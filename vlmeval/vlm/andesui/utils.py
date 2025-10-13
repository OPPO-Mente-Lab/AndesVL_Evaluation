import os
import sys
import random
import argparse
from collections import OrderedDict
import json
import time
import math
from PIL import Image
import torch.distributed as dist
import torch
import numpy as np

from model.modeling_andesvl import AndesVL

import transformers
from functools import partial
from flash_attn import flash_attn_varlen_func
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from flash_attn.ops.rms_norm import rms_norm
from transformers.modeling_outputs import CausalLMOutputWithPast
import numpy as np
from scipy.stats import uniform, beta

def sample_distribution(dist_type, size=1):
    """
    根据传入的分布类型进行采样
    
    参数:
    dist_type: str, 可选值为 'none', 'u1', 'u2', 'b1'
    size: int, 采样数量
    
    返回:
    numpy.array: 采样值数组
    """
    if dist_type == 'none':
        return np.ones(size)
    elif dist_type == 'u1':
        return uniform.rvs(loc=0.8, scale=0.4, size=size)
    elif dist_type == 'u2':
        return uniform.rvs(loc=0.5, scale=1.5, size=size)
    elif dist_type == 'b1':
        return beta.rvs(2, 5, size=size) * 1.5 + 0.5
    else:
        raise ValueError("Invalid distribution type")
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def log_version(save_dir, rank):
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    year_month = time.strftime("%Y%m", time.localtime(time.time()))
    if os.environ.get('HOSTNAME', "").startswith('dlc'):
        save_dir = os.path.join(save_dir, year_month, cur_time+"-"+os.environ['HOSTNAME'].split('-')[0])
    else:
        save_dir = os.path.join(save_dir, year_month, cur_time)
    if dist.is_initialized():  # Synchronize save_dir
        if dist.get_rank() == 0:
            objects = [save_dir]
        else:
            objects = [None]
        dist.broadcast_object_list(objects, src=0, device=torch.device("cuda"))
        save_dir = objects[0]
    return save_dir


def print_global_token_stats(pixel_values, world_size):
    org_sizes = [(p.shape[1], p.shape[2]) for p in pixel_values]
    tokens = [h * w // (14 * 14) for h, w in org_sizes]
    img_num = len(
        [t for t in tokens if t > 4]
    )  # 这里的img_num是所有图片的数量，注意4的情况是padding的图片
    token = sum(tokens)  # 这里的token是所有图片的token总和

    token_tensor = torch.tensor([token], dtype=torch.int, device="cuda")
    img_num_tensor = torch.tensor([img_num], dtype=torch.int, device="cuda")

    # 准备一个列表来收集所有节点的token张量
    gathered_tokens = [torch.zeros_like(token_tensor) for _ in range(world_size)]
    gathered_img_num = [torch.zeros_like(img_num_tensor) for _ in range(world_size)]

    # 使用dist.all_gather来收集所有节点的token数据
    dist.all_gather(gathered_tokens, token_tensor)
    dist.all_gather(gathered_img_num, img_num_tensor)

    # 将收集到的张量列表concatenate成一个张量
    all_tokens = torch.cat(gathered_tokens)
    max_tokens = all_tokens.max()
    min_tokens = all_tokens.min()
    avg_tokens = all_tokens.float().mean()

    # 获取img_num的最大值，最小值
    all_img_num = torch.cat(gathered_img_num)
    max_img_num = all_img_num.max()
    min_img_num = all_img_num.min()
    avg_img_num = all_img_num.float().mean()

    # 打印或返回统计结果
    result = (
        max_img_num.item(),
        min_img_num.item(),
        int(avg_img_num),
        max_tokens.item(),
        min_tokens.item(),
        int(avg_tokens.item()),
        int(avg_tokens.item() / 4),
    )
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--deepspeed_config_path", type=str, default="./configs/deepspeed_config.json"
    )
    parser.add_argument(
        "--training_config_path",
        type=str,
        default="./configs/training_config.json",
    )
    parser.add_argument(
        "--task_info_config_path", type=str, default="./configs/task_info.json"
    )
    parser.add_argument(
        "--task_weight_config_path", type=str, default="./configs/task_weight.json"
    )
    args = parser.parse_args()
    hparam = OrderedDict()
    hparam["local_rank"] = args.local_rank
    with open(args.deepspeed_config_path, "r", encoding="utf-8") as f:
        deepspeed_config = json.load(f)
    hparam["deepspeed_config"] = deepspeed_config
    with open(args.training_config_path, "r", encoding="utf-8") as f:
        training_config = json.load(f)
    hparam["training_config"] = training_config
    with open(args.task_info_config_path, "r", encoding="utf-8") as f:
        task_info = json.load(f)
    with open(args.task_weight_config_path, "r", encoding="utf-8") as f:
        task_weight = json.load(f)
    hparam["task_info"] = task_info
    hparam["task_weight"] = task_weight
    hparam["args"] = args
    return hparam


def get_model(hparam):
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    local_world_size = torch.cuda.device_count()

    m_rank, local_rank = divmod(rank, local_world_size)
    is_rank_loaded_model = torch.zeros(world_size, dtype=torch.int64).to(
        f"cuda:{local_rank}"
    )
    load_maps = {  # TODO: 这个是什么意思还有待研究
        0: 8,
        1: 8,
        2: 8,
        3: 8,
        4: 8,
        5: 8,
        6: 8,
        7: 8,
    }

    while is_rank_loaded_model.sum() != world_size:
        if (is_rank_loaded_model[rank] == 0) and (
            local_rank < load_maps[is_rank_loaded_model[ m_rank * local_world_size : (1 + m_rank) * local_world_size].sum().item()]
        ):
            print(f"> rank {rank}, local rank {local_rank} start load model ...")
            model = AndesVL(
                hparam["training_config"]["vit_path"],
                hparam["training_config"]["llm_path"],
                mlp_path=hparam["training_config"]["mlp_path"],
                local_rank=local_rank,
                zero_init=hparam["training_config"]["zero_init"],
                lora_rank=hparam["training_config"]["lora_rank"],
                vlora_rank=hparam["training_config"]["vlora_rank"],
            )
            if os.path.exists(hparam["training_config"]["ckpt_path"]):
                model.load_state_dict(
                    torch.load(hparam["training_config"]["ckpt_path"])
                )
            is_rank_loaded_model[rank] = 1
            print(f"> rank {rank}, local rank {local_rank} finish load model ...")
        elif is_rank_loaded_model[rank] == 1:
            print(f"> rank {rank}, local rank {local_rank} is load ...")
        else:
            print(f"> rank {rank}, local rank {local_rank} wait load model ...")

        torch.distributed.all_reduce(
            is_rank_loaded_model, op=torch.distributed.ReduceOp.MAX
        )
    print(f" > rank {rank}, finished load all model ...")

    torch.distributed.barrier()
    model.gradient_checkpointing_enable()
    return model


def get_dataset(hparam, data_state={}):
    from dataio.data_generator import DataGenerator

    dataloader = DataGenerator(
        llm_path=(
            hparam["training_config"]["llm_path"]
            if "llm_path" in hparam["training_config"]
            else None
        ),
        vit_path=(
            hparam["training_config"]["vit_path"]
            if "vit_path" in hparam["training_config"]
            else None
        ),
        detail=hparam["training_config"]["detail"],
        max_size=hparam["training_config"]["max_size"],
        tasks_info=hparam["task_info"],
        train_micro_batch_size_per_gpu=hparam["deepspeed_config"][
            "train_micro_batch_size_per_gpu"
        ],
        gradient_accumulation_steps=hparam["deepspeed_config"][
            "gradient_accumulation_steps"
        ],
        task_weight_config_path=hparam["args"].task_weight_config_path,
        max_length=hparam["training_config"]["max_length"],
        packing=hparam["training_config"]["packing"],
        data_state=data_state,
        random_seed=hparam["training_config"]['random_seed'] #dataloader的seed也是需要设置的。
    )
    return dataloader

def get_seq_lens(position_ids):
    """
    输入：position_ids，形状为（1，total_seq_len）的张量，其中包含多个（0，seq_len_1-1),(0,seq_len_2-1)的位置编码
    输出：cu_seqlens， max_seqlen，其中cu_seqlens是一个列表，包含每个位置编码的长度，max_seqlen是最大的长度
    """
    total_seq_len = position_ids.size(1) 
    seq_lens = []
    # 使用向量化操作来查找分段的长度
    diffs = (position_ids[0, 1:] == 0).nonzero(as_tuple=True)[0] + 1
    seq_lens = torch.diff(torch.cat((torch.tensor([0], device=position_ids.device), diffs, torch.tensor([total_seq_len], device=position_ids.device))))
    max_seqlen = seq_lens.max().item() # 计算最大长度
    # 计算 cu_seqlens
    cu_seqlens = torch.cat((torch.tensor([0], device=position_ids.device), torch.cumsum(seq_lens, dim=0)), dim=0).to(torch.int32)
    return cu_seqlens, max_seqlen

def rms_norm_forward(self, hidden_states):
    return rms_norm(hidden_states, self.weight, self.variance_epsilon)
def qwen2_flat_flash_forward(
    self,
    hidden_states,
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    output_attentions = False,
    use_cache = False,
    cache_position = None,
):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    #from liger_kernel.transformers import liger_rotary_pos_emb as apply_rotary_pos_emb #实测这个可能导致flash attention计算出现nan
    #NOTE: 1. 因为我们训练的上下文窗口较小，所以暂时不考虑qwen2的滑窗注意力。2. 这个不能用在推理的时候。
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cu_seqlens, max_seqlen = get_seq_lens(position_ids)
    
    rotary_seq_len = max_seqlen
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    
    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    #flash_attention
    attn_output = flash_attn_varlen_func(
                    query_states.squeeze(0),
                    key_states.squeeze(0),
                    value_states.squeeze(0),
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                ) #(total, nheads, headdim)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value

def qwen3_flat_flash_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_value = None,
    cache_position = None,
    **kwargs,
):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    bsz, q_len, _ = hidden_states.size()
    position_ids = kwargs['position_ids']
    cu_seqlens, max_seqlen = get_seq_lens(position_ids)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    #flash_attention
    attn_output = flash_attn_varlen_func(
                    query_states.squeeze(0),
                    key_states.squeeze(0),
                    value_states.squeeze(0),
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                ) #(total, nheads, headdim)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    attn_weights = None
    return attn_output, attn_weights


def llama_flat_flash_forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
        cache_position = None,
    ):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    #from liger_kernel.transformers import liger_rotary_pos_emb as apply_rotary_pos_emb
    #NOTE: transformers库后续版本的llama可能会去掉position_ids这个传参，后续需要注意一下。。。
    output_attentions = False
    bsz, q_len, _ = hidden_states.size()
    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cu_seqlens, max_seqlen = get_seq_lens(position_ids)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    #flash_attention
    attn_output = flash_attn_varlen_func(
                    query_states.squeeze(0),
                    key_states.squeeze(0),
                    value_states.squeeze(0),
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                ) #(total, nheads, headdim)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value


def lm_forward(
    self,
    input_ids = None,
    attention_mask = None,
    position_ids = None,
    past_key_values = None,
    inputs_embeds = None,
    labels = None,
    use_cache = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
    cache_position = None,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    #logits = logits.float() #我们把.float()操作给提到后面去了，这样可以避免对不需要计算loss的token的logits的显存占用。

    loss = None
    if labels is not None:
        cu_seqlens, max_seqlen = get_seq_lens(position_ids)
        loss_fct = CrossEntropyLoss(reduction='sum', inplace_backward=True, ignore_index=-100)
        average = 'token'  # 可以设置为 'token' 或 'sample'，暂不支持外部接口控制，需要在代码内部修改。
        total_loss = 0.0
        total_valid_tokens = 0
        sample_losses = []
        losses = []
        for i in range(len(cu_seqlens) - 1):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            if end_idx-start_idx==1:
                #长度为1的是pad的token，得跳过才行，不然shift之后的是空的张量。
                continue
            shift_logits = logits[..., start_idx:end_idx - 1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., start_idx + 1: end_idx].contiguous().view(-1).to(shift_logits.device)
            
            valid_positions = shift_labels != -100
            valid_shift_logits = shift_logits[valid_positions].float()
            #valid_shift_logits = shift_logits[valid_positions]
            valid_shift_labels = shift_labels[valid_positions]

            sample_loss = loss_fct(valid_shift_logits, valid_shift_labels)
            valid_count = valid_positions.sum().item()
            
            if average == 'token':
                total_loss += sample_loss
                total_valid_tokens += valid_count
            elif average == 'sample':
                sample_losses.append(sample_loss/ valid_count if valid_count > 0 else 0)
            else:
                raise ValueError(f"average must be 'token' or 'sample', but got {average}")
        
        if average == 'token':
            loss = total_loss / total_valid_tokens if total_valid_tokens > 0 else 0.0
        elif average == 'sample':
            loss = sum(sample_losses) / len(sample_losses) if len(sample_losses) > 0 else 0.0
            
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def lm_forward1(
    self,
    input_ids = None,
    attention_mask = None,
    position_ids = None,
    past_key_values = None,
    inputs_embeds = None,
    labels = None,
    use_cache = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
    cache_position = None,
):    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]  # shape: (1, total_seq_len, hidden_size)

    loss = None
    if labels is not None:
        # cu_seqlens包含每个序列结束位置的累积和
        # 例如: [0, 5, 8, 12] 表示第一个序列长度5,第二个长度3,第三个长度4
        cu_seqlens, max_seqlen = get_seq_lens(position_ids)  
        # cu_seqlens shape: (num_sequences + 1,)
        
        # 计算每个序列的长度
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]  # shape: (num_sequences,)
        valid_seq_mask = seqlens > 1  # shape: (num_sequences,)
        
        # 获取有效序列的起始和结束位置
        valid_starts = cu_seqlens[:-1][valid_seq_mask]  # shape: (num_valid_sequences,)
        valid_ends = cu_seqlens[1:][valid_seq_mask]     # shape: (num_valid_sequences,)
        
        # 创建所有有效序列中需要预测的位置的索引
        indices = torch.cat([torch.arange(start, end-1, device=labels.device, dtype=torch.int32) 
                           for start, end in zip(valid_starts, valid_ends)])
        # indices shape: (total_valid_positions,)
        # total_valid_positions = sum(valid_sequence_lengths - 1)
        
        # 获取标签位置(向右移动一位)
        label_indices = indices + 1  # shape: (total_valid_positions,)
        shift_labels = labels[:, label_indices].view(-1)  # shape: (total_valid_positions,)
        
        # 找出label!=-100的位置
        valid_positions = shift_labels != -100  # shape: (total_valid_positions,)
        final_indices = indices[valid_positions]  # shape: (num_valid_tokens,)
        valid_labels = shift_labels[valid_positions]  # shape: (num_valid_tokens,)
        
        # 只对需要计算loss的位置计算logits
        selected_hidden = hidden_states[:, final_indices, :]  
        # shape: (1, num_valid_tokens, hidden_size)
        
        valid_logits = self.lm_head(selected_hidden).float().view(-1, self.config.vocab_size)
        # shape: (num_valid_tokens, vocab_size)
        
        # 计算损失
        loss_fct = CrossEntropyLoss(reduction='mean', inplace_backward=True)
        loss = loss_fct(valid_logits, valid_labels)

    logits = None #暂时不支持返valid_positions 回logits

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

#'''
def lm_forward2(
    self,
    input_ids = None,
    attention_mask = None,
    position_ids = None,
    past_key_values = None,
    inputs_embeds = None,
    labels = None,
    use_cache = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
    cache_position = None,
):    
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]  # shape: (1, total_seq_len, hidden_size)

    loss = None
    if labels is not None:
        # cu_seqlens包含每个序列结束位置的累积和
        # 例如: [0, 5, 8, 12] 表示第一个序列长度5,第二个长度3,第三个长度4
        cu_seqlens, max_seqlen = get_seq_lens(position_ids)  
        # cu_seqlens shape: (num_sequences + 1,)
        
        # 计算每个序列的长度
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]  # shape: (num_sequences,)
        valid_seq_mask = seqlens > 1  # shape: (num_sequences,)
        
        # 获取有效序列的起始和结束位置
        valid_starts = cu_seqlens[:-1][valid_seq_mask]  # shape: (num_valid_sequences,)
        valid_ends = cu_seqlens[1:][valid_seq_mask]     # shape: (num_valid_sequences,)
        
        # 创建所有有效序列中需要预测的位置的索引
        indices = torch.cat([torch.arange(start, end-1, device=labels.device, dtype=torch.int32) 
                           for start, end in zip(valid_starts, valid_ends)])
        # indices shape: (total_valid_positions,)
        # total_valid_positions = sum(valid_sequence_lengths - 1)
        
        # 获取标签位置(向右移动一位)
        label_indices = indices + 1  # shape: (total_valid_positions,)
        shift_labels = labels[:, label_indices].view(-1)  # shape: (total_valid_positions,)
        
        # 找出label!=-100的位置
        valid_positions = shift_labels != -100  # shape: (total_valid_positions,)
        final_indices = indices[valid_positions]  # shape: (num_valid_tokens,)
        valid_labels = shift_labels[valid_positions]  # shape: (num_valid_tokens,)
        
        # 只对需要计算loss的位置计算logits
        selected_hidden = hidden_states[:, final_indices, :].view(-1, self.config.hidden_size)
        # shape: (num_valid_tokens, hidden_size)

        # 计算损失
        loss_fct = LigerFusedLinearCrossEntropyLoss(reduction='mean')
        loss = loss_fct(self.lm_head.weight, selected_hidden, valid_labels)

    if not return_dict:
        logits = None
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output
    logits = self.lm_head(hidden_states)
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
#'''
'''
def lm_forward2(
    self,
    input_ids = None,
    attention_mask = None,
    position_ids = None,
    past_key_values = None,
    inputs_embeds = None,
    labels = None,
    use_cache = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
    cache_position = None,
):    
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    average = 'sample'  # 可以设置为 'token' 或 'sample'

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]  # shape: (1, total_seq_len, hidden_size)

    loss = None
    if labels is not None:
        # cu_seqlens包含每个序列结束位置的累积和
        cu_seqlens, max_seqlen = get_seq_lens(position_ids)  
        # cu_seqlens shape: (num_sequences + 1,)
        
        # 计算每个序列的长度
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]  # shape: (num_sequences,)
        valid_seq_mask = seqlens > 1  # shape: (num_sequences,)
        
        # 获取有效序列的起始和结束位置
        valid_starts = cu_seqlens[:-1][valid_seq_mask]  # shape: (num_valid_sequences,)
        valid_ends = cu_seqlens[1:][valid_seq_mask]     # shape: (num_valid_sequences,)

        # 创建所有有效序列中需要预测的位置的索引
        indices = torch.cat([torch.arange(start, end-1, device=labels.device, dtype=torch.int32) 
                            for start, end in zip(valid_starts, valid_ends)])
        # indices shape: (total_valid_positions,)
        
        # 获取标签位置(向右移动一位)
        label_indices = indices + 1  # shape: (total_valid_positions,)
        shift_labels = labels[:, label_indices].view(-1)  # shape: (total_valid_positions,)
        
        # 找出label!=-100的位置
        valid_positions = shift_labels != -100  # shape: (total_valid_positions,)
        final_indices = indices[valid_positions]  # shape: (num_valid_tokens,)
        valid_labels = shift_labels[valid_positions]  # shape: (num_valid_tokens,)
        assert len(valid_labels) > 0

        # 只对需要计算loss的位置计算logits
        selected_hidden = hidden_states[:, final_indices, :].view(-1, self.config.hidden_size)
        # shape: (num_valid_tokens, hidden_size)

        if average == 'token':
            # token级别平均,直接计算所有token的平均loss
            loss_fct = LigerFusedLinearCrossEntropyLoss(reduction='mean')
            loss = loss_fct(self.lm_head.weight, selected_hidden, valid_labels)
        
        elif average == 'sample':
            # 创建sequence_ids用于追踪每个token属于哪个序列
            sequence_ids = torch.cat([torch.full((end-start-1,), i, device=labels.device, dtype=torch.int64)
                                    for i, (start, end) in enumerate(zip(valid_starts, valid_ends))])
            valid_sequence_ids = sequence_ids[valid_positions]
            
            # 计算所有token的loss
            loss_fct = LigerFusedLinearCrossEntropyLoss(reduction='none')
            token_losses = loss_fct(self.lm_head.weight, selected_hidden, valid_labels)
            
            # 使用scatter_mean计算每个序列的平均loss
            num_valid_sequences = len(valid_starts)  # 使用有效序列的数量
            sequence_losses = torch.zeros(num_valid_sequences, device=labels.device)
            sequence_counts = torch.zeros(num_valid_sequences, device=labels.device)
            
            # 累加每个序列的losses和计数
            sequence_losses.scatter_add_(0, valid_sequence_ids, token_losses)
            sequence_counts.scatter_add_(0, valid_sequence_ids, torch.ones_like(token_losses))
            
            # 计算所有有效序列的平均loss
            sequence_losses = sequence_losses / sequence_counts #NOTE: 如果我们除以sequence_counts**0.5，就是InternVL2.5的Loss了。
            loss = sequence_losses.mean()
        else:
            raise ValueError(f"average必须是'token'或'sample', 但获得了{average}")

    if not return_dict:
        logits = None
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    logits = self.lm_head(hidden_states)
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
'''

def fuse():    
    from liger_kernel.transformers import LigerSwiGLUMLP
    transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = rms_norm_forward
    transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm.forward = rms_norm_forward

    #NOTE: lm_forward2中不需要CrossEntropyLoss
    #transformers.models.llama.modeling_llama.CrossEntropyLoss = partial(CrossEntropyLoss, inplace_backward=True)
    #transformers.models.qwen2.modeling_qwen2.CrossEntropyLoss = partial(CrossEntropyLoss, inplace_backward=True)

    #NOTE: 这里面集成了RoPE的优化（可能导致nan，已去掉），以及no padding的优化
    #TODO: 这里最好是根据transformers的版本，进行优化。
    if not hasattr(transformers.models, "qwen3"):
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_flat_flash_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flat_flash_forward
    else:
        #这里是兼容qwen3的训练代码
        transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm.forward = rms_norm_forward
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = qwen3_flat_flash_forward
        transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.forward = lm_forward2
        transformers.models.qwen3.modeling_qwen3.Qwen3MLP = LigerSwiGLUMLP


    #NOTE: 这里主要集成了linear_corss_entropy的优化，以及大量label==-100的稀疏优化
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = lm_forward2
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.forward = lm_forward2


    transformers.models.llama.modeling_llama.LlamaMLP = LigerSwiGLUMLP
    transformers.models.qwen2.modeling_qwen2.Qwen2MLP = LigerSwiGLUMLP
    #这里是针对LayerNorm的(MLP connector以及ViT都在用)
    #import model
    #from liger_kernel.transformers import LigerLayerNorm
    #model.vision.qwen2vl.modeling_qwen2_vl_rope_navit.LayerNorm = LigerLayerNorm
    #model.vision.internvl.modeling_intern_vl_rope_navit.LayerNorm = LigerLayerNorm
    #model.modeling_andesvl.LayerNorm = LigerLayerNorm

def count_images(lst):
    count = 0
    is_counting = False
    for num in lst:
        if num == 1:
            if not is_counting:
                count += 1
                is_counting = True
        else:
            is_counting = False
    return count
#TODO：理论极端情况下可能出现短边长度为0的bug，需要修复一下。
def get_scaled_img_size(image_size, max_area, base, max_resolution=4480, upper=True):
    """计算缩放后的图片大小和包裹矩形的大小"""
    assert max_resolution%base==0
    # 计算原始图片的宽高比
    aspect_ratio = image_size[0] / image_size[1]
    # 计算包裹矩形的最大可能宽度和高度
    max_width = math.floor(math.sqrt(max_area * aspect_ratio))
    max_height = math.floor(math.sqrt(max_area / aspect_ratio))
    max_width, max_height = min(max_width, max_resolution), min(
        max_height, max_resolution
    )
    max_width, max_height = max(max_width, base), max(max_height, base)
    # 确保包裹矩形的宽度和高度都是base的整数倍
    if not upper:
        # 向下取整, 保证面积不会超过max_area
        max_width = max_width - max_width % base
        max_height = max_height - max_height % base
    else:
        # 向上取整，同时不超过max_resolution（单边最大长度）
        max_width = min(max_width + (base - max_width % base), max_resolution)
        max_height = min(max_height + (base - max_height % base), max_resolution)
    # 计算缩放因子
    scale_factor = min(max_width / image_size[0], max_height / image_size[1])
    # 计算缩放后的图片大小
    new_image_size = (
        round(image_size[0] * scale_factor),
        round(image_size[1] * scale_factor),
    )
    # 计算包裹矩形的大小
    bounding_box_size = (max_width, max_height)
    return new_image_size, bounding_box_size


def max_preprocess(
    img, max_size, base, background_color, max_resolution=4480, upper=True, force_resize=False
):
    """对图片进行预处理，使其面积接近max_size**2"""
    # 首先把图片resize到长度和宽度都低于max_resolution
    w, h = img.size
    if max(w, h) > max_resolution:
        scale = max_resolution / max(w, h)
        w, h = int(w * scale), int(h * scale)
    # 获取缩放后的图片大小和包裹矩形的大小
    new_image_size, bounding_box_size = get_scaled_img_size(
        (w, h), max_size**2, base, max_resolution, upper
    )
    if force_resize:
        return img.resize(bounding_box_size)
    # 创建一个新的画布
    canvas = Image.new("RGB", bounding_box_size, background_color)
    # 计算将图像粘贴到画布上的位置
    paste_width = (bounding_box_size[0] - new_image_size[0]) // 2
    paste_height = (bounding_box_size[1] - new_image_size[1]) // 2
    # 将图像粘贴到画布上
    canvas.paste(img.resize(new_image_size), (paste_width, paste_height))
    return canvas


def native_preprocess(
    img, max_size, base, background_color, max_resolution=4480, min_tokens=64
):
    # 对图片进行处理，使其宽度和高度都是base的整数倍
    # 如果图片的最长边超过max_resolution，就把图片resize到max_resolution以内
    w, h = img.size
    # 首先保证图片的最长边不超过max_resolution(ViT在极限长度)
    if max(w, h) > max_resolution:
        scale = max_resolution / max(w, h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h))
    if w * h > max_size**2:
        return max_preprocess(img, max_size, base, background_color, max_resolution)
    if w * h < (base * base * min_tokens):
        return max_preprocess(
            img,
            int(base * (min_tokens**0.5)),
            base,
            background_color,
            max_resolution,
        )  
    w1, h1 = w + base - w % base, h + base - h % base
    if w1 == w and h1 == h:
        return img
    else:
        # 创建一个新的(w1, h1)的画布，并把图片resize保证只有一侧存在白边的情况
        scale = min(w1 / w, h1 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h))
        canvas = Image.new("RGB", (w1, h1), background_color)
        canvas.paste(img, ((w1 - new_w) // 2, (h1 - new_h) // 2))
        return canvas

def resize_to_scale(img, scale, max_size):
    if scale==1:
        return img
    w, h = img.size
    if w*h >= max_size*max_size:
        return img
    new_sz = (int(w*scale), int(w*scale))
    return img.resize(new_sz)

def resize_to_area(img, area):
    "Resize image so that the area is `area`"
    ratio = (area / img.size[0] / img.size[1]) ** 0.5
    new_sz = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    return img.resize(new_sz)


def cal_num_of_slices(org_width, org_height, max_area, max_scale=9):
    """计算宽度和高度的切片数，这里直接按照子图最接近正方形的策略来切割"""
    org_area = org_height * org_width
    scale = org_area / (max_area)
    scale = math.ceil(scale)
    scale = min(scale, max_scale)

    def factorize(n):
        factors = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append((i / (n / i), i, n // i))
        return factors

    available_ratios = []
    log_origin_ratio = math.log(org_width / org_height)
    if scale == 1:
        available_ratios = factorize(scale)  # 这里只有1*1
    elif scale == 2:
        available_ratios = factorize(scale) + factorize(scale + 1)  # 这里只有1*2
    elif scale == max_scale:
        available_ratios = factorize(scale - 1) + factorize(scale)
    else:
        available_ratios = (
            factorize(scale - 1) + factorize(scale) + factorize(scale + 1)
        )
    min_dif = 1000
    best_w = 0
    best_h = 0
    for r, w_slice, h_slice in available_ratios:
        log_r = math.log(r)
        if min_dif > abs(log_r - log_origin_ratio):
            min_dif = abs(log_r - log_origin_ratio)
            best_w = w_slice
            best_h = h_slice
    return best_w, best_h


def pad_to_max(image, best_w, best_h, base, background_color=(0, 0, 0)):
    """对图片进行处理，使其宽度被best_w*base整除，高度被best_h*base整除（对高分辨率的大图切割之前的预处理）"""
    # NOTE: 这里返回的图片，是比原图更大的，因为我们只是padding，以保证子图
    width, height = image.size
    base_w = best_w * base
    base_h = best_h * base
    if width % base_w == 0 and height % base_h == 0:
        return image
    target_width = int(math.ceil(width / base_w) * base_w)
    target_height = int(math.ceil(height / base_h) * base_h)
    # 创建一个新的画布
    new_image = Image.new("RGB", (target_width, target_height), background_color)
    # 计算缩放比例
    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    # 调整图像大小
    resized_image = image.resize((new_width, new_height))
    # 计算粘贴位置
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    # 粘贴图像到新画布
    new_image.paste(resized_image, (paste_x, paste_y))
    return new_image



def slice_image_wh(image, best_w, best_h):
    origin_image_width = image.size[0]
    origin_image_height = image.size[1]
    assert origin_image_width % best_w == 0 and origin_image_height % best_h == 0
    slices = []
    for j in range(best_h):
        for i in range(best_w):
            box = (
                i * origin_image_width // best_w,
                j * origin_image_height // best_h,
                (i + 1) * origin_image_width // best_w,
                (j + 1) * origin_image_height // best_h,
            )
            region = image.crop(box).convert("RGB")
            slices.append(region)
    return slices


if __name__ == "__main__":
    import sys

    # print(sys.argv)
    sys.argv = [
        "main.py",
        "--deepspeed_config_path",
        "./configs/deepspeed_config.json",
        "--training_config_path",
        "./configs/training_config.json",
        "--task_info_config_path",
        "./configs/task_info.json",
        "--task_weight_config_path",
        "./configs/task_weight.json",
    ]
    sys.argv = [x.replace("./configs", "./configs/.test") for x in sys.argv]
    hparam = get_args()
    #for k in hparam["task_info"].keys():
    #    hparam["task_info"][k]["num_workers"] = 4
    #hparam['deepspeed_config']['train_micro_batch_size_per_gpu'] = 1
    data_state = {}
    dataloader = get_dataset(hparam, data_state=data_state)
    n = 0
    ns = []
    from tqdm import tqdm

    max_n = 1000
    samples = 0
    steps = 0
    global_batch_size = 1
    seed_everything(42)
    import time

    last_time = time.time()
    imgs = []
    l_tokens = []
    # for batch in tqdm(dataloader):
    for batch in dataloader:
        temp_time = time.time()
        # print(temp_time-last_time)
        last_time = temp_time
        n += batch["input_ids"].shape[0]
        # print(batch['input_ids'].shape[1])
        ns.append(
            batch["input_ids"].shape[1] * batch["input_ids"].shape[0]
        )  # 记录我们训练的token数量
        imgs.append(len(batch["pixel_values"]))  # 记录我们训练了的图片数量
        l_tokens.append(batch["image_flags"].ne(1).sum().item())  # 这个是我们训练了的
        steps += 1
        samples +=  batch['position_ids'][batch['image_flags']==0].eq(0).sum().item()
        print(
            f"token: {sum(ns)/steps}, l_token: {sum(l_tokens)/steps}, img: {sum(imgs)/steps}, samples: {samples/steps}"
        )
        if steps >= max_n:
            break
    # print(sum(ns)/n)
    # print("eos")
