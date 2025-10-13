import json
import torch
import os
import re
import time
import warnings
from .base import BaseModel
from PIL import Image
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import sys

sys.path.insert(0, "/mnt/data/group/wangnan/code/VLMS/AndesVL-V1") # 兼容starfire的cpfs挂载
from utils import count_images, get_scaled_img_size, max_preprocess, native_preprocess, cal_num_of_slices, resize_to_area, pad_to_max, slice_image_wh
from model.vision.qwen2vl.modeling_qwen2_vl_rope_navit import Qwen2VLVisionModel
from model.vision.internvl.modeling_intern_vl_rope_navit import InternVisionModel
from model.vision.qwen2_5_vl.modeling_qwen2_5_vl_navit_rope import Qwen2_5_VLVisionModel
from model.vision.siglip2.modeling_siglip2_navit_rope import Siglip2VisionModel
from model.vision.siglip2_naflex.modeling_siglip2 import Siglip2NavitVisionModel
from model.vision.aimv2.modeling_aimv2 import AIMv2Model
from ..smp import *


class AndesVL(nn.Module):
    def __init__(
        self,
        vit_path,
        llm_path,
        mlp_path=None,
        zero_init=False,
    ):
        super().__init__()
        vit_name = os.path.basename(vit_path).lower()
        if  zero_init:
            if 'intern' in vit_name:
                self.vision_encoder = InternVisionModel.from_pretrained(vit_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
            elif 'qwen' in vit_name:
                if '2.5' not in vit_name:
                    self.vision_encoder = Qwen2VLVisionModel.from_pretrained(vit_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
                else:
                    self.vision_encoder = Qwen2_5_VLVisionModel.from_pretrained(vit_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
            elif 'siglip2' in vit_name:
                if 'rope' in vit_name:
                    self.vision_encoder = Siglip2VisionModel.from_pretrained(vit_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
                else:
                    self.vision_encoder = Siglip2NavitVisionModel.from_pretrained(vit_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)  
            elif 'aimv2' in vit_name:
                self.vision_encoder = AIMv2Model.from_pretrained(vit_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16) 
            else:
                raise ValueError(f"Unknown vision model: {vit_path}")
            self.language_model = AutoModelForCausalLM.from_pretrained(llm_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
        else:
            if 'intern' in vit_name:
                self.vision_encoder = InternVisionModel.from_pretrained(vit_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16, device_map={"": 0})
            elif 'qwen' in vit_name:
                if '2.5' not in vit_name:
                    self.vision_encoder = Qwen2VLVisionModel.from_pretrained(vit_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16, device_map={"": 0})
                else:
                    self.vision_encoder = Qwen2_5_VLVisionModel.from_pretrained(vit_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16, device_map={"": 0})
            elif 'siglip2' in vit_name:
                if 'rope' in vit_name:
                    self.vision_encoder = Siglip2VisionModel.from_pretrained(vit_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16, device_map={"": 0})
                else:
                    self.vision_encoder = Siglip2NavitVisionModel.from_pretrained(vit_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16, device_map={"": 0})
            elif 'aimv2' in vit_name:
                self.vision_encoder = AIMv2Model.from_pretrained(vit_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16, device_map={"": 0})
            else:
                raise ValueError(f"Unknown vision model: {vit_path}")
            self.language_model = AutoModelForCausalLM.from_pretrained(llm_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16,device_map="auto")
        # vit_hidden_size = self.vision_encoder.config.hidden_size if 'intern' in vit_name else self.vision_encoder.config.embed_dim
        vit_hidden_size = self.vision_encoder.config.embed_dim if 'qwen' in vit_name else self.vision_encoder.config.hidden_size
        llm_hidden_size = self.language_model.config.hidden_size
        self.patch_size = self.vision_encoder.config.patch_size
        if 'intern' in vit_name:
            self.mlp = nn.Sequential(
                nn.LayerNorm(vit_hidden_size * 4),
                nn.Linear(vit_hidden_size * 4, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
        else:
            # qwen的patch merger的情况，注意因为维度的问题，我们把layernorm给放到vit里面去了。
            self.mlp = nn.Sequential(
                nn.Linear(vit_hidden_size * 4, vit_hidden_size * 4),
                nn.GELU(),
                nn.Linear(vit_hidden_size * 4, llm_hidden_size),
            )
        if not zero_init:
            self.mlp.to("cuda:0")
        if os.path.exists(mlp_path):
            self.mlp.load_state_dict(torch.load(mlp_path))
            
    def get_flated_pixel_values(self, pixel_values):
        # NOTE: 理论上我们可以在数据处理的时候完成这步，但是考虑到之前代码的复用方便，在此处才序列化。
        # 对pixel_values进行处理，变为序列化的图片，其中空间上相邻的四个patch在序列上相邻
        flated_pixel_values = []
        image_grid_hw = []
        for pv in pixel_values:
            # 获取图片的宽高
            c, h, w = pv.shape
            assert c==3 and h%self.patch_size==0 and w%self.patch_size==0, f"{c}, {w}, {h}, {self.patch_size}"
            image_grid_hw.append((h//self.patch_size, w//self.patch_size))
            fpv = pv.reshape(c, h//(2*self.patch_size), 2, self.patch_size, w//(2*self.patch_size), 2, self.patch_size)
            flated_pixel_values.append(fpv.permute(1, 4, 2, 5, 0, 3, 6).reshape(-1, c*self.patch_size*self.patch_size))
        flated_pixel_values = torch.cat(flated_pixel_values, dim=0) # (Len_img, C, H, W)
        image_grid_hw = torch.tensor(image_grid_hw, device=flated_pixel_values.device) # (N_img, 2)
        return flated_pixel_values, image_grid_hw

    def get_vit_embeds_and_merge(self, pixel_values, image_grid_hw, input_embeds, image_flags):
        """
        Args:
            pixel_values: (Len_img, H_vit0)， 拉平后的初始patch特征，按照序列维度拼接在一起
            image_grid_hw: (N_img, 2)， 每个图片的宽高
            input_embeds: (Bt, Lt, Ht)， 每个token的embedding
            image_flags: (Bt, Lt)， 每个token是否是图片
        """
        # 首先是直接过vit提取特征
        vit_embeds = self.vision_encoder(pixel_values, image_grid_hw)  # (Len_img, H_vit)
        # 然后我们把相邻的四个patch合并
        vit_embeds = vit_embeds.view(-1, vit_embeds.shape[-1]*4) # (Len_img//4, H_vit*4)
        # 然后过mlp
        vit_embeds = self.mlp(vit_embeds) # (Len_img//4, H_llm)
        # NOTE:TODO: 多图推理这里的截断必加
        vit_embeds = vit_embeds[:image_flags.sum()] # 推理的时候，后面的图片可能是padding的，所以这里需要截断
        # 然后和原始的input_embeds合并
        # assert vit_embeds.shape[0] == image_flags.sum(), f"{vit_embeds.shape[0]}, {image_flags.sum()}"
        Bt, Lt, Ht = input_embeds.shape
        input_embeds = input_embeds.reshape(-1, Ht)
        image_flags = image_flags.view(-1)
        input_embeds[image_flags == 1] = vit_embeds
        input_embeds = input_embeds.view(Bt, Lt, Ht) # 恢复形状
        return input_embeds 

    @torch.inference_mode()
    def generate(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        image_flags=None,  # (Bt, Lt)
        generation_config=None,
        output_hidden_states=None,
        return_dict=None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        input_embeds = self.language_model.get_input_embeddings()(input_ids)  # (Bt, Lt, Ht)
        if image_flags != None and (image_flags == 1).sum() > 0:
            flated_pixel_values, image_grid_hw = self.get_flated_pixel_values(pixel_values)
            input_embeds = self.get_vit_embeds_and_merge(flated_pixel_values, image_grid_hw, input_embeds, image_flags)
        outputs = self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            #return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )
        return outputs

def pad_sequence_left(sequences, batch_first=False, padding_value=0):
    # 计算最大长度
    max_size = max([s.size(0) for s in sequences])
    # 准备填充后的序列
    padded_seqs = []
    for seq in sequences:
        # 计算需要填充的长度
        num_padding = max_size - seq.size(0)
        # 创建一个填充tensor
        padding = torch.full((num_padding,), padding_value, dtype=seq.dtype)
        # 将填充tensor和原始tensor连接起来
        padded_seq = torch.cat((padding, seq), dim=0)
        padded_seqs.append(padded_seq)
    # 如果batch_first为True，转换维度
    if batch_first:
        return torch.stack(padded_seqs, 0)
    else:
        return torch.stack(padded_seqs, 1)

def get_global_max_pixel_values_len(current_rank_len: int):
    if not dist.is_initialized():
        return current_rank_len
    global_len = torch.tensor([current_rank_len], dtype=torch.int64, device='cuda')
    dist.all_reduce(global_len, op=dist.ReduceOp.MAX)
    return global_len.item()

class AndesVL_V1(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    # NOTE: 1. 这里现在是只支持deepspeed推理的，已经写死了，2. 这个model_path只是一个兼容参数，实际我们不会使用。
    def __init__(self, model_path, **kwargs):
        device = torch.cuda.current_device()
        self.device = device
        model_dir =  os.environ.get("model_dir")
        hp_path = os.path.join(os.path.split(model_dir)[0],"hparams.json")
        hparams = json.load(open(hp_path))
        llm_path = hparams['training_config']['llm_path']
        vit_path = hparams['training_config']['vit_path']
        llm_name = os.path.basename(llm_path)
        vit_name = os.path.basename(vit_path)
        if not hparams['training_config']['freeze_llm']:
            llm_path = f"{model_dir}/{llm_name}"
        if not hparams['training_config']['freeze_vit']:
            vit_path = f"{model_dir}/{vit_name}"
        self.llm_path = llm_path
        self.vit_path = vit_path
        self.mlp_path =  f"{model_dir}/mlp.bin"
        
        ds_config = {
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": int(os.environ.get("zero_stage", 0)),
            },
            "train_micro_batch_size_per_gpu": 1,
        }
        if os.path.exists(self.vit_path) and os.path.exists(self.llm_path) and os.path.exists(self.mlp_path):
            if dist.is_initialized():
                from transformers.integrations.deepspeed import HfDeepSpeedConfig
                hfdsc = HfDeepSpeedConfig(ds_config)
            self.model = AndesVL(vit_path=self.vit_path, llm_path=self.llm_path, mlp_path=self.mlp_path, zero_init=True)
            self.processor = CLIPImageProcessor.from_pretrained(self.vit_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True, use_fast=False)
            assert "<|im_end|>" in self.tokenizer.get_vocab()
            if self.tokenizer.eos_token!="<|im_end|>": 
                self.tokenizer.eos_token = "<|im_end|>"
        else:
            print("模型未切分，尝试直接导入pytorch_model.bin，请确保内存足够。")
            # NOTE: 这里我们仍然使用了zero init为True，因为我们要先把模型放到cpu上。
            self.model = AndesVL(vit_path=hparams['training_config']['vit_path'], 
                                 llm_path=hparams['training_config']['llm_path'], 
                                 mlp_path=self.mlp_path, 
                                 zero_init=True)
            self.model.load_state_dict(torch.load(os.path.join(model_dir, 'pytorch_model.bin')))
            self.processor = CLIPImageProcessor.from_pretrained(hparams['training_config']['vit_path'])
            self.tokenizer = AutoTokenizer.from_pretrained(hparams['training_config']['llm_path'], trust_remote_code=True, use_fast=False)
            assert "<|im_end|>" in self.tokenizer.get_vocab()
            if self.tokenizer.eos_token!="<|im_end|>": 
                self.tokenizer.eos_token = "<|im_end|>"
        self.patch_size = 14 if not hasattr(self.processor, 'patch_size') else self.processor.patch_size
        self.model.eval()
        if dist.is_initialized():
            ds_engine = deepspeed.initialize(model=self.model, config_params=ds_config)[0]
            ds_engine.module.eval()
            self.model  = ds_engine.module
        else:
            self.model.to(self.device)
        temperature=float(os.environ.get('temperature', 0.0))
        do_sample= eval(os.environ.get('do_sample', str(temperature > 0)))
        kwargs_default = dict(
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=int(os.environ.get("max_new_tokens", 2048)),
            top_p=float(os.environ.get('top_p', 1.0)),
            repetition_penalty=float(os.environ.get('repetition_penalty', 1.0))
        )
        kwargs_default.update(kwargs)
        self.extra_prompt = os.environ.get("extra_prompt","").strip()
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def message_to_prompting(self, message):
        s = []
        for m in message:
            if m['type']=="image":
                s.append(f"<img>{m['value']}</img>")
            elif m['type']=='text':
                s.append(m['value'])
            else:
                raise Exception(m)
        return "\n".join(s)

    # NOTE: 目前对于HallusionBench和POPE，可以通过设置环境变量thinking_mode=1来启用深度思考，其余benchmark保持和原设定一致
    def add_custom_prompt(self, prompt, dataset=None):
        """
        针对特定数据集，在prompt中添加自定义内容
        """
        thinking_mode = os.environ.get("thinking_mode", '0')=='1'
        if self.extra_prompt:
            prompt = prompt.strip() + "\n" + self.extra_prompt.strip()

        if dataset in ['RealWorldQA','Q-Bench1_VAL', 'DynaMath']:
            pass
        elif dataset in ['TextVQA_VAL']:
            prompt += "\nGive the final answer directly without any explanation."
        elif dataset in ['OCRBench']:
            if 'write out' in prompt:
                pass
            elif 'Answer this question using the text in the image directly.' not in prompt:
                prompt += '\nAnswer this question based on the text in the image directly.'
        elif dataset in ['ChartQA_TEST']:
            prompt = prompt.replace('Answer the question using a single word or phrase.',
                                'Answer the question using a single number or phrase.')
        elif dataset in ['MTVQA_TEST']:
            prompt = prompt.replace('\nAnswer the question using a word or phrase in the language of the question.', '')
        elif dataset in ['MathVista_MINI','MathVerse_MINI', 'MathVerse_MINI_Vision_Only']:
            if 'Choices:' in prompt:
                prompt = prompt.replace('Choices:', 'Options:').replace('Hint:', 'Context:')
                for i in range(1, 7):  # replace A ~ F
                    prompt = prompt.replace(f'({chr(64 + i)})', f'{chr(64 + i)}.')
                prompt += '\nAnswer with the option’s letter from the given choices directly.'
            else:
                prompt = prompt.replace('Hint:', 'Context:')
                prompt += '\nAnswer the question using a single word or phrase.'
        elif dataset in ['MMBench_DEV_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11',
                         'AI2D_TEST', 'AI2D_TEST_NO_MASK', 'MMMU_DEV_VAL','BLINK','MMMU_Pro_10c','SEEDBench2_Plus']:
            prompt = prompt.replace('Please select the correct answer from the options above.',
                                    'Answer with the option’s letter from the given choices directly.')                   
        elif dataset in ['MME']:
            prompt += '\nGive a very brief answer.'
        elif dataset in ['MMVet']:
            prompt += "\nProvide a step-by-step solution to the problem carefully. Conclue with 'The answer is ...'"
        elif dataset in ['MathVision', 'MathVision_MINI']:
            if 'Choices:' in prompt or "Options:" in prompt:
                prompt += "\nPlease show step-by-step reasoning, and answer the question with the option's letter from the given choices directly."
            else:
                prompt += '\nPlease show step-by-step reasoning, and answer the question with a number directly.'
        elif dataset in ['MMStar']:
            prompt = prompt.replace('Please select the correct answer from the options above.',
                                    "Please read the multiple-choice question carefully, reason step by step, and conclude by stating 'The answer is' followed by the correct letter only.")               
        # 幻觉
        elif dataset in ['HallusionBench', 'POPE']:
            if thinking_mode:
                prompt += '\nConclude with "The answer is yes" or "The answer is no" /think'
            else:
                prompt += '\nConclude with "The answer is yes" or "The answer is no", and give a brief explanation.'
        elif dataset in ['WildVision']:
            prompt += '\nAnswer this question in detail.'
        return prompt

    def process_img_content(self, tokenizer, processor, max_size, patch_size, content, hd=False):
        # 首先按照'\n<img>\S+?</img>\n'进行切分
        pix_value, img_flag, ids = [], [], []
        splits = re.split(r'(<img>\S+?</img>)',content)

        for s in splits:
            if re.match('<img>\S+?</img>',s):
                # 提取出图片的路径
                img_path = re.search(r'<img>(\S+)</img>',s).group(1)
                # img_path = img_path.replace('workspace','data')
                if os.path.exists(img_path):
                    # 读取图片
                    img = None  
                    for _ in range(3):
                        try:
                            img = Image.open(img_path).convert("RGB") # 这里偶尔读取失败
                            break
                        except Exception as e:
                            print(f"Error encountered while reading image! image_path: {img_path}, error: {e}")
                            time.sleep(1)
                    assert img is not None, f"Failed to read image: {img_path}"

                    # resize image to expand
                    if self.image_scale != 1:
                        img = img.resize((int(img.size[0] * self.image_scale),int(img.size[1] * self.image_scale)), Image.BILINEAR)
                    if self.min_size is not None:
                        min_pixels = int(self.min_size) ** 2
                        img_pixels = img.size[0] * img.size[1]
                        if img_pixels < min_pixels:
                            rate = math.sqrt(min_pixels / img_pixels)
                            img = img.resize((int(img.size[0]*rate),int(img.size[1]*rate)), Image.BILINEAR)
                    w, h = img.size
                    base = patch_size*2
                    w1, h1 = w+base-w%base, h+base-h%base
                    if (not hd) or w1*h1<=self.max_size**2:
                        # 不需要切图的场景
                        img = native_preprocess(img, self.max_size, 2*patch_size, tuple(int(x*255) for x in processor.image_mean),min_tokens=4)
                        pix = processor(img, return_tensors='pt').pixel_values.squeeze(0)
                        pix_value.append(pix)
                        vision_token_num = pix.shape[-2]*pix.shape[-1]//(4*patch_size*patch_size)
                        t = tokenizer.encode('<img>{}</img>'.format(tokenizer.eos_token*vision_token_num), add_special_tokens=False)
                        img_flag.extend([1 if ti==tokenizer.eos_token_id else 0 for ti in t])
                        ids.extend(t)
                    else:
                         #NOTE: 对于切图的场景，这里我们写死了，是按照448*448的大小来切。并且最多切9张。
                        s = int(os.environ.get("slice_num", 3))
                        max_size = int(os.environ.get("slice_size", 448))
                        best_w, best_h  = cal_num_of_slices(org_width=img.width, org_height=img.height,
                                                        max_area=max_size**2, max_scale=s*s) 
                        org_img = img
                        if best_w>1 or best_h>1:
                            # TODO: 这两个操作，可以融合一下。
                            # 保证有效面积达到max_size*max_size*best_w*best_h
                            img = resize_to_area(org_img, best_w*best_h*max_size*max_size)
                            # 保证子图的边长是可以被patch_size整除的
                            img = pad_to_max(img, best_w, best_h, 2*self.patch_size, tuple(int(x*255) for x in self.processor.image_mean))
                            # 进行切分 
                            slices = slice_image_wh(img, best_w, best_h)
                        else:
                            slices = []
                        # NOTE: 这里我们是对原图进行的max_preprocess，因为img已经被padding过了，二次padding的时候是无法获取原始图片的padding的。
                        global_img = max_preprocess(org_img, max_size if len(slices)>1 else self.max_size, 2*self.patch_size, tuple(int(x*255) for x in self.processor.image_mean),upper=True)
                        images = slices+[global_img]
                        # 先把第一张图片放进去
                        pix = self.processor(images[0], return_tensors='pt').pixel_values.squeeze(0)
                        pix_value.append(pix)
                        vision_token_num = pix.shape[-2]*pix.shape[-1]//(4*self.patch_size*self.patch_size)
                        if len(slices)!=0:
                            t = self.tokenizer.encode('<img>{}'.format(self.tokenizer.eos_token*vision_token_num), add_special_tokens=False)
                        else:
                            t = self.tokenizer.encode('<img>{}</img>'.format(self.tokenizer.eos_token*vision_token_num), add_special_tokens=False)
                        img_flag.extend([1 if ti==self.tokenizer.eos_token_id else 0 for ti in t])
                        ids.extend(t)
                        for iid, img in enumerate(images[1:],1):
                            pix = self.processor(img, return_tensors='pt').pixel_values.squeeze(0)
                            pix_value.append(pix)
                            vision_token_num = pix.shape[-2]*pix.shape[-1]//(4*self.patch_size*self.patch_size)
                            if iid!=len(images)-1:
                                if iid%best_w!=0:
                                    t = self.tokenizer.encode(',{}'.format(self.tokenizer.eos_token*vision_token_num), add_special_tokens=False)
                                else:
                                    t = self.tokenizer.encode('.{}'.format(self.tokenizer.eos_token*vision_token_num), add_special_tokens=False)
                                assert len(t)==vision_token_num+1
                            else:
                                # 末尾的图
                                t = self.tokenizer.encode('\n{}</img>'.format(self.tokenizer.eos_token*vision_token_num), add_special_tokens=False)
                                assert len(t)-len(self.tokenizer.encode('\n</img>', add_special_tokens=False))==vision_token_num
                            img_flag.extend([1 if ti==self.tokenizer.eos_token_id else 0 for ti in t])
                            ids.extend(t)
                else:
                    print(f'WARN: file {img_path} not exists')
                    raise Exception()
            else:
                t = tokenizer.encode(s, add_special_tokens=False)
                img_flag.extend([0 for _ in range(len(t))])
                ids.extend(t)
        return pix_value, img_flag, ids

    def set_image_size(self, dataset):
        # set image expand scale
        set_expand_2 = set(['InfoVQA_VAL', 'InfoVQA_TEST', 'AI2D_TEST', 'AI2D_TEST_NO_MASK', 'SEEDBench2_Plus', 'CCBench',
                            'RealWorldQA', 'MMVet', 'MME', 'MMMU', 'OCRBench'])
        set_expand_4 = set(['MMBench_DEV_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11'])
        if dataset in set_expand_2:
            default_image_scale = 2
        elif dataset in set_expand_4:
            default_image_scale = 4
        else:
            default_image_scale = 1
        self.image_scale = float(os.environ.get('image_scale', default_image_scale))

        # set image max_size and min_size
        self.max_size=int(os.environ.get("max_size", 1792))
        if dataset in ['DynaMath']:   # worst case 精度影响较大
            self.max_size = 3000
            print("Image max_size set 3000 for DynaMath.")
        self.min_size=os.environ.get("min_size", None)
        print(f"{dataset} dataset uses max_size={self.max_size}, min_size={self.min_size} and image_scale={self.image_scale}")

    @torch.inference_mode()
    def batch_generate(self, messages, dataset=None, verbose=True, post_process=True):
        if os.environ.get("hd",False)=="True":
            raise KeyError("高分辨率切图方案已经废弃！！！")
            # return self.batch_generate_hd(messages, dataset)
        self.set_image_size(dataset)

        all_input_ids = []
        all_image_flags = []
        all_pixel_values = []
        all_attention_masks = []
        prompt_list = []
        for message in messages:
            # set prompts for different datasets
            prompt = self.message_to_prompting(message)
            prompt = self.add_custom_prompt(prompt, dataset)
            if os.environ.get("completion", 'False')!="True":
                prompt = f"<|im_start|>user\n{prompt}{self.tokenizer.eos_token}\n<|im_start|>assistant\n" + os.environ.get("extra_start","")
            prompt_list.append(prompt)
            pixel_values, image_flags, input_ids = self.process_img_content(self.tokenizer, self.processor, self.max_size, self.patch_size, prompt)
            pixel_values = [p.bfloat16().to(self.device) for p in pixel_values]
            input_ids = torch.tensor(input_ids)
            image_flags = torch.tensor(image_flags)
            all_input_ids.append(input_ids)
            all_image_flags.append(image_flags)
            all_pixel_values.extend(pixel_values) # NOTE: 这里我们是extend
            all_attention_masks.append(torch.ones_like(input_ids))
        all_input_ids1 = pad_sequence_left(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        all_image_flags1 = pad_sequence_left(all_image_flags, batch_first=True, padding_value=0)
        all_attention_masks1 = pad_sequence_left(all_attention_masks, batch_first=True, padding_value=0)
        
        # 这个是针对zero3保证的padding
        global_len = get_global_max_pixel_values_len(len(all_pixel_values))
        for i in range(global_len-len(all_pixel_values)):
            all_pixel_values.append(torch.zeros(3, 2*self.patch_size, 2*self.patch_size).type_as(all_pixel_values[0]))
        
        with torch.autocast("cuda",torch.bfloat16):
            # streamer = TextStreamer(tokenizer,skip_prompt=True)
            streamer = None
            outputs = self.model.generate(input_ids=all_input_ids1.to(self.device),image_flags=all_image_flags1.to(self.device),pixel_values=all_pixel_values,synced_gpus=dist.is_initialized(),
                                eos_token_id=self.tokenizer.eos_token_id,attention_mask=all_attention_masks1.to(self.device),streamer=streamer,**self.kwargs)
        outputs = outputs[:, all_input_ids1.shape[1]:]
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        if os.environ.get('post_process', 'True') == 'False':
            post_process = False
        if post_process:
            if '</think>' in responses[0]:
                responses = [output[re.search(r'</think>', output).end():].strip() if re.search(r'</think>', output) else output for output in responses]
            if "<answer>" in responses[0]:
                responses = [re.findall('<answer>(.*?)</answer>', output)[0] if re.findall('<answer>(.*?)</answer>', output) else output for output in responses]
                responses = [re.findall('The answer is (.{1,100})', output, re.M)[0] if re.findall('The answer is (.{1,100})', output, re.M) else output for output in responses]
                responses = [re.findall('the answer is (.{1,100})', output, re.M)[0] if re.findall('the answer is (.{1,100})', output, re.M) else output for output in responses]
                
                responses = [output.rstrip('.') if output.strip().endswith('.')  else output for output in responses]
        if os.environ.get('verbose', 'True') == 'False':
            verbose = False
        if verbose:
            for idx in range(min(5, len(responses))):   # log太多，只展示5个
                print(f"\nQuery:{prompt_list[idx]}\nAnswer:{responses[idx]}\n")
        return responses

    @torch.inference_mode()
    def batch_generate_hd(self, messages, dataset=None):
        # 初始化空列表来存储所有输入的ID、图像标志、像素值和注意力掩码
        all_input_ids = []
        all_image_flags = []
        all_pixel_values = []
        all_attention_masks = []
        # 遍历每个消息
        s = 3
        max_size = 448
        for message in messages:
            prompt = self.message_to_prompting(message)
            if self.extra_prompt:
                prompt = prompt.strip() + "\n" + self.extra_prompt.strip()
            prompt = f"<|im_start|>user\n{prompt}{self.tokenizer.eos_token}\n<|im_start|>assistant\n"
            pixel_values, image_flags, input_ids = self.process_img_content(self.tokenizer, self.processor, max_size, self.patch_size, prompt, hd=True)
            pixel_values = [p.bfloat16().to(self.device) for p in pixel_values]
            input_ids = torch.tensor(input_ids)
            image_flags = torch.tensor(image_flags)
            all_input_ids.append(input_ids)
            all_image_flags.append(image_flags)
            all_pixel_values.extend(pixel_values)
            all_attention_masks.append(torch.ones_like(input_ids))
        all_input_ids1 = pad_sequence_left(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        all_image_flags1 = pad_sequence_left(all_image_flags, batch_first=True, padding_value=0)
        all_attention_masks1 = pad_sequence_left(all_attention_masks, batch_first=True, padding_value=0)
        
        # 这个是针对zero3保证的padding
        global_len = get_global_max_pixel_values_len(len(all_pixel_values))
        for i in range(global_len-len(all_pixel_values)):
            all_pixel_values.append(torch.zeros(3, 2*self.patch_size,  2*self.patch_size).type_as(all_pixel_values[0]))
        
        with torch.autocast("cuda",torch.bfloat16):
            streamer = None
            outputs = self.model.generate(input_ids=all_input_ids1.to(self.device),image_flags=all_image_flags1.to(self.device),pixel_values=all_pixel_values,synced_gpus=True,
                                eos_token_id=self.tokenizer.eos_token_id,attention_mask=all_attention_masks1.to(self.device),streamer=streamer,**self.kwargs)
        outputs = outputs[:, all_input_ids1.shape[1]:]
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return responses

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)