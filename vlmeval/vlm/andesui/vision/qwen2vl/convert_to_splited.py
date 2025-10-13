"""
这里我们导入一个qwen2-vl的模型，会输出三个文件：
vit/mlp/llm
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoTokenizer

class M(torch.nn.Module):
    def __init__(self,model,lm_head):
        super().__init__()
        self.model = model
        self.lm_head = lm_head

class V(torch.nn.Module):
    def __init__(self,model, ln):
        super().__init__()
        self.vision_model = model
        self.ln = ln

def conv3d_to_linear(conv3d_layer):
    # 获取原始的权重
    conv3d_weight = conv3d_layer.weight.data  # 形状为 (1280, 3, 2, 14, 14)
    # 对时间维度上的权重求和
    new_weight = conv3d_weight.sum(dim=2)  # 形状为 (1280, 3, 14, 14)
    # 展平成二维权重矩阵
    new_weight = new_weight.view(1280, -1)  # 形状为 (1280, 588)
    print(f"new_weight 的形状为：{new_weight.shape}")  # 应该输出 torch.Size([1280, 588])
    # 创建新的 Linear 层
    linear_layer = nn.Linear(14 * 14 * 3, 1280, bias=False)
    # 设置 Linear 层的权重
    linear_layer.weight.data = new_weight
    return linear_layer

vl = Qwen2VLForConditionalGeneration.from_pretrained(
    "/mnt/data/group/models/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
vl_llm = vl.model
#TODO: 下次记得保存的目录不要使用vit和llm，加上qwen的标识。
#TODO: vit也可以加上rope base的参数。
"""保存llm的部分。#TODO:下次保存胡时候，注意保存bfloat16的权重。(以及我们可能需要修改一下RoPE的base),
llm = AutoModelForCausalLM.from_pretrained("/mnt/data/group/models/Qwen2-7B-Instruct", torch_dtype=torch.bfloat16)
m = M(vl_llm, vl.lm_head)
llm.load_state_dict(m.state_dict())
llm.save_pretrained("/mnt/data/group/wangnan/model/Qwen2-VL-Separate/7B/qwen2_llm")
#NOTE: 必须要是slow的tokenizer，不然无法修改底层变量。
tk = AutoTokenizer.from_pretrained("/mnt/data/group/models/Qwen2-VL-7B-Instruct", use_fast=False)
additional_special_tokens = ['<|im_start|>',
                            '<|im_end|>',
                            '<|object_ref_start|>',
                            '<|object_ref_end|>',
                            '<|box_start|>',
                            '<|box_end|>',
                            '<|quad_start|>',
                            '<|quad_end|>',
                            '<img>',
                            '</img>',
                            '<|vision_pad|>',
                            '<|image_pad|>',
                            '<|video_pad|>']
tk._additional_special_tokens  = additional_special_tokens
tk.init_kwargs['additional_special_tokens'] = additional_special_tokens
tk.added_tokens_decoder[151652].content = '<img>'
tk.added_tokens_decoder[151653].content = '</img>'
tk.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
tk.save_pretrained("/mnt/data/group/wangnan/model/Qwen2-VL-Separate/7B/qwen2_llm")
"""

"""mlp的部分
mlp = vl.visual.merger.mlp
torch.save(mlp.state_dict(), "/mnt/data/group/wangnan/model/Qwen2-VL-Separate/7B/mlp.bin")
"""

"""vit的部分
#注意这里需要对ConV3D进行转换
visual = vl.visual
ln = visual.merger.ln_q
del visual.merger
visual.patch_embed.proj = conv3d_to_linear(visual.patch_embed.proj)
v = V(visual, ln)
torch.save(v.state_dict(), "/mnt/data/group/wangnan/model/Qwen2-VL-Separate/7B/qwen2_vision/pytorch_model.bin")
"""