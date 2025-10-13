"""
主要的变化：
1. 没有强制放大两倍。
2. 修复了图片粘贴没有居中的问题。
3. interleaved的输入暂不支持（主要是处理逻辑为了利用了单图的特性，不好改）
4. /mnt/data/group/wangnan/code/GPT4oEval_wn/infer_andesvl_ds2.py的bug，这里的代码可能存在bug。
"""
import json
import torch
import os
import warnings
from .base import BaseModel
import re
from PIL import Image
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
sys.path.append("/mnt/data/group/wangnan/model/InternVit-6B-980px-V1.5-Navit-Leaf")
# from modeling_intern_vit_navit_leaf_cat_packed_flash import  InternVisionModel
sys.path.append("/mnt/data/group/wangnan/model/InternVit-300M-980px-V2-Navit-Leaf")
# from modeling_intern_vit_ln_navit_leaf_cat_packed_flash import InternVisionModel as InternVisionModelLN

class AndesVLModel(nn.Module):
    INTERLEAVE = True
    def __init__(
        self,
        vit_path,
        llm_path,
        mlp_path=None,
        zero_init=False,
    ):
        super().__init__()
        self.patch_size = 14  # 这个是写死了的
        if zero_init:
            if "300m" in vit_path.lower():
                self.vision_encoder = InternVisionModelLN.from_pretrained(
                    vit_path, torch_dtype=torch.bfloat16
                )
            else:
                self.vision_encoder = InternVisionModel.from_pretrained(
                    vit_path, torch_dtype=torch.bfloat16
                )
            self.language_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
        else:
            if "300m" in vit_path.lower():
                self.vision_encoder = InternVisionModelLN.from_pretrained(
                    vit_path, torch_dtype=torch.bfloat16, device_map={"": 0}
                )
            else:
                self.vision_encoder = InternVisionModel.from_pretrained(
                    vit_path, torch_dtype=torch.bfloat16, device_map={"": 0}
                )
            self.language_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        vit_hidden_size, llm_hidden_size = (
            self.vision_encoder.config.hidden_size,
            self.language_model.config.hidden_size,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * 4),
            nn.Linear(vit_hidden_size * 4, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self.mlp.to(0)
        if os.path.exists(mlp_path):
            self.mlp.load_state_dict(torch.load(mlp_path))

    def pixel_shuffle(self, x):
        n, w, h, c = x.size()
        x = x.reshape(n, w, h // 2, c * 2)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(n, h // 2, w // 2, c * 4)
        return x

    def get_vit_embeds_and_merge(self, pixel_values, input_embeds, image_flags):
        org_sizes = [(p.shape[1], p.shape[2]) for p in pixel_values]
        tokens = [h * w // (self.patch_size * self.patch_size) for h, w in org_sizes]
        batch_size = math.floor(sum(tokens) / max(tokens))
        # TODO:NOTE:需要check一下，如果说，某个gpu没有的情况怎么处理
        vit_embeds = self.vision_encoder.vision_model(
            pixel_values=pixel_values, batch_size=batch_size
        ).last_hidden_state
        #NOTE: 这个操作仅仅在zero3 推理的时候需要。训练的时候，我们是有额外的处理逻辑
        img_num = sum(count_images(f) for f in image_flags)
        vit_embeds = vit_embeds[:img_num]
        vit_embeds = [
            vit_embed.reshape(
                1,
                org_size[0] // self.patch_size,
                org_size[1] // self.patch_size,
                vit_embed.shape[-1],
            )
            for vit_embed, org_size in zip(vit_embeds, org_sizes)
        ]
        vit_embeds = [self.pixel_shuffle(vit_embed) for vit_embed in vit_embeds]
        vit_embeds = [
            vit_embed.reshape(-1, vit_embed.shape[-1]) for vit_embed in vit_embeds
        ]  # [ (Lvi, Hv) ...]
        vit_embeds = torch.cat(vit_embeds, dim=0)  # (Lv1+Lv2+..., Hv)
        vit_embeds = self.mlp(vit_embeds)  #
        # 下面需要替换特征，需要先把文本和视觉嵌入转换为2D张量
        assert (
            vit_embeds.shape[0] == image_flags.sum()
        ), f"{vit_embeds.shape[0]}, {image_flags.sum()}"
        Bt, Lt, Ht = input_embeds.shape
        input_embeds = input_embeds.reshape(-1, Ht)
        image_flags = image_flags.view(-1)
        input_embeds[image_flags == 1] = vit_embeds
        input_embeds = input_embeds.view(Bt, Lt, Ht)  # 恢复形状
        return input_embeds

    @torch.no_grad()
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

        input_embeds = self.language_model.get_input_embeddings()(
            input_ids
        )  # (Bt, Lt, Ht)
        if image_flags != None and (image_flags == 1).sum() > 0:
            input_embeds = self.get_vit_embeds_and_merge(
                pixel_values, input_embeds, image_flags
            )
        outputs = self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )
        return outputs

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

def resize_to_area(img, area):
    "Resize image so that the area is `area`"
    ratio = (area / img.size[0] / img.size[1]) ** 0.5
    new_sz = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    return img.resize(new_sz)

def get_scaled_img_size(image_size, max_area, base, max_resolution=980, upper=False):
    '''计算缩放后的图片大小和包裹矩形的大小'''
    # 计算原始图片的宽高比
    aspect_ratio = image_size[0] / image_size[1]
    # 计算包裹矩形的最大可能宽度和高度【这里的包裹矩形是指面积为max_area，且宽高比和待处理图片面积一致的矩形】
    max_width = math.floor(math.sqrt(max_area * aspect_ratio))
    max_height = math.floor(math.sqrt(max_area / aspect_ratio))
    max_width, max_height = min(max_width, max_resolution), min(max_height, max_resolution)
    max_width, max_height = max(max_width, base), max(max_height, base)
    # 确保包裹矩形的宽度和高度都是base的整数倍
    if not upper:
        # 向下取整, 保证面积不会超过max_area
        max_width = max_width - max_width % base
        max_height = max_height - max_height % base
    else:
        # 向上取整，同时不超过max_resolution
        max_width = min(max_width + (base - max_width % base), max_resolution)
        max_height = min(max_height + (base - max_height % base), max_resolution)
    # 计算缩放因子
    scale_factor = min(max_width / image_size[0], max_height / image_size[1])
    # 计算缩放后的图片大小
    new_image_size = (round(image_size[0] * scale_factor), round(image_size[1] * scale_factor))
    # 计算包裹矩形的大小
    bounding_box_size = (max_width, max_height)
    return new_image_size, bounding_box_size

def max_preprocess(img, max_size, base, background_color, max_resolution=980, upper=False, force_resize=False):
    '''对图片进行预处理，使其面积接近max_size**2'''
    # 首先把图片resize到长度和宽度都低于max_resolution
    w, h = img.size
    if max(w, h) > max_resolution:
        scale = max_resolution / max(w, h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h))
    # 获取缩放后的图片大小和包裹矩形的大小
    new_image_size, bounding_box_size = get_scaled_img_size((w, h), max_size**2, base, max_resolution, upper)
    
    if force_resize:
        # 强制调整大小
        return img.resize(bounding_box_size)
    
    # 创建一个新的画布
    canvas = Image.new('RGB', bounding_box_size, background_color)
    # 计算将图像粘贴到画布上的位置
    paste_width = (bounding_box_size[0] - new_image_size[0]) // 2
    paste_height = (bounding_box_size[1] - new_image_size[1]) // 2
    # 将图像粘贴到画布上
    canvas.paste(img.resize(new_image_size), (paste_width, paste_height))
    return canvas

def native_preprocess(img, max_size, base, background_color, max_resolution=980, min_tokens=1, force_resize=False):
    # 对图片进行处理，使其宽度和高度都是base的整数倍
    # 如果图片的最长边超过max_resolution，就把图片resize到max_resolution以内
    w, h = img.size
    # 首先保证图片的最长边不超过max_resolution(ViT在极限长度)
    if max(w, h) > max_resolution:
        scale = max_resolution / max(w, h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h))
    # 获取直接向上取整的宽度和高度
    w1, h1 = w + base - w % base, h + base - h % base
    if w1 * h1 > max_size ** 2:
        return max_preprocess(img, max_size, base, background_color, max_resolution, force_resize=force_resize)
    if w1 * h1 < (base * base * min_tokens):
        return max_preprocess(img, int(base * (min_tokens ** 0.5)), base, background_color, max_resolution, upper=True, force_resize=force_resize)  # NOTE:对于过小的图片，我们向上取
    if w1 == w and h1 == h:
        return img
    else:
        if force_resize:
            img = img.resize((w1, h1))
            return img
        else:
            # 创建一个新的(w1, h1)的画布，并把图片resize保证只有一侧存在白边的情况
            scale = min(w1 / w, h1 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h))
            canvas = Image.new('RGB', (w1, h1), background_color)
            canvas.paste(img, ((w1 - new_w) // 2, (h1 - new_h) // 2))
            return canvas

def cal_num_of_slices(org_width, org_height, max_area, max_scale=9):
    '''计算宽度和高度的切片数，这里直接按照子图最接近正方形的策略来切割'''
    org_area = org_height*org_width
    scale = org_area/(max_area)  
    scale = math.ceil(scale)
    scale = min(scale, max_scale)
    def factorize(n):
        factors = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append((i/(n/i), i, n // i))
        return factors
    available_ratios = []
    log_origin_ratio = math.log(org_width/org_height)
    if scale==1:
        available_ratios = factorize(scale) #这里只有1*1
    elif scale==2:
        available_ratios = factorize(scale) + factorize(scale+1) #这里只有1*2
    elif scale==max_scale:
        available_ratios = factorize(scale-1) + factorize(scale)
    else:
        available_ratios = factorize(scale-1) + factorize(scale) + factorize(scale+1)
    min_dif = 1000 
    best_w = 0
    best_h = 0
    for (r,w_slice,h_slice) in available_ratios:
        log_r = math.log(r)
        if min_dif > abs(log_r - log_origin_ratio):
            min_dif = abs(log_r - log_origin_ratio)
            best_w = w_slice
            best_h = h_slice
    return best_w,best_h  


def slice_image_wh(image, best_w, best_h):
    origin_image_width  = image.size[0]
    origin_image_height = image.size[1]
    assert origin_image_width%best_w==0 and origin_image_height%best_h==0
    slices = []
    for j in range(best_h):
        for i in range(best_w):
            box = (i * origin_image_width//best_w, j * origin_image_height//best_h, (i + 1) * origin_image_width//best_w, (j + 1) * origin_image_height//best_h)
            region = image.crop(box).convert("RGB")
            slices.append(region)
    return slices


def pad_to_max(image, best_w, best_h, base, background_color=(0,0,0)):
    '''对图片进行处理，使其宽度被best_w*base整除，高度被best_h*base整除（对高分辨率的大图切割之前的预处理）'''
    # NOTE: 这里返回的图片，是比原图更大的，因为我们只是padding，以保证子图
    width, height = image.size
    base_w = best_w * base
    base_h = best_h * base
    if width % base_w == 0 and height % base_h == 0:
        return image
    target_width = int(math.ceil(width / base_w) * base_w)
    target_height = int(math.ceil(height / base_h) * base_h)
    # 创建一个新的画布
    new_image = Image.new('RGB', (target_width, target_height), background_color)
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
    global_len = torch.tensor([current_rank_len], dtype=torch.int64, device='cuda')
    dist.all_reduce(global_len, op=dist.ReduceOp.MAX)
    return global_len.item()

class AndesVL_V0(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path, **kwargs):
        #model_path只是一个占位符，实际上我们会从环境变量中获取模型路径（因为文件嵌套太多，主程序传参不方便）
        self.max_size = int(os.environ.get('max_size',448))
        self.patch_size = 14
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
        
        zero_init = True
        ds_config = {
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
            },
            "train_micro_batch_size_per_gpu": 1,
        }
        if not zero_init:
            self.model = AndesVLModel(vit_path=self.vit_path, llm_path=self.llm_path, mlp_path=self.mlp_path, local_rank=device)
        else:
            from transformers.deepspeed import HfDeepSpeedConfig
            hfdsc = HfDeepSpeedConfig(ds_config)
            self.model = AndesVLModel(vit_path=self.vit_path, llm_path=self.llm_path, mlp_path=self.mlp_path, zero_init=True)
        ds_engine = deepspeed.initialize(model=self.model, config_params=ds_config)[0]
        ds_engine.module.eval()
        self.model  = ds_engine.module

        self.processor = CLIPImageProcessor.from_pretrained(self.vit_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True, use_fast=False)
        assert "<|im_end|>" in self.tokenizer.get_vocab()
        if self.tokenizer.eos_token!="<|im_end|>": 
            self.tokenizer.eos_token = "<|im_end|>"
        kwargs_default = dict(do_sample=False, max_new_tokens=int(os.environ.get("max_new_tokens",16)), top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.extra_prompt = os.environ.get("extra_prompt","").strip()
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
    
    def message_to_promptimg(self, message):
        s = []
        for m in message:
            if m['type']=="image":
                s.append(f"<img>{m['value']}</img>")
            elif m['type']=='text':
                s.append(m['value'])
            else:
                raise Exception(m)
        return "\n".join(s)

    def process_img_content(self, tokenizer, processor, max_size, patch_size, content, hd=False):
        # 首先按照'\n<img>\S+?</img>\n'进行切分
        pix_value, img_flag, ids = [], [], []
        splits = re.split(r'(<img>\S+?</img>)',content)
        for s in splits:
            if re.match('<img>\S+?</img>',s):
                # 提取出图片的路径
                img_path = re.search(r'<img>(\S+)</img>',s).group(1)
                if os.path.exists(img_path):
                    # 读取图片
                    img = Image.open(img_path).convert("RGB")  #这里可能读取失败，暂时不写异常处理了。。。
                    w, h = img.size
                    base = patch_size*2
                    w1, h1 = w+base-w%base, h+base-h%base
                    if (not hd) or w1*h1<=self.max_size**2:
                        #不需要切图的场景
                        img = native_preprocess(img, max_size, 2*patch_size, tuple(int(x*255) for x in processor.image_mean),min_tokens=4)
                        pix = processor(img, return_tensors='pt').pixel_values.squeeze(0)
                        pix_value.append(pix)
                        vision_token_num = pix.shape[-2]*pix.shape[-1]//(4*patch_size*patch_size)
                        t = tokenizer.encode('<img>{}</img>'.format(tokenizer.eos_token*vision_token_num), add_special_tokens=False)
                        img_flag.extend([1 if ti==tokenizer.eos_token_id else 0 for ti in t])
                        ids.extend(t)
                    else:
                         #NOTE: 对于切图的场景，这里我们写死了，是按照448*448的大小来切。并且最多切9张。
                        s = 3
                        max_size = 448
                        best_w, best_h  = cal_num_of_slices(org_width=img.width, org_height=img.height,
                                                        max_area=max_size**2, max_scale=s*s) 
                        org_img = img
                        if best_w>1 or best_h>1:
                            #TODO: 这两个操作，可以融合一下。
                            #保证有效面积达到max_size*max_size*best_w*best_h
                            img = resize_to_area(org_img, best_w*best_h*max_size*max_size)
                            #保证子图的边长是可以被patch_size整除的
                            img = pad_to_max(img, best_w, best_h, 2*self.patch_size, tuple(int(x*255) for x in self.processor.image_mean))
                            #进行切分 
                            slices = slice_image_wh(img, best_w, best_h)
                        else:
                            slices = []
                        #NOTE: 这里我们是对原图进行的max_preprocess，因为img已经被padding过了，二次padding的时候是无法获取原始图片的padding的。
                        global_img = max_preprocess(org_img, max_size if len(slices)>1 else self.max_size, 2*self.patch_size, tuple(int(x*255) for x in self.processor.image_mean),upper=True)
                        images = slices+[global_img]
                        #先把第一张图片放进去
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
                                #末尾的图
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

    @torch.inference_mode()
    def batch_generate(self, messages, dataset=None):
        if os.environ.get("hd",False)=="True":
            return self.batch_generate_hd(messages, dataset)
        all_input_ids = []
        all_image_flags = []
        all_pixel_values = []
        all_attention_masks = []
        prompt_list = []
        for message in messages:
            prompt = self.message_to_promptimg(message)
            if self.extra_prompt:
                prompt = prompt.strip() + "\n" + self.extra_prompt.strip()
            prompt_list.append(prompt)
            prompt = f"<|im_start|>user\n{prompt}{self.tokenizer.eos_token}\n<|im_start|>assistant\n"
            pixel_values, image_flags, input_ids = self.process_img_content(self.tokenizer, self.processor, self.max_size, self.patch_size, prompt)
            pixel_values = [p.bfloat16().to(self.device) for p in pixel_values]
            input_ids = torch.tensor(input_ids)
            image_flags = torch.tensor(image_flags)
            all_input_ids.append(input_ids)
            all_image_flags.append(image_flags)
            all_pixel_values.extend(pixel_values) #NOTE: 这里我们是extend
            all_attention_masks.append(torch.ones_like(input_ids))
        all_input_ids1 = pad_sequence_left(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        all_image_flags1 = pad_sequence_left(all_image_flags, batch_first=True, padding_value=0)
        all_attention_masks1 = pad_sequence_left(all_attention_masks, batch_first=True, padding_value=0)
        
        #这个是针对zero3保证的padding
        global_len = get_global_max_pixel_values_len(len(all_pixel_values))
        for i in range(global_len-len(all_pixel_values)):
            all_pixel_values.append(torch.zeros(3, 28, 28).type_as(all_pixel_values[0]))
        
        with torch.autocast("cuda",torch.bfloat16):
            #streamer = TextStreamer(tokenizer,skip_prompt=True)
            streamer = None
            outputs = self.model.generate(input_ids=all_input_ids1.to(self.device),image_flags=all_image_flags1.to(self.device),pixel_values=all_pixel_values,synced_gpus=True,
                                eos_token_id=self.tokenizer.eos_token_id,attention_mask=all_attention_masks1.to(self.device),streamer=streamer,**self.kwargs)
        outputs = outputs[:, all_input_ids1.shape[1]:]
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        for p, r in zip(prompt_list, responses):
            print(f"\nQ:{p}\nA:{r}\n")
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
            prompt = self.message_to_promptimg(message)
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
            all_pixel_values.append(torch.zeros(3, 28, 28).type_as(all_pixel_values[0]))
        
        with torch.autocast("cuda",torch.bfloat16):
            streamer = None
            outputs = self.model.generate(input_ids=all_input_ids1.to(self.device),image_flags=all_image_flags1.to(self.device),pixel_values=all_pixel_values,synced_gpus=True,
                                eos_token_id=self.tokenizer.eos_token_id,attention_mask=all_attention_masks1.to(self.device),streamer=streamer,**self.kwargs)
        outputs = outputs[:, all_input_ids1.shape[1]:]
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return responses

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)