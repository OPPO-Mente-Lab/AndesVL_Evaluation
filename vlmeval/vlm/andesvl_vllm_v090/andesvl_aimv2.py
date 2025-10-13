import os
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, Optional, TypedDict, TypeVar, Union, Tuple, List

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import BatchEncoding, PretrainedConfig, TensorType

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from .aimv2_navit_rope import Aimv2VisionModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,MultiModalInputs,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import AnyTokenizer

from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)

from transformers import BatchEncoding, TensorType, AutoTokenizer


IMG_START = '<img>'
IMG_END = '</img>'
IMG_CONTEXT = '<|vision_pad|>'

IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


#NOTE:这里我们存在写死的参数：max_size、patch_size、max_resolution、image_mean、image_std，这些参数暂时是不能通过外部修改的。
class AndesVLImageProcessor:
    def __init__(self, 
                 max_size: int = 1792,
                 patch_size: int = 14,
                 max_resolution: int = 4172,
                 image_mean: Tuple[float, float, float] =(0.48145466, 0.4578275, 0.40821073),
                 image_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)):
        """
        初始化图像预处理器
        """
        self.max_size = max_size
        self.patch_size = patch_size
        self.max_resolution = max_resolution
        self.background_color = tuple(int(x*255) for x in image_mean)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=image_mean, std=image_std)
        ])
        self.base = 2 * patch_size
        self.min_area = self.base * self.base * 4  # 最小4个token

    def load_image(self, source: Union[str, Image.Image]) -> Image.Image:
        """加载图像"""
        if isinstance(source, Image.Image):
            img = source
        elif isinstance(source, str):
            if source.startswith('http'):
                response = requests.get(source)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
            elif os.path.exists(source):
                img = Image.open(source)
            elif source.startswith('data:image'):
                img = Image.open(BytesIO(base64.b64decode(source.split(',')[1])))
            else:
                raise ValueError("Unsupported image source")
        else:
            raise ValueError("Unsupported image source")
        return img.convert('RGB')

    def get_target_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """计算目标尺寸，考虑所有约束"""
        w, h = float(original_size[0]), float(original_size[1])
        image_scale = float(os.environ.get('image_scale', 1.0))
        min_size = int(os.environ.get('min_size', 0))
        max_size = int(os.environ.get('max_size', 1792))
        # print(f"image_scale: {image_scale}, min_size: {min_size}, max_size: {max_size}")

        # 处理图像缩放
        w *= image_scale
        h *= image_scale

        # 首先处理最大分辨率限制
        if max(w, h) > self.max_resolution:
            scale = self.max_resolution / max(w, h)
            w *= scale
            h *= scale
        # 确保最小面积
        if w * h < self.min_area:
            scale = math.sqrt(self.min_area / (w * h))
            w *= scale
            h *= scale
        
        # 处理最小尺寸限制
        if w * h < min_size ** 2:
            scale = math.sqrt((min_size ** 2) / (w * h))
            w *= scale
            h *= scale

        # 处理最大尺寸限制
        if w * h > max_size ** 2:
            scale = math.sqrt((max_size ** 2) / (w * h))
            w *= scale
            h *= scale
        # 确保是patch_size的整数倍并转为整数
        w = math.ceil(w / self.base) * self.base
        h = math.ceil(h / self.base) * self.base
        return int(w), int(h)
    
    def get_image_tokens(self, w, h):
        w, h = self.get_target_size((w, h))
        return (w // self.base) * (h // self.base)
    
    def get_max_andesvl_image_tokens(self) -> int:
        max_size = int(os.environ.get('max_size', 1792))
        aligned_dim = math.ceil(max_size / self.base) * self.base
        return (aligned_dim // self.base) ** 2
     
    def expand_image_tokens(self, text: str, image_tokens: list[int]) -> str:
        def replacement(m):
            token_count = image_tokens.pop(0)
            return f"<img>{'<|vision_pad|>' * token_count}</img>"
        return re.sub(r'<image>', replacement, text)

    def process(self, image_source: Union[str, Image.Image], do_transform: bool = True) -> torch.Tensor | Image.Image:
        """
        处理图像的主函数
        Returns: 处理后的张量 [C, H, W]
        """
        # 加载图像
        img = self.load_image(image_source)
        # 计算目标尺寸
        target_w, target_h = self.get_target_size(img.size)
        if target_w != img.size[0] or target_h != img.size[1]:
            # 创建目标画布
            canvas = Image.new("RGB", (target_w, target_h), self.background_color)
            # 计算保持纵横比的resize尺寸
            scale = min(target_w/img.size[0], target_h/img.size[1])
            resize_w = img.size[0] * scale
            resize_h = img.size[1] * scale
            # 只进行一次resize，在最后转为整数
            img_resized = img.resize((int(resize_w), int(resize_h)))
            # 居中粘贴
            paste_x = int((target_w - resize_w) / 2)
            paste_y = int((target_h - resize_h) / 2)
            canvas.paste(img_resized, (paste_x, paste_y))
            img = canvas
        return self.transform(img) if do_transform else img
    
    def get_flated_pixel_values(self, pixel_values: list):
        #NOTE:这个可以在数据预处理的时候实现，也可以在model内部的预处理实现，这里冗余了。
        flated_pixel_values = []
        image_grid_hw = []
        for pv in pixel_values:
            #获取图片的宽高
            c, h, w = pv.shape
            image_grid_hw.append((h//self.patch_size, w//self.patch_size))
            fpv = pv.reshape(c, h//(2*self.patch_size), 2, self.patch_size, w//(2*self.patch_size), 2, self.patch_size)
            flated_pixel_values.append(fpv.permute(1, 4, 2, 5, 0, 3, 6).reshape(-1, c*self.patch_size*self.patch_size))
        flated_pixel_values = torch.cat(flated_pixel_values, dim=0) # (Len_img, C, H, W)
        image_grid_hw = torch.tensor(image_grid_hw, device=flated_pixel_values.device) # (N_img, 2)
        return flated_pixel_values, image_grid_hw

image_processor = AndesVLImageProcessor(max_size=int(os.environ.get("max_size", 1792)))


class AndesVLProcessor:
    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_CONTEXT]

    def __call__(
        self,
        text: Optional[str] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> Mapping[str, NestedTensors]:
        #NOTE: vllm经常只传带<image>的prompt但是不传images，不知道目的是什么。
        assert isinstance(text, str)
        #if text:
        #    print(f"\nTEXT:{text}:TEXT\n")
        #1. 批量化处理
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            image_inputs = {}
        else:
            base = image_processor.base
            pixel_values = [image_processor.process(img) for img in images]
            image_tokens = [pv.shape[1]*pv.shape[2]//(base*base) for pv in pixel_values]
            text = image_processor.expand_image_tokens(text, image_tokens)
            flated_pixel_values, image_grid_hw = image_processor.get_flated_pixel_values(pixel_values)
            image_inputs = {"pixel_values": pixel_values, 'flated_pixel_values': flated_pixel_values, 'image_grid_hw':image_grid_hw}
        #这里实际的文本输入是一个字符串，因为是_call_hf_processor调用的，所以实际是1个str+n张图片。
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        #编码文本
        text_inputs = self.tokenizer(text, add_special_tokens=False)
        #NOTE:
        #1. 整体的处理逻辑和之前版本的vLLM还是类似的。
        return {
            **BatchEncoding(text_inputs, tensor_type=return_tensors),
            **image_inputs,
        }


class AndesVLProcessingInfo(BaseProcessingInfo):

    #NOTE:这里实际返回的是我们在这个文件中定义的AndesVLProcessor，而不是huggingface的processor
    def get_hf_processor(
        self,
        **kwargs: object,
    ) -> AndesVLProcessor:
        return self.ctx.init_processor(
            AndesVLProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        return image_processor.get_image_tokens(image_width, image_height)


# 针对多模态数据，需要实现get_dummy_text和get_dummy_mm_data两个方法。
class AndesVLDummyInputsBuilder(BaseDummyInputsBuilder[AndesVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return "<image>" * num_images
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        max_size = int(os.environ.get('max_size', 1792))
        target_width = target_height = max_size
        num_images = mm_counts.get("image", 0)
        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class AndesVLMultiModalProcessor(BaseMultiModalProcessor[AndesVLProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )
        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        image_token_id = hf_processor.image_token_id
        processed_outputs["image_token_id"] = torch.tensor(image_token_id)
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_hws = hf_inputs.get("image_grid_hw", torch.empty((0, 2)))
        image_grid_sizes = image_grid_hws.prod(-1)
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"), #这里batched对于list的张量是怎么处理的，我还不太清楚，是否需要替换为flat呢？
            #flated_pixel_values=MultiModalFieldConfig.flat("image"),
            flated_pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes),
            image_grid_hw=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )


    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        def get_replacement_andesvl(item_idx: int):
            #pvs = out_mm_kwargs['pixel_values'].shape
            #grid_hw = out_mm_kwargs['image_grid_hw'][item_idx]
            #assert pvs[0]==1
            assert isinstance(item_idx, int), f"{item_idx}"
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)
            num_image_tokens = self.info.get_num_image_tokens(
                image_width=image_size.width,
                image_height=image_size.height,
            )
            return  f"<img>{'<|vision_pad|>' * num_image_tokens}</img>"

        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=get_replacement_andesvl,
            )
        ]

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        """
        Process multi-modal inputs to be used in vLLM.

        The main steps are:

        1. Apply HF Processor on prompt text and multi-modal data together,
           outputting token IDs and processed tensors.
        2. Find and update sequences in the token IDs with placeholder tokens.
           The number of placeholder tokens equals the feature size of the
           multi-modal data outputted by the multi-modal encoder.
        3. Extract information about the placeholder tokens from the
           processed token IDs.
        """
        #import datetime
        #print(f"$DATE: {datetime.datetime.now()} INFO:{prompt}￥")
        mm_items = self._to_mm_items(mm_data)

        (
            prompt_ids,
            mm_kwargs,
            mm_hashes,
            is_update_applied,
        ) = self._cached_apply_hf_processor(
            prompt,
            mm_items,
            hf_processor_mm_kwargs,
            return_mm_hashes=return_mm_hashes,
        )
        #NOTE: 这里的prompt被更新了。
        prompt_ids, prompt, mm_placeholders = self._maybe_apply_prompt_updates(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            prompt_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            is_update_applied=is_update_applied,
        )

        mm_placeholder_ranges = {
            modality: [item.to_range() for item in placeholders]
            for modality, placeholders in mm_placeholders.items()
        }

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholder_ranges,
        )


@MULTIMODAL_REGISTRY.register_processor(AndesVLMultiModalProcessor,
                                        info=AndesVLProcessingInfo,
                                        dummy_inputs=AndesVLDummyInputsBuilder)
class AndesVLForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        # NOTE: 1. 这里主要是参考vllm中的internvl模块实现的，2. 这里没有考虑量化。
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.vision_encoder = Aimv2VisionModel(config.vision_config)
        self.patch_size = config.vision_config.patch_size
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(vit_hidden_size * 4, vit_hidden_size * 4),
            nn.GELU(),
            nn.Linear(vit_hidden_size * 4, llm_hidden_size),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(vllm_config.model_config.model)
        self.img_context_token_id = self.tokenizer.vocab[IMG_CONTEXT]

    def get_flated_pixel_values(self, pixel_values):
        """
            Args:
                pixel_values: 
        """
        #NOTE: 理论上我们可以在数据处理的时候完成这步，但是考虑到之前代码的复用方便，在此处才序列化。
        #对pixel_values进行处理，变为序列化的图片，其中空间上相邻的四个patch在序列上相邻
        flated_pixel_values = []
        image_grid_hw = []
        for pv in pixel_values:
            #获取图片的宽高
            c, h, w = pv.shape
            image_grid_hw.append((h//self.patch_size, w//self.patch_size))
            fpv = pv.reshape(c, h//(2*self.patch_size), 2, self.patch_size, w//(2*self.patch_size), 2, self.patch_size)
            flated_pixel_values.append(fpv.permute(1, 4, 2, 5, 0, 3, 6).reshape(-1, c*self.patch_size*self.patch_size))
        flated_pixel_values = torch.cat(flated_pixel_values, dim=0) # (Len_img, C, H, W)
        image_grid_hw = torch.tensor(image_grid_hw, device=flated_pixel_values.device) # (N_img, 2)
        return flated_pixel_values, image_grid_hw
    
    def get_multimodal_embeddings(self, **kwargs):
        pixel_values = kwargs['pixel_values']
        pixel_values1 = []
        for i in range(len(pixel_values)): #batch维度
            for j in range(len(pixel_values[i])):  # 样本内的第j张图片
                p = pixel_values[i][j]
                assert isinstance(p, torch.Tensor) and p.dim()==3, "All elements must be tensors"
                pixel_values1.append(p)
        
        flated_pixel_values, image_grid_hw = self.get_flated_pixel_values(pixel_values1)
        vit_embeds = self.vision_encoder(flated_pixel_values, image_grid_hw)  # (Len_img, H_vit)
        vit_embeds = vit_embeds.view(-1, vit_embeds.shape[-1]*4)
        vit_embeds = self.mlp(vit_embeds) 
        #下面是为了应对新版本的vllm的sanity check
        image_grid_sizes = image_grid_hw.prod(-1)//4
        ends = torch.cumsum(image_grid_sizes, 0)
        starts = torch.cat([torch.tensor([0], device=ends.device), ends[:-1]])
        vit_embeds = [vit_embeds[s:e] for s,e in zip(starts, ends)]
        return vit_embeds


    def get_input_embeddings(
        self,
        input_ids,
        multimodal_embeddings = None,
    ):
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.img_context_token_id)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) ->  IntermediateTensors:

        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            #TODO: 加上vision的embedding
            #kwargs['pixel_values'] #第一个维度是batch的维度，pixel_values[i][j]第i条样本的第j个图片
            if 'pixel_values' in kwargs:
                vision_embeddings = self.get_multimodal_embeddings(**kwargs)
                vision_embeddings = torch.cat(vision_embeddings, dim=0)
                inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            else:
                inputs_embeds = self.language_model.get_input_embeddings(input_ids)
            input_ids = None

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }
        hidden_states = self.language_model.model(**forward_kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)