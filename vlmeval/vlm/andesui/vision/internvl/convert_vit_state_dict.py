import torch
import torch.nn as nn
import sys
sys.path.append("/mnt/data/group/wangnan/model/InternVit-6B-980px-V1.5-Navit-Leaf")
from modeling_intern_vit_navit_leaf_cat_packed_flash import  InternVisionModel
sys.path.append("/mnt/data/group/wangnan/model/InternVit-300M-980px-V2-Navit-Leaf")
from modeling_intern_vit_ln_navit_leaf_cat_packed_flash import InternVisionModel as InternVisionModelLN

class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim, bias=False)

def conv2d_to_linear(conv2d_layer):
    # 获取原始的权重
    conv2d_weight = conv2d_layer.weight.data  # 形状为 (out_channels, in_channels, kernel_height, kernel_width)
    # 展平成二维权重矩阵
    out_channels, in_channels, kernel_height, kernel_width = conv2d_weight.shape
    new_weight = conv2d_weight.view(out_channels, -1)  # 形状为 (out_channels, in_channels * kernel_height * kernel_width)
    print(f"new_weight 的形状为：{new_weight.shape}")  # 应该输出 torch.Size([out_channels, in_channels * kernel_height * kernel_width])
    # 创建新的 Linear 层
    linear_layer = nn.Linear(in_channels * kernel_height * kernel_width, out_channels, bias=False)
    # 设置 Linear 层的权重
    linear_layer.weight.data = new_weight
    return linear_layer

#主要涉及两个操作：删除CLS token、删除底层的位置嵌入、Conv2D转Linear。
#我们暂时只处理两个模型： 8B的40B的，对应300M和6B的ViT
"""
model = InternVisionModelLN.from_pretrained("/mnt/data/group/wangnan/model/InternVL2-8B-AndesVL-Init/InternVit-300M-980px-V2-Navit-Leaf")
pe = PatchEmbed(model.config.hidden_size)
linear = conv2d_to_linear(model.vision_model.embeddings.patch_embedding)
pe.proj = linear
del model.vision_model.embeddings
model.vision_model.embeddings = pe
torch.save(model.state_dict(), "/mnt/data/group/wangnan/model/InternVL2-8B-AndesVL-Init/InternViT-300M-Navit-RoPE/pytorch_model.bin")
"""
model = InternVisionModel.from_pretrained("/mnt/data/group/wangnan/model/InternVL2-40B-AndesVL-Init/InternVit-6B-980px-V1.5-Navit-Leaf-Fix")
pe = PatchEmbed(model.config.hidden_size)
linear = conv2d_to_linear(model.vision_model.embeddings.patch_embedding)
pe.proj = linear
del model.vision_model.embeddings
model.vision_model.embeddings = pe
torch.save(model.state_dict(), "/mnt/data/group/wangnan/model/InternVL2-40B-AndesVL-Init/InternVit-6B-Navit-RoPE/pytorch_model.bin")