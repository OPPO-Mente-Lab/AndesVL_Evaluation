# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# A modified implementation of the SigLIP2 Vision Transformer
# Added support for 2D RoPE, NaViT-style dynamic resolution, and vLLM optimizations
# Compatible with vLLM distributed execution
from collections.abc import Iterable
from typing import Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.attention.layer import MultiHeadAttention
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.utils import divide
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from flash_attn import flash_attn_varlen_func

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    tensor: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class VisionRotaryEmbedding(nn.Module):
    """Vision 2D rotary embedding for dynamic resolution"""
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig,
        prefix: str,
    ):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches
        self.embed_dim = config.hidden_size
        self.preserve_original_pe = getattr(config, "preserve_original_pe", True)

        self.proj = nn.Linear(
            config.num_channels * config.patch_size * config.patch_size,
            config.hidden_size,
        )

        if self.preserve_original_pe:
            assert self.num_patches**0.5 == int(self.num_patches**0.5), "num_patches must be a perfect square"
            self.pos_embed = nn.Embedding(self.num_patches, self.embed_dim)
            self.original_grid_size = int(self.num_patches**0.5)
        else:
            self.pos_embed = None
            self.original_grid_size = 0

    def get_patch_coordinates(self, grid_hw: torch.Tensor, device: torch.device):
        """Generate patch coordinates matching 2x2 block scanning order."""
        all_h_coords, all_w_coords, all_target_sizes = [], [], []
        
        for h, w in grid_hw:
            h, w = h.item(), w.item()
            
            # Generate standard grid coordinates
            h_grid, w_grid = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing='ij'
            )
            
            # Reshape to block scanning order
            h_coords = h_grid.reshape(
                h//2, 2, w//2, 2
            ).permute(0, 2, 1, 3).flatten()
            
            w_coords = w_grid.reshape(
                h//2, 2, w//2, 2
            ).permute(0, 2, 1, 3).flatten()
            
            all_h_coords.append(h_coords)
            all_w_coords.append(w_coords)
            
            target_size = torch.tensor([h, w], device=device, dtype=torch.float32)
            all_target_sizes.append(target_size.expand(h * w, -1))

        return torch.cat(all_h_coords), torch.cat(all_w_coords), torch.cat(all_target_sizes)

    def abs_pos_embed(self, grid_hw: torch.Tensor, mode='bicubic') -> torch.Tensor:
        pos_embed_weight = self.pos_embed.weight
        pos_embed_2d = pos_embed_weight.transpose(0, 1).reshape(
            self.embed_dim, self.original_grid_size, self.original_grid_size
        ).unsqueeze(0).to(torch.float32)

        if grid_hw.numel() == 0:
            return torch.empty(0, self.embed_dim, device=pos_embed_2d.device, dtype=pos_embed_weight.dtype)
        
        h_coords, w_coords, target_sizes = self.get_patch_coordinates(grid_hw, pos_embed_2d.device)
        
        if h_coords.shape[0] == 0:
            return torch.empty(0, self.embed_dim, device=pos_embed_2d.device, dtype=pos_embed_weight.dtype)

        target_h = target_sizes[:, 0]
        target_w = target_sizes[:, 1]
        
        # Normalization formula for align_corners=False
        norm_w = (2.0 * (w_coords + 0.5) / target_w) - 1.0
        norm_h = (2.0 * (h_coords + 0.5) / target_h) - 1.0

        grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(0)

        interpolated_embed = F.grid_sample(
            pos_embed_2d, grid, mode=mode, align_corners=False,
            padding_mode='border'  
        )
        
        adapted_pos_embed = interpolated_embed.squeeze(0).squeeze(1).permute(1, 0)
        
        return adapted_pos_embed.to(pos_embed_weight.dtype)
            
    def forward(self, hidden_states: torch.Tensor, grid_hw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): input tensor of shape [seq_len, num_channels*patch_size*patch_size]
            grid_hw (torch.Tensor): tensor of shape [num_images, 2] with (h, w) for each image
        Returns:
            torch.Tensor: output tensor of shape [seq_len, embed_dim]
        """
        hidden_states = self.proj(hidden_states)
        
        if self.preserve_original_pe:
            pos_emb = self.abs_pos_embed(grid_hw)
            hidden_states = hidden_states + pos_emb
            
        return hidden_states


class Siglip2MLP(nn.Module):
    def __init__(self, config, quant_config: QuantizationConfig, prefix: str):
        super().__init__()
        self.config = config
        
        self.fc1 = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)  # SigLIP typically uses GELU
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class Siglip2Attention(nn.Module):

    def __init__(self, config, quant_config: QuantizationConfig, prefix: str):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5

        self.qkv = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            bias=True,  # SigLIP uses bias
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention: [seq_len, num_heads_per_partition, head_dim]
        q = q.view(seq_length, self.num_heads_per_partition, self.head_dim)
        k = k.view(seq_length, self.num_heads_per_partition, self.head_dim)
        v = v.view(seq_length, self.num_heads_per_partition, self.head_dim)
        
        # Apply rotary position embedding
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
            k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # Use flash attention
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
        ).reshape(seq_length, -1)
        
        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class Siglip2EncoderLayer(nn.Module):
    def __init__(self, config, quant_config: QuantizationConfig, prefix: str):
        super().__init__()
        self.embed_dim = config.hidden_size
        
        self.self_attn = Siglip2Attention(
            config, quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=getattr(config, "layer_norm_eps", 1e-6))
        self.mlp = Siglip2MLP(
            config, quant_config=quant_config, prefix=f"{prefix}.mlp"
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=getattr(config, "layer_norm_eps", 1e-6))

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)  # Remove .forward_native
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)  # Remove .forward_native
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Siglip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Siglip2EncoderLayer`].
    """

    def __init__(self, config, quant_config: QuantizationConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Siglip2EncoderLayer(config, quant_config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb,
    ):
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens,
                rotary_pos_emb,
            )
        return hidden_states


class Siglip2VisionTransformer(nn.Module):
    def __init__(self, config, quant_config: QuantizationConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = PatchEmbed(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.embeddings",
        )
        
        head_dim = config.hidden_size // config.num_attention_heads
        rope_theta = getattr(config, "rope_theta", 10000.0)
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2, rope_theta)
        
        self.encoder = Siglip2Encoder(
            config, quant_config=quant_config, prefix=f"{prefix}.encoder"
        )
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=getattr(config, "layer_norm_eps", 1e-6))

    def rot_pos_emb(self, grid_hw):
        pos_ids = []
        for h, w in grid_hw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // 2,
                2,
                w // 2,
                2,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // 2,
                2,
                w // 2,
                2,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_hw.max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_hw: torch.Tensor,
    ):
        hidden_states = self.embeddings(hidden_states, grid_hw)
        rotary_pos_emb = self.rot_pos_emb(grid_hw).to(hidden_states.device)
        cu_seqlens = (grid_hw[:, 0] * grid_hw[:, 1]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        hidden_states = self.encoder(
            hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = self.post_layernorm(hidden_states)  # Remove .forward_native
        return hidden_states


class Siglip2VisionModel(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vision_model = Siglip2VisionTransformer(
            config, quant_config=quant_config, prefix=f"{prefix}.vision_model"
        )

    def forward(
        self, hidden_states: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [num_patches, c*patch_size*patch_size] NaViT-style flattened patches
            grid_hw: [num_images, 2] tensor with (h, w) for each image
        """
        return self.vision_model(hidden_states, grid_hw)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv", ".q_proj", "q"),
            (".qkv", ".k_proj", "k"), 
            (".qkv", ".v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params