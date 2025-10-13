# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# A modified implementation of the AIMv2 Transformer
# Added support for 2D RoPE, window attention, and NaViT-style dynamic resolution
# Compatible with HuggingFace interface
from collections.abc import Iterable
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.attention.layer import MultiHeadAttention
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.utils import divide
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.transformers_utils.configs.ovis import AIMv2Config
from flash_attn import flash_attn_varlen_func
from flash_attn.layers.rotary import apply_rotary_emb

class AIMv2SwiGLUFFN(nn.Module):

    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig,
                 prefix: str):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.hidden_size
        bias = config.use_bias

        self.fc13 = MergedColumnParallelLinear(
            in_features,
            [hidden_features] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc13",
        )
        self.fc2 = RowParallelLinear(
            input_size=hidden_features,
            output_size=in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc13(x)
        x = self.act_fn(x)
        x, _ = self.fc2(x)
        return x


class AIMv2PatchEmbed(nn.Module):

    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        # Ensure temporal_patch_size is 1 for compatibility
        assert getattr(config, 'temporal_patch_size', 1) == 1
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _get_2d_weight(self):
        """Convert 2D conv weight to linear format for efficiency"""
        weight = self.proj.weight.view(self.config.hidden_size, -1)
        bias = self.proj.bias if self.proj.bias is not None else torch.zeros(
            self.config.hidden_size, device=weight.device)
        return weight, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: [num_patches, c*patch_size*patch_size] (NaViT style)
        # When temporal_patch_size=1: [num_patches, c*patch_size*patch_size]
        x = torch.nn.functional.linear(x, *self._get_2d_weight())
        x = self.norm.forward_native(x)
        return x


class AIMv2ViTPreprocessor(nn.Module):

    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.config = config
        num_patches = (config.image_size // config.patch_size) ** 2

        self.patchifier = AIMv2PatchEmbed(config)
        
        # Optional position embedding preservation
        self.preserve_original_pe = getattr(config, "preserve_original_pe", False)
        self.hidden_stride = getattr(config, "hidden_stride", 1)
        
        if self.preserve_original_pe:
            self.interpolate_pe_method = getattr(config, "interpolate_pe_method", "two_dim")
            self.pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, config.hidden_size)))

    def forward(self, x: torch.Tensor, grid_thws: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.patchifier(x)

        if self.preserve_original_pe and grid_thws is not None:
            pos_embed_new = torch.zeros_like(tokens)
            
            if self.interpolate_pe_method == 'one_dim':
                pos_embed = self.pos_embed.transpose(1, 2).to(tokens.device)
            elif self.interpolate_pe_method == 'two_dim':
                ori_h = ori_w = int(self.pos_embed.shape[1] ** 0.5)
                pos_embed = self.pos_embed.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2)
            else:
                raise TypeError("The interpolation method for pe should be one_dim, two_dim.")
            
            cnt = 0
            for t, h, w in grid_thws:
                num_patches = h * w
                thw = t * h * w
                
                if self.interpolate_pe_method == 'one_dim':
                    pe = F.interpolate(pos_embed, size=num_patches, mode='linear', align_corners=False).transpose(1, 2)
                elif self.interpolate_pe_method == 'two_dim':
                    pe = F.interpolate(pos_embed, size=(h, w), mode='bicubic', align_corners=False)
                    pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)
                
                pe = pe[0].repeat(t, 1)
                pe = pe.reshape(t, h // self.hidden_stride, self.hidden_stride, 
                              w // self.hidden_stride, self.hidden_stride, -1)
                pe = pe.permute(0, 1, 3, 2, 4, 5).reshape(thw, -1)
                pos_embed_new[cnt:cnt + thw] = pe
                cnt += thw

            tokens = tokens + pos_embed_new

        return tokens


class VisionRotaryEmbedding(nn.Module):
    """Vision 2D rotary embedding for dynamic resolution"""
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed


class AIMv2Attention(nn.Module):

    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig,
                 prefix: str):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5

        self.qkv = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )

        self.proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

        # RoPE flag
        self.use_rope = not getattr(config, "disable_rope", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention: [seq_len, num_heads_per_partition, head_dim]
        q = q.view(seq_length, self.num_heads_per_partition, self.head_dim)
        k = k.view(seq_length, self.num_heads_per_partition, self.head_dim)
        v = v.view(seq_length, self.num_heads_per_partition, self.head_dim)

        if self.use_rope and position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
            q = q.squeeze(0)
            k = k.squeeze(0)

        # use flash attention
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
        attn_output = attn_output.reshape(
            -1, self.num_heads_per_partition * self.head_dim
        )
        attn_output, _ = self.proj(attn_output)
        return attn_output


class AIMv2Block(nn.Module):

    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig,
                 prefix: str):
        super().__init__()
        self.attn = AIMv2Attention(config,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.attn")
        self.norm_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = AIMv2SwiGLUFFN(config,
                                  quant_config=quant_config,
                                  prefix=f"{prefix}.mlp")
        self.norm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_1.forward_native(x), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        x = x + self.mlp(self.norm_2.forward_native(x))
        return x


class AIMv2Transformer(nn.Module):

    def __init__(
        self,
        config: AIMv2Config,
        quant_config: QuantizationConfig,
        *,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            AIMv2Block(config, quant_config, prefix=f"{prefix}.blocks.{i}")
            for i in range(config.num_hidden_layers)
        ])
        
        # Always include post_trunk_norm (following HF implementation)
        self.post_trunk_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 2D RoPE
        self.use_rope = not getattr(config, "disable_rope", False)
        if self.use_rope:
            head_dim = config.hidden_size // config.num_attention_heads
            rope_dim = head_dim // 2  # Half for height, half for width
            self.rotary_pos_emb = VisionRotaryEmbedding(rope_dim)
            
        # Configuration for spatial merging and windowing
        self.hidden_stride = getattr(config, "hidden_stride", 1)
        self.patch_size = getattr(config, "patch_size", 16)
        self.window_size = getattr(config, "window_size", 224)  # Default window size
        self.spatial_merge_unit = self.hidden_stride * self.hidden_stride
        
        # Full attention block indexes (if None, all blocks use windowed attention)
        self.fullatt_block_indexes = getattr(config, "fullatt_block_indexes", None)

    def rot_pos_emb(self, grid_thws: torch.Tensor) -> torch.Tensor:
        """Generate 2D rotary position embeddings from grid info"""
        pos_ids = []
        for t, h, w in grid_thws:
            # Create 2D position indices
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.hidden_stride,
                self.hidden_stride,
                w // self.hidden_stride,
                self.hidden_stride,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.hidden_stride,
                self.hidden_stride,
                w // self.hidden_stride,
                self.hidden_stride,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        
        pos_ids = torch.cat(pos_ids, dim=0)  # [total_patches, 2]
        max_grid_size = grid_thws[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)  # [total_patches, rope_dim*2]
        return rotary_pos_emb

    def get_window_index(self, grid_thws: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """Get window indices for windowed attention"""
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.hidden_stride // self.patch_size

        for grid_t, grid_h, grid_w in grid_thws:
            llm_grid_h = grid_h // self.hidden_stride
            llm_grid_w = grid_w // self.hidden_stride
            
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def forward(
        self,
        tokens: torch.Tensor,
        grid_thws: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        # Generate 2D RoPE if enabled and grid info available
        rotary_pos_emb = self.rot_pos_emb(grid_thws).to(tokens.device)
        
        # Prepare window indices for windowed attention
        window_index, cu_window_seqlens = self.get_window_index(grid_thws)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=tokens.device,
            dtype=grid_thws.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = tokens.size()
        tokens = tokens.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        tokens = tokens[window_index, :, :]
        tokens = tokens.reshape(seq_len, -1)
        
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Prepare sequence lengths for attention
        cu_seqlens = torch.repeat_interleave(
            grid_thws[:, 1] * grid_thws[:, 2], grid_thws[:, 0]
        ).cumsum(
            dim=0,
            dtype=grid_thws.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Prepare reverse indices for window attention
        reverse_indices = torch.argsort(window_index)

        
        for index, block in enumerate(self.blocks):
            # Choose between full attention and windowed attention
            if self.fullatt_block_indexes is None or index in self.fullatt_block_indexes:
                cu_seqlens_tmp = cu_seqlens
            else:
                cu_seqlens_tmp = cu_window_seqlens if grid_thws is not None else cu_seqlens
            
            tokens = block(tokens, cu_seqlens_tmp, position_embeddings)

        # Apply post trunk norm
        tokens = self.post_trunk_norm.forward_native(tokens)
        
        # Restore original order if windowed attention was used
        seq_len = tokens.shape[0]
        tokens = tokens.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        tokens = tokens[reverse_indices, :].reshape(seq_len, -1)
        
        return tokens


class Aimv2VisionModel(torch.nn.Module):

    def __init__(self,
                 config: AIMv2Config,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.config = config
        self.preprocessor = AIMv2ViTPreprocessor(config)
        self.trunk = AIMv2Transformer(config,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.trunk")

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_hws: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [num_patches, c*temporal_patch_size*patch_size*patch_size] 
                         NaViT-style flattened patches
            grid_hws: [num_images, 2] tensor with (h, w) for each image
        """
        # NOTE: 这个是我们自研的ViT输入接口
        # Transform flattened pixel values to include temporal dimension
        assert self.config.temporal_patch_size == 1, "This implementation assumes temporal_patch_size is 1."
        pixel_values = torch.cat([hidden_states for _ in range(self.config.temporal_patch_size)], dim=1)

        # Add temporal dimension (t=1) to the grid info
        grid_t = torch.ones(grid_hws.shape[0], 1, device=grid_hws.device, dtype=grid_hws.dtype)
        grid_thws = torch.cat([grid_t, grid_hws], dim=1)

        x = self.preprocessor(pixel_values, grid_thws=grid_thws)
        x = self.trunk(x, grid_thws=grid_thws)
        
        return x

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".fc13", ".fc1", 0),
            (".fc13", ".fc3", 1),
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