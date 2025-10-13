from functools import partial
from typing import Optional
from einops import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from flash_attn import flash_attn_varlen_func

NORM2FN = {
    "rms_norm": RMSNorm,
    "layer_norm": nn.LayerNorm,
}


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_torch(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)")
    sin = repeat(sin, "... d -> ... 1 (2 d)")
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim]) * sin, x[..., ro_dim:]],
        dim=-1,
    )


def apply_rotary_pos_emb_vision(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    t_ = t.float()
    cos = freqs.cos()
    sin = freqs.sin()
    output = apply_rotary_emb_torch(t_, cos, sin).type_as(t)
    return output


class VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (
                self.theta
                ** (
                    torch.arange(
                        0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device
                    )
                    / self.dim
                )
            )
            seq = torch.arange(
                seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Linear(
            in_channels * patch_size * patch_size, embed_dim, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): shape [seq_len, in_channels*patch_size*patch_size]
        Returns:
            torch.Tensor: shape [seq_len, embed_dim]
        """
        target_dtype = self.proj.weight.dtype
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype))
        return hidden_states


class InternMLP(nn.Module):
    """MLP for Vision Transformers."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class InternVisionAttention(nn.Module):
    """Attention with Rotary Position Embedding."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # Additional dummy heads are used to enable TP for common GPU counts.(主要是6B的ViT只有25个head，我们使用的300M，应该没问题，以及似乎这个num_dummy_heads并没有被使用？)
        self.dummy_dim = (num_dummy_heads + self.num_heads) * self.head_dim
        self.num_heads_per_partition = divide(
            num_dummy_heads + self.num_heads, self.tp_size
        )

        self.qkv = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            num_dummy_heads + self.num_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )
        self.qk_normalization = config.qk_normalization
        if self.qk_normalization:
            self.q_norm = RMSNorm(
                self.embed_dim,
                eps=config.layer_norm_eps,
                var_hidden_size=self.embed_dim,
            )
            self.k_norm = RMSNorm(
                self.embed_dim,
                eps=config.layer_norm_eps,
                var_hidden_size=self.embed_dim,
            )
        self.proj = RowParallelLinear(
            self.dummy_dim,
            self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm.forward_native(q)
        k = self.k_norm.forward_native(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        return q, k

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb):
        # QKV projection and split
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # Apply QK normalization if needed
        if self.qk_normalization:
            q, k = self._apply_qk_norm(q, k)

        # Reshape for attention
        q = q.view(-1, self.num_heads_per_partition, self.head_dim)
        k = k.view(-1, self.num_heads_per_partition, self.head_dim)
        v = v.view(-1, self.num_heads_per_partition, self.head_dim)

        # Apply rotary embeddings
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # Flash attention
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=False,
        )

        # Reshape and project output
        attn_output = attn_output.reshape(
            -1, self.num_heads_per_partition * self.head_dim
        )
        attn_output, _ = self.proj(attn_output)

        return attn_output


class InternVisionEncoderLayer(nn.Module):
    """Encoder Layer with Rotary Position Embedding."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # TODO: 这里到底用什么attention，还需要check一下
        self.attn = InternVisionAttention(config, quant_config, prefix=f"{prefix}.attn")
        self.mlp = InternMLP(config, quant_config, prefix=f"{prefix}.mlp")
        self.norm1 = NORM2FN[config.norm_type](
            self.embed_dim, eps=config.layer_norm_eps
        )
        self.norm2 = NORM2FN[config.norm_type](
            self.embed_dim, eps=config.layer_norm_eps
        )

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = (
            hidden_states
            + self.attn(self.norm1(hidden_states), cu_seqlens, rotary_pos_emb)
            * self.ls1
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states)) * self.ls2
        return hidden_states


class InternVisionEncoder(nn.Module):
    """Encoder with Rotary Position Embedding."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList(
            [
                InternVisionEncoderLayer(
                    config,
                    quant_config,
                    num_dummy_heads=num_dummy_heads,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, rotary_pos_emb)
        return hidden_states


class InternVisionTransformer(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embeddings = PatchEmbed(
            config.patch_size, config.num_channels, config.hidden_size
        )
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2, config.rope_theta)
        self.encoder = InternVisionEncoder(
            config, quant_config, prefix=f"{prefix}.encoder"
        )

    def rot_pos_emb(self, grid_hw):
        """Calculate rotary position embeddings"""
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
        self, hidden_states: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): shape [seq_len, in_channels*patch_size*patch_size]
            grid_hw (torch.Tensor): shape [num_images, 2]
        Returns:
            torch.Tensor: shape [seq_len, embed_dim]
        """
        hidden_states = self.embeddings(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_hw)
        cu_seqlens = (grid_hw[:, 0] * grid_hw[:, 1]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        hidden_states = self.encoder(
            hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        return hidden_states


class InternVisionModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.vision_model = InternVisionTransformer(
            config=config, quant_config=quant_config, prefix=prefix
        )

    def forward(
        self, hidden_states: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        return self.vision_model(hidden_states, grid_hw)
