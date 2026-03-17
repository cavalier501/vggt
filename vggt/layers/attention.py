# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self._graph_capture_mode = False

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_fused(nn.Module):
    """
    Fused attention implementation for Aggregator inference blocks.

    Both eager and torch_compile paths share the same operator set:
    - RoPE: RotaryPositionEmbedding2D.forward -> npu_rotary_mul
    - Attention: npu_fused_infer_attention_score
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self._graph_backend = "eager"

    def set_graph_backend(self, mode: str) -> None:
        if mode not in {"eager", "torch_compile"}:
            raise ValueError(f"Unsupported graph backend mode: {mode}")
        self._graph_backend = mode

    def _apply_standard_rope(self, q: Tensor, k: Tensor, pos: Tensor | None) -> tuple[Tensor, Tensor]:
        if self.rope is None:
            return q, k
        return self.rope(q, pos), self.rope(k, pos)

    def _to_attention_dtype(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        attn_dtype = v.dtype
        if q.dtype != attn_dtype:
            q = q.to(attn_dtype)
        if k.dtype != attn_dtype:
            k = k.to(attn_dtype)
        return q, k, v

    def _run_fia_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        import torch_npu

        seq_len = q.shape[2]
        return torch_npu.npu_fused_infer_attention_score(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            actual_seq_lengths=[seq_len],
            actual_seq_lengths_kv=[seq_len],
            num_heads=self.num_heads,
            input_layout="BNSD",
            scale=float(self.scale),
            pre_tokens=65535,
            next_tokens=65535,
        )[0]

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self._graph_backend not in {"eager", "torch_compile"}:
            raise RuntimeError(f"Unknown graph backend mode: {self._graph_backend}")

        q, k = self._apply_standard_rope(q, k, pos)
        q, k, v = self._to_attention_dtype(q, k, v)
        x = self._run_fia_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionFusedAclgraphDeprecated(Attention_fused):
    """Deprecated ACLGraph backup for historical reference only.

    This class is intentionally not used by business code. It is kept as a code
    backup because some fused NPU attention operators do not work reliably with
    raw NPUGraph capture/replay.
    """

    def set_graph_backend(self, mode: str) -> None:
        if mode not in {"eager", "torch_compile", "aclgraph"}:
            raise ValueError(f"Unsupported graph backend mode: {mode}")
        self._graph_backend = mode

    def set_graph_capture_mode(self, enabled: bool) -> None:
        self.set_graph_backend("aclgraph" if enabled else "eager")

    def _run_aclgraph_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return attn @ v

    def forward(self, x: Tensor, pos=None) -> Tensor:
        if self._graph_backend != "aclgraph":
            return super().forward(x, pos=pos)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self._apply_standard_rope(q, k, pos)
        q, k, v = self._to_attention_dtype(q, k, v)
        x = self._run_aclgraph_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
