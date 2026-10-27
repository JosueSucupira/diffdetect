"""Layers for the DinoLizer vision transformer and head."""

from __future__ import annotations

from typing import Optional, Type

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class Attention(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Multi-head Self Attention with optional fused scaled dot product."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_head_dim: Optional[int] = None,
        dim_out: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Optional[Type[nn.Module]] = None,
        device=None,
        dtype=None,
    ) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required to initialize Attention")
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        dim_out = dim_out or dim
        head_dim = attn_head_dim
        if head_dim is None:
            if dim % num_heads != 0:
                raise ValueError("dim should be divisible by num_heads")
            head_dim = dim // num_heads
        if qk_norm or scale_norm:
            if norm_layer is None:
                raise ValueError("norm_layer must be provided if qk_norm or scale_norm is True")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, self.attn_dim * 3, bias=qkv_bias, **dd)
        self.q_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(self.attn_dim, **dd) if scale_norm else nn.Identity()
        self.proj = nn.Linear(self.attn_dim, dim_out, bias=proj_bias, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(b, n, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Two-layer MLP with GELU activation."""

    def __init__(self, emb_size: int) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required to initialize FeedForward")
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Linear(4 * emb_size, emb_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class Block(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Transformer block with attention and feedforward layers."""

    def __init__(self, emb_size: int, num_head: int) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required to initialize Block")
        super().__init__()
        self.att = Attention(dim=emb_size, num_heads=num_head)
        self.ll = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.0)
        self.ff = FeedForward(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.att(self.ll(x)))
        x = x + self.dropout(self.ff(self.ll(x)))
        return self.ll(x)
