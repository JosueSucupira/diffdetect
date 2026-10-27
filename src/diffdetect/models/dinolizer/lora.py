"""LoRA module for injecting low-rank adaptation into timm Vision Transformers."""

from __future__ import annotations

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class _LoRA_qkv_timm(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Wraps timm's self.qkv linear layer with low-rank adaptations on q and v."""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        r: int,
        alpha: int,
    ) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required to initialize _LoRA_qkv_timm")
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.r = r
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        scaling = self.alpha // self.r
        qkv[:, :, : self.dim] += scaling * new_q
        qkv[:, :, -self.dim :] += scaling * new_v
        return qkv


class LoRA_ViT_timm(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Applies Low-Rank Adaptation to a timm VisionTransformer."""

    def __init__(
        self,
        vit_model: nn.Module,
        r: int = 64,
        alpha: int = 64,
        num_classes: int = 0,
        lora_layer: Optional[list[int]] = None,
    ) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required to initialize LoRA_ViT_timm")
        super().__init__()
        assert r > 0, "r must be positive"
        assert alpha > 0, "alpha must be positive"

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        self.w_As: list[nn.Module] = []
        self.w_Bs: list[nn.Module] = []

        for param in vit_model.parameters():
            param.requires_grad = False

        for layer_idx, blk in enumerate(vit_model.blocks):
            if layer_idx not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.extend([w_a_linear_q, w_a_linear_v])
            self.w_Bs.extend([w_b_linear_q, w_b_linear_v])
            blk.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                r,
                alpha,
            )

        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.reset_classifier(num_classes=num_classes)

    def reset_parameters(self) -> None:
        for w_a in self.w_As:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_Bs:
            nn.init.zeros_(w_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_vit(x)
