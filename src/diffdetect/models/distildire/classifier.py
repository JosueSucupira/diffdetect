"""Student classifier network for DistilDIRE (ResNet50 with 6-channel input)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class DistilDIRE(nn.Module if nn is not None else object):  # type: ignore[misc]
    """ResNet50 student network accepting image (3 channels) + noise (3 channels)."""

    def __init__(self, device: str = "cpu") -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required to initialize DistilDIRE")
        super().__init__()
        import torchvision.models as tvm

        student = tvm.resnet50()
        self.student_backbone = nn.Sequential(
            OrderedDict(list(student.named_children())[:-2])
        )
        self.student_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 1),
        )
        self.student_backbone.conv1 = nn.Conv2d(
            6, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.device = device

    def forward(
        self,
        img: torch.Tensor,
        eps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        img = img.to(self.device)
        eps = eps.to(self.device)
        img_tens = torch.cat([img, eps], dim=1)
        feature = self.student_backbone(img_tens)
        logit = self.student_head(feature)
        return {"logit": logit, "feature": feature}
