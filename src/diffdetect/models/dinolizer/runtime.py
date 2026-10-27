"""Native PyTorch runtime for DinoLizer without external repository dependencies."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PIL import Image

from ...weights import ensure_weight
from .lora import LoRA_ViT_timm
from .wrapper import Wrapper5crops

if TYPE_CHECKING:
    from ...adapters.dinolizer import DinoLizerInference

WINDOW_SIZE = 504
STRIDE = 128
PIXEL_THRESHOLD = 0.5
TARGET_RESIZE_AREA = 1016 * 1016
INFERENCE_BATCH_SIZE = 50


class NativeDinoLizerRuntime:
    """Load and execute the DinoLizer model natively using timm and PyTorch."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        try:
            import numpy
            import timm
            import torch
            import torchvision.transforms.functional as transforms
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DinoLizer requires numpy, timm, torch, and torchvision; "
                "install the optional models environment: pip install '.[models]'"
            ) from exc

        if checkpoint_path is None:
            checkpoint_file = ensure_weight("dinolizer_bfree")
        else:
            checkpoint_file = Path(checkpoint_path).expanduser().resolve()
            if not checkpoint_file.is_file():
                raise FileNotFoundError(f"DinoLizer checkpoint not found: {checkpoint_file}")

        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        backbone = timm.create_model(
            "vit_base_patch14_reg4_dinov2.lvd142m",
            num_classes=1,
            pretrained=True,
        )
        model = Wrapper5crops(backbone, WINDOW_SIZE, return_features=True)
        model.model = LoRA_ViT_timm(vit_model=model.model, r=64, alpha=64, num_classes=0)

        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=True,
        )
        try:
            checkpoint_weights = checkpoint["state_dict"]
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid DinoLizer checkpoint") from exc
        if not isinstance(checkpoint_weights, Mapping):
            raise ValueError("invalid DinoLizer checkpoint state dictionary")

        state_dict: dict[str, torch.Tensor] = {}
        for name, value in checkpoint_weights.items():
            if not isinstance(name, str):
                raise ValueError("invalid DinoLizer checkpoint parameter name")
            normalized_name = name.removeprefix("net.").removeprefix("model.")
            if "pos_weight" not in normalized_name:
                state_dict[normalized_name] = value

        model.load_state_dict(state_dict)
        model = model.half().to(torch_device).eval()

        del backbone, checkpoint, checkpoint_weights, state_dict
        gc.collect()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

        self._numpy = numpy
        self._torch = torch
        self._transforms = transforms
        self._device = torch_device
        self._model = model

    def infer(self, image: Image.Image) -> DinoLizerInference:
        from ...adapters.dinolizer import (
            DinoLizerInference,
            _processed_size,
            _window_coordinates,
        )

        numpy = self._numpy
        torch = self._torch

        original_size = image.size
        processed_size = _processed_size(original_size)
        if processed_size != original_size:
            image = image.resize(processed_size, Image.Resampling.BICUBIC)

        coordinates = _window_coordinates(processed_size)
        width, height = processed_size
        logit_sums = numpy.zeros((height, width, 3), dtype=numpy.float32)
        weight_matrix = numpy.zeros((height, width), dtype=numpy.float32)

        for start in range(0, len(coordinates), INFERENCE_BATCH_SIZE):
            batch_coordinates = coordinates[start : start + INFERENCE_BATCH_SIZE]
            batch = torch.stack(
                [self._transform(image.crop(coord)) for coord in batch_coordinates]
            ).to(device=self._device, dtype=torch.float16)

            with torch.inference_mode():
                _, features = self._model(batch)
                patch_logits = self._model.cls_head(features[:, 5:, :])
                patch_logits = patch_logits.reshape(-1, 36, 36, 3).permute(0, 3, 1, 2)
                patch_logits = torch.nn.functional.interpolate(
                    patch_logits,
                    size=(WINDOW_SIZE, WINDOW_SIZE),
                    mode="bicubic",
                    align_corners=False,
                )
                patch_logits = patch_logits.permute(0, 2, 3, 1).cpu().numpy()

            for coordinate, window_logits in zip(batch_coordinates, patch_logits):
                left, top, right, bottom = coordinate
                logit_sums[top:bottom, left:right, :] += window_logits
                weight_matrix[top:bottom, left:right] += 1.0

        if numpy.any(weight_matrix == 0.0):
            raise RuntimeError("DinoLizer windows did not cover the processed image")

        averaged_logits = logit_sums / weight_matrix[:, :, numpy.newaxis]
        shifted_logits = averaged_logits - numpy.max(
            averaged_logits,
            axis=2,
            keepdims=True,
        )
        exponentials = numpy.exp(shifted_logits)
        probabilities = exponentials / numpy.sum(
            exponentials,
            axis=2,
            keepdims=True,
        )
        processed_map = probabilities[:, :, 2].astype(numpy.float32, copy=False)

        score = float(numpy.percentile(processed_map, 99))
        marked_area = float(numpy.mean(processed_map > PIXEL_THRESHOLD))
        localization_map = self._restore_original_size(processed_map, original_size)

        return DinoLizerInference(
            score=score,
            localization_map=localization_map,
            marked_area=marked_area,
            processed_size=processed_size,
            window_count=len(coordinates),
        )

    def _transform(self, image: Image.Image):
        tensor = self._transforms.to_tensor(image)
        return self._transforms.normalize(
            tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def _restore_original_size(
        self,
        probability_map,
        original_size: tuple[int, int],
    ) -> Image.Image:
        numpy = self._numpy
        localization_map = Image.fromarray(probability_map)
        if localization_map.size != original_size:
            localization_map = localization_map.resize(
                original_size,
                Image.Resampling.BICUBIC,
            )
        restored = numpy.asarray(localization_map, dtype=numpy.float32)
        restored = numpy.clip(restored, 0.0, 1.0)
        return Image.fromarray(restored.astype(numpy.float32, copy=False))
