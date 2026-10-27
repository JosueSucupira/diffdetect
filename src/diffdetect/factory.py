"""Factory functions for creating native detector instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters import DinoLizerDetector, DistilDIREDetector
from .detector import BaseDetector

DEFAULT_DISTILDIRE_THRESHOLD = 0.9877818822860718
DEFAULT_DINOLIZER_THRESHOLD = 0.5946570634841919


def create_detector(
    name: str,
    *,
    threshold: float | None = None,
    device: str = "cuda",
    runtime: Any | None = None,
    repository_path: str | Path | None = None,
    **kwargs: Any,
) -> BaseDetector:
    """Create a detector configured with native runtimes and automatic weights.

    Supported detector names:
        - "distildire": ImageNet DistilDIRE detector for fully synthetic diffusion images.
        - "dinolizer": DinoLizer localization detector for edited/inpainted images.
    """
    normalized_name = name.strip().lower()

    if normalized_name == "distildire":
        eff_threshold = (
            threshold if threshold is not None else DEFAULT_DISTILDIRE_THRESHOLD
        )
        if runtime is not None:
            return DistilDIREDetector(threshold=eff_threshold, runtime=runtime)
        if repository_path is not None:
            return DistilDIREDetector(
                threshold=eff_threshold,
                repository_path=repository_path,
                device=device,
                **kwargs,
            )
        from .models.distildire import NativeDistilDIRERuntime

        native_runtime = NativeDistilDIRERuntime(device=device, **kwargs)
        return DistilDIREDetector(threshold=eff_threshold, runtime=native_runtime)

    elif normalized_name == "dinolizer":
        eff_threshold = (
            threshold if threshold is not None else DEFAULT_DINOLIZER_THRESHOLD
        )
        if runtime is not None:
            return DinoLizerDetector(threshold=eff_threshold, runtime=runtime)
        if repository_path is not None:
            return DinoLizerDetector(
                threshold=eff_threshold,
                repository_path=repository_path,
                device=device,
                **kwargs,
            )
        from .models.dinolizer import NativeDinoLizerRuntime

        native_runtime = NativeDinoLizerRuntime(device=device, **kwargs)
        return DinoLizerDetector(threshold=eff_threshold, runtime=native_runtime)

    raise KeyError(
        f"unknown detector '{name}'. Available detectors: 'distildire', 'dinolizer'"
    )
