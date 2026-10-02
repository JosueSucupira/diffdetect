"""Adapter boundary for the external DinoLizer localization detector."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Protocol

from PIL import Image

from ..detector import BaseDetector, DetectionResult, DetectionTarget


DEFAULT_THRESHOLD = 0.5946570634841919
WINDOW_SIZE = 504
STRIDE = 128
PIXEL_THRESHOLD = 0.5


@dataclass(frozen=True)
class DinoLizerInference:
    """Detector-specific output produced by a DinoLizer runtime."""

    score: float
    localization_map: Image.Image
    marked_area: float
    processed_size: tuple[int, int]
    window_count: int


class DinoLizerRuntime(Protocol):
    """Small inference boundary used by the adapter."""

    def infer(self, image: Image.Image) -> DinoLizerInference:
        """Return calibrated image evidence and an original-size probability map."""


class DinoLizerDetector(BaseDetector):
    """Expose DinoLizer edited-image evidence through the common detector API.

    The injectable runtime keeps the framework independent of the external
    DinoLizer source tree, model checkpoint, and deep-learning dependencies.
    The official runtime is added separately from this adapter boundary.
    """

    name = "dinolizer"
    target = DetectionTarget.EDITED

    def __init__(
        self,
        *,
        runtime: DinoLizerRuntime,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.threshold = _probability(threshold, "threshold")
        if not callable(getattr(runtime, "infer", None)):
            raise TypeError("runtime must provide a callable infer(image) method")
        self._runtime = runtime

    def predict(self, image: Image.Image) -> DetectionResult:
        """Calculate DinoLizer evidence for one image."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")

        rgb_image = image.convert("RGB")
        inference = self._runtime.infer(rgb_image)
        if not isinstance(inference, DinoLizerInference):
            raise TypeError("runtime must return DinoLizerInference")

        score = _probability(inference.score, "score")
        marked_area = _probability(inference.marked_area, "marked_area")
        processed_size = _size(inference.processed_size, "processed_size")
        window_count = _positive_integer(inference.window_count, "window_count")
        localization_map = _localization_map(inference.localization_map, image.size)

        return DetectionResult(
            detected=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            localization_map=localization_map,
            metadata={
                "score_name": "p99_probability",
                "marked_area": marked_area,
                "original_size": image.size,
                "processed_size": processed_size,
                "window_count": window_count,
                "window_size": WINDOW_SIZE,
                "stride": STRIDE,
                "pixel_threshold": PIXEL_THRESHOLD,
            },
        )


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return converted


def _size(value: object, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return value


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _localization_map(value: object, original_size: tuple[int, int]) -> Image.Image:
    if not isinstance(value, Image.Image):
        raise TypeError("localization_map must be a PIL image")
    if value.mode != "F":
        raise ValueError("localization_map must use floating-point mode F")
    if value.size != original_size:
        raise ValueError("localization_map must match the original image size")

    if any(
        not isfinite(probability) or not 0.0 <= probability <= 1.0
        for probability in value.getdata()
    ):
        raise ValueError("localization_map values must be finite and between 0 and 1")
    return value
