"""Common contract for detector adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from numbers import Real

from PIL import Image


class DetectionTarget(str, Enum):
    """Evidence that a specialized detector can investigate."""

    SYNTHETIC = "synthetic"
    EDITED = "edited"


@dataclass(frozen=True)
class DetectionResult:
    """A detector's decision and optional supporting information.

    ``score`` stays on the detector's own scale. It must not be compared with
    scores from other detectors without an explicit calibration step.
    ``localization_map`` may contain a mask or a heatmap of edited regions.
    ``metadata`` retains detector-specific context without changing the common
    decision contract.
    """

    detected: bool
    score: float | None = None
    threshold: float | None = None
    localization_map: Image.Image | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.detected) is not bool:
            raise TypeError("detected must be a bool")

        for name in ("score", "threshold"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number or None")
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.threshold is not None and self.score is None:
            raise ValueError("threshold requires a score")
        if self.localization_map is not None and not isinstance(
            self.localization_map, Image.Image
        ):
            raise TypeError("localization_map must be a PIL image or None")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if any(not isinstance(key, str) for key in self.metadata):
            raise TypeError("metadata keys must be strings")

        object.__setattr__(self, "metadata", dict(self.metadata))


class BaseDetector(ABC):
    """Base class for model-specific adapters.

    Each adapter declares a unique ``name`` and the target it investigates.
    ``predict`` receives an RGB PIL image and returns one ``DetectionResult``.
    Loading weights and preprocessing remain the adapter's responsibility.
    """

    name: str
    target: DetectionTarget

    @abstractmethod
    def predict(self, image: Image.Image) -> DetectionResult:
        """Return evidence for this adapter's target class."""
