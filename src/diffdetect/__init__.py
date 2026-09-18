"""Public API for DiffDetect detector adapters."""

from .detector import BaseDetector, DetectionResult, DetectionTarget
from .registry import DetectorRegistry
from .runner import DetectorRun, DetectorRunner

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "DetectionTarget",
    "DetectorRegistry",
    "DetectorRun",
    "DetectorRunner",
]
