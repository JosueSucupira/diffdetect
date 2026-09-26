"""Public API for DiffDetect detector adapters."""

from .adapters import DistilDIREDetector
from .classification import ClassificationResult, DecisionReason, ImageClass, classify
from .detector import BaseDetector, DetectionResult, DetectionTarget
from .registry import DetectorRegistry
from .runner import DetectorRun, DetectorRunner

__all__ = [
    "BaseDetector",
    "ClassificationResult",
    "DecisionReason",
    "DetectionResult",
    "DetectionTarget",
    "DistilDIREDetector",
    "DetectorRegistry",
    "DetectorRun",
    "DetectorRunner",
    "ImageClass",
    "classify",
]
