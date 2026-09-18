"""Public API for DiffDetect detector adapters."""

from .detector import BaseDetector, DetectionResult, DetectionTarget
from .registry import DetectorRegistry

__all__ = ["BaseDetector", "DetectionResult", "DetectionTarget", "DetectorRegistry"]
