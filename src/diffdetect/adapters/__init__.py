"""Concrete adapters for external image detectors."""

from .dinolizer import DinoLizerDetector, DinoLizerInference
from .distildire import DistilDIREDetector

__all__ = ["DinoLizerDetector", "DinoLizerInference", "DistilDIREDetector"]
