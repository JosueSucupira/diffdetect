"""Native DistilDIRE model architecture and inference runtime."""

from __future__ import annotations

from .classifier import DistilDIRE
from .runtime import NativeDistilDIRERuntime

__all__ = [
    "DistilDIRE",
    "NativeDistilDIRERuntime",
]
