"""Class definitions for image authenticity."""

from __future__ import annotations

from enum import Enum


class ImageClass(str, Enum):
    """Classes used by the three-class analysis."""

    REAL = "real"
    SYNTHETIC = "synthetic"
    EDITED = "edited"
