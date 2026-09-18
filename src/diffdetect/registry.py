"""Registration and construction of detector adapters."""

from __future__ import annotations

from inspect import isabstract, isclass
from typing import Any, TypeVar

from .detector import BaseDetector, DetectionTarget


DetectorType = TypeVar("DetectorType", bound=BaseDetector)


class DetectorRegistry:
    """Map detector names to concrete adapter classes.

    Each registry is independent. Registering an adapter does not load model
    weights; construction happens only when ``create`` is called.
    """

    def __init__(self) -> None:
        self._detectors: dict[str, type[BaseDetector]] = {}

    def register(self, detector_type: type[DetectorType]) -> type[DetectorType]:
        """Validate and add an adapter class. Can also be used as a decorator."""

        if not isclass(detector_type) or not issubclass(detector_type, BaseDetector):
            raise TypeError("detector_type must be a BaseDetector subclass")
        if isabstract(detector_type):
            raise TypeError("cannot register an abstract detector")

        name = getattr(detector_type, "name", None)
        target = getattr(detector_type, "target", None)
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("detector name must be a non-empty string without outer spaces")
        if not isinstance(target, DetectionTarget):
            raise ValueError("detector target must be a DetectionTarget")
        if name in self._detectors:
            raise ValueError(f"detector already registered: {name}")

        self._detectors[name] = detector_type
        return detector_type

    def create(self, name: str, **kwargs: Any) -> BaseDetector:
        """Build one registered adapter, passing configuration to its constructor."""

        try:
            detector_type = self._detectors[name]
        except KeyError as exc:
            raise KeyError(f"unknown detector: {name}") from exc
        return detector_type(**kwargs)

    def names(self) -> tuple[str, ...]:
        """List registered names in registration order."""

        return tuple(self._detectors)
