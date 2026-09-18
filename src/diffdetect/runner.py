"""Run detector adapters on a shared image input."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from PIL import Image

from .detector import BaseDetector, DetectionResult, DetectionTarget


@dataclass(frozen=True)
class DetectorRun:
    """One detector's result or error, with its execution time."""

    name: str
    target: DetectionTarget
    duration_seconds: float
    result: DetectionResult | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ValueError("exactly one of result or error must be set")

    @property
    def status(self) -> str:
        return "success" if self.error is None else "error"


class DetectorRunner:
    """Execute selected detector instances through a single image API."""

    def __init__(self, detectors: Iterable[BaseDetector]) -> None:
        self._detectors: dict[str, BaseDetector] = {}
        for detector in detectors:
            if not isinstance(detector, BaseDetector):
                raise TypeError("all detectors must extend BaseDetector")
            if (
                not isinstance(detector.name, str)
                or not detector.name
                or detector.name != detector.name.strip()
            ):
                raise ValueError("detector name must be a non-empty string without outer spaces")
            if not isinstance(detector.target, DetectionTarget):
                raise ValueError("detector target must be a DetectionTarget")
            if detector.name in self._detectors:
                raise ValueError(f"duplicate detector: {detector.name}")
            self._detectors[detector.name] = detector
        if not self._detectors:
            raise ValueError("at least one detector is required")

    def names(self) -> tuple[str, ...]:
        """List loaded detector instances in execution order."""

        return tuple(self._detectors)

    def run(
        self,
        image: str | Path | Image.Image,
        *,
        selected: Iterable[str] | None = None,
    ) -> tuple[DetectorRun, ...]:
        """Run selected detectors and retain each success or failure.

        The input is converted to RGB once. Every detector receives its own
        copy, so a model that modifies an image cannot affect later detectors.
        Invalid input images raise before any detector is run.
        """

        detectors = self._select(selected)
        rgb_image = _load_rgb(image)
        runs: list[DetectorRun] = []

        for detector in detectors:
            started = perf_counter()
            try:
                result = detector.predict(rgb_image.copy())
                if not isinstance(result, DetectionResult):
                    raise TypeError("predict() must return DetectionResult")
                runs.append(
                    DetectorRun(
                        name=detector.name,
                        target=detector.target,
                        duration_seconds=perf_counter() - started,
                        result=result,
                    )
                )
            except Exception as exc:
                runs.append(
                    DetectorRun(
                        name=detector.name,
                        target=detector.target,
                        duration_seconds=perf_counter() - started,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        return tuple(runs)

    def _select(self, selected: Iterable[str] | None) -> tuple[BaseDetector, ...]:
        if selected is None:
            return tuple(self._detectors.values())
        if isinstance(selected, str):
            selected = (selected,)
        names = tuple(selected)
        if not names:
            raise ValueError("select at least one detector")
        if len(names) != len(set(names)):
            raise ValueError("selected detector names must be unique")
        unknown = set(names) - self._detectors.keys()
        if unknown:
            raise KeyError(f"unknown detectors: {', '.join(sorted(unknown))}")
        return tuple(self._detectors[name] for name in names)


def _load_rgb(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as loaded:
            return loaded.convert("RGB")
    raise TypeError("image must be a path or PIL image")
