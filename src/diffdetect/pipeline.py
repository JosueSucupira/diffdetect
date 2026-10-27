"""High-level unified pipeline for DiffDetect image analysis."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .classification import ImageClass
from .detector import BaseDetector
from .factory import (
    DEFAULT_DINOLIZER_THRESHOLD,
    DEFAULT_DISTILDIRE_THRESHOLD,
    create_detector,
)
from .hierarchical import (
    HierarchicalClassificationResult,
    HierarchicalDecisionReason,
    HierarchicalPolicy,
    classify_hierarchical,
    load_default_hierarchical_policy,
    load_hierarchical_policy,
)
from .runner import DetectorRun, DetectorRunner


@dataclass(frozen=True)
class DiffDetectResult:
    """Consolidated high-level result from DiffDetect execution."""

    decision: HierarchicalClassificationResult

    @property
    def label(self) -> ImageClass | None:
        """Final predicted class (REAL, SYNTHETIC, EDITED), or None if inconclusive."""
        return self.decision.label

    @property
    def reason(self) -> HierarchicalDecisionReason:
        """The stage or error condition that determined the final outcome."""
        return self.decision.reason

    @property
    def inconclusive(self) -> bool:
        """True if the decision abstained or could not reach a confident label."""
        return self.decision.inconclusive

    @property
    def p_artificial(self) -> float | None:
        """Calibrated probability that the image has AI intervention (synthetic or edited)."""
        return self.decision.p_artificial

    @property
    def p_synthetic_given_artificial(self) -> float | None:
        """Calibrated probability of full synthesis given that Level 1 predicted artificial."""
        return self.decision.p_synthetic_given_artificial

    @property
    def runs(self) -> tuple[DetectorRun, ...]:
        """Raw detector runs with individual scores and execution durations."""
        return self.decision.runs

    @property
    def scores(self) -> dict[str, float | None]:
        """Mapping from detector name to raw detection score."""
        return {
            run.name: (run.result.score if run.result else None)
            for run in self.decision.runs
        }

    @property
    def localization_map(self) -> Image.Image | None:
        """Floating-point localization map (mode F) if an edited detector returned one."""
        for run in self.decision.runs:
            if run.result and run.result.localization_map is not None:
                return run.result.localization_map
        return None

    @property
    def marked_area(self) -> float | None:
        """Fraction of pixels marked as edited in the localization map, if available."""
        for run in self.decision.runs:
            if run.result and "marked_area" in run.result.metadata:
                return float(run.result.metadata["marked_area"])
        return None


class DiffDetect:
    """Unified entrypoint for real, diffusion-generated, and diffusion-edited image detection."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        policy: HierarchicalPolicy | str | Path | None = None,
        detectors: Sequence[BaseDetector] | None = None,
        distildire_threshold: float = DEFAULT_DISTILDIRE_THRESHOLD,
        dinolizer_threshold: float = DEFAULT_DINOLIZER_THRESHOLD,
        classifier_weights: str | Path | None = None,
        adm_weights: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        if policy is None:
            self.policy = load_default_hierarchical_policy()
        elif isinstance(policy, (str, Path)):
            self.policy = load_hierarchical_policy(policy)
        elif isinstance(policy, HierarchicalPolicy):
            self.policy = policy
        else:
            raise TypeError("policy must be a HierarchicalPolicy, path, or None")

        if detectors is not None:
            self.detectors = tuple(detectors)
        else:
            distildire = create_detector(
                "distildire",
                threshold=distildire_threshold,
                device=device,
                classifier_weights=classifier_weights,
                adm_weights=adm_weights,
                **kwargs,
            )
            dinolizer = create_detector(
                "dinolizer",
                threshold=dinolizer_threshold,
                device=device,
                checkpoint_path=checkpoint_path,
                **kwargs,
            )
            self.detectors = (distildire, dinolizer)

        self.runner = DetectorRunner(self.detectors)

    def predict(self, image: str | Path | Image.Image) -> DiffDetectResult:
        """Run all detectors on an image and apply hierarchical classification."""
        runs = self.runner.run(image)
        decision = classify_hierarchical(runs, self.policy)
        return DiffDetectResult(decision)

    def predict_batch(
        self,
        images: Iterable[str | Path | Image.Image],
    ) -> list[DiffDetectResult]:
        """Run prediction across a sequence of images."""
        return [self.predict(image) for image in images]
