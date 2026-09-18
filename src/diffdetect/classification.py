"""Combine specialized detector decisions into the three output classes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from .detector import DetectionTarget
from .runner import DetectorRun


class ImageClass(str, Enum):
    """Classes used by the three-class analysis."""

    REAL = "real"
    SYNTHETIC = "synthetic"
    EDITED = "edited"


class DecisionReason(str, Enum):
    """Why a class was returned or the analysis was inconclusive."""

    NO_EVIDENCE = "no_evidence"
    SYNTHETIC_EVIDENCE = "synthetic_evidence"
    EDITED_EVIDENCE = "edited_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    DETECTOR_ERROR = "detector_error"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"


@dataclass(frozen=True)
class ClassificationResult:
    """Final label, decision reason, and unchanged individual detector runs.

    ``label=None`` is an inconclusive status, not an additional image class.
    """

    label: ImageClass | None
    reason: DecisionReason
    runs: tuple[DetectorRun, ...]

    @property
    def inconclusive(self) -> bool:
        return self.label is None


def classify(runs: Iterable[DetectorRun]) -> ClassificationResult:
    """Apply a conservative three-class rule to detector executions.

    Every detector must succeed, and at least one successful detector must
    cover each target. Within a target, any positive detector is evidence for
    that target. Positive evidence for both targets is inconclusive. Raw scores
    are not compared across detectors.
    """

    records = tuple(runs)
    if any(not isinstance(run, DetectorRun) for run in records):
        raise TypeError("runs must contain only DetectorRun values")

    if any(run.error is not None for run in records):
        return ClassificationResult(None, DecisionReason.DETECTOR_ERROR, records)

    covered = {run.target for run in records}
    if covered != {DetectionTarget.SYNTHETIC, DetectionTarget.EDITED}:
        return ClassificationResult(None, DecisionReason.INSUFFICIENT_COVERAGE, records)

    positive = {
        run.target for run in records if run.result is not None and run.result.detected
    }
    if len(positive) == 2:
        return ClassificationResult(None, DecisionReason.CONFLICTING_EVIDENCE, records)
    if DetectionTarget.SYNTHETIC in positive:
        return ClassificationResult(
            ImageClass.SYNTHETIC, DecisionReason.SYNTHETIC_EVIDENCE, records
        )
    if DetectionTarget.EDITED in positive:
        return ClassificationResult(ImageClass.EDITED, DecisionReason.EDITED_EVIDENCE, records)
    return ClassificationResult(ImageClass.REAL, DecisionReason.NO_EVIDENCE, records)
