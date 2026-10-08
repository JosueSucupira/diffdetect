"""Pure-Python runtime for a frozen hierarchical aggregation policy."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path

from .classification import ImageClass
from .detector import DetectionTarget
from .runner import DetectorRun


POLICY_SCHEMA_VERSION = "diffdetect-hierarchical-policy-v1"
HIERARCHICAL_FEATURES = (
    "distildire_score",
    "dinolizer_score",
    "dinolizer_marked_area",
)


class HierarchicalDecisionReason(str, Enum):
    """Stage that produced a label or an inconclusive decision."""

    LEVEL1_REAL = "level1_real"
    LEVEL1_INCONCLUSIVE = "level1_inconclusive"
    LEVEL2_SYNTHETIC = "level2_synthetic"
    LEVEL2_EDITED = "level2_edited"
    LEVEL2_INCONCLUSIVE = "level2_inconclusive"
    DETECTOR_ERROR = "detector_error"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    INVALID_FEATURES = "invalid_features"


@dataclass(frozen=True)
class LogisticPolicyModel:
    """Serialized coefficients for one binary logistic model."""

    coefficients: tuple[float, ...]
    intercept: float


@dataclass(frozen=True)
class HierarchicalPolicy:
    """Validated coefficients, scaler, and thresholds used at runtime."""

    policy_version: str
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    level1: LogisticPolicyModel
    level2: LogisticPolicyModel
    level1_threshold: float
    level2_threshold: float
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.policy_version:
            raise ValueError("policy_version must not be empty")
        if self.feature_names != HIERARCHICAL_FEATURES:
            raise ValueError("unexpected hierarchical feature order")
        feature_count = len(self.feature_names)
        for name, values in (
            ("feature_means", self.feature_means),
            ("feature_scales", self.feature_scales),
            ("level1 coefficients", self.level1.coefficients),
            ("level2 coefficients", self.level2.coefficients),
        ):
            if len(values) != feature_count:
                raise ValueError(f"{name} must contain {feature_count} values")
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain only finite values")
        if any(scale <= 0.0 for scale in self.feature_scales):
            raise ValueError("feature scales must be positive")
        for name, model in (("level1", self.level1), ("level2", self.level2)):
            if not math.isfinite(model.intercept):
                raise ValueError(f"{name} intercept must be finite")
        for name, threshold in (
            ("level1_threshold", self.level1_threshold),
            ("level2_threshold", self.level2_threshold),
        ):
            if not 0.5 <= threshold < 1.0:
                raise ValueError(f"{name} must be in [0.5, 1.0)")
        _validate_sha256(self.artifact_sha256, "artifact_sha256")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        artifact_sha256: str,
    ) -> HierarchicalPolicy:
        """Load the runtime subset of the versioned JSON policy schema."""

        if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError("unexpected hierarchical policy schema version")
        try:
            scaler = _mapping(payload["scaler"], "scaler")
            models = _mapping(payload["models"], "models")
            level1 = _mapping(models["level1"], "models.level1")
            level2 = _mapping(models["level2"], "models.level2")
            thresholds = _mapping(
                payload["confidence_thresholds"],
                "confidence_thresholds",
            )
            return cls(
                policy_version=_text(payload["policy_version"], "policy_version"),
                feature_names=_text_tuple(payload["feature_names"], "feature_names"),
                feature_means=_float_tuple(scaler["means"], "scaler.means"),
                feature_scales=_float_tuple(scaler["scales"], "scaler.scales"),
                level1=LogisticPolicyModel(
                    coefficients=_float_tuple(
                        level1["coefficients"],
                        "models.level1.coefficients",
                    ),
                    intercept=_finite_float(
                        level1["intercept"],
                        "models.level1.intercept",
                    ),
                ),
                level2=LogisticPolicyModel(
                    coefficients=_float_tuple(
                        level2["coefficients"],
                        "models.level2.coefficients",
                    ),
                    intercept=_finite_float(
                        level2["intercept"],
                        "models.level2.intercept",
                    ),
                ),
                level1_threshold=_finite_float(
                    thresholds["level1"],
                    "confidence_thresholds.level1",
                ),
                level2_threshold=_finite_float(
                    thresholds["level2"],
                    "confidence_thresholds.level2",
                ),
                artifact_sha256=artifact_sha256,
            )
        except KeyError as exc:
            raise ValueError(f"policy is missing field: {exc.args[0]}") from exc


@dataclass(frozen=True)
class HierarchicalClassificationResult:
    """Hierarchical decision with probabilities and unchanged detector runs."""

    label: ImageClass | None
    reason: HierarchicalDecisionReason
    runs: tuple[DetectorRun, ...]
    policy_version: str
    policy_sha256: str
    standardized_features: tuple[tuple[str, float], ...]
    p_artificial: float | None
    p_synthetic_given_artificial: float | None
    level1_threshold: float
    level2_threshold: float

    @property
    def inconclusive(self) -> bool:
        return self.label is None


def load_hierarchical_policy(path: str | Path) -> HierarchicalPolicy:
    """Read a policy JSON and bind its exact file digest to runtime results."""

    policy_path = Path(path)
    data = policy_path.read_bytes()
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("hierarchical policy is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("hierarchical policy must be a JSON object")
    return HierarchicalPolicy.from_mapping(
        payload,
        artifact_sha256=hashlib.sha256(data).hexdigest(),
    )


def classify_hierarchical(
    runs: Iterable[DetectorRun],
    policy: HierarchicalPolicy,
) -> HierarchicalClassificationResult:
    """Apply the frozen two-level policy without importing scikit-learn."""

    if not isinstance(policy, HierarchicalPolicy):
        raise TypeError("policy must be a HierarchicalPolicy")
    records = tuple(runs)
    if any(not isinstance(run, DetectorRun) for run in records):
        raise TypeError("runs must contain only DetectorRun values")

    def result(
        label: ImageClass | None,
        reason: HierarchicalDecisionReason,
        *,
        features: tuple[tuple[str, float], ...] = (),
        p_artificial: float | None = None,
        p_level2: float | None = None,
    ) -> HierarchicalClassificationResult:
        return HierarchicalClassificationResult(
            label=label,
            reason=reason,
            runs=records,
            policy_version=policy.policy_version,
            policy_sha256=policy.artifact_sha256,
            standardized_features=features,
            p_artificial=p_artificial,
            p_synthetic_given_artificial=p_level2,
            level1_threshold=policy.level1_threshold,
            level2_threshold=policy.level2_threshold,
        )

    if any(run.error is not None for run in records):
        return result(None, HierarchicalDecisionReason.DETECTOR_ERROR)

    named: dict[str, DetectorRun] = {}
    for run in records:
        if run.name in named:
            return result(None, HierarchicalDecisionReason.INSUFFICIENT_COVERAGE)
        named[run.name] = run
    if set(named) != {"distildire", "dinolizer"}:
        return result(None, HierarchicalDecisionReason.INSUFFICIENT_COVERAGE)

    distildire = named["distildire"]
    dinolizer = named["dinolizer"]
    if (
        distildire.target is not DetectionTarget.SYNTHETIC
        or dinolizer.target is not DetectionTarget.EDITED
        or distildire.result is None
        or dinolizer.result is None
    ):
        return result(None, HierarchicalDecisionReason.INSUFFICIENT_COVERAGE)

    raw_values = (
        distildire.result.score,
        dinolizer.result.score,
        dinolizer.result.metadata.get("marked_area"),
    )
    if any(not _bounded_probability(value) for value in raw_values):
        return result(None, HierarchicalDecisionReason.INVALID_FEATURES)

    values = tuple(float(value) for value in raw_values)
    standardized_values = tuple(
        (value - mean) / scale
        for value, mean, scale in zip(
            values,
            policy.feature_means,
            policy.feature_scales,
        )
    )
    standardized_features = tuple(zip(policy.feature_names, standardized_values))
    p_artificial = _logistic_probability(policy.level1, standardized_values)

    if p_artificial <= 1.0 - policy.level1_threshold:
        return result(
            ImageClass.REAL,
            HierarchicalDecisionReason.LEVEL1_REAL,
            features=standardized_features,
            p_artificial=p_artificial,
        )
    if p_artificial < policy.level1_threshold:
        return result(
            None,
            HierarchicalDecisionReason.LEVEL1_INCONCLUSIVE,
            features=standardized_features,
            p_artificial=p_artificial,
        )

    p_level2 = _logistic_probability(policy.level2, standardized_values)
    if p_level2 <= 1.0 - policy.level2_threshold:
        return result(
            ImageClass.EDITED,
            HierarchicalDecisionReason.LEVEL2_EDITED,
            features=standardized_features,
            p_artificial=p_artificial,
            p_level2=p_level2,
        )
    if p_level2 >= policy.level2_threshold:
        return result(
            ImageClass.SYNTHETIC,
            HierarchicalDecisionReason.LEVEL2_SYNTHETIC,
            features=standardized_features,
            p_artificial=p_artificial,
            p_level2=p_level2,
        )
    return result(
        None,
        HierarchicalDecisionReason.LEVEL2_INCONCLUSIVE,
        features=standardized_features,
        p_artificial=p_artificial,
        p_level2=p_level2,
    )


def _logistic_probability(
    model: LogisticPolicyModel,
    features: Sequence[float],
) -> float:
    logit = model.intercept + sum(
        coefficient * value
        for coefficient, value in zip(model.coefficients, features)
    )
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _bounded_probability(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return tuple(_text(item, name) for item in value)


def _float_tuple(value: object, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return tuple(_finite_float(item, name) for item in value)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must contain real numbers")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must contain finite numbers")
    return converted


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must contain 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must contain 64 hexadecimal characters") from exc
