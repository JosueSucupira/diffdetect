"""Offline fitting utilities for the frozen hierarchical aggregation policy."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from diffdetect import HIERARCHICAL_FEATURES, POLICY_SCHEMA_VERSION


POLICY_VERSION = "diffdetect-hierarchical-aggregation-v1"
CALIBRATION_PROTOCOL_VERSION = "diffdetect-hierarchical-calibration-v1"
CALIBRATION_MANIFEST_SHA256 = (
    "5a23bee4f802b9c88dd41030711aee173d27a3f8974371153439c0c363cde3a5"
)
CALIBRATION_RESULTS_SHA256 = (
    "f104b2ea9b08e25122382583e081a6216b079f26f30c358e6c660edd53f7f27b"
)
CV_SEED = 20261007
CV_FOLDS = 5
MINIMUM_COVERAGE = 0.80
THRESHOLD_GRID = tuple(round(value / 100, 2) for value in range(50, 96))
TRUE_CLASSES = ("real", "synthetic", "edited")
MODEL_CONFIGURATION = {
    "type": "logistic_regression",
    "penalty": "l2",
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 1000,
}

EVIDENCE_FIELDS = (
    "protocol_version",
    "manifest_sha256",
    "manifest_index",
    "sample_id",
    "true_class",
    "source_dataset",
    "source_revision",
    "source_partition",
    "source_id",
    "input_sha256",
    "environment_id",
    "attempt",
    "run_started_utc",
    "distildire_score",
    "distildire_threshold",
    "distildire_detected",
    "distildire_duration_seconds",
    "distildire_error",
    "dinolizer_score",
    "dinolizer_threshold",
    "dinolizer_detected",
    "dinolizer_duration_seconds",
    "dinolizer_error",
    "dinolizer_marked_area",
    "dinolizer_processed_width",
    "dinolizer_processed_height",
    "dinolizer_window_count",
    "localization_width",
    "localization_height",
    "localization_mode",
    "total_duration_seconds",
)

OOF_FIELDS = (
    "manifest_index",
    "sample_id",
    "true_class",
    "source_dataset",
    "source_id",
    "group_id",
    "fold",
    *HIERARCHICAL_FEATURES,
    "p_artificial",
    "p_synthetic_given_artificial",
    "level1_threshold",
    "level2_threshold",
    "hierarchical_label",
    "decision_reason",
    "inconclusive",
    "correct",
)

FOLD_FIELDS = (
    "manifest_index",
    "sample_id",
    "true_class",
    "source_dataset",
    "source_id",
    "group_id",
    "fold",
)

THRESHOLD_FIELDS = (
    "level1_threshold",
    "level2_threshold",
    "eligible",
    "coverage",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "conclusive_accuracy",
    "conclusive_macro_f1",
)


@dataclass(frozen=True)
class EvidenceRow:
    manifest_index: int
    sample_id: str
    true_class: str
    source_dataset: str
    source_id: str
    input_sha256: str
    group_id: str
    features: tuple[float, ...]


@dataclass(frozen=True)
class EvidenceBundle:
    rows: tuple[EvidenceRow, ...]
    results_sha256: str
    results_metadata_sha256: str
    manifest_sha256: str
    execution_metadata: Mapping[str, object]


@dataclass(frozen=True)
class VariantFit:
    feature_names: tuple[str, ...]
    feature_indexes: tuple[int, ...]
    folds: tuple[int, ...]
    p_artificial: tuple[float, ...]
    p_level2: tuple[float, ...]
    selected_thresholds: tuple[float, float]
    selected_predictions: tuple[str | None, ...]
    selected_reasons: tuple[str, ...]
    selected_metrics: Mapping[str, object]
    threshold_table: tuple[Mapping[str, object], ...]
    scaler_means: tuple[float, ...]
    scaler_scales: tuple[float, ...]
    level1_coefficients: tuple[float, ...]
    level1_intercept: float
    level2_coefficients: tuple[float, ...]
    level2_intercept: float


def load_calibration_evidence(
    *,
    results_path: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    expected_results_sha256: str | None = CALIBRATION_RESULTS_SHA256,
    expected_manifest_sha256: str | None = CALIBRATION_MANIFEST_SHA256,
    expected_samples_per_class: int | None = 256,
) -> EvidenceBundle:
    """Validate and join the frozen raw results to their manifest identities."""

    results_path = _file(results_path, "results")
    metadata_path = _file(metadata_path, "results_metadata")
    manifest_path = _file(manifest_path, "manifest")

    results_sha256 = file_digest(results_path)
    manifest_sha256 = file_digest(manifest_path)
    if expected_results_sha256 and results_sha256 != expected_results_sha256:
        raise ValueError("calibration result SHA-256 does not match")
    if expected_manifest_sha256 and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("calibration manifest SHA-256 does not match")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("calibration result metadata is not valid JSON") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError("calibration result metadata must be a JSON object")
    if metadata.get("status") != "complete":
        raise ValueError("calibration result metadata is not complete")
    if metadata.get("aggregation_policy_applied") is not False:
        raise ValueError("calibration results must contain raw evidence only")
    if metadata.get("results_sha256") != results_sha256:
        raise ValueError("calibration result metadata SHA-256 does not match CSV")
    if metadata.get("manifest_sha256") != manifest_sha256:
        raise ValueError("calibration metadata manifest SHA-256 does not match")

    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    manifest_by_id = {row.get("sample_id", ""): row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("calibration manifest sample identifiers must be unique")

    with results_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(EVIDENCE_FIELDS):
            raise ValueError("calibration result columns do not match frozen schema")
        raw_rows = list(reader)

    records: list[EvidenceRow] = []
    for raw in raw_rows:
        sample_id = raw["sample_id"]
        manifest = manifest_by_id.get(sample_id)
        if manifest is None:
            raise ValueError(f"result sample is absent from manifest: {sample_id}")
        for field in (
            "manifest_index",
            "true_class",
            "source_dataset",
            "source_id",
            "input_sha256",
        ):
            if raw[field] != manifest[field]:
                raise ValueError(f"result field {field} does not match manifest")
        if raw["protocol_version"] != CALIBRATION_PROTOCOL_VERSION:
            raise ValueError("unexpected calibration protocol version")
        if raw["manifest_sha256"] != manifest_sha256:
            raise ValueError("result row manifest SHA-256 does not match")
        if raw["distildire_error"].strip() or raw["dinolizer_error"].strip():
            raise ValueError(f"detector error in calibration row: {sample_id}")

        features = tuple(
            _bounded_float(raw[name], name)
            for name in HIERARCHICAL_FEATURES
        )
        true_class = raw["true_class"]
        if true_class not in TRUE_CLASSES:
            raise ValueError(f"invalid true class: {true_class}")
        source_dataset = raw["source_dataset"]
        source_id = raw["source_id"]
        records.append(
            EvidenceRow(
                manifest_index=_positive_int(raw["manifest_index"], "manifest_index"),
                sample_id=sample_id,
                true_class=true_class,
                source_dataset=source_dataset,
                source_id=source_id,
                input_sha256=raw["input_sha256"],
                group_id=_group_id(source_dataset, source_id),
                features=features,
            )
        )

    records.sort(key=lambda record: record.manifest_index)
    if tuple(record.manifest_index for record in records) != tuple(
        range(1, len(records) + 1)
    ):
        raise ValueError("calibration result indexes must be consecutive")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("calibration result sample identifiers must be unique")
    if len(records) != len(manifest_rows):
        raise ValueError("calibration results do not cover the complete manifest")

    counts = Counter(record.true_class for record in records)
    if expected_samples_per_class is not None:
        expected_counts = {
            true_class: expected_samples_per_class for true_class in TRUE_CLASSES
        }
        if counts != expected_counts:
            raise ValueError(f"calibration results are not balanced: {dict(counts)}")
    _validate_groups(records)
    if metadata.get("completed_samples") != len(records):
        raise ValueError("metadata completed sample count does not match CSV")

    return EvidenceBundle(
        rows=tuple(records),
        results_sha256=results_sha256,
        results_metadata_sha256=file_digest(metadata_path),
        manifest_sha256=manifest_sha256,
        execution_metadata=metadata,
    )


def grouped_fold_assignments(
    rows: Sequence[EvidenceRow],
    *,
    folds: int = CV_FOLDS,
    seed: int = CV_SEED,
) -> tuple[tuple[int, ...], tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]]:
    """Create deterministic stratified group folds and expose their indexes."""

    try:
        import numpy
        from sklearn.model_selection import StratifiedGroupKFold
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "fit-policy requires the 'experiments' optional dependencies"
        ) from exc

    labels = numpy.asarray([row.true_class for row in rows], dtype=object)
    groups = numpy.asarray([row.group_id for row in rows], dtype=object)
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    assignments = numpy.zeros(len(rows), dtype=int)
    indexes = []
    placeholder = numpy.zeros((len(rows), 1), dtype=float)
    for fold, (train, validation) in enumerate(
        splitter.split(placeholder, labels, groups),
        start=1,
    ):
        assignments[validation] = fold
        indexes.append((tuple(int(value) for value in train), tuple(int(value) for value in validation)))
    if any(value == 0 for value in assignments):
        raise RuntimeError("grouped cross-validation left samples without a fold")
    for group in set(groups):
        group_folds = {
            int(assignments[index])
            for index, value in enumerate(groups)
            if value == group
        }
        if len(group_folds) != 1:
            raise RuntimeError(f"group was split across folds: {group}")
    return tuple(int(value) for value in assignments), tuple(indexes)


def fit_variant(
    rows: Sequence[EvidenceRow],
    *,
    feature_names: Sequence[str],
    fold_indexes: Sequence[tuple[Sequence[int], Sequence[int]]],
    fold_assignments: Sequence[int],
    minimum_coverage: float = MINIMUM_COVERAGE,
) -> VariantFit:
    """Fit one feature variant with leakage-safe out-of-fold probabilities."""

    try:
        import numpy
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "fit-policy requires the 'experiments' optional dependencies"
        ) from exc

    names = tuple(feature_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("feature_names must be non-empty and unique")
    try:
        feature_indexes = tuple(HIERARCHICAL_FEATURES.index(name) for name in names)
    except ValueError as exc:
        raise ValueError("unknown hierarchical feature") from exc

    matrix = numpy.asarray(
        [[row.features[index] for index in feature_indexes] for row in rows],
        dtype=float,
    )
    labels = numpy.asarray([row.true_class for row in rows], dtype=object)
    p_artificial = numpy.full(len(rows), numpy.nan, dtype=float)
    p_level2 = numpy.full(len(rows), numpy.nan, dtype=float)

    for train_indexes, validation_indexes in fold_indexes:
        train = numpy.asarray(train_indexes, dtype=int)
        validation = numpy.asarray(validation_indexes, dtype=int)
        scaler = StandardScaler().fit(matrix[train])
        training_features = scaler.transform(matrix[train])
        validation_features = scaler.transform(matrix[validation])

        level1_targets = (labels[train] != "real").astype(int)
        level1 = _new_logistic_regression().fit(training_features, level1_targets)
        p_artificial[validation] = level1.predict_proba(validation_features)[:, 1]

        artificial_training = labels[train] != "real"
        level2_targets = (labels[train][artificial_training] == "synthetic").astype(int)
        level2 = _new_logistic_regression().fit(
            training_features[artificial_training],
            level2_targets,
        )
        p_level2[validation] = level2.predict_proba(validation_features)[:, 1]

    if not numpy.all(numpy.isfinite(p_artificial)) or not numpy.all(
        numpy.isfinite(p_level2)
    ):
        raise RuntimeError("cross-validation produced non-finite probabilities")

    threshold_table, selected = select_thresholds(
        [row.true_class for row in rows],
        p_artificial,
        p_level2,
        minimum_coverage=minimum_coverage,
    )
    level1_threshold = float(selected["level1_threshold"])
    level2_threshold = float(selected["level2_threshold"])
    predictions, reasons = hierarchical_predictions(
        p_artificial,
        p_level2,
        level1_threshold,
        level2_threshold,
    )

    full_scaler = StandardScaler().fit(matrix)
    standardized = full_scaler.transform(matrix)
    full_level1 = _new_logistic_regression().fit(
        standardized,
        (labels != "real").astype(int),
    )
    artificial = labels != "real"
    full_level2 = _new_logistic_regression().fit(
        standardized[artificial],
        (labels[artificial] == "synthetic").astype(int),
    )

    return VariantFit(
        feature_names=names,
        feature_indexes=feature_indexes,
        folds=tuple(int(value) for value in fold_assignments),
        p_artificial=tuple(float(value) for value in p_artificial),
        p_level2=tuple(float(value) for value in p_level2),
        selected_thresholds=(level1_threshold, level2_threshold),
        selected_predictions=tuple(predictions),
        selected_reasons=tuple(reasons),
        selected_metrics=multiclass_metrics(
            [row.true_class for row in rows],
            predictions,
        ),
        threshold_table=tuple(threshold_table),
        scaler_means=tuple(float(value) for value in full_scaler.mean_),
        scaler_scales=tuple(float(value) for value in full_scaler.scale_),
        level1_coefficients=tuple(float(value) for value in full_level1.coef_[0]),
        level1_intercept=float(full_level1.intercept_[0]),
        level2_coefficients=tuple(float(value) for value in full_level2.coef_[0]),
        level2_intercept=float(full_level2.intercept_[0]),
    )


def select_thresholds(
    true_labels: Sequence[str],
    p_artificial: Sequence[float],
    p_level2: Sequence[float],
    *,
    minimum_coverage: float = MINIMUM_COVERAGE,
) -> tuple[list[dict[str, object]], Mapping[str, object]]:
    """Evaluate the prespecified grid and select by the frozen tie-break order."""

    table = []
    for level1_threshold in THRESHOLD_GRID:
        for level2_threshold in THRESHOLD_GRID:
            predictions, _ = hierarchical_predictions(
                p_artificial,
                p_level2,
                level1_threshold,
                level2_threshold,
            )
            metrics = multiclass_metrics(true_labels, predictions)
            end_to_end = metrics["end_to_end"]
            row = {
                "level1_threshold": level1_threshold,
                "level2_threshold": level2_threshold,
                "eligible": end_to_end["coverage"] >= minimum_coverage,
                "coverage": end_to_end["coverage"],
                "accuracy": end_to_end["accuracy"],
                "balanced_accuracy": end_to_end["balanced_accuracy"],
                "macro_precision": end_to_end["macro_precision"],
                "macro_recall": end_to_end["macro_recall"],
                "macro_f1": end_to_end["macro_f1"],
                "conclusive_accuracy": end_to_end["conclusive_accuracy"],
                "conclusive_macro_f1": end_to_end["conclusive_macro_f1"],
            }
            table.append(row)

    eligible = [row for row in table if row["eligible"]]
    if not eligible:
        raise RuntimeError(
            f"no confidence-threshold pair reaches {minimum_coverage:.0%} coverage"
        )
    selected = max(
        eligible,
        key=lambda row: (
            row["macro_f1"],
            row["balanced_accuracy"],
            row["coverage"],
            -(row["level1_threshold"] + row["level2_threshold"]),
            -row["level1_threshold"],
            -row["level2_threshold"],
        ),
    )
    return table, selected


def hierarchical_predictions(
    p_artificial: Sequence[float],
    p_level2: Sequence[float],
    level1_threshold: float,
    level2_threshold: float,
) -> tuple[list[str | None], list[str]]:
    """Apply both confidence regions to aligned probability sequences."""

    if len(p_artificial) != len(p_level2):
        raise ValueError("probability sequences must have equal lengths")
    predictions: list[str | None] = []
    reasons = []
    for artificial, synthetic in zip(p_artificial, p_level2):
        artificial = float(artificial)
        synthetic = float(synthetic)
        if not math.isfinite(artificial) or not math.isfinite(synthetic):
            raise ValueError("probabilities must be finite")
        if artificial <= 1.0 - level1_threshold:
            predictions.append("real")
            reasons.append("level1_real")
        elif artificial < level1_threshold:
            predictions.append(None)
            reasons.append("level1_inconclusive")
        elif synthetic <= 1.0 - level2_threshold:
            predictions.append("edited")
            reasons.append("level2_edited")
        elif synthetic >= level2_threshold:
            predictions.append("synthetic")
            reasons.append("level2_synthetic")
        else:
            predictions.append(None)
            reasons.append("level2_inconclusive")
    return predictions, reasons


def multiclass_metrics(
    true_labels: Sequence[str],
    predictions: Sequence[str | None],
) -> dict[str, object]:
    """Calculate three-class metrics with abstentions retained as misses."""

    if len(true_labels) != len(predictions) or not true_labels:
        raise ValueError("labels and predictions must be non-empty and aligned")
    if any(label not in TRUE_CLASSES for label in true_labels):
        raise ValueError("true labels contain an unknown class")
    if any(label is not None and label not in TRUE_CLASSES for label in predictions):
        raise ValueError("predictions contain an unknown class")

    total = len(true_labels)
    conclusive_indexes = [
        index for index, prediction in enumerate(predictions) if prediction is not None
    ]
    correct = sum(
        truth == prediction for truth, prediction in zip(true_labels, predictions)
    )
    per_class = {}
    recalls = []
    precisions = []
    f1_values = []
    conclusive_f1_values = []
    confusion = {
        truth: {prediction: 0 for prediction in TRUE_CLASSES}
        for truth in TRUE_CLASSES
    }
    outcomes = {
        truth: {**{prediction: 0 for prediction in TRUE_CLASSES}, "inconclusive": 0}
        for truth in TRUE_CLASSES
    }

    for truth, prediction in zip(true_labels, predictions):
        outcomes[truth][prediction or "inconclusive"] += 1
        if prediction is not None:
            confusion[truth][prediction] += 1

    for target in TRUE_CLASSES:
        true_positive = sum(
            truth == target and prediction == target
            for truth, prediction in zip(true_labels, predictions)
        )
        actual = sum(truth == target for truth in true_labels)
        predicted = sum(prediction == target for prediction in predictions)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[target] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual,
            "predicted": predicted,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)

        conclusive_actual = sum(
            true_labels[index] == target for index in conclusive_indexes
        )
        conclusive_predicted = sum(
            predictions[index] == target for index in conclusive_indexes
        )
        conclusive_true_positive = sum(
            true_labels[index] == target and predictions[index] == target
            for index in conclusive_indexes
        )
        conclusive_precision = (
            conclusive_true_positive / conclusive_predicted
            if conclusive_predicted
            else 0.0
        )
        conclusive_recall = (
            conclusive_true_positive / conclusive_actual
            if conclusive_actual
            else 0.0
        )
        conclusive_f1_values.append(
            2.0 * conclusive_precision * conclusive_recall
            / (conclusive_precision + conclusive_recall)
            if conclusive_precision + conclusive_recall
            else 0.0
        )

    conclusive_count = len(conclusive_indexes)
    conclusive_correct = sum(
        true_labels[index] == predictions[index] for index in conclusive_indexes
    )
    return {
        "sample_count": total,
        "outcome_table": outcomes,
        "conclusive_confusion_matrix": confusion,
        "end_to_end": {
            "accuracy": correct / total,
            "balanced_accuracy": sum(recalls) / len(TRUE_CLASSES),
            "macro_precision": sum(precisions) / len(TRUE_CLASSES),
            "macro_recall": sum(recalls) / len(TRUE_CLASSES),
            "macro_f1": sum(f1_values) / len(TRUE_CLASSES),
            "coverage": conclusive_count / total,
            "abstention_rate": 1.0 - conclusive_count / total,
            "conclusive_samples": conclusive_count,
            "correct_samples": correct,
            "conclusive_accuracy": (
                conclusive_correct / conclusive_count if conclusive_count else 0.0
            ),
            "conclusive_macro_f1": sum(conclusive_f1_values) / len(TRUE_CLASSES),
            "per_class": per_class,
        },
    }


def fit_and_write_policy(
    *,
    bundle: EvidenceBundle,
    policy_output: str | Path,
    oof_output: str | Path,
    folds_output: str | Path,
    thresholds_output: str | Path,
    metrics_output: str | Path,
    checksums_output: str | Path,
) -> dict[str, object]:
    """Fit the primary hierarchy and prespecified ablations, then write artifacts."""

    rows = bundle.rows
    assignments, indexes = grouped_fold_assignments(rows)
    primary = fit_variant(
        rows,
        feature_names=HIERARCHICAL_FEATURES,
        fold_indexes=indexes,
        fold_assignments=assignments,
    )
    without_distildire = fit_variant(
        rows,
        feature_names=("dinolizer_score", "dinolizer_marked_area"),
        fold_indexes=indexes,
        fold_assignments=assignments,
    )
    without_marked_area = fit_variant(
        rows,
        feature_names=("distildire_score", "dinolizer_score"),
        fold_indexes=indexes,
        fold_assignments=assignments,
    )

    created_utc = datetime.now(timezone.utc).isoformat()
    level1_threshold, level2_threshold = primary.selected_thresholds
    execution_artifacts = bundle.execution_metadata.get("artifacts", {})
    policy = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "created_utc": created_utc,
        "feature_names": list(primary.feature_names),
        "feature_ranges": {
            name: {"minimum": 0.0, "maximum": 1.0}
            for name in primary.feature_names
        },
        "scaler": {
            "type": "standard_scaler",
            "means": list(primary.scaler_means),
            "scales": list(primary.scaler_scales),
        },
        "models": {
            "level1": {
                "negative_class": "real",
                "positive_class": "artificial",
                "coefficients": list(primary.level1_coefficients),
                "intercept": primary.level1_intercept,
                "configuration": MODEL_CONFIGURATION,
            },
            "level2": {
                "training_classes": ["edited", "synthetic"],
                "negative_class": "edited",
                "positive_class": "synthetic",
                "coefficients": list(primary.level2_coefficients),
                "intercept": primary.level2_intercept,
                "configuration": MODEL_CONFIGURATION,
            },
        },
        "confidence_thresholds": {
            "level1": level1_threshold,
            "level2": level2_threshold,
        },
        "detectors": {
            "distildire": {
                "name": "distildire",
                "target": "synthetic",
                "repository_revision": execution_artifacts.get("distildire_revision"),
                "classifier_sha256": execution_artifacts.get(
                    "distildire_classifier_sha256"
                ),
                "adm_md5": execution_artifacts.get("distildire_adm_md5"),
                "score_definition": "sigmoid of the DistilDIRE classifier logit",
                "boolean_threshold_for_audit_only": execution_artifacts.get(
                    "distildire_threshold"
                ),
            },
            "dinolizer": {
                "name": "dinolizer",
                "target": "edited",
                "repository_revision": execution_artifacts.get("dinolizer_revision"),
                "checkpoint_sha256": execution_artifacts.get(
                    "dinolizer_checkpoint_sha256"
                ),
                "score_definition": (
                    "99th percentile of the edited-class probability map"
                ),
                "marked_area_definition": (
                    "proportion of pixels with edited probability at least 0.5"
                ),
                "boolean_threshold_for_audit_only": execution_artifacts.get(
                    "dinolizer_threshold"
                ),
            },
        },
        "calibration": {
            "protocol_version": CALIBRATION_PROTOCOL_VERSION,
            "manifest_sha256": bundle.manifest_sha256,
            "results_sha256": bundle.results_sha256,
            "results_metadata_sha256": bundle.results_metadata_sha256,
            "sample_count": len(rows),
            "class_counts": dict(sorted(Counter(row.true_class for row in rows).items())),
            "previous_validation_used_for_fitting": False,
        },
        "cross_validation": {
            "splitter": "StratifiedGroupKFold",
            "folds": CV_FOLDS,
            "shuffle": True,
            "seed": CV_SEED,
            "grouping_rule": {
                "cocoglide": "source_id; real/edited pair kept together",
                "diffusiondb": "source_id",
            },
            "standardization_scope": "training fold only",
            "threshold_grid": {
                "minimum": THRESHOLD_GRID[0],
                "maximum": THRESHOLD_GRID[-1],
                "step": 0.01,
            },
            "minimum_coverage": MINIMUM_COVERAGE,
            "selection_order": [
                "highest end-to-end macro_f1",
                "highest balanced_accuracy",
                "highest coverage",
                "smallest threshold sum",
                "smallest level1 threshold",
                "smallest level2 threshold",
            ],
            "out_of_fold_metrics": primary.selected_metrics,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
    }
    _write_json(policy_output, policy)
    policy_sha256 = file_digest(policy_output)

    _write_csv(
        folds_output,
        FOLD_FIELDS,
        (
            {
                "manifest_index": row.manifest_index,
                "sample_id": row.sample_id,
                "true_class": row.true_class,
                "source_dataset": row.source_dataset,
                "source_id": row.source_id,
                "group_id": row.group_id,
                "fold": fold,
            }
            for row, fold in zip(rows, assignments)
        ),
    )
    _write_csv(
        oof_output,
        OOF_FIELDS,
        (
            {
                "manifest_index": row.manifest_index,
                "sample_id": row.sample_id,
                "true_class": row.true_class,
                "source_dataset": row.source_dataset,
                "source_id": row.source_id,
                "group_id": row.group_id,
                "fold": fold,
                **dict(zip(HIERARCHICAL_FEATURES, row.features)),
                "p_artificial": p_artificial,
                "p_synthetic_given_artificial": p_level2,
                "level1_threshold": level1_threshold,
                "level2_threshold": level2_threshold,
                "hierarchical_label": prediction or "",
                "decision_reason": reason,
                "inconclusive": prediction is None,
                "correct": prediction == row.true_class,
            }
            for row, fold, p_artificial, p_level2, prediction, reason in zip(
                rows,
                assignments,
                primary.p_artificial,
                primary.p_level2,
                primary.selected_predictions,
                primary.selected_reasons,
            )
        ),
    )
    _write_csv(thresholds_output, THRESHOLD_FIELDS, primary.threshold_table)

    fold_counts = {
        str(fold): dict(
            sorted(
                Counter(
                    row.true_class
                    for row, assigned in zip(rows, assignments)
                    if assigned == fold
                ).items()
            )
        )
        for fold in range(1, CV_FOLDS + 1)
    }
    metrics = {
        "schema_version": "diffdetect-hierarchical-calibration-metrics-v1",
        "created_utc": created_utc,
        "policy_file": Path(policy_output).name,
        "policy_sha256": policy_sha256,
        "source": {
            "manifest_sha256": bundle.manifest_sha256,
            "results_sha256": bundle.results_sha256,
            "results_metadata_sha256": bundle.results_metadata_sha256,
        },
        "fold_class_counts": fold_counts,
        "primary": _variant_summary(primary),
        "ablations": {
            "without_distildire_score": _variant_summary(without_distildire),
            "without_dinolizer_marked_area": _variant_summary(without_marked_area),
        },
    }
    _write_json(metrics_output, metrics)

    artifact_paths = (
        Path(policy_output),
        Path(oof_output),
        Path(folds_output),
        Path(thresholds_output),
        Path(metrics_output),
    )
    checksum_path = Path(checksums_output)
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checksum_path.with_name(f".{checksum_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(
            f"{bundle.manifest_sha256}  hierarchical-calibration-manifest.csv\n"
        )
        stream.write(f"{bundle.results_sha256}  hierarchical-calibration.csv\n")
        stream.write(
            f"{bundle.results_metadata_sha256}  "
            "hierarchical-calibration.csv.metadata.json\n"
        )
        for path in artifact_paths:
            stream.write(f"{file_digest(path)}  {path.name}\n")
    temporary.replace(checksum_path)
    return metrics


def _variant_summary(variant: VariantFit) -> dict[str, object]:
    return {
        "feature_names": list(variant.feature_names),
        "selected_thresholds": {
            "level1": variant.selected_thresholds[0],
            "level2": variant.selected_thresholds[1],
        },
        "out_of_fold_metrics": variant.selected_metrics,
    }


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_logistic_regression():
    from sklearn.linear_model import LogisticRegression

    # lbfgs uses L2 regularization by default. Leaving ``penalty`` at its
    # version-specific default avoids the scikit-learn 1.8 deprecation while
    # preserving the protocol's fixed L2 model on older supported releases.
    return LogisticRegression(
        C=MODEL_CONFIGURATION["C"],
        class_weight=MODEL_CONFIGURATION["class_weight"],
        solver=MODEL_CONFIGURATION["solver"],
        max_iter=MODEL_CONFIGURATION["max_iter"],
        random_state=CV_SEED,
    )


def _validate_groups(rows: Sequence[EvidenceRow]) -> None:
    grouped: dict[str, list[EvidenceRow]] = {}
    for row in rows:
        grouped.setdefault(row.group_id, []).append(row)
    for group_id, members in grouped.items():
        classes = {member.true_class for member in members}
        if group_id.startswith("cocoglide:"):
            if len(members) != 2 or classes != {"real", "edited"}:
                raise ValueError(f"invalid CocoGlide calibration group: {group_id}")
        elif len(members) != 1 or classes != {"synthetic"}:
            raise ValueError(f"invalid DiffusionDB calibration group: {group_id}")


def _group_id(source_dataset: str, source_id: str) -> str:
    if not source_id:
        raise ValueError("source_id must not be empty")
    if source_dataset == "nebula/CocoGlide":
        return f"cocoglide:{source_id}"
    if source_dataset == "poloclub/diffusiondb":
        return f"diffusiondb:{source_id}"
    raise ValueError(f"unexpected source dataset: {source_dataset}")


def _bounded_float(value: object, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return converted


def _positive_int(value: object, name: str) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if converted <= 0:
        raise ValueError(f"{name} must be positive")
    return converted


def _file(value: str | Path, name: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")
    return path.resolve()


def _write_json(path: str | Path, value: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(output)


def _write_csv(
    path: str | Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, object]] | object,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
