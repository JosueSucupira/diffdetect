"""Read-only retrospective evaluation of a frozen hierarchical policy."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from diffdetect import (
    DetectionResult,
    DetectionTarget,
    DetectorRun,
    classify_hierarchical,
    load_hierarchical_policy,
)
if __package__:
    from experiments.hierarchical_training import TRUE_CLASSES, multiclass_metrics
else:
    from hierarchical_training import TRUE_CLASSES, multiclass_metrics


BASELINE_RESULTS_SHA256 = (
    "b6788b97a249ccf6d4ab2c7fc1ea4d3149a9707371aa3ad1eeb9f588743a141a"
)
BASELINE_MANIFEST_SHA256 = (
    "9dcda4b93d8c73df7b1240629bec209566cd6119bb7f800c2906b8c3826b8777"
)
BOOTSTRAP_SEED = 20261007
BOOTSTRAP_RESAMPLES = 2000
OUTCOMES = (*TRUE_CLASSES, "inconclusive")

PREDICTION_FIELDS = (
    "manifest_index",
    "sample_id",
    "true_class",
    "source_dataset",
    "source_id",
    "group_id",
    "distildire_score",
    "dinolizer_score",
    "dinolizer_marked_area",
    "baseline_label",
    "baseline_reason",
    "hierarchical_label",
    "hierarchical_reason",
    "hierarchical_inconclusive",
    "p_artificial",
    "p_synthetic_given_artificial",
    "standardized_distildire_score",
    "standardized_dinolizer_score",
    "standardized_dinolizer_marked_area",
    "level1_threshold",
    "level2_threshold",
    "policy_sha256",
)


def evaluate_retrospective(
    *,
    policy_path: str | Path,
    expected_policy_sha256: str,
    baseline_results_path: str | Path,
    baseline_metadata_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Apply an already-frozen policy and write comparison artifacts."""

    policy_path = _file(policy_path, "policy")
    baseline_results_path = _file(baseline_results_path, "baseline_results")
    baseline_metadata_path = _file(baseline_metadata_path, "baseline_metadata")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_policy_sha256 = file_digest(policy_path)
    if actual_policy_sha256 != expected_policy_sha256:
        raise ValueError("frozen policy SHA-256 does not match")
    if file_digest(baseline_results_path) != BASELINE_RESULTS_SHA256:
        raise ValueError("baseline result SHA-256 does not match")

    metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("baseline metadata is not complete")
    if metadata.get("results_sha256") != BASELINE_RESULTS_SHA256:
        raise ValueError("baseline metadata result SHA-256 does not match")
    if metadata.get("manifest_sha256") != BASELINE_MANIFEST_SHA256:
        raise ValueError("baseline metadata manifest SHA-256 does not match")

    with baseline_results_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _validate_baseline_rows(rows)
    policy = load_hierarchical_policy(policy_path)

    baseline_labels: list[str | None] = []
    hierarchical_labels: list[str | None] = []
    decisions = []
    output_rows = []
    group_ids = []
    for row in rows:
        baseline_label = row["final_label"] or None
        decision = classify_hierarchical(_runs_from_row(row), policy)
        standardized = dict(decision.standardized_features)
        group_id = _group_id(row["source_dataset"], row["source_id"])
        baseline_labels.append(baseline_label)
        hierarchical_label = decision.label.value if decision.label is not None else None
        hierarchical_labels.append(hierarchical_label)
        decisions.append(decision)
        group_ids.append(group_id)
        output_rows.append(
            {
                "manifest_index": row["manifest_index"],
                "sample_id": row["sample_id"],
                "true_class": row["true_class"],
                "source_dataset": row["source_dataset"],
                "source_id": row["source_id"],
                "group_id": group_id,
                "distildire_score": row["distildire_score"],
                "dinolizer_score": row["dinolizer_score"],
                "dinolizer_marked_area": row["dinolizer_marked_area"],
                "baseline_label": baseline_label or "",
                "baseline_reason": row["decision_reason"],
                "hierarchical_label": hierarchical_label or "",
                "hierarchical_reason": decision.reason.value,
                "hierarchical_inconclusive": decision.inconclusive,
                "p_artificial": _optional_number(decision.p_artificial),
                "p_synthetic_given_artificial": _optional_number(
                    decision.p_synthetic_given_artificial
                ),
                "standardized_distildire_score": standardized.get(
                    "distildire_score", ""
                ),
                "standardized_dinolizer_score": standardized.get(
                    "dinolizer_score", ""
                ),
                "standardized_dinolizer_marked_area": standardized.get(
                    "dinolizer_marked_area", ""
                ),
                "level1_threshold": decision.level1_threshold,
                "level2_threshold": decision.level2_threshold,
                "policy_sha256": decision.policy_sha256,
            }
        )

    true_labels = [row["true_class"] for row in rows]
    baseline_metrics = multiclass_metrics(true_labels, baseline_labels)
    hierarchical_metrics = multiclass_metrics(true_labels, hierarchical_labels)
    aggregation_error_reasons = {
        "detector_error",
        "insufficient_coverage",
        "invalid_features",
    }
    aggregation_errors = sum(
        decision.reason.value in aggregation_error_reasons for decision in decisions
    )
    coverage_by_class = _coverage_by_class(true_labels, hierarchical_labels)
    stage_metrics = _stage_metrics(rows, decisions, hierarchical_labels)
    bootstrap = paired_group_bootstrap(
        true_labels=true_labels,
        baseline_predictions=baseline_labels,
        hierarchical_predictions=hierarchical_labels,
        group_ids=group_ids,
    )

    baseline_end = baseline_metrics["end_to_end"]
    hierarchical_end = hierarchical_metrics["end_to_end"]
    recalls = {
        name: values["recall"]
        for name, values in hierarchical_end["per_class"].items()
    }
    acceptance = {
        "zero_aggregation_errors": aggregation_errors == 0,
        "coverage_at_least_80_percent": hierarchical_end["coverage"] >= 0.80,
        "balanced_accuracy_at_least_70_percent": (
            hierarchical_end["balanced_accuracy"] >= 0.70
        ),
        "macro_f1_at_least_65_percent": hierarchical_end["macro_f1"] >= 0.65,
        "every_class_recall_at_least_50_percent": all(
            value >= 0.50 for value in recalls.values()
        ),
        "balanced_accuracy_above_boolean_baseline": (
            hierarchical_end["balanced_accuracy"]
            > baseline_end["balanced_accuracy"]
        ),
        "macro_f1_above_boolean_baseline": (
            hierarchical_end["macro_f1"] > baseline_end["macro_f1"]
        ),
    }
    acceptance["all_engineering_gates_passed"] = all(acceptance.values())

    predictions_path = output_dir / "hierarchical-retrospective-predictions.csv"
    outcomes_path = output_dir / "hierarchical-retrospective-outcomes.csv"
    confusion_path = output_dir / "hierarchical-retrospective-confusion.csv"
    outcomes_png = output_dir / "hierarchical-retrospective-outcomes.png"
    confusion_png = output_dir / "hierarchical-retrospective-confusion.png"
    metrics_path = output_dir / "hierarchical-retrospective-metrics.json"
    checksums_path = output_dir / "hierarchical-retrospective-artifacts.sha256"

    _write_csv(predictions_path, PREDICTION_FIELDS, output_rows)
    outcome_rows = _outcome_rows(baseline_metrics, hierarchical_metrics)
    confusion_rows = _confusion_rows(baseline_metrics, hierarchical_metrics)
    _write_csv(
        outcomes_path,
        ("policy", "true_class", *OUTCOMES),
        outcome_rows,
    )
    _write_csv(
        confusion_path,
        ("policy", "true_class", *TRUE_CLASSES),
        confusion_rows,
    )
    _draw_table_figure(
        outcomes_png,
        "Outcomes by true class",
        outcome_rows,
        OUTCOMES,
    )
    _draw_table_figure(
        confusion_png,
        "Conclusive confusion matrices",
        confusion_rows,
        TRUE_CLASSES,
    )

    metrics = {
        "schema_version": "diffdetect-hierarchical-retrospective-metrics-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "baseline_results_file": baseline_results_path.name,
            "baseline_results_sha256": BASELINE_RESULTS_SHA256,
            "baseline_metadata_sha256": file_digest(baseline_metadata_path),
            "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
            "policy_file": policy_path.name,
            "policy_sha256": actual_policy_sha256,
            "policy_calibration_previous_validation_used_for_fitting": False,
        },
        "sample_count": len(rows),
        "class_counts": dict(sorted(Counter(true_labels).items())),
        "aggregation_errors": aggregation_errors,
        "coverage_by_true_class": coverage_by_class,
        "boolean_baseline": baseline_metrics,
        "hierarchical": hierarchical_metrics,
        "stages": stage_metrics,
        "paired_group_bootstrap": bootstrap,
        "engineering_acceptance": acceptance,
    }
    _write_json(metrics_path, metrics)

    artifact_paths = (
        predictions_path,
        outcomes_path,
        confusion_path,
        outcomes_png,
        confusion_png,
        metrics_path,
    )
    temporary_checksums = checksums_path.with_name(f".{checksums_path.name}.tmp")
    with temporary_checksums.open("w", encoding="utf-8") as stream:
        stream.write(f"{actual_policy_sha256}  {policy_path.name}\n")
        stream.write(
            f"{BASELINE_RESULTS_SHA256}  {baseline_results_path.name}\n"
        )
        stream.write(
            f"{file_digest(baseline_metadata_path)}  {baseline_metadata_path.name}\n"
        )
        for path in artifact_paths:
            stream.write(f"{file_digest(path)}  {path.name}\n")
    temporary_checksums.replace(checksums_path)
    return metrics


def paired_group_bootstrap(
    *,
    true_labels: Sequence[str],
    baseline_predictions: Sequence[str | None],
    hierarchical_predictions: Sequence[str | None],
    group_ids: Sequence[str],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Bootstrap paired metric differences by source group."""

    grouped: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(index)
    groups = sorted(grouped)
    generator = random.Random(seed)
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1", "coverage")
    differences = {name: [] for name in metric_names}

    for _ in range(resamples):
        sampled_indexes = []
        for _ in groups:
            sampled_indexes.extend(grouped[generator.choice(groups)])
        sample_truth = [true_labels[index] for index in sampled_indexes]
        sample_baseline = [baseline_predictions[index] for index in sampled_indexes]
        sample_hierarchy = [
            hierarchical_predictions[index] for index in sampled_indexes
        ]
        baseline = multiclass_metrics(sample_truth, sample_baseline)["end_to_end"]
        hierarchy = multiclass_metrics(sample_truth, sample_hierarchy)["end_to_end"]
        for name in metric_names:
            differences[name].append(hierarchy[name] - baseline[name])

    baseline = multiclass_metrics(true_labels, baseline_predictions)["end_to_end"]
    hierarchy = multiclass_metrics(true_labels, hierarchical_predictions)["end_to_end"]
    return {
        "method": "paired source-group bootstrap with replacement",
        "seed": seed,
        "resamples": resamples,
        "confidence_level": 0.95,
        "group_count": len(groups),
        "differences_hierarchical_minus_boolean": {
            name: {
                "observed": hierarchy[name] - baseline[name],
                "bootstrap_mean": sum(values) / len(values),
                "lower_95": _percentile(values, 0.025),
                "upper_95": _percentile(values, 0.975),
            }
            for name, values in differences.items()
        },
    }


def _stage_metrics(rows, decisions, hierarchical_labels) -> dict[str, object]:
    level1_truth = [
        "real" if row["true_class"] == "real" else "artificial" for row in rows
    ]
    level1_predictions = []
    for decision in decisions:
        if decision.p_artificial is None:
            level1_predictions.append(None)
        elif decision.p_artificial <= 1.0 - decision.level1_threshold:
            level1_predictions.append("real")
        elif decision.p_artificial >= decision.level1_threshold:
            level1_predictions.append("artificial")
        else:
            level1_predictions.append(None)

    artificial_indexes = [
        index for index, row in enumerate(rows) if row["true_class"] != "real"
    ]
    level2_truth = [rows[index]["true_class"] for index in artificial_indexes]
    level2_end_to_end = [
        hierarchical_labels[index]
        if hierarchical_labels[index] in {"synthetic", "edited"}
        else None
        for index in artificial_indexes
    ]
    reached_level2 = [
        index
        for index in artificial_indexes
        if decisions[index].p_synthetic_given_artificial is not None
    ]
    conditional_truth = [rows[index]["true_class"] for index in reached_level2]
    conditional_predictions = [
        hierarchical_labels[index]
        if hierarchical_labels[index] in {"synthetic", "edited"}
        else None
        for index in reached_level2
    ]
    return {
        "level1_real_vs_artificial": _classification_metrics_for_classes(
            level1_truth,
            level1_predictions,
            ("real", "artificial"),
        ),
        "level2_synthetic_vs_edited_end_to_end": _classification_metrics_for_classes(
            level2_truth,
            level2_end_to_end,
            ("synthetic", "edited"),
        ),
        "level2_synthetic_vs_edited_conditional_on_reaching_level2": (
            _classification_metrics_for_classes(
                conditional_truth,
                conditional_predictions,
                ("synthetic", "edited"),
            )
        ),
    }


def _classification_metrics_for_classes(truth, predictions, classes):
    total = len(truth)
    conclusive = sum(prediction is not None for prediction in predictions)
    correct = sum(left == right for left, right in zip(truth, predictions))
    per_class = {}
    recalls = []
    f1s = []
    for target in classes:
        true_positive = sum(
            left == target and right == target
            for left, right in zip(truth, predictions)
        )
        actual = sum(left == target for left in truth)
        predicted = sum(right == target for right in predictions)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[target] = {"precision": precision, "recall": recall, "f1": f1}
        recalls.append(recall)
        f1s.append(f1)
    return {
        "sample_count": total,
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": sum(recalls) / len(classes),
        "macro_f1": sum(f1s) / len(classes),
        "coverage": conclusive / total if total else 0.0,
        "per_class": per_class,
    }


def _runs_from_row(row: Mapping[str, str]) -> tuple[DetectorRun, DetectorRun]:
    distildire = DetectorRun(
        name="distildire",
        target=DetectionTarget.SYNTHETIC,
        duration_seconds=float(row["distildire_duration_seconds"]),
        result=DetectionResult(
            detected=_boolean(row["distildire_detected"]),
            score=_bounded_float(row["distildire_score"]),
            threshold=_bounded_float(row["distildire_threshold"]),
        ),
    )
    dinolizer = DetectorRun(
        name="dinolizer",
        target=DetectionTarget.EDITED,
        duration_seconds=float(row["dinolizer_duration_seconds"]),
        result=DetectionResult(
            detected=_boolean(row["dinolizer_detected"]),
            score=_bounded_float(row["dinolizer_score"]),
            threshold=_bounded_float(row["dinolizer_threshold"]),
            metadata={"marked_area": _bounded_float(row["dinolizer_marked_area"])},
        ),
    )
    return distildire, dinolizer


def _validate_baseline_rows(rows: Sequence[Mapping[str, str]]) -> None:
    if len(rows) != 768:
        raise ValueError("baseline must contain 768 rows")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("baseline sample identifiers must be unique")
    if Counter(row["true_class"] for row in rows) != {
        "real": 256,
        "synthetic": 256,
        "edited": 256,
    }:
        raise ValueError("baseline class counts do not match")
    for row in rows:
        if row["manifest_sha256"] != BASELINE_MANIFEST_SHA256:
            raise ValueError("baseline row manifest SHA-256 does not match")
        if row["distildire_error"].strip() or row["dinolizer_error"].strip():
            raise ValueError("baseline contains a detector error")
        if row["final_label"] and row["final_label"] not in TRUE_CLASSES:
            raise ValueError("baseline contains an invalid final label")
        for name in (
            "distildire_score",
            "dinolizer_score",
            "dinolizer_marked_area",
        ):
            _bounded_float(row[name])


def _coverage_by_class(truth, predictions):
    result = {}
    for target in TRUE_CLASSES:
        indexes = [index for index, label in enumerate(truth) if label == target]
        conclusive = sum(predictions[index] is not None for index in indexes)
        result[target] = {
            "support": len(indexes),
            "conclusive": conclusive,
            "coverage": conclusive / len(indexes),
            "abstention_rate": 1.0 - conclusive / len(indexes),
        }
    return result


def _outcome_rows(baseline, hierarchical):
    rows = []
    for name, metrics in (("boolean", baseline), ("hierarchical", hierarchical)):
        for truth in TRUE_CLASSES:
            rows.append(
                {
                    "policy": name,
                    "true_class": truth,
                    **metrics["outcome_table"][truth],
                }
            )
    return rows


def _confusion_rows(baseline, hierarchical):
    rows = []
    for name, metrics in (("boolean", baseline), ("hierarchical", hierarchical)):
        for truth in TRUE_CLASSES:
            rows.append(
                {
                    "policy": name,
                    "true_class": truth,
                    **metrics["conclusive_confusion_matrix"][truth],
                }
            )
    return rows


def _draw_table_figure(path, title, rows, value_fields):
    cell_width = 118
    cell_height = 42
    label_width = 190
    panel_gap = 30
    policies = ("boolean", "hierarchical")
    width = label_width + len(value_fields) * cell_width + 30
    panel_height = 55 + (len(TRUE_CLASSES) + 1) * cell_height
    height = 55 + len(policies) * panel_height + panel_gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((15, 15), title, fill="black")
    y = 50
    for policy in policies:
        draw.text((15, y), policy, fill="black")
        y += 28
        for column, field in enumerate(value_fields):
            draw.text(
                (label_width + column * cell_width + 6, y + 10),
                field,
                fill="black",
            )
        y += cell_height
        policy_rows = [row for row in rows if row["policy"] == policy]
        maximum = max(
            1,
            max(int(row[field]) for row in policy_rows for field in value_fields),
        )
        for row in policy_rows:
            draw.text((15, y + 12), row["true_class"], fill="black")
            for column, field in enumerate(value_fields):
                value = int(row[field])
                intensity = int(245 - 165 * value / maximum)
                left = label_width + column * cell_width
                draw.rectangle(
                    (left, y, left + cell_width - 3, y + cell_height - 3),
                    fill=(intensity, intensity, 255),
                    outline=(100, 100, 140),
                )
                draw.text((left + 48, y + 12), str(value), fill="black")
            y += cell_height
        y += panel_gap
    image.save(path)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _group_id(source_dataset: str, source_id: str) -> str:
    if source_dataset == "nebula/CocoGlide":
        prefix = "cocoglide"
    elif source_dataset == "poloclub/diffusiondb":
        prefix = "diffusiondb"
    else:
        raise ValueError(f"unexpected source dataset: {source_dataset}")
    return f"{prefix}:{source_id}"


def _bounded_float(value: object) -> float:
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError("feature must be finite and between 0 and 1")
    return converted


def _boolean(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _optional_number(value: float | None) -> float | str:
    return "" if value is None else value


def _file(path: str | Path, name: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is not a file: {resolved}")
    return resolved.resolve()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str | Path, value: Mapping[str, object]) -> None:
    output = Path(path)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(output)


def _write_csv(path, fields, rows):
    output = Path(path)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
