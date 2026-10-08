"""Freeze and evaluate an independent hierarchical confirmation corpus.

This module deliberately exposes no fitting or threshold-selection path.  It
validates a prespecified external catalog, runs the already-pinned detectors,
and applies the already-frozen hierarchical policy without modification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import perf_counter

from PIL import Image, ImageDraw

from diffdetect import (
    DinoLizerDetector,
    DistilDIREDetector,
    DetectorRun,
    DetectorRunner,
    classify_hierarchical,
    load_hierarchical_policy,
)

if __package__:
    from experiments.hierarchical_aggregation import (
        DINOLIZER_THRESHOLD,
        DISTILDIRE_THRESHOLD,
        RESULT_FIELDS,
        _gpu_environment,
        _reset_gpu_peak_memory,
        _software_environment,
        _utc_now,
        _verify_model_artifacts,
        _write_json,
        calibration_result_row,
        file_digest,
    )
    from experiments.hierarchical_retrospective import (
        _coverage_by_class,
        _runs_from_row,
        _stage_metrics,
    )
    from experiments.hierarchical_training import TRUE_CLASSES, multiclass_metrics
else:
    from hierarchical_aggregation import (  # type: ignore[no-redef]
        DINOLIZER_THRESHOLD,
        DISTILDIRE_THRESHOLD,
        RESULT_FIELDS,
        _gpu_environment,
        _reset_gpu_peak_memory,
        _software_environment,
        _utc_now,
        _verify_model_artifacts,
        _write_json,
        calibration_result_row,
        file_digest,
    )
    from hierarchical_retrospective import (  # type: ignore[no-redef]
        _coverage_by_class,
        _runs_from_row,
        _stage_metrics,
    )
    from hierarchical_training import TRUE_CLASSES, multiclass_metrics


PROTOCOL_VERSION = "diffdetect-hierarchical-external-confirmation-v1"
SAMPLES_PER_CLASS = 128
EXPECTED_POLICY_SHA256 = (
    "c1214e33d698daa7cda082cd2c20ef0e4553d8d63b46ac77494e9685a50a1187"
)
BOOTSTRAP_SEED = 20261008
BOOTSTRAP_RESAMPLES = 2000
MIN_DATASETS_PER_CLASS = 2
MAX_DATASET_SHARE_PER_CLASS = 0.50
MIN_SYNTHETIC_METHODS = 4
MIN_EDITING_METHODS = 2
MAX_SYNTHETIC_METHOD_SHARE = 0.25
MAX_EDITING_METHOD_SHARE = 0.50

CANDIDATE_FIELDS = (
    "sample_id",
    "true_class",
    "source_dataset",
    "source_revision",
    "source_partition",
    "source_id",
    "origin_dataset",
    "origin_id",
    "origin_relative_path",
    "origin_input_sha256",
    "group_id",
    "source_root",
    "relative_path",
    "source_metadata_sha256",
    "input_sha256",
    "width",
    "height",
    "generation_method",
    "editing_method",
)

MANIFEST_FIELDS = (
    "protocol_version",
    "manifest_index",
    *CANDIDATE_FIELDS,
    "exclusion_manifest_sha256s",
)

PREDICTION_FIELDS = (
    "manifest_index",
    "sample_id",
    "true_class",
    "source_dataset",
    "source_id",
    "group_id",
    "generation_method",
    "editing_method",
    "distildire_score",
    "dinolizer_score",
    "dinolizer_marked_area",
    "hierarchical_label",
    "hierarchical_reason",
    "hierarchical_inconclusive",
    "p_artificial",
    "p_synthetic_given_artificial",
    "level1_threshold",
    "level2_threshold",
    "policy_sha256",
)


@dataclass(frozen=True)
class ExternalManifestRecord:
    protocol_version: str
    manifest_index: int
    sample_id: str
    true_class: str
    source_dataset: str
    source_revision: str
    source_partition: str
    source_id: str
    origin_dataset: str
    origin_id: str
    origin_relative_path: str
    origin_input_sha256: str
    group_id: str
    source_root: str
    relative_path: str
    source_metadata_sha256: str
    input_sha256: str
    width: int
    height: int
    generation_method: str
    editing_method: str
    exclusion_manifest_sha256s: str

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "ExternalManifestRecord":
        if set(row) != set(MANIFEST_FIELDS):
            raise ValueError("external manifest columns do not match the frozen schema")
        try:
            record = cls(
                protocol_version=row["protocol_version"],
                manifest_index=int(row["manifest_index"]),
                sample_id=row["sample_id"],
                true_class=row["true_class"],
                source_dataset=row["source_dataset"],
                source_revision=row["source_revision"],
                source_partition=row["source_partition"],
                source_id=row["source_id"],
                origin_dataset=row["origin_dataset"],
                origin_id=row["origin_id"],
                origin_relative_path=row["origin_relative_path"],
                origin_input_sha256=row["origin_input_sha256"],
                group_id=row["group_id"],
                source_root=row["source_root"],
                relative_path=row["relative_path"],
                source_metadata_sha256=row["source_metadata_sha256"],
                input_sha256=row["input_sha256"],
                width=int(row["width"]),
                height=int(row["height"]),
                generation_method=row["generation_method"],
                editing_method=row["editing_method"],
                exclusion_manifest_sha256s=row["exclusion_manifest_sha256s"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("external manifest contains an invalid field") from exc
        record.validate()
        return record

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unexpected protocol version: {self.protocol_version}")
        if self.manifest_index <= 0:
            raise ValueError("manifest_index must be positive")
        required = (
            self.sample_id,
            self.source_dataset,
            self.source_revision,
            self.source_partition,
            self.source_id,
            self.group_id,
            self.source_root,
        )
        if any(not value.strip() for value in required):
            raise ValueError("external source identity fields must not be empty")
        if self.true_class not in TRUE_CLASSES:
            raise ValueError(f"invalid true class: {self.true_class}")
        _validate_relative_path(self.relative_path)
        _validate_digest(self.source_metadata_sha256, "source metadata SHA-256")
        _validate_digest(self.input_sha256, "input SHA-256")
        exclusion_hashes = self.exclusion_manifest_sha256s.split(";")
        if len(exclusion_hashes) < 2 or len(exclusion_hashes) != len(set(exclusion_hashes)):
            raise ValueError("at least two unique exclusion manifest hashes are required")
        for digest in exclusion_hashes:
            _validate_digest(digest, "exclusion manifest SHA-256")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("manifest dimensions must be positive")
        if self.true_class == "real":
            if (
                self.generation_method
                or self.editing_method
                or self.origin_dataset
                or self.origin_id
                or self.origin_relative_path
                or self.origin_input_sha256
            ):
                raise ValueError("real rows must not declare generation, editing, or origin fields")
        elif self.true_class == "synthetic":
            if (
                not self.generation_method
                or self.editing_method
                or self.origin_dataset
                or self.origin_id
                or self.origin_relative_path
                or self.origin_input_sha256
            ):
                raise ValueError("synthetic rows require only generation_method")
        elif not all(
            (
                self.editing_method,
                self.origin_dataset,
                self.origin_id,
                self.origin_relative_path,
                self.origin_input_sha256,
            )
        ):
            raise ValueError("edited rows require editing_method and complete origin identity")
        if self.origin_input_sha256:
            _validate_relative_path(self.origin_relative_path)
            _validate_digest(self.origin_input_sha256, "origin input SHA-256")

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExcludedIdentities:
    manifest_sha256s: tuple[str, ...]
    sample_ids: frozenset[str]
    source_ids: frozenset[tuple[str, str]]
    input_sha256s: frozenset[str]


def freeze_manifest(
    *,
    catalog_path: str | Path,
    roots: Mapping[str, str | Path],
    exclusion_manifests: Sequence[str | Path],
) -> tuple[ExternalManifestRecord, ...]:
    """Validate a fully curated catalog and return deterministically ordered rows."""

    catalog_path = _file(catalog_path, "catalog")
    excluded = load_exclusions(exclusion_manifests)
    with catalog_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CANDIDATE_FIELDS):
            raise ValueError("candidate catalog columns do not match the frozen schema")
        rows = list(reader)

    exclusion_value = ";".join(excluded.manifest_sha256s)
    records = []
    for row in rows:
        try:
            record = ExternalManifestRecord(
                protocol_version=PROTOCOL_VERSION,
                manifest_index=1,
                sample_id=row["sample_id"],
                true_class=row["true_class"],
                source_dataset=row["source_dataset"],
                source_revision=row["source_revision"],
                source_partition=row["source_partition"],
                source_id=row["source_id"],
                origin_dataset=row["origin_dataset"],
                origin_id=row["origin_id"],
                origin_relative_path=row["origin_relative_path"],
                origin_input_sha256=row["origin_input_sha256"],
                group_id=row["group_id"],
                source_root=row["source_root"],
                relative_path=row["relative_path"],
                source_metadata_sha256=row["source_metadata_sha256"],
                input_sha256=row["input_sha256"],
                width=int(row["width"]),
                height=int(row["height"]),
                generation_method=row["generation_method"],
                editing_method=row["editing_method"],
                exclusion_manifest_sha256s=exclusion_value,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("candidate catalog contains an invalid field") from exc
        record.validate()
        _assert_not_excluded(record, excluded)
        _verify_record_file(record, roots)
        records.append(record)

    _validate_collection(records)
    records.sort(key=lambda record: _selection_digest(record.sample_id))
    return tuple(
        ExternalManifestRecord(**{**record.to_row(), "manifest_index": index})
        for index, record in enumerate(records, start=1)
    )


def load_exclusions(paths: Sequence[str | Path]) -> ExcludedIdentities:
    if len(paths) < 2:
        raise ValueError("baseline and calibration exclusion manifests are required")
    sample_ids: set[str] = set()
    source_ids: set[tuple[str, str]] = set()
    input_sha256s: set[str] = set()
    hashes = []
    required = {"sample_id", "source_dataset", "source_id", "input_sha256"}
    for value in paths:
        path = _file(value, "exclusion manifest")
        digest = file_digest(path)
        if digest in hashes:
            raise ValueError("duplicate exclusion manifest")
        hashes.append(digest)
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("exclusion manifest does not contain identity fields")
            for row in reader:
                sample_ids.add(row["sample_id"])
                source_ids.update(
                    _canonical_source_identities(row["source_dataset"], row["source_id"])
                )
                _validate_digest(row["input_sha256"], "excluded input SHA-256")
                input_sha256s.add(row["input_sha256"])
    return ExcludedIdentities(
        manifest_sha256s=tuple(sorted(hashes)),
        sample_ids=frozenset(sample_ids),
        source_ids=frozenset(source_ids),
        input_sha256s=frozenset(input_sha256s),
    )


def write_manifest(records: Sequence[ExternalManifestRecord], path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            record.validate()
            writer.writerow(record.to_row())
    temporary.replace(output)
    return file_digest(output)


def load_manifest(path: str | Path) -> tuple[ExternalManifestRecord, ...]:
    path = _file(path, "manifest")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(MANIFEST_FIELDS):
            raise ValueError("external manifest columns do not match the frozen schema")
        records = tuple(ExternalManifestRecord.from_row(row) for row in reader)
    _validate_collection(records)
    if tuple(record.manifest_index for record in records) != tuple(
        range(1, len(records) + 1)
    ):
        raise ValueError("manifest indexes must be consecutive and ordered")
    return records


def parse_roots(values: Sequence[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in roots:
            raise ValueError("roots must use unique NAME=/absolute/path values")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"root is not a directory: {path}")
        roots[name] = path
    if not roots:
        raise ValueError("at least one root is required")
    return roots


def verify_inputs(
    records: Iterable[ExternalManifestRecord], roots: Mapping[str, str | Path]
) -> dict[str, Path]:
    return {record.sample_id: _verify_record_file(record, roots) for record in records}


def select_smoke_records(
    records: Sequence[ExternalManifestRecord],
) -> tuple[ExternalManifestRecord, ...]:
    selected = {}
    for record in records:
        selected.setdefault(record.true_class, record)
    if set(selected) != set(TRUE_CLASSES):
        raise ValueError("smoke selection requires all three classes")
    return tuple(sorted(selected.values(), key=lambda record: record.manifest_index))


def run_external(args: argparse.Namespace) -> None:
    manifest_path = _file(args.manifest, "manifest")
    records = load_manifest(manifest_path)
    roots = parse_roots(args.root)
    resolved = verify_inputs(records, roots)
    selected = select_smoke_records(records) if args.smoke else records
    manifest_sha256 = file_digest(manifest_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _existing_sample_ids(output, manifest_sha256) if args.resume else set()
    if output.exists() and not args.resume:
        raise FileExistsError(f"output already exists; use --resume: {output}")
    pending = tuple(record for record in selected if record.sample_id not in completed)

    artifact_metadata = _verify_model_artifacts(args)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "output": str(output),
        "environment_id": args.environment_id,
        "smoke": bool(args.smoke),
        "resume": bool(args.resume),
        "selected_samples": len(selected),
        "already_completed": len(completed),
        "status": "running",
        "started_utc": _utc_now(),
        "software": _software_environment(),
        "artifacts": artifact_metadata,
        "model_order": ["distildire", "dinolizer"],
        "aggregation_policy_applied": False,
    }
    metadata_output = output.with_suffix(output.suffix + ".metadata.json")
    _write_json(metadata_output, metadata)

    if not pending:
        metadata.update(
            status="complete",
            completed_samples=len(completed),
            finished_utc=_utc_now(),
            gpu=_gpu_environment(),
            results_sha256=file_digest(output),
        )
        _write_json(metadata_output, metadata)
        return

    _reset_gpu_peak_memory()
    runner = DetectorRunner(
        [
            DistilDIREDetector(
                repository_path=args.distildire_repository,
                classifier_weights=args.distildire_classifier,
                adm_weights=args.distildire_adm,
                threshold=DISTILDIRE_THRESHOLD,
                device=args.device,
            ),
            DinoLizerDetector(
                repository_path=args.dinolizer_repository,
                checkpoint_path=args.dinolizer_checkpoint,
                threshold=DINOLIZER_THRESHOLD,
                device=args.device,
            ),
        ]
    )
    append = output.exists()
    processed = 0
    try:
        with output.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, lineterminator="\n")
            if not append:
                writer.writeheader()
            for record in pending:
                started_utc = _utc_now()
                started = perf_counter()
                runs = runner.run(resolved[record.sample_id])
                row = calibration_result_row(
                    record=record,  # compatible evidence fields
                    runs=runs,
                    manifest_sha256=manifest_sha256,
                    environment_id=args.environment_id,
                    attempt=1,
                    run_started_utc=started_utc,
                    total_duration_seconds=perf_counter() - started,
                )
                row["protocol_version"] = PROTOCOL_VERSION
                writer.writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
                processed += 1
                metadata.update(
                    processed_this_execution=processed,
                    last_sample_id=record.sample_id,
                )
                _write_json(metadata_output, metadata)
    except BaseException:
        metadata.update(
            status="interrupted",
            processed_this_execution=processed,
            finished_utc=_utc_now(),
            gpu=_gpu_environment(),
        )
        _write_json(metadata_output, metadata)
        raise

    metadata.update(
        status="complete",
        processed_this_execution=processed,
        completed_samples=len(completed) + processed,
        finished_utc=_utc_now(),
        gpu=_gpu_environment(),
        results_sha256=file_digest(output),
    )
    _write_json(metadata_output, metadata)


def evaluate_external(
    *,
    policy_path: str | Path,
    expected_policy_sha256: str,
    manifest_path: str | Path,
    results_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Apply the frozen hierarchy to external evidence without fitting."""

    if expected_policy_sha256 != EXPECTED_POLICY_SHA256:
        raise ValueError("policy SHA-256 is not the preregistered frozen policy")
    policy_path = _file(policy_path, "policy")
    manifest_path = _file(manifest_path, "manifest")
    results_path = _file(results_path, "results")
    metadata_path = _file(metadata_path, "metadata")
    if file_digest(policy_path) != expected_policy_sha256:
        raise ValueError("frozen policy SHA-256 does not match")
    manifest_sha256 = file_digest(manifest_path)
    manifest = load_manifest(manifest_path)
    by_sample = {record.sample_id: record for record in manifest}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("external inference metadata is not complete")
    if metadata.get("manifest_sha256") != manifest_sha256:
        raise ValueError("external metadata manifest SHA-256 does not match")
    results_sha256 = file_digest(results_path)
    if metadata.get("results_sha256") != results_sha256:
        raise ValueError("external result SHA-256 does not match metadata")

    with results_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(RESULT_FIELDS):
            raise ValueError("external result columns do not match the evidence schema")
        rows = list(reader)
    _validate_result_rows(rows, by_sample, manifest_sha256)
    rows.sort(key=lambda row: by_sample[row["sample_id"]].manifest_index)
    policy = load_hierarchical_policy(policy_path)
    predictions = []
    decisions = []
    prediction_rows = []
    for row in rows:
        record = by_sample[row["sample_id"]]
        decision = classify_hierarchical(_runs_from_row(row), policy)
        label = decision.label.value if decision.label is not None else None
        decisions.append(decision)
        predictions.append(label)
        prediction_rows.append(
            {
                "manifest_index": record.manifest_index,
                "sample_id": record.sample_id,
                "true_class": record.true_class,
                "source_dataset": record.source_dataset,
                "source_id": record.source_id,
                "group_id": record.group_id,
                "generation_method": record.generation_method,
                "editing_method": record.editing_method,
                "distildire_score": row["distildire_score"],
                "dinolizer_score": row["dinolizer_score"],
                "dinolizer_marked_area": row["dinolizer_marked_area"],
                "hierarchical_label": label or "",
                "hierarchical_reason": decision.reason.value,
                "hierarchical_inconclusive": decision.inconclusive,
                "p_artificial": _optional(decision.p_artificial),
                "p_synthetic_given_artificial": _optional(
                    decision.p_synthetic_given_artificial
                ),
                "level1_threshold": decision.level1_threshold,
                "level2_threshold": decision.level2_threshold,
                "policy_sha256": decision.policy_sha256,
            }
        )

    truth = [record.true_class for record in manifest]
    groups = [record.group_id for record in manifest]
    metrics = multiclass_metrics(truth, predictions)
    end_to_end = metrics["end_to_end"]
    errors = sum(
        decision.reason.value in {"detector_error", "insufficient_coverage", "invalid_features"}
        for decision in decisions
    )
    acceptance = {
        "zero_aggregation_errors": errors == 0,
        "coverage_at_least_80_percent": end_to_end["coverage"] >= 0.80,
        "balanced_accuracy_at_least_70_percent": end_to_end["balanced_accuracy"] >= 0.70,
        "macro_f1_at_least_65_percent": end_to_end["macro_f1"] >= 0.65,
        "every_class_recall_at_least_50_percent": all(
            values["recall"] >= 0.50
            for values in end_to_end["per_class"].values()
        ),
    }
    acceptance["all_confirmation_gates_passed"] = all(acceptance.values())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "hierarchical-external-predictions.csv"
    confusion_path = output_dir / "hierarchical-external-confusion.csv"
    confusion_png = output_dir / "hierarchical-external-confusion.png"
    metrics_path = output_dir / "hierarchical-external-metrics.json"
    checksums_path = output_dir / "hierarchical-external-artifacts.sha256"
    _write_csv(predictions_path, PREDICTION_FIELDS, prediction_rows)
    confusion_rows = [
        {"true_class": name, **metrics["outcome_table"][name]}
        for name in TRUE_CLASSES
    ]
    _write_csv(confusion_path, ("true_class", *TRUE_CLASSES, "inconclusive"), confusion_rows)
    _draw_confusion(confusion_png, confusion_rows)
    report = {
        "schema_version": "diffdetect-hierarchical-external-metrics-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "protocol_version": PROTOCOL_VERSION,
            "manifest_sha256": manifest_sha256,
            "results_sha256": results_sha256,
            "metadata_sha256": file_digest(metadata_path),
            "policy_sha256": expected_policy_sha256,
            "fitting_performed": False,
            "threshold_selection_performed": False,
        },
        "sample_count": len(manifest),
        "class_counts": dict(sorted(Counter(truth).items())),
        "aggregation_errors": errors,
        "coverage_by_true_class": _coverage_by_class(truth, predictions),
        "hierarchical": metrics,
        "stages": _stage_metrics(rows, decisions, predictions),
        "group_bootstrap": group_bootstrap(
            truth=truth,
            predictions=predictions,
            group_ids=groups,
            resamples=bootstrap_resamples,
        ),
        "confirmation_acceptance": acceptance,
    }
    _write_json(metrics_path, report)
    artifacts = (predictions_path, confusion_path, confusion_png, metrics_path)
    with checksums_path.open("w", encoding="utf-8") as stream:
        stream.write(f"{expected_policy_sha256}  {policy_path.name}\n")
        stream.write(f"{manifest_sha256}  {manifest_path.name}\n")
        stream.write(f"{results_sha256}  {results_path.name}\n")
        stream.write(f"{file_digest(metadata_path)}  {metadata_path.name}\n")
        for path in artifacts:
            stream.write(f"{file_digest(path)}  {path.name}\n")
    return report


def group_bootstrap(
    *,
    truth: Sequence[str],
    predictions: Sequence[str | None],
    group_ids: Sequence[str],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    grouped: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(index)
    groups = sorted(grouped)
    generator = random.Random(seed)
    names = ("accuracy", "balanced_accuracy", "macro_f1", "coverage")
    samples = {name: [] for name in names}
    for _ in range(resamples):
        indexes = []
        for _ in groups:
            indexes.extend(grouped[generator.choice(groups)])
        measured = multiclass_metrics(
            [truth[index] for index in indexes],
            [predictions[index] for index in indexes],
        )["end_to_end"]
        for name in names:
            samples[name].append(measured[name])
    observed = multiclass_metrics(truth, predictions)["end_to_end"]
    return {
        "method": "source-group bootstrap with replacement",
        "seed": seed,
        "resamples": resamples,
        "confidence_level": 0.95,
        "group_count": len(groups),
        "metrics": {
            name: {
                "observed": observed[name],
                "bootstrap_mean": sum(values) / len(values),
                "lower_95": _percentile(values, 0.025),
                "upper_95": _percentile(values, 0.975),
            }
            for name, values in samples.items()
        },
    }


def _validate_collection(records: Sequence[ExternalManifestRecord]) -> None:
    expected_total = SAMPLES_PER_CLASS * len(TRUE_CLASSES)
    if len(records) != expected_total:
        raise ValueError(f"expected {expected_total} external rows, found {len(records)}")
    counts = Counter(record.true_class for record in records)
    expected = {name: SAMPLES_PER_CLASS for name in TRUE_CLASSES}
    if counts != expected:
        raise ValueError(f"external manifest is not balanced: {dict(counts)}")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("external sample identifiers must be unique")
    if len({record.input_sha256 for record in records}) != len(records):
        raise ValueError("external input hashes must be unique")
    exclusion_sets = {record.exclusion_manifest_sha256s for record in records}
    if len(exclusion_sets) != 1:
        raise ValueError("external rows must reference the same exclusion manifests")
    for true_class in TRUE_CLASSES:
        class_rows = [record for record in records if record.true_class == true_class]
        datasets = Counter(record.source_dataset for record in class_rows)
        if len(datasets) < MIN_DATASETS_PER_CLASS:
            raise ValueError(f"{true_class} requires at least two source datasets")
        if max(datasets.values()) > SAMPLES_PER_CLASS * MAX_DATASET_SHARE_PER_CLASS:
            raise ValueError(f"one {true_class} dataset exceeds the 50% quota")
    synthetic_methods = Counter(
        record.generation_method for record in records if record.true_class == "synthetic"
    )
    if len(synthetic_methods) < MIN_SYNTHETIC_METHODS:
        raise ValueError("synthetic class requires at least four generation methods")
    if max(synthetic_methods.values()) > SAMPLES_PER_CLASS * MAX_SYNTHETIC_METHOD_SHARE:
        raise ValueError("one generation method exceeds the 25% quota")
    editing_methods = Counter(
        record.editing_method for record in records if record.true_class == "edited"
    )
    if len(editing_methods) < MIN_EDITING_METHODS:
        raise ValueError("edited class requires at least two editing methods")
    if max(editing_methods.values()) > SAMPLES_PER_CLASS * MAX_EDITING_METHOD_SHARE:
        raise ValueError("one editing method exceeds the 50% quota")


def _validate_result_rows(rows, records, manifest_sha256):
    if len(rows) != len(records) or {row["sample_id"] for row in rows} != set(records):
        raise ValueError("external results do not cover the frozen manifest exactly")
    for row in rows:
        record = records[row["sample_id"]]
        if row["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("external result protocol version does not match")
        if row["manifest_sha256"] != manifest_sha256:
            raise ValueError("external result manifest SHA-256 does not match")
        if row["true_class"] != record.true_class:
            raise ValueError("external result true class does not match manifest")
        if int(row["manifest_index"]) != record.manifest_index:
            raise ValueError("external result manifest index does not match manifest")
        if row["distildire_error"].strip() or row["dinolizer_error"].strip():
            raise ValueError("external results contain a detector error")
        for name in ("distildire_score", "dinolizer_score", "dinolizer_marked_area"):
            _bounded_float(row[name])


def _assert_not_excluded(record, excluded):
    if record.sample_id in excluded.sample_ids:
        raise ValueError(f"sample overlaps an earlier manifest: {record.sample_id}")
    record_identities = _canonical_source_identities(
        record.source_dataset, record.source_id
    )
    if record.origin_dataset:
        record_identities.update(
            _canonical_source_identities(record.origin_dataset, record.origin_id)
        )
    if record_identities & excluded.source_ids:
        raise ValueError(f"source overlaps an earlier manifest: {record.sample_id}")
    if record.input_sha256 in excluded.input_sha256s:
        raise ValueError(f"input overlaps an earlier manifest: {record.sample_id}")
    if record.origin_input_sha256 in excluded.input_sha256s:
        raise ValueError(f"origin overlaps an earlier manifest: {record.sample_id}")


def _canonical_source_identities(dataset, source_id):
    dataset_key = dataset.strip().casefold()
    source_key = source_id.strip().casefold()
    identities = {(dataset_key, source_key)}
    if dataset_key in {"nebula/cocoglide", "coco", "mscoco", "ms-coco"}:
        numeric_id = source_key.removeprefix("coco_")
        if numeric_id.isdigit():
            numeric_id = numeric_id.zfill(12)
        identities.add(("coco", numeric_id))
    if dataset_key in {"poloclub/diffusiondb", "diffusiondb"}:
        identities.add(("diffusiondb", source_key))
    return identities


def _verify_record_file(record, roots):
    if record.source_root not in roots:
        raise KeyError(f"missing root for {record.source_root}")
    root = Path(roots[record.source_root]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source root is not a directory: {root}")
    relative = PurePosixPath(record.relative_path)
    path = root.joinpath(*relative.parts).resolve()
    if root == path or root not in path.parents or not path.is_file():
        raise ValueError(f"manifest path is invalid: {record.relative_path}")
    if file_digest(path) != record.input_sha256:
        raise ValueError(f"input SHA-256 does not match for {record.sample_id}")
    try:
        with Image.open(path) as image:
            size = image.size
            image.verify()
    except Exception as exc:
        raise ValueError(f"invalid image for {record.sample_id}") from exc
    if size != (record.width, record.height):
        raise ValueError(f"input dimensions do not match for {record.sample_id}")
    if record.origin_relative_path:
        origin_relative = PurePosixPath(record.origin_relative_path)
        origin_path = root.joinpath(*origin_relative.parts).resolve()
        if root == origin_path or root not in origin_path.parents or not origin_path.is_file():
            raise ValueError(f"origin path is invalid: {record.origin_relative_path}")
        if file_digest(origin_path) != record.origin_input_sha256:
            raise ValueError(f"origin SHA-256 does not match for {record.sample_id}")
    return path


def _existing_sample_ids(output, manifest_sha256):
    if not output.exists():
        return set()
    with output.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(RESULT_FIELDS):
            raise ValueError("existing result columns do not match the evidence schema")
        rows = list(reader)
    if any(row["manifest_sha256"] != manifest_sha256 for row in rows):
        raise ValueError("existing results belong to another manifest")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("existing results contain duplicate sample identifiers")
    return set(sample_ids)


def _draw_confusion(path, rows):
    fields = (*TRUE_CLASSES, "inconclusive")
    cell_width, cell_height, label_width = 125, 44, 155
    width = label_width + len(fields) * cell_width + 20
    height = 75 + (len(TRUE_CLASSES) + 1) * cell_height
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), "External confirmation outcomes", fill="black")
    y = 48
    for column, field in enumerate(fields):
        draw.text((label_width + column * cell_width + 5, y), field, fill="black")
    y += cell_height
    maximum = max(1, max(int(row[field]) for row in rows for field in fields))
    for row in rows:
        draw.text((12, y + 12), row["true_class"], fill="black")
        for column, field in enumerate(fields):
            value = int(row[field])
            intensity = int(245 - 165 * value / maximum)
            left = label_width + column * cell_width
            draw.rectangle(
                (left, y, left + cell_width - 3, y + cell_height - 3),
                fill=(intensity, intensity, 255),
                outline=(100, 100, 140),
            )
            draw.text((left + 52, y + 12), str(value), fill="black")
        y += cell_height
    image.save(path)


def _prepare_command(args):
    records = freeze_manifest(
        catalog_path=args.catalog,
        roots=parse_roots(args.root),
        exclusion_manifests=args.exclude_manifest,
    )
    digest = write_manifest(records, args.output)
    print(f"wrote {len(records)} records to {args.output}")
    print(f"class counts: {dict(sorted(Counter(r.true_class for r in records).items()))}")
    print(f"manifest SHA-256: {digest}")


def _evaluate_command(args):
    outputs = (
        "hierarchical-external-predictions.csv",
        "hierarchical-external-confusion.csv",
        "hierarchical-external-confusion.png",
        "hierarchical-external-metrics.json",
        "hierarchical-external-artifacts.sha256",
    )
    existing = [
        Path(args.output_dir) / name
        for name in outputs
        if (Path(args.output_dir) / name).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError("external artifacts already exist: " + ", ".join(map(str, existing)))
    report = evaluate_external(
        policy_path=args.policy,
        expected_policy_sha256=args.policy_sha256,
        manifest_path=args.manifest,
        results_path=args.results,
        metadata_path=args.results_metadata,
        output_dir=args.output_dir,
    )
    end = report["hierarchical"]["end_to_end"]
    print(f"coverage: {end['coverage']:.6f}")
    print(f"balanced accuracy: {end['balanced_accuracy']:.6f}")
    print(f"macro F1: {end['macro_f1']:.6f}")
    print(
        "all confirmation gates passed: "
        f"{report['confirmation_acceptance']['all_confirmation_gates_passed']}"
    )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-manifest")
    prepare.add_argument("--catalog", required=True, type=Path)
    prepare.add_argument("--root", required=True, action="append")
    prepare.add_argument("--exclude-manifest", required=True, action="append", type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.set_defaults(handler=_prepare_command)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--root", required=True, action="append")
    run.add_argument("--distildire-repository", required=True, type=Path)
    run.add_argument("--distildire-classifier", required=True, type=Path)
    run.add_argument("--distildire-adm", required=True, type=Path)
    run.add_argument("--dinolizer-repository", required=True, type=Path)
    run.add_argument("--dinolizer-checkpoint", required=True, type=Path)
    run.add_argument("--device", default="cuda")
    run.add_argument("--environment-id", required=True)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.set_defaults(handler=run_external)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--policy", required=True, type=Path)
    evaluate.add_argument("--policy-sha256", required=True)
    evaluate.add_argument("--manifest", required=True, type=Path)
    evaluate.add_argument("--results", required=True, type=Path)
    evaluate.add_argument("--results-metadata", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.set_defaults(handler=_evaluate_command)
    return parser


def _selection_digest(value):
    return hashlib.sha256(f"{PROTOCOL_VERSION}:{value}".encode()).hexdigest()


def _validate_relative_path(value):
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.as_posix() != value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ValueError(f"path must be a normalized relative path: {value}")


def _validate_digest(value, name):
    if len(value) != 64:
        raise ValueError(f"{name} must contain 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must contain 64 hexadecimal characters") from exc


def _bounded_float(value):
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("feature must be finite and between 0 and 1")
    return number


def _percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _optional(value):
    return "" if value is None else value


def _file(path, name):
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is not a file: {resolved}")
    return resolved.resolve()


def _write_csv(path, fields, rows):
    output = Path(path)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
