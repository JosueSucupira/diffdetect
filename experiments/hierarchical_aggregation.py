"""Prepare the frozen calibration corpus for hierarchical aggregation.

The calibration manifest is intentionally separate from the earlier
three-class validation manifest. Run ``prepare-manifest`` before fitting or
evaluating any aggregation policy, review the generated rows, and commit the
manifest before model inference begins.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Iterable, Mapping, Sequence

from PIL import Image

from diffdetect import (
    DinoLizerDetector,
    DistilDIREDetector,
    DetectorRun,
    DetectorRunner,
    ImageClass,
)


PROTOCOL_VERSION = "diffdetect-hierarchical-calibration-v1"
SAMPLES_PER_CLASS = 256
DIFFUSIONDB_PART = 1800
COCOGLIDE_REVISION = "275f045df7caa2544dd28de7fa86b044ab661bd0"
DIFFUSIONDB_REVISION = "fb620fbe49fa4420e0734bd9c0df11f51176b61f"
COCOGLIDE_CALIBRATION_SHA256 = (
    "6ec4fa85098466cdafcb592afdd1be59c9c474ac09679f9da557522986e86244"
)
THREE_CLASS_VALIDATION_MANIFEST_SHA256 = (
    "9dcda4b93d8c73df7b1240629bec209566cd6119bb7f800c2906b8c3826b8777"
)
NSFW_LIMIT = 0.1
TRUE_CLASSES = tuple(image_class.value for image_class in ImageClass)
DISTILDIRE_REVISION = "5de48cf"
DISTILDIRE_CLASSIFIER_SHA256 = (
    "e6b76a13ae49eb83d39fb9b1f7de86bf9f63d1dddf0225ca7ad03690e6bc53bc"
)
DISTILDIRE_ADM_MD5 = "fd9dd2335b8736d521de0aed54bd90ca"
DISTILDIRE_THRESHOLD = 0.9877818822860718
DINOLIZER_REVISION = "3241ce530a685e6e0560db4e0d8aaa9f28de6fc6"
DINOLIZER_CHECKPOINT_SHA256 = (
    "96cb26f2536919d67b75a5aff195b683a1d8590fe596e6f1260ab7950883e2a2"
)
DINOLIZER_THRESHOLD = 0.5946570634841919

MANIFEST_FIELDS = (
    "protocol_version",
    "manifest_index",
    "sample_id",
    "true_class",
    "source_dataset",
    "source_revision",
    "source_partition",
    "source_id",
    "source_root",
    "relative_path",
    "source_metadata_sha256",
    "exclusion_manifest_sha256",
    "input_sha256",
    "width",
    "height",
)

EXCLUSION_FIELDS = {
    "sample_id",
    "true_class",
    "source_dataset",
    "source_id",
    "input_sha256",
}

RESULT_FIELDS = (
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


@dataclass(frozen=True)
class CalibrationManifestRecord:
    """One immutable input in the hierarchical calibration corpus."""

    protocol_version: str
    manifest_index: int
    sample_id: str
    true_class: str
    source_dataset: str
    source_revision: str
    source_partition: str
    source_id: str
    source_root: str
    relative_path: str
    source_metadata_sha256: str
    exclusion_manifest_sha256: str
    input_sha256: str
    width: int
    height: int

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> CalibrationManifestRecord:
        missing = set(MANIFEST_FIELDS) - row.keys()
        if missing:
            raise ValueError(f"manifest row is missing fields: {', '.join(sorted(missing))}")
        try:
            return cls(
                protocol_version=row["protocol_version"],
                manifest_index=int(row["manifest_index"]),
                sample_id=row["sample_id"],
                true_class=row["true_class"],
                source_dataset=row["source_dataset"],
                source_revision=row["source_revision"],
                source_partition=row["source_partition"],
                source_id=row["source_id"],
                source_root=row["source_root"],
                relative_path=row["relative_path"],
                source_metadata_sha256=row["source_metadata_sha256"],
                exclusion_manifest_sha256=row["exclusion_manifest_sha256"],
                input_sha256=row["input_sha256"],
                width=int(row["width"]),
                height=int(row["height"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("manifest row contains an invalid numeric field") from exc

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unexpected protocol version: {self.protocol_version}")
        if self.manifest_index <= 0:
            raise ValueError("manifest_index must be positive")
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.true_class not in TRUE_CLASSES:
            raise ValueError(f"invalid true class: {self.true_class}")
        if any(
            not value
            for value in (
                self.source_dataset,
                self.source_revision,
                self.source_partition,
                self.source_id,
            )
        ):
            raise ValueError("manifest source fields must not be empty")
        if self.source_root not in {"cocoglide", "diffusiondb"}:
            raise ValueError(f"invalid source root: {self.source_root}")
        expected_source = {
            "cocoglide": ("nebula/CocoGlide", COCOGLIDE_REVISION),
            "diffusiondb": ("poloclub/diffusiondb", DIFFUSIONDB_REVISION),
        }[self.source_root]
        if (self.source_dataset, self.source_revision) != expected_source:
            raise ValueError(f"unexpected source identity for {self.source_root}")
        expected_partition = (
            self.source_partition == "calibration"
            if self.source_root == "cocoglide"
            else self.source_partition.startswith("part-")
        )
        if not expected_partition:
            raise ValueError(f"unexpected source partition: {self.source_partition}")
        _validate_relative_path(self.relative_path)
        _validate_hex_digest(self.source_metadata_sha256, "source metadata SHA-256")
        _validate_hex_digest(
            self.exclusion_manifest_sha256,
            "exclusion manifest SHA-256",
        )
        _validate_hex_digest(self.input_sha256, "input SHA-256")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("manifest dimensions must be positive")

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExcludedIdentities:
    """Identities that cannot enter the new calibration corpus."""

    manifest_sha256: str
    sample_ids: frozenset[str]
    source_ids: frozenset[tuple[str, str]]
    input_sha256s: frozenset[str]


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    """Calculate a file digest without loading the whole file into memory."""

    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_excluded_identities(manifest_path: str | Path) -> ExcludedIdentities:
    """Load source and content identities from an earlier frozen manifest."""

    manifest_path = _file(manifest_path, "exclude_manifest")
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not EXCLUSION_FIELDS.issubset(reader.fieldnames):
            raise ValueError("exclusion manifest does not match the expected schema")
        rows = list(reader)
    if not rows:
        raise ValueError("exclusion manifest must contain at least one row")

    sample_ids: set[str] = set()
    source_ids: set[tuple[str, str]] = set()
    input_sha256s: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        source_dataset = row["source_dataset"]
        source_id = row["source_id"]
        input_sha256 = row["input_sha256"]
        if not sample_id or not source_dataset or not source_id:
            raise ValueError("exclusion manifest identities must not be empty")
        _validate_hex_digest(input_sha256, "excluded input SHA-256")
        if sample_id in sample_ids:
            raise ValueError("exclusion manifest contains duplicate sample identifiers")
        sample_ids.add(sample_id)
        source_ids.add((source_dataset, source_id))
        input_sha256s.add(input_sha256)

    return ExcludedIdentities(
        manifest_sha256=file_digest(manifest_path),
        sample_ids=frozenset(sample_ids),
        source_ids=frozenset(source_ids),
        input_sha256s=frozenset(input_sha256s),
    )


def build_calibration_manifest(
    *,
    cocoglide_results: str | Path,
    cocoglide_root: str | Path,
    diffusiondb_metadata: str | Path,
    diffusiondb_root: str | Path,
    exclude_manifest: str | Path,
    samples_per_class: int = SAMPLES_PER_CLASS,
    verify_cocoglide_results: bool = True,
    verify_exclude_manifest: bool = True,
) -> tuple[CalibrationManifestRecord, ...]:
    """Build a balanced calibration manifest with explicit overlap checks."""

    if type(samples_per_class) is not int or samples_per_class <= 0:
        raise ValueError("samples_per_class must be a positive integer")

    cocoglide_results = _file(cocoglide_results, "cocoglide_results")
    cocoglide_root = _directory(cocoglide_root, "cocoglide_root")
    diffusiondb_metadata = _file(diffusiondb_metadata, "diffusiondb_metadata")
    diffusiondb_root = _directory(diffusiondb_root, "diffusiondb_root")
    excluded = load_excluded_identities(exclude_manifest)
    if (
        verify_exclude_manifest
        and excluded.manifest_sha256 != THREE_CLASS_VALIDATION_MANIFEST_SHA256
    ):
        raise ValueError("three-class validation manifest SHA-256 does not match")

    cocoglide_metadata_sha256 = file_digest(cocoglide_results)
    if (
        verify_cocoglide_results
        and cocoglide_metadata_sha256 != COCOGLIDE_CALIBRATION_SHA256
    ):
        raise ValueError("CocoGlide calibration results SHA-256 does not match")
    diffusiondb_metadata_sha256 = file_digest(diffusiondb_metadata)

    records = _cocoglide_records(
        results_path=cocoglide_results,
        image_root=cocoglide_root,
        metadata_sha256=cocoglide_metadata_sha256,
        exclusion_manifest_sha256=excluded.manifest_sha256,
        samples_per_class=samples_per_class,
    )
    _assert_no_excluded_overlap(records, excluded)

    records.extend(
        _diffusiondb_records(
            metadata_path=diffusiondb_metadata,
            image_root=diffusiondb_root,
            metadata_sha256=diffusiondb_metadata_sha256,
            exclusion_manifest_sha256=excluded.manifest_sha256,
            excluded=excluded,
            samples_per_class=samples_per_class,
        )
    )
    _validate_record_collection(records, samples_per_class=samples_per_class)
    _assert_no_excluded_overlap(records, excluded)

    records.sort(key=lambda record: _selection_digest(f"order:{record.sample_id}"))
    return tuple(
        CalibrationManifestRecord(**{**record.to_row(), "manifest_index": index})
        for index, record in enumerate(records, start=1)
    )


def write_calibration_manifest(
    records: Sequence[CalibrationManifestRecord],
    output_path: str | Path,
) -> str:
    """Write a calibration manifest atomically and return its SHA-256."""

    output = Path(output_path)
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


def load_calibration_manifest(
    manifest_path: str | Path,
    *,
    samples_per_class: int = SAMPLES_PER_CLASS,
) -> tuple[CalibrationManifestRecord, ...]:
    """Load and validate a frozen hierarchical calibration manifest."""

    manifest_path = _file(manifest_path, "manifest")
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(MANIFEST_FIELDS):
            raise ValueError("manifest columns do not match the frozen schema")
        records = tuple(CalibrationManifestRecord.from_row(row) for row in reader)
    _validate_record_collection(records, samples_per_class=samples_per_class)
    if tuple(record.manifest_index for record in records) != tuple(
        range(1, len(records) + 1)
    ):
        raise ValueError("manifest indexes must be consecutive and ordered")
    exclusion_hashes = {record.exclusion_manifest_sha256 for record in records}
    if len(exclusion_hashes) != 1:
        raise ValueError("manifest rows must reference one exclusion manifest")
    return records


def resolve_calibration_path(
    record: CalibrationManifestRecord,
    roots: Mapping[str, str | Path],
) -> Path:
    """Resolve one manifest path while preventing traversal outside its root."""

    if record.source_root not in roots:
        raise KeyError(f"missing root for {record.source_root}")
    root = _directory(roots[record.source_root], record.source_root)
    relative = PurePosixPath(record.relative_path)
    candidate = root.joinpath(*relative.parts).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"manifest path escapes {record.source_root} root")
    if not candidate.is_file():
        raise FileNotFoundError(f"manifest image does not exist: {candidate}")
    return candidate


def verify_calibration_inputs(
    records: Iterable[CalibrationManifestRecord],
    roots: Mapping[str, str | Path],
) -> dict[str, Path]:
    """Verify every calibration input hash and dimension before inference."""

    resolved: dict[str, Path] = {}
    for record in records:
        path = resolve_calibration_path(record, roots)
        if file_digest(path) != record.input_sha256:
            raise ValueError(f"input SHA-256 does not match for {record.sample_id}")
        width, height = _verified_image_size(path)
        if (width, height) != (record.width, record.height):
            raise ValueError(f"input dimensions do not match for {record.sample_id}")
        resolved[record.sample_id] = path
    return resolved


def select_smoke_records(
    records: Sequence[CalibrationManifestRecord],
) -> tuple[CalibrationManifestRecord, ...]:
    """Select the earliest calibration-manifest record from each class."""

    selected: dict[str, CalibrationManifestRecord] = {}
    for record in records:
        selected.setdefault(record.true_class, record)
    if set(selected) != set(TRUE_CLASSES):
        raise ValueError("smoke selection requires all three true classes")
    return tuple(sorted(selected.values(), key=lambda record: record.manifest_index))


def calibration_result_row(
    *,
    record: CalibrationManifestRecord,
    runs: Sequence[DetectorRun],
    manifest_sha256: str,
    environment_id: str,
    attempt: int,
    run_started_utc: str,
    total_duration_seconds: float,
) -> dict[str, object]:
    """Flatten raw detector evidence without applying an aggregation policy."""

    by_name = {run.name: run for run in runs}
    if len(runs) != 2 or set(by_name) != {"distildire", "dinolizer"}:
        raise ValueError("expected exactly DistilDIRE and DinoLizer runs")
    distildire = by_name["distildire"]
    dinolizer = by_name["dinolizer"]
    distildire_result = distildire.result
    dinolizer_result = dinolizer.result
    dinolizer_metadata = (
        dict(dinolizer_result.metadata) if dinolizer_result is not None else {}
    )
    processed_size = dinolizer_metadata.get("processed_size")
    if not (
        isinstance(processed_size, tuple)
        and len(processed_size) == 2
        and all(type(value) is int for value in processed_size)
    ):
        processed_size = (None, None)
    localization = (
        dinolizer_result.localization_map if dinolizer_result is not None else None
    )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_sha256": manifest_sha256,
        "manifest_index": record.manifest_index,
        "sample_id": record.sample_id,
        "true_class": record.true_class,
        "source_dataset": record.source_dataset,
        "source_revision": record.source_revision,
        "source_partition": record.source_partition,
        "source_id": record.source_id,
        "input_sha256": record.input_sha256,
        "environment_id": environment_id,
        "attempt": attempt,
        "run_started_utc": run_started_utc,
        "distildire_score": _result_value(distildire_result, "score"),
        "distildire_threshold": _result_value(distildire_result, "threshold"),
        "distildire_detected": _result_value(distildire_result, "detected"),
        "distildire_duration_seconds": distildire.duration_seconds,
        "distildire_error": distildire.error or "",
        "dinolizer_score": _result_value(dinolizer_result, "score"),
        "dinolizer_threshold": _result_value(dinolizer_result, "threshold"),
        "dinolizer_detected": _result_value(dinolizer_result, "detected"),
        "dinolizer_duration_seconds": dinolizer.duration_seconds,
        "dinolizer_error": dinolizer.error or "",
        "dinolizer_marked_area": dinolizer_metadata.get("marked_area", ""),
        "dinolizer_processed_width": processed_size[0] or "",
        "dinolizer_processed_height": processed_size[1] or "",
        "dinolizer_window_count": dinolizer_metadata.get("window_count", ""),
        "localization_width": localization.width if localization is not None else "",
        "localization_height": localization.height if localization is not None else "",
        "localization_mode": localization.mode if localization is not None else "",
        "total_duration_seconds": total_duration_seconds,
    }


def run_calibration(args: argparse.Namespace) -> None:
    """Run a smoke or complete detector pass over the calibration manifest."""

    manifest_path = _file(args.manifest, "manifest")
    records = load_calibration_manifest(manifest_path)
    roots = {
        "cocoglide": args.cocoglide_root,
        "diffusiondb": args.diffusiondb_root,
    }
    resolved_paths = verify_calibration_inputs(records, roots)
    selected = select_smoke_records(records) if args.smoke else records
    manifest_sha256 = file_digest(manifest_path)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _existing_sample_ids(output, manifest_sha256) if args.resume else set()
    if output.exists() and not args.resume:
        raise FileExistsError(f"output already exists; use --resume: {output}")
    pending = tuple(record for record in selected if record.sample_id not in completed)

    artifact_metadata = _verify_model_artifacts(args)
    execution_metadata = {
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
        "first_pending_sample_includes_lazy_model_loading": bool(pending),
    }
    metadata_output = output.with_suffix(output.suffix + ".metadata.json")
    _write_json(metadata_output, execution_metadata)

    if not pending:
        execution_metadata.update(
            status="complete",
            completed_samples=len(completed),
            finished_utc=_utc_now(),
            gpu=_gpu_environment(),
        )
        _write_json(metadata_output, execution_metadata)
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
    processed_this_execution = 0
    try:
        with output.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, lineterminator="\n")
            if not append:
                writer.writeheader()
            for record in pending:
                started_utc = _utc_now()
                started = perf_counter()
                runs = runner.run(resolved_paths[record.sample_id])
                total_duration = perf_counter() - started
                writer.writerow(
                    calibration_result_row(
                        record=record,
                        runs=runs,
                        manifest_sha256=manifest_sha256,
                        environment_id=args.environment_id,
                        attempt=1,
                        run_started_utc=started_utc,
                        total_duration_seconds=total_duration,
                    )
                )
                stream.flush()
                os.fsync(stream.fileno())
                processed_this_execution += 1
                execution_metadata["processed_this_execution"] = (
                    processed_this_execution
                )
                execution_metadata["last_sample_id"] = record.sample_id
                _write_json(metadata_output, execution_metadata)
    except BaseException:
        execution_metadata.update(
            status="interrupted",
            processed_this_execution=processed_this_execution,
            finished_utc=_utc_now(),
            gpu=_gpu_environment(),
        )
        _write_json(metadata_output, execution_metadata)
        raise

    execution_metadata.update(
        status="complete",
        processed_this_execution=processed_this_execution,
        completed_samples=len(completed) + processed_this_execution,
        finished_utc=_utc_now(),
        gpu=_gpu_environment(),
        results_sha256=file_digest(output),
    )
    _write_json(metadata_output, execution_metadata)


def _cocoglide_records(
    *,
    results_path: Path,
    image_root: Path,
    metadata_sha256: str,
    exclusion_manifest_sha256: str,
    samples_per_class: int,
) -> list[CalibrationManifestRecord]:
    records: list[CalibrationManifestRecord] = []
    with results_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "split",
            "coco_id",
            "expected_class",
            "image_file",
            "original_width",
            "original_height",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CocoGlide calibration CSV does not match the schema")
        for row in reader:
            if row["split"] != "calibration":
                continue
            true_class = {
                "real": ImageClass.REAL.value,
                "manipulated": ImageClass.EDITED.value,
            }.get(row["expected_class"])
            if true_class is None:
                raise ValueError(f"unknown CocoGlide class: {row['expected_class']}")
            relative_path = _normalize_relative_path(row["image_file"])
            path = _rooted_file(image_root, relative_path)
            width, height = _verified_image_size(path)
            expected_size = (int(row["original_width"]), int(row["original_height"]))
            if (width, height) != expected_size:
                raise ValueError(f"CocoGlide dimensions do not match for {path.name}")
            coco_id = row["coco_id"]
            records.append(
                CalibrationManifestRecord(
                    protocol_version=PROTOCOL_VERSION,
                    manifest_index=1,
                    sample_id=f"cocoglide:{coco_id}:{true_class}",
                    true_class=true_class,
                    source_dataset="nebula/CocoGlide",
                    source_revision=COCOGLIDE_REVISION,
                    source_partition="calibration",
                    source_id=coco_id,
                    source_root="cocoglide",
                    relative_path=relative_path,
                    source_metadata_sha256=metadata_sha256,
                    exclusion_manifest_sha256=exclusion_manifest_sha256,
                    input_sha256=file_digest(path),
                    width=width,
                    height=height,
                )
            )

    counts = Counter(record.true_class for record in records)
    expected = {
        ImageClass.REAL.value: samples_per_class,
        ImageClass.EDITED.value: samples_per_class,
    }
    if counts != expected:
        raise ValueError(f"unexpected CocoGlide calibration counts: {dict(counts)}")
    return records


def _diffusiondb_records(
    *,
    metadata_path: Path,
    image_root: Path,
    metadata_sha256: str,
    exclusion_manifest_sha256: str,
    excluded: ExcludedIdentities,
    samples_per_class: int,
) -> list[CalibrationManifestRecord]:
    rows = _read_diffusiondb_metadata(metadata_path, image_root)
    selected: list[tuple[Mapping[str, object], str, Path, str]] = []
    selected_hashes: set[str] = set()

    for part_id in _ordered_available_parts(image_root):
        eligible: list[tuple[str, Mapping[str, object], str, Path]] = []
        for row in rows.get(part_id, ()):
            image_name = str(row.get("image_name", ""))
            if not image_name or not _safe_diffusiondb_row(row):
                continue
            if ("poloclub/diffusiondb", image_name) in excluded.source_ids:
                continue
            resolved = _find_diffusiondb_image(image_root, part_id, image_name)
            if resolved is None:
                continue
            relative = resolved.relative_to(image_root).as_posix()
            eligible.append(
                (_selection_digest(image_name), row, relative, resolved)
            )
        eligible.sort(key=lambda item: item[0])
        for _, row, relative, resolved in eligible:
            input_sha256 = file_digest(resolved)
            if input_sha256 in excluded.input_sha256s or input_sha256 in selected_hashes:
                continue
            selected.append((row, relative, resolved, input_sha256))
            selected_hashes.add(input_sha256)
            if len(selected) == samples_per_class:
                break
        if len(selected) == samples_per_class:
            break

    if len(selected) != samples_per_class:
        raise ValueError(
            f"expected {samples_per_class} eligible DiffusionDB images, "
            f"found {len(selected)} without validation overlap"
        )

    records: list[CalibrationManifestRecord] = []
    for row, relative_path, path, input_sha256 in selected:
        image_name = str(row["image_name"])
        part_id = int(row["part_id"])
        width, height = _verified_image_size(path)
        expected_size = (int(row["width"]), int(row["height"]))
        if (width, height) != expected_size:
            raise ValueError(f"DiffusionDB dimensions do not match for {image_name}")
        records.append(
            CalibrationManifestRecord(
                protocol_version=PROTOCOL_VERSION,
                manifest_index=1,
                sample_id=f"diffusiondb:{image_name}",
                true_class=ImageClass.SYNTHETIC.value,
                source_dataset="poloclub/diffusiondb",
                source_revision=DIFFUSIONDB_REVISION,
                source_partition=f"part-{part_id:06d}",
                source_id=image_name,
                source_root="diffusiondb",
                relative_path=relative_path,
                source_metadata_sha256=metadata_sha256,
                exclusion_manifest_sha256=exclusion_manifest_sha256,
                input_sha256=input_sha256,
                width=width,
                height=height,
            )
        )
    return records


def _read_diffusiondb_metadata(
    metadata_path: Path,
    image_root: Path,
) -> dict[int, list[Mapping[str, object]]]:
    required = {
        "image_name",
        "part_id",
        "width",
        "height",
        "image_nsfw",
        "prompt_nsfw",
    }
    available_parts = _ordered_available_parts(image_root)
    if metadata_path.suffix.lower() == ".csv":
        with metadata_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("DiffusionDB metadata CSV does not match the schema")
            rows = [row for row in reader if int(row["part_id"]) in available_parts]
    elif metadata_path.suffix.lower() in {".parquet", ".pq"}:
        try:
            pandas = __import__("pandas")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "reading DiffusionDB Parquet metadata requires pandas and pyarrow"
            ) from exc
        frame = pandas.read_parquet(
            metadata_path,
            columns=sorted(required),
            filters=[("part_id", "in", list(available_parts))],
        )
        rows = frame.to_dict(orient="records")
    else:
        raise ValueError("DiffusionDB metadata must be CSV or Parquet")

    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        part_id = int(row["part_id"])
        grouped.setdefault(part_id, []).append(row)
    return grouped


def _ordered_available_parts(image_root: Path) -> tuple[int, ...]:
    parts: set[int] = set()
    if any(path.is_file() for path in image_root.glob("*.png")):
        parts.add(DIFFUSIONDB_PART)
    for parent in (image_root, image_root / "images"):
        if not parent.is_dir():
            continue
        for path in parent.glob("part-[0-9][0-9][0-9][0-9][0-9][0-9]"):
            if path.is_dir():
                parts.add(int(path.name.removeprefix("part-")))
    if DIFFUSIONDB_PART not in parts:
        raise FileNotFoundError(
            f"DiffusionDB part {DIFFUSIONDB_PART:06d} is not available under "
            f"{image_root}"
        )
    return tuple(sorted(parts, key=lambda part: (part - DIFFUSIONDB_PART) % 2000))


def _find_diffusiondb_image(
    image_root: Path,
    part_id: int,
    image_name: str,
) -> Path | None:
    _validate_filename(image_name)
    candidates = (
        image_root / image_name,
        image_root / f"part-{part_id:06d}" / image_name,
        image_root / "images" / f"part-{part_id:06d}" / image_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if image_root == resolved or image_root not in resolved.parents:
                raise ValueError("DiffusionDB image path escapes its root")
            return resolved
    return None


def _safe_diffusiondb_row(row: Mapping[str, object]) -> bool:
    try:
        image_nsfw = float(row["image_nsfw"])
        prompt_nsfw = float(row["prompt_nsfw"])
        width = int(row["width"])
        height = int(row["height"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(image_nsfw)
        and math.isfinite(prompt_nsfw)
        and image_nsfw <= NSFW_LIMIT
        and prompt_nsfw <= NSFW_LIMIT
        and width > 0
        and height > 0
    )


def _assert_no_excluded_overlap(
    records: Iterable[CalibrationManifestRecord],
    excluded: ExcludedIdentities,
) -> None:
    for record in records:
        if record.sample_id in excluded.sample_ids:
            raise ValueError(f"sample overlaps exclusion manifest: {record.sample_id}")
        source_identity = (record.source_dataset, record.source_id)
        if source_identity in excluded.source_ids:
            raise ValueError(
                "source overlaps exclusion manifest: "
                f"{record.source_dataset}:{record.source_id}"
            )
        if record.input_sha256 in excluded.input_sha256s:
            raise ValueError(f"input overlaps exclusion manifest: {record.sample_id}")


def _validate_record_collection(
    records: Iterable[CalibrationManifestRecord],
    *,
    samples_per_class: int,
) -> None:
    records = tuple(records)
    expected_total = samples_per_class * len(TRUE_CLASSES)
    if len(records) != expected_total:
        raise ValueError(f"expected {expected_total} records, found {len(records)}")
    counts = Counter(record.true_class for record in records)
    expected_counts = {true_class: samples_per_class for true_class in TRUE_CLASSES}
    if counts != expected_counts:
        raise ValueError(f"manifest is not balanced: {dict(counts)}")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("manifest sample identifiers must be unique")
    if len({record.input_sha256 for record in records}) != len(records):
        raise ValueError("manifest input hashes must be unique")


def _verify_model_artifacts(args: argparse.Namespace) -> dict[str, object]:
    distildire_repository = _directory(
        args.distildire_repository,
        "distildire_repository",
    )
    dinolizer_repository = _directory(args.dinolizer_repository, "dinolizer_repository")
    classifier = _file(args.distildire_classifier, "distildire_classifier")
    adm = _file(args.distildire_adm, "distildire_adm")
    dinolizer_checkpoint = _file(args.dinolizer_checkpoint, "dinolizer_checkpoint")

    distildire_head = _git_head(distildire_repository)
    dinolizer_head = _git_head(dinolizer_repository)
    if not distildire_head.startswith(DISTILDIRE_REVISION):
        raise ValueError(f"unexpected DistilDIRE revision: {distildire_head}")
    if dinolizer_head != DINOLIZER_REVISION:
        raise ValueError(f"unexpected DinoLizer revision: {dinolizer_head}")

    classifier_sha256 = file_digest(classifier)
    adm_md5 = file_digest(adm, "md5")
    dinolizer_sha256 = file_digest(dinolizer_checkpoint)
    if classifier_sha256 != DISTILDIRE_CLASSIFIER_SHA256:
        raise ValueError("DistilDIRE classifier SHA-256 does not match")
    if adm_md5 != DISTILDIRE_ADM_MD5:
        raise ValueError("ADM checkpoint MD5 does not match")
    if dinolizer_sha256 != DINOLIZER_CHECKPOINT_SHA256:
        raise ValueError("DinoLizer checkpoint SHA-256 does not match")

    return {
        "distildire_repository": str(distildire_repository),
        "distildire_revision": distildire_head,
        "distildire_classifier": str(classifier),
        "distildire_classifier_sha256": classifier_sha256,
        "distildire_adm": str(adm),
        "distildire_adm_md5": adm_md5,
        "dinolizer_repository": str(dinolizer_repository),
        "dinolizer_revision": dinolizer_head,
        "dinolizer_checkpoint": str(dinolizer_checkpoint),
        "dinolizer_checkpoint_sha256": dinolizer_sha256,
        "distildire_threshold": DISTILDIRE_THRESHOLD,
        "dinolizer_threshold": DINOLIZER_THRESHOLD,
        "device": args.device,
    }


def _existing_sample_ids(output: Path, manifest_sha256: str) -> set[str]:
    if not output.exists():
        return set()
    with output.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(RESULT_FIELDS):
            raise ValueError("existing result columns do not match the frozen schema")
        rows = list(reader)
    if any(row["manifest_sha256"] != manifest_sha256 for row in rows):
        raise ValueError("existing results belong to another manifest")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("existing results contain duplicate sample identifiers")
    return set(sample_ids)


def _software_environment() -> dict[str, object]:
    packages = {}
    for distribution in (
        "diffdetect",
        "numpy",
        "Pillow",
        "torch",
        "torchvision",
        "timm",
        "albumentations",
        "einops",
        "safetensors",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "gpu_at_start": _gpu_environment(),
    }


def _gpu_environment() -> dict[str, object]:
    try:
        torch = __import__("torch")
    except ModuleNotFoundError:
        return {"torch_available": False}
    cuda_available = bool(torch.cuda.is_available())
    data: dict[str, object] = {
        "torch_available": True,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
    }
    if cuda_available:
        data.update(
            device_name=torch.cuda.get_device_name(0),
            device_count=torch.cuda.device_count(),
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    return data


def _reset_gpu_peak_memory() -> None:
    try:
        torch = __import__("torch")
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _selection_digest(value: str) -> str:
    return hashlib.sha256(f"{PROTOCOL_VERSION}:{value}".encode()).hexdigest()


def _verified_image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            size = image.size
            image.verify()
    except Exception as exc:
        raise ValueError(f"invalid input image: {path}") from exc
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"invalid input dimensions: {path}")
    return size


def _rooted_file(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    candidate = root.joinpath(*relative.parts).resolve()
    if root == candidate or root not in candidate.parents:
        raise ValueError("input path escapes its configured root")
    if not candidate.is_file():
        raise FileNotFoundError(f"input image does not exist: {candidate}")
    return candidate


def _normalize_relative_path(value: str) -> str:
    _validate_relative_path(value)
    return PurePosixPath(value).as_posix()


def _validate_relative_path(value: str) -> None:
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


def _validate_filename(value: str) -> None:
    if PurePosixPath(value).name != value or value in {"", ".", ".."}:
        raise ValueError(f"invalid image filename: {value}")


def _validate_hex_digest(value: str, name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must contain 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must contain 64 hexadecimal characters") from exc


def _directory(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} is not a directory: {path}")
    return path.resolve()


def _file(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")
    return path.resolve()


def _result_value(result, name: str):
    return getattr(result, name) if result is not None else ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _prepare_manifest_command(args: argparse.Namespace) -> None:
    records = build_calibration_manifest(
        cocoglide_results=args.cocoglide_results,
        cocoglide_root=args.cocoglide_root,
        diffusiondb_metadata=args.diffusiondb_metadata,
        diffusiondb_root=args.diffusiondb_root,
        exclude_manifest=args.exclude_manifest,
    )
    digest = write_calibration_manifest(records, args.output)
    counts = Counter(record.true_class for record in records)
    print(f"wrote {len(records)} records to {args.output}")
    print(f"class counts: {dict(sorted(counts.items()))}")
    print(f"manifest SHA-256: {digest}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-manifest",
        help="freeze and verify the balanced 768-image calibration manifest",
    )
    prepare.add_argument("--cocoglide-results", required=True, type=Path)
    prepare.add_argument("--cocoglide-root", required=True, type=Path)
    prepare.add_argument("--diffusiondb-metadata", required=True, type=Path)
    prepare.add_argument("--diffusiondb-root", required=True, type=Path)
    prepare.add_argument("--exclude-manifest", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.set_defaults(handler=_prepare_manifest_command)

    run = commands.add_parser(
        "run",
        help="run the smoke test or complete frozen calibration inference",
    )
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--cocoglide-root", required=True, type=Path)
    run.add_argument("--diffusiondb-root", required=True, type=Path)
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
    run.set_defaults(handler=run_calibration)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
