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
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from PIL import Image

from diffdetect import ImageClass


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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
