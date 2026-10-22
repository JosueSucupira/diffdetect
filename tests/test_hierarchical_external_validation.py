import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from experiments import hierarchical_external_validation as external
from experiments.hierarchical_aggregation import RESULT_FIELDS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/experiments/results"
POLICY = RESULTS / "hierarchical-policy.json"


class HierarchicalExternalValidationTests(unittest.TestCase):
    def test_freezes_balanced_catalog_deterministically_without_overlap(self):
        with tempfile.TemporaryDirectory() as directory, self._small_protocol():
            fixture = self._fixture(Path(directory))
            first = external.freeze_manifest(**fixture["freeze_args"])
            second = external.freeze_manifest(**fixture["freeze_args"])
            self.assertEqual(first, second)
            self.assertEqual(len(first), 12)
            self.assertEqual(
                [record.manifest_index for record in first],
                list(range(1, 13)),
            )
            self.assertEqual(
                {record.true_class for record in first},
                {"real", "synthetic", "edited"},
            )
            self.assertTrue(
                all(len(record.exclusion_manifest_sha256s.split(";")) == 2 for record in first)
            )
            manifest = Path(directory) / "manifest.csv"
            digest = external.write_manifest(first, manifest)
            self.assertEqual(digest, external.file_digest(manifest))
            self.assertEqual(external.load_manifest(manifest), first)

    def test_rejects_overlap_with_earlier_manifest(self):
        with tempfile.TemporaryDirectory() as directory, self._small_protocol():
            fixture = self._fixture(Path(directory), overlap=True)
            with self.assertRaisesRegex(ValueError, "overlaps an earlier manifest"):
                external.freeze_manifest(**fixture["freeze_args"])

    def test_rejects_edited_derivative_of_excluded_coco_origin(self):
        with tempfile.TemporaryDirectory() as directory, self._small_protocol():
            fixture = self._fixture(Path(directory), origin_overlap=True)
            with self.assertRaisesRegex(ValueError, "source overlaps an earlier manifest"):
                external.freeze_manifest(**fixture["freeze_args"])

    def test_rejects_edited_origin_that_matches_an_external_input(self):
        with tempfile.TemporaryDirectory() as directory, self._small_protocol():
            fixture = self._fixture(Path(directory), external_origin_hash_overlap=True)
            with self.assertRaisesRegex(
                ValueError, "edited origin overlaps an external input"
            ):
                external.freeze_manifest(**fixture["freeze_args"])

    def test_rejects_reused_open_images_identity_across_external_sources(self):
        with tempfile.TemporaryDirectory() as directory, self._small_protocol():
            fixture = self._fixture(Path(directory), external_origin_id_overlap=True)
            with self.assertRaisesRegex(ValueError, "external source identity is reused"):
                external.freeze_manifest(**fixture["freeze_args"])

    def test_evaluates_only_the_preregistered_policy_without_fitting(self):
        with tempfile.TemporaryDirectory() as directory, self._small_protocol():
            root = Path(directory)
            fixture = self._fixture(root)
            records = external.freeze_manifest(**fixture["freeze_args"])
            manifest = root / "manifest.csv"
            external.write_manifest(records, manifest)
            raw = root / "raw.csv"
            self._write_results(raw, records, external.file_digest(manifest))
            metadata = raw.with_suffix(".csv.metadata.json")
            metadata.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "manifest_sha256": external.file_digest(manifest),
                        "results_sha256": external.file_digest(raw),
                    }
                ),
                encoding="utf-8",
            )
            output = root / "outputs"
            report = external.evaluate_external(
                policy_path=POLICY,
                expected_policy_sha256=external.EXPECTED_POLICY_SHA256,
                manifest_path=manifest,
                results_path=raw,
                metadata_path=metadata,
                output_dir=output,
                bootstrap_resamples=10,
            )
            self.assertEqual(report["sample_count"], 12)
            self.assertFalse(report["source"]["fitting_performed"])
            self.assertFalse(report["source"]["threshold_selection_performed"])
            self.assertTrue((output / "hierarchical-external-metrics.json").is_file())
            self.assertTrue((output / "hierarchical-external-confusion.png").is_file())
            self.assertNotIn("fit", external._parser()._subparsers._group_actions[0].choices)

    def test_rejects_any_policy_hash_other_than_frozen_artifact(self):
        with self.assertRaisesRegex(ValueError, "preregistered frozen policy"):
            external.evaluate_external(
                policy_path=POLICY,
                expected_policy_sha256="0" * 64,
                manifest_path="missing",
                results_path="missing",
                metadata_path="missing",
                output_dir="missing",
            )

    def test_group_bootstrap_is_deterministic(self):
        values = {
            "truth": ("real", "synthetic", "edited") * 4,
            "predictions": ("real", "synthetic", "edited") * 4,
            "group_ids": tuple(f"group-{index // 2}" for index in range(12)),
            "resamples": 20,
            "seed": 123,
        }
        self.assertEqual(external.group_bootstrap(**values), external.group_bootstrap(**values))

    @staticmethod
    def _small_protocol():
        return patch.multiple(
            external,
            SAMPLES_PER_CLASS=4,
            MIN_SYNTHETIC_METHODS=2,
            MAX_SYNTHETIC_METHOD_SHARE=0.50,
        )

    def _fixture(
        self,
        root: Path,
        overlap: bool = False,
        origin_overlap: bool = False,
        external_origin_hash_overlap: bool = False,
        external_origin_id_overlap: bool = False,
    ):
        image_roots = {"source-a": root / "source-a", "source-b": root / "source-b"}
        for path in image_roots.values():
            path.mkdir()
        rows = []
        for class_index, true_class in enumerate(("real", "synthetic", "edited")):
            for index in range(4):
                source_root = "source-a" if index < 2 else "source-b"
                source_dataset = f"dataset-{source_root[-1]}-{true_class}"
                source_id = f"{true_class}-{index}"
                if true_class == "real" and index == 0 and external_origin_id_overlap:
                    source_dataset = "Open Images V6 validation"
                    source_id = "0123456789abcdef"
                filename = f"{true_class}-{index}.png"
                image = image_roots[source_root] / filename
                Image.new(
                    "RGB",
                    (2 + index, 3),
                    (index * 20, 30 + class_index * 40, 40),
                ).save(image)
                origin = image_roots[source_root] / f"origin-{true_class}-{index}.png"
                if true_class == "edited":
                    if external_origin_hash_overlap and index == 0:
                        origin.write_bytes(
                            (image_roots["source-a"] / "real-0.png").read_bytes()
                        )
                    else:
                        Image.new("RGB", (4, 4), (90, index * 20, 10)).save(origin)
                rows.append(
                    {
                        "sample_id": f"external:{true_class}:{index}",
                        "true_class": true_class,
                        "source_dataset": source_dataset,
                        "source_revision": "revision-1",
                        "source_partition": "test",
                        "source_id": source_id,
                        "origin_dataset": (
                            "Open Images V7 validation"
                            if true_class == "edited"
                            and index == 0
                            and external_origin_id_overlap
                            else "coco"
                            if true_class == "edited"
                            else ""
                        ),
                        "origin_id": (
                            "000000123456"
                            if true_class == "edited" and index == 0 and origin_overlap
                            else "0123456789abcdef"
                            if true_class == "edited"
                            and index == 0
                            and external_origin_id_overlap
                            else f"origin-{index}"
                            if true_class == "edited"
                            else ""
                        ),
                        "origin_relative_path": (
                            origin.name if true_class == "edited" else ""
                        ),
                        "origin_input_sha256": (
                            external.file_digest(origin)
                            if true_class == "edited"
                            else ""
                        ),
                        "group_id": f"group:{true_class}:{index}",
                        "source_root": source_root,
                        "relative_path": filename,
                        "source_metadata_sha256": "a" * 64,
                        "input_sha256": external.file_digest(image),
                        "width": 2 + index,
                        "height": 3,
                        "generation_method": (
                            f"generator-{index % 2}" if true_class == "synthetic" else ""
                        ),
                        "editing_method": (
                            f"editor-{index % 2}" if true_class == "edited" else ""
                        ),
                    }
                )
        exclusions = []
        for number in range(2):
            path = root / f"exclude-{number}.csv"
            excluded_sample = rows[0]["sample_id"] if overlap and number == 0 else f"old:{number}"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("sample_id", "source_dataset", "source_id", "input_sha256"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": excluded_sample,
                        "source_dataset": (
                            "nebula/CocoGlide"
                            if origin_overlap and number == 0
                            else f"old-dataset-{number}"
                        ),
                        "source_id": (
                            "123456"
                            if origin_overlap and number == 0
                            else f"old-{number}"
                        ),
                        "input_sha256": str(number + 1) * 64,
                    }
                )
            exclusions.append(path)
        catalog = root / "catalog.csv"
        with catalog.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=external.CANDIDATE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return {
            "freeze_args": {
                "catalog_path": catalog,
                "roots": image_roots,
                "exclusion_manifests": exclusions,
            }
        }

    @staticmethod
    def _write_results(path, records, manifest_sha256):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            for record in records:
                artificial = record.true_class != "real"
                edited = record.true_class == "edited"
                writer.writerow(
                    {
                        "protocol_version": external.PROTOCOL_VERSION,
                        "manifest_sha256": manifest_sha256,
                        "manifest_index": record.manifest_index,
                        "sample_id": record.sample_id,
                        "true_class": record.true_class,
                        "source_dataset": record.source_dataset,
                        "source_revision": record.source_revision,
                        "source_partition": record.source_partition,
                        "source_id": record.source_id,
                        "input_sha256": record.input_sha256,
                        "environment_id": "test",
                        "attempt": 1,
                        "run_started_utc": "2026-10-08T00:00:00+00:00",
                        "distildire_score": 0.9 if artificial else 0.1,
                        "distildire_threshold": external.DISTILDIRE_THRESHOLD,
                        "distildire_detected": artificial,
                        "distildire_duration_seconds": 0.1,
                        "distildire_error": "",
                        "dinolizer_score": 0.9 if edited else 0.1,
                        "dinolizer_threshold": external.DINOLIZER_THRESHOLD,
                        "dinolizer_detected": edited,
                        "dinolizer_duration_seconds": 0.1,
                        "dinolizer_error": "",
                        "dinolizer_marked_area": 0.2 if edited else 0.0,
                        "dinolizer_processed_width": record.width,
                        "dinolizer_processed_height": record.height,
                        "dinolizer_window_count": 1,
                        "localization_width": record.width,
                        "localization_height": record.height,
                        "localization_mode": "L",
                        "total_duration_seconds": 0.2,
                    }
                )


if __name__ == "__main__":
    unittest.main()
