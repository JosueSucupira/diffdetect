import csv
import json
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from diffdetect import BaseDetector, DetectionResult, DetectionTarget, DetectorRun
from experiments.hierarchical_aggregation import (
    DIFFUSIONDB_PART,
    PROTOCOL_VERSION,
    RESULT_FIELDS,
    build_calibration_manifest,
    calibration_result_row,
    file_digest,
    load_calibration_manifest,
    load_excluded_identities,
    resolve_calibration_path,
    run_calibration,
    select_smoke_records,
    verify_calibration_inputs,
    write_calibration_manifest,
)


class FixedExperimentDetector(BaseDetector):
    def __init__(self, name, target, detected):
        self.name = name
        self.target = target
        self.detected = detected

    def predict(self, image):
        metadata = {}
        localization_map = None
        if self.target is DetectionTarget.EDITED:
            metadata = {
                "marked_area": 0.1,
                "processed_size": (1016, 1016),
                "window_count": 25,
            }
            localization_map = Image.new("F", image.size, color=0.1)
        return DetectionResult(
            detected=self.detected,
            score=0.9 if self.detected else 0.1,
            threshold=0.5,
            localization_map=localization_map,
            metadata=metadata,
        )


class HierarchicalAggregationExperimentTests(unittest.TestCase):
    def test_builds_balanced_deterministic_manifest_without_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._create_fixture(root)

            first = build_calibration_manifest(**fixture, samples_per_class=2)
            second = build_calibration_manifest(**fixture, samples_per_class=2)

            self.assertEqual(first, second)
            self.assertEqual(len(first), 6)
            self.assertEqual(
                Counter(record.true_class for record in first),
                {"real": 2, "synthetic": 2, "edited": 2},
            )
            self.assertEqual(
                tuple(record.manifest_index for record in first),
                (1, 2, 3, 4, 5, 6),
            )

            excluded = load_excluded_identities(fixture["exclude_manifest"])
            self.assertTrue(
                all(
                    record.exclusion_manifest_sha256 == excluded.manifest_sha256
                    for record in first
                )
            )
            self.assertNotIn("diffusiondb:excluded.png", {r.sample_id for r in first})
            self.assertNotIn(
                "diffusiondb:duplicate-content.png",
                {r.sample_id for r in first},
            )
            self.assertFalse(
                {record.input_sha256 for record in first} & excluded.input_sha256s
            )

            manifest = root / "calibration-manifest.csv"
            digest = write_calibration_manifest(first, manifest)
            self.assertEqual(digest, file_digest(manifest))
            loaded = load_calibration_manifest(manifest, samples_per_class=2)
            self.assertEqual(loaded, first)

            roots = {
                "cocoglide": fixture["cocoglide_root"],
                "diffusiondb": fixture["diffusiondb_root"],
            }
            resolved = verify_calibration_inputs(loaded, roots)
            self.assertEqual(set(resolved), {record.sample_id for record in loaded})

    def test_rejects_cocoglide_source_overlap_with_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._create_fixture(root, excluded_coco_id="000001")

            with self.assertRaisesRegex(ValueError, "overlaps exclusion manifest"):
                build_calibration_manifest(**fixture, samples_per_class=2)

    def test_verifies_frozen_exclusion_manifest_digest_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._create_fixture(Path(directory))
            fixture.pop("verify_exclude_manifest")

            with self.assertRaisesRegex(ValueError, "validation manifest SHA-256"):
                build_calibration_manifest(**fixture, samples_per_class=2)

    def test_rejects_tampered_input_and_unsafe_relative_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._create_fixture(root)
            records = build_calibration_manifest(**fixture, samples_per_class=2)
            roots = {
                "cocoglide": fixture["cocoglide_root"],
                "diffusiondb": fixture["diffusiondb_root"],
            }

            target = records[0]
            path = resolve_calibration_path(target, roots)
            Image.new("RGB", (4, 3), color="black").save(path)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_calibration_inputs(records, roots)

            with self.assertRaisesRegex(ValueError, "relative path"):
                replace(target, relative_path="../escape.png").validate()

    def test_requires_selected_diffusiondb_part(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._create_fixture(root)
            for image in Path(fixture["diffusiondb_root"]).glob("*.png"):
                image.unlink()

            with self.assertRaisesRegex(FileNotFoundError, f"{DIFFUSIONDB_PART:06d}"):
                build_calibration_manifest(**fixture, samples_per_class=2)

    def test_flattens_raw_detector_evidence_without_final_label(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._create_fixture(Path(directory))
            record = build_calibration_manifest(**fixture, samples_per_class=2)[0]
            runs = (
                DetectorRun(
                    name="distildire",
                    target=DetectionTarget.SYNTHETIC,
                    duration_seconds=0.2,
                    result=DetectionResult(
                        detected=True,
                        score=0.99,
                        threshold=0.98,
                    ),
                ),
                DetectorRun(
                    name="dinolizer",
                    target=DetectionTarget.EDITED,
                    duration_seconds=0.3,
                    result=DetectionResult(
                        detected=True,
                        score=0.8,
                        threshold=0.6,
                        localization_map=Image.new("F", (4, 3), color=0.25),
                        metadata={
                            "marked_area": 0.25,
                            "processed_size": (1016, 1016),
                            "window_count": 25,
                        },
                    ),
                ),
            )

            row = calibration_result_row(
                record=record,
                runs=runs,
                manifest_sha256="a" * 64,
                environment_id="colab-t4-test",
                attempt=1,
                run_started_utc="2026-10-07T12:00:00+00:00",
                total_duration_seconds=0.5,
            )

            self.assertEqual(set(row), set(RESULT_FIELDS))
            self.assertNotIn("final_label", row)
            self.assertEqual(row["dinolizer_marked_area"], 0.25)
            self.assertEqual(row["dinolizer_processed_width"], 1016)
            self.assertEqual(row["localization_mode"], "F")

    def test_smoke_run_writes_raw_results_and_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "input.png"
            Image.new("RGB", (4, 3), color="white").save(image_path)
            image_sha256 = file_digest(image_path)
            manifest_path = root / "manifest.csv"
            manifest_path.write_text("frozen manifest fixture\n", encoding="utf-8")
            output = root / "smoke.csv"
            records = tuple(
                self._record_for_run(
                    index=index,
                    true_class=true_class,
                    image_sha256=image_sha256,
                )
                for index, true_class in enumerate(
                    ("real", "synthetic", "edited"),
                    start=1,
                )
            )
            self.assertEqual(len(select_smoke_records(records)), 3)
            resolved = {record.sample_id: image_path for record in records}
            args = Namespace(
                manifest=manifest_path,
                cocoglide_root=root,
                diffusiondb_root=root,
                distildire_repository=root,
                distildire_classifier=image_path,
                distildire_adm=image_path,
                dinolizer_repository=root,
                dinolizer_checkpoint=image_path,
                device="cpu",
                environment_id="unit-test",
                output=output,
                smoke=True,
                resume=False,
            )

            with self._patched_run_dependencies(records, resolved):
                run_calibration(args)

            with output.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, list(RESULT_FIELDS))
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["sample_id"] for row in rows}, set(resolved))
            self.assertNotIn("final_label", reader.fieldnames)

            metadata_path = output.with_suffix(".csv.metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["completed_samples"], 3)
            self.assertFalse(metadata["aggregation_policy_applied"])

            args.resume = True
            with self._patched_run_dependencies(records, resolved):
                run_calibration(args)
            with output.open("r", encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)

    @contextmanager
    def _patched_run_dependencies(self, records, resolved):
        patches = (
            patch(
                "experiments.hierarchical_aggregation.load_calibration_manifest",
                return_value=records,
            ),
            patch(
                "experiments.hierarchical_aggregation.verify_calibration_inputs",
                return_value=resolved,
            ),
            patch(
                "experiments.hierarchical_aggregation._verify_model_artifacts",
                return_value={"verified": True},
            ),
            patch(
                "experiments.hierarchical_aggregation._software_environment",
                return_value={"test": True},
            ),
            patch(
                "experiments.hierarchical_aggregation._gpu_environment",
                return_value={"test": True},
            ),
            patch("experiments.hierarchical_aggregation._reset_gpu_peak_memory"),
            patch(
                "experiments.hierarchical_aggregation.DistilDIREDetector",
                return_value=FixedExperimentDetector(
                    "distildire",
                    DetectionTarget.SYNTHETIC,
                    True,
                ),
            ),
            patch(
                "experiments.hierarchical_aggregation.DinoLizerDetector",
                return_value=FixedExperimentDetector(
                    "dinolizer",
                    DetectionTarget.EDITED,
                    False,
                ),
            ),
        )
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            yield

    def _record_for_run(self, *, index, true_class, image_sha256):
        from experiments.hierarchical_aggregation import CalibrationManifestRecord

        is_synthetic = true_class == "synthetic"
        return CalibrationManifestRecord(
            protocol_version=PROTOCOL_VERSION,
            manifest_index=index,
            sample_id=f"sample-{true_class}",
            true_class=true_class,
            source_dataset=(
                "poloclub/diffusiondb" if is_synthetic else "nebula/CocoGlide"
            ),
            source_revision=(
                "fb620fbe49fa4420e0734bd9c0df11f51176b61f"
                if is_synthetic
                else "275f045df7caa2544dd28de7fa86b044ab661bd0"
            ),
            source_partition="part-001800" if is_synthetic else "calibration",
            source_id=f"source-{index}",
            source_root="diffusiondb" if is_synthetic else "cocoglide",
            relative_path="input.png",
            source_metadata_sha256="b" * 64,
            exclusion_manifest_sha256="c" * 64,
            input_sha256=image_sha256,
            width=4,
            height=3,
        )

    def _create_fixture(
        self,
        root: Path,
        *,
        excluded_coco_id: str | None = None,
    ) -> dict[str, object]:
        cocoglide_root = root / "cocoglide"
        diffusiondb_root = root / "diffusiondb"
        cocoglide_root.mkdir()
        diffusiondb_root.mkdir()

        cocoglide_results = root / "cocoglide-calibration.csv"
        coco_fields = (
            "split",
            "coco_id",
            "expected_class",
            "image_file",
            "original_width",
            "original_height",
        )
        coco_rows = []
        for coco_id, colors in (
            ("000001", ("red", "green")),
            ("000002", ("blue", "yellow")),
        ):
            for expected_class, color in zip(("real", "manipulated"), colors):
                filename = f"{coco_id}_{expected_class}.png"
                Image.new("RGB", (4, 3), color=color).save(cocoglide_root / filename)
                coco_rows.append(
                    {
                        "split": "calibration",
                        "coco_id": coco_id,
                        "expected_class": expected_class,
                        "image_file": filename,
                        "original_width": 4,
                        "original_height": 3,
                    }
                )
        with cocoglide_results.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=coco_fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(coco_rows)

        diffusiondb_metadata = root / "diffusiondb.csv"
        diffusion_fields = (
            "image_name",
            "part_id",
            "width",
            "height",
            "image_nsfw",
            "prompt_nsfw",
        )
        diffusion_rows = []
        image_digests = {}
        for index, filename in enumerate(
            (
                "excluded.png",
                "duplicate-content.png",
                "synthetic-1.png",
                "synthetic-2.png",
                "synthetic-3.png",
            )
        ):
            path = diffusiondb_root / filename
            color_index = 0 if filename == "duplicate-content.png" else index
            Image.new("RGB", (4, 3), color=(color_index * 40, 10, 10)).save(path)
            image_digests[filename] = file_digest(path)
            diffusion_rows.append(
                {
                    "image_name": filename,
                    "part_id": DIFFUSIONDB_PART,
                    "width": 4,
                    "height": 3,
                    "image_nsfw": 0.01,
                    "prompt_nsfw": 0.01,
                }
            )
        with diffusiondb_metadata.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=diffusion_fields,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(diffusion_rows)

        exclude_manifest = root / "validation-manifest.csv"
        exclusion_fields = (
            "sample_id",
            "true_class",
            "source_dataset",
            "source_id",
            "input_sha256",
        )
        exclusion_rows = [
            {
                "sample_id": "diffusiondb:excluded.png",
                "true_class": "synthetic",
                "source_dataset": "poloclub/diffusiondb",
                "source_id": "excluded.png",
                "input_sha256": image_digests["excluded.png"],
            }
        ]
        if excluded_coco_id is not None:
            path = cocoglide_root / f"{excluded_coco_id}_real.png"
            exclusion_rows.append(
                {
                    "sample_id": f"cocoglide:{excluded_coco_id}:real",
                    "true_class": "real",
                    "source_dataset": "nebula/CocoGlide",
                    "source_id": excluded_coco_id,
                    "input_sha256": file_digest(path),
                }
            )
        with exclude_manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=exclusion_fields,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(exclusion_rows)

        return {
            "cocoglide_results": cocoglide_results,
            "cocoglide_root": cocoglide_root,
            "diffusiondb_metadata": diffusiondb_metadata,
            "diffusiondb_root": diffusiondb_root,
            "exclude_manifest": exclude_manifest,
            "verify_cocoglide_results": False,
            "verify_exclude_manifest": False,
        }


if __name__ == "__main__":
    unittest.main()
