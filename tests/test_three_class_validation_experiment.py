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
from experiments.three_class_validation import (
    RESULT_FIELDS,
    ManifestRecord,
    PROTOCOL_VERSION,
    build_manifest,
    file_digest,
    load_manifest,
    resolve_manifest_path,
    result_row,
    run_validation,
    select_smoke_records,
    verify_manifest_inputs,
    write_manifest,
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


class ThreeClassValidationExperimentTests(unittest.TestCase):
    def test_builds_balanced_deterministic_manifest_and_verifies_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._create_fixture(root)

            first = build_manifest(**fixture, samples_per_class=2)
            second = build_manifest(**fixture, samples_per_class=2)

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

            manifest = root / "manifest.csv"
            digest = write_manifest(first, manifest)
            self.assertEqual(digest, file_digest(manifest))
            loaded = load_manifest(manifest, samples_per_class=2)
            self.assertEqual(loaded, first)

            roots = {
                "cocoglide": fixture["cocoglide_root"],
                "diffusiondb": fixture["diffusiondb_root"],
            }
            resolved = verify_manifest_inputs(loaded, roots)
            self.assertEqual(set(resolved), {record.sample_id for record in loaded})

            smoke = select_smoke_records(loaded)
            self.assertEqual(len(smoke), 3)
            self.assertEqual(
                {record.true_class for record in smoke},
                {"real", "synthetic", "edited"},
            )

    def test_detects_tampered_input_and_unsafe_manifest_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._create_fixture(root)
            records = build_manifest(**fixture, samples_per_class=2)
            roots = {
                "cocoglide": fixture["cocoglide_root"],
                "diffusiondb": fixture["diffusiondb_root"],
            }

            target = records[0]
            path = resolve_manifest_path(target, roots)
            Image.new("RGB", (4, 3), color="black").save(path)

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_manifest_inputs(records, roots)

            with self.assertRaisesRegex(ValueError, "relative path"):
                replace(target, relative_path="../escape.png").validate()

    def test_flattens_detector_results_and_inconclusive_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._create_fixture(Path(directory))
            record = build_manifest(**fixture, samples_per_class=2)[0]
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

            row = result_row(
                record=record,
                runs=runs,
                manifest_sha256="a" * 64,
                environment_id="colab-t4-test",
                attempt=1,
                run_started_utc="2026-10-05T12:00:00+00:00",
                total_duration_seconds=0.5,
            )

            self.assertEqual(row["protocol_version"], PROTOCOL_VERSION)
            self.assertEqual(row["decision_reason"], "conflicting_evidence")
            self.assertEqual(row["final_label"], "")
            self.assertTrue(row["inconclusive"])
            self.assertEqual(row["dinolizer_processed_width"], 1016)
            self.assertEqual(row["localization_mode"], "F")

    def test_smoke_run_writes_incremental_results_and_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "input.png"
            Image.new("RGB", (4, 3), color="white").save(image_path)
            image_sha256 = file_digest(image_path)
            manifest_path = root / "manifest.csv"
            manifest_path.write_text("frozen manifest fixture\n", encoding="utf-8")
            output = root / "smoke.csv"
            records = tuple(
                ManifestRecord(
                    protocol_version=PROTOCOL_VERSION,
                    manifest_index=index,
                    sample_id=f"sample-{true_class}",
                    true_class=true_class,
                    source_dataset=(
                        "poloclub/diffusiondb"
                        if true_class == "synthetic"
                        else "nebula/CocoGlide"
                    ),
                    source_revision=(
                        "fb620fbe49fa4420e0734bd9c0df11f51176b61f"
                        if true_class == "synthetic"
                        else "275f045df7caa2544dd28de7fa86b044ab661bd0"
                    ),
                    source_partition="fixture",
                    source_id=f"source-{index}",
                    source_root=(
                        "diffusiondb" if true_class == "synthetic" else "cocoglide"
                    ),
                    relative_path="input.png",
                    source_metadata_sha256="b" * 64,
                    input_sha256=image_sha256,
                    width=4,
                    height=3,
                )
                for index, true_class in enumerate(
                    ("real", "synthetic", "edited"), start=1
                )
            )
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
                run_validation(args)

            with output.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, list(RESULT_FIELDS))
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["sample_id"] for row in rows}, set(resolved))

            metadata_path = output.with_suffix(".csv.metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["completed_samples"], 3)

            args.resume = True
            with self._patched_run_dependencies(records, resolved):
                run_validation(args)
            with output.open("r", encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)

    @contextmanager
    def _patched_run_dependencies(self, records, resolved):
        patches = (
            patch(
                "experiments.three_class_validation.load_manifest",
                return_value=records,
            ),
            patch(
                "experiments.three_class_validation.verify_manifest_inputs",
                return_value=resolved,
            ),
            patch(
                "experiments.three_class_validation._verify_model_artifacts",
                return_value={"verified": True},
            ),
            patch(
                "experiments.three_class_validation._software_environment",
                return_value={"test": True},
            ),
            patch(
                "experiments.three_class_validation._gpu_environment",
                return_value={"test": True},
            ),
            patch("experiments.three_class_validation._reset_gpu_peak_memory"),
            patch(
                "experiments.three_class_validation.DistilDIREDetector",
                return_value=FixedExperimentDetector(
                    "distildire", DetectionTarget.SYNTHETIC, True
                ),
            ),
            patch(
                "experiments.three_class_validation.DinoLizerDetector",
                return_value=FixedExperimentDetector(
                    "dinolizer", DetectionTarget.EDITED, False
                ),
            ),
        )
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            yield

    def _create_fixture(self, root: Path) -> dict[str, object]:
        cocoglide_root = root / "cocoglide"
        diffusiondb_root = root / "diffusiondb"
        cocoglide_root.mkdir()
        diffusiondb_root.mkdir()

        cocoglide_results = root / "cocoglide-results.csv"
        fields = (
            "split",
            "coco_id",
            "expected_class",
            "image_file",
            "original_width",
            "original_height",
        )
        rows = []
        for coco_id, color in (("000001", "red"), ("000002", "blue")):
            for expected_class in ("real", "manipulated"):
                filename = f"{coco_id}_{expected_class}.png"
                Image.new("RGB", (4, 3), color=color).save(cocoglide_root / filename)
                rows.append(
                    {
                        "split": "validation",
                        "coco_id": coco_id,
                        "expected_class": expected_class,
                        "image_file": filename,
                        "original_width": 4,
                        "original_height": 3,
                    }
                )
        with cocoglide_results.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

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
        for index, nsfw in ((1, 0.01), (2, 0.02), (3, 0.9)):
            filename = f"synthetic-{index}.png"
            Image.new("RGB", (4, 3), color=(index * 20, 10, 10)).save(
                diffusiondb_root / filename
            )
            diffusion_rows.append(
                {
                    "image_name": filename,
                    "part_id": 1079,
                    "width": 4,
                    "height": 3,
                    "image_nsfw": nsfw,
                    "prompt_nsfw": 0.01,
                }
            )
        with diffusiondb_metadata.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=diffusion_fields)
            writer.writeheader()
            writer.writerows(diffusion_rows)

        return {
            "cocoglide_results": cocoglide_results,
            "cocoglide_root": cocoglide_root,
            "diffusiondb_metadata": diffusiondb_metadata,
            "diffusiondb_root": diffusiondb_root,
            "verify_cocoglide_results": False,
        }


if __name__ == "__main__":
    unittest.main()
