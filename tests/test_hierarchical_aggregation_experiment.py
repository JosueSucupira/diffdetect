import csv
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from PIL import Image

from experiments.hierarchical_aggregation import (
    DIFFUSIONDB_PART,
    PROTOCOL_VERSION,
    build_calibration_manifest,
    file_digest,
    load_calibration_manifest,
    load_excluded_identities,
    resolve_calibration_path,
    verify_calibration_inputs,
    write_calibration_manifest,
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
