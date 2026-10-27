"""Unit tests for the unified DiffDetect pipeline and factory."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from diffdetect import (
    BaseDetector,
    DetectionResult,
    DetectionTarget,
    DiffDetect,
    DiffDetectResult,
    ImageClass,
    create_detector,
    load_default_hierarchical_policy,
)
from diffdetect.adapters.dinolizer import DinoLizerInference


class FakeSyntheticDetector(BaseDetector):
    name = "distildire"
    target = DetectionTarget.SYNTHETIC

    def __init__(self, score: float = 0.99) -> None:
        self.score = score

    def predict(self, image: Image.Image) -> DetectionResult:
        return DetectionResult(
            detected=self.score >= 0.98,
            score=self.score,
            threshold=0.98,
        )


class FakeEditedDetector(BaseDetector):
    name = "dinolizer"
    target = DetectionTarget.EDITED

    def __init__(self, score: float = 0.85, marked_area: float = 0.15) -> None:
        self.score = score
        self.marked_area = marked_area

    def predict(self, image: Image.Image) -> DetectionResult:
        localization = Image.new("F", image.size, color=0.5)
        return DetectionResult(
            detected=self.score >= 0.59,
            score=self.score,
            threshold=0.59,
            localization_map=localization,
            metadata={"marked_area": self.marked_area},
        )


class PipelineTests(unittest.TestCase):
    def test_load_default_policy(self) -> None:
        policy = load_default_hierarchical_policy()
        self.assertEqual(policy.policy_version, "diffdetect-hierarchical-aggregation-v1")
        self.assertEqual(
            policy.artifact_sha256,
            "c1214e33d698daa7cda082cd2c20ef0e4553d8d63b46ac77494e9685a50a1187",
        )

    def test_create_detector_with_custom_runtime(self) -> None:
        mock_runtime = MagicMock()
        mock_runtime.score.return_value = 0.95

        detector = create_detector("distildire", threshold=0.9, runtime=mock_runtime)
        self.assertEqual(detector.name, "distildire")
        self.assertEqual(detector.target, DetectionTarget.SYNTHETIC)

        result = detector.predict(Image.new("RGB", (10, 10)))
        self.assertTrue(result.detected)
        self.assertEqual(result.score, 0.95)

        mock_dino_runtime = MagicMock()
        mock_dino_runtime.infer.return_value = DinoLizerInference(
            score=0.8,
            localization_map=Image.new("F", (10, 10)),
            marked_area=0.1,
            processed_size=(1016, 1016),
            window_count=1,
        )
        dino = create_detector("dinolizer", threshold=0.5, runtime=mock_dino_runtime)
        self.assertEqual(dino.name, "dinolizer")
        self.assertEqual(dino.target, DetectionTarget.EDITED)

        dino_res = dino.predict(Image.new("RGB", (10, 10)))
        self.assertTrue(dino_res.detected)

        with self.assertRaises(KeyError):
            create_detector("unsupported_detector")

    def test_diffdetect_predict_single_and_batch(self) -> None:
        synth_det = FakeSyntheticDetector(score=0.99)
        edit_det = FakeEditedDetector(score=0.85, marked_area=0.25)

        pipeline = DiffDetect(detectors=[synth_det, edit_det])
        img = Image.new("RGB", (32, 32))

        result = pipeline.predict(img)
        self.assertIsInstance(result, DiffDetectResult)
        self.assertIn(result.label, (ImageClass.REAL, ImageClass.SYNTHETIC, ImageClass.EDITED, None))
        self.assertIsNotNone(result.reason)
        self.assertIsInstance(result.p_artificial, float)
        self.assertIsNotNone(result.localization_map)
        self.assertEqual(result.marked_area, 0.25)
        self.assertIn("distildire", result.scores)
        self.assertIn("dinolizer", result.scores)

        batch_results = pipeline.predict_batch([img, img])
        self.assertEqual(len(batch_results), 2)
        self.assertEqual(batch_results[0].label, result.label)

    def test_diffdetect_rejects_invalid_policy(self) -> None:
        with self.assertRaises(TypeError):
            DiffDetect(policy=12345)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
