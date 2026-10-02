import unittest

from PIL import Image

from diffdetect import BaseDetector, DetectionResult, DetectionTarget


class IncompleteDetector(BaseDetector):
    name = "incomplete"
    target = DetectionTarget.SYNTHETIC


class ExampleDetector(BaseDetector):
    name = "example"
    target = DetectionTarget.EDITED

    def predict(self, image: Image.Image) -> DetectionResult:
        return DetectionResult(
            detected=True,
            score=-0.4,
            threshold=-0.5,
            localization_map=Image.new("L", image.size),
        )


class DetectorContractTests(unittest.TestCase):
    def test_adapter_must_implement_predict(self):
        with self.assertRaises(TypeError):
            IncompleteDetector()

    def test_result_preserves_detector_specific_evidence(self):
        result = ExampleDetector().predict(Image.new("RGB", (8, 8)))
        self.assertTrue(result.detected)
        self.assertEqual(result.score, -0.4)
        self.assertEqual(result.threshold, -0.5)
        self.assertEqual(result.localization_map.size, (8, 8))
        self.assertEqual(result.metadata, {})

    def test_result_copies_detector_specific_metadata(self):
        metadata = {
            "marked_area": 0.25,
            "processed_size": (1016, 1016),
            "window_count": 25,
        }

        result = DetectionResult(detected=True, metadata=metadata)
        metadata["window_count"] = 1

        self.assertEqual(result.metadata["marked_area"], 0.25)
        self.assertEqual(result.metadata["processed_size"], (1016, 1016))
        self.assertEqual(result.metadata["window_count"], 25)

    def test_result_rejects_invalid_metadata(self):
        with self.assertRaisesRegex(TypeError, "metadata must be a mapping"):
            DetectionResult(detected=True, metadata=[("window_count", 25)])
        with self.assertRaisesRegex(TypeError, "metadata keys must be strings"):
            DetectionResult(detected=True, metadata={1: "invalid"})

    def test_invalid_decisions_cannot_cross_the_adapter_boundary(self):
        with self.assertRaises(TypeError):
            DetectionResult(detected="yes")
        with self.assertRaises(ValueError):
            DetectionResult(detected=True, score=float("nan"))
        with self.assertRaises(ValueError):
            DetectionResult(detected=True, threshold=0.5)


if __name__ == "__main__":
    unittest.main()
