import unittest

from PIL import Image

from diffdetect import DetectionTarget, DetectorRegistry, DistilDIREDetector


class RecordingRuntime:
    def __init__(self, score: float):
        self.output_score = score
        self.images: list[Image.Image] = []

    def score(self, image: Image.Image) -> float:
        self.images.append(image)
        return self.output_score


class DistilDIREAdapterTests(unittest.TestCase):
    def test_returns_synthetic_evidence_with_configured_threshold(self):
        runtime = RecordingRuntime(0.99)
        detector = DistilDIREDetector(threshold=0.98, runtime=runtime)

        result = detector.predict(Image.new("L", (12, 8)))

        self.assertEqual(detector.name, "distildire")
        self.assertEqual(detector.target, DetectionTarget.SYNTHETIC)
        self.assertTrue(result.detected)
        self.assertEqual(result.score, 0.99)
        self.assertEqual(result.threshold, 0.98)
        self.assertEqual(runtime.images[0].mode, "RGB")
        self.assertEqual(runtime.images[0].size, (12, 8))

    def test_score_below_threshold_is_not_synthetic_evidence(self):
        detector = DistilDIREDetector(
            threshold=0.987782,
            runtime=RecordingRuntime(0.91),
        )

        result = detector.predict(Image.new("RGB", (4, 4)))

        self.assertFalse(result.detected)

    def test_rejects_invalid_probability_boundaries(self):
        for threshold in (-0.1, 1.1, float("nan")):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    DistilDIREDetector(
                        threshold=threshold,
                        runtime=RecordingRuntime(0.5),
                    )

        detector = DistilDIREDetector(
            threshold=0.5,
            runtime=RecordingRuntime(float("inf")),
        )
        with self.assertRaises(ValueError):
            detector.predict(Image.new("RGB", (2, 2)))

    def test_requires_complete_model_configuration_without_runtime(self):
        with self.assertRaisesRegex(ValueError, "repository_path"):
            DistilDIREDetector(threshold=0.5)

    def test_can_be_constructed_by_the_registry(self):
        registry = DetectorRegistry()
        registry.register(DistilDIREDetector)

        detector = registry.create(
            "distildire",
            threshold=0.8,
            runtime=RecordingRuntime(0.7),
        )

        self.assertIsInstance(detector, DistilDIREDetector)
        self.assertFalse(detector.predict(Image.new("RGB", (2, 2))).detected)


if __name__ == "__main__":
    unittest.main()
