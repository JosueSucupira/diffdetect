import unittest

from PIL import Image

from diffdetect import (
    DetectionTarget,
    DetectorRegistry,
    DinoLizerDetector,
    DinoLizerInference,
)


class RecordingRuntime:
    def __init__(
        self,
        *,
        score: float = 0.75,
        marked_area: float = 0.2,
        processed_size: tuple[int, int] = (1016, 1016),
        window_count: int = 25,
    ) -> None:
        self.score = score
        self.marked_area = marked_area
        self.processed_size = processed_size
        self.window_count = window_count
        self.images: list[Image.Image] = []

    def infer(self, image: Image.Image) -> DinoLizerInference:
        self.images.append(image)
        return DinoLizerInference(
            score=self.score,
            localization_map=Image.new("F", image.size, color=0.25),
            marked_area=self.marked_area,
            processed_size=self.processed_size,
            window_count=self.window_count,
        )


class StaticRuntime:
    def __init__(self, output: object) -> None:
        self.output = output

    def infer(self, image: Image.Image) -> object:
        return self.output


class DinoLizerAdapterTests(unittest.TestCase):
    def test_returns_edited_evidence_and_metadata(self):
        runtime = RecordingRuntime()
        detector = DinoLizerDetector(runtime=runtime)

        result = detector.predict(Image.new("L", (12, 8)))

        self.assertEqual(detector.name, "dinolizer")
        self.assertEqual(detector.target, DetectionTarget.EDITED)
        self.assertTrue(result.detected)
        self.assertEqual(result.score, 0.75)
        self.assertEqual(result.threshold, 0.5946570634841919)
        self.assertEqual(result.localization_map.mode, "F")
        self.assertEqual(result.localization_map.size, (12, 8))
        self.assertEqual(runtime.images[0].mode, "RGB")
        self.assertEqual(runtime.images[0].size, (12, 8))
        self.assertEqual(
            result.metadata,
            {
                "score_name": "p99_probability",
                "marked_area": 0.2,
                "original_size": (12, 8),
                "processed_size": (1016, 1016),
                "window_count": 25,
                "window_size": 504,
                "stride": 128,
                "pixel_threshold": 0.5,
            },
        )

    def test_uses_configurable_inclusive_threshold(self):
        equal = DinoLizerDetector(
            runtime=RecordingRuntime(score=0.8),
            threshold=0.8,
        )
        below = DinoLizerDetector(
            runtime=RecordingRuntime(score=0.79),
            threshold=0.8,
        )

        self.assertTrue(equal.predict(Image.new("RGB", (2, 2))).detected)
        self.assertFalse(below.predict(Image.new("RGB", (2, 2))).detected)

    def test_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(TypeError, "runtime"):
            DinoLizerDetector(runtime=object())

        for threshold in (-0.1, 1.1, float("nan"), True):
            with self.subTest(threshold=threshold):
                with self.assertRaises((TypeError, ValueError)):
                    DinoLizerDetector(
                        runtime=RecordingRuntime(),
                        threshold=threshold,
                    )

    def test_rejects_invalid_runtime_output(self):
        detector = DinoLizerDetector(runtime=StaticRuntime(object()))

        with self.assertRaisesRegex(TypeError, "DinoLizerInference"):
            detector.predict(Image.new("RGB", (2, 2)))

        invalid_values = (
            {"score": float("inf")},
            {"marked_area": -0.1},
            {"processed_size": (1016, 0)},
            {"window_count": 0},
        )
        for changes in invalid_values:
            with self.subTest(changes=changes):
                values = {
                    "score": 0.7,
                    "localization_map": Image.new("F", (2, 2), color=0.5),
                    "marked_area": 0.2,
                    "processed_size": (1016, 1016),
                    "window_count": 25,
                }
                values.update(changes)
                detector = DinoLizerDetector(
                    runtime=StaticRuntime(DinoLizerInference(**values))
                )
                with self.assertRaises((TypeError, ValueError)):
                    detector.predict(Image.new("RGB", (2, 2)))

    def test_rejects_invalid_localization_map(self):
        nan_map = Image.new("F", (2, 2), color=0.5)
        nan_map.putpixel((0, 0), float("nan"))
        invalid_maps = (
            Image.new("L", (2, 2)),
            Image.new("F", (3, 2), color=0.5),
            Image.new("F", (2, 2), color=1.1),
            nan_map,
        )
        for localization_map in invalid_maps:
            with self.subTest(mode=localization_map.mode, size=localization_map.size):
                output = DinoLizerInference(
                    score=0.7,
                    localization_map=localization_map,
                    marked_area=0.2,
                    processed_size=(1016, 1016),
                    window_count=25,
                )
                detector = DinoLizerDetector(runtime=StaticRuntime(output))
                with self.assertRaises((TypeError, ValueError)):
                    detector.predict(Image.new("RGB", (2, 2)))

    def test_can_be_constructed_by_the_registry(self):
        registry = DetectorRegistry()
        registry.register(DinoLizerDetector)

        detector = registry.create("dinolizer", runtime=RecordingRuntime())

        self.assertIsInstance(detector, DinoLizerDetector)


if __name__ == "__main__":
    unittest.main()
