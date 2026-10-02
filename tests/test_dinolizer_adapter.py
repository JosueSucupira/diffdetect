import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

from PIL import Image

from diffdetect import (
    DetectionTarget,
    DetectorRegistry,
    DinoLizerDetector,
    DinoLizerInference,
)
from diffdetect.adapters.dinolizer import (
    _import_official_modules,
    _processed_size,
    _window_coordinates,
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

        with self.assertRaisesRegex(ValueError, "repository_path, checkpoint_path"):
            DinoLizerDetector()

        with TemporaryDirectory() as directory:
            repository = Path(directory)
            checkpoint = repository / "model.ckpt"
            checkpoint.write_bytes(b"checkpoint placeholder")

            with self.assertRaisesRegex(ValueError, "custom runtime"):
                DinoLizerDetector(
                    repository_path=repository,
                    checkpoint_path=checkpoint,
                    runtime=RecordingRuntime(),
                )

            for device in ("", "   ", None):
                with self.subTest(device=device):
                    with self.assertRaisesRegex(ValueError, "device"):
                        DinoLizerDetector(
                            repository_path=repository,
                            checkpoint_path=checkpoint,
                            device=device,
                        )

        with self.assertRaisesRegex(FileNotFoundError, "repository_path"):
            DinoLizerDetector(
                repository_path="missing-repository",
                checkpoint_path="missing-checkpoint",
            )

    def test_loads_official_runtime_lazily_and_reuses_it(self):
        runtime = RecordingRuntime()
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            checkpoint = repository / "model.ckpt"
            checkpoint.write_bytes(b"checkpoint placeholder")

            with patch(
                "diffdetect.adapters.dinolizer._TorchDinoLizerRuntime",
                return_value=runtime,
            ) as runtime_factory:
                detector = DinoLizerDetector(
                    repository_path=repository,
                    checkpoint_path=checkpoint,
                    device="cpu",
                )

                runtime_factory.assert_not_called()
                detector.predict(Image.new("RGB", (2, 2)))
                detector.predict(Image.new("RGB", (2, 2)))

                runtime_factory.assert_called_once_with(
                    repository.resolve(),
                    checkpoint.resolve(),
                    "cpu",
                )
                self.assertEqual(len(runtime.images), 2)

    def test_isolates_generic_module_names_used_by_external_repositories(self):
        previous_networks = ModuleType("networks")
        previous_utils = ModuleType("utils")
        original_path = list(sys.path)

        def fake_import(name: str) -> ModuleType:
            root = name.split(".", maxsplit=1)[0]
            sys.modules.setdefault(root, ModuleType(root))
            module = ModuleType(name)
            sys.modules[name] = module
            return module

        with patch.dict(
            sys.modules,
            {"networks": previous_networks, "utils": previous_utils},
        ):
            with patch(
                "diffdetect.adapters.dinolizer.importlib.import_module",
                side_effect=fake_import,
            ):
                lora_module, wrapper_module = _import_official_modules(
                    Path("/external/dinolizer")
                )

            self.assertEqual(lora_module.__name__, "networks.LORA")
            self.assertEqual(wrapper_module.__name__, "networks.wrapper5crops")
            self.assertIs(sys.modules["networks"], previous_networks)
            self.assertIs(sys.modules["utils"], previous_utils)
            self.assertNotIn("networks.LORA", sys.modules)
            self.assertNotIn("networks.wrapper5crops", sys.modules)

        self.assertEqual(sys.path, original_path)

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

    def test_matches_official_resize_geometry(self):
        self.assertEqual(_processed_size((256, 256)), (1016, 1016))
        self.assertEqual(_processed_size((512, 512)), (512, 512))

        for original_size in ((200, 800), (100, 1000), (1000, 100)):
            with self.subTest(original_size=original_size):
                processed_width, processed_height = _processed_size(original_size)
                self.assertGreaterEqual(processed_width, 504)
                self.assertGreaterEqual(processed_height, 504)
                processed_ratio = processed_width / processed_height
                original_ratio = original_size[0] / original_size[1]
                self.assertLess(
                    abs(processed_ratio - original_ratio) / original_ratio,
                    0.002,
                )

        for invalid_size in ((0, 20), (20, -1), (20.0, 20), [20, 20]):
            with self.subTest(invalid_size=invalid_size):
                with self.assertRaises(ValueError):
                    _processed_size(invalid_size)

    def test_sliding_windows_cover_edges_without_duplicates(self):
        self.assertEqual(_window_coordinates((504, 504)), ((0, 0, 504, 504),))

        coordinates = _window_coordinates((1016, 1016))
        self.assertEqual(len(coordinates), 25)
        self.assertEqual(coordinates[0], (0, 0, 504, 504))
        self.assertEqual(coordinates[-1], (512, 512, 1016, 1016))
        self.assertEqual(len(set(coordinates)), len(coordinates))

        irregular = _window_coordinates((600, 700))
        self.assertIn((96, 196, 600, 700), irregular)
        self.assertEqual(len(set(irregular)), len(irregular))

        invalid_arguments = (
            ((503, 504), {}),
            ((504, 504), {"window_size": 0}),
            ((504, 504), {"stride": 0}),
        )
        for image_size, arguments in invalid_arguments:
            with self.subTest(image_size=image_size, arguments=arguments):
                with self.assertRaises(ValueError):
                    _window_coordinates(image_size, **arguments)


if __name__ == "__main__":
    unittest.main()
