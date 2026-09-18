import unittest

from PIL import Image

from diffdetect import BaseDetector, DetectionResult, DetectionTarget, DetectorRegistry


class SyntheticAdapter(BaseDetector):
    name = "synthetic_example"
    target = DetectionTarget.SYNTHETIC

    def __init__(self, cutoff: float = 0.5):
        self.cutoff = cutoff

    def predict(self, image: Image.Image) -> DetectionResult:
        return DetectionResult(detected=self.cutoff < 0.7, threshold=self.cutoff, score=0.7)


class NoTargetAdapter(BaseDetector):
    name = "no_target"

    def predict(self, image: Image.Image) -> DetectionResult:
        return DetectionResult(detected=False)


class IncompleteAdapter(BaseDetector):
    name = "incomplete"
    target = DetectionTarget.EDITED


class RegistryTests(unittest.TestCase):
    def test_register_list_and_create_with_configuration(self):
        registry = DetectorRegistry()
        registry.register(SyntheticAdapter)

        self.assertEqual(registry.names(), ("synthetic_example",))
        detector = registry.create("synthetic_example", cutoff=0.8)
        self.assertIsInstance(detector, SyntheticAdapter)
        self.assertEqual(detector.cutoff, 0.8)
        self.assertFalse(detector.predict(Image.new("RGB", (2, 2))).detected)

    def test_registry_instances_are_independent(self):
        first = DetectorRegistry()
        second = DetectorRegistry()
        first.register(SyntheticAdapter)

        self.assertEqual(first.names(), ("synthetic_example",))
        self.assertEqual(second.names(), ())

    def test_rejects_duplicate_and_invalid_adapters(self):
        registry = DetectorRegistry()
        registry.register(SyntheticAdapter)
        with self.assertRaises(ValueError):
            registry.register(SyntheticAdapter)
        with self.assertRaises(ValueError):
            registry.register(NoTargetAdapter)
        with self.assertRaises(TypeError):
            registry.register(IncompleteAdapter)
        with self.assertRaises(TypeError):
            registry.register(object)

    def test_unknown_name_is_explicit(self):
        with self.assertRaisesRegex(KeyError, "unknown detector: missing"):
            DetectorRegistry().create("missing")


if __name__ == "__main__":
    unittest.main()
