import tempfile
import unittest
from pathlib import Path

from PIL import Image

from diffdetect import BaseDetector, DetectionResult, DetectionTarget, DetectorRunner


class RecordingDetector(BaseDetector):
    def __init__(self, name, target, *, fail=False, mutate=False):
        self.name = name
        self.target = target
        self.fail = fail
        self.mutate = mutate
        self.calls = 0
        self.seen_pixel = None

    def predict(self, image):
        self.calls += 1
        if image.mode != "RGB":
            raise ValueError("image was not RGB")
        self.seen_pixel = image.getpixel((0, 0))
        if self.mutate:
            image.putpixel((0, 0), (0, 0, 0))
        if self.fail:
            raise RuntimeError("model unavailable")
        return DetectionResult(detected=False, score=0.2)


class WrongOutputDetector(RecordingDetector):
    def predict(self, image):
        return "not a DetectionResult"


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.synthetic = RecordingDetector("synthetic", DetectionTarget.SYNTHETIC)
        self.edited = RecordingDetector("edited", DetectionTarget.EDITED)
        self.image = Image.new("L", (4, 4), color=255)

    def test_selected_detector_receives_rgb_image_from_file(self):
        runner = DetectorRunner([self.synthetic, self.edited])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            self.image.save(path)
            runs = runner.run(path, selected=["edited"])

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].name, "edited")
        self.assertEqual(runs[0].status, "success")
        self.assertGreaterEqual(runs[0].duration_seconds, 0)
        self.assertEqual(self.synthetic.calls, 0)
        self.assertEqual(self.edited.seen_pixel, (255, 255, 255))

    def test_detectors_cannot_change_each_others_input(self):
        mutating = RecordingDetector("mutating", DetectionTarget.SYNTHETIC, mutate=True)
        runner = DetectorRunner([mutating, self.edited])
        runs = runner.run(self.image)

        self.assertEqual([run.status for run in runs], ["success", "success"])
        self.assertEqual(self.edited.seen_pixel, (255, 255, 255))
        self.assertEqual(self.image.getpixel((0, 0)), 255)

    def test_failure_is_recorded_without_stopping_other_detectors(self):
        failing = RecordingDetector("failing", DetectionTarget.SYNTHETIC, fail=True)
        runs = DetectorRunner([failing, self.edited]).run(self.image)

        self.assertEqual([run.status for run in runs], ["error", "success"])
        self.assertIn("model unavailable", runs[0].error)
        self.assertIsNone(runs[0].result)
        self.assertIsNotNone(runs[1].result)

    def test_invalid_output_and_selection_are_explicit(self):
        wrong = WrongOutputDetector("wrong", DetectionTarget.SYNTHETIC)
        runner = DetectorRunner([wrong, self.edited])
        self.assertIn("TypeError", runner.run(self.image, selected="wrong")[0].error)
        with self.assertRaises(KeyError):
            runner.run(self.image, selected=["missing"])
        with self.assertRaises(ValueError):
            runner.run(self.image, selected=["edited", "edited"])

    def test_unreadable_image_never_runs_a_detector(self):
        runner = DetectorRunner([self.synthetic])
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.png"
            with self.assertRaises(FileNotFoundError):
                runner.run(missing)
        self.assertEqual(self.synthetic.calls, 0)


if __name__ == "__main__":
    unittest.main()
