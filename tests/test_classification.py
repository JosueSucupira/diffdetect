import unittest

from PIL import Image

from diffdetect import (
    BaseDetector,
    DecisionReason,
    DetectionResult,
    DetectionTarget,
    DetectorRun,
    DetectorRunner,
    ImageClass,
    classify,
)


def successful_run(name, target, detected, *, score=None):
    return DetectorRun(
        name=name,
        target=target,
        duration_seconds=0.01,
        result=DetectionResult(detected=detected, score=score),
    )


class FixedDetector(BaseDetector):
    def __init__(self, name, target, detected):
        self.name = name
        self.target = target
        self.detected = detected

    def predict(self, image):
        return DetectionResult(detected=self.detected)


class ClassificationTests(unittest.TestCase):
    def test_three_classes_and_conflict(self):
        cases = (
            (False, False, ImageClass.REAL, DecisionReason.NO_EVIDENCE),
            (True, False, ImageClass.SYNTHETIC, DecisionReason.SYNTHETIC_EVIDENCE),
            (False, True, ImageClass.EDITED, DecisionReason.EDITED_EVIDENCE),
            (True, True, None, DecisionReason.CONFLICTING_EVIDENCE),
        )
        for synthetic, edited, expected_label, expected_reason in cases:
            with self.subTest(synthetic=synthetic, edited=edited):
                runs = (
                    successful_run("synth", DetectionTarget.SYNTHETIC, synthetic),
                    successful_run("edit", DetectionTarget.EDITED, edited),
                )
                decision = classify(runs)
                self.assertEqual(decision.label, expected_label)
                self.assertEqual(decision.reason, expected_reason)
                self.assertEqual(decision.runs, runs)

    def test_missing_target_is_inconclusive_even_with_positive_evidence(self):
        decision = classify((successful_run("synth", DetectionTarget.SYNTHETIC, True),))
        self.assertIsNone(decision.label)
        self.assertEqual(decision.reason, DecisionReason.INSUFFICIENT_COVERAGE)

    def test_detector_error_is_inconclusive(self):
        runs = (
            successful_run("synth", DetectionTarget.SYNTHETIC, True),
            DetectorRun("edit", DetectionTarget.EDITED, 0.01, error="RuntimeError: failed"),
        )
        decision = classify(runs)
        self.assertIsNone(decision.label)
        self.assertEqual(decision.reason, DecisionReason.DETECTOR_ERROR)

    def test_any_positive_per_target_without_comparing_raw_scores(self):
        runs = (
            successful_run("synth_a", DetectionTarget.SYNTHETIC, False, score=1000),
            successful_run("synth_b", DetectionTarget.SYNTHETIC, True, score=-1000),
            successful_run("edit", DetectionTarget.EDITED, False, score=1000),
        )
        decision = classify(runs)
        self.assertEqual(decision.label, ImageClass.SYNTHETIC)

    def test_runner_output_can_be_classified(self):
        runner = DetectorRunner(
            [
                FixedDetector("synth", DetectionTarget.SYNTHETIC, False),
                FixedDetector("edit", DetectionTarget.EDITED, True),
            ]
        )
        decision = classify(runner.run(Image.new("RGB", (2, 2))))
        self.assertEqual(decision.label, ImageClass.EDITED)


if __name__ == "__main__":
    unittest.main()
