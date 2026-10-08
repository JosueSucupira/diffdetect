import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from diffdetect import (
    DecisionReason,
    DetectionResult,
    DetectionTarget,
    DetectorRun,
    HierarchicalDecisionReason,
    HierarchicalPolicy,
    ImageClass,
    LogisticPolicyModel,
    classify,
    classify_hierarchical,
    load_hierarchical_policy,
)


def policy():
    return HierarchicalPolicy(
        policy_version="test-policy-v1",
        feature_names=(
            "distildire_score",
            "dinolizer_score",
            "dinolizer_marked_area",
        ),
        feature_means=(0.0, 0.0, 0.0),
        feature_scales=(1.0, 1.0, 1.0),
        level1=LogisticPolicyModel((10.0, 0.0, 0.0), -5.0),
        level2=LogisticPolicyModel((0.0, 0.0, -10.0), 5.0),
        level1_threshold=0.8,
        level2_threshold=0.8,
        artifact_sha256="a" * 64,
    )


def runs(distildire_score, dinolizer_score=0.5, marked_area=0.5):
    return (
        DetectorRun(
            "distildire",
            DetectionTarget.SYNTHETIC,
            0.1,
            result=DetectionResult(
                detected=distildire_score >= 0.5,
                score=distildire_score,
                threshold=0.5,
            ),
        ),
        DetectorRun(
            "dinolizer",
            DetectionTarget.EDITED,
            0.1,
            result=DetectionResult(
                detected=dinolizer_score >= 0.5,
                score=dinolizer_score,
                threshold=0.5,
                metadata={"marked_area": marked_area},
            ),
        ),
    )


class HierarchicalClassificationTests(unittest.TestCase):
    def test_returns_all_classes_and_both_abstention_stages(self):
        cases = (
            ((0.1, 0.5, 0.5), ImageClass.REAL, HierarchicalDecisionReason.LEVEL1_REAL),
            ((0.5, 0.5, 0.5), None, HierarchicalDecisionReason.LEVEL1_INCONCLUSIVE),
            (
                (0.9, 0.5, 0.1),
                ImageClass.SYNTHETIC,
                HierarchicalDecisionReason.LEVEL2_SYNTHETIC,
            ),
            (
                (0.9, 0.5, 0.9),
                ImageClass.EDITED,
                HierarchicalDecisionReason.LEVEL2_EDITED,
            ),
            ((0.9, 0.5, 0.5), None, HierarchicalDecisionReason.LEVEL2_INCONCLUSIVE),
        )
        for features, expected_label, expected_reason in cases:
            with self.subTest(features=features):
                decision = classify_hierarchical(runs(*features), policy())
                self.assertEqual(decision.label, expected_label)
                self.assertEqual(decision.reason, expected_reason)
                self.assertEqual(decision.policy_sha256, "a" * 64)
                self.assertEqual(decision.runs, runs(*features))

    def test_rejects_errors_missing_targets_and_invalid_features(self):
        detector_error = (
            DetectorRun(
                "distildire",
                DetectionTarget.SYNTHETIC,
                0.1,
                error="RuntimeError: failed",
            ),
            runs(0.9)[1],
        )
        self.assertEqual(
            classify_hierarchical(detector_error, policy()).reason,
            HierarchicalDecisionReason.DETECTOR_ERROR,
        )
        self.assertEqual(
            classify_hierarchical((runs(0.9)[0],), policy()).reason,
            HierarchicalDecisionReason.INSUFFICIENT_COVERAGE,
        )
        invalid = (
            runs(0.9)[0],
            DetectorRun(
                "dinolizer",
                DetectionTarget.EDITED,
                0.1,
                result=DetectionResult(True, score=0.8, metadata={}),
            ),
        )
        self.assertEqual(
            classify_hierarchical(invalid, policy()).reason,
            HierarchicalDecisionReason.INVALID_FEATURES,
        )

    def test_probability_evaluation_is_stable_for_large_logits(self):
        extreme = HierarchicalPolicy(
            **{
                **policy().__dict__,
                "level1": LogisticPolicyModel((10000.0, 0.0, 0.0), -5000.0),
            }
        )
        self.assertEqual(
            classify_hierarchical(runs(0.0), extreme).label,
            ImageClass.REAL,
        )
        self.assertEqual(
            classify_hierarchical(runs(1.0, marked_area=0.0), extreme).label,
            ImageClass.SYNTHETIC,
        )

    def test_loads_policy_and_binds_exact_file_hash(self):
        payload = {
            "schema_version": "diffdetect-hierarchical-policy-v1",
            "policy_version": "test-policy-v1",
            "feature_names": [
                "distildire_score",
                "dinolizer_score",
                "dinolizer_marked_area",
            ],
            "scaler": {"means": [0.0, 0.0, 0.0], "scales": [1.0, 1.0, 1.0]},
            "models": {
                "level1": {"coefficients": [10.0, 0.0, 0.0], "intercept": -5.0},
                "level2": {"coefficients": [0.0, 0.0, -10.0], "intercept": 5.0},
            },
            "confidence_thresholds": {"level1": 0.8, "level2": 0.8},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_hierarchical_policy(path)
            self.assertEqual(
                loaded.artifact_sha256,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_boolean_classifier_remains_independent(self):
        evidence = runs(0.9, dinolizer_score=0.9, marked_area=0.1)
        hierarchical = classify_hierarchical(evidence, policy())
        boolean = classify(evidence)
        self.assertEqual(hierarchical.label, ImageClass.SYNTHETIC)
        self.assertIsNone(boolean.label)
        self.assertEqual(boolean.reason, DecisionReason.CONFLICTING_EVIDENCE)


if __name__ == "__main__":
    unittest.main()
