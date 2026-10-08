import tempfile
import unittest
from pathlib import Path

from experiments.hierarchical_retrospective import (
    evaluate_retrospective,
    file_digest,
    paired_group_bootstrap,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/experiments/results"
POLICY = RESULTS / "hierarchical-policy.json"


class HierarchicalRetrospectiveTests(unittest.TestCase):
    def test_paired_group_bootstrap_is_deterministic(self):
        values = {
            "true_labels": ("real", "edited", "synthetic") * 4,
            "baseline_predictions": (None, None, "edited") * 4,
            "hierarchical_predictions": ("real", "edited", "synthetic") * 4,
            "group_ids": tuple(f"group-{index // 2}" for index in range(12)),
            "resamples": 20,
            "seed": 123,
        }
        first = paired_group_bootstrap(**values)
        second = paired_group_bootstrap(**values)
        self.assertEqual(first, second)
        self.assertEqual(first["resamples"], 20)
        self.assertGreater(
            first["differences_hierarchical_minus_boolean"]["macro_f1"]["observed"],
            0.0,
        )

    def test_evaluates_frozen_policy_without_refitting(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = evaluate_retrospective(
                policy_path=POLICY,
                expected_policy_sha256=file_digest(POLICY),
                baseline_results_path=RESULTS / "three-class-validation-clean-cache.csv",
                baseline_metadata_path=(
                    RESULTS / "three-class-validation-clean-cache.csv.metadata.json"
                ),
                output_dir=directory,
            )
            self.assertEqual(metrics["sample_count"], 768)
            self.assertEqual(metrics["aggregation_errors"], 0)
            self.assertEqual(
                metrics["boolean_baseline"]["end_to_end"]["coverage"],
                0.40625,
            )
            self.assertEqual(
                metrics["boolean_baseline"]["end_to_end"]["accuracy"],
                0.06640625,
            )
            self.assertTrue(
                metrics["source"][
                    "policy_calibration_previous_validation_used_for_fitting"
                ]
                is False
            )
            self.assertTrue(
                (Path(directory) / "hierarchical-retrospective-predictions.csv").is_file()
            )

    def test_rejects_wrong_policy_hash_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "policy SHA-256"):
                evaluate_retrospective(
                    policy_path=POLICY,
                    expected_policy_sha256="0" * 64,
                    baseline_results_path=(
                        RESULTS / "three-class-validation-clean-cache.csv"
                    ),
                    baseline_metadata_path=(
                        RESULTS / "three-class-validation-clean-cache.csv.metadata.json"
                    ),
                    output_dir=directory,
                )


if __name__ == "__main__":
    unittest.main()
