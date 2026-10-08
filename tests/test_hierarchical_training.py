import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.hierarchical_training import (
    CALIBRATION_MANIFEST_SHA256,
    CALIBRATION_RESULTS_SHA256,
    CV_FOLDS,
    HIERARCHICAL_FEATURES,
    fit_and_write_policy,
    fit_variant,
    grouped_fold_assignments,
    load_calibration_evidence,
    select_thresholds,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/experiments/results"


class HierarchicalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_calibration_evidence(
            results_path=RESULTS / "hierarchical-calibration.csv",
            metadata_path=RESULTS / "hierarchical-calibration.csv.metadata.json",
            manifest_path=RESULTS / "hierarchical-calibration-manifest.csv",
        )

    def test_loads_complete_frozen_evidence_and_validates_hashes(self):
        self.assertEqual(self.bundle.results_sha256, CALIBRATION_RESULTS_SHA256)
        self.assertEqual(self.bundle.manifest_sha256, CALIBRATION_MANIFEST_SHA256)
        self.assertEqual(len(self.bundle.rows), 768)
        self.assertEqual(
            Counter(row.true_class for row in self.bundle.rows),
            {"real": 256, "synthetic": 256, "edited": 256},
        )
        self.assertTrue(
            all(len(row.features) == len(HIERARCHICAL_FEATURES) for row in self.bundle.rows)
        )

    def test_grouped_folds_keep_cocoglide_pairs_together(self):
        assignments, indexes = grouped_fold_assignments(self.bundle.rows)
        self.assertEqual(len(indexes), CV_FOLDS)
        self.assertEqual(set(assignments), set(range(1, CV_FOLDS + 1)))

        group_folds = {}
        for row, fold in zip(self.bundle.rows, assignments):
            group_folds.setdefault(row.group_id, set()).add(fold)
        self.assertTrue(all(len(folds) == 1 for folds in group_folds.values()))

        for fold in range(1, CV_FOLDS + 1):
            classes = Counter(
                row.true_class
                for row, assigned in zip(self.bundle.rows, assignments)
                if assigned == fold
            )
            self.assertEqual(set(classes), {"real", "synthetic", "edited"})

    def test_threshold_selection_uses_smallest_pair_after_metric_ties(self):
        labels = ("real", "synthetic", "edited") * 4
        p_artificial = tuple(
            0.1 if label == "real" else 0.9 for label in labels
        )
        p_level2 = tuple(
            0.9 if label == "synthetic" else 0.1 for label in labels
        )
        _, selected = select_thresholds(labels, p_artificial, p_level2)
        self.assertEqual(selected["macro_f1"], 1.0)
        self.assertEqual(selected["level1_threshold"], 0.5)
        self.assertEqual(selected["level2_threshold"], 0.5)

    def test_fits_deterministically_and_writes_loadable_artifacts(self):
        assignments, indexes = grouped_fold_assignments(self.bundle.rows)
        first = fit_variant(
            self.bundle.rows,
            feature_names=HIERARCHICAL_FEATURES,
            fold_indexes=indexes,
            fold_assignments=assignments,
        )
        second = fit_variant(
            self.bundle.rows,
            feature_names=HIERARCHICAL_FEATURES,
            fold_indexes=indexes,
            fold_assignments=assignments,
        )
        self.assertEqual(first.selected_thresholds, second.selected_thresholds)
        self.assertEqual(first.p_artificial, second.p_artificial)
        self.assertEqual(first.p_level2, second.p_level2)
        self.assertGreaterEqual(
            first.selected_metrics["end_to_end"]["coverage"],
            0.80,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {
                "policy_output": root / "policy.json",
                "oof_output": root / "oof.csv",
                "folds_output": root / "folds.csv",
                "thresholds_output": root / "thresholds.csv",
                "metrics_output": root / "metrics.json",
                "checksums_output": root / "artifacts.sha256",
            }
            metrics = fit_and_write_policy(bundle=self.bundle, **outputs)
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            self.assertEqual(
                metrics["primary"]["selected_thresholds"],
                {
                    "level1": first.selected_thresholds[0],
                    "level2": first.selected_thresholds[1],
                },
            )


if __name__ == "__main__":
    unittest.main()
