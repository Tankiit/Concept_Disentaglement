import unittest

import numpy as np
import pandas as pd

from real.acquisition_gap import (DATASETS, PRED, exact_errors, heldout_population,
                                  out_of_fold_picks, t_star, validate_evaluator)


class EvaluatorTests(unittest.TestCase):
    def test_reproduces_frozen_heldout_rows(self):
        """Exact one-step errors must match heldout_exact_one_step.csv (8 rows)."""
        for spec in DATASETS.values():
            pop, N = heldout_population(spec)
            validate_evaluator(spec, pop, N, t_star(pop, N))     # asserts internally

    def test_label_channel_is_stratified_half_half(self):
        """Label error must not depend on the held-out source sizes (weight 1/2 each)."""
        sh = [np.array([[3.0, 5.0], [4.0, 6.0], [7.0, 2.0]])]
        a = exact_errors(sh, {0: dict(n=10.0, rate=0.3), 1: dict(n=90.0, rate=0.7)}, 100.0, 0.5)
        b = exact_errors(sh, {0: dict(n=90.0, rate=0.3), 1: dict(n=10.0, rate=0.7)}, 100.0, 0.5)
        self.assertAlmostEqual(a[0, 1], b[0, 1], places=15)       # label: size-invariant
        self.assertNotAlmostEqual(a[0, 0], b[0, 0], places=6)     # identity: size-dependent


class PickRuleTests(unittest.TestCase):
    def _frame(self):
        rng = np.random.default_rng(0)
        rows = []
        for cell, label_best in (((0, 2), True), ((512, 2), False)):
            for _ in range(20):
                e_id, e_lab = (0.2, 0.1) if label_best else (0.1, 0.2)
                rows.append(dict(n_identity=cell[0], n_labels_per_source=cell[1],
                                 mean=rng.random(), variance=rng.random(), mi=rng.random(),
                                 e_identity=e_id, e_label=e_lab))
        return pd.DataFrame(rows)

    def test_budget_majority_direction(self):
        """A cell where label is always best must be assigned the label channel."""
        f = self._frame()
        _, budget = out_of_fold_picks(f)
        best = np.where(f.e_identity <= f.e_label, 0, 1)
        self.assertTrue((budget == best).all())

    def test_gap_zero_when_pick_is_best(self):
        f = self._frame()
        best = np.where(f.e_identity <= f.e_label, 0, 1)
        gap = np.where(best == 0, f.e_identity, f.e_label) - np.minimum(f.e_identity, f.e_label)
        self.assertTrue((gap == 0).all())

    def test_pred_has_ranges_and_threshold(self):
        for k in ("P1_belief_only_pos_frac", "P2_model_pick_pos_frac", "P3_budget_only_pos_frac"):
            lo, hi = PRED[k]["range"]
            self.assertLess(lo, hi)
        self.assertIn("10%", PRED["practitioner_threshold"])


if __name__ == "__main__":
    unittest.main()
