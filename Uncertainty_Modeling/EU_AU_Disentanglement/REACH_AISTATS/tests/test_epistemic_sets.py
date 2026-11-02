"""epistemic_sets: set representation, contours, score derivations, adapter, tables.

Hand-built stacks pin the semantics: agreeing certain samples -> degenerate set;
two disagreeing halves -> maximal envelope; nesting; adapter fits only what it
is given and separates separable interval features; CSVs round-trip.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from epistemic_sets import (Contour, CredalAdapter, EpistemicSet,
                            EpistemicSetRepresenter, FEATURE_NAMES,
                            RepresentationBuilder, interval_features, set_scores,
                            ece, nll)


def certain_stack(n=4, k=3, col=0):
    """All T samples agree, one-hot on col -> degenerate (point) set."""
    P = np.zeros((5, n, k))
    P[:, :, col] = 1.0
    return P


def halves_stack(n=4, k=2):
    """Half the samples one-hot class 0, half one-hot class 1 -> maximal envelope."""
    P = np.zeros((6, n, k))
    P[:3, :, 0] = 1.0
    P[3:, :, 1] = 1.0
    return P


class TestRepresentationBuilder(unittest.TestCase):
    def test_base_class_is_abstract_contract(self):
        with self.assertRaises(NotImplementedError):
            RepresentationBuilder().build(np.ones((2, 3, 4)))

    def test_representer_is_a_representation_builder(self):
        self.assertIsInstance(EpistemicSetRepresenter(), RepresentationBuilder)


class TestEpistemicSet(unittest.TestCase):
    def test_envelope_contains_mean_and_median(self):
        S = EpistemicSetRepresenter().build(
            np.random.default_rng(0).dirichlet(np.ones(3), size=(7, 5)))
        self.assertTrue((S.lower <= S.mean + 1e-12).all())
        self.assertTrue((S.mean <= S.upper + 1e-12).all())
        self.assertTrue((S.lower <= S.median + 1e-12).all())
        self.assertTrue((S.median <= S.upper + 1e-12).all())
        self.assertEqual(S.width.shape, (5, 3))
        self.assertEqual(S.n_samples, 7)

    def test_agreeing_certain_samples_give_degenerate_set(self):
        S = EpistemicSetRepresenter().build(certain_stack())
        self.assertTrue(np.allclose(S.width, 0))
        # point set: every class outside the certain one has interval [0, 0]
        self.assertTrue((S.upper[:, 1:] == 0).all())

    def test_disagreeing_halves_give_maximal_envelope(self):
        S = EpistemicSetRepresenter().build(halves_stack())
        self.assertTrue(np.allclose(S.lower, 0))
        self.assertTrue(np.allclose(S.upper, 1))
        self.assertTrue(np.allclose(S.mean, 0.5))
        self.assertTrue(np.allclose(S.median, 0.5))

    def test_width_nests_with_sample_count(self):
        rng = np.random.default_rng(1)
        stacked = rng.dirichlet(np.ones(3), size=(100, 6))
        S_full = EpistemicSetRepresenter().build(stacked)
        S_half = EpistemicSetRepresenter().build(stacked[:10])
        self.assertTrue((S_half.width <= S_full.width + 1e-12).all())


class TestContours(unittest.TestCase):
    def test_contours_are_nested_in_alpha(self):
        S = EpistemicSetRepresenter().build(
            np.random.default_rng(2).dirichlet(np.ones(4), size=(8, 3)))
        for row in range(3):
            cs = S.contours(row, alphas=(0.1, 0.3, 0.6))
            for lo, hi in zip(cs, cs[1:]):          # alpha ascending
                self.assertTrue((hi.optimistic <= lo.optimistic).all())
                self.assertTrue((hi.pessimistic <= lo.pessimistic).all())

    def test_pessimistic_subset_of_optimistic(self):
        S = EpistemicSetRepresenter().build(
            np.random.default_rng(3).dirichlet(np.ones(4), size=(8, 3)))
        for c in S.contours(0, alphas=(0.2, 0.5, 0.8)):
            self.assertTrue((c.pessimistic <= c.optimistic).all())
            self.assertLessEqual(c.card_pessimistic, c.card_optimistic)

    def test_point_set_contour_is_the_certain_class(self):
        S = EpistemicSetRepresenter().build(certain_stack(n=2, k=3, col=1))
        cs = S.contours(0, alphas=(0.25, 0.5))
        self.assertEqual(cs[0].card_pessimistic, 1)
        self.assertTrue(cs[0].pessimistic[1])
        self.assertEqual(cs[1].card_pessimistic, 1)

    def test_contour_dataclass_cardinalities(self):
        c = Contour(alpha=0.5, optimistic=np.array([True, True, False]),
                    pessimistic=np.array([True, False, False]))
        self.assertEqual((c.card_optimistic, c.card_pessimistic), (2, 1))


class TestSetScores(unittest.TestCase):
    def test_scores_on_point_set(self):
        S = EpistemicSetRepresenter().build(certain_stack(n=3, k=4, col=2))
        sc = set_scores(S, k_unknown=2)
        self.assertTrue(np.allclose(sc["width_novel"], 0))
        self.assertTrue(np.allclose(sc["set_lower"], 1))
        self.assertTrue(np.allclose(sc["set_upper"], 1))
        self.assertTrue(np.allclose(sc["set_mid"], sc["set_median"]))
        self.assertTrue(np.allclose(sc["width_mean"], 0))
        self.assertEqual(set(sc), set(FEATURE_NAMES))

    def test_scores_on_maximal_envelope(self):
        S = EpistemicSetRepresenter().build(halves_stack())
        sc = set_scores(S, k_unknown=0)
        self.assertTrue(np.allclose(sc["set_lower"], 0))
        self.assertTrue(np.allclose(sc["set_upper"], 1))
        self.assertTrue(np.allclose(sc["width_novel"], 1))
        self.assertTrue(np.allclose(sc["set_median"], 0.5))
        self.assertTrue(np.allclose(sc["card_mean"], 2))   # both classes in every optimistic cut

    def test_residual_mode_targets_known_block(self):
        # k_unknown=None: novel score = 1 - max known prob; bounds follow the envelope
        S = EpistemicSetRepresenter().build(certain_stack(n=2, k=3, col=0))
        sc = set_scores(S)
        self.assertTrue(np.allclose(sc["set_lower"], 0))   # max upper_c = 1 -> lower 0
        self.assertTrue(np.allclose(sc["set_upper"], 0))

    def test_interval_features_matrix_shape_and_order(self):
        S = EpistemicSetRepresenter().build(halves_stack())
        sc = set_scores(S, k_unknown=1)
        F_ = interval_features(sc)
        self.assertEqual(F_.shape, (4, len(FEATURE_NAMES)))
        self.assertTrue(np.allclose(F_[:, FEATURE_NAMES.index("width_novel")], 1))


class TestCredalAdapter(unittest.TestCase):
    def test_separates_separable_interval_features(self):
        rng = np.random.default_rng(4)
        neg = rng.normal(0.0, 0.3, size=(200, len(FEATURE_NAMES)))
        pos = rng.normal(2.0, 0.3, size=(200, len(FEATURE_NAMES)))
        X = np.vstack([neg, pos])
        y = np.r_[np.zeros(200), np.ones(200)]
        for kind in ("logistic", "mlp"):
            ad = CredalAdapter(kind=kind, seed=0).fit(X, y)
            s = ad.decision_function(X)
            ranks = np.argsort(np.argsort(s))          # ranks over the pooled scores
            self.assertLess(ranks[:200].mean(), ranks[200:].mean())
            self.assertGreater((s[200:] > s[:200]).mean(), 0.95)

    def test_fit_is_deterministic_given_seed(self):
        X = np.random.default_rng(5).normal(size=(100, len(FEATURE_NAMES)))
        y = (X[:, 0] > 0).astype(int)
        s1 = CredalAdapter(kind="logistic", seed=1).fit(X, y).decision_function(X)
        s2 = CredalAdapter(kind="logistic", seed=1).fit(X, y).decision_function(X)
        self.assertTrue(np.allclose(s1, s2))

    def test_rejects_unknown_kind(self):
        with self.assertRaises(AssertionError):
            CredalAdapter(kind="forest")


class TestCalibration(unittest.TestCase):
    def test_perfectly_calibrated_confidence_has_near_zero_ece(self):
        rng = np.random.default_rng(6)
        conf = rng.uniform(0.5, 1.0, 4000)
        correct = (rng.uniform(size=4000) < conf)
        self.assertLess(ece(conf, correct), 0.05)

    def test_overconfident_confidence_has_positive_ece(self):
        conf = np.full(1000, 0.99)
        correct = np.zeros(1000)
        self.assertGreater(ece(conf, correct), 0.5)

    def test_nll_bounds(self):
        self.assertEqual(nll(np.full(3, 0.5), np.array([1, 0, 1])), np.log(2))


class TestTabularRoundTrip(unittest.TestCase):
    def test_rows_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rows.csv"
            df = pd.DataFrame({"a": [1.5, 2.5], "b": ["x", "y"]})
            df.to_csv(p, index=False)
            back = pd.read_csv(p)
            self.assertTrue(df.equals(back))


if __name__ == "__main__":
    unittest.main()
