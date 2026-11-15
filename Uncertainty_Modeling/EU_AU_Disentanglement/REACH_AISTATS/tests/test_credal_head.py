"""credal_head: relative-likelihood credal set from the ensemble, and its discipline.

Pins: the KL-ball reduction (bounds invert the binary KL exactly, closed forms at
the simplex edges, no feasible h beats a bound), the collapse theorem (the set
depends on the stack only through its mean -- unlike the min/max envelope), tau
hygiene (bounds are a pure function of (stack, tau)), kkc-only AU bin edges, the
near/far recipe, the PRED-first guard, and the verdict scorer on toy frames.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import credal_head as ch
from credal_head import (PRED, PRIMARY_TAU, TAUS, _binary_kl, au_bins,
                         credal_head_width, evaluate_pred, kl_ball_bounds,
                         near_far_tags, width_from_bounds)
from epistemic_sets import EpistemicSetRepresenter


def full_kl(p, h):
    p, h = np.asarray(p, float), np.asarray(h, float)
    m = p > 0
    return float((p[m] * np.log(p[m] / h[m])).sum())


def attainer(p_bar, y0, q):
    """The extremal h: h(y0) = q, the rest of p_bar rescaled proportionally."""
    h = p_bar * (1 - q) / (1 - p_bar[y0])
    h[y0] = q
    return h


class TestBinaryKL(unittest.TestCase):
    def test_zero_on_diagonal_positive_off(self):
        p = np.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(_binary_kl(p, p), 0.0, atol=1e-12)
        self.assertTrue((_binary_kl(p, p + 0.05) > 0).all())

    def test_edge_p_zero_is_minus_log_one_minus_q(self):
        q = np.array([0.1, 0.4, 0.7])
        np.testing.assert_allclose(_binary_kl(np.zeros(3), q), -np.log1p(-q), rtol=1e-12)


class TestKLBallBounds(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.P = rng.dirichlet(np.ones(4) * 1.5, size=50)          # interior rows

    def test_closed_forms_at_simplex_edges(self):
        for tau in TAUS:
            lo, hi = kl_ball_bounds(np.array([[0.0, 1.0]]), tau)
            np.testing.assert_allclose(lo[0], [0.0, tau], atol=1e-12)
            np.testing.assert_allclose(hi[0], [1.0 - tau, 1.0], atol=1e-12)

    def test_bounds_invert_the_binary_kl(self):
        r = -np.log(PRIMARY_TAU)
        lo, hi = kl_ball_bounds(self.P, PRIMARY_TAU)
        np.testing.assert_allclose(_binary_kl(self.P, hi), r, rtol=1e-7)
        ok = self.P > 1e-3          # below ~r/700 the true lower root underflows float64
        np.testing.assert_allclose(_binary_kl(self.P[ok], lo[ok]), r, rtol=1e-7)
        self.assertTrue((lo[~ok] < 1e-30).all())

    def test_attainer_is_in_the_ball_and_reduction_is_exact(self):
        r = -np.log(PRIMARY_TAU)
        lo, hi = kl_ball_bounds(self.P, PRIMARY_TAU)
        for i in range(5):
            for y0 in range(4):
                for q in (lo[i, y0], hi[i, y0]):
                    h = attainer(self.P[i], y0, q)
                    self.assertAlmostEqual(h.sum(), 1.0, places=12)
                    kl = full_kl(self.P[i], h)
                    self.assertAlmostEqual(kl, float(_binary_kl(self.P[i, y0], q)), places=9)
                    self.assertLessEqual(kl, r + 1e-8)

    def test_no_feasible_h_beats_a_bound(self):
        rng = np.random.default_rng(1)
        r = -np.log(PRIMARY_TAU)
        lo, hi = kl_ball_bounds(self.P[:10], PRIMARY_TAU)
        hits = 0
        for i in range(10):
            H = rng.dirichlet(self.P[i] * 40 + 1e-3, size=4000)
            ok = np.array([full_kl(self.P[i], h) <= r for h in H])
            hits += ok.sum()
            if ok.any():
                self.assertTrue((H[ok] <= hi[i] + 1e-9).all())
                self.assertTrue((H[ok] >= lo[i] - 1e-9).all())
        self.assertGreater(hits, 100)                  # the check actually sampled the ball

    def test_containment_and_nesting_in_tau(self):
        lo90, hi90 = kl_ball_bounds(self.P, 0.90)
        lo95, hi95 = kl_ball_bounds(self.P, 0.95)
        self.assertTrue((lo90 <= self.P).all() and (self.P <= hi90).all())
        self.assertTrue((lo90 <= lo95 + 1e-12).all() and (hi95 <= hi90 + 1e-12).all())

    def test_tau_outside_unit_interval_rejected(self):
        for bad in (0.0, 1.0, 1.2):
            with self.assertRaises(AssertionError):
                kl_ball_bounds(self.P, bad)

    def test_tiny_mass_sits_at_the_floor(self):
        """KL(p_bar || h): raising a class with no mean mass costs only the mass taken
        from the others, so its upper bound is 1 - tau, not ~0 (the PRED_AMENDMENT
        trigger: the reverse direction D(q || p) would give ~0.005)."""
        for tau in TAUS:
            lo, hi = kl_ball_bounds(np.array([[1e-10, 1 - 1e-10]]), tau)
            self.assertAlmostEqual(float(hi[0, 0]), 1 - tau, places=8)
            self.assertGreaterEqual(lo[0, 0], 0.0)

    def test_width_curve_symmetric_with_floor_and_peak(self):
        grid = np.linspace(0, 1, 2001)
        for tau in TAUS:
            lo, hi = kl_ball_bounds(np.stack([grid, 1 - grid], 1), tau)
            g = hi[:, 0] - lo[:, 0]
            np.testing.assert_allclose(g, g[::-1], atol=1e-9)        # g(p) = g(1 - p)
            self.assertAlmostEqual(float(g[0]), 1 - tau, places=9)    # floor at both ends
            self.assertAlmostEqual(float(g[-1]), 1 - tau, places=9)
            self.assertAlmostEqual(float(g[1000]), np.sqrt(1 - tau ** 2), places=9)
            self.assertTrue((np.diff(g[:1001]) > 0).all())           # increasing to 1/2

    def test_auroc_identity_with_the_ambiguity_score(self):
        from example_wrn50 import auroc_fast
        rng = np.random.default_rng(6)
        p_known = rng.beta(0.3, 8.0, 300)
        p_novel = rng.beta(2.0, 1.0, 200)               # includes near-certain novel rows
        def width(p):
            lo, hi = kl_ball_bounds(np.stack([1 - p, p], 1), PRIMARY_TAU)
            return hi[:, 1] - lo[:, 1]
        a = auroc_fast(width(p_known), width(p_novel))
        b = auroc_fast(-np.abs(p_known - 0.5), -np.abs(p_novel - 0.5))
        self.assertAlmostEqual(a, b, places=12)
        self.assertLess(a, auroc_fast(p_known, p_novel))   # symmetry costs separation here


class TestCollapseTheorem(unittest.TestCase):
    def test_same_mean_same_set_unlike_the_envelope(self):
        agree = np.full((6, 1, 2), 0.5)                 # all members say (0.5, 0.5)
        split = np.zeros((6, 1, 2))
        split[:3, :, 0] = 1.0                           # half one-hot 0, half one-hot 1
        split[3:, :, 1] = 1.0
        a = credal_head_width(agree, PRIMARY_TAU)
        b = credal_head_width(split, PRIMARY_TAU)
        np.testing.assert_allclose(a["credal_width_unk"], b["credal_width_unk"], atol=1e-12)
        np.testing.assert_allclose(a["p_lo_t090"], b["p_lo_t090"], atol=1e-12)
        env_a = EpistemicSetRepresenter().build(agree).width[0, -1]
        env_b = EpistemicSetRepresenter().build(split).width[0, -1]
        self.assertAlmostEqual(env_a, 0.0)
        self.assertAlmostEqual(env_b, 1.0)              # the hull sees the disagreement

    def test_logits_input_equals_prob_input(self):
        rng = np.random.default_rng(2)
        Z = rng.normal(size=(5, 7, 3)) * 3
        P = np.exp(Z - Z.max(-1, keepdims=True))
        P /= P.sum(-1, keepdims=True)
        a, b = credal_head_width(Z, 0.95), credal_head_width(P, 0.95)
        np.testing.assert_allclose(a["credal_width_unk"], b["credal_width_unk"], atol=1e-12)


class TestTauHygiene(unittest.TestCase):
    def test_taus_are_the_fixed_convention(self):
        self.assertEqual(TAUS, (0.90, 0.95))
        self.assertIn(PRIMARY_TAU, TAUS)

    def test_bounds_are_a_pure_function_of_stack_and_tau(self):
        params = list(inspect.signature(credal_head_width).parameters)
        self.assertEqual(params, ["ens", "tau", "k_unknown"])   # no labels, no masks
        rng = np.random.default_rng(3)
        S = rng.dirichlet(np.ones(4), size=(5, 9))
        S0 = S.copy()
        a, b = credal_head_width(S, PRIMARY_TAU), credal_head_width(S, PRIMARY_TAU)
        np.testing.assert_array_equal(S, S0)                    # input not mutated
        np.testing.assert_array_equal(a["credal_width_unk"], b["credal_width_unk"])


class TestWidthFromBounds(unittest.TestCase):
    def test_unknown_slot_and_mean_gap(self):
        lo = np.array([[0.0, 0.1, 0.2]])
        hi = np.array([[0.3, 0.2, 0.6]])
        w = width_from_bounds(lo, hi, k_unknown=-1)
        self.assertAlmostEqual(float(w["credal_width_unk"][0]), 0.4)
        self.assertAlmostEqual(float(w["credal_width_mean"][0]), (0.3 + 0.1 + 0.4) / 3)
        self.assertNotIn("credal_width_unk", width_from_bounds(lo, hi))


class TestAUBins(unittest.TestCase):
    def test_edges_come_from_known_rows_only(self):
        rng = np.random.default_rng(4)
        au = rng.random(400)
        known = np.arange(400) < 300
        b1 = au_bins(au, known)
        au2 = au.copy()
        au2[~known] = rng.random(100) * 50              # move only the novel rows
        b2 = au_bins(au2, known)
        np.testing.assert_array_equal(b1[known], b2[known])
        self.assertTrue(((b1 >= 0) & (b1 <= 9)).all() and ((b2 >= 0) & (b2 <= 9)).all())
        self.assertEqual(int(b2[~known].max()), 9)      # far-out novel rows still assigned
        counts = np.bincount(b1[known], minlength=10)
        self.assertTrue((np.abs(counts - 30) <= 1).all())   # deciles of the known rows


class TestNearFar(unittest.TestCase):
    def test_recipe_tags_near_and_far(self):
        rng = np.random.default_rng(5)
        X = np.concatenate([rng.normal(0, 1, (40, 2)),       # kkc 0
                            rng.normal(10, 1, (40, 2)),      # kkc 1
                            rng.normal(0.5, 1, (20, 2)),     # novel 2: inside class 0
                            rng.normal(60, 1, (20, 2))])     # novel 3: far away
        y = np.repeat([0, 1, 2, 3], [40, 40, 20, 20])
        world = {"X": X, "y": y, "kkc": [0, 1], "kuc": [], "uuc": [2, 3],
                 "idx": {"train": np.arange(80)}}
        tag, meds, cutoff = near_far_tags(world, k=5, q=0.95)
        self.assertEqual(tag, {2: "near", 3: "far"})
        self.assertEqual(cutoff, 1.0)
        self.assertLess(meds[2], 1.0 + 1e-9)
        self.assertGreater(meds[3], 1.0)


class TestPredFirstGuard(unittest.TestCase):
    def test_pred_names_every_check_and_the_rule(self):
        self.assertTrue(PRED["written_before_scoring"])
        for key in ("P1_direction", "P2_collapse_corollary", "P3_vs_envelope",
                    "P4_dependence", "P5_near_far_crossover", "P6_tau_invariance",
                    "P7_matched_au", "verdict_rule"):
            self.assertIn(key, PRED)

    def test_amendment_keeps_p1_and_records_its_trigger(self):
        A = ch.PRED_AMENDMENT
        self.assertTrue(A["made_before_real_data_scoring"])
        self.assertIn("scored as written", A["P1_status"])
        self.assertIn("P1_amended", A)
        self.assertIn("ratio_band", PRED["P1_direction"])        # original left verbatim

    def test_main_refuses_a_different_pred_before_any_work(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "pred.json").write_text(json.dumps({"other": 1}))
            with self.assertRaises(AssertionError):
                ch.main(world_seeds=(), out_dir=d)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["pred.json"])

    def test_main_refuses_a_different_amendment(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "pred.json").write_text(json.dumps(PRED))
            (Path(d) / "pred_amendment.json").write_text(json.dumps({"other": 1}))
            with self.assertRaises(AssertionError):
                ch.main(world_seeds=(), out_dir=d)


class TestEvaluatePred(unittest.TestCase):
    def frames(self, cred_auroc):
        arms = {"credal_width_unk_t090": cred_auroc, "credal_width_unk_t095": cred_auroc,
                "P_unknown": 0.946, "set_median": 0.950, "width_novel": 0.916}
        summary = pd.DataFrame([{"world_seed": 0, "method": m,
                                 "auroc_pos_uuc_neg_kkc": a} for m, a in arms.items()])
        matched = pd.DataFrame([{"world_seed": 0, "bin": b, **{a: 0.8 for a in ch.MATCHED_ARMS},
                                 "P_unknown": 0.9} for b in range(3)])
        cors = pd.DataFrame([{"world_seed": 0, "spearman_cred90_vs_p_unknown": 0.8,
                              "spearman_cred90_vs_eu_bald": 0.3,
                              "spearman_width_novel_vs_eu_bald": 0.7}])
        p_unk = np.array([1e-4, 2e-4, 0.5, 0.9])
        lo, hi = kl_ball_bounds(np.stack([1 - p_unk, p_unk], 1), PRIMARY_TAU)
        rows = pd.DataFrame({
            "world_seed": 0,
            "group": ["kkc", "kkc", "uuc", "uuc"],
            "novel_tag": ["kkc", "kkc", "near", "far"],
            "credal_width_unk_t090": hi[:, 1] - lo[:, 1],     # g(p): 0.100, 0.100, 0.436, 0.273
            "P_unknown": p_unk.astype(np.float32),
            "p_bar_unk_f64": p_unk})
        return summary, matched, cors, rows

    def test_baseline_reading_when_below_existing_readouts(self):
        v = evaluate_pred(*self.frames(0.90))
        self.assertFalse(v["P1_ratio_uuc_over_kkc"]["pass"])   # original P1: unattainable
        self.assertTrue(v["P1_amended"]["pass"])
        self.assertTrue(v["identity_ambiguity_auroc"]["pass"])
        self.assertTrue(v["P2a_below_p_unknown"]["pass"] and v["P2b_below_set_median"]["pass"])
        self.assertTrue(v["P5_near_far_crossover"]["pass"])
        self.assertTrue(v["P7_matched_au"]["pass"])
        self.assertTrue(v["overall"].startswith("not informative"))
        json.dumps(v)                                   # verdict.json must serialize

    def test_escalates_when_the_arm_beats_a_readout(self):
        v = evaluate_pred(*self.frames(0.97))
        self.assertFalse(v["P2a_below_p_unknown"]["pass"])
        self.assertTrue(v["overall"].startswith("INFORMATIVE"))

    def test_near_far_untestable_when_a_tag_is_absent(self):
        s, m, c, rows = self.frames(0.90)
        rows.loc[rows.novel_tag == "far", "novel_tag"] = "near"
        self.assertIsNone(evaluate_pred(s, m, c, rows)["P5_near_far_crossover"]["pass"])


if __name__ == '__main__':
    unittest.main()
