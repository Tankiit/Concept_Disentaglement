import json
import unittest

import numpy as np

from real.beta_acquisition import action_values
from real.credal_action_band import (
    BASE_PRIOR, CHANNELS, DATASETS, PRED, break_even_strength,
    credal_action_band, family_evidence_discount, family_fixed_strength,
    family_tv_ball, load_selected, tv_distances, verdict)


COUNTS = np.array([[25.0, 7.0], [132.0, 380.0], [347.0, 165.0]])   # synthetic, camelyon-like
BASE_SHAPES = COUNTS + BASE_PRIOR


class FamilyTests(unittest.TestCase):
    def test_families_contain_base_posterior(self):
        for fam in (family_tv_ball(COUNTS, 0.36),
                    family_fixed_strength(COUNTS),
                    family_evidence_discount(COUNTS, 0.36)):
            self.assertTrue(np.any(np.all(fam == BASE_SHAPES, axis=(1, 2))))

    def test_tv_ball_zero_radius_is_the_point(self):
        band = credal_action_band(family_tv_ball(COUNTS, 0.0))
        point = action_values(BASE_SHAPES)
        for i, ch in enumerate(CHANNELS):
            lo, hi = band[ch]
            self.assertAlmostEqual(lo, float(point[i]), places=15)
            self.assertAlmostEqual(hi, float(point[i]), places=15)

    def test_band_contains_point_value(self):
        point = action_values(BASE_SHAPES)
        for fam in (family_tv_ball(COUNTS, 0.2),
                    family_fixed_strength(COUNTS),
                    family_evidence_discount(COUNTS, 0.2)):
            band = credal_action_band(fam)
            for i, ch in enumerate(CHANNELS):
                self.assertLessEqual(band[ch][0], float(point[i]) + 1e-18)
                self.assertGreaterEqual(band[ch][1], float(point[i]) - 1e-18)

    def test_band_monotone_in_tv(self):
        small = credal_action_band(family_tv_ball(COUNTS, 0.1))
        big = credal_action_band(family_tv_ball(COUNTS, 0.4))
        for ch in CHANNELS:
            self.assertLessEqual(big[ch][0], small[ch][0])
            self.assertGreaterEqual(big[ch][1], small[ch][1])

    def test_evidence_discount_lamda_one_is_base(self):
        fam = family_evidence_discount(COUNTS, 0.0)
        self.assertTrue(np.all(fam == BASE_SHAPES))

    def test_strength_family_extremes(self):
        fam = family_fixed_strength(COUNTS)
        self.assertEqual(fam.shape, (125, 3, 2))
        self.assertAlmostEqual(float(fam[:, :, :].min()), float(COUNTS.min() + 0.25))


class VerdictTests(unittest.TestCase):
    def test_disjoint_bands_are_robust(self):
        bands = {"source_id": (1.0, 2.0), "reference_label": (5.0, 6.0)}
        v = verdict(bands, (1.5, 5.5), margin=2.0)
        self.assertEqual(v["verdict"], "robust")
        self.assertEqual(v["winner"], "reference_label")
        self.assertTrue(v["margin_robust"])          # 5.0 / 2.0 = 2.5 >= 2

    def test_overlapping_bands_are_fragile(self):
        bands = {"source_id": (1.0, 4.0), "reference_label": (3.0, 6.0)}
        v = verdict(bands, (2.0, 4.0), margin=2.0)
        self.assertEqual(v["verdict"], "fragile")

    def test_separation_below_margin_not_margin_robust(self):
        bands = {"source_id": (10.0, 12.0), "reference_label": (4.0, 5.0)}
        v = verdict(bands, (11.0, 4.5), margin=2.0)  # 10/5 = 2 >= 2 edge
        self.assertTrue(v["margin_robust"])
        v = verdict(bands, (11.0, 4.5), margin=2.1)
        self.assertFalse(v["margin_robust"])

    def test_zero_loser_band_gives_infinite_separation(self):
        bands = {"source_id": (0.0, 0.0), "reference_label": (1.0, 2.0)}
        v = verdict(bands, (0.0, 1.5), margin=2.0)
        self.assertEqual(v["verdict"], "robust")
        self.assertEqual(v["separation"], float("inf"))

    def test_band_has_teeth_on_near_tie_state(self):
        """A state whose two channel values nearly tie must be fragile somewhere in
        the stress family (the test must be able to say fragile)."""
        rng = np.random.default_rng(0)
        found_fragile = False
        for _ in range(200):
            counts = rng.integers(1, 30, size=(3, 2)).astype(float)
            point = action_values(counts + BASE_PRIOR)
            lo, hi = min(point), max(point)
            if lo <= 0 or hi / lo >= 2.0:
                continue                       # need a near tie
            v = verdict(credal_action_band(family_fixed_strength(counts)), tuple(map(float, point)), 2.0)
            if v["verdict"] == "fragile":
                found_fragile = True
                break
        self.assertTrue(found_fragile)

    def test_break_even_finite_and_above_one_for_clear_preference(self):
        k = break_even_strength(COUNTS, tuple(map(float, action_values(BASE_SHAPES))))
        self.assertGreater(k, 1.0)


class FrozenInputTests(unittest.TestCase):
    """Consistency of the frozen artifacts and PRED coverage. These tests never
    compute the credal bands of the four selected states (that is the run's
    outcome, predicted in PRED before scoring)."""

    def test_selected_states_reproduce_recorded_values(self):
        for dataset in DATASETS:
            states, margin = load_selected(dataset)
            self.assertEqual(margin, 2.0)
            for sid, st in states.items():
                vi, vl = st["point"]
                self.assertGreater(max(vi, vl), 0.0)
                # frozen held-out rule matches the run-level result.json (asserted
                # inside load_selected); here just check internal consistency
                other = "reference_label" if st["preferred"] == "source_id" else "source_id"
                self.assertEqual(st["confirmed"],
                                 st["ese"][st["preferred"]] < st["ese"][other])

    def test_tv_distances_match_tutorial(self):
        for dataset, spec in DATASETS.items():
            per_source, labels, tv_max = tv_distances(dataset)
            self.assertEqual(len(per_source), len(spec["tv_tutorial"]))
            for (_, want) in spec["tv_tutorial"]:
                self.assertTrue(any(abs(v - want) < 2e-4 for v in per_source.values()))
            self.assertEqual(tv_max, max(per_source.values()))
        self.assertAlmostEqual(tv_distances("camelyon17")[2], 0.3625, places=4)
        self.assertAlmostEqual(tv_distances("fmow")[2], 0.0145, places=4)

    def test_pred_covers_selected_states(self):
        for dataset in DATASETS:
            states, _ = load_selected(dataset)
            self.assertEqual(set(PRED["states"][dataset]),
                             {str(s) for s in states})
        for ds in PRED["states"].values():
            for st in ds.values():
                self.assertIn(st["pred"], ("robust", "fragile"))
                self.assertIn(st["winner"], CHANNELS)
                self.assertIn(st["recorded_heldout"], ("confirmed", "failed"))

    def test_pred_json_roundtrip(self):
        json.loads(json.dumps(PRED))            # serialisable, no numpy types
        self.assertTrue(PRED["quantitative"])   # quantitative items exist


if __name__ == "__main__":
    unittest.main()
