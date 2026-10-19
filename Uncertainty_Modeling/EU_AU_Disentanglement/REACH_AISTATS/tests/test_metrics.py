"""metrics: ensemble MI/AU/MSP decomposition, separation, reporting."""
import unittest
import numpy as np

from metrics import fill_data_and_split, fill_scores, fill_separation, new_metrics, report


def deterministic_disagreement(n=3):
    """(T=2, N, K=2): members are certain but disagree -> MI = ln 2, AU = 0."""
    P = np.zeros((2, n, 2))
    P[0, :, 0], P[1, :, 1] = 1., 1.
    return P


class TestFillScores(unittest.TestCase):
    def test_identical_members_have_zero_mi(self):
        P = np.full((4, 5, 3), 1 / 3)
        rows = fill_scores(new_metrics(), P, {"known": np.ones(5, bool)})
        self.assertTrue(np.allclose(rows["MI"], 0, atol=1e-8))
        self.assertTrue(np.allclose(rows["AU"], np.log(3), atol=1e-6))

    def test_disagreeing_deterministic_members_give_mi_ln2_and_au_zero(self):
        P = deterministic_disagreement()
        M = new_metrics()
        rows = fill_scores(M, P, {"known": np.ones(3, bool)})
        self.assertTrue(np.allclose(rows["MI"], np.log(2), atol=1e-8))
        self.assertTrue(np.allclose(rows["AU"], 0, atol=1e-8))
        entry = M["scores"]["MI"]["known"]
        self.assertEqual(entry["n"], 3)
        self.assertAlmostEqual(entry["mean"], np.log(2), places=6)
        self.assertEqual(M["scores"]["MSP"]["known"]["mean"], 0.5)

    def test_empty_group_is_recorded_as_none(self):
        P = np.full((2, 4, 2), 0.5)
        M = new_metrics()
        fill_scores(M, P, {"known": np.ones(4, bool), "uuc": np.zeros(4, bool)})
        self.assertIsNone(M["scores"]["MI"]["uuc"]["mean"])
        self.assertEqual(M["scores"]["MI"]["uuc"]["n"], 0)


class TestFillSeparation(unittest.TestCase):
    def test_perfectly_separated_scores_give_auroc_one(self):
        P = deterministic_disagreement(6)
        masks = {"known": np.array([1, 1, 1, 0, 0, 0], bool),
                 "kuc": np.zeros(6, bool),
                 "uuc": np.array([0, 0, 0, 1, 1, 1], bool)}
        M = new_metrics()
        rows = fill_scores(M, P, masks)
        # make known rows look certain, unknown rows look uncertain
        rows["MSP"][:3], rows["MSP"][3:] = 0.99, 0.10
        rows["MI"][:3], rows["MI"][3:] = 0.01, 0.90
        fill_separation(M, rows, masks)
        self.assertEqual(M["separate"]["auroc_known_vs_uuc"]["MSP"], 1.0)  # MSP enters as 1-MSP
        self.assertEqual(M["separate"]["auroc_known_vs_uuc"]["MI"], 1.0)
        self.assertEqual(M["separate"]["auroc_known_vs_kuc"], {})          # empty kuc -> no pairs


class TestFillDataAndSplit(unittest.TestCase):
    def test_toy_world_fills_everything(self):
        M = new_metrics()
        X = np.random.default_rng(0).normal(size=(6, 3))
        y = np.array([0, 0, 1, 1, 2, 2])
        out = dict(idx={"train": np.array([0, 1]), "val": np.array([2, 3]), "test": np.array([4, 5])},
                   kkc=[0], kuc=[1], uuc=[2])
        fill_data_and_split(M, X, y, out)
        self.assertEqual(M["data"]["n_rows"], 6)
        self.assertEqual(M["data"]["classes_present"], [0, 1, 2])
        self.assertTrue(M["split"]["leak_free"])
        self.assertEqual(M["split"]["uuc_in_test"], [2])


class TestReport(unittest.TestCase):
    def test_report_flags_small_groups_without_crashing(self):
        M = new_metrics()
        P = np.full((2, 4, 2), 0.5)
        fill_scores(M, P, {"known": np.ones(4, bool), "uuc": np.array([1, 0, 0, 0], bool)})
        fill_separation(M, {k: np.zeros(4) for k in ("MI", "AU", "MSP")},
                        {"known": np.ones(4, bool), "uuc": np.array([1, 0, 0, 0], bool)})
        report(M)  # uuc has n=1 < 200 -> should be flagged, not crash


if __name__ == "__main__":
    unittest.main()
