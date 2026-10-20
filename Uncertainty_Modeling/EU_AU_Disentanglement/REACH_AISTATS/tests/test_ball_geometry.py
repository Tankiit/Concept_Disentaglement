"""Regression checks for public methods respecting the requested geometry."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.ball import MMDBall, weight_interval


class TestGeometry(unittest.TestCase):
    def test_direct_audit_uses_passed_context(self):
        from real import audit
        ctx = dict(groups=[(0,)], d=2, set_kind='box')
        with patch.object(audit, 'true_weights', return_value=np.ones(1)), \
             patch.object(audit, '_draw_log', return_value=np.array([0])), \
             patch.object(audit, '_reference', return_value=(np.array([[1., 0.]]),
                         np.array([[1., 0.]]), .1, .1)), \
             patch.object(audit.MMDBall, 'from_log', side_effect=RuntimeError('stop')) as build:
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                audit.run_cell(ctx, 1, 1, 0)
            self.assertEqual(build.call_args.args[-1], 'box')

    def test_feasibility_distinguishes_box_from_l2(self):
        args = (np.array([[1., 0.]]), np.array([.5, .5]), .6)
        self.assertTrue(MMDBall(*args, set_kind='box').feasible())
        self.assertFalse(MMDBall(*args, set_kind='l2').feasible())

    def test_weight_extrema_use_l2(self):
        ball = MMDBall(np.eye(2), np.array([.5, .5]), .2, set_kind='l2')
        result = weight_interval(ball, 0)
        self.assertAlmostEqual(result['lower'], .5-.2/np.sqrt(2), places=6)
        self.assertAlmostEqual(result['upper'], .5+.2/np.sqrt(2), places=6)
        self.assertAlmostEqual(result['width'], .4/np.sqrt(2), places=6)


if __name__ == '__main__':
    unittest.main()
