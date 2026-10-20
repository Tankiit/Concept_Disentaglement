"""Analytic controls for the experimental joint-mass LP."""
import unittest
import numpy as np
from investigate import joint_interval, bounds


class TestJoint(unittest.TestCase):
    def test_exact_identified_posterior(self):
        a = np.array([[.8,.2],[.1,.9]])
        w = np.array([.7,.3]); p = w@a
        for c in range(2):
            lo, hi, status = joint_interval(a, a, p, p, c)
            np.testing.assert_allclose([lo,hi], w[-1]*a[-1,c]/p[c], atol=1e-7)
            self.assertEqual(status, 'optimal')

    def test_incompatibility(self):
        a = np.array([[.8,.2],[.1,.9]])
        p = np.array([.95,.05])
        self.assertEqual(joint_interval(a,a,p,p,0)[2], 'empty_set')

    def test_zero_count_bound(self):
        lo, hi = bounds(np.array([0,200]), .025)
        self.assertEqual(lo[0], 0)
        self.assertEqual(hi[1], 1)
        self.assertTrue(0 < hi[0] < .03)


if __name__ == '__main__':
    unittest.main()
