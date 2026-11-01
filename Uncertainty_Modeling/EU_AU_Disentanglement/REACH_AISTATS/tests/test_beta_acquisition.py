import unittest
import numpy as np
from real.beta_acquisition import *


class BetaAcquisitionTests(unittest.TestCase):
    def test_analytic_values_equal_exact_updates(self):
        for shapes in [np.ones((3,2)),np.array([[2,8],[3,12],[15,9]]),np.array([[1,1],[110,400],[140,370]])]:
            vi,vl=action_values(shapes)
            self.assertAlmostEqual(float(vi),enumerate_one_step(shapes,'source_id'),places=13)
            self.assertAlmostEqual(float(vl),enumerate_one_step(shapes,'reference_label'),places=13)

    def test_equal_source_means_identity_has_no_one_step_value(self):
        vi,vl=action_values(np.ones((3,2)))
        self.assertEqual(float(vi),0.)
        self.assertGreater(float(vl),0.)

    def test_moments_match_independent_monte_carlo(self):
        s=np.array([[3,7],[5,11],[12,8]])
        rng=np.random.default_rng(4)
        draws=rng.beta(s[:,0],s[:,1],size=(300000,3))
        t=(1-draws[:,0])*draws[:,1]+draws[:,0]*draws[:,2]
        m,v=moments(s)
        self.assertAlmostEqual(float(m),t.mean(),delta=.001)
        self.assertAlmostEqual(float(v),t.var(),delta=.0002)

    def test_quadrature_converges(self):
        s=np.array([[2,8],[20,70],[35,60]])
        a=epistemic_mi(s,24);b=epistemic_mi(s,48)
        self.assertGreater(a,0.)
        self.assertAlmostEqual(a,b,places=10)


if __name__=='__main__':unittest.main()
