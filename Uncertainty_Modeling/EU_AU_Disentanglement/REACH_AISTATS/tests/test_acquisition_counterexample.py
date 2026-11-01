import unittest
import numpy as np
from real.acquisition_counterexample import *


class CounterexampleTests(unittest.TestCase):
    def test_target_and_predictive_laws_match(self):
        for p in [P_LOW,P_HIGH]:
            self.assertAlmostEqual((1-p)*0+p*1,.5*p+.5*p)
        self.assertAlmostEqual(float(predictive_eu(.5)),.19274475702175742)

    def test_exact_one_observation_reversal(self):
        for world,good,bad in [('mixture_unknown','source_id','reference_label'),
                               ('label_unknown','reference_label','source_id')]:
            a=exact_expected(world,good,1); b=exact_expected(world,bad,1)
            self.assertAlmostEqual(a['variance_reduction'],.0324)
            self.assertAlmostEqual(b['variance_reduction'],0)
            self.assertAlmostEqual(exact_expected(world,'inputs',1)['variance_reduction'],0)

    def test_update_and_uncertainty_bounds(self):
        self.assertAlmostEqual(float(posterior_q(.5,1,1)),.8)
        self.assertAlmostEqual(float(posterior_q(.5,1,0)),.2)
        lo,hi=eu_bounds(.25,.75)
        grid=predictive_eu(np.linspace(.25,.75,10001))
        self.assertAlmostEqual(float(lo),float(grid.min()))
        self.assertAlmostEqual(float(hi),float(grid.max()))

    def test_exact_symmetry_and_bayes_risk_monotonicity(self):
        previous=.09
        for n in [0,1,2,5,10,20,50,100]:
            a=exact_expected('mixture_unknown','source_id',n)
            b=exact_expected('label_unknown','reference_label',n)
            self.assertEqual(a,b)
            self.assertLessEqual(a['expected_variance'],previous+1e-12)
            previous=a['expected_variance']

    def test_rejection_is_not_zero_width(self):
        lo,hi,empty=confidence_set(1000,500,True)
        self.assertTrue(empty);self.assertTrue(np.isnan(lo));self.assertTrue(np.isnan(hi))
        lo,hi,empty=confidence_set(1000,200,True)
        self.assertEqual(float(lo),.2);self.assertEqual(float(hi),.2)


if __name__=='__main__':unittest.main()
