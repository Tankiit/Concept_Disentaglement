"""Analytic and geometry checks for probability-ratio interval inference."""
import unittest
import numpy as np
from real.source_acquisition import (polytope, ratio_interval, membership_bounds,
                                    anchored_label_bounds, frequency_box)


class SourceAcquisitionTests(unittest.TestCase):
    def test_confounding_and_identity_oracle(self):
        A=np.array([[1.,0.],[0.,1.],[.5,.5]])
        p=np.array([.5,.5]); w=np.array([.35,.35,.3])
        out=membership_bounds(A,A,(p,p),None,0)['candidate_membership']
        np.testing.assert_allclose([out['lower'],out['upper']],[0,1])
        out=membership_bounds(A,A,(p,p),(w,w),0)['candidate_membership']
        np.testing.assert_allclose([out['lower'],out['upper']],[.3,.3])

    def test_reference_box_ratio_has_analytic_extremes(self):
        lo=np.array([[.2,.6],[.6,.2]])
        hi=np.array([[.4,.8],[.8,.4]])
        P=polytope(lo,hi,weights=(np.array([.5,.5]),np.array([.5,.5])))
        out=ratio_interval(P,[1,0,0,0],[1,0,1,0])
        np.testing.assert_allclose([out['lower'],out['upper']],[.2/(.2+.8),.4/(.4+.6)])
        label=anchored_label_bounds(lo,hi,[.1,.7],[.3,.9],0)
        # Low label uses largest first-source share; high label uses smallest.
        np.testing.assert_allclose([label['lower'],label['upper']],[.4*.1+.6*.7,.2*.3+.8*.9])

    def test_empty_and_undefined_are_distinct(self):
        A=np.array([[1.,0.],[1.,0.]])
        out=ratio_interval(polytope(A,A),[0,1,0,0],[0,1,0,1])
        self.assertEqual(out['status'],'undefined_denominator_policy')
        out=ratio_interval(polytope(A,A,marginal=(np.array([0.,1.]),np.array([0.,1.]))),
                           [1,0,0,0],[1,0,1,0])
        self.assertEqual(out['status'],'empty')
        self.assertTrue(np.isnan(out['width']))

    def test_nested_model_sets_contract(self):
        lo=np.zeros((3,3)); hi=np.ones((3,3))
        wide=membership_bounds(lo,hi,None,None,0)['candidate_membership']
        A=np.array([[.7,.2,.1],[.1,.7,.2],[.2,.1,.7]])
        w=np.array([.4,.3,.3]); p=w@A
        narrow=membership_bounds(A,A,(p,p),None,0)['candidate_membership']
        self.assertLessEqual(narrow['width'],wide['width'])
        self.assertAlmostEqual(narrow['width'],0)
        self.assertAlmostEqual(narrow['lower'],w[2]*A[2,0]/p[0])

    def test_zero_observations_are_uninformative(self):
        lo,hi=frequency_box(np.zeros(3),.05)
        np.testing.assert_equal(lo,0); np.testing.assert_equal(hi,1)


if __name__=='__main__':
    unittest.main()
