"""Independent numerical checks of EU endpoints and model-state safety."""
import math
import unittest
import numpy as np
from scipy.optimize import minimize
import torch
from torch import nn
from real.epistemic_weight_set import bound_epistemic, weighted_eu, DropoutEpistemicRepresenter

class EpistemicSetTests(unittest.TestCase):
    def test_analytic_binary_and_agreement(self):
        p = torch.tensor([[[1., 0.], [0., 1.]], [[.8, .2], [.8, .2]]], dtype=torch.float64)
        out = bound_epistemic(p, rho=.5)
        self.assertAlmostEqual(out.lower[0].item(), -.75*math.log(.75)-.25*math.log(.25), places=9)
        self.assertAlmostEqual(out.upper[0].item(), math.log(2), places=9)
        self.assertLess(out.upper[1].item(), 1e-9)
        self.assertLess(bound_epistemic(p, rho=1).lower.max().item(), 1e-9)

    def test_collapse_nesting_and_random_feasible_weights(self):
        torch.manual_seed(31)
        p = torch.randn(4, 5, 3, dtype=torch.float64).softmax(-1)
        zero = bound_epistemic(p, rho=0)
        self.assertTrue(torch.equal(zero.lower, zero.upper))
        small = bound_epistemic(p, rho=.25, max_iter=400)
        large = bound_epistemic(p, rho=.75, max_iter=400)
        self.assertTrue(torch.all(large.lower <= small.lower+1e-9))
        self.assertTrue(torch.all(large.upper >= small.upper-2e-6))
        for _ in range(20):
            q = torch.distributions.Dirichlet(torch.ones(4, 5, dtype=torch.float64)).sample()
            value = weighted_eu(p, .25/5+.75*q)
            self.assertTrue(torch.all(value >= large.lower-1e-9))
            self.assertTrue(torch.all(value <= large.upper+1e-9))

    def test_independent_scipy_maximum(self):
        rng = np.random.default_rng(81)
        p = rng.dirichlet(np.ones(3), size=(3, 5))
        result = bound_epistemic(torch.tensor(p), rho=.5, max_iter=500, tolerance=1e-7)
        for i, row in enumerate(p):
            h = -(row*np.log(row)).sum(-1)
            def neg_mi(q):
                w = .5/len(row)+.5*q
                mean = w@row
                return (mean*np.log(mean)).sum()+w@h
            ref = minimize(neg_mi, np.full(5, .2), method='SLSQP', bounds=[(0, 1)]*5,
                constraints={'type':'eq', 'fun':lambda q:q.sum()-1}, options={'ftol':1e-12, 'maxiter':1000})
            self.assertTrue(ref.success, ref.message)
            optimum = -ref.fun
            self.assertLessEqual(result.upper_attained[i].item(), optimum+1e-7)
            self.assertGreaterEqual(result.upper[i].item(), optimum-1e-7)
            self.assertLess(result.optimization_gap[i].item(), 2e-5)

    def test_iteration_limit_still_encloses(self):
        torch.manual_seed(5)
        p = torch.randn(3, 7, 4, dtype=torch.float64).softmax(-1)
        early = bound_epistemic(p, max_iter=1)
        refined = bound_epistemic(p, max_iter=400)
        self.assertTrue(torch.all(early.upper >= refined.upper_attained-1e-9))

    def test_wrapper_preserves_modes_and_rng(self):
        torch.manual_seed(3)
        model = nn.Sequential(nn.Linear(3, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(.5), nn.Linear(8, 2))
        model.train(); model[1].eval()
        modes = [m.training for m in model.modules()]
        x = torch.randn(4, 3)
        state = torch.random.get_rng_state().clone()
        wrapper = DropoutEpistemicRepresenter(model, passes=6)
        a, b = wrapper(x), wrapper.predict(x)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(a.probabilities, b.probabilities))
        self.assertTrue(torch.all(a.lower <= a.upper))

    def test_invalid_probabilities(self):
        for p in [torch.full((2, 3, 4), float('nan')), torch.ones(2, 3, 4), -torch.ones(2, 3, 4)]:
            with self.assertRaises(ValueError): bound_epistemic(p)

if __name__ == '__main__': unittest.main()
