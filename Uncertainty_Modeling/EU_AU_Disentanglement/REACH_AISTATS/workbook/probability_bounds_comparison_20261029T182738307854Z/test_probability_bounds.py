import unittest
import numpy as np
import torch
from probly.method import dropout
from probly.representer import representer
from probly.quantification import quantify
from probly.evaluation.ood import evaluate_ood
from real.probability_bounds import (envelope_from_samples,DropoutEnvelopeRepresenter,
                                    SourceMixtureBoundsRepresenter,evaluate_probability_ood)


class ProbabilityBoundTests(unittest.TestCase):
    def test_exact_event_envelope_and_quantify(self):
        p=torch.tensor([[[.6,.3,.1],[.1,.6,.3]]])
        out=envelope_from_samples(p)
        torch.testing.assert_close(out.lower,torch.tensor([[.1,.3,.1]]))
        torch.testing.assert_close(out.upper,torch.tensor([[.6,.6,.3]]))
        torch.testing.assert_close(quantify(out).epistemic,torch.tensor([.5]))
        lo,hi=DropoutEnvelopeRepresenter.event_bounds(out,[0,1])
        torch.testing.assert_close(lo,torch.tensor([.7]))
        torch.testing.assert_close(hi,torch.tensor([.9]))
        self.assertGreater(float(out.upper[:,:2].sum()),float(hi[0]))

    def test_actual_probly_sampler_and_ood_api(self):
        model=dropout(torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.ReLU(),torch.nn.Linear(8,2)),
                      p=.25,predictor_type='logit_classifier')
        model.eval()
        modes=[m.training for m in model.modules()]
        rep=DropoutEnvelopeRepresenter(representer(model,num_samples=5))
        out=rep.represent(torch.ones(3,4))
        self.assertEqual(out.lower.shape,(3,2))
        self.assertTrue(torch.all(out.lower<=out.nominal))
        self.assertTrue(torch.all(out.nominal<=out.upper))
        self.assertEqual(modes,[m.training for m in model.modules()])
        score=quantify(out).epistemic.numpy()
        self.assertAlmostEqual(evaluate_probability_ood(score,score)['auroc'],.5)

    def test_ties_are_not_perfect_detection(self):
        out=evaluate_probability_ood(np.zeros(4),np.zeros(4))
        self.assertEqual(out['auroc'],.5)
        self.assertIn('probly_raw_auroc',out)

    def test_source_oracles_and_novel_event(self):
        A=np.array([[1.,0.],[0.,1.],[.5,.5]])
        p=np.array([.5,.5]); w=np.array([.35,.35,.3])
        rep=SourceMixtureBoundsRepresenter(None,A,A,(p,p))
        out=rep.represent_cells([0,1])
        np.testing.assert_allclose(out.candidate_lower,0)
        np.testing.assert_allclose(out.candidate_upper,1)
        exact=SourceMixtureBoundsRepresenter(None,A,A,(p,p),(w,w))
        out=exact.represent_cells([0,1])
        np.testing.assert_allclose(out.lower,[[1,0],[0,1]])
        np.testing.assert_allclose(out.candidate_lower,.3)
        np.testing.assert_allclose(out.candidate_upper,.3)
        np.testing.assert_allclose(quantify(out).epistemic,0)
        lo,hi=exact.event_bounds([0,1],[0,1])
        np.testing.assert_allclose(lo,1);np.testing.assert_allclose(hi,1)

    def test_more_samples_expand_envelope_not_confidence(self):
        p=torch.tensor([[[.5,.5],[.8,.2],[.05,.95]]])
        a=envelope_from_samples(p[:,:2]); b=envelope_from_samples(p)
        self.assertTrue(torch.all(b.lower<=a.lower))
        self.assertTrue(torch.all(b.upper>=a.upper))

    def test_invalid_probability_samples_fail(self):
        with self.assertRaises(ValueError):
            envelope_from_samples(torch.ones(2,3,4))


if __name__=='__main__':unittest.main()
