import unittest
import numpy as np
from real.three_target_characterization import *

class CharacterizationTests(unittest.TestCase):
    def test_same_full_predictive_law_all_families_and_states(self):
        signatures=[]
        for name in FAMILIES:
            models=family(name)
            for state in ('unresolved','oracle_U0','oracle_U1'):
                q=state_weights(models,state);signatures.append(predictive_signature(models,q))
                self.assertAlmostEqual(predictive_mi(models,q),float(entropy(.5)-entropy(.4)))
        self.assertTrue(all(s==signatures[0] for s in signatures))

    def test_analytic_target_ranges(self):
        expected={'known_membership':(0,[.25,.75],[.25,.25]),
                  'candidate_membership':(2,[.2,.6],[.2,.2]),
                  'reference_target':(1,[.3,.7],[.5,.7])}
        for name,(j,unresolved,conditioned) in expected.items():
            models=family(name)
            for state,truth in [('unresolved',unresolved),('oracle_U0',conditioned)]:
                t=target_statistics(models,state_weights(models,state))
                np.testing.assert_allclose([t['lower'][j],t['upper'][j]],truth,atol=1e-14)

    def test_lp_certifies_membership_ranges(self):
        for name,j in [('known_membership',0),('candidate_membership',2)]:
            models=family(name)
            for state in ('unresolved','oracle_U0','oracle_U1'):
                b=membership_lp_certificate(name,state);t=target_statistics(models,state_weights(models,state))
                self.assertEqual(b['status'],'ok');self.assertLess(b['residual'],1e-10)
                np.testing.assert_allclose([b['lower'],b['upper']],[t['lower'][j],t['upper'][j]],atol=1e-12)

    def test_finite_observations_do_not_delete_positive_likelihood_hypotheses(self):
        for name in FAMILIES:
            models=family(name)
            for state in ('unresolved','oracle_U0','oracle_U1'):
                q=state_weights(models,state)
                for channel in ('unlabeled_input','source_identity','deployment_label','reference_label'):
                    L=likelihood(models,channel);np.testing.assert_allclose(L.sum(1),1)
                    self.assertTrue((L>0).all())
                    a=expected_update(models,q,channel)
                    np.testing.assert_allclose(a['width_reduction'],0,atol=1e-13)
                    self.assertGreaterEqual(a['variance_reduction'].min(),-1e-13)

    def test_source_ids_reduce_hidden_membership_variance_without_changing_mi(self):
        for name,j in [('known_membership',0),('candidate_membership',2)]:
            models=family(name)
            before=expected_update(models,state_weights(models),'source_identity')
            after=expected_update(models,state_weights(models,'oracle_U0'),'source_identity')
            self.assertGreater(before['variance_reduction'][j],0)
            self.assertAlmostEqual(after['variance_reduction'][j],0)
            self.assertAlmostEqual(before['mi_reduction'],0)

    def test_oracle_actions_resolve_declared_targets(self):
        for name,j in [('known_membership',0),('candidate_membership',2)]:
            m=family(name);q=state_weights(m)
            self.assertAlmostEqual(expected_update(m,q,'source_weights',True)['expected_width'][j],0)
            self.assertAlmostEqual(expected_update(m,q,'input_law',True)['width_reduction'][j],0)
        m=family('reference_target');q=state_weights(m)
        self.assertAlmostEqual(expected_update(m,q,'deployment_label_law',True)['expected_width'][1],.2)
        self.assertAlmostEqual(expected_update(m,q,'reference_label_laws',True)['expected_width'][1],0)

    def test_aligned_reference_representation_is_sufficient(self):
        m=family('aligned_reference_control')
        for state in ('unresolved','oracle_U0','oracle_U1'):
            q=state_weights(m,state);s=target_statistics(m,q)
            self.assertAlmostEqual(s['lower'][1],.4);self.assertAlmostEqual(s['upper'][1],.6)
            np.testing.assert_allclose([x['targets'][1] for x in m],[x['p'] for x in m])

    def test_full_rank_input_controls_identify_membership(self):
        for row in identifiable_source_controls():
            self.assertEqual(row['affine_rank'],2);self.assertEqual(row['status'],'ok')
            self.assertAlmostEqual(row['lower'],row['truth']);self.assertAlmostEqual(row['upper'],row['truth'])

if __name__=='__main__':unittest.main()
