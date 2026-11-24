import unittest
from fractions import Fraction as F
from real.informative_cell_counterexample import (
    P, WORLDS, ACTIONS, BUDGETS, joint, predictive_joint, rates,
    count_outcomes, variance, risk, sequence_risk, value,
)


class InformativeCellTests(unittest.TestCase):
    def test_joint_predictive_equality(self):
        for p in P:
            self.assertEqual(predictive_joint('M', p), predictive_joint('L', p))
            for w in WORLDS:
                self.assertEqual(sum(joint(w, p).values()), 1)
                self.assertEqual(sum(v for (c,y),v in predictive_joint(w,p).items() if y), p)
        self.assertEqual(list(predictive_joint('M', P[0]).values()),
                         [F(3,5), F(1,20), F(1,5), F(3,20)])

    def test_likelihoods_and_one_step(self):
        expected = {'M': ((F(1,5),F(4,5)), (F(0),F(0)), (F(7,20),F(13,20))),
                    'L': ((F(1,2),F(1,2)), (F(1,5),F(4,5)), (F(7,20),F(13,20)))}
        for w in WORLDS:
            for a, r in zip(ACTIONS, expected[w]):
                self.assertEqual(rates(w,a), r)
        for w in WORLDS:
            self.assertEqual(value(w,'cell'), F(81,10000))
            self.assertEqual([post for _,_,post in count_outcomes(rates(w,'cell'),1)],
                             [F(7,20),F(13,20)])
        self.assertEqual(value('M','identity'), F(81,2500))
        self.assertEqual(value('L','source0_label'), F(81,2500))
        self.assertEqual(value('M','source0_label'), 0)
        self.assertEqual(value('L','identity'), 0)
        self.assertEqual(list(count_outcomes(rates('M','source0_label'),1)), [(0,F(1),F(1,2))])

    def test_count_and_ordered_enumeration(self):
        for q in (F(1,4),F(1,2),F(3,4)):
            for w in WORLDS:
                for a in ACTIONS:
                    for n in range(7):
                        channel = rates(w,a)
                        self.assertEqual(sum(m for _,m,_ in count_outcomes(channel,n,q)),1)
                        self.assertEqual(risk(channel,n,q),sequence_risk(channel,n,q))

    def test_multibudget_and_support(self):
        for w in WORLDS:
            for a in ACTIONS:
                last = variance(F(1,2))
                for n in BUDGETS:
                    channel = rates(w,a)
                    current = risk(channel,n)
                    self.assertLessEqual(current,last)
                    self.assertGreater(current,0)
                    for _,_,q in count_outcomes(channel,n):
                        self.assertTrue(0 < q < 1)
                    last = current
        self.assertEqual(risk(rates('M','cell'),2),F(819,10900))

    def test_minimax_certificate(self):
        # Regret is affine in action probabilities. Check the affine identity
        # on all three simplex vertices; this certifies it over the simplex.
        for selected in ACTIONS:
            regrets = [F(81,2500)-value(w,selected) for w in WORLDS]
            self.assertEqual(sum(regrets)/2, F(81,5000)+F(81,10000)*(selected=='cell'))
        for w in WORLDS:
            self.assertEqual(F(81,2500)-(value(w,'identity')+value(w,'source0_label'))/2,
                             F(81,5000))


if __name__ == '__main__':
    unittest.main()
