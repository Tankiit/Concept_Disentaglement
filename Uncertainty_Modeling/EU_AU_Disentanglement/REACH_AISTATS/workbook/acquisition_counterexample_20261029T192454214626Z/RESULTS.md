# Counterexample: identical predictive uncertainty, reversed useful acquisition

Both worlds have the same target T=P(Y=1), the same predictive law over Bernoulli(0.2)/Bernoulli(0.8), and equal hypothesis weights.

- Initial epistemic MI: 0.192744757 nats.
- Same epistemic interval for prior weights [0.25,0.75]: [0.147044215, 0.192744757] nats.
- Initial target-probability posterior variance: 0.09.
- One informative acquisition reduces expected variance to 0.0576: value 0.0324.
- World M: source identities have value 0.0324; source-0 labels have value 0.
- World L: source-0 labels have value 0.0324; source identities have value 0.
- Inputs have value 0 in both worlds.
- Any policy restricted to the shared predictive representation has worst-world one-step regret at least 0.0162 (allowing randomized choices).

This proves predictive-representation-only acquisition insufficiency in the declared model. It does not prove that epistemic measures are incorrect, that static intervals repair the problem, or that DAPPr training fails. Correct Bayesian updates using the full channel likelihood succeed.

The notebook includes the algebraic proof, exact outcome enumeration, 10,000 paired repetitions, fixed-n confidence-set illustration, and clearly separated probability and epistemic intervals.
