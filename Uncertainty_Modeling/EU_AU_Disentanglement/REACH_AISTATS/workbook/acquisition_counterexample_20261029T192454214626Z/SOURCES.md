### Sources and theoretical dependencies

- Hüllermeier & Waegeman (2021), [Aleatoric and epistemic uncertainty](https://doi.org/10.1007/s10994-021-05946-3): distinction between outcome randomness, uncertainty about predictive probabilities, and set-valued representations.
- [Probly](https://github.com/pwhofman/probly): independently check predictive mutual information on the actual categorical second-order representation. Record the installed version. No neural dropout sampling is needed because the two predictive members are exact.
- Hoeffding (1963), [Probability inequalities](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf): fixed-sample confidence-set illustration, clearly separated from posterior/epistemic intervals.
- Ni et al., [DAPPr](https://arxiv.org/abs/2605.00600): motivation for comparing predictive uncertainty representations. We do not assert that DAPPr training produces matched scores on these worlds, and do not fabricate neural logits to claim an empirical DAPPr failure.

**Derived directly below:** Bayes' rule, Bernoulli entropy, posterior variance as squared-error Bayes risk, the law of total variance, and the two-world acquisition-sufficiency counterexample. The counterexample is our construction for this study; no novelty claim against the literature is made.

The acquisition laws, priors and hypothesis set are specified, not learned from the four image datasets. Results are conditional on these choices. All sources, equations, code, draws, outputs and plots are saved under a fresh workbook directory.