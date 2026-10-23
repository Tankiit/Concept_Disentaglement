# Exact mutual-information target check

Date: 2026-10-23. Status: executed deterministic calculation and seeded posterior-update checks.

See [formulation review](../../FORMULATION_REVIEW.md) for definitions, assumptions, primary sources, and interpretation.

## Results

Under equal prior weights on the two candidate worlds:

| Quantity | Value |
| --- | ---: |
| Posterior world probabilities, for any observed dataset | (0.5, 0.5) |
| Posterior world entropy | 1 bit |
| MI between next contaminated cell and world | 0 bits |
| MI between next clean-source cell and world | 0.0281477333 bits |
| Contour-membership MI where the two regions disagree | 1 bit |
| Contour-membership MI where the two regions agree | 0 bits |
| Posterior variance of clean center x coordinate | 0.04 |

![Target-aligned information](target_information.png)

The two colored curves are the candidate clean contours. The bright crescents are points whose membership in the clean central region differs between worlds. The figure is a spatial target-information map, not a confidence band.

## Verification

- Used the saved, validated cell laws from experiment 01 (interior cell width 0.25, including tail bins).
- Evaluated information via weighted KL divergence in bits, avoiding entropy-subtraction cancellation.
- Confirmed zero mixture-observation MI and positive clean-observation MI.
- Verified identical posterior weights after sampled count vectors at n = 10, 100, 1000, 10000, and 1000000, using seed 20261023. These are implementation checks, not repeated-sample performance estimates; equality for all datasets follows analytically.
- Confirmed the contour-membership map is one bit exactly on the disagreement region and zero elsewhere on the plotted grid.
- Checked zero mixture MI for five nondegenerate prior weights. Residual world entropy and membership MI depend on the prior; the one-bit result is specific to equal weights.

No neural model, posterior approximation, bootstrap, or fitted classifier is involved. Both candidate world descriptions are supplied as the diagnostic hypothesis space; the true world identity is not supplied. This gives the learner more structural knowledge than a generic unrestricted-contamination model but still leaves it unable to identify the true contour.

## Interpretation

The observation score overlooks contour ambiguity only if interpreted as measuring that different target. The contour-target score does not fail in this example. No empirical contraction is claimed: mixture MI is zero for every n, while the posterior contour ambiguity is constant.

## Reproduce

After running experiment 01, from the repository root:

```bash
python Uncertainty_Modeling/EU_AU_Disentanglement/Epistemic_Contour_Failure/experiments/02_mutual_information_target_check/run.py
```

Dependencies: numpy, scipy, matplotlib (same environment as experiment 01). Outputs: `summary.json` and `target_information.png`.

## Theoretical decision record and papers used (added 2026-10-23)

- **Evaluate a specific observation channel, not an unspecified epistemic score.** Houlsby et al. [1], Section 2, describe selecting an observation that “maximises the decrease in expected posterior entropy”. We use that information-gain interpretation for I(J;W|D). A zero value says another mixture observation cannot reduce uncertainty; it does not say the contour is known. The formula is adapted to categorical cells rather than presented as a reproduction of their classifier.
- **Compare outcomes under the same posterior.** Changing from mixture cell J to contour membership B_x changes the target, while preserving the same candidate worlds and prior. This is our diagnostic design, not a theorem taken from [1]. It exposes interpretation dependence without posterior approximation.
- **Retain the successful target-aligned control.** The positive membership MI is new understanding from this experiment: it rules out claiming that this example disproves target-specific MI. The one-bit result depends on equal prior weights, not merely on contour separation.
- **Treat general criticism of the entropy decomposition as existing work.** Wimmer et al. [2], abstract, state that they “identify various incoherencies that call their appropriateness into question”. We use this source to motivate scrutiny and flag prior art, not to claim our particular construction follows from their results. A full novelty comparison remains pending.
- **Use two supplied candidate worlds.** This is an exact, correctly specified restricted model, not inference over unrestricted contamination. It prevents approximation and misspecification from being explanations for the zero observation MI, but limits generality.

Papers consulted:

[1] Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, and Máté Lengyel (2011), *Bayesian Active Learning for Classification and Preference Learning*. [Original paper, Section 2](https://mlg.eng.cam.ac.uk/pub/pdf/HouHusGha11a.pdf). Used for the MI/information-gain interpretation in the first bullet; supporting quote is placed there.

[2] Lisa Wimmer, Yusuf Sale, Paul Hofman, Bernd Bischl, and Eyke Hüllermeier (2023), *Quantifying Aleatoric and Epistemic Uncertainty in Machine Learning: Are Conditional Entropy and Mutual Information Appropriate Measures?*, UAI. [Paper, abstract and introduction/background](https://arxiv.org/pdf/2209.03302). Used for motivation and prior-art scope in the fourth bullet; supporting quote is placed there.
