# Selecting the first epistemic formulation

Date: 2026-10-23. Status: initial source review and exact diagnostic; not a novelty review.

## Framework correction (2026-10-23)

The user confirmed a frequentist evaluation framework. The MI/posterior selection below is a historical investigation, not an agreed Bayesian direction. The current theory and procedure-selection criteria are in [experiment 05](experiments/05_frequentist_theory/RESULTS.md); prioritize fixed-target confidence coverage and explicit resampling targets.

## Historical decision

Start with the information-theoretic quantity commonly interpreted as predictive epistemic uncertainty:

    I(Y; Theta | x, D)
      = H(Y | x, D) - E_{Theta | D}[H(Y | x, Theta)].

The predicted variable Y is part of the definition. It cannot be silently replaced by uncertainty about any latent property of the data-generating process. Our first diagnostic compares observation prediction with contour membership using the same exact posterior. It removes approximation and misspecification as explanations, while explicitly revealing the change of target.

## Primary sources inspected

- [Houlsby et al., Bayesian Active Learning for Classification and Preference Learning (2011), Section 2](https://mlg.eng.cam.ac.uk/pub/pdf/HouHusGha11a.pdf): the acquisition objective is expected reduction in parameter entropy, expressed through predictive entropies. This is information supplied by a particular potential observation, not an assertion that zero information gain means the parameters are already known.
- [Wimmer et al., Quantifying Aleatoric and Epistemic Uncertainty in Machine Learning: Are Conditional Entropy and Mutual Information Appropriate Measures? (UAI 2023), introduction and background](https://arxiv.org/pdf/2209.03302): explicitly examines the conditional-entropy/mutual-information decomposition used for classification uncertainty and reports conceptual objections. This is directly relevant prior work; criticism of this decomposition in general is already established and cannot be claimed as our novelty.

These sources motivate the measure and its interpretation. The Gaussian two-world calculation below is our own diagnostic, not a result attributed to either source. A systematic comparison with the specific objections and counterexamples in Wimmer et al. remains necessary.

## Candidate assessment

| Candidate | Quantity actually measured | Role in this study | Main limitation of an apparent failure |
| --- | --- | --- | --- |
| Predictive mutual information | Dependence between a specified future outcome and posterior model uncertainty | First exact diagnostic; compare different outcomes explicitly | Zero information from mixture observations does not imply certainty about latent contours |
| Posterior uncertainty of the contour itself | Posterior distribution of the clean target | Necessary positive control | Depends on prior and contamination model; cannot be assumed to collapse |
| Gaussian posterior ignoring contamination | Uncertainty conditional on a single-Gaussian observation model | Later misspecification baseline | Narrow wrong contours may establish model misspecification rather than a defect in Bayesian uncertainty |
| Bootstrap contour variation | Sampling variability of a chosen fitting procedure | Later practical baseline | Stability around a biased estimator is not automatically clean-target uncertainty |
| Ensemble/dropout classification disagreement | Variation in predictions of a defined classifier | Defer until a prediction task is specified | Neither directly defines contour uncertainty in the current unlabeled setup |

## Exact two-world analysis

Use the two complete candidate worlds from the first construction, including their respective contamination laws. Let W be the world index with prior probability 1/2 each. This is a deliberately restricted, correctly specified diagnostic model: one of these two worlds generates the observations. It is not a prior over all possible contamination distributions.

Because their observed cell laws coincide, for every possible dataset D and every sample size,

    P(W=0 | D) = P(W=1 | D) = 1/2.

The posterior does not contract. Its entropy remains one bit, and the posterior variance of the clean center's x coordinate remains 0.04.

For J, the cell of the next contaminated observation,

    I(J; W | D) = 0.

This follows because P(J | W=0,D) = P(J | W=1,D) = m. It is exactly zero from the outset, not a score observed to shrink with n. Correctly interpreted, it says that another observation of the same type supplies no information about W.

For the target B_x(W), which indicates whether a fixed spatial point x is inside the clean 95%-mass region,

    I(B_x; W | D) = H(B_x | D),

because B_x is deterministic given W. This equals one bit in the symmetric difference between the two regions and zero elsewhere. The contour-target MI therefore detects the relevant spatial ambiguity. B_x is an evaluable target variable, not a label assumed available in the training data.

A clean-source cell observation provides a third control: I(J_clean; W | D) is positive (approximately 0.028148 bits on the selected grid). Changing the evidence type can resolve ambiguity even though more contaminated observations cannot.

## Supported conclusion

**Predictive information gain about an observation channel is not a complete measure of uncertainty about the clean contour.** Even exact inference can yield zero for the former while the latter remains unresolved.

This is an interpretation/target mismatch. It is not a demonstrated mathematical failure of mutual information, and it does not refute target-specific predictive epistemic uncertainty. In this experiment, the target-aligned measure and full posterior preserve ambiguity correctly. This control must remain in the paper rather than be omitted in pursuit of the initial hypothesis.

A possible scoped paper claim is: “Observation-based epistemic scores need not diagnose uncertainty about latent clean-distribution contours under source ambiguity.” Whether this adds a novel contribution beyond known nonidentifiability and task dependence is open.

## Next discriminating experiment

Compare a contamination-ignoring posterior contour estimate with a contamination-aware posterior/control on the same observations. First establish each model's target and derive its limiting behavior. Then evaluate repeated-sample contour coverage and width. Label any demonstrated contraction under misspecification accordingly. A stronger critique of an existing definition requires a defensible target-aligned counterexample; the present check does not supply one.

Executed diagnostic: [results and reproduction](experiments/02_mutual_information_target_check/RESULTS.md).
