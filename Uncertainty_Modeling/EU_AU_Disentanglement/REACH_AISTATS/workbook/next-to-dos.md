1. A continuous “camouflage” experiment. Gradually mix the candidate cell law toward a mixture of known-source laws. Track the transition from distinguishable to weakly distinguishable to exactly confounded. Compare finite-sample widths with exact identification floors. This separates “we need more data” from “this information cannot resolve the ambiguity.”
   Completed first control: [ten-source camouflage workbook](camouflage_2026-10-21/RESULTS.md). Exact and estimated references, two log budgets, five camouflage levels, and analytically checked identification floors; probability-space simulation, not image morphing. Source-identity acquisition and independent reference-budget sweeps remain follow-ups.
2. Give REACH the wrong candidate. Supply Fashion reference information but deploy Blood, or deploy mixtures of several novel datasets while declaring only one. Record rejection, coverage failures and narrow-but-wrong intervals. This tests whether REACH recognizes when its explanation of the data is inadequate.
3. Make the labels tell a different story while inputs stay identical. Hold all source-cell laws and deployment weights fixed, but change the source-specific target rates. Input observations cannot reveal that change. Acquiring new target labels should distinguish the worlds. This would make the role of \(W_{\mathrm{label}}\) much more tangible.
4. Stress the representation with image transformations. Sweep rotation, blur, occlusion or contrast on a fixed subset of digits. Compare original-model accuracy and confidence with REACH intervals, endpoints and rejection. This tests whether the representation hides a change that matters for predictions; source-law shift must be reported explicitly.
5. Remove the exact-reference advantage—especially for Blood. Estimate source laws from progressively smaller reference samples, including missing rare cells. Does Blood’s apparent certainty survive once empirical zeros become uncertain? This is the most immediate realism check.
For the meeting, I would propose finite-reference validation next, followed by the camouflage and identical-input/different-target experiments. Together, they test statistical uncertainty, structural ambiguity, and the distinct information supplied by labels.

## Completed 2026-10-21: individual digits and held-out digits

- Restored ten individual known digit sources for EMNIST/Fashion/Blood; ran
  paired exact versus estimated source-law controls with fixed targets/logs.
- Trained a fresh six-output CNN on digits 0–5 only; excluded 6–9 from training
  and partition fitting. Tested 6–9 as the aggregate held-out candidate.
- Compared matching candidate information, an unrestricted unknown source,
  and known-only compatibility, including a clean deployment control.
- Ran 20 simulation seeds and independent checks on 10,140 intervals.
- Results and plots: [digit-source experiment](digit_sources_2026-10-21/RESULTS.md).

Finite-reference uncertainty remains consequential. Held-out candidate
membership is still broad with matching references, and known-only tests fail
to reject the shifted held-out mixture at the tested finite-reference budgets.
Source laws and target labels remain finite-pool/synthetic controls: this is not
yet a reference/deployment shift experiment or validation across training seeds.

Next: vary source overlap and reference budgets while retaining individual-digit
sources and the same epistemic targets; add reference/deployment mismatch and
multiple training seeds. Keep interval endpoints and compatibility results
alongside widths. Do not treat narrower intervals alone as the success criterion.

## Completed: paired boundary stress test

See [boundary disagreement versus REACH](boundary_epistemic_2026-10-21/RESULTS.md).
On identical MNIST feature trajectories, dropout disagreement falls deep past
the boundary and prediction-only REACH returns narrow incorrect intervals.
Margin/support cells instead reject the clean source family. This transfers
the notebook's idea; it is not a rerun of its CLIP experiment. Label intervals
remain unchanged within cells; residual mass is not semantic novelty.

Next controls: margin-only versus support-only; displacement-matched random
perturbations; pure source-weight shifts with stable source laws. Separate
distribution-shift rejection from prediction-error detection and evaluate
additional training seeds and realizable image transformations.
