1. A continuous “camouflage” experiment. Gradually mix the candidate cell law toward a mixture of known-source laws. Track the transition from distinguishable to weakly distinguishable to exactly confounded. Compare finite-sample widths with exact identification floors. This separates “we need more data” from “this information cannot resolve the ambiguity.”
   Completed first control: [ten-source camouflage workbook](camouflage_2026-10-21/RESULTS.md). Exact and estimated references, two log budgets, five camouflage levels, and analytically checked identification floors; probability-space simulation, not image morphing. Completed [source-identity acquisition follow-up](camouflage_identities_2026-10-21/RESULTS.md): reliable identities reduce W_cls and W_unk at 99% and 100% camouflage while the reference-anchored W_label remains unchanged. Completed [reference-budget sweep](camouflage_references_2026-10-21/RESULTS.md): with 8,192 identities fixed, increasing digit references mainly reduces W_cls and also W_label; increasing candidate references mainly reduces W_unk. Next: vary target-label acquisition with other evidence fixed.
2. Give REACH the wrong candidate. Supply Fashion reference information but deploy Blood, or deploy mixtures of several novel datasets while declaring only one. Record rejection, coverage failures and narrow-but-wrong intervals. This tests whether REACH recognizes when its explanation of the data is inadequate.
3. Make the labels tell a different story while inputs stay identical. Hold all source-cell laws and deployment weights fixed, but change the source-specific target rates. Input observations cannot reveal that change. Acquiring new target labels should distinguish the worlds. This would make the role of \(W_{\mathrm{label}}\) much more tangible.
4. Stress the representation with image transformations. Sweep rotation, blur, occlusion or contrast on a fixed subset of digits. Compare original-model accuracy and confidence with REACH intervals, endpoints and rejection. This tests whether the representation hides a change that matters for predictions; source-law shift must be reported explicitly.
5. Remove the exact-reference advantage—especially for Blood. Estimate source laws from progressively smaller reference samples, including missing rare cells. Does Blood’s apparent certainty survive once empirical zeros become uncertain? This is the most immediate realism check.
For the meeting, I would propose finite-reference validation next, followed by the camouflage and identical-input/different-target experiments. Together, they test statistical uncertainty, structural ambiguity, and the distinct information supplied by labels.

## Completed 2026-10-21: candidate misspecification

See [candidate-law misspecification](candidate_mismatch_2026-10-21/RESULTS.md).
Matched controls remain compatible and cover their simulated targets. Fashion
deployed with Blood references can yield extremely narrow, incorrect W_unk
intervals without rejection. Source identities expose this mismatch, but do not
guarantee detection for every wrong law at finite reference budgets.
This completes the single wrong-candidate control in item 2; mixtures of multiple
novel datasets remain untested. A useful follow-up is a graded mismatch sweep
to locate where noncoverage starts relative to rejection, with endpoints and
candidate prevalence reported alongside widths.

## Completed 2026-10-21: target-label acquisition

See [target-label acquisition under camouflage](camouflage_labels_2026-10-21/RESULTS.md).
With all other evidence fixed, increasing labels per digit from 512 to 32,768
reduces mean W_label from 0.324 to 0.081 with baseline references. Exact target
rates leave 0.045 from reference uncertainty; exact rates and source laws give
zero width. W_cls and W_unk endpoints are reused and unchanged by definition.
The target-label sweep proposed in item 1 is complete. The identical-input,
different-target-world comparison in item 3 remains separate and unexecuted.
Next priority: candidate misspecification and coverage/compatibility failures.

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
