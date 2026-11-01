# Controlled three-target characterization: executed results

All 12 states have the same second-order law over P(cell,label), with predictive MI 0.020135513551 nats. Probly agrees to numerical precision.

| Target / family | Unresolved hull | Oracle U=0 hull | Oracle U=1 hull |
|---|---|---|---|
| known_membership | [0.25, 0.75] | [0.25, 0.25] | [0.75, 0.75] |
| candidate_membership | [0.20, 0.60] | [0.20, 0.20] | [0.60, 0.60] |
| reference_target | [0.30, 0.70] | [0.50, 0.70] | [0.30, 0.50] |
| aligned_reference_control | [0.40, 0.60] | [0.40, 0.60] | [0.40, 0.60] |

Source-identity observations reduce expected membership variance while preserving the entire predictive law for every realized outcome:

- known_membership: one-observation variance reduction 0.01250000; predictive MI reduction zero.
- candidate_membership: one-observation variance reduction 0.00666667; predictive MI reduction zero.

All finite-observation support widths remain unchanged because all likelihoods are positive. Oracle information can shrink widths. These are distinct conclusions.

Positive controls pass: aligned reference-target uncertainty is determined by the predictive law; known affinely independent source laws plus the exact marginal identify source membership (8 LP checks). Six independent membership LP certificates agree with analytic endpoints.

The three failure witnesses are separate constructions, not an additive decomposition or a simultaneous independent three-component result. The reference-target failure requires reference/deployment mismatch. No natural-image or trained-estimator failure is established here.

## Artifacts

[Static intervals](target_intervals.png), [acquisition responses](acquisition_responses.png), [target table](target_intervals.csv), [predictive summaries](predictive_summaries.csv), [finite responses](finite_acquisition_responses.csv), [oracle responses](oracle_acquisition_responses.csv), [LP checks](membership_lp_checks.csv), [sufficient source controls](identifiable_source_controls.csv), [Probly check](probly_check.csv), [proof and sources](THREE_TARGET_CHARACTERIZATION.md), [executed notebook](executed.ipynb).

Remaining paper work: image-conditioned target-matched evidence under estimated source laws; transfer and containment checks; representation and discretization sensitivity; precise comparison with existing sufficiency/identification theory. Acquisition-policy superiority is not required by this characterization thesis.
