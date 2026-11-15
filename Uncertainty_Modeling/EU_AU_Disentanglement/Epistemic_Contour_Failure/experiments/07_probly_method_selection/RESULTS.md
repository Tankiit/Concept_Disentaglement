# probly: method selection and integration audit

Date: 2026-10-23. Status: repository/API discovery and initial primary-source review; no probly experiment executed or package installed.

## Decision

Adopt the user-suggested probly library as the candidate UQ implementation layer. Keep the agreed frequentist evaluation: fixed data-generating truths and repeated-sample contour error, coverage where defined, and uncertainty behavior. Using a library that includes Bayesian methods does not require adopting posterior inference as the study's framework. Start with ensembles; MC dropout is a possible additional method under evaluation, not a change of research philosophy.

This addresses experiment 06's need to identify contemporary UQ implementations, but does not by itself solve the mapping from predictive uncertainty to clean-contour uncertainty.

## What was verified

- The [repository README](https://github.com/pwhofman/probly) describes representation, quantification, and decomposition of predictive uncertainty; its description includes “decompose its predictive uncertainty into aleatoric and epistemic components.” This motivates using the library to inspect the actual epistemic quantities, not treating all confidence intervals as named epistemic estimators.
- The inspected [method exports](https://raw.githubusercontent.com/pwhofman/probly/main/src/probly/method/__init__.py) include `ensemble`, `dropout`, `evidential_classification`, `evidential_regression`, and several credal methods.
- The inspected [quantification exports](https://raw.githubusercontent.com/pwhofman/probly/main/src/probly/quantification/__init__.py) include `mutual_information`, `sample_variance`, `max_disagreement`, `SecondOrderEntropyDecomposition`, and `SecondOrderVarianceDecomposition`. Export presence is not a verified end-to-end adapter for this experiment.
- [Quantification dispatch source](https://raw.githubusercontent.com/pwhofman/probly/main/src/probly/quantification/_quantification.py) selects implementations by representation. Therefore `quantify(...).epistemic` must not be assumed to mean the same formula for every representation. Record the selected decomposition and formula in each run.
- The current interpreter does not have probly installed. Existing `REACH_AISTATS/example.py` already imports its dropout and quantification interfaces, but that is not evidence that the active environment contains it or that those old imports match the current version. Reuse library knowledge, not REACH's methodology.
- The README states Python 3.12 or later. Web views of main-branch files can reflect different crawl times; some retrieved exports differed across pages. Pin a release or commit and inspect that consistent checkout before runtime integration. No reproducible version has yet been selected, and no successful import is claimed.

## Proposed first comparison

| Layer | Initial choice | What is evaluated |
| --- | --- | --- |
| Member generation | Ensemble first | Variation between fitted models; document independent initialization versus bootstrap resampling |
| Quantification | Entropy/MI and variance decompositions where supported by the pinned representation | Exact formula and its target, not just a field named epistemic |
| Contour extraction | Derive each member's alpha-contour from its fitted distribution | Clean-contour error and variation of the contour location |
| Reference | Experiment 06's contamination-aware confidence set | Repeated-sample coverage under declared contamination assumptions |

Do not equate a deep ensemble with the earlier bootstrap refits: they differ in how member variability is generated. Also do not assume independently optimizing a simple identifiable Gaussian likelihood reproduces the behavior of a neural deep ensemble. Model architecture and training objective must be specified explicitly.

## Papers consulted, quotations, and usage

1. Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. [Original paper abstract](https://arxiv.org/abs/1612.01474). The authors describe “an alternative to Bayesian NNs”. Used to prioritize an ensemble-based starting point compatible with the user's concern about a Bayesian-centered detour. Only the abstract was consulted in this step; exact architecture/training choices require reading the method before implementation. This source does not assert that a generic ensemble spread is a clean-contour confidence band.
2. Yarin Gal and Zoubin Ghahramani (2016), *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. [Original paper abstract](https://arxiv.org/abs/1506.02142). The authors characterize dropout training as “approximate Bayesian inference in deep Gaussian processes”. Used to label MC dropout's theoretical origin transparently if included. Testing it with frequentist coverage does not make our evaluation Bayesian. Only the abstract was consulted; no contour coverage guarantee is attributed to it.

The software links above are implementation sources, not research-paper evidence. The clean-contour target and information restrictions come from our existing proposal, not from these papers.

## Required bridge to the existing experiment

Each fitted member must produce a normalized distribution or a contour estimate from the permitted observations. For a Gaussian member theta_b, both its cell probabilities p_j(theta_b) and its clean-model alpha-contour are available. They answer different questions:

- Entropy/MI across cell probability vectors concerns disagreement about the cell of a future observation.
- Variation of extracted contours concerns variation of estimated contour location.
- A member's deterministic inside/outside contour indicator gives a different target again; it is not an observed class label.

Experiment 02 already shows why these distinctions matter. Report the predictive epistemic score alongside contour error and contour variability; do not assert their equivalence. A scalar score has no nominal coverage until an explicit, justified set construction is supplied. A 95% fraction of ensemble members is not automatically a 95% frequentist confidence set.

Do not manufacture supervised clean-contour labels from evaluator truth. Those labels would add information absent from the current finite-cell source-mixture setup. A classification or regression adapter requiring such labels must be treated as a different experiment. Prefer a density-estimation interface if available, or explicitly document a custom adapter and the difference from the original method.

## Next executable step

1. Pin probly and create an isolated environment; validate representation and decomposition APIs with small known probability vectors.
2. Read the chosen ensemble method and backend implementation at that version, and the original method paper beyond the abstract.
3. Specify an unsupervised grouped-density model and permissible member-generation mechanisms. Confirm that the intended library method supports it; otherwise document the required adapter before claiming a probly method replication.
4. Verify contour extraction on the clean control, then repeat the existing fixed-truth contamination evaluation. Include optimization variation and finite-member sensitivity.

No installation, model training, coverage results, or numerical comparison are claimed in this audit.

## Follow-up status

[Experiment 08](../08_probly_adapter/RESULTS.md) completed isolated installation of probly 0.9.1, inspected the pinned API and original ensemble method beyond the abstract, validated quantification against known formulas, and ran a Gaussian-bootstrap adapter pilot. The no-install/no-execution statements above describe this initial audit only. The neural ensemble factory integration remains pending; Gaussian bootstrap results are not a deep-ensemble replication.
