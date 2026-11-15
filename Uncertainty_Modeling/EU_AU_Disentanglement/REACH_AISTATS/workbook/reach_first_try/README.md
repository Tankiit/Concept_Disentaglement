# REACH first try

Plan recorded: 2026-10-16.
Status: planning only. No experiments have been executed for this workbook entry.

## Objective

Understand how REACH constructs probability intervals and which acquisitions
reduce their widths. Start with a controlled synthetic world before adapting
the analysis to our cached CIFAR-10 features.

REACH keeps the predictor or partition fixed and asks which deployment
distributions remain compatible with the available data, and how much their
posterior probabilities differ. Our preceding MC-dropout experiment instead
measured disagreement between stochastic predictions.

## 1. Trace one REACH calculation

Use the existing synthetic world with two known sources, one candidate novel
source, and four input cells. Source identity and the binary target label are
separate variables.

Trace the source distributions and deployment observations into a feasible
set of mixture weights, then compute the minimum and maximum posterior
probabilities at one cell. Display the true probability, interval, width,
and assumptions.

| Quantity | What varies |
| --- | --- |
| W_cls | Known-source posterior over plausible mixture weights |
| W_label | Known-source target-label probability over label-rate intervals, with reference weights fixed |
| W_unk | Candidate-novel-source posterior over plausible mixture weights |
| W_total | Target-label probability allowing the modeled uncertainties to vary together |

These widths are distinct estimands, not additive uncertainty components.
REACH does not estimate aleatoric uncertainty.

Relevant code: `data/worlds.py`, `models/ball.py`, `models/widths.py`.
Paths in this plan are relative to REACH_AISTATS.

## 2. Test responses to acquisitions

Hold the world and evaluation cells fixed. Initially use the true source
conditionals to isolate the acquisition effects. Vary one factor at a time:

| Intervention | Question |
| --- | --- |
| Increase unlabelled deployment budget n | Do W_cls and W_unk tighten as the mixture becomes better constrained? |
| Increase labelled budget per known source m | Does W_label tighten as label rates become better constrained? |
| Increase known symmetric label-flip rate eta | How does noisier supervision affect label uncertainty? |

Repeat across seeds, using paired observations or nested data budgets where
supported by the existing acquisition code. Report interval widths, changes
with budget, coverage of the true probabilities, and solver status. Inspect
trends rather than require monotonic contraction in every individual run.

Record the budget grid, seeds, confidence allocation, and choice of box or
exact L2 feasible set before running. Treat this first study as exploratory.
The existing `train/predictions.py` hypotheses are not guarantees.

Relevant code: `data/acquisitions.py`, `train/experiments.py`,
`train/predictions.py`, `test/inference.py`.

## 3. Examine non-identification

Compare the existing separated and confounded worlds. In the confounded
world, the candidate novel distribution is a mixture of known distributions.
Unlabelled data alone cannot distinguish novelty from changed known-source
proportions. Investigate what uncertainty remains as the log budget grows.

Interpret reducibility relative to the acquisition and modeling assumptions.

## 4. Estimate source conditionals from finite reference data

Replace oracle conditionals with estimates from a finite reference pool.
Study how reference and deployment budgets jointly influence intervals.
Trace the appropriate interval implementation before making a coverage claim:
the repository has several experimental variants, and adding a radius alone
does not validate every posterior interval.

## 5. Adapt to real CIFAR-10 features

Reuse the existing frozen ResNet-18 cache. Explicitly define:

- A fixed partition into cells.
- Independent development, reference, deployment-log, and evaluation pools.
- Known sources, binary target labels where required, and candidate novelty distribution.
- The acquisition intervention and evaluation criterion.

The preceding three-known-class dropout setup cannot be passed directly into
the two-known-source, binary-target prototype without defining this mapping.
The generic cached-data adapter expects separate arrays for these pools; its
assumptions must be checked, including the declared mixture family and target
structure. Evaluation labels must not inform fitting or acquisition choices.

Relevant code: `train/cache.py`, `real/head.py`, `data/pools.py`, and the
applicable real-data experiment modules after their assumptions are reviewed.

## Scope and first deliverables

Begin with steps 1 and 2 only when execution is requested. Save:

1. A worked single-cell calculation with assumptions and true probabilities.
2. A reproducible experiment configuration and script.
3. Raw per-seed results and an acquisition-response summary table.
4. Plots of widths against acquisition budgets, with coverage and solver diagnostics.
5. Interpretation of supported patterns, unresolved questions, and failures.

The supplied paper reports limitations and failed acquisition-selection
experiments. The aim is to reproduce and understand both successes and
failures, not assume that widths necessarily identify the best acquisition.
