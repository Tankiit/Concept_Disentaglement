# Understand features.ipynb — 2026-10-16

Status updated 2026-10-19: first execution stage (steps 1–2) completed.
See [EXECUTION_REPORT.md](EXECUTION_REPORT.md) and [CELL_GUIDE.md](CELL_GUIDE.md).
Original notebook cells and REACH experiments remain unexecuted; pool bookkeeping
was checked using real labels and row-ID markers.

## Goal

Understand the original notebook's data preparation, feature diagnostics,
REACH experiments, and interpretation of results. Distinguish saved historical
outputs from results we reproduce locally.

## 1. Map cells and check prerequisites

Create a cell-by-cell guide showing inputs, outputs, dependencies, and file
writes. Notebook cell numbers below are zero-based positions, not execution counts.

- Cells 0–5: cache inventory, clean/corrupted features, normalization checks.
- Cells 6–9: disjoint data pools and exported datasets.
- Cells 10–14: visualization and nearest-neighbour diagnostics.
- Cells 15–19: controlled REACH acquisition experiments and width summaries.
- Cell 20: prevalence correction, accuracy comparisons, and crossover budget.

Inventory local files before execution. The notebook expects CLIP caches
`cifar10_train.npz`, `cifar10_test.npz`, and a brightness-corruption cache for
its corruption demonstration. Currently our available feature cache is
`timm_resnet18_c10test.npz`; it is not a replacement for the expected CLIP
train/test/corruption caches. Decide explicitly between obtaining the original
features and creating a documented adaptation. Do not silently rename caches.

The saved notebook outputs and claims about damaged local downloads refer to
an earlier environment. Treat them as historical evidence to inspect, not
verified facts about this machine.

## 2. Trace the four pools

Read `data/pools.py` alongside notebook cells 6–9. Build a table of the pools:

| Pool | Source | Purpose |
| --- | --- | --- |
| Development | First 10,000 rows of a seeded training permutation | Fit the frozen head/partition |
| Reference | Remaining training rows, optionally sampled per class | Estimate source distributions from labels |
| Deployment log | First half of official test images | Observe deployment distribution |
| Evaluation | Second half of official test images | Evaluate on separate images |

Trace original row IDs and verify disjointness. Explain the difference between
having ground-truth labels stored for experimental evaluation and using them
as supervision. Check which functions read which labels.

For corruption, inspect how the same log image indices map to severity blocks,
and what label alignment checks do and do not establish. Evaluation stays clean
in this demonstration; explain the resulting scope of the comparison.

## 3. Understand the feature diagnostics

Trace PCA/t-SNE and nearest-neighbour calculations. Distinguish exploratory
visualization from quantitative distances in the original feature space.

Compare the all-ten-class case with the labelled-classes-0–5 case. Explain
leave-one-out reference distances, the empirical 95th-percentile threshold,
and why its flagged fraction is not automatically the true novel fraction.

Review saved outputs first. Reproduce plots only after inputs are available
and execution is requested. Keep seeds and subsamples explicit; notebook RNG
state depends on cell execution order.

## 4. Trace one REACH run before a sweep

Read `real/head.py` and `real/audit.py`, then follow `build_context()` and one
`run_cell()` configuration. Record:

- Known classes 0–4 and novel classes 5–9 (different from the earlier six-known-class diagnostic).
- Frozen head, cell definitions, estimated source distributions, and mixture weights.
- Roles of n, m, delta, eta, and seed.
- Meaning of q_mode='oracle'; it does not imply every quantity is known exactly.
- L2 feasible set, uncertainty radii, interval endpoints, solver status, and coverage checks.
- Difference between plug-in and joint widths, especially W_unk versus W_unk_joint.

Inspect how log sampling works, including whether rows are sampled with
replacement when n exceeds the available pool. Verify how the notebook's
log-half restriction affects truth conditionals and evaluation assumptions.

Deliverable: a worked example connecting inputs to one cell's interval and width.

## 5. Review and, later, reproduce the controlled sweep

Notebook configuration: five seeds, delta in {0, 0.05, 0.2}, baseline
(n=1000, m=20, eta=0), and one-factor sweeps:

| Factor | Values |
| --- | --- |
| n | 100, 1,000, 10,000 |
| m | 5, 20, 100 |
| eta | 0, 0.1, 0.2 |

Separate hypotheses from observed outcomes. A predictions hash alone does
not establish historical preregistration. Record repeated baseline settings
and avoid counting them as independent replications.

Inspect width distributions, coverage, solver diagnostics, and per-seed changes.
The saved output includes a solver-accuracy warning, nonmonotonic W_cls behavior,
and near-unit W_unk_joint. Investigate these rather than assuming success.
A broad interval covering truth may still provide little useful information.

## 6. Audit the final performance analysis

Before interpreting or reproducing the final cell:

- Check the sign of the quantities named excess_risk: the notebook subtracts
  oracle accuracy from method accuracy, reversing the usual excess-error difference.
- Check whether the fixed evaluation set represents each simulated deployment
  mixture, or whether reweighting is required for the intended comparison.
- Clarify what is oracle: true weights combined with estimated conditionals do
  not automatically constitute a Bayes-optimal predictor.
- Check class-ID versus output-index assumptions in accuracy calculations.
- Treat the W_cls <= W_label crossover as an exploratory comparison of distinct
  estimands, not a validated acquisition decision rule.

Document findings before proposing any changes to the original notebook.

## Outputs for a future execution

Save the cell guide, input/provenance inventory, pool table, one-cell REACH
walkthrough, execution configuration, raw results, plots, and interpretation
in this folder. Preserve the original notebook's saved outputs and distinguish
any adapted experiment from a reproduction.

Initial execution scope was steps 1 and 2 only; completed on 2026-10-19. The separate
`workbook/reach_first_try/` plan remains pending.
