# Original REACH width plots

Run `plot.py` with the reach-aistats Python environment. Source: the saved
`fashion_blood_reach_2026-10-19/reach_rows.csv`; no intervals are recomputed.
These plots describe the original encoder/partition and EMNIST-letter novelty
reference, including its mismatch with Fashion/Blood. They do not describe the
newer boundary partitions or their global prevalence intervals.

- `reach_widths.png` / `.pdf`: all five widths, averaged equally over the ten
  known-class cells within a seed, then equally over three seeds.
- `reach_widths_outlier.png` / `.pdf`: the separate outlier cell.
- `reach_width_sweeps.png` / `.pdf`: acquisition curves, one width per row.
  Shading is the min–max of three seed-level averages, not a confidence interval.
- CSVs preserve the per-seed averages and mean/min/max summaries.

Each width is upper minus lower on a probability scale. A width of 1 spans
the full [0,1] range. Heatmap annotations are rounded to three decimals.

| Width | Target and quantities varied |
|---|---|
| W_cls | Predicted-class probability conditional on the cell and known status; vary mixture weights with plug-in conditional entries. |
| W_label | Same known-class target; vary reference conditional entries at fixed uniform anchor weights. |
| W_joint | Same known-class target; vary both mixture weights and reference entries. |
| W_unk | Novelty probability conditional on the cell; vary mixture weights with plug-in entries. |
| W_unk_joint | Same novelty target; vary both weights and conditional entries. |

These widths are not additive EU/AU components. W_label is reference-estimation
uncertainty, not an estimate of irreducible aleatoric uncertainty. It is identical
across domains here because it uses the same known references and fixed anchor.

Baseline: deployment n=1000, known references m=200 per class, novelty references
r=50 per EMNIST letter. Each sweep changes one budget at a time. Increasing m
from 50 to 1000 reduces mean W_label from .812 to .425; most other widths remain
near 1. More deployment or letter-reference samples alone do not resolve this
original construction's broad intervals. This plot does not isolate the causal
contribution of representation quality versus uncertainty-region construction.

Validation: all 924 rows have finite widths in [0,1] within numerical tolerance;
the four widths with saved endpoints agree with upper-minus-lower to 1e-6.
