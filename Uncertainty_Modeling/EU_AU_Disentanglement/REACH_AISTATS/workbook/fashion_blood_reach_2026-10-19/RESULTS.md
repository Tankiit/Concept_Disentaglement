# Results — Fashion-MNIST and BloodMNIST

Completed 84 REACH calls (924 cell records), with the previous frozen model,
partition and references. MNIST cell assignments and entropy reproduce the
previous run; known-digit accuracy remains 98.04%.
Input pool IDs are disjoint within each dataset. Finite bounds, ordering and
width identities passed checks. Class status counts: {'optimal': 633, 'zero_mass_policy': 291}.
Captured warnings: 0.
Missing joint novelty bounds: 0.

## OOD diagnostics (separate from REACH)

| Novel domain | Entropy AUROC | Distance AUROC | Outlier cell fraction |
| --- | ---: | ---: | ---: |
| EMNIST Letters | 0.837 | 0.86426 | 31.1% |
| KMNIST | 0.907 | 0.94056 | 56.6% |
| Fashion-MNIST | 0.945 | 0.87772 | 22.1% |
| BloodMNIST grayscale | 0.908 | 0.99996 | 99.9% |

The first two rows are prior-run results with the same frozen model. Distance
is the nearest-class Mahalanobis distance used by the fixed LDA partition.
Blood images are almost perfectly ranked above digits by this score. This
can reflect background/texture as well as content; it is not evidence of
medical understanding. Fashion entropy separates well, while most fashion
images still fall below the fixed outlier threshold. AUROC measures ranking,
so good AUROC need not imply high detection at that particular threshold.

## REACH result

At the baseline (n=1000, m=200, r=50), mean joint novelty width is approximately
1 in all four new scenarios. The outlier cell also has width 1. No declared
mixture incompatibility is flagged in any of the 84 runs.

With m increased to 1000, mean known-cell joint novelty widths at 5% novelty
are about 0.9792 (Fashion) and 0.9829 (Blood). At 20% novelty they remain about
1. W_label decreases from about 0.6404 to 0.4251, as in the previous pilot.
Increasing n or r alone does not materially resolve the broad joint intervals.

Thus changing to more visually distinct datasets did not solve the precision
problem at these budgets. Strong feature-distance OOD ranking coexists with
uninformative REACH bounds. High observed interval coverage is largely due
to wide intervals and does not validate the mismatched EMNIST novelty model.

These widths describe cell probabilities, not per-image medical diagnoses.
The source references are still EMNIST, so Fashion and Blood are deliberately
undeclared novelty. Obtaining matching references would be a different
information-access condition and has not been tested here. Likewise an RGB
medical condition, intensity-matched controls, more informative reference
budgets, and oracle-conditional diagnostics remain untested.

Only three simulation seeds and one trained encoder are used. No claim of
population coverage or general OOD superiority is made from this pilot.
