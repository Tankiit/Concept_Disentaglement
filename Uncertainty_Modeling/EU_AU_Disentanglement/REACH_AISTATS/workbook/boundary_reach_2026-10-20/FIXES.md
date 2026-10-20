# Collaborator experiment corrections — 2026-10-20

The REACH branch through 8cdf0ec was merged into reach-mateus-workbook. Earlier
workbook artifacts and the L2/context fixes remain intact. The original notebook
with historical outputs is available from that commit; corrected notebook outputs
are deliberately empty because the CLIP `features.pt` cache is unavailable locally.
The local timm/ResNet cache is not substituted and labeled as CLIP.

## Fixed

* Five-epoch MLP now steps its own optimizer. A regression executes the actual
  notebook training block and verifies that the head parameters change.
* Aggregate novelty comparisons explicitly exclude KUC; per-class comparisons
  exclude the other UUC classes. The low-EU subset actually filters rows.
* MC sampling activates only dropout, freezes batch-normalization statistics,
  and restores every module's previous mode. Deterministic predictions and
  attacks no longer inherit active dropout from an earlier cell.
* The main head is returned to evaluation mode. Seeds are set before model
  initialization. The example passes the actual feature dimension to its head.
  Optional FAISS is no longer required merely to import/run the example; exact
  sklearn neighbors provide the same Euclidean-distance interface when absent.
* PGD enforces a Euclidean norm budget at each step without a final normalization
  that could move the returned feature outside the budget. Radial and spherical
  diagnostics remain explicitly separate. The depth sweep uses the exact nearest
  linear decision plane and signed logit margins, not a probability-margin formula
  that becomes zero for a second-place true class.
* Pair-margin controls use the true class and a distinct competitor; invalid
  zero-step ratios are marked undefined. Prediction flips mean a change from
  the clean prediction, not simply a negative true-label margin.
* Clean/attacked bootstraps resample corresponding images together. Pooled-depth
  comparisons resample image IDs as clusters. The symmetry table's "0 in CI?"
  boolean now has the correct orientation.
* Superiority comparisons bootstrap paired AUROC differences. They are labeled
  exploratory, conditional on the trained models and not multiplicity-adjusted.
  CV fold-seed ranges are labeled as ranges, not confidence intervals. The
  notebook no longer treats dropout EU as independent information beyond TU and
  AU: EU is exactly TU minus AU. Bootstrap EU is assessed separately.
* Added all-output margin/MSP baselines alongside the known-only variants.
* Thresholds for the confidence/EU quadrants come from validation KKC rows.
  The notebook reports both warnings and possible missed warnings; neither
  intersection is claimed to contain 10% of known rows by construction.
* `metrics.fill_scores` now returns row-bootstrap intervals for the reported
  group statistic, conditional on fixed members. The previous member-quantile
  intervals were not confidence intervals for that statistic. The test for
  deterministic disagreement now checks that its constant log(2) score also has
  a log(2) interval, rather than the misleading zero jackknife spread.
* The joint-count prototype now checks actual feasible cell mass before applying
  its zero-support policy. A zero log lower bound alone does not imply a feasible
  zero denominator. It also supports global residual mass (`cell=None`) for the
  new boundary experiment; this is distinct from per-cell posterior intervals.

## Validation scope

All 54 targeted tests pass: 28 core REACH, 3 geometry/context, 8 metrics,
9 uncertainty-audit, and 6 joint-region tests. Saved experiment checks also pass
for all 500 inference runs and 5,000 scored rows: valid finite intervals, clean
controls, disjoint fitting/calibration IDs, and the expected deep wrong predictions.

Core REACH, geometry/context, metrics, uncertainty-audit, and joint-region suites
are run separately from unavailable CIFAR-cache integration tests. Regression
controls cover exact boundary crossings, attack budgets, dropout/BatchNorm state,
class/subset masks, paired and clustered resampling, optimizer updates, and
notebook compilation. The corrected example is smoke-tested on small generated
features. The full original CLIP notebook is not claimed to have been rerun.

The completed MNIST boundary experiment is documented in RESULTS.md. It supplies
new evidence using an available frozen encoder and explicitly records its own
head, calibration, finite-pool sampling assumptions and limitations.
