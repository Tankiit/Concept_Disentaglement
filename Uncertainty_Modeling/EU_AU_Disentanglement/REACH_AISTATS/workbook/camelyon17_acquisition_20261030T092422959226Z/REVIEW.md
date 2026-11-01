# Initial WILDS acquisition implementation — 2026-10-30

Camelyon17 image and counterexample notebooks executed successfully. FMoW
notebooks and both local-data adapters are implemented and fixture-tested, but
**FMoW has not been run on real data**: its download is pending with the user.

## Camelyon17 real-record counterexample

Run: `workbook/camelyon17_counterexample_20261030T092024988890Z/`.
Of 600 training information states, 39 pairs met fixed matching criteria.
The selected pair's MI relative difference is **0.0170009%**. Mean predictions
are 0.578622 and 0.578479. Both concern the same binary population prevalence.

| Selected state | Expected held-out MSE after one identity | After one target label | Preferred acquisition |
|---|---:|---:|---|
| Identity-preferred | 0.04568537 | 0.04629771 | Source identity |
| Label-preferred | 0.04633004 | 0.04577468 | Target label |

Both prespecified held-out advantages are positive. The 2,000-repeat acquisition
curves also retain the expected ordering through budget 128 in this run.
This supports a selected real-record analogue of the toy mechanism under the
declared Beta model and revelation protocol. It is not an exact-equality theorem,
a neural prediction counterexample, or an estimate of how common such states are.
The counterexample conditions on recorded hospital and tumor fields, not pixels.

## Camelyon17 image experiment

Final diagnostic run: `workbook/camelyon17_acquisition_20261030T092422959226Z/`.
18,000 selected images, frozen ImageNet ResNet18, matched CE/DAPPr heads,
30 training epochs, 30 Probly MC passes. Original data hashes were checked
against all 21 files in the pinned public mirror. This confirms mirror integrity,
not independent equivalence to the unavailable official download archive.

| Score | Domain OOD AUROC |
|---|---:|
| MC predictive MI | 0.6897 |
| MC probability envelope width | 0.6853 |
| DAPPr original EU | 0.5973 |

These are pilot scores on custom patient-disjoint splits, not official WILDS
benchmark numbers, a reproduction of the original DAPPr architecture, or evidence
that the source bounds outperform either method.

At additional budget 2,048, mean known-source membership width decreases from
1.000 to 0.823 under identity acquisition. Candidate membership width decreases
from 0.986 to 0.721. Target-label acquisition decreases reference target width
from 0.859 to 0.623. The probabilities concern separate events and have distinct
acquisition assumptions. No additive decomposition is implied.

**The bounds are still broad and transfer is imperfect.** Empirical source-cell
total-variation differences between reference and evaluation are 0.3115, 0.1995
and 0.3635 for hospitals 0, 3 and 2 respectively. At budget 2,048 under source-ID
acquisition, known-membership intervals contain the fixed empirical held-out
target in 17 of 20 cell/repeat checks (85%). This is a descriptive containment
diagnostic, not estimated population confidence coverage. It prevents interpreting
shrinking widths as universally valid improvement under patient shift.

No feasible sets were empty in this run; 30.5% of all interval queries used the
explicit [0,1] policy because conditioning mass could be zero. The minimum
conditioning mass and missing-cell behavior are therefore material limitations.

The custom evaluation split has only one patient in hospital 0, two in hospital 3
and one in hospital 2. These image counts do not provide thousands of independent
patient observations. Multi-split, multi-seed validation and a richer treatment
of within-source patient variation are needed before a CVPR-scale claim.

## Implementation checks

18 tests passed, covering group separation, label-independent split assignment,
both FMoW file formats, cell-conditional target bounds, posterior formulas,
credible enclosure extrema and finite DAPPr gradients. All four notebooks pass
schema validation and code-cell compilation. Each code cell has a preceding
narrative and a goal. The two completed runs include executed notebooks,
observations, models/posteriors, figures, package versions and artifact hashes.

Run instructions and FMoW folder requirements are in `WILDS_EXPERIMENTS.md`.
