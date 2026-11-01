# Initial WILDS acquisition implementation — 2026-10-30

Camelyon17 and FMoW image and counterexample notebooks executed successfully.
FMoW was loaded from `/home/mtleao/projects/Concept_Disentaglement/fMoW`:
32 readable Parquet files, 141,696 rows, 14,612,709,837 bytes. Local SHA-256
hashes are archived; they were not independently compared against upstream hashes.
Mixed ISO timestamp precision required an adapter fix, covered by a regression test.

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

## FMoW real-record counterexample: prespecified endpoint failed

Run: `workbook/fmow_counterexample_20261030T094929883846Z/`.
Of 600 states, 32 pairs met the fixed matching criteria. The selected pair's
MI relative difference is **0.0643531%**. The binary event is class 42,
selected as the modal known-source training class before held-out evaluation.

| Model-selected state | Expected held-out MSE after one identity | After one target label | Empirical preference |
|---|---:|---:|---|
| Identity-preferred | 0.00019807175 | 0.00017691726 | Target label |
| Label-preferred | 0.00002065736 | 0.00002140581 | Source identity |

Both acquisition advantages have the opposite sign to the prespecified
prediction. **The held-out confirmation failed.** Although the empirical best
actions differ, we do not relabel the states after seeing held-out outcomes or
present that as passing the chosen endpoint. No thresholds or pair selection
were changed. Posterior Bayes action values are expectations under the declared
model, whereas held-out MSE depends on the actual empirical population and
initial estimation bias. These quantities need not rank actions identically.

## FMoW image experiment

Run: `workbook/fmow_acquisition_20261030T094932286096Z/`.
The same 18,000-image, frozen-encoder, 30-epoch and 30-MC-pass pilot ran on GPU.
The custom group-disjoint experiment concerns Americas/Europe versus Asia using
pre-2016 observations; it is not the official WILDS temporal benchmark.

| Score | Domain OOD AUROC |
|---|---:|
| MC predictive MI | 0.5737 |
| MC probability envelope width | 0.5458 |
| DAPPr original EU | 0.5605 |

At additional budget 2,048, source IDs reduced mean known-membership width from
0.999 to 0.884 and candidate-membership width from 0.987 to 0.860. Target labels
reduced reference target width from 0.778 to 0.324. Additional inputs alone had
little effect under this configuration. The remaining membership intervals are
still broad; none of these width changes proves superiority over neural scores.

Empirical source-cell total-variation differences were 0.0135, 0.0125 and 0.0145.
All evaluated intervals contained their corresponding empirical targets, but this
is descriptive containment with often-wide intervals, not proof of population
calibration. No feasible sets were empty; 11.67% of interval queries used the
explicit [0,1] policy because conditioning mass could be zero.

The two datasets therefore give mixed evidence: the selected Camelyon counterexample
passes its held-out endpoint, FMoW does not. Both show target-specific acquisition
responses within the declared bound model, with substantial remaining limitations.

## Implementation checks

18 tests passed, covering group separation, label-independent split assignment,
both FMoW file formats, cell-conditional target bounds, posterior formulas,
credible enclosure extrema and finite DAPPr gradients. All four notebooks pass
schema validation and code-cell compilation. Each code cell has a preceding
narrative and a goal. The four completed experiments include executed notebooks,
observations, models/posteriors, figures, package versions and artifact hashes.

Run instructions and FMoW folder requirements are in `WILDS_EXPERIMENTS.md`.
