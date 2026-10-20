# CIFAR-10 dropout uncertainty — 2026-10-16

Goal: repeat the known-versus-unknown uncertainty experiment on real images,
then compare dropout disagreement and predictive entropy as novelty scores.

Run with the `reach-aistats` environment:

```bash
python workbook/cifar10_dropout_uncertainty_2026-10-16/run_experiment.py
```

The script downloads the CIFAR-10 test split from the uoft-cs/cifar10
Hugging Face mirror and pretrained timm ResNet-18 weights on first use.
The mirror file is verified against its SHA-256 digest. The original
Toronto archive download was stopped because it was unusually slow. Frozen features for all 10,000 official CIFAR-10 test images are cached
in `cache/features/timm_resnet18_c10test.npz`. The source image download is
stored in `cache/cifar10/test.parquet`. Install the additional reader with
`python -m pip install -r workbook/cifar10_dropout_uncertainty_2026-10-16/requirements.txt`. This follows the original example's convention:
the official test set is repartitioned into our own train/validation/test
sets. Results are for this open-set protocol, not the standard CIFAR-10 benchmark.

With seed 0, the three known classes are airplane (0), automobile (1), and
ship (8). The two validation-only unknown classes are deer (4) and frog (6).
The five test-only unknown classes are bird (2), cat (3), dog (5), horse (7),
and truck (9). This refers to what the small head sees; the pretrained
ImageNet backbone may already encode these object categories.

Known classes contribute 2,100 training rows, 600 validation rows, and 300
test rows. Validation also contains 2,000 unknown rows; test contains 5,000.

The original `Head` and `train` functions train a small classifier for 40
epochs on known classes. The input dimension is read from the feature array,
and the initialization is seeded before constructing the head. Feature
extraction uses CUDA if available; the small head runs on CPU.

Scoring uses 30 dropout passes. The workbook explicitly restores evaluation
mode between batches to avoid the original `evaluate()` mode issue. Known
accuracy uses deterministic predictions with dropout disabled. Both EU and
predictive entropy rank unknown examples as positive for AUROC. No threshold
or hyperparameters are tuned on the test results.

Artifacts:

- `output.txt`: captured initial run log.
- `training.csv`: training loss per epoch.
- `head.pt`: trained head state dictionary.
- `scores.csv`: one row per validation/test example, retaining original cache index.
- `results.json`: configuration and aggregate results.
- `feature_provenance.json`: source checksum and pretrained model configuration.
- `uncertainty_comparison.png`: test score distributions and ROC curves.

Histograms use density normalization because the two groups have different
sizes. This is one seed and one choice of held-out classes; it does not
establish performance across other class assignments or data distributions.

## Observed results

| Metric | Validation | Test |
| --- | ---: | ---: |
| Known-class accuracy | 95.0% | 96.3% |
| Mean known EU | 0.01260 | 0.01140 |
| Mean unknown EU | 0.04349 | 0.04269 |
| EU AUROC | 0.8673 | 0.8332 |
| Predictive entropy AUROC | 0.8932 | 0.8380 |

Unknown examples have higher average dropout disagreement in this run.
Test EU ranks an unknown above a known example approximately 83.3% of the
time (ties count as half). Predictive entropy performs similarly; this
single run does not establish a meaningful advantage for either score.
The prior synthetic experiment used a different feature distribution and
training configuration, so the comparison is illustrative, not controlled.

Verified 7,900 unique validation/test row IDs, finite scores, and the identity
EU = predictive entropy - mean pass entropy. The splitter also checks that
training, validation, and test row sets are disjoint.

## Split tables

See [split_table.md](split_table.md) for counts by class and split,
`split_summary.csv` for the aggregate data, and `split_examples.csv` for
all 10,000 original row IDs with split assignments and available scores.
