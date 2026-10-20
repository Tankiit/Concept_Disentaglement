# Understand data splits — 2026-10-16

Goal: understand how `oss_from_cache.py` loads cached features, splits known and unknown classes, and builds traceable data batches before following their use in `example.py`.

Keep this session's notes, experiments, and results in this folder.

## Step 1: cached features and class splits

A cache is an NPZ file with `X` (N rows, d feature coordinates) and `y`
(N integer class labels). Each row represents one example. `load_cached()`
loads these arrays as float32 and int64. It does not extract image features.

`make_oss()` wraps the arrays in a TensorDataset, then delegates splitting to
PyTorch-OOD's DynamicOSS. There are two separate decisions:

1. Assign entire classes to KKC (known target classes), KUC (known outlier
   classes available during training), or UUC (classes withheld from training).
2. Partition rows within KKC and KUC classes across train/validation/test.
   Validation UUC and test UUC instead receive separate whole classes.

For ten classes, defaults `kuc=0, uuc_val=2, uuc_test=5` leave three KKC
classes, no KUC classes, two validation-only UUC classes, and five test-only
UUC classes. The `world['uuc']` field is the union of validation and test UUC
classes. The installed splitter selects disjoint validation/test unknown
classes, although the project's wrapper has a TODO about checking this.

The 70/20/10 proportions apply within KKC/KUC classes, not to the complete
input dataset. The installed splitter computes train and validation counts
from those fractions and gives test the remaining rows; it does not use
`test_size` directly in that calculation.

## Experiment 1: trace a synthetic dataset

Run from REACH_AISTATS:

```bash
python workbook/understand_data_splits_2026-10-16/inspect_splits.py
```

This uses 1,000 synthetic rows: ten classes, 100 rows per class, four arbitrary
features per row. It calls the original project's functions without changing
them. No image data or pretrained models are downloaded. Results are saved
in `split_output.txt` from the initial run.

| Split | KKC rows | KUC rows | UUC rows | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 210 | 0 | 0 | 210 |
| Validation | 60 | 0 | 200 | 260 |
| Test | 30 | 0 | 500 | 530 |

With seed 0, KKC classes are 0, 1, 8; validation UUC classes are 4, 6;
test UUC classes are 2, 3, 5, 7, 9. No rows overlap across splits (the wrapper's
assertions pass), and validation/test unknown class sets do not overlap.
These class IDs are arbitrary assignments, not semantic statements about CIFAR.

`make_loader()` produces dictionary batches:

- `X`: shape (batch_size, d), feature vectors.
- `y`: shape (batch_size,), original class IDs, not remapped classifier outputs.
- `index`: shape (batch_size,), row IDs in the original cache.
- Optional extra fields: per-row metadata supplied by keyword.

The experiment verifies that `batch['X']` equals the original `X` indexed by
`batch['index']`. This traceability lets later experiments attach scores or
region tags to the correct examples.

## Next step

Follow how `example.py` selects KKC training rows, remaps their original labels
to classifier output indices, and trains its dropout head. Then inspect the
uncertainty calculation and known-versus-unknown evaluation.

## Step 2: from original labels to classifier training

Run `python workbook/understand_data_splits_2026-10-16/inspect_training.py` from
REACH_AISTATS. The captured result is in `training_output.txt`.

The classifier has one output per KKC class. With seed 0 these classes are
0, 1, 8, so its three output indices mean:

| Original label | Output index |
| --- | --- |
| 0 | 0 |
| 1 | 1 |
| 8 | 2 |
| Any unknown class | -1 (sentinel, not a classifier output) |

The lookup tensor `lut` performs this translation. PyTorch cross-entropy
expects target indices in `0..k-1`; passing the original label 8 to a
three-output head would be invalid. Unknown rows must be excluded from this
training loss; -1 is not a valid target under the default cross-entropy settings.
The training loader selects KKC rows before calling `train()`.

The head maps features through Linear -> ReLU -> Dropout -> Linear. Its output
is a batch-by-k matrix of logits (unnormalized scores). Cross-entropy accepts
these logits directly. The optimizer updates only this small head; cached
features remain fixed and no image backbone is trained in this step.

For each batch, `train()` remaps labels, clears gradients, computes loss,
backpropagates, and performs an Adam update. An epoch visits the training data
once, and the reported epoch loss is weighted by each batch's row count.

The toy experiment uses ten-dimensional, deliberately separable class centres
with small noise. It explicitly sets `Head(d=10, hidden=32, k=3)` and uses the
original `train()` function. Training loss fell from 0.8845 to 0.1922 over 20
epochs, with 100% accuracy on 60 known-class validation rows. This demonstrates
the training plumbing, not performance on images or successful OOD detection.

Implementation details to remember:

- `example.py` constructs `Head(k=...)` with a default input dimension of 512.
  Other feature dimensions require explicitly setting `d`.
- `train(seed=0)` seeds PyTorch after head construction in the original example.
  To reproduce initial weights, seed before constructing the head, as this
  workbook experiment does.
- Training and feature extraction default to CPU in `example.py`; installing
  CUDA-enabled PyTorch does not automatically move this code to the GPU.

Next: inspect stochastic dropout predictions, entropy, and BALD uncertainty.

## Step 3: repeated dropout predictions and uncertainty

Run `python workbook/understand_data_splits_2026-10-16/inspect_uncertainty.py`
from REACH_AISTATS. Captured results are in `uncertainty_output.txt`.
The script trains a fresh head on the same synthetic data construction as
Step 2. Its random-number consumption differs from that walkthrough, so it
is a separate run, not a continuation of the identical trained weights.

For each input, `eu_bald()` performs 30 forward passes with dropout enabled.
Each pass randomly masks hidden activations; weights remain fixed. The model
returns logits, softmax turns them into probabilities, and stacking the passes
gives a tensor with shape (30, number_of_rows, number_of_known_classes).
`torch.no_grad()` disables gradient tracking; it does not disable dropout.

For probability vector p, entropy is H(p) = -sum_k p_k log(p_k).
The code computes:

    mean_prediction = mean_t p_t
    predictive_entropy = H(mean_prediction)
    mean_pass_entropy = mean_t H(p_t)
    EU = predictive_entropy - mean_pass_entropy

Natural logarithms make the units nats. EU measures disagreement among the
sampled predictions. The first term can be large even when the predictions
agree: repeated [0.5, 0.5] predictions have high entropy but zero EU.

| Controlled two-class example | H(mean) | Mean H | EU |
| --- | ---: | ---: | ---: |
| Both passes predict [0.99, 0.01] | 0.0560 | 0.0560 | 0.0000 |
| Both passes predict [0.5, 0.5] | 0.6931 | 0.6931 | 0.0000 |
| Passes predict [0.99, 0.01] and [0.01, 0.99] | 0.6931 | 0.0560 | 0.6371 |

The original script calls the difference EU/BALD. Treat it as a model-based
approximation of epistemic uncertainty. The average per-pass entropy is not
by itself a verified measurement of the dataset's irreducible label noise.
This calculation is separate from the REACH probability-width framework.

In this synthetic run, the prediction tensor had shape (30, 530, 3):

| Test group | Rows | Mean EU |
| --- | ---: | ---: |
| Known | 30 | 0.0569 |
| Unknown | 500 | 0.0641 |

The unknown-positive AUROC was 0.5707. It measures how often an unknown row
gets a larger score than a known row, counting ties as half a win. A value
of 0.5 corresponds to chance ranking; 1.0 means perfect ranking on these
samples. The modest separation here shows that this score need not be a
strong novelty detector. This is one synthetic run, not a general estimate.
Labels are used to evaluate the ranking, not to compute the EU score itself.
The workbook uses scikit-learn's AUROC implementation to avoid the original
example's slow nested Python pairwise comparisons.

Checks performed:

- The explicit entropy calculation matches original `eu_bald()` with the
  same batch and random seed.
- No model weights change while computing uncertainty.
- Repeated predictions are identical with dropout disabled for this head.
- Original `eu_bald()` leaves the head in training mode. Since `evaluate()`
  switches to evaluation mode only before its loop, subsequent batches can
  have classification loss computed with dropout active. The workbook sampler
  restores the previous mode; the original project implementation is unchanged.

Next: review the real-feature example's assumptions and decide which experiment
we want to reproduce before connecting dropout scores to the REACH widths.
