# Execute the original scripts — 2026-10-16

We executed the original `example.py` unchanged and directly executed
`oss_from_cache.py`. We also called the latter's loader functions through a
small workbook driver. The separate REACH plan remains pending.

## Reproduce

From REACH_AISTATS, with the reach-aistats environment active:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u example.py --source timm --model resnet18
python oss_from_cache.py
python workbook/original_scripts_2026-10-16/inspect_original_loaders.py
```

The thread settings limit CPU parallelism. The first command reuses our
existing `cache/features/timm_resnet18_c10test.npz`; it does not download or
extract again when that file exists. Using `--source cache` instead would
look for the absent CLIP cache `cache/features/cifar10_test.npz`.

## Reading example_output.txt

1. Two cache summaries appear because both the extraction/cache-check helper
   and the split helper load the same file. This does not mean training twice.
2. The script creates known/unknown splits and trains one dropout head on the
   three known classes for 40 epochs. The printed loss samples are epochs
   1, 6, 11, ..., 36, followed by epoch 40. Loss fell from 0.9882 to 0.0686.
3. The split/group table reports classification loss and average dropout EU.
   Unknown-group loss is NaN by design: those labels have no classifier output,
   so no classification loss is accumulated for them.
4. The final lines report unknown-positive EU AUROC: validation 0.862, test 0.846.

| Split | Group | Loss | Mean EU | Rows |
| --- | --- | ---: | ---: | ---: |
| Validation | Known | 0.1641 | 0.0136 | 600 |
| Validation | Unknown | Not defined | 0.0436 | 2000 |
| Test | Known | 0.0833 | 0.0113 | 300 |
| Test | Unknown | Not defined | 0.0433 | 5000 |

This original entry point does not report accuracy or predictive-entropy
AUROC, and does not save a checkpoint or per-example scores. The workbook
captures its printed output.

Its output need not exactly match the previous workbook experiment: the
original seeds training after head initialization, uses different evaluation
batching/RNG consumption, and leaves training mode enabled inside eu_bald().
In particular, later known-class validation batches can compute loss with
dropout active. We preserved that original behavior for this execution.

## Reading loader_output.txt

Running `python oss_from_cache.py` succeeds without output because the file
defines functions and classes but has no executable experiment entry point.
`oss_direct_output.txt` is consequently empty.

`inspect_original_loaders.py` calls the original `group_loaders()` on the real
feature cache and prints the first batch from each nonempty split/group.
Each batch holds X (features), y (original labels), index (original cache row),
and group (0 = KKC, 1 = KUC, 2 = UUC). Empty KUC loaders are expected with kuc=0.
This driver inspects data loading only; it trains no model.
