# First execution stage — 2026-10-19

Completed the first stage specified in the plan: cell mapping and four-pool
bookkeeping. Feature diagnostics, model fitting, and the REACH sweep remain
pending. The original notebook and original project modules are unchanged.

## What ran

`inspect_pools.py` verifies SHA-256 checksums of the downloaded CIFAR-10
train/test Parquet files and reads their actual labels. It calls the original
`data.pools.load_pools()` and `check_disjoint()` functions with data access
substituted locally: X contains original row IDs as markers, not CLIP features.
This verifies the original pool selection and row/label alignment without
claiming to have executed feature-dependent calculations.

Reproduce from REACH_AISTATS with reach-aistats active:

```bash
python workbook/understand_features_notebook_2026-10-16/inspect_pools.py
```

## Verified pool sizes

| Pool | 200 references/class | Full reference | Source |
| --- | ---: | ---: | --- |
| Development | 10,000 | 10,000 | Training split |
| Reference | 2,000 | 40,000 | Training split |
| Deployment log | 5,000 | 5,000 | First test half |
| Evaluation | 5,000 | 5,000 | Second test half |

Assertions passed for development/reference disjointness, log/evaluation
disjointness, feature-marker/row/label alignment, exhaustive test partition,
and exhaustive development/full-reference training partition. Row identity
is (source dataset, row index), not index alone across train and test.
These checks establish bookkeeping, not statistical independence or absence
of duplicate image content across the original datasets.

## Comparison with saved notebook output

Log class counts match exactly: [488,505,512,497,507,488,491,495,504,513].
Evaluation class counts also match exactly.

Full-reference counts differ:

- Current: [4018,3964,3991,3950,4016,4006,4014,3990,4001,4050].
- Saved notebook: [3971,4002,3976,4025,3970,4032,4023,4008,3983,4010].

Both have 40,000 rows, but exact historical training-row assignment is not
reproduced. The original CLIP training cache/order and historical runtime are
unavailable, so the cause is unresolved; possible differences in row ordering
or historical execution must not be presented as verified explanations.

## Prerequisites still missing

The inventory currently contains only the ResNet-18 test-feature cache.
Missing original inputs: CLIP train features, CLIP test features, and
CIFAR-10-C brightness features. We now have the raw clean training/test data
needed for local CLIP extraction, but did not extract embeddings in this stage.
The corruption alignment check is reviewed in the cell guide but was not run.
Matching class labels at an index is a useful check, not proof that two arrays
contain the same underlying image; provenance/order are still necessary.

## Artifacts

- CELL_GUIDE.md: cells, dependencies, writes, and implementation findings.
- pool_output.txt: captured bookkeeping execution.
- pool_summary.csv: sizes and per-class counts for both configurations.
- pool_assignments.csv: assignments identified by configuration/source/index.
- pool_provenance.json: input URLs, checksums, and pool-code checksum.
- cache_inventory.json: actual available feature cache schema.

Next stage: obtain or regenerate explicitly documented CLIP inputs, then run
the feature diagnostics and one audited REACH configuration before the sweep.
The independent reach_first_try plan remains pending.
