# REPRODUCE_CVPR.md — reproducing every CVPR-line experiment

Verified end-to-end on 2026-11-10 (this machine, this environment). Nothing in
`results/` was overwritten: every re-run went to a scratch directory and was compared
against the recorded outputs. `verify_cvpr_repro.py` (beside this file) re-does the
whole check in one command.

## 1. Scope: what the CVPR experiments are

Both live CVPR lines share one code root (`REACH_AISTATS/` on branch `REACH`):

| # | experiment | entry point | recorded outputs | feeds |
| - | ---------- | ----------- | ---------------- | ----- |
| 1 | Epistemic-set readouts (headline) | `epistemic_sets.py` | `results/epistemic_sets/` | epistemic-set line |
| 2 | Credal-head baseline arm | `credal_head.py` | `results/credal_head/` | epistemic-set line |
| 3 | Step-0 reconciliation (negative-set finding) | `step0_reconcile.py` | `results/step0_reconciliation/` | both lines (discipline) |
| 4 | WRN50 envelope-width arm (6 plain-mode runs) | `example_wrn50.py --mode plain` | `results/oss_metrics{,_fmnist,_svhn,_cifar100,_svhn_id,_svhn_id_mnist}/` | envelope exploratory arm |
| 5 | WRN50 stability + path diagnostics | `example_wrn50.py --stability / --paths` | `results/oss_metrics/wrn50_stability.json`, `wrn50_paths*` | envelope exploratory arm |
| 6 | wrn_compare smoke (transformed-vs-clean) | `python -m wrn_compare.run --smoke` | `results/wrn_smoke/` | adjacency checks (no scientific claim) |
| 7 | Finite-world novel controls | `novel_variations.py` | in-memory tables (no files) | controls |
| 8 | Notebook protocol home | `eu_blindspot.ipynb` | in-notebook | exploratory (superseded by 3) |
| 9 | Unit suite | `python -m pytest tests/` | — | all |

The AISTATS-era widths code (`main.py`, `gates.py`, `models/`, `lean/`, `real/probability_bounds.py`
on the workbook branch) is out of scope for CVPR and not covered here.

## 2. Environment

Conda env `torch-multimodal` (the only env on this machine with the full stack; identified
from `__pycache__` tags cpython-312 and the notebook's `language_info`):

```
python      3.12.4
torch       2.10.0
pytorch_ood 0.2.1
probly      0.9.1
scikit-learn 1.5.1
numpy       2.2.6
pandas      2.3.2
scipy       1.16.2
matplotlib  3.10.8
timm        1.0.21
faiss       1.9.0
pytest      8.1.1
```

These satisfy `requirements-novel.txt` (minimums). `OMP_NUM_THREADS=1` is required —
torch+sklearn libomp pools explode to thousands of threads otherwise (noted in the plan,
24 Oct). All scripts pin `torch.set_num_threads(1)` themselves; export the variable anyway.

## 3. Feature caches (inputs — NOT in git)

`cache/` is git-ignored (`EU_AU_Disentaglement/.gitignore`), so **no clone brings the
caches**; copy them out-of-band or re-extract. Pinned SHA-256 (also enforced by
`verify_cvpr_repro.py`):

| file | sha256 | what it is |
| ---- | ------ | ---------- |
| `cache/features/features.pt` | `814295a9…54d3` | CLIP ViT-B/16, 512-d, L2-normalized; keys `cifar10_train`/`cifar10_test` + 3 CIFAR-10-C corruptions. Used by `step0_reconcile.py`, notebook. |
| `cache/features/cifar10_test.npz` | `85144428…2156` | same CLIP features as `features.pt['cifar10_test']` (verified identical, max abs diff 0.0). Used by `epistemic_sets.py`, `credal_head.py`, `example.py`, `oss_from_cache.py`. |
| `cache/features/cifar10_train.npz` | `c23e78a3…3367` | CLIP train features (10k-row npz family). |
| `cache/features/timm_wide_resnet50_2_{c10test,mnisttest,fashionmnisttest,svhntest,cifar100test}.npz` | pinned in `verify_cvpr_repro.py` | timm `wide_resnet50_2.racm_in1k` features for the WRN50 arm. |

Regeneration routes (weights all cached locally: `timm/wide_resnet50_2.racm_in1k`,
`timm/resnet18.a1_in1k` in the HF cache; DINOv2 ViT-S/14 in the torch-hub cache):

- CLIP cache: `features.ipynb` / `data/features.py` (or the Modal extractor
  `real/modal_extract.py` — the original provenance; do not `modal volume get` feature
  files, 12/17 local CIFAR-10-C NPZ transfers are truncated).
- timm caches: `example_wrn50.py` extraction path or `example.py --source timm --model resnet18`.
- Raw datasets live under `/Users/tanmoy/data` (CIFAR-10/100) and `cache/{cifar10,mnist}`.

## 4. One-command verification

```
OMP_NUM_THREADS=1 python verify_cvpr_repro.py            # inputs + suite + experiments 1-3
OMP_NUM_THREADS=1 python verify_cvpr_repro.py --wrn50    # + WRN50 arm (6 runs, ~10 min)
```

Writes only to a scratch dir (default: tempdir; `--out` to choose). Exit code 0 = all
checks pass. Verdicts: `IDENTICAL` (byte-equal), `~TOLERANCE` (numeric match, atol 1e-9 /
rtol 1e-5), `FAIL`.

## 5. Verified outcomes (2026-11-10)

| experiment | verdict | detail |
| ---------- | ------- | ------ |
| test suite | **147 passed** | 11.4 s; matches recorded count |
| `epistemic_sets.py` | **byte-identical** | all 5 files (rows.csv 26,500×23, summary, per_class, summary_agg, manifest) |
| `credal_head.py` | **byte-identical** | all 10 files incl. PRED verdicts and the built-in `verify_against_delivered` (max abs diff 0.00e+00 vs delivered rows) |
| `step0_reconcile.py` | **byte-identical** | summary.json; head-seed/world-seed spreads reproduced (0.975/0.938/0.982/0.902) |
| WRN50 plain ×6 | **within tolerance** | headline AUROC deltas ≤0.005 (svhn_id: 0.013); rank-disagreement diagnostics looser (see §6) |
| WRN50 `--stability` | **within tolerance** | std ±0.001–0.002 AUROC at T=30 (recorded ±0.002); mean width monotone in T: 0.134/0.186/0.240 (recorded 0.140/0.193/0.249) |
| WRN50 `--paths` | **completes; diagnostics head-dependent** | EU/distance/margin trajectories reproduce in structure; per-pair margins move ≤0.08 across head retrainings (16 of 132 numeric fields exceed 0.05). Requires `HF_HUB_OFFLINE=1` (weights cached) and an extracted CIFAR root (`--data-root /Users/tanmoy/data`); with the default `cache/cifar10` (tarball only, unextracted) + online hub check it can hang on network |
| `wrn_compare.run --smoke` | **data tables byte-identical** | rows/summary/summary_per_seed/clean_accuracy/density_minus_EU/pool_assignments; manifest differs only in `out`/`device` fields; `validation.json` is a hand-written session record, not a script output |
| `novel_variations.py` | **deterministic** | double-run check: identical row ids; max numeric diff 0.0 |
| `eu_blindspot.ipynb` | executes end-to-end (nbconvert, 51 cells) | cell 8's signature leak value **0.664 reproduces exactly** despite the unseeded head (the leak dominates head init); exact cell-6-dependent numbers are NOT bit-reproducible by design — documented in the `step0_reconcile.py` header; step-0 is the reproducible replacement and brackets the notebook's readouts (head-seed spread 0.975–0.979) |
| `metrics.py` | regenerable | writes `results/oss_metrics/metrics.json` in place — move the recorded file first if byte-comparison is wanted; exploratory skeleton, superseded by 3 |

Runtime on this machine (M-series, CPU/MPS): experiments 1–3 ≈ 100 s total; each WRN50
plain run ≈ 15–30 s; stability ≈ 25 s; suite 11 s.

## 6. Known nondeterminism — do not chase these

1. **WRN50 head-training nondeterminism (mps AND cpu).** Recorded in the plan (24 Oct:
   "the 0.011 run-to-run gap is head-training nondeterminism, not dropout draws").
   Re-verified today: an mps repeat moved pooled AUROC by 0.004–0.006, and two identical
   CPU invocations were *not* byte-identical (0.9041 vs 0.9059). Expectation for
   reproduction: headline AUROCs within ~±0.005 (worst observed 0.013 on svhn_id);
   order-statistic diagnostics (`rank_disagreement.max_gap`, `n_divergent`) move more.
   The arm's *conclusions* (EU–width pooled ties, CIs straddling zero off-MNIST) are stable.
2. **Notebook cell 6** built the K+1 head before any `torch.manual_seed`, so its exact
   numbers came from an unreproducible ambient RNG state. `step0_reconcile.py` rebuilds
   the protocol with seeding fixed and shows the notebook's readouts sit inside the
   head-seed spread (0.975–0.979). Quote step-0, not the notebook.
3. **Local CIFAR-10-C NPZs are truncated** (12/17 files) — CIFAR-10-C is not part of the
   core CVPR protocol; anything touching it runs on Modal (`real/modal_extract.py`).
4. Figures (`comparison.png`, `wrn50_scatter.png`, …) are rendering outputs and are not
   byte-compared; the data tables behind them are.
5. The DINOv2 hardness arm (parked CVPR line) computes on the Modal volume `reach-aistats`;
   its weights are cached locally but the extraction pipeline is not local.
6. **`--paths` network hang:** `example_wrn50.py --paths` loads the timm backbone with
   `pretrained=True` and reads raw CIFAR-10 from `--data-root`. With the hub reachable-but-blocked
   and `cache/cifar10` holding only the tar.gz, the run sat idle >12 min. Fix (verified):
   `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` + `--data-root /Users/tanmoy/data`
   (extracted `cifar-10-batches-py`) → completes in ~20 s.

## 7. Fresh-repo checklist (the CVPR carry)

1. Clone branch `REACH` of this repo; sparse-checkout `REACH_AISTATS/` (the
   `reach-mateus-workbook` branch is the tutorial/ICML snapshot and does NOT contain the
   CVPR code).
2. Create the env from §2 (conda or venv; `requirements-novel.txt` for minimums).
3. Copy the §3 caches out-of-band (git-ignored) or re-extract.
4. `OMP_NUM_THREADS=1 python verify_cvpr_repro.py --wrn50` → expect §5.
