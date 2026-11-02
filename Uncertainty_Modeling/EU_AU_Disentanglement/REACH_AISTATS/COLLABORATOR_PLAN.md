# Collaborator plan — CVPR 2027: scoring hard unseen classes beyond AUROC

Status: working draft, 19 Oct 2026; protocol concretised 20 Oct 2026. Everything marked
exploratory comes from one seed on cached CIFAR-10 features and has not been replicated.
The previous occupant of this file (the self-contained synthetic-experiments plan, REACH
AISTATS era) is archived unchanged at `COLLABORATOR_PLAN_synthetic_2026-10.md`.

## One-paragraph pitch

Open-set uncertainty scores that tie on pooled AUROC can behave very differently on the unseen
classes that are hardest to detect. In our controlled protocol, four uncertainty scores sit
within 0.004 AUROC of each other (0.949 to 0.953), and an uncertainty score and a margin score
fail on different held-out classes. We propose to evaluate novelty scores with a measure that
reports performance on the hardest unseen examples, using a hardness definition that does not
come from the model being scored, and to test whether it changes which methods look best.

## The question

Does a hardness-aware measure change which novelty scores look best, compared with pooled AUROC?

Pooled AUROC averages over all held-out rows, so easy classes dominate and the number depends on
which rows serve as negatives. A measure that reports performance on the hardest unseen
examples, with a stated definition of hard, would tell a practitioner which score to trust when
it matters. It is useful only if it changes a conclusion, stays stable across protocol choices,
and defines hardness without using the scored model.

## What we have so far (exploratory)

- Pooled AUROC on held-out test rows (200 known negatives, 5000 held-out positives): four
  uncertainty scores are within 0.004 of each other, and a margin score is lower (table below).
  Intervals are 95% bootstrap over rows.
- Per class the picture differs. EU is below 0.5 on one held-out class (AUROC 0.21 to 0.25)
  where margin reaches 0.67, and margin is below 0.5 on three classes (0.39 to 0.50) where EU is
  above 0.5 (0.54 to 0.70). Class identities assume standard CIFAR-10 order and still need to
  be checked.
- Adding EU to a margin-plus-entropy model gains at most 0.010 AUROC, on 3 of 7 held-out
  classes, and nothing on the held-out rows that look known.
- Pooled AUROC for the learned unknown output moved from 0.970 to 0.664 between two runs that
  are not yet reconciled. This is the kind of instability a better measure should avoid.
  **(Resolved 1 Nov — not instability: the two numbers used different negative sets. See the
  1 Nov addendum and `results/step0_reconciliation/summary.json`.)**

### Score table

Known test rows as negatives, held-out test rows as positives; one split and seed.

| score | AUROC | 95% CI | FPR at 95% TPR |
| --- | --- | --- | --- |
| EU, dropout | 0.949 | 0.934 to 0.962 | 0.213 |
| EU, bootstrap | 0.952 | 0.937 to 0.967 | 0.170 |
| TU | 0.952 | 0.936 to 0.966 | 0.187 |
| AU | 0.953 | 0.936 to 0.967 | 0.193 |
| margin / MSP (identical values, to be checked) | 0.790 | 0.754 to 0.825 | 0.387 |

### Head dependence

Same protocol; gain is the cross-validated AUROC gain from adding EU to a margin-plus-entropy
model.

| head | EU AUROC | margin AUROC | gain from EU |
| --- | --- | --- | --- |
| MLP, 300 epochs | 0.947 | 0.767 | 0.000 |
| MLP, 5 epochs | 0.378 | 0.532 | +0.007 |
| ETF fixed readout | 0.393 | 0.769 | 0.000 |

An EU AUROC below 0.5 means the score ranks held-out rows as more certain than known rows. This
head comparison belongs to the margin analysis planned for the ICML paper, so which paper
reports it is still open.

## Hypotheses (to be written down with numbers before each run)

1. **H1, rank reversal.** At least two scores that tie on pooled AUROC are separated by a
   hardness-stratified measure, with an interval on the difference that excludes zero. Test:
   paired bootstrap of the AUROC difference. The expected pair is named in a prediction file
   before the run.
2. **H2, independent hardness.** The reversal in H1 holds under at least two hardness
   definitions that do not use the scored model, for example distance in a second backbone's
   features and a class taxonomy.
3. **H3, stability.** The proposed measure varies less than pooled AUROC across negative-set
   choices, class mixes and seeds.

Each hypothesis can fail. If H1 fails, the measure adds nothing over AUROC on this protocol and
the paper does not go forward in this form.

## What the audience will demand

- A benchmark with graded difficulty: six or more known classes, or a standard split with near
  and far novel classes. The current evidence has two known classes, which is thin.
- Several published methods scored under the new measure: MSP, MaxLogit, energy, Mahalanobis,
  kNN, ViM and the learned unknown output.
- A hardness definition that does not come from the scored model, and a check that conclusions
  survive a second definition.
- A comparison with existing evaluation practice (near and far OOD splits, joint
  recognise-and-reject measures). Their definitions have not been verified here and need a
  literature check.
- A named negative set in every metric, and thresholds chosen on known validation rows only.

## Planned contributions (each falsifiable)

1. A hardness-aware evaluation measure, with the candidate definitions compared: class-macro
   AUROC, worst-class AUROC, hardness-stratified AUROC, and recall on the hardest fraction at a
   fixed false-alarm rate.
2. Evidence on whether it changes method rankings relative to pooled AUROC.
3. A stability analysis across negative sets, class mixes and seeds.
4. A reproducible protocol on cached features.

## Concrete protocol (proposed — freeze in a prereg file before any confirmatory run)

> **PARKED 1 Nov 2026 (owner decision).** Steps 2–7 of this hardness program are on hold
> pending the CVPR-line decision (epistemic-set line, see the 1 Nov evening addendum).
> The 2 Nov go/no-go is stood down while parked. Reactivation restarts at step 2
> (measures into `metrics.py` + tests) with the step-3 DINOv2 prerequisite re-checked;
> steps 0–1 remain closed and their discipline (named negative sets, class identities)
> applies to every line.

Everything below runs on the assets that already exist in this directory. Nothing here trains a
backbone or touches raw images.

### Data and splits

- **Rows.** CIFAR-10 train + test on cached CLIP features (`cache/features/features.pt`, keys
  `cifar10_train` / `cifar10_test`; 512-d, L2-normalized; local copy verified loadable
  2026-10-20).
- **Split mechanism.** `make_oss` (eu_blindspot.ipynb / example.py) wrapping
  `pytorch_ood.dataset.ossim.DynamicOSS`. The exploratory run used `kuc=1, uuc_val=2,
  uuc_test=5, seed=0`: kkc = 2 known classes (200 known test negatives = 2 x 100), 5000
  held-out positives (5 uuc classes x 1000).
- **Graded-difficulty grid** (answers the "two known classes is thin" demand; on 10 classes,
  more known classes means fewer novel test classes per split — that trade-off *is* the
  class-mix axis H3 needs):

  | arm | kkc | kuc | uuc_val | uuc_test | DynamicOSS seeds |
  | --- | --- | --- | --- | --- | --- |
  | A (current) | 2 | 1 | 2 | 5 | 0-9 |
  | B | 4 | 1 | 2 | 3 | 0-9 |
  | C | 6 | 1 | 1 | 2 | 0-19 (uuc_test shrinks to 2 classes, so more seeds) |

- **Threshold discipline.** Every metric names its negative set ("kkc test rows", etc.).
  Operating thresholds (blind-spot quantiles, FPR targets) come from kkc **val** rows only;
  uuc_val rows are diagnostics-only, never thresholds. `metrics.report()` flags groups below
  `MIN_GROUP = 200`.
- **Class identities.** Assert the split's class ids against the standard CIFAR-10 order in
  code and print per-class n before any confirmatory run — the per-class inversion claims
  depend on it.
- **Optional benchmark arm** (supplementary, only if on schedule 3 Nov): CIFAR-100 with a
  coarse-group near/far split using `y_coarse`, already in the cached c100g NPZs on Modal.

### Feature spaces

| space | source | role |
| --- | --- | --- |
| CLIP ViT-B/16, 512-d | `cache/features/features.pt` (local, intact for train/test) | the space all scored models live in |
| DINOv2 ViT-S/14, 384-d | Modal volume `reach-aistats`, extractor `real/modal_extract.py` | hardness space 1 — never scored |
| timm wide_resnet50_2, 2048-d | `example_wrn50.py` extractor (HF-hub weights; caches `cache/features/timm_wide_resnet50_2_*.npz`, CIFAR-10 + MNIST test) | backbone-swap demo arm (exploratory) — envelope-vs-BALD checks live here |

- Verify full CIFAR-10 train+test DINOv2 features exist on the volume; if not, extract on
  Modal. Do not `modal volume get` feature files to local (12/17 local NPZs are truncated
  transfers); compute hardness on Modal and ship home only the per-row scalar arrays.
- A third backbone, if a reviewer demands it, comes from `example.py`'s timm loader.

### Scored models (heads)

All trained per split per seed on cached CLIP features (laptop-scale: minutes). The CVPR paper
scores these two EU estimators and names them every time:

- **EU, MC-dropout**: `HeadDrop` MLP 512-128-ReLU-dropout(0.5)-(K+1), 300 epochs Adam, T = 30
  dropout-enabled passes (BALD decomposition as in the notebook).
- **EU, bootstrap**: 12 data-draw members, BALD over members.

The K+1 outlier-exposure head maps all kuc ids to the learned unknown output; dropout inside
the head is what makes the MC passes possible. The linear head, mlp@5 and ETF-readout regimes
stay in the ICML margin paper's scope (see the head-dependence table above) — they are context
here until the ownership decision lands.

### Scored methods (the benchmark set)

Eleven methods, all computable from cached features + the trained heads. Use `pytorch-ood`
implementations where available (already a dependency); the rest go into a new
`real/hardness_eval.py`.

1. MSP (max softmax of mean probs)
2. MaxLogit
3. Energy (temperature 1)
4. Margin (top1 - top2 logit; the notebook's `neg_margin`)
5. EU, MC-dropout (T = 30)
6. EU, bootstrap (12 members)
7. TU and AU (context; estimator named wherever quoted)
8. Learned unknown output (P(unknown) of the K+1 head) — **gated on step 0** below
9. Mahalanobis (shared covariance, class means on kkc train features)
10. kNN score (Sun et al. 2022; k = 10, k = 50 as sensitivity; per-class tau_c calibration
    machinery already exists in `metrics.fill_knn`)
11. ViM (logits + feature residual; rescaling calibrated on kkc val rows)
12. *Exploratory:* credal envelope width (probly probability-interval set over the T dropout
    samples; `example_wrn50.credal_intervals`). Ties BALD on pooled AUROC here (+0.001 to
    +0.002); include only if the hardest-tercile separation replicates — see addendum below.

Hyperparameters (T, member count, k, temperature, ViM subspace dim) freeze in the prereg; no
per-method tuning on test rows.

### Hardness definitions (independent of the scored model)

Three candidate sources; the first two are the H2 pair.

1. **DINOv2 kNN distance (primary).** Per uuc test row: mean distance to its k = 10 nearest
   kkc-train rows in DINOv2 ViT-S/14 space, normalized by that known class's tau_95 of
   within-class neighbour distances (same calibration idea as `fill_knn`). Per-row hardness
   gives per-split tercile strata (cut points within the split's uuc test rows); per-class
   hardness = class mean, which drives worst-class ordering. Computed on Modal.
2. **Taxonomy near/far (secondary).** CIFAR-10 classes inherit the documented CIFAR-100 coarse
   grouping: vehicles {airplane, car, ship, truck}; animals {bird, cat, deer, dog, frog,
   horse}. A uuc class sharing a coarse group with at least one kkc class is near (harder);
   otherwise far. Print and verify the mapping table in code.
3. **Leave-one-method-out consensus miss rate (robustness only).** Hardness of a row = mean
   miss rate of all *other* methods at their known-val FPR95 thresholds. Independent of any
   single scored method; not independent of the panel — disclosed wherever used.

Circularity check: report Spearman correlation between DINOv2 hardness and (a) CLIP-space
Mahalanobis, (b) CLIP-space kNN score, on uuc rows. If DINOv2 hardness is near-proxy for a CLIP
distance detector (rho > 0.9), the taxonomy arm carries the independence claim and the text says
so.

### Measures

| measure | definition | status |
| --- | --- | --- |
| Pooled AUROC | current practice | baseline column |
| Class-macro AUROC | mean per-uuc-class AUROC vs named negatives, with per-class CIs | main text |
| Worst-class AUROC | min per-class AUROC | main text |
| Hardness-stratified AUROC | AUROC per hardness tercile; headline = hardest tercile | main text; Figure 1 |
| Recall on hardest fraction | recall on hardest tercile at 5% FPR, threshold from kkc val rows | Table 1 |

H3 stability summary: per measure, SD across seeds and CI width of paired method differences,
under negative-set variants (all kkc test rows vs leave-one-known-class-out), the class-mix
grid (arms A/B/C) and seeds.

### Statistics and preregistration

- Within-split contrasts: paired bootstrap over rows, n_boot = 1000 (the notebook
  `score_table` convention).
- Across splits: seed-paired analysis over the DynamicOSS seeds; head-training seeds 0-4 nested
  per split; every split reported, no selection.
- House style: `real/hardness_prereg.py` freezes the named H1 pair, grids, measures,
  hyperparameters and stop condition before confirmatory runs; the manifest records its SHA.
  Any grid change after looking at results makes the run exploratory, and it is labeled as
  such.
- Unit tests: extend `tests/test_metrics.py` with hand-built-score tests for macro, worst-class
  and stratified AUROC and the recall@FPR measure.

### Compute and where it runs

- Head training and scoring: laptop-scale (the notebook ran the current protocol
  interactively); the full grid is roughly 40 splits x ~3 heads, hours not days.
- DINOv2 hardness: computed on Modal next to the volume (`real/modal_audit.py` pattern); only
  per-row scalar arrays come back.
- Local CIFAR-10-C NPZs are untrustworthy (12/17 truncated); CIFAR-10-C is not needed for the
  core protocol, and anything that touches it runs on Modal.
- Deliverables per run: `results/hardness/<run>/{summary.json, per_class.json, manifest.json}`
  via `metrics.to_json`, the manifest carrying `predictions_sha256`.

## Figure 1 and Table 1 (candidates)

- Figure 1: AUROC against hardness stratum, one line per score. Scores that tie on the pooled
  number should separate along the hardness axis.
- Table 1: pooled AUROC, class-macro AUROC, worst-class AUROC, hardness-stratified summary, and
  recall on the hardest fraction at a fixed false-alarm rate, each with confidence intervals.

## Concrete steps, in order

| # | step | when | exit criterion |
| --- | --- | --- | --- |
| 0 | Reconcile the unknown-output runs: diff the two runs behind 0.970 and 0.664 (head config, epochs, split seed, normalization, sign convention); write findings to `results/`. Also resolve the margin/MSP identity flag (print both score vectors; check which head's logits and sign each used). | 20-21 Oct | **DONE 1 Nov** — root cause: negative sets (kkc+kuc vs kkc-only), mixture identity reproduces 0.665 exactly; margin/MSP identity is a K=2 mathematical identity. `results/step0_reconciliation/summary.json`, `step0_reconcile.py` |
| 1 | Verify class identities in code: assert split class ids against standard CIFAR-10 order; print per-class n. | 22 Oct | **DONE 1 Nov** — `metrics.CIFAR10_CLASSES` + names in `fill_data_and_split`/`report()`, asserted against the notebook's recorded split; tests in `tests/test_metrics.py` (101 passing) |
| 2 | Implement + unit-test the measures in `metrics.py`: macro AUROC, worst-class AUROC, stratified AUROC, recall@FPR95 on hardest fraction. Tests in `tests/test_metrics.py`. | 22-24 Oct | tests pass; `report()` flags small groups |
| 3 | DINOv2 hardness on Modal (parallel with 2): verify/extract full train+test; compute per-row kNN hardness + tau normalization on Modal; ship back (N,) arrays only. | by 26 Oct | hardness array for all 60k rows on local disk, small file |
| 4 | Freeze the prereg: `real/hardness_prereg.py` with the named H1 pair (prediction: EU and margin separate under hardness-stratified AUROC while tying on pooled), grids, seeds, hyperparameters, stop condition; record SHA. | by 27 Oct | manifest hash exists before any confirmatory run |
| 5 | Confirmatory run A: arm A (kkc=2) x seeds 0-9, all 11 methods, all 5 measures. H1 paired bootstrap. | 27-30 Oct | per-method table with CIs; H1 verdict per the frozen rule |
| 6 | Confirmatory run B: arms B and C (kkc in {4, 6}). H2: recompute strata under taxonomy and consensus hardness; circularity correlations. | 30 Oct - 2 Nov | reversal replicates (or fails) under both hardness definitions |
| 7 | H3 stability: negative-set variants, class-mix grid, seed variance. | 31 Oct - 3 Nov | stability table per measure |
| 8 | **Go/no-go.** H1 holds (separation with CI excluding zero, replicating across arms and both hardness definitions) -> proceed. H1 fails -> the measure adds nothing over AUROC here; stop or re-scope per the stop condition. | **2 Nov** | *stood down — program parked 1 Nov (owner decision); the CVPR-line decision supersedes this gate* |
| 9 | Benchmark arm + figures: CIFAR-100 near/far if go and on schedule; draft Figure 1 and Table 1; write the half-page verified-definitions note on near/far OOD and joint recognise-and-reject practice (sources read, not summarized). | 3-9 Nov | figures drafted; literature note in `05_Literature_and_Scoop_Analysis/` or `09_Results_Wiki/` |
| 10 | Enroll by 10 Nov; write 10-16 Nov (paper due 16 Nov, no extension); supplementary by 23 Nov. | 10-23 Nov | submitted |

## Related work to position against

Cite after re-reading: Young 2026 (arXiv 2606.12646), Mucsanyi et al. 2024, Wimmer et al. 2023,
Hendrycks et al. 2019 (outlier exposure), Dhamija et al. 2018 (background class), ViM, NuSA and
NECO. For the evaluation side, check existing benchmarks with near and far novel classes and
joint recognise-and-reject measures; their definitions have not been verified here. Several
descriptions come from search summaries. The verified-definitions note (step 9) is where the
near/far and joint-measure conventions get settled with citations that were actually read.

## What we do not claim

- No new detector. EU, entropy and margin are scored methods, and the measure is the subject.
- No claim about ensembles in general. We name the estimator that was run.
- No margin or attack analysis. That material belongs to the ICML paper.

## Relationship to the ICML paper

The ICML paper owns the margin analysis (EU as a function of the logit margin, attack depth,
epoch dependence) and the head comparison. This paper treats EU as one method among several and
reuses none of those figures or experiments. If the per-class inversion appears in both, one
paper reports it as an observation and the other explains it. Which paper owns it is undecided.

## Risks

1. Circular hardness: if hardness is measured in the scored model's feature space, distance-based
   scores win by construction. *(Mitigated: hardness lives in DINOv2 space and in the taxonomy;
   circularity correlations are reported, and the taxonomy arm carries the claim if they fail.)*
2. Two known classes give too little graded difficulty. *(Mitigated: arms B and C take kkc to 4
   and 6; arm C is the "six known classes" answer.)*
3. The measure may change no ranking. That is the H1 stop condition, tested at the 2 Nov gate.
4. A measure paper needs several published methods. *(Mitigated: the 11-method set, most from
   pytorch-ood.)*
5. The 0.664 versus 0.970 gap is unexplained and affects every table that includes the unknown
   output. *(Resolved 1 Nov: negative-set definition difference, not instability — see 1 Nov
   addendum. The unknown output is unblocked under the naming discipline.)*
6. Results come from a single split and seed so far. *(Mitigated: seeds 0-9 per arm before the
   go/no-go.)*

## Addendum 24 Oct 2026 — envelope width vs BALD checks (example_wrn50.py)

First numbers for the wide_resnet50_2 CIFAR-10 vs MNIST demo (see [[04_OSS_Worlds_wrn50]] in the
vault): mean class-probability interval width (probly credal set over T=30 dropout samples) tracks
BALD closely — pooled AUROC 0.915 vs 0.913 (+0.002, a tie). This motivates the same
beyond-pooled-AUROC question as this plan, on a second score pair. Checks now implemented and
tested (suite: `python -m pytest tests/`, 90 passing):

1. **Per-example rank divergence** — `rank_disagreement` (`tests: test_rank_disagreement`):
   share of test rows where EU and width percentile ranks differ >10 %, plus scatter
   `wrn50_scatter.png` colored ID/OOD. Feeds the hardness-stratified comparison directly.
2. **Dropout-sampling stability** — `sampling_stability` (`tests: test_sampling_stability_structure`):
   head, rows, preprocessing fixed; only dropout masks redrawn; T ∈ {10, 30, 100}, 20 repeats.
   Spread here is attributable to dropout sampling alone (the controlled attribution the run-to-run
   0.011 gap lacked). Envelope nesting property tested: width over a pass subset ≤ width over the
   full set (`tests: test_credal_width_nesting_monotone`), so width scales are T-dependent.
3. **Per-pair path retention** — `choose_control_partners` (`tests: test_choose_control_partners`):
   CIFAR controls now declare same-class vs different-class relation per pair (fails loudly if
   unconstructible); per-pair final-step EU/width/distance in `wrn50_paths_summary.json →
   per_pair_final`.

Standing caveat agreed with collaborator framing: 0.915 is scalar-score AUROC, not interval-valued
$[A_L, A_U]$ AUROC; envelopes are not membership intervals.

## Addendum 24 Oct 2026 (evening) — repo update after new implementation pass

The plan now has direct code support for two new execution paths that were not present in the
earlier draft, and corresponding tests are in place.

### New code paths added

1. `novel_variations.py` (finite-world controls, no training required):
  `make_novel_world` adds confounded/interpolate/near/custom novel-law controls,
  calibration is known-only, and `run(**OPTIONS)` returns rows + per-cell summaries with fixed
  threshold discipline.
2. `example_wrn50.py` (feature-space benchmark runner):
  adds two world modes under one interface:
  - `--mode oss` (DynamicOSS budgeted split),
  - `--mode plain` (fixed known classes + pooled OOD, including cross-dataset OOD such as
    CIFAR-10 vs MNIST with label-offset safety).
3. Device and dimensionality wiring:
  WRN50 (2048-d) heads are now explicit (no accidental 512-d assumptions), and extraction,
  training, and scoring run on configured `cpu`/`mps`/`cuda`.

### Evaluation diagnostics now implemented

1. Rank divergence (`rank_disagreement`) between EU and credal-width ordering.
2. Sampling-only variability (`sampling_stability`) over redraws of dropout masks at fixed data
  and fixed head.
3. Hardness-stratified reporting helpers (`stratified_tercile_auroc`,
  `divergent_hardness_concentration`) for "hardest slice" comparisons.
4. Path-level diagnostics and control-pair construction checks (`choose_control_partners`, path
  trace utilities).

### Results added 25 Oct (terciles + attribution, CIFAR-10 vs MNIST, wrn50)

- Hardness-stratified AUROC (terciles by frozen-feature kNN distance, bootstrap CI on
  width − EU): easy −0.000 [−0.002, +0.001]; mid **+0.013 [+0.004, +0.023]** (CI excludes 0);
  hardest +0.014 [−0.001, +0.029] (directional, not established). Both scores degrade with
  hardness (0.89 → 0.84 → 0.79/0.80). Write-up: vault `04_Experiments/04_OSS_Worlds_wrn50.md`.
- Divergent rows are NOT hardness-concentrated (35.8 % in hardest tercile vs 33.3 % base).
- Dropout-only sampling moves AUROC ±0.002 at T=30, so the 0.011 run-to-run gap is head-training
  nondeterminism, not dropout draws. Mean envelope width grows with T (nesting): 0.140/0.193/0.249
  at T=10/30/100 — width values are never comparable across T.
- probly rendering (`plot_credal_examples` → `wrn50_credal_set.png`): the rank-divergent known row
  contests 2/10 classes with envelope bars nearly as wide as the OOD row's, which spreads over
  ~6 classes — envelope width alone cannot separate them.

### Multi-dataset readiness — swept 25 Oct

`example_wrn50.py` takes any torchvision test split as ID or OOD source (registry:
cifar10, cifar100, mnist, fashionmnist, svhn; `--id-dataset` / `--ood-dataset`, per-dataset
caches `cache/features/timm_<model>_<dataset>test.npz`). Four-dataset sweep run (ID = CIFAR-10,
OOD = whole test split; artifacts `results/oss_metrics_{fmnist,svhn,cifar100}/`): pooled EU-vs-width
AUROC tie on every dataset (largest gap +0.004); the MNIST hardest-tercile hint (+0.014, ns) did
NOT replicate on FashionMNIST/SVHN/CIFAR-100 (all CIs straddle 0); rank divergence ranges
7 % (CIFAR-100) to 44 % (SVHN) of rows. Two infra notes: an `auroc_fast` (exact Mann-Whitney
midranks, parity-tested) replaces example.auroc's quadratic Python loop in the stratified/stability
paths — bootstrap 10 min → 1 s; and this env needs `OMP_NUM_THREADS=1` (torch+sklearn libomp pools
exploded to 2,500 threads). Large downloads (CIFAR-100) run from the local tarball at
`/Users/tanmoy/data` — no Modal trip needed for the wrn50 line.

### Test status reflected in tree

New/expanded tests exist for both tracks:

- `tests/test_novel_variations.py`
- `tests/test_example_wrn50.py`
- `tests/test_wrn_compare.py`

These cover endpoint laws, calibration behavior, prefix consistency, WRN50 plain/OSS smoke runs,
credal-envelope properties, kNN distance correctness, and stratified metrics behavior.

### Immediate gating tasks before confirmatory claims

1. Freeze one reporting configuration for the WRN50 path:
  exact `mode`, known/novel class policy, `T`, `k`, and calibration seed policy.
2. Record one manifest-style run summary per mode (`oss`, `plain`) with hashes and config blocks,
  then stop changing defaults until the 2 Nov gate.
3. Keep the current caveat explicit in all tables: envelope-width results are scalar-score AUROC
  comparisons, not interval-valued AUROC intervals.
4. Merge only the collaborator-facing markdown plan in git; keep auxiliary markdown notes local,
  consistent with repository ignore policy.

## Addendum 1 Nov 2026 — steps 0 and 1 closed

`step0_reconcile.py` rebuilt the notebook protocol exactly (world seed 0 reproduced bit-for-bit:
kkc automobile(1)+ship(8), kuc deer(4), uuc the other seven; 2100/2600/5300 rows) and resolved
both flags. Full numbers: `results/step0_reconciliation/summary.json`.

**0.970 vs 0.664 — negative sets, not instability.** Notebook cell 8's code
`roc_auc_score(uuc_mask, p_unknown)` takes negatives = every non-uuc row = kkc (200) **plus the
100 kuc deer rows trained on the unknown output**. Deer rows score P(unknown)~0.99 and beat
almost every uuc row, dragging AUROC to 0.665 (the notebook's 0.664-0.667). With kkc-only
negatives the same head gives **0.975** — the plan's 0.970 is a normal honest readout. The
mixture identity proves the mechanism: (200*0.975 + 100*0.045)/300 = 0.665 exactly. Head seeds
0-4 move the honest readout 0.975-0.979; world seeds 0-3 move it 0.902-0.982 (novel-class mix,
not head noise); aggregation choice (deterministic, mean-logits) moves it <=0.007. The kuc
readout (deer positives) is 1.000 — a trained-on ceiling, never a detection number.

**Consequence for the exploratory score table above:** cell 43's `~is_novel` negatives share
the leak, so every number there (EU 0.949-0.953, margin/MSP 0.790) is a kkc+kuc-negative
number. EU barely moves (0.949 kkc-only vs 0.941 with deer negatives) because deer EU is low,
but all known-vs-novel scores must be recomputed under named negatives before the prereg
freezes them.

**margin/MSP identity — resolved, not a bug.** At K=2 known outputs, max softmax =
sigmoid(top1-top2), strictly monotone in the margin: identical ranking and AUROC by
construction (measured 0.73557 vs 0.73546; spearman 0.999991, the residue being float64
softmax saturation ties). Arm A (kkc=2) can never separate margin from MSP; arms B/C will, and
the confirmatory tables carry both. Also recorded: cell 43 computed its logits after
`head.train()` was restored, so the notebook's margin/MSP/EU table basis is a dropout-active
stochastic forward; the prereg fixes eval mode.

**Step 1 (class identities).** `metrics.py` now carries `CIFAR10_CLASSES` (standard
torchvision order) and `cifar10_names()`; `fill_data_and_split` records names and per-split
per-class counts, `report()` prints them every run, and `cifar10_names` asserts ids are
CIFAR-10 ids. Tests added (`tests/test_metrics.py`; suite 101 passing).

**Gate decision.** The learned unknown output is UNBLOCKED for confirmatory tables under the
naming discipline: kkc-only negatives for detection claims, kuc readout disclosed as a
trained-on ceiling, never an unnamed pooled negative set. The prereg's negative-set clause is
the fix; no property of the method changes.

## Addendum 1 Nov 2026 (evening) — collaborator meeting: epistemic sets, and Tanmoy's side delivered

Meeting outcomes recorded (from the 1 Nov notes): the joint story moves to **epistemic
uncertainty as a set** (upper/lower bounds) on ID/OOD open-set simulation via pytorch-ood,
cached features only. CVPR angle: a simple **credal adapter** ("cradle adapter" in the
notes) as a post-hoc trainer for confidence/calibration; ICML may pivot toward likelihood/
decalibration aspects (open). Mateus replicates the last paper discussed and applies
credal/epistemic sets; February visit likely.

**Tanmoy's parts are complete** — `epistemic_sets.py` + `tests/test_epistemic_sets.py`
(21 tests; suite 122 passing), results in `results/epistemic_sets/`:

- `RepresentationBuilder` base class + `EpistemicSetRepresenter`: (T, N, K) sampled
  predictives -> per-class lower/upper/mean/median envelope + nested alpha-cut **contour**
  family (optimistic/pessimistic sets per alpha; nesting unit-tested). The upstream base
  class lives in the collaborator's codebase; ours is a local declaration of the same
  one-method contract (`build`), to re-base if it becomes available.
- Scalar scores **from the set**: lower / upper / midpoint ("average") / median-of-samples /
  width on the novel column / mean width / mean contour cardinality.
- **CredalAdapter**: logistic + MLP over interval features, fit on the val split only
  (kkc val negatives + uuc_val positives — disclosed training signal, never a threshold
  source), probabilities via `predict_proba` for calibration readouts.
- K+1 dropout head (notebook cell-6 protocol, canonical home now `epistemic_sets.py`),
  DynamicOSS worlds kuc=1/uuc_val=2/uuc_test=5, world seeds 0-4, cached CLIP features.
- Tabular per the meeting: `rows.csv` (26,500 rows x 23 cols, per-row scores with class
  names and groups), `summary.csv` / `summary_agg.csv`, `per_class.csv`, `manifest.json`
  (sha256 + negative-set + adapter-fit disclosures), `comparison.png`.

**Results (pooled AUROC, pos uuc | neg kkc, mean over 5 world seeds):**

| method | mean | worst-class |
| --- | --- | --- |
| set_median (median P(unk) over passes) | **0.950** | **0.855** |
| set_lower (min P(unk) over passes) | 0.949 | 0.854 |
| P_unknown (mean P(unk)) | 0.946 | 0.839 |
| width_novel (interval width on the unknown column) | 0.916 | 0.811 |
| EU/BALD | 0.888 | 0.774 |
| TU / MSP / AU / margin | 0.883-0.886 | 0.77-0.78 |
| energy / MaxLogit (raw eval logits) | 0.869 | 0.698 |
| credal_adapter_mlp / logistic | 0.81 / 0.78 (sd 0.18-0.21) | 0.62 / 0.53 |

Read: **every direct interval readout beats every single-valued score**; the median of the
set is the best pooled and hardest-class number; even the width alone (pure set-ness)
beats BALD. Honest negatives: the *learned* adapters do not transfer across novel-class
mixes (fit on uuc_val bird/frog, tested on five other classes; sd 0.2) and are worse
calibrated than MSP on known rows (ECE 0.45-0.59 vs 0.01) — as trained for separation,
not calibration; the "post-hoc trainer improves calibration" claim is NOT yet supported
and needs a correctness-supervised fit (deferred to Mateus's replication axis).
Macro AUROC equals pooled here because uuc test classes are balanced (documented in code).
Standing caveat unchanged: scalar-score AUROC only; envelopes are not membership intervals.

**Open decision for the 2 Nov gate:** the hardness-measure confirmatory program (steps
3-7 below) and this epistemic-set line now compete for the CVPR slot. The set-result
above is a cleaner positive than anything in the hardness line so far; both share the
negative-set naming discipline and the class-identity machinery.

## Decisions still open

1. Hardness source. *Recommendation: DINOv2 kNN distance primary, taxonomy near/far secondary,
   consensus miss rate robustness-only.*
2. Data: six or more known classes, or a standard benchmark split. *Recommendation: CIFAR-10
   arm grid A/B/C as primary; CIFAR-100 near/far only as a supplementary benchmark arm, built
   only if step 8 passes on time.*
3. Which candidate measures go into the main text. *Recommendation: class-macro AUROC and
   hardest-tercile stratified AUROC headline; worst-class in main text; recall@FPR95 in Table
   1.*
4. Which paper owns the per-class inversion and the head comparison. Still open; owner decision.
5. Whether to submit this to CVPR and the margin paper to ICML, or only one of them.
   *Recommendation: decide at the 2 Nov go/no-go; CVPR only if H1 holds.*
   **Update 1 Nov: the hardness program is parked (see the parked note above); this
   decision is now folded into the CVPR-line choice — hardness-measure vs epistemic-set
   story — recorded in the 1 Nov evening addendum.*

## Dates

From the CVPR 2027 call for papers; verify page limit and overlap rules in the 2027 author
guidelines.

- Enrollment: 10 Nov 2026
- Paper: 16 Nov 2026, no extension
- Supplementary: 23 Nov 2026
- Dual-submission window: 16 Nov 2026 to 22 Feb 2027. A substantially similar paper cannot be
  under review elsewhere in this window. ICML 2027 dates are projections, not official.
