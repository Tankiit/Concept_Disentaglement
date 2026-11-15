# CVPR 2027 plan — certified novelty and the failure of pooled AUROC

Status: 10 Nov 2026. Author/owner: Tanmoy. Working draft; supersedes the CVPR framing
in `COLLABORATOR_PLAN.md` §"one-paragraph pitch" (that program is parked; its discipline
and step-0 findings carry over). Everything below is regenerable: each result names its
script and run directory; all recorded outputs are wiki-synced (53 run pages).

## 0. Dates and binding constraints

- **Enrollment due TODAY (10 Nov 2026). Paper 16 Nov, no extension. Supplementary 23 Nov.**
- Dual-submission window 16 Nov–22 Feb: the AISTATS widths paper (W_cls/W_lab/W_unk,
  LP/MMD attribution) stays OUT of CVPR. Only the dropout-envelope width appears, as a score.
- House rules that bind every table: gates before claims; every AUROC names its negative
  set; thresholds from kkc val rows only; preregs frozen (amendments are new files);
  matched arms; every number regenerable; wiki backup after each batch.
- Standing caveat on every set result: scalar-score AUROCs and certificate rates only;
  envelopes are probability intervals from T sampled predictives, not membership intervals
  of a formally identified credal set. Say "sensitivity/certainty bands" in the main text;
  "credal" once, in the appendix.

## 1. Verified result inventory (as of 10 Nov)

### 1.1 Epistemic-set readouts (CIFAR-10, CLIP ViT-B/16, DynamicOSS kkc=2/kuc=1,
uuc_val=2/uuc_test=5, world seeds 0–4, K+1 MC-dropout head; `results/epistemic_sets/`)

Pooled AUROC (pos uuc | neg kkc), mean over 5 worlds [min–max]:

| readout | mean | range |
| --- | --- | --- |
| set_median | 0.950 | 0.908–0.987 |
| set_lower | 0.949 | 0.905–0.985 |
| P_unknown (mean) | 0.946 | 0.904–0.984 |
| set_upper = set_mid | 0.941 | 0.897–0.980 |
| credal τ-width (0.924) / width_novel (0.916) | — | — |
| EU/BALD 0.888, TU 0.886, MSP 0.885, AU 0.885, margin 0.883 | — | — |
| energy/MaxLogit (K+1 logits) 0.869 | — | — |
| width_mean 0.886 (≤ BALD in 5/5), card_mean 0.631 (near chance) | — | — |

**Fair-baseline finding (10 Nov audit, deterministic replays of the delivered heads).**
The classic scores as delivered are computed over all K+1 outputs including the unknown
slot. Restricted to the known outputs (standard definition), on the same heads:
energy(known) = 0.988/0.933/0.992/0.904/0.898 (mean 0.943), maxlogit(known) 0.942,
BALD(known) 0.914, MSP(known) 0.907. **The unknown-output readouts do not beat these:
3/5 worlds vs energy(known), a world-dependent tie.** The +0.06–0.08 margins in the
delivered table are the K+1-formulation artifact. Do not headline "sets beat baselines".

### 1.2 Certificates (the spine; `results/certificate_audit/`, PRED frozen before scoring)

Envelope bounds on the unknown column, level ½: *certainly novel* = every sampled model
above ½; *certainly known* = every one below; else undecided. T=30.

| world | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| false novelty certs on knowns | 0 | 0 | 0 | 0 | 0 |
| knowns certified known | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 |
| uuc far: certified novel | 0.73 | 0.66 | 0.87 | 0.33 | 0.64 |
| uuc near: certified novel | 0.02 | 0.01 | 0.49 | 0.00 | 0.00 |
| uuc near: certified KNOWN (wrong) | 0.90 | 0.95 | 0.49 | 0.99 | 0.94 |
| absorbed rows (P_unk ≤ 0.01) certified known | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| absorbed share of uuc (near-tagged share) | .31 (.99) | .36 (.98) | .19 (.99) | .74 (.76) | .52 (.99) |

Certificate-pair statement: **a sound novelty certificate (zero false certificates in
every world) coexists with an overconfident knownness certificate that mis-certifies
49–99% of near unseen rows and 100% of absorbed rows.** Pooled AUROC ≥ 0.90 throughout.

Supporting checks (PRED frozen, 12/15 sub-verdicts as predicted; 3 failures listed in
`summary.json`):
- **Kill test**: a P_unknown threshold at zero FPR on kkc val rows does NOT reproduce the
  certificate (coverage differs by −0.15 to +0.08). Both are sound on test knowns (the
  threshold never slipped here). The certificate's edge is structural: no calibration
  data, no outlier fragility — world 3's val threshold was set by a single val row
  (0.9989), collapsing its coverage to 0.048 vs the certificate's 0.131.
- **T-robustness**: coverage(100)/coverage(10) = 0.79–0.93; monotone by nesting
  (prefixes share the mask sequence); soundness exact at all T. Not an undersampling
  artifact.
- **Two constructions agree on definites**: sampled hull vs τ-relative-likelihood (KL
  ball) — zero direct novel↔known conflicts (impossible by the p̄ sandwich: hull-novel
  ⇒ p̄ ≥ ½, τ-known ⇒ p̄ < ½); they differ on 4–12% of uuc rows, always via undecided,
  always in the direction that disagreement REMOVES certificates (τ coverage ≥ hull, 5/5).
- Figure: `results/certificate_audit/certificates.png` (per world: known / uuc-near /
  uuc-far × certainly-known / undecided / certainly-novel).

### 1.3 Exact structural results (theorems + unit tests; hold by construction)

- Negative-set mixture identity (every AUROC names its negatives; step 0:
  (200·0.975+100·0.045)/300 = 0.665 reproduced the notebook's 0.664 exactly).
- K=2: MSP ≡ sigmoid(margin), identical ranks (Spearman 0.999991).
- Relative-likelihood credal set = KL ball around p̄: no disagreement information;
  τ-invariant ranks (τ=0.90 ≡ 0.95, identical numbers); unknown-slot width symmetric in
  p ⇒ AUROC ≡ −|p̄−½| exactly.
- Envelope nesting: width grows with T (mean 0.14/0.19/0.25 at T=10/30/100) — widths
  never comparable across T; dropout-only noise ±0.002 at T=30.

### 1.4 The universal width negative (the only cross-dataset credal result)

Mean class-probability envelope width never beats BALD by more than 0.006 in ANY of 11
settings: 5 CLIP worlds (−0.003 mean, loses 5/5) + 6 WRN50 dataset pairs (CIFAR-10 vs
MNIST/FMNIST/SVHN/CIFAR-100: ±0.006; **SVHN as ID: −0.033**, replicated with independent
head retrainings). Contour cardinality near chance (0.63). Both width constructions
(hull and τ-set) lose to the mean/bounds readouts in 5/5 worlds; Spearman(width, BALD)
≈ 0.86. Width carries no detection information beyond predictive uncertainty.

### 1.5 Instability findings (the beyond-pooled-AUROC evidence)

- EU vs MSP: pooled tie (+0.003 mean) but per-world bootstrap CIs exclude 0 in all 5
  worlds — positive in 2, negative in 3. Ranking among classic scores is a property of
  the class mix, not the method.
- Per-class failure sets differ by method (set readouts' worst: truck with vehicle
  knowns, 2/2 worlds where it can occur; BALD's worst: horse/deer where set_median is
  1.000). Near classes are harder than far for unknown-output readouts (5/5); the
  near/far order reverses for classic uncertainty scores in 2/5.
- Absorption: 19–74% of unseen rows get P(unknown) ≤ 0.01 at pooled AUROC ≥ 0.90
  (per-world 0.31/0.36/0.19/0.74/0.52).

### 1.6 Negative inventory (report as results, never as bugs)

Learned credal adapters: worse than the mean they are built from in 5/5 worlds, below-
chance classes, ECE 0.01–0.80 vs MSP's 0.002–0.023 (5/5). Hardest-tercile width
advantage did not replicate; monotone hardness degradation holds 3/6 dataset pairs; the
recorded tercile design starves hard strata of negatives (n_known down to 1) — the
parked protocol's uuc-only strata fix was never run. Energy/MaxLogit over K+1 logits
collapse on classes near the kuc class (0.344 on truck, world 4) — a baseline-definition
artifact; use known-logit variants only.

## 2. What holds vs what does not (one screen)

HOLDS (universal in everything tested): certificate soundness (0 false novelty certs,
all worlds, all T, both constructions) + the overconfidence certificate; near→mis-
certified, far→certified; kill test non-redundancy; T-robustness; the width negative
(11/11 settings); the exact theorems; adapters/calibration failures; absorbed→certified-
known (100%, 5/5).

DOES NOT HOLD (must not be claimed): "every interval readout beats every single score"
(width_mean, card_mean, set_upper all fail); "sets beat baselines" under fair known-logit
baselines (3/5 world-dependent tie); "width ties BALD" universally (SVHN-ID −0.033);
any set-ness attribution for width_novel's BALD win; adapter calibration gains.

## 3. Paper skeleton (claims → evidence)

1. **Intro/thesis**: pooled AUROC and point thresholds hide what a set of predictives
   certifies; certificates expose both soundness and overconfidence that AUROC cannot see.
2. **Setup**: DynamicOSS worlds, K+1 MC-dropout head, envelope and τ-set constructions,
   named-negative discipline.
3. **Certificates** (Fig 1 = certificates.png; Tab 1 = §1.2 table): sound novelty
   certificate + overconfident knownness certificate; kill test; T-robustness; two
   constructions (theorem: no direct conflicts; disagreement only removes).
4. **What pooled AUROC hides** (Tab 2): EU-vs-MSP world-resolved reversals; per-class
   failure-set divergence; absorption vs AUROC ≥ 0.90.
5. **The width negative** (Tab 3, 11 settings, 2 backbones): neither width construction
   carries information beyond BALD; τ-collapse theorem.
6. **Honest negatives**: adapters, calibration, tercile design.
7. **Limitations**: one dataset for the positives (unless item 6 below lands), kkc=2,
   5 exploratory worlds + confirmatory prereg status, certificates relative to the
   sampled envelope not to truth.

## 4. Validation agenda — status

Items 1–5 of the 10 Nov proposal: **DONE today** (`results/certificate_audit/`, PRED
frozen first; 12/15 predicted sub-verdicts held; the 3 misses are recorded and are
themselves informative — near rows are mis-certified rather than undecided; the val
threshold never slipped). Item 6 (second dataset) is the remaining one; freeze the
prereg BEFORE running:

- **Datasets/backbones** (caches exist, sha-pinned in `verify_cvpr_repro.py`): timm
  wide_resnet50_2 features for MNIST, FashionMNIST, SVHN, CIFAR-100 (test splits), plus
  CIFAR-10 on WRN50 as the backbone-swap arm. Same pipeline (`epistemic_sets.run_world`
  with `feat_path` override; kuc=1, uuc_val=2, uuc_test=5; for CIFAR-100 cap kkc at the
  remaining classes and disclose the regime change).
- **Prereg predictions to freeze**: (P1) soundness holds — 0 false novelty certificates
  on knowns in every world of every dataset; (P2) coverage range shifts (does not
  reproduce CIFAR-10's 0.13–0.72); (P3) far > near certified rate in every world;
  (P4) absorbed → certified-known ≥ 95% in every world; (P5) width_mean ≤ BALD + 0.01
  everywhere (extends the negative). Seeds 0–4 per dataset; every number from the
  prereg'd script; any post-hoc change logged as amendment.
- **Do NOT explore** on these datasets before freezing (they are the confirmatory set).

## 5. Not worth trying (settled negatives)

Anything that reweights or learns over widths (adapters 5/5 fail; universal width
negative). Tercile stratification over pooled rows. Card-based scores.

## 6. Timeline (10 → 16 Nov)

- **10 Nov (today)**: ENROLL. Freeze item-6 prereg (`real/certificates_prereg.py`,
  pattern of `real/hardness_prereg.py`). Commit the working tree (deferred since 1 Nov:
  epistemic_sets, credal_head, certificate_audit, verify_cvpr_repro, this plan).
- **11 Nov**: run item 6 (laptop-scale, minutes per world); update tables; regenerate
  collaborator handoff zip.
- **12–14 Nov**: draft per §3; figures from recorded outputs only.
- **15 Nov**: claims audit against §2 (every sentence in the paper maps to a row).
- **16 Nov**: submit. Supplementary (23 Nov): full per-seed tables, replays, theorems.

## 7. Risks and fallbacks

- Item 6 soundness fails somewhere → the certificate story becomes CIFAR-10-only;
  fallback spine = §1.4 + §1.5 (instability + width negative, fully cross-dataset).
- Reviewer "certificates are just thresholds" → kill-test table (non-redundancy, outlier
  fragility of the val threshold, no calibration data needed).
- Reviewer "2 known classes is thin" → disclosed limitation; arms B/C (kkc=4/6) exist in
  the parked plan as future work; do not run new arms in the window.
- Single-dataset objection on positives → honest framing + cross-dataset negatives.

## 8. Hygiene (blockers before submission)

- **Bug**: `epistemic_sets.py:400` writes class names into `per_class.csv`'s auroc column
  (`"auroc": list(pc)` → `list(pc.values())`). Fix + regenerate per-class tables from
  `rows.csv` (analysis fix, log in manifest).
- `results/` and caches are git-ignored: the wiki sync is the backup (done today:
  203 files, 53 run pages, gates PASS).
- This file matches the repo's `*.MD` ignore rule (auxiliary notes stay local); force-add
  if it should travel with the branch.
- Known nondeterminism: WRN50 arm only (head training; ±0.005). All CLIP-world results
  byte-identical on re-run (`verify_cvpr_repro.py`).

## 9. UPDATE 10 Nov (evening) — item 6 EXECUTED under the frozen prereg; GATE FAILED

`real/certificates_prereg.py` frozen (sha fe2559a4…) before any run; executed by
`certificates_multidata.py` on 12 datasets × 5 worlds (mnist, fashionmnist, svhn,
cifar10@WRN50, cifar100, bloodmnist, pathmnist, organa/c/s-mnist, emnist-letters,
eurosat; CAMELYON17/FMoW not on disk and not class-taxonomies for this protocol —
PathMNIST/EuroSAT are the disclosed stand-ins). Results: `results/certificates_multidata/`.

- **P1 (the gate): FAIL — 111 false novelty certificates over 19,739 known test rows,
  in 11 of 12 datasets** (only cifar100 is clean, where the certificate never fires at
  all: coverage 0.000, 95% absorbed, kkc=92 regime collapse). Worst rates per known-row
  basis: organsmnist 12/199, svhn 11/262, bloodmnist 9/310. The CIFAR-10/CLIP zero was
  feature-space luck, not a property of the certificate. Per the frozen stop rule, no
  post-hoc rescoping of P1.
- **P3 FAIL**: far>near certified coverage flips in 9 worlds (eurosat w0 far 0.005 vs
  near 0.171; organamnist w2–4; organsmnist; fashionmnist w2/w4; bloodmnist w0). The
  near/far certificate direction is also not universal.
- **P4 PASS, and it is now the project's most robust result: absorbed unseen rows
  (P_unknown ≤ 0.01, 3–98% of uuc by dataset) receive the certainly-known certificate
  at rate 1.000 in all 60 worlds × 12 datasets.**
- P2 PASS (coverage escapes the CIFAR-10 band: 0.000–0.55 mean by dataset).
- P5 PASS: width − BALD ≤ +0.007 everywhere; the universal width negative now spans
  12 datasets + 5 CLIP worlds, two width constructions.

**Consequence (per the 10 Nov gate this plan was written under): the certificate paper
is not submittable for 16 Nov; the acquisition paper keeps the CVPR slot.** The
certificate/overconfidence findings survive as exploratory material for a later venue,
rescoped honestly: the K+1 envelope *mis-certifies* known rows at small, dataset-dependent
rates (0–6%) — the sampled set itself is wrong on those rows — while the overconfidence
half (absorbed → certainly known, 100%) and the width negative are universal.
