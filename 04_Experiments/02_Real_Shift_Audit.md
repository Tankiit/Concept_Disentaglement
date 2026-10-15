---
title: Real Shift Audit (CLIP features)
type: experiment-spec
status: active
tags:
  - experiments
  - real-data
  - clip
  - label-shift
---

# 🔍 Real Shift Audit

## Pipeline

```bash
modal run real/modal_extract.py            # CLIP features (A10G) → Modal volume
modal volume get reach-aistats features cache/
python main.py audit --grouping super4  --out results/audit_super4_s50   # label shift, held-out
python main.py audit --grouping classes --out results/audit_classes_s50
```

Also in `real/`: `modal_audit.py` (audit-side Modal entry), `paper_figs.py` (figure generation from saved results).

## What It Tests

The synthetic machinery on **cached real features** under a held-out label-shift protocol:
- Acceptance and accepted-error rates, widths, and a **mixture-incompatibility flag** (the real-data analogue of the `conditionals` tilt residual — does the declared family survive contact with the data?).
- Two groupings: `super4` (coarse) and `classes` (fine) — resolution of the audit depends on grouping granularity, echoing the ball-resolution gate.

## Standing Constraints

- Features are **cached and frozen** — no encoder training, no fine-tuning; the classifier is frozen by construction of the method.
- The audit is evidence about the **pipeline running on real data**, not a leaderboard claim — see [[01_Project_Scope]] non-claims.

Related: [[00_Experiment_Index]] · [[03_Allocation_Dose_and_Preregisters]].
