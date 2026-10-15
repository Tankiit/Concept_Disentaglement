---
title: Synthetic Response Matrix
type: experiment-spec
status: active
tags:
  - experiments
  - synthetic
  - response-matrix
---

# 📈 Synthetic Response Matrix

## Purpose

The synthetic program validates **machinery**: intervals cover truth, they contract when they should, and the designed failures still fail. It does not settle manuscript definitions (owner work, per `COLLABORATOR_PLAN.md`).

## The Five Synthetic Experiments

### train — coverage & responsiveness
- Varies log size, labeled sample size, label noise across seeds; checks coverage per budget and signed contrasts per acquisition.
- Compares prevalence estimation against an **EM-based bootstrap baseline** (`em_baselines.json`).
- Outputs the pilot bundle + a manifest with SHA-256 of every source file.

### separation — slow resolution is honest
- Shrinks the tangent-space gap between the two known sources toward singularity.
- **Pass:** $\Wcls$ widens as the gap shrinks while coverage holds at **every** gap. Widening without coverage loss is the correct behavior; coverage loss anywhere is a kill.

### matched — the total-width trap
- Selects two arms matched on baseline `W_total`, evaluates log-gain vs label-gain asymmetry.
- Establishes that one total width does **not** identify which acquisition helps — the motivation for pinning ([[02_MMD_Ball_and_Intervals]]).

### conditionals — radius and family detection
- With reference radius: nominal weight coverage. Without: lost. Under tilt: residual above radius flags a wrong declared family.

### response — the matrix
```bash
python main.py response --seeds 128 --out results/response_s128              # ~5 min
python main.py response --seeds 128 --set l2 --out results/response_l2_s128  # exact MMD ball
```
Feeds the `ball_resolution_gt_1` and `response_matrix_done` gates in [[02_Decision_Gates]].

## Seed Discipline

- 8-seed evidence is not a constraint — compute is cheap (256 seeds ≈ 2.7 min; all ten paired contrasts hit ±0.01 bootstrap half-width, worst 0.0063).
- Shards must use **disjoint `--seed-start` ranges**; pool by concatenating `rows.csv`, never by hand-merging `summary.json`. The binding constraint is deciding what the paper claims, not CPU time.

Related: [[00_Experiment_Index]] · [[01_Core_Claim_and_Widths]].
