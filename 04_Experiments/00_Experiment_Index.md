---
title: Experiment Index
type: index
status: active
tags:
  - experiments
  - index
---

# 🧪 Experiment Index

All commands run from `REACH_AISTATS/`. `--help` works on the top level and each subcommand. Quick sanity: `python main.py test` (~1 s, 18+ tests).

| Command | Question it asks | What a pass looks like | Note |
| --- | --- | --- | --- |
| `train` | Do intervals cover truth, and do widths respond to each acquisition? | Coverage ≥ nominal per budget; signed contrasts in the predicted direction | Pilot bundle: contrasts, coverage, precision, EM baselines, widths.png + SHA-256 manifest |
| `separation` | Does a near-singular source embedding resolve slowly? | $\Wcls$ widens as tangent gap shrinks **while coverage holds at every gap** | [[01_Response_Matrix]] |
| `matched` | Does one total width tell you which acquisition helps? | Two arms matched on baseline `W_total` with asymmetry in log-gain minus label-gain | Rule 7 in [[01_Operating_Rules]] |
| `conditionals` | Is the reference radius optional? Is a wrong family detected? | Nominal weight coverage with the radius, lost without it; residual above radius under tilt | [[02_MMD_Ball_and_Intervals]] |
| `evaluate` | Does the pipeline run on real cached features? | Acceptance, accepted error, width, mixture-incompatibility flag | cache.npz input |
| `response` | Full synthetic response matrix | Seeds 128; `--set l2` for exact MMD ball | Feeds `response_matrix_done` gate |
| `audit` | Real label-shift audit on cached CLIP features | Groupings `super4` / `classes`, held-out | See [[02_Real_Shift_Audit]] |

Smoke test:
```bash
python main.py train --out results/smoke --seeds 2 --n 16 32 --m 8 --eta 0 \
                     --worlds separated --bootstrap 2
```

Deep dives: [[01_Response_Matrix]] · [[02_Real_Shift_Audit]] · [[03_Allocation_Dose_and_Preregisters]].
Sharding protocol and timings: [[01_Repo_Map_and_Commands]].
