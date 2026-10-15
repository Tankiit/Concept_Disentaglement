---
title: Decision Gates (gates.py)
type: gates
status: active
tags:
  - gates
  - kill-rules
---

# 🚦 Decision Gates

Source of truth: `gates.py` in REACH_AISTATS — "one yes/no per kill rule, read from saved results. **The first failing gate blocks everything after it.**"

```bash
python gates.py    # exit 0 = all green, exit 1 = at least one red
```

## The Four Gates (in order)

| # | Gate | Checks | Read from |
| --- | --- | --- | --- |
| 1 | `ball_resolution_gt_1` | Ball resolution above the floor for synthetic, `response_s128`, and `response_l2_s128` | `ball_resolution_gate.min_ball_resolution` in each summary |
| 2 | `invariants_green` | Solver-level invariants hold | saved invariant summaries |
| 3 | `a12_coverage_ok` | A1/A2 coverage at/above nominal per budget | coverage summaries |
| 4 | `response_matrix_done` | The full response matrix has been produced | `response` run summaries |

## Operating Semantics

- Gates read **saved results** — run the experiments first, then the gate; it never recomputes.
- Ordered kill rule: a failure at gate $k$ invalidates any statement that depends on gates $> k$. Fix in order.
- Gate results feed the [[00_Dashboard]] status board. Rule 1 of [[01_Operating_Rules]]: gates before claims.

## Kill/Pivot Triggers

- Ball resolution at/below floor → the log cannot resolve the mixture family; do not quote widths, increase log or report slow resolution honestly (`separation` world documents this mode).
- Coverage below nominal at any budget → the interval machinery is broken; everything downstream is void.
- Response matrix incomplete → no acquisition-attribution claim may be made.

Related: [[01_Response_Matrix]] · [[02_MMD_Ball_and_Intervals]].
