---
title: Allocation, Dose, and the Prereg Registry
type: experiment-spec
status: active
tags:
  - experiments
  - prereg
  - allocation
  - dose-response
---

# 📋 Allocation, Dose, and the Prereg Registry

`real/` contains the pre-registered study program. Files ending `_prereg*.py` are **frozen designs**; amendments are new files (see Rule 3 in [[01_Operating_Rules]]).

## Registry

| Study | Prereg file(s) | Analysis / amendment | Topic |
| --- | --- | --- | --- |
| Allocation | `alloc_prereg.py` | `alloc.py` | Budget allocation across acquisitions — the "which data next" decision (recent commit: allocation analysis added) |
| Dose–response | `gauss_dose_prereg.py`, (exchange) `exchange_prereg.py` | `dose.py`, `gauss.py`, `exchange.py` | Width vs amount-of-data curves per acquisition |
| Bridge v1 | `bridge_prereg.py` | `bridge.py`, `bridge_analysis.py` | Synthetic → real bridge |
| Bridge v2 | `bridge2_prereg.py` | `bridge2.py`, `bridge2_analysis.py`, `bridge2_amendment.py` | v2 with a logged amendment — the canonical amendment pattern |
| Blind spot | `blindspot_prereg.py`, `blindspot_prereg_v2.py` | `blindspot.py`, `blindspot_v2.py` | Where the ball cannot see (slot/unseen-class behavior) |
| Expectations | — | `expectations.py`, `head.py` | Supporting estimation machinery |

## Operating Notes

- Before running any study: read its prereg **in full**; the prereg defines the pass/fail labels, not the outcome.
- After a run: decision labels go to the [[00_Dashboard]] status board; a failed prereg decision is a **result**, not a bug (unless a code bug is documented in an amendment).
- v2 of a study supersedes v1 only via its own prereg — never by silently editing the original.

Related: [[01_Core_Claim_and_Widths]] (from widths to an acquisition) · [[02_Decision_Gates]].
