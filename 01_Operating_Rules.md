---
title: Research & LLM Operating Rules
type: protocol
project: REACH (Acquisition-Attributed Epistemic Widths)
status: mandatory
tags:
  - protocol
  - operating-rules
  - reproducibility
---

# 📜 Research & LLM Operating Rules

These rules govern all automated and manual research activities in this repository. They are binding and must be verified before and after any experiment or claim.

---

## 🔒 The 8 Core Operating Principles

### 1. Gates Before Claims
> **Directive:** Run `python gates.py` before quoting any result. The gates are ordered kill rules — **the first failing gate blocks everything after it**. A red gate means the number does not exist for reporting purposes.

### 2. Never Report a Width Without Its Coverage
> **Directive:** A width is only meaningful if the interval covers truth. Every width claim must be accompanied by coverage at or above nominal for that budget. A narrower interval at broken coverage is **not** a win.
> - Width comparisons across acquisitions happen only at matched baseline `W_total` (that is what the `matched` command exists for).

### 3. Preregistrations Are Binding
> **Directive:** Files matching `real/*_prereg*.py` are frozen study designs. Amendments are written as **new files** (the `bridge2_amendment.py` pattern), never as edits to an existing prereg. Analysis code may be fixed only for bugs, logged in the amendment.

### 4. Every Number Is Regenerable
> **Directive:** Every reported number must be reproducible from the released scripts and result files (this is the paper's public AI-use commitment). No hand-merged `summary.json` files — pool by concatenating `rows.csv` across **disjoint-seed** shards and re-analyse from Python.

### 5. Scope Discipline: Epistemic Only
> **Directive:** REACH quantifies **epistemic uncertainty only**. Labels and label noise enter as information interventions; there is no aleatoric estimator and density-ratio estimation stays diagnostic. Never let a claim drift into aleatoric territory or into "true coverage" language that the finite-cell model cannot support.

### 6. Disjoint Seeds or It Didn't Happen
> **Directive:** Sharded runs must use disjoint `--seed-start` ranges. Overlapping seeds recompute the same draws and buy nothing. Two shards agreeing on every decision label is the first-class replication check.

### 7. Matched Arms or Don't Compare
> **Directive:** Total width alone does not say which acquisition helps (the `matched` experiment exists because of this). Any "acquisition A beats B" statement requires arms matched on baseline `W_total` with the asymmetry read off log-gain vs label-gain.

### 8. Paper, Lean, and Tests Move Together
> **Directive:** A proposition edit in `paper/` requires re-checking `lean/Reach.lean` (statements are machine-checked with standard axioms only) and the corresponding tests in `test/`. The three artifacts must not drift.

---

## 🧪 Session Checklist

Before any work: read [[00_Dashboard]], run `gates.py`, run `main.py test`.
After any work: re-run gates, confirm shard seeds were disjoint, update [[00_Dashboard]] status board and `last_modified`.
