---
title: "REACH: Acquisition-Attributed Epistemic Widths for Frozen Classifiers"
type: project-wiki
status: active
priority: 1
target: AISTATS 2027
owner: Tanmoy
review: weekly
tags:
  - epistemic-uncertainty
  - label-shift
  - mmd-ball
  - acquisition
  - frozen-classifier
  - AISTATS2027
next_action: "Run `python gates.py` in REACH_AISTATS — the first failing gate blocks everything after it"
created: 2026-10-14
last_modified: 2026-10-14
---

# REACH: Which Data Would Reduce This Uncertainty?

> **Target:** AISTATS 2027
> **Status:** Active — submission artifacts built (`paper/main_submission.pdf`, supplement zip)
> **One-Line Goal:** For a frozen classifier with a labeled reference and an unlabelled log, estimate **per input** which acquisition — more log, more labels, or class discovery — would shrink which epistemic width, and which width stays unidentified.

---

## 🗂️ Vault Structure & Quick Navigation

```
REACH/
├── 00_Dashboard.md                       <-- You are here (master hub & status board)
├── 01_Operating_Rules.md                 <-- Binding research & reporting principles
├── 02_Theory/
│   ├── 01_Core_Claim_and_Widths.md       <-- The three widths and what contracts under which acquisition
│   └── 02_MMD_Ball_and_Intervals.md      <-- Uncertainty set, LP intervals, pinning, joint inclusion
├── 03_Scope_and_Gates/
│   ├── 01_Project_Scope.md               <-- Epistemic-only scope; explicit non-claims
│   └── 02_Decision_Gates.md              <-- gates.py kill rules (first failure blocks)
├── 04_Experiments/
│   ├── 00_Experiment_Index.md            <-- Command → question → pass-looks-like table
│   ├── 01_Response_Matrix.md             <-- Synthetic: train / separation / matched / conditionals / response
│   ├── 02_Real_Shift_Audit.md            <-- CLIP features, label-shift audit on cached data
│   └── 03_Allocation_Dose_and_Preregisters.md  <-- real/ prereg registry: alloc, dose, bridge, blindspot, exchange
├── 05_Literature_and_Scoop_Analysis/
│   └── 01_Positioning.md                 <-- vs label-shift estimation, EU taxonomies, conformal under shift
├── 06_Implementation/
│   └── 01_Repo_Map_and_Commands.md       <-- Repo layout, entry points, sharding protocol
├── 07_Paper_Roadmap/
│   └── 01_AISTATS_Status.md              <-- Sections, Lean formalization, AI statement, build
├── 08_Glossary.md                        <-- Widths, ball, pinning, cells, preregs, gates
└── Project_Canvas.canvas                 <-- Pipeline at a glance
```

Code root: `Uncertainty_Modeling/EU_AU_Disentanglement/REACH_AISTATS/` (paper, lean, tests, real-data scripts). Older top-level scripts (`latent_subspace.py`, `attention_prompt.py`, `wilds_expts.py`) belong to the broader EU/AU disentanglement thread — see [[01_Project_Scope]].

---

## 📊 Status Board

| Component | Status | Evidence |
| --- | --- | --- |
| Paper draft | Complete, built | `paper/main.pdf`, `paper/main_submission.pdf`, supplement zip |
| Lean formalization | Present | `lean/Reach.lean`, `lean/Axioms.lean` (standard axioms only, per AI statement) |
| Test suite | Green assumed — verify | `python main.py test` (18+ numerical/integration tests, ~1 s) |
| Kill gates | Run before any claim | `python gates.py` → see [[02_Decision_Gates]] |
| Synthetic response matrix | Runbook defined | `response --seeds 128`, `--set l2` exact MMD ball |
| Real-data audit | Scripts ready | `real/modal_extract.py` (CLIP, A10G) → `audit --grouping super4/classes` |
| Preregistered studies | Registry live | `real/*_prereg*.py` — see [[03_Allocation_Dose_and_Preregisters]] |

---

## ✅ Next Actions

1. Run `python gates.py` — confirm all four gates green before quoting any number.
2. Run `python main.py test` to confirm the environment still passes the suite.
3. Check AISTATS 2027 cycle deadlines against current artifact state (`main_submission.pdf` already built — confirm what remains editable).
4. Review [[03_Allocation_Dose_and_Preregisters]] for open preregistered studies vs completed ones.

---

## 🔗 Key Links

- Theory: [[01_Core_Claim_and_Widths]] · [[02_MMD_Ball_and_Intervals]]
- Rules: [[01_Operating_Rules]]
- Experiments: [[00_Experiment_Index]]
- Paper: [[01_AISTATS_Status]]
