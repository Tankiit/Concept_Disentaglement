---
title: AISTATS 2027 Paper Status
type: paper-roadmap
status: active
tags:
  - paper
  - aistats2027
---

# 📄 AISTATS 2027 Paper Status

**Title:** *Which Data Would Reduce This Uncertainty? Acquisition-Attributed Epistemic Widths for Frozen Classifiers*
**Venue:** AISTATS 2027 (anonymous submission — `aistats2027.sty`, ` Anonymous Authors`).

## Artifacts

| Artifact | Path (from `REACH_AISTATS/`) | State |
| --- | --- | --- |
| Main source | `paper/main.tex` + `sections/{intro,setup,theory,related,experiments,limitations}.tex` | Complete draft |
| Appendix + checklist | `sections/appendix.tex`, `sections/checklist.tex` | Present |
| Builds | `paper/main.pdf`, `paper/main_submission.pdf` | Built |
| Supplement | `paper/REACH_AISTATS_supplement.zip`, `REACH_AISTATS_latex.zip` | Packaged |
| Build helper | `paper/build_modal.py` | Remote build path |
| Bibliography | `paper/refs.bib` | Curated; see [[01_Positioning]] |
| Lean | `lean/Reach.lean`, `lean/Axioms.lean` | Deterministic proof steps machine-checked, standard axioms only |

## Section Map

- **intro / setup:** the question, the three widths ($\Wcls$, $\Wlab$, $\Wunk$), pinning, joint intervals — [[01_Core_Claim_and_Widths]].
- **theory:** when more logging narrows prevalence and coverage widths under frozen class-conditionals; label width immovable by logging.
- **related:** label shift, open set, EU taxonomies, distribution-free under shift — [[01_Positioning]].
- **experiments:** synthetic response matrix + real audit — [[01_Response_Matrix]], [[02_Real_Shift_Audit]].
- **limitations:** finite-cell prototype scope — [[01_Project_Scope]].

## AI Use Statement (already in draft)

Publicly commits to: AI for code/prereg drafting, Lean formalization drafts, related-work checking, simulated reviews; **not** for the research question or frozen abstract; all numbers regenerable from released scripts; Lean checked with standard axioms; every citation verified. This is a **binding commitment** — see Rules 3–4 in [[01_Operating_Rules]].

## Before Any Revision

1. `python gates.py` green (Rule 1).
2. Re-check the two sharp-edge references (`li2026matrix`, `young2026epistemic`) for updates.
3. Any proposition edit → re-check `lean/` and `test/` (Rule 8).
4. Confirm the AISTATS 2027 cycle deadline state before rebuilding `main_submission.pdf`.
