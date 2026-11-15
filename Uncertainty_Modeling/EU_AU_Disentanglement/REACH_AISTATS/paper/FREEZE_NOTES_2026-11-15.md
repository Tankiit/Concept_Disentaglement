# Freeze notes — 15 Nov 2026 (pre-deadline pass on `acquisition_draft.tex`)

Deadline: CVPR 2027 paper, 16 Nov AoE. Source of truth: `acquisition_draft.tex`
(compiled with `tectonic`; PDF rebuilt 15 Nov 18:11). `acquisition_draft.md` is
STALE (last synced 12 Nov) and is now marked as such at the top.

## 1. The incoming revision plan vs the live draft

A revision plan arrived 15 Nov written against a stale copy (same drift as the
14 Nov external review, whose TODO@93 / states-390/566-in-intro quotes matched
no file — always grep the live `.tex` before acting). Claim by claim:

| Plan claim | Live `.tex` (14 Nov 22:25 state) |
| --- | --- |
| Abstract (l.008) says "regret" + "mixed-mixture framework" | Neither string exists in `.tex` or `.md`. Abstract never says "establish" either. |
| Intro ll.083–085 still say "regret" | Fixed 14 Nov: "regret" only in Appendix A; §3 Proposition says "worst-case loss in expected squared error". |
| Fig 1 = method figure per 14 Nov decision; PDF shows a width table; text describes two matched states | Backwards. The recorded 14 Nov decision (pre-freeze review, point 1): the real-patch contraction grid STAYS; the mixture schematic moved to the tutorial. Live Fig 1 = `fig1_main.pdf` (patch + contraction bars); ¶2 and the caption match it. |
| TODO at l.101 | None in the live `.tex`. |
| Figs 13/14 | The live draft has 11 figures; the real-record pair figures are Figs 7–8. |
| Generic opening paragraph | Partly true (opener was abstract); addressed today with a concrete ¶1. |
| Practitioner sentence missing | True; addressed today via the licensed sentences. |

## 2. The two open items (answers, with artifacts)

**Q1 — How many selected pairs are behind the real-record figures?**
One pair per dataset, out of 600 candidate states each. 39 qualifying pairs on
CAMELYON17, 32 on FMoW; the frozen rule evaluates only the best pair (smallest
relative MI difference, tie-break on state IDs). CAMELYON17 = states 390
(identity-preferred) + 566 (label-preferred); FMoW = 287 + 561.
Artifacts: `workbook/camelyon17_counterexample_20261030T092024988890Z/` and
`workbook/fmow_counterexample_20261030T094929883846Z/` (`result.json` has
`candidates/matches`, `qualifying_pairs.csv`, `selected_states.csv`,
`heldout_exact_one_step.csv`).

**Q2 — Were states 390, 566, 287, 561 chosen without looking at held-out error?**
Yes. `real/wilds_counterexample.py` l.29 freezes
`selection='smallest relative MI difference; then state IDs; never use
evaluation outcomes'`. The selection code (ll. 60–81) uses only model
quantities (mean/variance/MI, model channel values from `candidate_states.csv`),
writes `selected_states.csv`, and only then computes the held-out endpoint
(`heldout_exact_one_step.csv`, l.106). The FMoW held-out failure was retained,
not relabelled (draft §4.2 states this). No affirmative "establish" appears in
the draft, so nothing needed weakening.

## 3. Terminology (as built — one name per concept)

- **The act:** acquisition / acquiring observations (title, §4–5). "Collection"
  wording used in the new ¶1 and the Fig 1 caption lead ("the data collected").
- **The four options:** *observation channels* — unlabeled deployment inputs,
  source identities, reference inputs, target labels (defined §2.2). The stray
  "acquisition channel" occurrences (abstract, intro ¶3, contribution 3, §2.3)
  were renamed today; 8 occurrences now, 0 of the old compound.
- **The estimand:** *target* (three named targets, §2.1).
- **The intervals:** *compatible-set intervals*; "interval width"/"width" for
  their contraction — matches Fig 1 panel (b) "Width contraction", so "width"
  stays (a rename to "bounds" would desync text and figure the day before the
  deadline). `W_*` notation belongs to the AISTATS paper, not this draft.
- **Retired today:** "acquisition channel" (→ observation channel), "data
  bought" (→ data collected). "Acquisition" stays as the standard term for the
  act throughout — the incoming plan's "collecting" sweep was calibrated to the
  stale copy and was not applied wholesale.
- Word-search status: "regret" = Appendix A only; "credal" = 0; both verified
  by grep after the edits.

## 4. Edits made today

1. New concrete ¶1 (hospital scenario, no method words) before the existing
   opener; italic question and Fig 1 reference untouched.
2. §5.2: new paragraph with the two licensed acquisition-gap sentences
   (FMoW robust-and-wrong, 588/600 robust / 33% wrong; CAMELYON17 wrong more
   often than right below ~32 identity observations, 56% vs 30%), per the
   10 Nov collaborator scheduling (CVPR takes only these two sentences from
   that thread; numbers frozen in vault `04_Experiments/06_Acquisition_Gap_and_Sensitivity_Band.md`).
3. Intro: one-sentence practitioner caution after the contributions list.
4. Terminology: 4× "acquisition channel" → "observation channel"; caption
   "data bought" → "data collected".
5. `acquisition_draft.md` marked STALE; PDF recompiled (tectonic, clean).

## 5. Still open before submission

- 14 Nov review item 3: `tests/test_two_world.py` (rational enumeration of the
  channel values 0.0324/0.0081/0, minimax 81/5000) — not written yet.
- Commit decision: 13–15 Nov changes uncommitted on `reach-mateus-workbook`
  (fig1 polish, licensed sentences, these edits).
- Supplementary (23 Nov): consider packaging the 600-state audit runs backing
  the new §5.2 sentences (`real/acquisition_gap.py`, `real/band_beyond_budget.py`,
  workbook run dirs, wiki page 04_Experiments/06).
- The audit sentences in §5.2 cite recorded runs not described elsewhere in the
  paper; if reviewers need the protocol, it goes in the supplementary, not the
  main text (per the 10 Nov scheduling).

## 6. Tiny ImageNet addition (15 Nov, evening)

Collaborator-delivered folder (Downloads copy) archived verbatim at
`workbook/TinyImageNet_REACH_Results_v2full/` (summary / intervals /
exact_identification_benchmarks / reference_eval_transfer CSVs, audit.json,
their own 6-panel figure PDF/PNG). `audit.json` records: sources = fixed
per-image corruptions (clean/blur/noise), event = classifier top-1 error,
centroids from the fit split and known sources only, eval labels never used to
build the world, Jeffreys event-rate smoothing, 2% uniform-cell replacement,
same budgets (0–1e5), 6 repeats, and same baseline (500/200-per-source/100/0)
as the four-dataset protocol.

Verified numbers (all from the CSVs; QA re-printed by `real/fig_tin.py`):
- Exact input-only floors: known-membership 0.8628 (confounded) vs 0.0
  (separable); candidate-membership 0.9543 (confounded).
- Identities → 0.3176 (separable) / 0.3294 (confounded) at 1e5; inputs →
  0.6385 (separable) / 0.9989 (confounded, ≥ 0.993 at every budget).
- 25 of 30 confounded known-membership input intervals are returned by the
  undefined-denominator policy (widths 0.9935–1.0) — disclosed via hollow
  markers + caption. 67/720 intervals use the policy overall; all 720 contain
  their declared truth.
- Why identities flatten at ~0.32–0.33 instead of 0 (answered from the data,
  now in Appendix B): identities pin the mixture weights but add no reference
  observations; the residual is cell-law uncertainty at the probe cell, which
  only references reduce — the reference arm approaches the complementary
  floor (0.40 at 1e5, separable) from the other side.

Draft edits: abstract dataset list (+ "corruption-defined sources"), intro
contribution (3) clause, §4.1 retitled "Five image-derived source models" +
setup ¶ + confounding sentence, §5.1 floors ¶ + Figure `fig:tin-floor` +
coverage sentence (720/720, 67 policy), Appendix B (`app:tin`) with the
recorded 6-panel figure + audit line, `deng2009` bibitem, commented-out
acknowledgment block before the bibliography.
`real/fig_tin.py` → `paper/figures/fig_tin_floor.{pdf,png}` at exactly
6.50 × 2.20 in (natural size = full text width), house palette (IDs blue,
unlabeled grey), direct end labels, dashed exact floor in (b) only.

Plan-template corrections vs the incoming instructions: its caption template
said "(a) inputs shrink the bound slowly toward 0" (actually 0.64 at 1e5, floor
exactly 0) and "(b) identities reach 0.33" (confirmed); it did not know that
the (b) inputs curve is ~1.0 largely BY POLICY (hollow markers added). The
incoming plan again carried stale line anchors (TODO@101, "four
controlled"@126) — mapped by intent onto the live draft.

Authorship: per CVPR 2027 Author Guidelines the author list is locked at the
registration deadline (10 Nov) — only reordering is possible after; adding the
Tiny ImageNet contributor as an author is therefore likely impossible unless
they were already registered. Verify in the OpenReview console; the
acknowledgment block is ready for camera-ready.
