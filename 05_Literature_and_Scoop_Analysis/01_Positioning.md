---
title: Positioning and Prior Work
type: literature
status: active
tags:
  - literature
  - positioning
  - scoop-analysis
---

# 📚 Positioning and Prior Work

From `paper/refs.bib`. Every citation was checked against the cited work (per the paper's AI statement) — keep that true for any addition.

## REACH vs the Label-Shift Estimation Line

Saerens et al. 2002; Lipton et al. 2018 (BBSE); Azizzadenesheli 2019; Alexandari 2020 (MMU); Garg 2020 (unified view); Tian 2023 (ELSA); Dussap 2023 (robust quantification); Li 2026 (matrix-inversion → constraint regions for importance weights).

- **They:** estimate/adjust under label shift, mostly point estimates or confidence regions for **weights**.
- **REACH:** attributes uncertainty to **acquisitions** per input, with an unseen-class slot; weights are only the intermediate object. Closest in spirit: Li 2026's constraint regions — difference is REACH's question ("which data") and the three-width decomposition.

## Open-Set and Mixture Lines

Garg 2022 (open-set label shift), Zhang 2013 (target+conditional shift), Ramaswamy 2016 (mixture proportion via kernel embeddings), Iyer 2014 (MMD class-ratio bounds).

- **They:** detect or adapt under shifted/open mixture structure.
- **REACH:** declares the family up front and certifies **what remains unidentified** when the log cannot exclude densities — detection appears only as the tilt residual diagnostic (`conditionals`).

## Uncertainty Taxonomies

Hüllermeier 2021; Depeweg 2018; Lakshminarayanan 2017 (deep ensembles); Gruber 2023; **Young 2026 — "Epistemic Uncertainty Is Not the Reducible Kind"**.

- Young 2026 is the philosophical sharp edge: reducibility ≠ epistemicness. REACH's answer is operational — epistemic is defined **relative to a declared acquisition protocol**: the width that a specific acquisition (log/labels/discovery) would narrow. Cite and engage explicitly; this framing is a selling point, not a threat.

## Distribution-Free Uncertainty Under Shift

Podkopaev 2021; Si 2023 (PAC prediction sets under label shift); Zhao 2021 (active learning under label shift).

- **They:** valid prediction sets / active learning under label shift.
- **REACH:** no set-validity claim; intervals are LP-certified over the declared ball. Zhao 2021 is the nearest active-acquisition neighbor — REACH differs by attributing **which kind of data** at the width level before any learner is updated.

## Feature Backbones / Real Data

Radford 2021 (CLIP), Oquab 2024 (DINOv2), Hendrycks 2019 (CIFAR-10-C), Recht 2018 (CIFAR-10 test reuse) — used only as **frozen cached features** for the audit; no claims about the backbones themselves.

## Scoop Watch

- Re-check `li2026matrix` and `young2026epistemic` before any revision — both are recent and adjacent.
- Any new "acquisition-attributed uncertainty" phrasing in OpenReview/arXiv should land here with a one-line delta.
