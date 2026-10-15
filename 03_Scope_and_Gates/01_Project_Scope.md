---
title: Project Scope
type: scope
status: active
tags:
  - scope
  - non-claims
---

# 🧭 Project Scope

## REACH Is

A **finite-cell prototype for acquisition-conditional epistemic uncertainty**:
- A fixed finite partition of input space (cells), two known sources plus **one candidate novel source**, binary target.
- A declared **mixture family** and a declared **acquisition protocol** as inputs.
- Output: which uncertainty about target label probabilities **contracts** when you add unlabelled inputs, labels, or class discovery — and which **stays unidentified**.

## REACH Is Not (Explicit Non-Claims)

| Non-claim | Reason |
| --- | --- |
| Aleatoric uncertainty estimation | Out of scope — labels/noise enter only as information interventions |
| Density-ratio estimation as a deliverable | Stays **diagnostic** only |
| A trained model / neural optimizer | No training loop; `train` means statistical fitting, not gradient descent |
| Any-classifier generality claims | Frozen classifier with labeled reference + unlabelled log is the setting |
| Post-hoc "true scene" style guarantees | Intervals are LP-certified over the declared ball; nothing beyond |

## Repo Boundaries

- **This project's code root:** `Uncertainty_Modeling/EU_AU_Disentanglement/REACH_AISTATS/` (main.py, gates.py, paper/, lean/, test/, original/, real/).
- **Adjacent but separate:** top-level `latent_subspace.py`, `attention_prompt.py`, `wilds_expts.py`, `variant_latent_space.png` and `backup/` belong to the broader EU/AU disentanglement thread and the sibling `EU_CBM` project. Do not cite them as REACH evidence.
- Real-data path is **cached features only** (CLIP via Modal on A10G → `modal volume get reach-aistats`); no encoder training in this repo.

## Deferred / H2+

- Beyond-binary targets and $>2$ known sources beyond the prototype's two.
- Actively comparing against EM/BBSE-style point estimators beyond the existing bootstrap baseline in `train` (already covered) — further baselines only if a gate demands it.
