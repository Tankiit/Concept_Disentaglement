---
title: Glossary
type: reference
status: active
tags:
  - glossary
---

# 📖 Glossary

| Term | Definition |
| --- | --- |
| **REACH** | The method/prototype: acquisition-conditional epistemic uncertainty over a finite cell partition via LP intervals. |
| **Frozen classifier** | The predictor is fixed; data acquisition, not learning, is the only intervention. |
| **Cell** | One element of the fixed finite partition of input space; widths are reported per probe cell $x$. |
| **Log** | The unlabelled deployment data; the only evidence constraining the mixture. |
| **Labeled reference** | The labeled sample fixing the anchor posteriors; the only thing more labels shrink. |
| **MMD ball** | The set of input densities (mixture weights over anchors + unseen slot) the log cannot exclude; `--set l2` = exact variant. |
| **Unseen-class slot** | The $K{+}1$-th mixture component standing in for a novel source. |
| **$\Wcls$ / prevalence width** | Range of the source-posterior over the ball; narrowed by more log. |
| **$\Wlab$ / label width** | Range of the label posterior at fixed anchors over labeled-reference uncertainty; narrowed by more labels. |
| **$\Wunk$ / coverage width** | Range of the unseen-slot posterior $w_{K+1}q(x)/w^\top A(x)$; narrowed by class discovery. |
| **Inclusion** | Interval validity for joint intervals ("coverage" is reserved for $\Wunk$). |
| **Pinning** | Evaluating the three widths at shared configurations so acquisition attribution is on a common scale. |
| **Acquisition** | One of: unlabelled log, labels, class discovery — the three levers. |
| **Acquisition protocol** | The declared rules for how data can be acquired; epistemicness is defined relative to it. |
| **Tilt residual** | Above-radius residual under family tilt; diagnostic that the declared mixture family is wrong. |
| **Ball resolution** | How finely the ball separates candidate mixtures; gate `ball_resolution_gt_1`. |
| **Gates** | Ordered kill rules in `gates.py`; first failure blocks everything after. |
| **Prereg** | A frozen study design file `real/*_prereg*.py`; amendments are new files. |
| **Shard** | An independent replication over a disjoint seed range; pooled via `rows.csv` only. |
| **EM bootstrap baseline** | The comparison point for prevalence estimation in `train` (`em_baselines.json`). |
| **Matched arms** | Two acquisition arms equalized on baseline `W_total`; the only valid width-vs-width comparison. |

Related: [[01_Core_Claim_and_Widths]] · [[02_MMD_Ball_and_Intervals]] · [[02_Decision_Gates]].
