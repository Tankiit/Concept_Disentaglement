---
title: Core Claim and the Three Widths
type: theory
status: stable
tags:
  - theory
  - epistemic-widths
  - acquisition
---

# 🎯 Core Claim and the Three Widths

**Paper title:** *Which Data Would Reduce This Uncertainty? Acquisition-Attributed Epistemic Widths for Frozen Classifiers* (AISTATS 2027 submission).

## One-Paragraph Claim

For a **frozen classifier** with a labeled reference set and an unlabelled log, REACH estimates, **per input**, which acquisition would reduce which uncertainty. From the log it forms an [[02_MMD_Ball_and_Intervals|MMD ball]] of input densities the log cannot exclude, spanning prevalence mixtures of the known classes **plus a slot for an unseen class**. The spread of induced label posteriors defines three epistemic widths, each tied to a different acquisition. Under frozen class-conditionals, class distinguishability and support determine when more logging narrows prevalence and coverage widths — while the label width, fixed by the labeled reference, does not move.

## The Three Widths (per probe cell $x$, source $k$)

| Width | Symbol | Defined as | Narrowed by |
| --- | --- | --- | --- |
| Prevalence | $\Wcls(x)$ | Range of $w_k A_k(x) / \sum_{j\le K} w_j A_j(x)$ over $w \in \mathcal{W}_n$ | More **unlabelled log** |
| Label | $\Wlab(x)$ | Range of the label posterior at fixed anchor weights over labeled-reference uncertainty (rows of $A$ fixed; ranges over entries $A_k(x)$) | More **labels** |
| Coverage | $\Wunk(x)$ | Range of $w_{K+1} q(x) / w^\top A(x)$ — mass on the unseen-class slot | **Class discovery** |

Notes from `paper/sections/setup.tex`:
- $\Wlab$ instances share a **role** (the width only labels narrow) but not an estimand across settings; `prop:b` covers only the fixed-anchor instance.
- $\Wcls$ and $\Wunk$ evaluate the posterior at $\hat A$; **joint intervals** carry inclusion of the target. The word "coverage" is reserved for the width $\Wunk$, not for interval validity.

## Attribution by Pinning

The three widths range over **different posteriors**, so their largest points are not on a common scale. Attribution is by **pinning**: evaluate all three at common pinned configurations so the comparison is meaningful. See [[02_MMD_Ball_and_Intervals]].

## From Widths to an Acquisition

At budget $B$, with one label costing $c$ unlabelled inputs, REACH predicts how more log or more labels would change the **joint width**, holding current estimates fixed. The `matched` experiment is the empirical guardrail — see [[01_Response_Matrix]].

## What Stays Unidentified

Which uncertainty **stays unidentified** under a declared mixture family and acquisition protocol is itself a first-class output: every quantity is an interval computed by LP over an explicit uncertainty set, so a claim either survives the solver or it does not. The `separation` experiment shows the failure mode: as the tangent-space gap between sources shrinks toward singularity, $\Wcls$ widens **while coverage must hold at every gap** — slow resolution is honest, broken coverage is not.
