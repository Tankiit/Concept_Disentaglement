---
title: MMD Ball, LP Intervals, and Pinning
type: theory
status: stable
tags:
  - theory
  - mmd
  - linear-programming
  - intervals
---

# ⭕ MMD Ball, LP Intervals, and Pinning

## The Uncertainty Set

From the unlabelled log, REACH forms a **maximum mean discrepancy (MMD) ball** of input densities: every mixture $w$ over the known-class anchors plus the unseen-class slot whose MMD to the empirical log distribution is below the ball radius. The log **cannot exclude** anything inside the ball — that is the entire epistemic state.

- Default ball: kernel-MMD radius from the log (`--set` default).
- `--set l2` gives the **exact MMD ball** variant used in the response matrix.
- The weight set $\mathcal{W}_n$ is the intersection of the ball with the simplex (prevalences of $K$ known sources + slot $K{+}1$).

## Everything Is an Interval by LP

Every reported quantity is an **interval computed by linear programming** over the explicit uncertainty set (Charnes–Cooper for the fractional posteriors — `charnes1962programming` in the bib). Consequences:

1. A claim either **survives the solver** or it does not — no asymptotic hand-waving.
2. Widths are certified ranges, not point estimates with error bars.
3. The `invariants_green` gate checks the solver-level invariants before any downstream number is trusted.

## Joint Intervals vs Widths

- $\Wcls$ and $\Wunk$ evaluate posteriors at the point estimate $\hat A$.
- **Joint intervals** additionally propagate the anchored uncertainty in $A$, and carry **inclusion** of the target as their validity statement.
- Terminology discipline: "coverage" means the width $\Wunk$ (the unseen-class slot); interval validity is called **inclusion**.

## Pinning (Common-Scale Attribution)

The three widths range over different posteriors, so raw maxima are incomparable. **Pinning** evaluates them at shared pinned configurations (e.g., fixed anchor weights for $\Wlab$; ball-extremal $w$ for $\Wcls$/$\Wunk$), putting acquisition attribution on a common scale. This is the theoretical basis for the `matched` experiment's log-gain vs label-gain asymmetry.

## Conditionals and the Reference Radius

The `conditionals` experiment establishes:
- With the reference radius: **nominal weight coverage** holds.
- Without it: coverage is lost (the radius is not optional).
- Under family tilt: **residual above radius** detects a wrong declared mixture family — the diagnostic that the ball, not the budget, is the binding constraint.

Related: [[01_Core_Claim_and_Widths]] · [[02_Decision_Gates]] (ball-resolution gate) · [[01_Positioning]] (vs label-shift confidence regions).
