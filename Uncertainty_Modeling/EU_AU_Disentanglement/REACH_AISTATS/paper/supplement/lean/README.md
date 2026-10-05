# Machine-checked cores of the §3 statements (Lean 4 + Mathlib)

```bash
cd lean
lake exe cache get   # prebuilt Mathlib, pinned in lake-manifest.json
lake build           # checks Reach/Basic.lean
lake env lean Axioms.lean   # every theorem: [propext, Classical.choice, Quot.sound] only, no sorry
```

| Lean theorem | Paper | What it states |
| --- | --- | --- |
| `ball_contains_truth` | Lemma 1 | On the concentration event (‖p̂−p*‖ ≤ ε, ‖Â_k−A_k‖ ≤ r), the true weights satisfy the ball constraint with radius ε + r. |
| `diam_bound` | Prop. B′ | If ‖Lv‖ ≥ σ‖v‖ on tangent vectors, two weights in the ball are within 2ρ/σ. |
| `simplex_sub_sum_zero` | Prop. B′ | Differences of simplex weights are tangent. |
| `ratio_lipschitz` | Prop. B′, C(i) | \|u/v − u′/v′\| ≤ (\|u−u′\| + \|v−v′\|)/v for 0 ≤ u′ ≤ v′. |
| `posterior_lipschitz` | Prop. B′ | Posterior at a probe moves by ≤ 2·a_max·‖w−w′‖₁ / v(w). |
| `two_point`, `width_ge` | Prop. B | Same log law under θ, θ′ ⇒ some data set's interval contains both ⇒ width ≥ \|θ−θ′\|. |
| `novel_not_identified`, `novel_segment_in_simplex` | Prop. C(ii) | q in the known hull ⇒ a segment of simplex weights with the same marginal and different novel weight. |

**Not formalised:** the concentration inequality (bounded-kernel MMD / McDiarmid; Gretton et al.
2012, Thm 7), which Mathlib does not provide. It enters Lemma 1 only as the hypothesis `hp`.
Also not formalised: the √K conversion between ℓ1 and ℓ2 used to state Prop. B′ with σ in ℓ2,
and Prop. C(iii), which is a policy (vacuous interval off support), not a theorem.
