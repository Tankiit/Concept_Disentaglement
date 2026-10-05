import Mathlib

open Finset BigOperators MeasureTheory

namespace Reach

/-! ## Setting

`S` indexes sources (known sources plus the candidate novel source), `C` indexes cells.
A row `A s : C → ℝ` is the cell distribution of source `s`; a weight vector `w : S → ℝ`
mixes them into the marginal `∑ s, w s • A s`. -/

variable {S : Type*} [Fintype S]

/-- The marginal over cells induced by weights `w` and rows `A`. -/
def marginal {E : Type*} [AddCommGroup E] [Module ℝ E] (w : S → ℝ) (A : S → E) : E :=
  ∑ s, w s • A s

/-- Weights in the probability simplex. -/
def InSimplex (w : S → ℝ) : Prop := (∀ s, 0 ≤ w s) ∧ ∑ s, w s = 1

/-! ## Lemma 1, deterministic step

If the empirical marginal is within `ε` of the true marginal and every estimated row is within
`r` of its true row, then the true weights satisfy the ball constraint with radius `ε + r`.
The probability of the hypotheses is the concentration statement, which is cited, not proved. -/

theorem ball_contains_truth {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (A Ahat : S → E) (w : S → ℝ) (hw : InSimplex w) (phat : E) (ε r : ℝ)
    (hp : ‖marginal w A - phat‖ ≤ ε) (hA : ∀ s, ‖Ahat s - A s‖ ≤ r) :
    ‖marginal w Ahat - phat‖ ≤ ε + r := by
  have hsplit : marginal w Ahat - phat = (marginal w Ahat - marginal w A) + (marginal w A - phat) := by
    abel
  have hdiff : marginal w Ahat - marginal w A = ∑ s, w s • (Ahat s - A s) := by
    simp [marginal, smul_sub, Finset.sum_sub_distrib]
  have hrow : ‖∑ s, w s • (Ahat s - A s)‖ ≤ r := by
    calc ‖∑ s, w s • (Ahat s - A s)‖ ≤ ∑ s, ‖w s • (Ahat s - A s)‖ := norm_sum_le _ _
      _ = ∑ s, w s * ‖Ahat s - A s‖ := by
          refine Finset.sum_congr rfl fun s _ => ?_
          rw [norm_smul, Real.norm_of_nonneg (hw.1 s)]
      _ ≤ ∑ s, w s * r := by
          refine Finset.sum_le_sum fun s _ => ?_
          exact mul_le_mul_of_nonneg_left (hA s) (hw.1 s)
      _ = r := by rw [← Finset.sum_mul, hw.2, one_mul]
  calc ‖marginal w Ahat - phat‖
      = ‖(marginal w Ahat - marginal w A) + (marginal w A - phat)‖ := by rw [← hsplit]
    _ ≤ ‖marginal w Ahat - marginal w A‖ + ‖marginal w A - phat‖ := norm_add_le _ _
    _ ≤ r + ε := by rw [hdiff]; exact add_le_add hrow hp
    _ = ε + r := add_comm _ _

/-! ## Proposition B′: diameter of the ball

Two weight vectors in the ball differ by a tangent vector (it sums to zero). If the embedding map
is bounded below by `σ` on tangent vectors, their distance is at most `2ρ / σ`. -/

theorem diam_bound {V E : Type*} [NormedAddCommGroup V] [NormedAddCommGroup E]
    [NormedSpace ℝ V] [NormedSpace ℝ E] (L : V →ₗ[ℝ] E) (T : Submodule ℝ V) (σ ρ : ℝ)
    (hσ : ∀ v ∈ T, σ * ‖v‖ ≤ ‖L v‖) (phat : E) (w w' : V)
    (hw : ‖L w - phat‖ ≤ ρ) (hw' : ‖L w' - phat‖ ≤ ρ) (hT : w - w' ∈ T) :
    σ * ‖w - w'‖ ≤ 2 * ρ := by
  calc σ * ‖w - w'‖ ≤ ‖L (w - w')‖ := hσ _ hT
    _ = ‖(L w - phat) - (L w' - phat)‖ := by rw [map_sub]; congr 1; abel
    _ ≤ ‖L w - phat‖ + ‖L w' - phat‖ := norm_sub_le _ _
    _ ≤ ρ + ρ := add_le_add hw hw'
    _ = 2 * ρ := by ring

/-- Differences of simplex weights are tangent: they sum to zero. -/
theorem simplex_sub_sum_zero (w w' : S → ℝ) (hw : InSimplex w) (hw' : InSimplex w') :
    ∑ s, (w - w') s = 0 := by
  simp [Finset.sum_sub_distrib, hw.2, hw'.2]

/-! ## Proposition B′ / C(i): ratio perturbation

A posterior at a probe is a ratio `u / v` of linear functionals of the weights, with `0 ≤ u ≤ v`.
Its change is controlled by the changes of numerator and denominator over the smaller-indexed
denominator. Combined with `diam_bound` this gives the width bound of Prop. B′. -/

theorem ratio_lipschitz (u v u' v' : ℝ) (hv : 0 < v) (hv' : 0 < v') (hu' : 0 ≤ u')
    (huv' : u' ≤ v') :
    |u / v - u' / v'| ≤ (|u - u'| + |v - v'|) / v := by
  have key : u / v - u' / v' = (u - u') / v + (u' / v') * ((v' - v) / v) := by
    field_simp
    ring
  have hfrac : 0 ≤ u' / v' := div_nonneg hu' hv'.le
  have hle1 : u' / v' ≤ 1 := (div_le_one hv').mpr huv'
  rw [key]
  calc |(u - u') / v + (u' / v') * ((v' - v) / v)|
      ≤ |(u - u') / v| + |(u' / v') * ((v' - v) / v)| := abs_add_le _ _
    _ = |u - u'| / v + (u' / v') * (|v - v'| / v) := by
        rw [abs_div, abs_of_pos hv, abs_mul, abs_of_nonneg hfrac, abs_div, abs_of_pos hv,
          abs_sub_comm v' v]
    _ ≤ |u - u'| / v + 1 * (|v - v'| / v) := by
        gcongr
    _ = (|u - u'| + |v - v'|) / v := by ring

/-- Prop. B′ at the level of the posterior. At a probe with source masses `a s ∈ [0, amax]`,
the posterior of source `y`, `f(w) = w y * a y / ∑ s, w s * a s`, moves by at most
`2 amax ‖w - w'‖₁ / v(w)`, where `v(w)` is the probe mass under `w`. With `diam_bound` (and
`‖·‖₁ ≤ √K ‖·‖₂`) this is the width bound `W_cls ≤ L(x) · 2(ε_n + r) / σ` of the paper. -/
theorem posterior_lipschitz (a w w' : S → ℝ) (y : S) (amax : ℝ)
    (ha0 : ∀ s, 0 ≤ a s) (ha : ∀ s, a s ≤ amax) (hw' : ∀ s, 0 ≤ w' s)
    (hv : 0 < ∑ s, w s * a s) (hv' : 0 < ∑ s, w' s * a s) :
    |w y * a y / ∑ s, w s * a s - w' y * a y / ∑ s, w' s * a s|
      ≤ 2 * amax * (∑ s, |w s - w' s|) / ∑ s, w s * a s := by
  set D := ∑ s, |w s - w' s|
  have hamax : 0 ≤ amax := le_trans (ha0 y) (ha y)
  have hnum : |w y * a y - w' y * a y| ≤ amax * D := by
    rw [← sub_mul, abs_mul, abs_of_nonneg (ha0 y)]
    calc |w y - w' y| * a y ≤ |w y - w' y| * amax := mul_le_mul_of_nonneg_left (ha y) (abs_nonneg _)
      _ = amax * |w y - w' y| := mul_comm _ _
      _ ≤ amax * D := mul_le_mul_of_nonneg_left
          (Finset.single_le_sum (f := fun s => |w s - w' s|) (fun s _ => abs_nonneg _)
            (Finset.mem_univ y)) hamax
  have hden : |∑ s, w s * a s - ∑ s, w' s * a s| ≤ amax * D := by
    rw [← Finset.sum_sub_distrib]
    calc |∑ s, (w s * a s - w' s * a s)| ≤ ∑ s, |w s * a s - w' s * a s| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ s, amax * |w s - w' s| := by
          refine Finset.sum_le_sum fun s _ => ?_
          rw [← sub_mul, abs_mul, abs_of_nonneg (ha0 s), mul_comm]
          exact mul_le_mul_of_nonneg_right (ha s) (abs_nonneg _)
      _ = amax * D := by rw [Finset.mul_sum]
  have hu' : 0 ≤ w' y * a y := mul_nonneg (hw' y) (ha0 y)
  have huv' : w' y * a y ≤ ∑ s, w' s * a s :=
    Finset.single_le_sum (f := fun s => w' s * a s) (fun s _ => mul_nonneg (hw' s) (ha0 s))
      (Finset.mem_univ y)
  calc _ ≤ (|w y * a y - w' y * a y| + |∑ s, w s * a s - ∑ s, w' s * a s|) / ∑ s, w s * a s :=
        ratio_lipschitz _ _ _ _ hv hv' hu' huv'
    _ ≤ (amax * D + amax * D) / ∑ s, w s * a s := by gcongr
    _ = 2 * amax * D / ∑ s, w s * a s := by ring

/-! ## Proposition B: label-information limit (two-point argument)

If two label laws `θ, θ'` induce the same law of the unlabelled data, then any set-valued
estimator that covers the truth with probability `≥ 1 - α` under both, with `α < 1/2`, must
contain both values for some data set. No measurability of the estimator is needed: outer
measure is subadditive. -/

theorem two_point {Ω Θ : Type*} [MeasurableSpace Ω] (P : Θ → Measure Ω)
    [hP : ∀ θ, IsProbabilityMeasure (P θ)] (θ θ' : Θ) (hsame : P θ = P θ')
    (CI : Ω → Set Θ) (α : ENNReal) (hα : 2 * α < 1)
    (hcov : ∀ t, P t {ω | t ∉ CI ω} ≤ α) :
    ∃ ω, θ ∈ CI ω ∧ θ' ∈ CI ω := by
  by_contra h
  push_neg at h
  have hunion : (Set.univ : Set Ω) ⊆ {ω | θ ∉ CI ω} ∪ {ω | θ' ∉ CI ω} := by
    intro ω _
    by_cases hθ : θ ∈ CI ω
    · exact Or.inr (h ω hθ)
    · exact Or.inl hθ
  have hle : P θ Set.univ ≤ 2 * α :=
    calc P θ Set.univ ≤ P θ ({ω | θ ∉ CI ω} ∪ {ω | θ' ∉ CI ω}) := measure_mono hunion
      _ ≤ P θ {ω | θ ∉ CI ω} + P θ {ω | θ' ∉ CI ω} := measure_union_le _ _
      _ = P θ {ω | θ ∉ CI ω} + P θ' {ω | θ' ∉ CI ω} := by rw [hsame]
      _ ≤ α + α := add_le_add (hcov θ) (hcov θ')
      _ = 2 * α := by rw [two_mul]
  rw [measure_univ] at hle
  exact absurd (lt_of_le_of_lt hle hα) (lt_irrefl 1)

/-- Consequence for real parameters: some data set gives an interval of width at least
`|θ - θ'|`, so the width cannot shrink below it from the unlabelled data alone, at any `n`. -/
theorem width_ge {Ω : Type*} [MeasurableSpace Ω] (P : ℝ → Measure Ω)
    [∀ θ, IsProbabilityMeasure (P θ)] (θ θ' : ℝ) (hsame : P θ = P θ')
    (CI : Ω → Set ℝ) (hbdd : ∀ ω, Bornology.IsBounded (CI ω)) (α : ENNReal) (hα : 2 * α < 1)
    (hcov : ∀ t, P t {ω | t ∉ CI ω} ≤ α) :
    ∃ ω, |θ - θ'| ≤ Metric.diam (CI ω) := by
  obtain ⟨ω, h1, h2⟩ := two_point P θ θ' hsame CI α hα hcov
  exact ⟨ω, by rw [← Real.dist_eq]; exact Metric.dist_le_diam_of_mem (hbdd ω) h1 h2⟩

/-! ## Proposition C(ii): novelty is not identified when `q` lies in the known hull

If the novel row is a convex combination `c` of the other rows, moving weight along
`e_nov - c` leaves the marginal unchanged while changing the novel weight. For small steps the
weights stay in the simplex, so the identified set for the novel weight has positive length. -/

/-- Weights moved by `t` along `e_nov - c`. -/
def shift [DecidableEq S] (w c : S → ℝ) (nov : S) (t : ℝ) : S → ℝ :=
  fun s => w s + t * ((if s = nov then 1 else 0) - c s)

theorem novel_not_identified [DecidableEq S] {E : Type*} [AddCommGroup E] [Module ℝ E]
    (A : S → E) (nov : S) (c : S → ℝ) (hc_nov : c nov = 0)
    (hq : A nov = ∑ s, c s • A s) (w : S → ℝ) (t : ℝ) :
    marginal (shift w c nov t) A = marginal w A ∧ shift w c nov t nov = w nov + t := by
  constructor
  · have hdir : ∑ s, ((if s = nov then (1 : ℝ) else 0) - c s) • A s = 0 := by
      simp only [sub_smul, Finset.sum_sub_distrib, ite_smul, one_smul, zero_smul,
        Finset.sum_ite_eq', Finset.mem_univ, if_true]
      rw [← hq, sub_self]
    have hsplit : marginal (shift w c nov t) A
        = marginal w A + t • ∑ s, ((if s = nov then (1 : ℝ) else 0) - c s) • A s := by
      simp only [marginal, shift, add_smul, mul_smul, Finset.sum_add_distrib, Finset.smul_sum]
    rw [hsplit, hdir, smul_zero, add_zero]
  · simp [shift, hc_nov]

/-- The moved weights stay in the simplex for `0 ≤ t` small enough. -/
theorem novel_segment_in_simplex [DecidableEq S] (nov : S) (c : S → ℝ) (hc : InSimplex c)
    (hc_nov : c nov = 0) (w : S → ℝ) (hw : InSimplex w) (t : ℝ) (ht0 : 0 ≤ t)
    (ht : ∀ s, t * c s ≤ w s) :
    InSimplex (shift w c nov t) := by
  refine ⟨fun s => ?_, ?_⟩
  · simp only [shift]
    by_cases hs : s = nov
    · subst hs
      simp only [if_true, hc_nov, sub_zero, mul_one]
      linarith [hw.1 s]
    · simp only [hs, if_false, zero_sub, mul_neg]
      linarith [ht s]
  · simp only [shift, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_sub_distrib,
      Finset.sum_ite_eq', Finset.mem_univ, if_true, hw.2, hc.2, sub_self, mul_zero, add_zero]

end Reach
