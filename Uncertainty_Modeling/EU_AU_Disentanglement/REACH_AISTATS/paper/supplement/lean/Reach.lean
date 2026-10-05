/-
REACH: machine-checked cores of the §3 statements.

What is formalised (deterministic or measure-theoretic cores):
  * `ball_contains_truth`      Lemma 1, deterministic step: concentration event ⇒ w* ∈ ball.
  * `diam_bound`               Prop. B′, ball diameter: σ‖w − w'‖ ≤ 2ρ on the tangent space.
  * `ratio_lipschitz`          Prop. B′/C(i), ratio perturbation: |u/v − u'/v'| ≤ (|u−u'|+|v−v'|)/v.
  * `two_point`, `width_ge`    Prop. B, label-information limit (Le Cam two-point).
  * `novel_not_identified`     Prop. C(ii), q in the known hull ⇒ a segment of weights with the
                               same marginal and different novel weight.

What is NOT formalised: the concentration inequality itself (McDiarmid / bounded-kernel MMD,
Gretton et al. 2012), which Mathlib does not provide; it enters only as a hypothesis.
-/
import Reach.Basic
