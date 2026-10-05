"""Pre-registration v2 for the blind-spot test. Written 2026-10-02 23:09 CEST, AFTER the v1 run
(results/blindspot, 19:59) and after a five-seat review of it, and BEFORE any v2 code was written.

WHAT WAS ALREADY SEEN (so these cells are not blind):
  * v1: "always add labels" had lower regret than the v1 REACH rule (0.0096 vs 0.0318 nats/cell)
    at price 1:50; the accuracy-tuned MI map was worse (0.1165); the H2 CI covers 0 under a
    seed x shift-family cluster bootstrap.
  * v1 rule defects: (a) the predicted log gain was NaN when the hypothetical larger-n ball (same
    histogram, smaller radius) was empty, and NaN fell through to "labels"; (b) the predicted
    label gain priced 25 more CLEAN labels per source, while the action bought 100 SHIFTED labels.
  * In v1 rows, labels pay on CIFAR-10-C even when the incompatibility flag does not fire
    (+0.085 nats), and slightly hurt on clean CIFAR-10 / CIFAR-10.1 (-0.006).
  * The v1 file blindspot_prereg.py states "written 2026-10-03"; it was written 2026-10-02 19:32.

V2 RULE (fixed here; no tuning on outcomes):
  off_frame  if the declared mixture is incompatible (best-fit residual > ball radius) or the cell
             is the outlier cell                                              -> add_labels
  otherwise compare predicted changes of the joint width W_joint(c), each re-solved holding the
  current estimates fixed:
    g_log  = W_now - W(ball centred at the best-fit mixture  w_hat^T A_hat, radius eps_{n0+B} + r)
             (centring at the fitted mixture fixes v1 defect (a): this ball is never empty)
    g_lab  = W_now - W(rows radius and entry half-widths of the estimator the action actually
             produces: (M0 A_hat + counts_k)/(M0 + n_k), n_k = B/(c K) shifted labels per source,
             bounded by (M0 r + n_k r(n_k))/(M0 + n_k) under frozen conditionals)  (fixes (b))
    if max(g_log, g_lab) <= 0 -> do_nothing ; elif g_log >= g_lab -> reweight ; else -> add_labels

ACTIONS AND PRICES. Budget B = 5000 units; reweight grows the log by B images; add_labels buys B/c
shifted labels. Prices c in {10, 50, 250} (500, 100, 20 labels); M0 = 50 primary, {10, 200}
sensitivity at c = 50. do_nothing has payoff 0. Payoff and pools exactly as in v1
(blindspot_prereg.py), same splits, same seeds (calibration 0-9, evaluation 10-19).

POLICIES COMPARED (all scored by regret = payoff(best of the three actions) - payoff(chosen)):
  REACH-v2; constants (do_nothing, reweight, add_labels); flag-only (flag -> labels, else reweight);
  flag-or-nothing (flag -> labels, else do_nothing); MI map over {nothing, reweight, labels} per MI
  tercile, tuned on calibration-seed REGRET (27 maps); MI+flag (flag -> labels, else the regret-tuned
  MI map); oracle.

INFERENCE. Two-way cluster bootstrap over seeds x shift families (clean, CIFAR-10.1, each of the 15
corruptions), 2000 replicates, 95% percentile CIs on regret differences.

CLAIMS:
  H2a (primary, per price): REACH-v2 regret minus the best constant policy's regret. We register NO
      direction at c = 50 (v1 already showed add_labels ahead there). At c = 250 we predict REACH-v2
      <= best constant (labels expensive); at c = 10 no prediction.
  H2b: REACH-v2 minus flag-only, and minus MI+flag. Prediction: per-cell width information adds little
      beyond the flag ('~0'); a negative CI would contradict this in REACH's favour.
  H1u: on unflagged cells, the realised advantage of labels over reweighting where REACH-v2 recommends
      labels minus where it does not, signed with CI. No direction registered.
  H4 (frontier): the regret of each policy as a function of price, reported in full.

IF REACH-v2 LOSES TO A CONSTANT OR TO THE FLAG: report it as the result. The paper then claims that
the widths certify coverage and attribute uncertainty, and that the acquisition signal on these
shifts comes from the incompatibility test, not the width comparison.
"""
PRICES = (10, 50, 250)
M0_PRIMARY = 50
M0_SENSITIVITY = (10, 200)
BUDGET = 5000
N0 = 1000
M_CLEAN = 2000
POOLS = dict(log=.5, label=.2, eval=.3)
CALIBRATION_SEEDS = range(0, 10)
EVALUATION_SEEDS = range(10, 20)
CLUSTER_BOOTSTRAP = 2000
