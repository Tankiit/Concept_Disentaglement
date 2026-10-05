"""Pre-registration: stronger attribution test with an oracle-free pinned-estimate rule (bridge v2).
Written 2026-10-04, AFTER the bridge v1 results (results/bridge/README.md) and BEFORE any code for v2.

ALREADY SEEN: all bridge v1 results, seeds 0-29, every real setting listed below. v1 showed the raw
argmax of (W_cls, W_label, W_unk) loses to the best input-independent predictor (budget-majority);
a logistic map of the widths fitted on calibration seeds beat it slightly (exploratory). Labels (rows)
dominate 80-99% of real inputs, so the real test has little headroom; hence the synthetic arm below.

NEW RULE (REACH-pin; fixed here, no fitting). At probe x with total tight joint interval width W(x)
(real/bridge.py), re-solve with one source pinned at its CURRENT ESTIMATE:
  log    : weights w0 = argmin_w dist(p_hat, box(w)) over the simplex; histogram <- x0, the point of
           box(w0) nearest p_hat; eps <- 0.
  rows   : every row entry interval <- A_hat (unseen row: q_hat); eps <- max(eps_n, residual of the
           best Euclidean fit of p_hat by mixtures of A_hat) so the set stays non-empty.
  unseen : w_unseen <- w0_unseen.
Dhat_s = W - W_pin_s; REACH-pin chooses argmax_s Dhat_s (sources available in the setting).

OUTCOMES. (O) oracle removal exactly as in v1 (bridge_prereg.py), resolvable threshold 0.01.
(F) finite acquisition, real label-shift CIFAR-10 settings, m0 in {125, 500}: realised reductions of W
from n -> 4n (log) and m -> 4m (labels); choice restricted to {log, rows}.
Score: regret = best realised reduction - reduction of the chosen source, averaged over inputs.

COMPARATORS: budget-majority (most frequent oracle/finite-best label per (setting, n, m) on
calibration seeds; for synthetic, per (world config, n, m) pooled over world draws), REACH (raw
argmax, v1), REACH-cal (v1 logistic map, refit on calibration seeds), radius bookkeeping.

SETTINGS.
  Real: the v1 parts L_c10_*, H_c10_* (CLIP, DINOv2, ResNet-50), L_c100_*, H_c100_* (CLIP, DINOv2).
        Calibration seeds 0-9, EVALUATION seeds 30-59 (never run before). Budgets as v1.
        One shared log draw of 128000 per seed (prefix-paired), so v1 logs are not reproduced.
  Synthetic categorical worlds: K in {3, 6, 10} known sources, d in {10, 50} cells, rows ~
        Dirichlet(beta 1_d) with beta in {0.2, 1, 5} (distinguishability), unseen row q = lam * fresh
        Dirichlet row + (1 - lam) * random mixture of the known rows, lam in {0, 0.3, 1} (separation),
        delta = 0.1, prevalences Dirichlet(1) per seed; 4 world draws per config; 10 probe cells per
        world (seeded); n in {300, 3000, 30000}, m in {30, 300, 3000}; q estimated from 2000 draws.
        Calibration seeds 0-4, evaluation seeds 5-9. Truth rows exact.

CLAIMS (registered):
  P1 real: REACH-pin regret (O) < budget-majority, CI excluding 0, pooled over CIFAR-10 H and L settings
     (two-way bootstrap seed x setting). Reported per encoder, per arm, and for CIFAR-100.
  P2 synthetic: REACH-pin regret (O) < budget-majority, CI excluding 0 (two-way bootstrap seed x world).
  P3 synthetic: REACH-pin accuracy on resolvable inputs >= 0.70.
  P4 real finite acquisition (F): REACH-pin regret < budget-majority, CI excluding 0.
  P5 (no direction): REACH-pin vs REACH-cal, real and synthetic.
  P6 inclusion of the total interval >= 0.95 in every arm (real and synthetic).
FAILURES are reported as failures.
"""
REAL_PARTS = ('L_c10_clip', 'L_c10_dinov2', 'L_c10_resnet50', 'H_c10_clip', 'H_c10_dinov2', 'H_c10_resnet50',
              'L_c100_clip', 'L_c100_dinov2', 'H_c100_clip', 'H_c100_dinov2')
REAL_CAL_SEEDS = tuple(range(0, 10))
REAL_EVAL_SEEDS = tuple(range(30, 60))
LOG_MAX = 128000
FINITE_M = (125, 500)
FINITE_FACTOR = 4
SYN_K = (3, 6, 10)
SYN_D = (10, 50)
SYN_BETA = (.2, 1., 5.)
SYN_LAM = (0., .3, 1.)
SYN_DRAWS = 4
SYN_PROBES = 10
SYN_N = (300, 3000, 30000)
SYN_M = (30, 300, 3000)
SYN_Q_DRAWS = 2000
SYN_CAL_SEEDS = tuple(range(0, 5))
SYN_EVAL_SEEDS = tuple(range(5, 10))
DELTA = .1
RESOLVABLE = .01
