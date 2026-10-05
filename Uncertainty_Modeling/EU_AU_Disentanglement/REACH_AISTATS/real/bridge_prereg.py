"""Pre-registration: real-data bridge for the per-component tests (CIFAR-10 / CIFAR-100, label shift,
held-out class, domain shift). Written 2026-10-03 ~19:40 CEST, BEFORE any code for this test
(real/bridge.py does not exist yet). The SHA-256 of this file is recorded in every manifest.

WHY. After three review rounds the open objection is that the attribution of epistemic width to an
acquisition (log -> prevalence, labels -> rows, deployment labels -> unseen mass) has no evidence
beyond construction, and that per-input widths add nothing over budget-level radius bookkeeping.
This test scores the attribution against an ORACLE on real data, and asks whether the per-input
widths beat the best input-independent predictor at the same budget.

ALREADY SEEN (not blind): audit results for held-out classes (8, 9) at delta 0.1 (CLIP, DINOv2), the
CIFAR-10-C coverage tables (CLIP, super4 and classes), all exchange/blind-spot results. NOT seen: any
oracle-removal quantity, any single-held-out-class run, CIFAR-100 held-out with the groups below,
any row-drift measurement.

TARGET. At a probe cell x, T(x) = P(source = yhat(x) | x) over ALL sources in play (known + unseen),
yhat(x) = argmax_k A_hat_k(x) over known sources. Total certified interval I(x): the tight joint set
(real/exchange.py): w in the simplex, known row entries in per-entry Hoeffding intervals, the unseen
row in [q_hat - q_r, q_hat + q_r], the log histogram within eps_n of the box. Alpha split as in the
audit (log / references / entries).

ORACLE REMOVAL. W = width of I(x). Remove one source at a time, re-solve, Delta_s = W - W_{-s}:
  log      : histogram <- true marginal p* = w*^T A, eps_n <- 0        (infinite log)
  rows     : every row entry interval <- the true row (deployment pool)  (infinite labels)
  unseen   : add the constraint w_unseen = delta*                        (known unseen mass)
  shift    : (domain-shift arm only) rows <- the SHIFTED true rows, eps as is
Oracle label o(x) = argmax_s Delta_s over inputs with max_s Delta_s >= 0.01 ("resolvable"); others
are reported as a share and excluded from accuracy.

PREDICTORS of o(x), all fitted (when fitted at all) on calibration seeds 0-9, scored on 10-29:
  REACH         : argmax(W_cls -> log, W_label -> rows, W_unk -> unseen) at x (published definitions,
                  plug-in widths; no fitting).
  REACH-cal     : multinomial logistic regression of o on (log W_cls, log W_label, log W_unk, log eps_n,
                  log r) [secondary].
  budget-major. : the most frequent oracle label at the same (setting, n, m) on calibration seeds
                  (the best input-independent predictor: "bookkeeping with hindsight").
  radius        : log if eps_n > entry half-width at m else rows (data-free bookkeeping).
  ens-MI        : tercile map of ensemble MI to {log, rows, unseen}, tuned on calibration accuracy.
Score: oracle regret = max_s Delta_s - Delta_{predicted}, averaged over resolvable inputs (width units);
accuracy secondary. CIs: bootstrap over seeds (held-out: two-way, seed x novel class).

SETTINGS. Head as in the audit (PCA-32 + equal-prior LDA + outlier cell). Dirichlet(1) prevalences.
Budgets n in {500, 2000, 8000, 32000} x m in {125, 500, 2000}. 30 seeds.
  L  label shift : CIFAR-10 x {CLIP, DINOv2, ResNet-50} x {super4, classes}; CIFAR-100 c100_2
                   (trees, vehicles_1) x {CLIP, DINOv2}.
  H  held out    : CIFAR-10, grouping classes, novel = each single class (9 known sources) x
                   {CLIP, DINOv2, ResNet-50}; CIFAR-100, known = c100_2, novel = one of
                   vehicles_2 (41,69,81,85,89), outdoor scenes (23,33,49,60,71), flowers
                   (54,62,70,82,92), people (2,11,35,46,98), fish (1,32,67,73,91) x {CLIP, DINOv2}.
                   delta* = 0.1. The unseen row is estimated from the withheld classes' train images.
  S  domain shift: CLIP (the only backbone with CIFAR-10-C features), super4, deployment =
                   CIFAR-10-C 15 corruptions x 5 severities and CIFAR-10.1; references and head clean.
                   n in {2000, 32000} x m in {125, 2000}; 20 seeds (cal 0-5, eval 6-19).

CLAIMS (registered):
  A1 (attribution beats bookkeeping, H): REACH regret < budget-majority regret, CI excluding 0,
     pooled over CIFAR-10 encoders; reported per encoder and for CIFAR-100.
  A2 (same, L): REACH regret < budget-majority regret, CI excluding 0, pooled over CIFAR-10 settings.
  A3 (no direction): REACH-cal vs REACH; ens-MI vs budget-majority.
  D1 (prevalence width and distinguishability, L): across all (setting, n, m) units, regression of
     log mean W_cls on log(eps_n + r_m) and log sigma(A_hat): sigma coefficient < 0, CI excluding 0.
     The Prop. B' bound holds on >= 95% of probes (a theorem on the event; failure = code error).
  D2 (coverage width and separation, H): over novel candidates, Spearman(s, mean W_unk at n = 32000,
     m = 2000) < 0, CI excluding 0 (bootstrap over seeds; permutation p reported), s = hull distance
     of the true unseen row from the true known rows. Spearman(s, n-slope of log W_unk) < 0.
  D3 (label width tracks realised row error, L): the anchored posterior with true rows lies inside
     the W_label interval on >= 95% of (probe, seed, m); within each m, Spearman(W_label, |anchored
     posterior error|) > 0 pooled over settings.
  D4 (domain shift as a measured violation, S): row drift rho = max_k ||A_shift_k - A_k||_2 /
     (eps_n + r_m) per (pool, n, m). Spearman(rho, inclusion rate of I) < 0 and Spearman(rho,
     incompatibility rate) > 0. Descriptive: inclusion for rho < 1 vs rho >= 1; share of resolvable
     inputs where 'shift' is the oracle label, and the incompatibility flag's sensitivity on them.
FAILURES are reported as failures; no claim is re-registered after results are seen.
"""
BUDGETS_N = (500, 2000, 8000, 32000)
BUDGETS_M = (125, 500, 2000)
SHIFT_N = (2000, 32000)
SHIFT_M = (125, 2000)
SEEDS = 30
SHIFT_SEEDS = 20
CAL_SEEDS = range(0, 10)
SHIFT_CAL_SEEDS = range(0, 6)
DELTA = .1
RESOLVABLE = .01
C100_KNOWN = 'c100_2'
C100_NOVEL = dict(vehicles_2=(41, 69, 81, 85, 89), outdoor=(23, 33, 49, 60, 71),
                  flowers=(54, 62, 70, 82, 92), people=(2, 11, 35, 46, 98), fish=(1, 32, 67, 73, 91))
