"""Pre-registration: exchange-rate and certification-acquisition test (v3).
Written 2026-10-03 ~06:52 CEST, after two review rounds, BEFORE any code for this test and BEFORE any
ResNet-50 feature exists. Hash recorded in every manifest of this test.

WHY. Review round 2 found (DA-N1, EIC N4, R2 M1): the synthetic attribution is mostly by construction
(rows known), and on real data every width responds to every acquisition. The theory (Prop. B') says
something sharper that has not been tested: the prevalence width depends on the two acquisitions only
through the radius eps_n + r_m, so each acquisition's effect is proportional to the share of the radius
it controls, and the better acquisition at matched cost changes with the starting budgets and the price.
That is a quantitative, falsifiable claim on data, under the paper's own assumption (frozen
class-conditionals), and it implies a decision on which constant policies must lose.

SETTINGS (primary = blind):
  P1 real, BLIND: CIFAR-10, frozen ResNet-50 (torchvision ImageNet weights, never inspected), PCA-32 +
     equal-prior LDA head, four superclass sources, pure label shift with Dirichlet(1) prevalences.
  P2 synthetic, BLIND: world 'separated' with the two known rows ESTIMATED from m references (r > 0),
     the unseen row declared.
  S1 real, SEEN (replication, not confirmatory): same as P1 with CLIP ViT-B/32 features.
Base points (n0, m0) in {500, 2000, 8000} x {125, 500, 2000} (synthetic: m0 in {32, 128, 512});
prices c in {10, 50, 250}; budget B = 5000 units: more log = +B images; labels = +B/(c K) clean
labelled references per source. 30 seeds per base point, Dirichlet prevalences per seed (real).

OUTCOME. Realised reduction of the mean certified width over known-class cells, i.e. the width at the
starting budgets minus the width after actually acquiring (prefix-extended streams). Certified width =
REACH tight joint interval (below). Secondary: plug-in W_cls, REACH joint (v2 definition), constraint
region after Li et al.

TIGHT JOINT INTERVAL (new, valid on the same event): weights w in the simplex such that the Euclidean
distance from the log histogram to the box {x : sum_j w_j lo_j(c) <= x_c <= sum_j w_j hi_j(c)} is at most
eps_n, where [lo, hi] are the per-entry row intervals; this replaces "ball of radius eps_n + r around
w^T A_hat AND entry ranges" (which counts reference error twice). It is a second-order cone constraint.
Prediction T1: tight width <= REACH joint width on >= 90% of probes. T2: no direction vs the Li region.

DECISION RULES compared (scored by regret = best realised reduction - chosen one's, per episode):
  REACH: re-solve the tight width at the post-acquisition radii holding current estimates fixed
         (log: ball at the fitted mixture, radius eps_{n0+B}; labels: entry intervals at m0 + B/(cK));
         choose the larger predicted reduction.
  radius-only: data-free bookkeeping; choose log iff eps_{n0} - eps_{n0+B} > r_{m0} - r_{m0+B/(cK)}.
  constants: always log, always labels.  ensemble MI: tercile map tuned on calibration regret (seeds 0-9).
Evaluation seeds 10-29. CIs: bootstrap over seeds (clusters = seed; base points and prices share seeds).

CLAIMS (registered):
  E1 (exchange rate): across base points and prices, the realised log-minus-label reduction has the sign
     predicted by REACH in >= 75% of (base, price) cells, in P1 and P2.
  E2 (decision value): REACH regret < min(always-log, always-labels) regret, CI excluding 0, in P1 and P2.
  E3 (does data add beyond bookkeeping): REACH vs radius-only, no direction registered.
  E4 (tight interval): T1, T2 above; tight interval covers >= 0.95 per budget in P1 and P2.
IF E2 FAILS in P1: we report that the exchange rate does not yield decision value on real features.
"""
BASES_N = (500, 2000, 8000)
BASES_M_REAL = (125, 500, 2000)
BASES_M_SYNTH = (32, 128, 512)
PRICES = (10, 50, 250)
BUDGET = 5000
SEEDS = 30
CALIBRATION_SEEDS = range(0, 10)
EVALUATION_SEEDS = range(10, 30)
