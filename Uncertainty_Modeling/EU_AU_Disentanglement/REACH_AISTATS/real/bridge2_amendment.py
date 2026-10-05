"""Amendment 1 to real/bridge2_prereg.py (SHA d8e97dab), written 2026-10-04 ~15:40 CEST, BEFORE any
claim was scored and before any full run. Only the synthetic world design and budgets change; claims,
rule, comparators, real settings and seeds are unchanged.

WHAT WAS SEEN: smoke runs only (24 episodes per part) and a width probe on single worlds (seed 5,
which is an evaluation seed; no oracle-vs-predictor comparison was computed). The registered dense
worlds (rows ~ Dirichlet(beta 1_d), d in {10, 50}) give a total interval of width ~1.00 at every
budget up to n = 1e5, m = 3e4: per-entry radii exceed most entries (zero-mass [0, 1] policy) and
dense rows overlap. Every predictor would then be scored on almost no resolvable inputs.

REPLACEMENT ("classifier worlds", mimicking a frozen head's confusion rows):
  K known sources, `per` cells per source plus one outlier cell (d = K*per + 1). Row k puts mass acc
  on its own cells (split by Dirichlet(1)) and 1 - acc on Dirichlet(1) noise over all cells (outlier
  entry down-weighted x0.2). Unseen row q = lam * (0.5 e_outlier + 0.5 Dirichlet(1)) + (1 - lam) *
  random mixture of the known rows.
  (K, per) in {(3,1), (3,3), (6,1), (6,2), (10,1)}; acc in {0.6, 0.8, 0.95}; lam in {0, 0.3, 1};
  4 draws per config; probes = all cells (at most 10, seeded subset otherwise);
  n in {500, 5000, 50000}; m in {50, 500, 5000, 50000} (the largest m gives the log a chance to
  dominate; probe runs showed rows dominate otherwise). Seeds unchanged (cal 0-4, eval 5-9).
"""
SYN_KP = ((3, 1), (3, 3), (6, 1), (6, 2), (10, 1))
SYN_ACC = (.6, .8, .95)
SYN_LAM = (0., .3, 1.)
SYN_DRAWS = 4
SYN_PROBES = 10
SYN_N = (500, 5000, 50000)
SYN_M = (50, 500, 5000, 50000)
