"""Pre-registration: Gaussian-feature world and per-component dose-responses.
Written 2026-10-04 (after bridge v2 results, results/bridge2/README.md) BEFORE any code for this test.

ALREADY SEEN: bridge v1 (raw widths: D1 sigma and D2 separation failed on real data, pooled across
different K; D3, D4 held) and bridge v2 (REACH-pin passes on real C10 and classifier worlds).

PART G - GAUSSIAN-FEATURE WORLD (realism: real head, controlled geometry)
  Features in R^64; known classes k = 0..3 and one unseen class 4, x ~ N(mu_c, I). mu_k = Delta u_k with
  u_k random unit vectors; unseen mean mu_4 = lam_g Delta u_new + (1 - lam_g)(mu_0 + mu_1)/2.
  Delta in {1.5, 2.5, 4}, lam_g in {0, 0.5, 1}, 3 world draws. Pools: 5000 train images per class
  (dev/ref split exactly as CIFAR), 2000 test per class. The frozen head is real/head.py (PCA-32,
  equal-prior LDA, outlier cell); the bridge v2 code is reused unchanged (REACH-pin, oracle removal).
  Held-out arm: delta = 0.1, Dirichlet(1) prevalences, budgets of bridge v1, cal seeds 0-9, eval 30-49.
  Shift arm: known classes only; deployment features shifted by tau v (v random unit vector), tau in
  {0, 0.25, 0.5, 1, 2}; n in {2000, 32000} x m in {125, 2000}; seeds 0-19. rho = row drift / radius.
  G1: REACH-pin oracle regret < budget-majority (held-out arm), CI excluding 0 (two-way seed x world).
  G2: inclusion of the total interval >= 0.95 in the held-out arm and in the shift arm at tau = 0.
  G3: shift arm: Spearman(rho, inclusion) < 0 and Spearman(rho, incompatibility) > 0 over units;
      inclusion >= 0.95 on units with rho < 1.
  G4: held-out arm at n = 32000, m = 2000: Spearman(Delta, mean W_cls) < 0 and
      Spearman(s_true, mean W_unk) < 0 over the 27 worlds, seed-bootstrap CIs excluding 0.

PART D - PER-COMPONENT DOSE-RESPONSE (classifier worlds of real/bridge2_amendment.py, K = 4, per = 1,
  so K and d are FIXED within every test).
  D-cls: acc in {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}, lam = 1, 4 draws; n in {500, 2000, 8000, 32000,
     128000}; KNOWN rows (r = 0, the setting of Prop. B'), 10 seeds; probes = the K known cells.
     Dcls-1 (n-rate): over probes non-saturated (W_cls < 0.95 at every n), OLS slope of log mean W_cls on
       log n lies in [-0.75, -0.25] (theory: -1/2), seed-bootstrap CI inside [-0.9, -0.1].
     Dcls-2 (distinguishability): at n = 8000, regression of log mean W_cls on log sigma(A) across the
       28 worlds has a negative slope, CI excluding 0.
     Dcls-3 (probe mass): within each world, Spearman over known probes of (probe known mass under
       w*, W_cls) at n = 8000 is negative on >= 70% of worlds.
  D-label: acc in {0.6, 0.8, 0.95}, lam = 1, 4 draws; noisy references with symmetric flip rate eta in
     {0, 0.05, 0.1, 0.2, 0.3}, debiased as in real/audit.py (_reference), m in {125, 500, 2000}, n = 8000;
     10 seeds. Dlab-1: at every (world, m), mean W_label is non-decreasing in eta (share of (world, m)
     with Spearman(eta, W_label) = 1 is >= 0.8). Dlab-2: the anchored true posterior lies in the label
     interval on >= 95% of (probe, seed) at every eta. Dlab-3: pooled within (world, m), Spearman
     (W_label, |anchored error|) > 0. Dlab-4: W_label at a fixed seed is identical across n in
     {2000, 32000} (exact; the n-invariance of Prop. B, a construction check).
  D-unk: acc = 0.9, lam in {0, 0.1, 0.2, 0.4, 0.7, 1}, 4 draws; KNOWN rows and q; n in {500, 2000, 8000,
     32000, 128000}; 10 seeds; probes = cells with q(x) > 0.
     Dunk-1: at n = 128000, Spearman over the 24 worlds of (s_true, mean W_unk) < 0, CI excluding 0.
     Dunk-2: n-slope of log mean W_unk: worlds with lam = 0 (s = 0): slope CI includes 0 or is > -0.1
       (no contraction, Prop. C(ii)); worlds with lam = 1: slope < -0.25 with CI excluding -0.1.
FAILURES are reported as failures.
"""
G_DELTA = (1.5, 2.5, 4.)
G_LAM = (0., .5, 1.)
G_DRAWS = 3
G_DIM = 64
G_TRAIN = 5000
G_TEST = 2000
G_TAU = (0., .25, .5, 1., 2.)
G_CAL = tuple(range(0, 10))
G_EVAL = tuple(range(30, 50))
G_SHIFT_SEEDS = tuple(range(0, 20))
D_CLS_ACC = (.5, .6, .7, .8, .9, .95, .99)
D_N = (500, 2000, 8000, 32000, 128000)
D_LAB_ACC = (.6, .8, .95)
D_ETA = (0., .05, .1, .2, .3)
D_LAB_M = (125, 500, 2000)
D_UNK_LAM = (0., .1, .2, .4, .7, 1.)
D_UNK_ACC = .9
D_DRAWS = 4
D_SEEDS = tuple(range(10))
D_K = 4
