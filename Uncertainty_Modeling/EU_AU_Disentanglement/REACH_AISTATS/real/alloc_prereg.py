"""Pre-registration: class-wise label allocation by per-source pinning (REACH-alloc).
Written 2026-10-05, after the round-4 review (scratchpad editorial decision round 4) and BEFORE any code
for this test. SHA-256 recorded in every manifest.

WHY. Round 4: REACH-pin estimates its oracle-removal target well (P1), but no outcome outside the interval
machinery shows decision value, and acquisitions are bought per deployment, not per input. Here the
decision is real and per deployment: how to split a budget of new labelled references across the K known
sources.

DECISION. Start: log of n images, m0 labelled references per known source. Budget B = K * m0 new labels.
Allocation b (sum b_k = B, integers, m0 + b_k capped at the available stream; excess redistributed in
proportion to the rule's weights among uncapped sources).
  uniform      b_k = B / K
  proportional b_k proportional to w_hat_k (known part of the best Euclidean fit of p_hat by A_hat)
  REACH-alloc  b_k proportional to max(Dhat_k, 0), Dhat_k = mean over probe cells of W(x) - W_{pin row k}(x),
               W the tight joint interval of P(source = yhat(x) | x) (real/bridge2.py), row k pinned at
               A_hat_k (its entry interval collapsed). All Dhat_k <= 0 -> uniform.
  oracle-alloc as REACH-alloc with row k pinned at the TRUE row (reference only; not a competitor).
The labels are then actually acquired (prefix-extended reference streams) and the state re-estimated.

OUTCOMES (per episode):
  O1 realised mean tight-interval width over probe cells after acquisition (yhat fixed at the start).
  O2 realised prevalence error after acquisition: L1 distance between the known-source weights of the
     best Euclidean fit of p_hat by the new A_hat (renormalised) and the true known-source weights.
     O2 does not use the interval machinery.

SETTINGS.
  Real: CIFAR-10 label shift, grouping super4 and classes, encoders CLIP, DINOv2, ResNet-50 (6 settings);
        CIFAR-100 c100_2, CLIP and DINOv2 (2 settings, reported separately). Dirichlet(1) prevalences.
        n in {2000, 32000}, m0 in {125, 500}. Seeds 60-89, never run before in any test.
  Synthetic: classifier worlds (real/bridge2_amendment.py recipe), (K, per) in {(3,1),(3,3),(6,1),(6,2),
        (10,1)}, accuracy in {0.6, 0.8, 0.95}, lam = 1, 4 draws; n in {2000, 32000}, m0 in {50, 500};
        seeds 10-19 (never run before). The unseen row is estimated once (2000 draws) and fixed across rules.

CLAIMS (registered; paired differences per episode, two-way bootstrap seed x setting/world, 2000 draws):
  AL1 real C10: O1(REACH-alloc) < O1(uniform) and < O1(proportional), both CIs excluding 0.
  AL2 synthetic: same as AL1.
  AL3 real C10: O2(REACH-alloc) < O2(uniform), CI excluding 0.
  AL4 synthetic: O2(REACH-alloc) < O2(uniform), CI excluding 0.
  AL5 (no direction): REACH-alloc vs oracle-alloc gap; CIFAR-100; per (n, m0) breakdown.
FAILURES are reported as failures.
"""
REAL_PARTS = ('c10_clip', 'c10_dinov2', 'c10_resnet50', 'c100_clip', 'c100_dinov2')
REAL_N = (2000, 32000)
REAL_M0 = (125, 500)
REAL_SEEDS = tuple(range(60, 90))
REAL_CAP = 4000
SYN_KP = ((3, 1), (3, 3), (6, 1), (6, 2), (10, 1))
SYN_ACC = (.6, .8, .95)
SYN_LAM = 1.
SYN_DRAWS = 4
SYN_N = (2000, 32000)
SYN_M0 = (50, 500)
SYN_SEEDS = tuple(range(10, 20))
SYN_CAP = 50000
N_BOOT = 2000
