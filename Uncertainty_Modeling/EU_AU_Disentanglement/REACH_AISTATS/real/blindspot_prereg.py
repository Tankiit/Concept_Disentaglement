"""Pre-registration for the natural-shift blind-spot test (written 2026-10-03, before any code
for the test existed and before CIFAR-10.1 features were extracted). SHA-256 goes in the manifest.

QUESTION. On shifts nobody constructed there is no ground-truth W. The claim is behavioural:
does REACH's per-cell regime say which acquisition will actually pay, where a scalar epistemic
score does not?

DATA. Shift splits: clean CIFAR-10 test (label shift only), CIFAR-10.1 v6 (natural re-collection),
and CIFAR-10-C (15 corruptions x severities 1..5). Frozen CLIP ViT-B/32 -> PCA-32 -> equal-prior
LDA head fitted on clean training data (as in real/audit.py). Sources: the four superclasses.
Each shift split is partitioned once (fixed seed) into disjoint pools: 50% log, 20% label, 30% eval.

EPISODE (shift split, seed). Deployment prevalence pi ~ Dirichlet(1) over sources (per seed).
Current state S0: clean reference rows A_clean from m = 2000 clean images per source; an
unlabelled log of n0 = 1000 shifted images drawn by pi; EMQ prevalence -> posterior per cell.

ACTIONS at matched cost, unlabelled : label = 1 : 50, budget B = 5000 units:
  reweight_from_log : log grows to n0 + B = 6000 shifted images (prefix-extended); EMQ refit on A_clean.
  add_labels        : B/50 = 100 labelled shifted images drawn by pi; rows re-estimated as
                      (M0 * A_clean + shifted counts) / (M0 + shifted n) with prior strength M0 = 50;
                      EMQ refit on the log of size n0.
PAYOFF of an action at cell c: reduction of held-out risk in that cell, risk = importance-weighted
(to pi) mean negative log-likelihood of the true source under the posterior, over eval-pool inputs
mapped to c. Payoffs are measured, never predicted.

REACH REGIME (computed from S0 only, no outcome data), per cell c:
  off_frame     if the declared mixture is incompatible at S0 (best-fit residual > ball radius)
                or c is the outlier cell                                -> recommend add_labels
  log_resolves  else, if the predicted reduction of the joint width W_joint(c) from n0 -> n0+B
                exceeds that from m -> m + B/50/K (labels spread over K sources) -> recommend reweight
  label_limited otherwise                                               -> recommend add_labels
Predicted reductions re-solve the ball at the larger budget with the S0 estimates (no new data).

BASELINE. Ensemble mutual information at S0 (members resample BOTH the log and the references,
so the baseline does see the log). EU terciles are fixed on the pooled calibration seeds. Its
tercile -> action map is the one with the best matching accuracy on calibration seeds 0..9 (an
oracle-tuned map, favourable to the baseline). REACH's rule is not tuned. Evaluation seeds 10..19.

PRE-REGISTERED CLAIMS (evaluation seeds; units = (episode, cell), weighted by eval mass):
  H1 (blind spot): within a fixed baseline-EU tercile, the realised advantage of labels over
     reweighting, D = payoff(add_labels) - payoff(reweight), is larger in REACH {off_frame,
     label_limited} cells than in log_resolves cells. Signed difference with a seed-bootstrap 95%
     CI; supported if the CI excludes 0 with positive sign in at least 2 of 3 terciles.
  H2 (decision value): REACH's recommendation has lower regret than the tuned baseline map,
     regret = payoff(best action) - payoff(recommended). Reported as the difference with CI.
  H3 (sanity): realised D increases with CIFAR-10-C severity (labels pay more as conditionals
     move), Spearman over severities > 0.

IF THEY AGREE. If H2's CI covers 0 and H1 fails, we write: on these shifts a scalar EU with a
tuned threshold chooses the acquisition as well as REACH's regimes; REACH's contribution there is
the coverage guarantee, not the decision. That is a reportable negative result, not a reason to
change the rule, the budget, or the data.

FIXED IN ADVANCE: n0, B, the 1:50 price, M0, the pool split, the seeds, the regime rule, the
baseline and its tuning protocol. ImageNet-R / DomainNet are not part of this registration.
"""
N0 = 1000
BUDGET = 5000
PRICE = 50
M0 = 50
M_CLEAN = 2000
POOLS = dict(log=.5, label=.2, eval=.3)
CALIBRATION_SEEDS = range(0, 10)
EVALUATION_SEEDS = range(10, 20)
