# Visual evidence acquisition: protocol before execution

Date: 2026-10-22. Status: design completed; no new experiment results.

## Question and decision

Can the three target intervals help choose useful evidence at a fixed annotation
budget, beyond directly estimating those targets from the same annotations?
The next pilot is a **fixed-allocation comparison**, not yet an adaptive acquisition
algorithm. Establish useful tradeoffs before designing a policy around them.

Use MNIST digit identity as known source and Fashion as the declared candidate.
Use actual digit parity as a meaningful binary label. Keep the original frozen CNN
as the first representation; add standardized raw pixels with PCA fitted on the
training split as a transparent second representation. This tests dependence on
representation, not pretrained-model novelty. A learned second encoder and multiple
training seeds remain necessary follow-ups if the pilot is useful.

Do not continue optimizing the audit-to-TV construction: direct use of the same
audit confidence set already dominates it. Preserve its negative result.

## An essential limitation of the label task

For digit source s, theta_s = 1{s is odd} is known exactly from the task definition.
The original reference-anchor target remains

    T_label(c) = sum_s rho_s A_s(c) theta_s / sum_s rho_s A_s(c).

It can still have nonzero epistemic width because the source-cell laws A are
uncertain. There is no separate unknown theta to estimate in this task. Revealing
a digit identity reveals parity for free. Charging separately or pretending this
is three independent evidence channels would manufacture the desired result.

This pilot therefore tests a real visual consequence of the existing targets,
but cannot validate a distinct target-label acquisition advantage. That limitation
is a result of the task definition, not a reason to reintroduce synthetic labels.

Classifier correctness is NOT a drop-in replacement for the earlier synthetic
label: generally P(correct | source, cell) differs from P(correct | source).
Using one marginal theta_s would impose an unjustified conditional-independence
assumption. A later correctness/attribute task needs either a defensible source
design or an explicit extension to joint source-cell-label laws. That extension
must be derived and compared with the original estimand before implementation.

## Targets and evaluation populations

- T_cls(k,c): known-digit membership conditional on known status and cell.
- T_unk(c): Fashion membership conditional on cell; not unrestricted unknown detection.
- T_label(c): odd-digit probability under a uniform reference mixture of known digits.

W denotes the range of its target over the specified evidence set. These are
neither additive uncertainty components nor interchangeable targets. W_label
refers to the anchor population, not the deployment mixture. Evaluate its decisions
on a separately balanced known-digit probe population. Never use it as the
deployment parity posterior without changing the estimand.

Use all known classes 0–9 in the first pilot. Hold the six-known/four-held-out
digit case for a later replication, so task and representation changes are not
confounded. Fashion is unknown to the task head; no claim about concepts unseen
in encoder pretraining is needed here.

## Data and representation protocol

1. Freeze representation and partition using only the official training split.
   Reuse the existing CNN weights, recording their hash. Fit the pixel baseline's
   standardization/PCA and the same type of source-supervised cell partition on
   training images only. Fix PCA dimension 16 and ten class cells plus an outlier
   cell; calibrate the outlier threshold using a training-only holdout.
2. Reuse cached official test pools as an explicitly exploratory finite population.
   For each of ten split seeds, assign disjoint baseline-reference, acquisition,
   deployment-log and probe roles before inference; save image IDs and hashes.
   Pair all methods and representations on these assignments. Different seeds
   may reuse images and are not independent training replications.
3. Start from the audited pilot's 200 known references per digit and 512 matched
   Fashion references. Build a 4,000-image deployment log at 20% Fashion and known
   digit weights proportional to 1,...,10. Freeze these choices before scoring.
4. Reserve acquisition capacity before assigning the log: 256 random deployment
   audit images, 256 additional known references and 256 additional Fashion
   references, disjoint from initial evidence and probes. Check feasibility by
   digit before committing manifests; do not silently reuse images or truncate
   a source. The known-reference acquisition pool is sampled uniformly within
   each requested digit, allocating remainders by fixed digit order.
5. Treat exact full-pool laws as evaluation truths only. Report this finite-population
   interpretation and previous exploratory reuse. Reserve fresh data/splits for
   confirmatory claims; a saved probe is not independent of a full-pool census.

The random deployment audit must represent the same designed mixture, rather than
being a sample selected for high uncertainty. Sample source counts randomly from
that mixture, then images without replacement within source, checking capacity.
Its realized class counts are observations, not fixed identities supplied for free.

## What one annotation purchase reveals

| Action | Sampling access | Observed information | Cost convention |
| --- | --- | --- | --- |
| Random deployment audit | Random images from the deployed mixture | Known digit or Fashion identity, cell, and parity when known | One fully revealed image |
| Known reference | An accessible catalog for a requested digit | Digit, cell, parity | One verified catalog image; catalog access assumed |
| Candidate reference | An accessible Fashion catalog | Candidate identity and cell | One verified catalog image; catalog access assumed |

Every method gets every field acquired under its action. In particular, deployment
audits supply weight information AND source-cell observations, not just identities.
Implement bounds valid for their conditional random source counts. Do not treat
the same image as independent evidence twice. Labels already attached to initial
references are available to all methods.

Image counts are an annotation proxy, not measured human time. Report the full
action-cost vector and compare relative reference/audit costs 0.5, 1 and 2. An
accessible class-indexed catalog is an explicit service assumption, not a free
way to discover candidates in the wild. Do not call this clinical annotation.

## Fixed allocations and required comparators

Additional budgets B = 0, 64, 128, 256 verified images. For B>0 compare:

1. All random deployment audits.
2. All known references, balanced over the ten sources.
3. All candidate references.
4. An even split over the three actions, with remainder assigned in that order.

Use nested prefixes within actions and pair observations whenever allocations
share an action. Each arm spends exactly B images. Avoid choosing an allocation
using the true source laws, target errors, or final evaluation labels.

Required inference comparators on each acquired evidence set:

- REACH joint-mass intervals, retaining all available source-cell and identity
  constraints. Label intervals use the fixed reference anchor and exact parity.
- Direct candidate-law estimation from verified candidate images, with the same
  evidence and confidence budget; no audit-derived TV compression.
- Direct empirical target intervals from deployment audits for T_cls and T_unk,
  and from a balanced known-reference sample for anchor T_label. Condition on
  observed cell/sample counts, use simultaneous bounds, and return [0,1] for
  unsupported cells. Do not substitute the deployment parity rate for the anchor.

The last comparator matters: narrowing mixture-model intervals is not a practical
gain if direct target estimation from the same labels already performs as well.
Methods may use less information by construction; document this instead of
artificially withholding acquired fields from stronger baselines.

## Confidence and downstream measurements

Implement finite-population-valid bounds for the sampling design. For random
source/cell counts, justify conditional bounds and their unconditional coverage;
do not reuse iid Clopper–Pearson intervals without checking the sampling model.
Preallocate a total error budget of 0.05 across evidence families, inspected
budget checkpoints and arms if making a simultaneous selection claim. Fixed-setting
coverage and simultaneous coverage must be reported separately. No optional stopping
claim follows from pointwise intervals. Write the allocation in config before runs.

Primary output: paired budget curves of width, truth coverage and rejection for
all three targets, with per-cell counts. Report unweighted and population-weighted
summaries and the outlier separately. Empty feasible sets are rejections.

Downstream decisions, fixed before running:

- Candidate: certify known when the T_unk upper endpoint <=0.1, candidate when
  its lower endpoint >=0.9; otherwise abstain.
- Digit: certify k only if its T_cls lower endpoint >=0.9; evaluate conditional
  on known status and separately report the candidate gate.
- Anchor parity: certify odd at lower endpoint >=0.9 or even at upper endpoint
  <=0.1; otherwise abstain. Evaluate only on the anchor population.

Report fraction of probe mass certified, conditional error among certified images,
and whether the certified population-probability statements are true under the
finite-pool census. Probability thresholds do not guarantee every individual
prediction is correct. Also show risk/abstention curves over a prespecified grid
0.05, 0.1, 0.2; do not select the best threshold after seeing results.

## Gates and next implementation task

Before running the grid: implement manifest/cost accounting, source-count-conditional
bounds and the direct empirical comparator. Validate disjointness, no hidden use
of source labels, shared annotation fields, zero-count behavior and equality of
expenditure. Run one-seed smoke checks before ten-seed evaluation.

Continue to adaptive allocation only if fixed allocations exhibit reproducible,
useful target-specific tradeoffs and modeled inference adds value over direct
estimation at acceptable coverage. No method or CVPR contribution is established
by this protocol. If intervals remain conservative or direct estimation wins,
record that outcome rather than tune the benchmark around favorable cells.

For a genuinely distinct label-acquisition channel, the next scientific design
decision is an annotated visual attribute varying within each source, together
with a justified conditional-label model. That is a separate extension, not a
claim this parity pilot can answer.
