# REACH: CVPR research roadmap

Last updated: 2026-10-22.
Target venue: **CVPR**, as requested by Mateus. Submission year and deadline have
not been selected or verified. The existing REACH_AISTATS directory name is historical.

## Resume here

Read this file before choosing the next research task; do not rely on chat memory.
Then read the relevant linked experiment reports and inspect the actual code and
artifacts. Update this roadmap when work is completed, assumptions change, or the
user changes priorities. Check the working branch and local changes before edits;
the intended workbook branch has been `reach-mateus-workbook`.

**Immediate next task:** compare sharper sampling-valid source-row bounds against
the current Hoeffding boxes, on the same saved evidence and confidence budget.
The [oracle diagnosis](visual_acquisition_2026-10-22/DIAGNOSIS.md) is complete:
3,696 records; exact sources reduce W_cls from 1.000 to 0.310 at 256 audits,
and exact sources plus exact log identify every target. Source-law bounds are a
major bottleneck; relaxing across-setting multiplicity alone barely changes W_cls.
Investigate finite-population tail inversion with explicit archive-size assumptions
before expanding representations or seeds. Accounting and direct baselines are
implemented; 1,056 records validated. No method certifies a decision at the tested
thresholds. The [full protocol](visual_acquisition_2026-10-22/PROTOCOL.md) remains
pending, including the pixel/PCA representation and repeated splits.
Parity is determined by digit identity, so this pilot cannot establish a separate
target-label acquisition advantage. A richer attribute task requires an explicit
conditional-label model; do not silently substitute marginal correctness rates.
Preserve the three epistemic targets as the central objective.
The image-audit pilot is complete and gives a negative result for the conservative
audit-to-TV approach; do not keep tuning it as a claimed method gain.
The initial audit and graded mismatch control are completed; literature novelty
clearance and a publishable theory/method contribution remain open. This roadmap
records research priorities; it does not authorize submission or publication.

## Honest assessment

The current studies are useful controlled evidence, but are not yet a strong
standalone CVPR paper. Additional datasets or replacing the CNN with CLIP do not
by themselves establish novelty. Several acquisition responses follow directly
from the definitions and must be presented as consistency checks rather than
discoveries. No acceptance prediction is implied by the venue choice.

The strongest existing combination is:

- Camouflage separates finite-evidence uncertainty from exact nonidentifiability.
- Acquisition interventions characterize what identities, references, and target
  labels contribute to each target.
- Misspecification demonstrates narrow, incorrect intervals without rejection;
  compatibility alone does not validate the assumed source family.

Current limitations: one principal frozen MNIST representation, mostly simulation
from saved cell laws, synthetic binary target labels, reliable external source
identities, and small numbers of simulation seeds. Simulation seeds are not
independent training seeds. These studies are not yet broad real-image validation.

## Candidate central research question

**What can each epistemic target be identified from, which observations resolve
its ambiguity, and how sensitive are those conclusions to an incorrect source model?**

Working framing: **three distinct epistemic targets with different information
requirements**. Do not claim three independent or additive epistemic components:
uncertainty in known-source laws affects multiple widths.

Current workbooks use W_cls, W_label, and W_unk. The earlier conversation also
used W_joint; reconcile notation against the reference manuscript explicitly
before writing a paper rather than silently treating these names as synonyms.
The focus remains epistemic uncertainty, not an epistemic/aleatoric decomposition.

## Ordered work plan

### 1. Contribution and theory audit — initial pass completed 2026-10-22

Deliverables: [audit and claim/proof plan](contribution_audit_2026-10-22/AUDIT.md),
[numerical counterexample check](contribution_audit_2026-10-22/theory_checks.json).
Found substantial overlap with BBSE, MaC-LP, open-set label shift, and acquisition-relative
uncertainty. Resolved W_unk versus joint-interval terminology and documented PDF/TeX
differences. The candidate contribution is now sensitivity to model error with
target-specific information requirements, not novelty of the LP alone. Focused
review completed; full algorithmic comparison and novelty clearance are unfinished.

- Define each estimand, conditioning event, evidence channel, and assumption.
- Distinguish source identities from binary target labels, and empirical cell-law
  simulations from experiments on independent images.
- Map REACH against mixture estimation, quantification under label shift, partial
  identification, and OOD uncertainty methods. Complete a focused literature review.
- Identify what is new beyond three applications of interval estimation.
- Investigate identification conditions, exact ambiguity floors, and what added
  evidence can resolve. State which properties are mathematical consequences of
  the construction and which require empirical evidence.

Deliverable: a concise contribution matrix and formal claim/proof plan, including
counterexamples and assumptions. Decision point: if there is no distinct useful
result or method, revise the contribution before expanding benchmarks.

### 2. Graded candidate mismatch — control completed 2026-10-22

Deliverable: [results and failure-boundary plot](graded_mismatch_2026-10-22/RESULTS.md).
Fixed Fashion deployment, two mismatch directions, seven grid points, exact/finite
laws, 0/8,192 identities, ten seeds: 560 runs and 18,480 validated interval records.
Every tested declared family fits the population input law with alternative weights.
Estimated EMNIST references at t=0.4 with identities give 50/100 known-cell candidate
intervals covering truth and no rejections. This motivates step 3. This finite-cell
control does not replace real-image validation or establish a new method.

- Interpolate continuously between matching and wrong candidate laws, preserving
  a fixed deployed world and pairing observation draws across settings.
- Include matching controls, exact and estimated references, and identity budgets.
- Track interval endpoints, widths, target error, coverage, and global rejection
  together. Report nonempty-interval statistics separately from rejected runs.
- Locate where coverage fails before incompatibility becomes detectable.
- Report candidate prevalence and per-cell results; do not hide rare/outlier cells
  behind averages. Predefine thresholds and experimental choices.

Deliverable: a failure-boundary plot and an explanation of how the assumed mixture
family can fit inputs while giving incorrect source memberships.

### 3. Sensitivity to source-model error — coordinate-box baseline completed 2026-10-22

Deliverable: [sensitivity baseline and plots](candidate_sensitivity_2026-10-22/RESULTS.md).
Three cases, seven fixed L-infinity error allowances, 0/8,192 identities, ten seeds:
420 runs and 13,860 validated intervals. Partial EMNIST mismatch with identities:
delta=0.05 changes mean W_unk 0.095→0.247 and observed coverage 50/100→100/100.
This is not a calibrated error budget. Severe Blood mismatch needs delta=1 on
the tested grid; without identities the recovered intervals are vacuous. Matching
controls also widen. Nested-set and conditional-inclusion reasoning documented.
Follow-up: [coupled TV comparison](candidate_tv_2026-10-22/RESULTS.md) completed:
420 runs, 13,860 intervals, six analytic checks. No coverage/rejection differences
from boxes; modest average tightening concentrated without identities, with a few
large local gains. No broad method improvement established.
[Budget justification protocol](candidate_tv_2026-10-22/BUDGET_JUSTIFICATION.md)
documents an independent-audit TV bound, error-budget accounting, and a necessary
direct-estimation comparator. The [candidate image-audit pilot](candidate_audit_2026-10-22/RESULTS.md)
has now executed it: real cached reference/audit/deployment images are disjoint
within each seed; target labels and optional identity-only controls remain synthetic.
480 runs and 15,840 intervals validated. At 2,048 audited images, direct mean
W_unk is 0.586 versus audit-TV 1.000 without identities, and 0.316 versus 0.664
with identities; both cover all recorded targets. Direct-set containment explains
why this conservative audit-to-radius approach cannot beat direct use of the same
audit confidence set. Preserve this negative result. Novelty, general calibration
and useful real-image decisions remain open; step 3 is not a demonstrated novel method.

- Allow explicitly bounded deviations from declared source laws and compute
  sensitivity intervals across a deviation budget.
- Justify the budget using available evidence, or present it openly as a
  sensitivity parameter. Do not choose it using the true deployed law and call
  the result a practical procedure.
- Compare nominal and sensitivity intervals using coverage, informativeness,
  rejection, and acquisition costs where relevant.
- Establish guarantees conditional on the stated deviation assumption, if possible.

Deliverable: a principled method or diagnostic that helps a user understand which
conclusions survive misspecification. Widening everything to [0,1] is not sufficient.

### 4. Real-image validation and CVPR-facing contribution — one-seed smoke completed

The [visual acquisition protocol](visual_acquisition_2026-10-22/PROTOCOL.md) specifies
fixed equal-budget actions, direct target-estimation baselines, probability-threshold
decisions and stop/go criteria. It explicitly exposes the redundancy between digit
identity and parity annotations. The [smoke results](visual_acquisition_2026-10-22/RESULTS.md)
validate image/cost accounting but show no certified decisions. At 256 audits,
mean W_unk contracts from 0.934 to 0.808; W_cls stays at 1.000. Do not expand to
adaptive acquisition on the basis of width contraction alone.

- Use genuinely disjoint image reference/deployment sets and multiple training
  seeds; vary the representation as well as sampling evidence.
- Add a second representation (potentially a pretrained encoder such as CLIP),
  chosen for a concrete visual question, not to substitute model size for novelty.
- Define unknown classes relative to the task head separately from pretraining
  exposure. Do not claim a pretrained encoder has never seen those concepts.
- Include controlled image transformations or semantic shifts and a meaningful
  target-label task. Avoid treating synthetic binary labels as clinical labels.
- Study partitions: the conclusions are conditional on what the representation
  and cell map preserve. Prevent deployment-label leakage during partition fitting.
- Demonstrate a useful visual-learning consequence, such as selecting which
  annotations to acquire or identifying conclusions that are unsafe under shift.

Deliverable: reproducible evidence that the proposed contribution survives real
image sampling, representation changes, and realistic information constraints.
These are project recommendations, not claims about formal CVPR requirements.

## Comparisons and evaluation rules

- Compare interval claims against appropriate mixture-estimation and
  partial-identification methods targeting the same estimands and using the same
  information. Include straightforward controls, not only neural uncertainty scores.
- Dropout, ensembles, and OOD scores address related predictive questions, but
  generally do not estimate these exact targets. Evaluate them on a clearly shared
  downstream task rather than equating their scores with posterior interval widths.
- Report coverage and width together; narrow is not automatically good, and broad
  is not automatically failure. Empty feasible sets are rejections, not width zero.
- Failure to reject is not proof of correct specification.
- Separate oracle controls from deployable procedures. Account for annotation
  budgets and whether candidate identities are realistically obtainable.
- Keep correctly specified and misspecified results separate; nominal guarantees
  do not automatically survive a wrong source family.

## Existing evidence to reuse

| Study | Main role | Report |
| --- | --- | --- |
| Boundary stress | Representation-dependent compatibility and narrow wrong intervals | [Results](boundary_epistemic_2026-10-21/RESULTS.md) |
| Camouflage | Weak versus exact nonidentifiability | [Results](camouflage_2026-10-21/RESULTS.md) |
| Source identities | Additional evidence can resolve input-only ambiguity | [Results](camouflage_identities_2026-10-21/RESULTS.md) |
| Reference acquisition | Known and candidate references affect targets differently | [Results](camouflage_references_2026-10-21/RESULTS.md) |
| Target labels | Finite-label and finite-reference contributions to W_label | [Results](camouflage_labels_2026-10-21/RESULTS.md) |
| Candidate mismatch | Wrong laws can yield narrow, wrong intervals without rejection | [Results](candidate_mismatch_2026-10-21/RESULTS.md) |

Concrete warning example: Fashion deployment with estimated Blood references and
no identities gives mean known-cell W_unk approximately 0.001, with 20/100 intervals
covering the actual target and no rejections across ten seeds. Matching Fashion
references give mean width 0.249 and 100/100 coverage. These are dependent simulated
intervals, not 100 independent calibration trials. External identities expose the
Fashion/Blood mismatch in 10/10 seeds, but Fashion/EMNIST only in 7/10 with estimated
references. See the report for precise definitions and other cells.

## Initial literature anchors — review incomplete

- [Quantification Under Prior Probability Shift: the Ratio Estimator and its Extensions](https://jmlr2020.csail.mit.edu/papers/v20/18-456.html):
  related prevalence confidence intervals and incorporation of target labels.
- [Understanding Failures in Out-of-Distribution Detection with Deep Generative Models](https://proceedings.mlr.press/v139/zhang21g.html):
  limits on OOD detection without assumptions on relevant out-distributions.

These anchors were checked during the initial assessment, not an exhaustive
novelty review. General claims that more evidence helps or arbitrary OOD detection
is impossible are not sufficient contributions on their own.

## Working quality bar

Before drafting a submission around these experiments, establish a distinct
formal result or useful method, credible comparisons, and convincing real-image
validation. If that cannot be established, narrow or change the claim rather than
inflate the number of experiments. Preserve negative results and limitations.
