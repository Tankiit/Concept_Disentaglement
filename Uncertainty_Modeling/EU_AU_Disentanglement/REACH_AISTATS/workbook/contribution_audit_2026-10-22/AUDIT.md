# REACH contribution and theory audit

Date: 2026-10-22. Initial focused audit, not an exhaustive novelty clearance.
Target: CVPR. No manuscript claims or proofs were silently replaced.

## Decision

Do not position the LP confidence set, acquisition-relative uncertainty, or
input-only mixture nonidentifiability as new by themselves. Proceed with the
graded mismatch diagnostic to establish the scope of a possible contribution:
**target-specific sensitivity to a wrong visual source model, and information
that exposes or resolves that error**. This is a research direction, not yet a
demonstrated novel method. A deployable sensitivity procedure and real-image
evaluation remain required before claiming a strong CVPR contribution.

## Definitions checked against local sources

Read [reference PDF](../AISTATS2027_REACH_.pdf), especially Eq. (5) and identification
discussion, and [editable manuscript](../../paper/submission/main.tex), setup and
theory sections. They are different manuscript versions. The PDF separates source
identity from binary target labels and uses two known sources; recent workbooks
generalize that construction to ten known sources. The TeX also discusses a
different real-data label estimand and plug-in versus joint intervals. Do not
merge their claims without an explicit reconciliation.

Let rows A_s(c) be source-conditional cell probabilities, w source proportions,
u the declared candidate, and rho fixed positive known-source anchor weights.

| Target | Functional | Interpretation |
| --- | --- | --- |
| T_cls(k,c) | w_k A_k(c) / sum_known w_j A_j(c) | Known-source membership conditional on known status and cell |
| T_unk(c) | w_u A_u(c) / sum_all w_j A_j(c) | Declared candidate membership conditional on cell |
| T_label(c) | sum_known rho_j A_j(c) theta_j / sum_known rho_j A_j(c) | Binary target probability under a fixed reference mixture |

W is the supremum minus infimum of the corresponding functional on its specified
evidence set. A vanishing possible denominator triggers [0,1] by policy; an empty
deployment set is rejection. The reference-anchor label calculation can remain
defined after deployment rejection because it uses a separate evidence set.

**Notation resolution:** neither inspected source defines W_joint as the third
target. The third target is W_unk. “Joint” in the editable manuscript describes
propagating multiple uncertain quantities into an interval, not a fourth or
replacement estimand. Preserve W_unk in code and explicitly clarify this in writing.

## Implementation versus manuscript

Current workbook inference uses normalized nonnegative joint masses Z_sc with
row sums w_s, bounds l_sc w_s <= Z_sc <= h_sc w_s, bounded deployment column sums,
and optional bounded row sums from identities. Ratios are optimized by a
Charnes–Cooper LP. The coordinate bounds use simultaneous Clopper–Pearson bounds;
synthetic label rates use Hoeffding bounds with a known invertible noise channel.

This is not the editable manuscript's Euclidean MMD-ball plug-in construction.
Rate claims for that construction do not automatically prove contraction of the
current normalized joint-mass region. W_label currently ranges over independent
known-row boxes and theta intervals, not deployment constraints. Its input/identity
invariance is by design; unknown source laws could otherwise be informed by joint
deployment evidence. Rows and target labels are simulated independently here.

## Contribution matrix and closest primary sources

| Proposed claim | Prior overlap / evidence | Assessment |
| --- | --- | --- |
| Infer mixture weights from classifier cells | [BBSE, Lipton et al. 2018](https://proceedings.mlr.press/v80/lipton18a.html) uses predictor confusion matrices for label-shift estimation with invertibility conditions. | Established foundation, not a novelty claim. |
| Finite-sample matrix-constraint regions and LP intervals | [Li, Han, Ma 2026](https://arxiv.org/html/2609.14802v1) develops MaC-LP with estimated matrices and diameter bounds. | Closest required statistical baseline; compare actual regions and targets before claiming improvement. |
| Novel-class proportion identification | [Garg et al. 2022](https://arxiv.org/abs/2207.13048) studies open-set label shift and identification conditions. [Ramaswamy et al. 2016](https://proceedings.mlr.press/v48/ramaswamy16.html) studies mixture proportion estimation under structural assumptions. | A declared extra row is easier than an unspecified unknown law; do not claim generic open-set detection. |
| Reducibility depends on acquired information | [Young 2026](https://arxiv.org/html/2606.12646v1) explicitly treats acquisition-relative reducibility and ensemble limitations. | The general framing is already present; REACH needs an operational distinction beyond terminology. |
| Confidence intervals and target-label acquisitions under shift | [Vaz et al. 2019](https://jmlr2020.csail.mit.edu/papers/v20/18-456.html) studies quantification intervals and available target labels. | Acquisition benefits alone are insufficient novelty. |
| Three posterior targets respond differently | Existing workbooks; several responses follow from excluded evidence channels. | Useful organization and controls, not yet a sufficient standalone contribution. |
| Identify when wrong models remain compatible yet give wrong target conclusions | Endpoint mismatch experiment plus proposed graded study. | Promising diagnostic focus; novelty and practical value remain to establish. |

The primary Li and Young articles were inspected beyond their abstracts; the
others were checked at the primary abstract/landing-page level in this focused
pass. Full comparison of estimands, assumptions, and algorithms remains open.
Do not interpret absence of a matching claim in this pass as proof of novelty.

## Formal claims and proof plan

### A. Correct-specification inclusion — elementary proof

If all row, column, identity and theta confidence events hold and reference laws
equal deployment laws, Z*_sc = w*_s A*_s(c) lies in the joint region. Extremizing
each defined target therefore includes its truth. A union bound over evidence
families supplies simultaneous inclusion at a fixed experimental setting. No
independence between these events is necessary. This gives no guarantee under
wrong candidate laws or simultaneously across an unadjusted adaptive budget grid.

Status: direct proof, established confidence-set reasoning, not new.

### B. Target identification is constancy on the observational fiber

For exact A and p, F_A(p)={w in simplex: A^T w=p}. A target is identified iff it
is constant on F_A(p). Full row rank is sufficient for identifying w, not necessary
for identifying a particular target. Rank deficiency alone does not imply every
cell width is positive. For q=sum beta_k A_k, the segment
w_u=t, w_k=s_k-beta_k t preserves p wherever all weights remain nonnegative.
The target-specific range over that segment is the relevant floor.

Status: direct algebra; already substantially present in the reference PDF.

### C. Compatibility cannot certify a misspecified candidate — exact counterexample

Use two cells and one known source a=(1,0). The actual candidate is q=(0,1),
with actual candidate weight rho=0.2, giving p=(0.8,0.2). Declare instead
q'=(0.5,0.5). The wrong model fits p exactly with candidate weight 0.4 and known
weight 0.6. Both two-row matrices are invertible. At cell 0, actual candidate
membership is 0, while the uniquely fitted wrong model gives 0.4*0.5/0.8=0.25.
Thus exact input knowledge and exact knowledge of the declared law yield a
zero-width, wrong interval without rejection. This is not a finite-sample effect.

Source references describe the declared family in both worlds; unlabeled logs
alone cannot decide whether that family is appropriate for actual deployment.
Reliable identities distinguish the two weight vectors. Conversely, identities
alone need not identify uncertain source-cell laws.

Status: algebraic counterexample to an overly strong REACH claim, not asserted
as a new general impossibility theorem. Numerically checked in theory_checks.py.

### D. Exact identities reveal the mismatch residual

Keep known laws and actual w fixed. With q_t=(1-t)q+t q_bad, the wrong model's
marginal at that same w differs from truth by rho*t*(q_bad-q). Hence its total
variation distance is rho*t*TV(q_bad,q). With exact w, exact reference laws and
exact inputs, any positive residual is incompatible. Without identities, weights
can compensate: population compatibility is p in conv(rows(A_t)).

For any alternative v, the stochastic mixture map obeys
||A_t^T(v-w)||_1 <= ||v-w||_1. Consequently, if input tolerance is epsilon_p in L1
and admissible identities restrict ||v-w||_1 <= epsilon_w, then
rho*t*||q_bad-q||_1 > epsilon_p+epsilon_w is sufficient for rejection with exact
laws. Additional source uncertainty requires an additional tolerance term.
This sufficient inequality is not the threshold of our coordinate-box procedure.

Status: direct triangle-inequality proof. Use population residuals alongside the
finite experiment; do not substitute them for finite-sample rejection rates.

### E. Sensitivity-region extension — next method task

A simple baseline expands candidate coordinate bounds by a specified delta and
retains row normalization. If actual candidate coordinate errors are bounded by
delta, the true row lies in the expanded box on the reference confidence event;
claim A then applies. Increasing delta nests regions, so nonempty target ranges
cannot shrink. This is a basic robustification, not an established novel method.
TV ambiguity with coupled coordinates may be more informative but needs an
explicit formulation and comparison. Budget calibration and real-image usefulness
are unresolved. Do not calibrate delta using hidden deployment truth.

## Experiment decision and remaining gates

Run a fixed Fashion deployment with declared laws interpolating toward Blood and
EMNIST. Preserve the previous ten seeds and evidence streams. Record all three
targets, rejection, per-cell coverage/width/error, and population residuals.
This follows from C–D and investigates a failure boundary, not an attempt to
manufacture a positive method result. Keep the reverse-direction endpoint study
as evidence that detection is asymmetric.

Remaining before a paper claim: broader related-work review, formal comparison
with MaC-LP, validated sensitivity/acquisition method, and independent real-image
evaluation with a meaningful visual decision. No theory in this audit has been
machine-checked, and existing manuscript Lean claims have not been revalidated.
