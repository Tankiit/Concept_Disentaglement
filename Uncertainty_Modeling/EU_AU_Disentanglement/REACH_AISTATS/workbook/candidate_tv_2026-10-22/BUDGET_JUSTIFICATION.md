# How could a sensitivity budget be justified?

Date: 2026-10-22. Proposed protocol; not executed or calibrated by this workbook.

## What the present sweep supports

Delta is an explicit assumption about the actual candidate distribution relative
to the nominal candidate law. Report sensitivity across delta. Do not call a
selected value calibrated merely because simulated coverage was good, or because
the expanded evidence region becomes compatible.

The minimum expansion restoring compatibility says that **some** explanation fits.
It does not upper-bound how wrong the declared source law is. The earlier
counterexample has zero minimum compatibility budget but a wrong candidate posterior.
Unlabeled deployment histograms alone cannot certify a useful upper bound in such
observationally equivalent worlds. Source prevalences alone also leave unknown
conditional cell laws unresolved in general.

## Option A: independently audited candidate inputs

Collect representative deployment inputs with reliable candidate provenance and
their frozen cell assignments. Compare their candidate-cell histogram with an
independent nominal candidate-reference histogram. This requires actual candidate
identification; head predictions do not supply ground truth. Stratified acquisition
must be representative within the candidate population, or its selection must be
modeled. Audit only a fixed deployment period unless temporal stability is justified.

Let simultaneous coordinate confidence boxes for actual and nominal laws be
[l^D,u^D] and [l^R,u^R]. A conservative upper bound is

`U = min(1, 0.5 sum_c max(u^D_c-l^R_c, u^R_c-l^D_c))`.

Proof: for every cell, the two distributions' absolute difference is at most
the displayed maximum on the joint confidence event. Summing gives a TV upper
bound. The bound ignores simplex coupling and may be loose. Choosing delta=U
then meets the TV assumption on that event, without access to latent simulation
truth. This is a basic conservative baseline, not a proposed new theorem.

Use a held-out audit stream or explicitly account for reuse. If existing candidate
reference bounds are reused, count their failure event once. Audit error must also
enter the overall budget: current experiments already allocate 0.05 across five
families. Adding a 0.01 audit event without reallocating would yield at best 0.06
overall failure, not 0.05. For example reserve 0.01 for audit and 0.008 for each of
the five original families, then rerun inference. Do not retrofit the old results
with a new guarantee. Any adaptive repeated audit needs appropriate simultaneous
or sequential accounting.

**Critical comparator:** the audited candidate inputs themselves estimate the
deployment candidate law. Compare direct intervals using these samples against
the sensitivity method using the same audit and reference budget. If direct
estimation is equally or more effective, that limits the case for a sensitivity
method. Include label/provenance acquisition costs in this comparison.

## Option B: external operational assumption

A bounded replacement/contamination mechanism q_D=(1-epsilon)q_R+epsilon r
implies TV(q_D,q_R)<=epsilon for arbitrary r. A defensible upper bound on the
fraction replaced therefore implies a TV allowance. A value chosen for convenience
does not. The mechanism is an assumption about cells under the fixed representation;
the identity of corrupted images and their selection can matter in other models.

## Option C: sensitivity reporting only

When independent audits or an operational bound are unavailable, report intervals
as functions of delta and the budgets at which a specified conclusion changes.
Do not attach unconditional coverage or announce that compatibility calibrated
the budget. This is the honest interpretation of the current fixed-grid studies.

## Decision for the next experiment

Before adding a sophisticated robust region, run a held-out candidate audit with
predeclared allocation and compare its conservative TV bound against direct
candidate-law estimation at equal evidence cost. Test on actual held-out images
where feasible, with explicit source provenance. This is still pending.
