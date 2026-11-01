# Controlled same-representation / three-target characterization

## Scope and definitions

This is a constructed finite-cell study, not a natural-image experiment, a learned
uncertainty estimator or a novelty claim. It tests static target ambiguity and
reducibility separately. All distributions, priors and observation channels are
declared. No hypothesis is chosen using experimental outcomes.

There are three sources S=0,1,2 and three cells C=0,1,2. Sources 0/1 are known;
source 2 is the specified candidate. Every source has the same known input law
a=(1/2,3/10,1/5). Labels are Bernoulli with source-specific rates theta_s, constant
across cells in this controlled construction. All cell and source probabilities
are strictly positive. At cell 0 (equivalently at any cell), the three targets are

- k = P(S=0 | C,S in {0,1}) = w0/(w0+w1).
- l = P_ref(Y=1 | C) = (theta0+theta1)/2, with fixed equal reference weights.
- n = P(S=2 | C) = w2.

The label-independence-of-cell assumption is special to this example, not imported
into the earlier WILDS image analyses. The candidate distribution here is known
and exactly confounded with the known sources. It is not arbitrary unknown OOD.

H,U are independent uniform binary variables. Put p_H=2/5 or 3/5. The unresolved
information state has four equally weighted hypotheses. Two more-informed states
condition that SAME model and prior on an exact oracle revelation U=0 or U=1.
This revelation is deliberately idealized and is not counted as a single ordinary
source observation. No prior, target definition or family is changed within a
comparison. Full information states differ; their predictive projections do not.

R is the full second-order law over joint predictive distributions P(C,Y), with
the exact cell law retained. It is stronger than a scalar MI or marginal prediction.
For every family and state below,

    R = 1/2 delta_[a x Bernoulli(2/5)] + 1/2 delta_[a x Bernoulli(3/5)].

Thus mean label probability, predictive entropy, MI, finite predictive probability
envelopes, and any fixed functional of R are identical. Mixture weights over the
two predictive atoms can additionally range over [1/4,3/4]; the resulting MI
interval is also identical. This sensitivity interval is NOT a confidence interval.

For the compatible finite hypothesis support M(e), the target interval is the
interval hull [min_M t, max_M t]. Not every interior point need be a member of the
finite target set. These are support extrema, not credible intervals. Posterior
variance is reported separately as squared-error Bayes risk under the given prior.

## Proposition 1: the factorization criterion

For fixed shared auxiliary assumptions, a representation R determines a target
interval A on the specified state space if and only if A is constant on each
fiber of R: R(e)=R(e') implies A(e)=A(e').

Proof: if A=F(R), equality of R implies equality of A. Conversely, if A is constant
on every fiber, define F(r) as A(e) for any e with R(e)=r; this is well-defined.
The same argument applies to a specified acquisition-response functional, but
the response includes its channel likelihood. Static and response sufficiency
are distinct. This elementary criterion does not prove anything about all scalar
functions, all epistemic definitions or all possible models.

## Proposition 2: known-source ambiguity is not determined by R

Let r_U=1/4 or 3/4; w=(4r_U/5,4(1-r_U)/5,1/5), and theta_s=p_H for all s.
The deployment label rate is sum_s w_s theta_s=p_H, independent of U, and the
cell law is a. The known-source target is k=r_U.

Before the oracle-U observation, k has hull [1/4,3/4]. After U=0 it is {1/4};
after U=1 it is {3/4}. R is unchanged in both cases. This proves static
insufficiency for k under this information restriction. The other targets remain
l=p_H and n=1/5. Exact unlabeled input laws cannot reduce the k interval because
every hypothesis induces the same a. Exact source weights identify U and k.

## Proposition 3: candidate membership ambiguity is not determined by R

Let c_U=1/5 or 3/5; w=((1-c_U)/2,(1-c_U)/2,c_U), theta_s=p_H.
Again the deployment label rate is p_H and its predictive law is independent of U.
The candidate target n=c_U has hull [1/5,3/5] before U is known, and a singleton
after U=0 or U=1. Meanwhile k=1/2 and l=p_H.

All known-source conditioning masses are at least 2/5. This example is genuine
ambiguity with a well-defined target, not the solver's zero-denominator [0,1]
fallback. Exact source weights resolve n; exact input laws do not.

## Proposition 4: reference-target ambiguity can differ when deployment and reference differ

Fix w=(3/5,1/5,1/5). Put d_U=-1/10 or 1/10 and

    theta0=p_H+d_U, theta1=p_H-3d_U, theta2=p_H.

All rates lie strictly between zero and one. The deployment rate is

    (3/5)(p_H+d_U)+(1/5)(p_H-3d_U)+(1/5)p_H = p_H.

The reference rate, however, is l=(theta0+theta1)/2=p_H-d_U. Its initial hull
is [3/10,7/10]. After U=0 it is [1/2,7/10]; after U=1 it is [3/10,1/2]. The
width falls from 2/5 to 1/5 without changing R. Source targets remain k=3/4,n=1/5.

Even exact deployment label probabilities identify H but leave l with width 1/5
while U remains unknown. Exact known-source label rates identify H,U and l.
The failure condition is the mismatch between the reference target and what the
deployment predictive law records. It is not evidence that predictive uncertainty
cannot characterize its own aligned target.

## Proposition 5: an aligned reference target is a positive control

Fix w=(2/5,2/5,1/5) and theta=(p_H+d_U,p_H-d_U,p_H). Now both deployment and
reference label rates equal p_H. The target l is a function of each predictive
atom in R; its induced law and support hull [2/5,3/5] are recoverable from R and
do not change when U is disclosed. Its posterior variance is also recovered from
R. Its response to an unstratified deployment label is determined by the Bernoulli
likelihood of the same target and is identical across U-information states.

This does not make scalar MI sufficient for arbitrary target laws, and does not
make R sufficient for every source-stratified acquisition channel. The latter can
still depend on hidden U. The positive statement is deliberately restricted to
the aligned target and deployment-label channel.

## Proposition 6: restoring source identifiability gives another positive control

Replace the identical rows by the known source-cell matrix

    A = [[.7,.2,.1], [.2,.7,.1], [.1,.2,.7]].

Its affine difference rank is two. The exact marginal b=w^T A uniquely determines
w on the three-source simplex. Hence (A,b) determines k(c) and n(c) at every cell
with positive denominators. The code checks this independently by ratio LPs and
direct Bayes ratios for both membership families and both U values.

This control deliberately changes the input geometry. The enriched representation
includes the exact input marginal and the known rows; it is not an assertion that
a scalar label-MI score suddenly identifies sources. Distinct weights now have
distinct observable input marginals, so the previous input-law equivalence is gone.

## Finite observations versus oracle observations

Ordinary channels are: one unlabeled cell; one representative source identity;
one unstratified deployment target label; or one label sampled from a known source
chosen uniformly from {0,1}, with that source recorded by the design. The finite
outcome likelihood is evaluated under every hypothesis and Bayes' rule updates
all weights. These are observation probes, not a learned acquisition policy.

Every ordinary outcome has positive likelihood under all initially supported
hypotheses. Therefore no finite sequence removes a hypothesis, and exact support
intervals remain unchanged. Yet posterior variance can decrease. The expected
variance reduction is exactly Var(E[t | outcome]) by total variance. In the two
membership families, source identities reduce target variance while R and its
MI are unchanged for EVERY outcome: U is updated, H is not. Oracle-U-conditioned
states have zero membership variance and no further source-ID reduction.

Oracle channels disclose an exact input law, source weights, deployment label
law, or known-source label laws. They partition the finite hypothesis set and can
shrink support intervals. Their results are identification limits, not finite-cost
sampling guarantees. In this example all sources have identical cell laws, so
more source-reference inputs are also uninformative about the hidden weights.

This separation is essential: a contracting posterior variance is not a contracting
identification interval; a noncontracting support interval is not evidence that
Bayesian learning did nothing. No finite-sample coverage guarantee is asserted.

## Conclusions permitted and not permitted

The three constructions establish, separately under explicit conditions, that the
same full deployment predictive representation can coexist with different support
ambiguity about each declared target. They do not show that all three ambiguities
must differ simultaneously, or that the three quantities are additive components.
The aligned-reference and full-rank-source controls locate conditions where the
appropriate representation does characterize the target.

This connects the earlier same-predictive-law mechanism to all three targets in a
controlled model. It does not establish prevalence in natural images, source-law
transfer, failure of a trained DAPPr model, or novelty relative to existing theory.
The source family, prior and extra oracle information are deliberately specified.

## Sources and their exact roles

- Existing `acquisition_counterexample.ipynb` and its archived `PROOF_AND_PROTOCOL.md`:
  earlier same-predictive-law / different-channel construction. The present study
  adds separate target-range witnesses and positive controls.
- Existing `source_acquisition_four_datasets.ipynb` and `real/source_acquisition.py`:
  finite-cell source definitions and independent ratio-LP certification.
- Hüllermeier & Waegeman (2021), https://doi.org/10.1007/s10994-021-05946-3:
  conceptual distinction between predictive disagreement and incomplete knowledge.
- Charnes & Cooper (1962), https://doi.org/10.1002/nav.3800090303:
  the existing linear-fractional solver used for independent endpoint checks.
- Probly, https://github.com/pwhofman/probly: independent MI calculation on the
  two exact categorical predictive atoms. No dropout or neural fitting is used.

All propositions above are derived explicitly for this construction. Bayes' rule,
finite support extrema, Bernoulli entropy and the law of total variance suffice.
No Hoeffding bound, DAPPr loss, empirical dataset or new literature novelty claim
is used. Sources are reused from the earlier archived studies; no new paper fetch
is needed for these exact calculations.
