# Epistemic uncertainty failure under contamination

Integrated narrative: [Research report — experiments 01–10](RESEARCH_REPORT.md), 24 October 2026.

Date: 2026-10-23
Status: frequentist evaluation framework confirmed; experiments 01–04 ingested and the theory completed in [experiment 05](experiments/05_frequentist_theory/RESULTS.md) and the full-cell frequentist comparison executed in [experiment 06](experiments/06_full_cell_frequentist/RESULTS.md).

## Agreed framework

The clean contour and contamination law are fixed unknown targets. Evaluate uncertainty by repeated-sample whole-contour coverage, estimation error, and set width under declared information and source assumptions. No prior is required. Experiments 03–04 are retained as baseline/sensitivity investigations; their Bayesian direction was introduced by the assistant, not agreed as the paper's framework.

The main theoretical reference is a contamination-aware frequentist confidence construction. It is a standard-ingredients evaluation control, not REACH or the replacement definition reserved for paper two. A confidence procedure failing under contamination does not, by itself, disprove all epistemic uncertainty definitions.

## Ingested evidence

- [01: indistinguishable worlds](experiments/01_indistinguishable_worlds/RESULTS.md): irreducible clean-contour ambiguity; the obstruction belongs to established robust-statistics theory.
- [02: MI target check](experiments/02_mutual_information_target_check/RESULTS.md): observation MI and contour uncertainty differ; target-aligned MI preserved ambiguity.
- [03: posterior comparison](experiments/03_posterior_contour_comparison/RESULTS.md): narrow wrong uncertainty under a misspecified observation model, measured by repeated sampling.
- [04: broader model](experiments/04_general_contamination_model/RESULTS.md): source assumptions and grids change conclusions; posterior prior sensitivity is a side investigation.
- [05: frequentist theory](experiments/05_frequentist_theory/RESULTS.md): confidence projection and proof, exact frequentist counterexample, width constraint, and certified off-grid safeguard.

- [06: full-cell frequentist comparison](experiments/06_full_cell_frequentist/RESULTS.md): Wald and bootstrap contour sets lose coverage under contamination while a conservative contamination-aware outer set retains it; reference width remains numerically loose.

- [07: probly method-selection audit](experiments/07_probly_method_selection/RESULTS.md): user-selected UQ library; ensemble-first integration proposed, with target/adapter and version checks required. No probly run yet.

Record theoretical decisions, new understanding, and sources with supporting quotations in each RESULTS.md; see [local research instructions](AGENTS.md).

## Paper statement

> This paper investigates a fundamental limitation of existing epistemic uncertainty formulations through a controlled finite-cell source-mixture study. Using a spherical Gaussian with controlled contamination, we aim to establish conditions under which epistemic uncertainty fails to reflect uncertainty about the clean distribution’s density contour. We characterize this failure across sample sizes, contamination levels, and spatial regions through theoretical analysis and empirical evaluation.

This is the first paper: establish a precise failure and its scope. REACH is set aside. A replacement formulation, an epistemic covariate set, and a proposed shift decomposition belong to a separate follow-up paper. OpenOOD may motivate later evaluation, but is not the core experiment here. Earlier REACH workbooks are historical context, not the specification for this study.

## Research question and intended claim

When estimating a clean Gaussian's contour from finite, contaminated observations, do selected epistemic uncertainty formulations reflect the knowledge that remains unresolved about that contour?

Working hypothesis: a specified reported uncertainty can contract even when the observations do not uniquely determine the clean contour. Persistent contour error alone is weaker evidence: it can arise from bias or model misspecification without establishing a failure of the quantity the method claims to estimate.

We must distinguish an estimator that poorly approximates its stated epistemic quantity from a correctly estimated quantity that is insufficient for the intended knowledge claim. The main conceptual interest is the latter. We cannot conclude that every definition of epistemic uncertainty fails from a counterexample for selected measures.

## Controlled model and information available

Start in two dimensions. The clean source is

\[
P_\theta=\mathcal N(\mu,\sigma^2 I_2),\qquad \theta=(\mu,\sigma),
\]

and the observed source mixture is

\[
M=(1-\varepsilon)P_\theta+\varepsilon Q.
\]

Here Q is a contamination distribution. Begin with known 0 < epsilon < 1/2 and unknown Q; subsequently study constrained Q, unknown epsilon, and clean reference observations. These are proposed information regimes, not interchangeable assumptions.

Partition the entire observation space into fixed cells C_1,...,C_K, including an overflow cell if using a bounded plotting grid. Define

\[
p_j(\theta)=P_\theta(C_j),\quad q_j=Q(C_j),\quad
m_j=(1-\varepsilon)p_j(\theta)+\varepsilon q_j.
\]

The initial learner observes cell counts N ~ Multinomial(n,m), not source identities. The evaluator knows the generating parameters and contamination. Evaluator knowledge must not leak into learner inputs. If a later method receives raw coordinates, it needs a separate analysis: identical cell probabilities do not imply identical raw-data distributions.

The primary experiment contaminates the observations used for inference. Clean training followed by shifted test observations is a distinct, secondary protocol.

## Target: the clean contour

Use the boundary of the clean Gaussian's central region containing a fixed probability alpha (a proposed initial choice is alpha = 0.95):

\[
\Gamma_\alpha(\theta)=\{x:\|x-\mu\|=
\sigma\sqrt{-2\log(1-\alpha)}\}.
\]

This is a density contour specified through enclosed probability, rather than a fixed numerical density threshold. Fix this convention before experiments. The learner knows the Gaussian family but not its true parameters; otherwise the contour could already be known. Under this family, estimated contours are circles with uncertain centers and radii. Irregular empirical contours require an explicitly different, more flexible estimator.

A Gaussian has no hard support boundary. This contour is the target boundary, not a separator that perfectly identifies contamination. Source membership can remain ambiguous even when both source distributions are known.

## Initial theoretical exploration: indistinguishable clean contours

The following is a direct finite-cell construction, not a novelty claim or an established failure theorem for any particular epistemic measure.

Take two spherical Gaussian parameters theta_0 and theta_1 with distinct target contours. Let their cell laws be p_0 and p_1 and define

\[
\delta=\operatorname{TV}(p_0,p_1)
=\tfrac12\sum_j|p_{0j}-p_{1j}|.
\]

If

\[
\delta\leq\frac{\varepsilon}{1-\varepsilon},
\]

there is a shared observed law m and contamination laws q_0,q_1 satisfying

\[
m=(1-\varepsilon)p_0+\varepsilon q_0
 =(1-\varepsilon)p_1+\varepsilon q_1.
\]

Construction: since sum_j max(p_0j,p_1j) = 1 + delta, let

\[
R=1-(1-\varepsilon)(1+\delta)\geq0,
\qquad m_j=(1-\varepsilon)\max(p_{0j},p_{1j})+Rr_j,
\]

for any probability vector r. Set q_ij = [m_j-(1-epsilon)p_ij]/epsilon. Each q_i is nonnegative and sums to one. Conversely, a shared m must dominate both scaled clean laws, which yields the same necessary inequality. The construction assumes unrestricted contamination cell laws and allows Q to differ between the two possible worlds. It need not hold for a fixed known Q or a restrictive contamination family.

Thus the observed counts have exactly the same distribution in both worlds, at every n, while the clean contours differ. Close distinct Gaussian parameters provide candidate pairs satisfying the inequality for positive epsilon. Finite-cell aggregation can create additional ambiguity; compare partition resolutions and verify that the selected contour separation is meaningful.

Let d be Hausdorff distance between the two contours. For any contour estimator using only the counts, the triangle inequality and the common observation law imply

\[
\max_{i\in\{0,1\}}\mathbb E_i
[d_H(\widehat\Gamma,\Gamma_i)]\geq d/2.
\]

For a confidence set of contours with coverage at least 1-beta in each world, the same observation law and the union bound imply that it contains both contours with probability at least 1-2 beta. On that event its diameter in Hausdorff distance is at least d. Uniformly valid uncertainty therefore cannot collapse to zero in this setting.

This establishes an information limitation. It does NOT yet prove that an existing epistemic formulation fails: we must select the formulation, specify its target and assumptions, and establish its behavior under the same evidence. A method that retains the ambiguity would be a successful control.

## Bridge to a failure claim

For each selected measure, document:

1. Its mathematical definition, intended target, and conditioning assumptions.
2. Whether it measures parameter uncertainty, predictive disagreement, contour uncertainty, or another quantity.
3. The exact mapping from its output to a claim about the clean contour, if such a mapping is justified.
4. Whether contraction under the indistinguishable-world construction can be proved, or only observed empirically.
5. Whether the mechanism is approximation error, restrictive model assumptions, target mismatch, or a limitation of the definition itself.

Primary candidates are clean-model frequentist contour confidence procedures and resampling-based contour uncertainty. Experiment 05 establishes an exact sign-statistic frequentist counterexample under contamination; general bootstrap and full-parameter results are not yet established. Previously explored posterior procedures remain optional baselines. Classifier disagreement or mutual information would require a specified prediction task and a justified connection to the contour; their scores cannot automatically be treated as contour confidence bands.

Correct uncertainty about the observed mixture is not automatically incorrect uncertainty about the clean source. The paper must make the interpretation being challenged explicit and avoid claiming that a method violates guarantees outside its stated assumptions.

## Empirical evaluation plan

- Clean control: epsilon = 0, to assess ordinary finite-sample behavior.
- Exact indistinguishable-world pair: same m, different clean contours, with paired observations to isolate the information limitation.
- Geometric contamination: vary location, spread, and overlap using specified Q families; check identifiability separately rather than assuming the exact construction applies.
- Information controls: known Q, clean references, or observed source identities, which can remove or reduce ambiguity.
- Resolution controls: vary cell size and compare exact population cell laws with finite counts to separate aggregation effects from sampling uncertainty.

Sweep sample size, contamination level, and contour position. Report contour error, uncertainty width, pointwise coverage, simultaneous whole-contour coverage, and repeated-dataset variability. Specify how pointwise bands are constructed when the center is uncertain. Report Hausdorff error for global contour recovery. Scalar uncertainty scores need their own justified evaluation; do not label them confidence bands without a calibrated construction.

Show the true clean contour, the estimated contour, and the reported uncertainty on the same plots. Separate finite-sample effects, unresolved source ambiguity, contamination bias, and cell discretization. No numerical result is claimed in this proposal.

## Next steps and status

1. **Completed:** fix the clean contour target, finite-cell evidence, and frequentist coverage criterion; distinguish whole-contour coverage from pointwise and identified-set coverage.
2. **Completed:** establish indistinguishable worlds and necessary confidence-set width; recognize their established robust-statistics context.
3. **Completed in experiment 05:** derive a finite-sample contamination-aware confidence projection, exact feasibility test, general biased-center failure criterion, exact frequentist counterexample, and an off-grid outer safeguard.
4. **Completed in experiment 06:** full-cell comparison of Wald, bootstrap, and a continuous-box reference on matched data. Coverage checks and numerical refinement checks completed; tighter reference diameter remains open.
5. **Partly completed:** clean off-grid and geometric-contamination controls with coverage intervals, width bounds, and contour error. Broader contamination sweeps, precise clean-data calibration, and spatial envelopes remain open.
6. **Active:** use probly as the candidate UQ implementation layer, beginning with an ensemble integration audit and explicit entropy/variance targets. Experiment 08 pins probly 0.9.1 and validates its quantification layer on an unsupervised Gaussian bootstrap adapter, with a 180-dataset pilot. Predictive MI contracts while contaminated contour error persists. This is not a neural deep-ensemble replication; experiment 09 now exercises the actual PyTorch ensemble transformation with a normalized three-component neural mixture-density adapter. Its 20-dataset pilot includes matched controls and optimization/contour checks; active parameter bounds, convergence, clean calibration, and architecture sensitivity remain open. Derive behavior and assess novelty; do not infer a universal failure from the present controls.

### Experiment 10 sensitivity update

The matched bounds/component-count test is complete: four configurations, 20 fixed datasets, 400 retained neural members. At n=100,000, contaminated clean-contour error persists (mean 0.348–0.376 across configurations/truths). Low-sample contour errors and member spreads are highly model-sensitive; flexible fits can produce disconnected tail islands even on clean data. Expanded bounds remain active and optimization tolerances are not met uniformly. The next priority is separating partition/within-cell geometry from source ambiguity and establishing clean calibration before broader neural-method claims. See [experiment 10](experiments/10_bounds_components_sensitivity/RESULTS.md).
