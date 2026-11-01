# Same epistemic uncertainty, opposite useful acquisitions
## An exact counterexample to acquisition sufficiency of predictive uncertainty

**Question:** Can predictive epistemic uncertainty alone tell us which additional information will reduce uncertainty about a fixed target?

**Construction:** two source models induce exactly the same two-member distribution over target predictions. Their scalar mutual information, probability support and lower/upper epistemic values all agree. Nevertheless, one benefits from source identities and the other from source-stratified target labels. We prove this algebraically, enumerate exact expectations, and verify the response with repeated acquisition draws.

This is a minimal analytic experiment, deliberately not another four-image-dataset benchmark. It removes optimization, model fitting, OOD rankings and representation-learning failures from the argument. Prior image notebooks remain unchanged.

**Claim boundary:** a function of the predictive representation alone cannot universally determine acquisition value. A Bayesian procedure given the full acquisition likelihood can distinguish the situations and update correctly; we implement those updates. This is not a claim that mutual information is numerically wrong, that all epistemic methods fail, or that intervals alone solve the problem.

### Sources and theoretical dependencies

- Hüllermeier & Waegeman (2021), [Aleatoric and epistemic uncertainty](https://doi.org/10.1007/s10994-021-05946-3): distinction between outcome randomness, uncertainty about predictive probabilities, and set-valued representations.
- [Probly](https://github.com/pwhofman/probly): independently check predictive mutual information on the actual categorical second-order representation. Record the installed version. No neural dropout sampling is needed because the two predictive members are exact.
- Hoeffding (1963), [Probability inequalities](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf): fixed-sample confidence-set illustration, clearly separated from posterior/epistemic intervals.
- Ni et al., [DAPPr](https://arxiv.org/abs/2605.00600): motivation for comparing predictive uncertainty representations. We do not assert that DAPPr training produces matched scores on these worlds, and do not fabricate neural logits to claim an empirical DAPPr failure.

**Derived directly below:** Bayes' rule, Bernoulli entropy, posterior variance as squared-error Bayes risk, the law of total variance, and the two-world acquisition-sufficiency counterexample. The counterexample is our construction for this study; no novelty claim against the literature is made.

The acquisition laws, priors and hypothesis set are specified, not learned from the four image datasets. Results are conditional on these choices. All sources, equations, code, draws, outputs and plots are saved under a fresh workbook directory.

### 1. Same target, different unknown mechanism

In both worlds there are two sources $S\in\{0,1\}$ and one observable input cell $X=0$. The target is the **deployment probability** $T=P(Y=1\mid X=0)=P(Y=1)$. It is the same estimand in both worlds, not two different uncertainty targets.

A latent hypothesis $H\in\{L,H\}$ selects $p_L=0.2$ or $p_H=0.8$. Initially $P(H=H)=q=0.5$.

| Quantity | World M: mixture unknown | World L: target rates unknown |
|---|---|---|
| Deployment source proportion $P(S=1)$ | $p_H$ or $p_L$ | Known $1/2$ |
| Source-0 target rate $P(Y=1\mid S=0)$ | Known 0 | $p_H$ or $p_L$ |
| Source-1 target rate $P(Y=1\mid S=1)$ | Known 1 | Same $p_H$ or $p_L$ |
| Observable input | Always $X=0$ | Always $X=0$ |
| Deployment target probability | $(1-p)0+p1=p$ | $(1/2)p+(1/2)p=p$ |

Therefore both worlds induce exactly

$$Q_q=(1-q)\delta_{\mathrm{Bern}(0.2)}+q\delta_{\mathrm{Bern}(0.8)}.$$

The worlds have different known structural assumptions. Those assumptions are available to an acquisition-aware rule, but absent from a rule restricted to $Q_q$. We do not claim the full parameter posteriors or full information states are identical.

**Fundamental:** projecting a parameter model to target predictions can erase which unknown parameters generated those predictions.

### 2. Equal predictive uncertainty, including upper/lower epistemic values

Let $h(p)=-p\log p-(1-p)\log(1-p)$. Predictive mutual information is

$$U(q)=h(0.2+0.6q)-(1-q)h(0.2)-qh(0.8).$$

At $q=1/2$, both worlds have mean prediction 0.5, probability support $\{0.2,0.8\}$, total predictive entropy $h(0.5)$, and epistemic MI $h(0.5)-h(0.2)$.

To represent uncertainty about the hypothesis weight, explicitly allow $q\in[0.25,0.75]$. This is a declared prior-sensitivity family, **not** a confidence set estimated from data. Define

$$[\underline U,\overline U]=[\min_q U(q),\max_q U(q)].$$

Concavity makes the minimum an endpoint value; symmetry makes the maximum occur at $q=1/2$. The resulting interval is identical in the two worlds. This directly demonstrates that merely replacing one scalar by upper/lower epistemic values does not supply the missing acquisition likelihood.

Below Probly checks the nominal MI. The interval calculation uses the explicit weighted MI formula, avoiding the installed version's known nonuniform-sample-weight path.

### 3. Specify what each acquisition actually observes

Each acquisition costs one observation; there are no hidden noisy source identities or unknown label-noise rates.

| Acquisition | World M | World L |
|---|---|---|
| Unlabeled input | $X=0$, same under both hypotheses | $X=0$, same under both hypotheses |
| Deployment source identity | $S\sim\mathrm{Bern}(p)$ | $S\sim\mathrm{Bern}(1/2)$, independent of H |
| Source-stratified reference label | Query source 0; $Y=0$, independent of H | Query source 0; $Y\sim\mathrm{Bern}(p)$ |

The reference source is fixed by the sampling design and known reliably. A reference label is not an unstratified deployment label: the latter would be Bernoulli(p) in both worlds and would help both. Distinguishing these channels is essential.

Bayes' rule updates the hypothesis weight after each acquisition. For an informative channel, $k$ successes in $n$ draws give

$$\operatorname{logit}(q_{n,k})=\operatorname{logit}(q)+k\log(0.8/0.2)+(n-k)\log(0.2/0.8).$$

For uninformative channels the likelihoods agree, so the posterior remains q. This is a correct update, not an artificially frozen baseline. The same evidence updates both endpoints of the prior-weight range using this monotone map.

### 4. Exact proposition: same predictive representation cannot determine action value

Use expected reduction in posterior variance of the **same target probability** T as acquisition value. Under squared error, the optimal estimate is the posterior mean and its Bayes risk is posterior variance. This is not the variance of the binary outcome Y.

Initially $\operatorname{Var}(T)=0.6^2q(1-q)$, which is 0.09 at q=1/2. For one informative Bernoulli observation, posterior weights become 0.8 or 0.2, with equal marginal probability. The remaining variance is $0.6^2(0.8)(0.2)=0.0576$. Hence expected risk reduction is **0.0324**. An uninformative observation leaves variance 0.09, so its reduction is **0**.

Thus source identity is strictly preferable in M; reference target label is strictly preferable in L. Both cost the same, and the predictive representation is exactly the same.

**Proof of insufficiency:** any deterministic policy that receives only Q must choose the same action in both worlds. It therefore cannot choose the uniquely best action in both. A randomized Q-only policy choosing identities with probability r incurs one-step regret $0.0324(1-r)$ in M and $0.0324r$ in L (ignoring inputs, which only worsen regret). Its worst-world regret is at least **0.0162**, attained at r=1/2. This proves a limitation of predictive-representation-only policies, not of policies given the full likelihoods.

**Stronger than scalar matching:** any scalar or interval that is solely a function of Q also agrees across worlds. This includes the nominal MI and prior-sensitive MI interval above. An arbitrary scalar could encode structural information, so we do not claim every possible scalar is insufficient.

**Transition:** enumerate exact acquisition outcomes to verify the proof and its behavior at larger budgets.

### 5. Repeated acquisitions: verify exact expectations and inspect interval behavior

Draw the true hypothesis once per replicate from the specified q=1/2 prior, then draw nested acquisition streams. All channels within that replicate share its hypothesis. The two informative channels use paired observations across worlds, so differences cannot be attributed to Monte Carlo noise.

For each budget, save posterior q, target mean, target variance, predictive MI, and lower/upper MI after updating the weight family. The expected endpoint curves average over the nominal q=1/2 observation law; they are **not robust expectations over all priors**. Individual MI realizations need not decrease at every step, even though expected posterior variance cannot increase under this correctly specified Bayesian experiment.

We also include a separate fixed-n confidence-set illustration for the true target T. Under an informative channel, retain p=0.2 or p=0.8 if its distance from the empirical success rate is at most sqrt(log(2/alpha)/(2n)). Hoeffding gives at least 1-alpha inclusion at each fixed budget. Under uninformative channels retain both. Empty sets are flagged as rejection with NaN endpoints; never count them as zero uncertainty. Means of these widths are reported **conditional on non-rejection**, alongside rejection and inclusion rates. No simultaneous-over-time guarantee is claimed.

Important distinction: Bayesian support still contains both hypotheses after any finite full-support Bernoulli sample (ignoring floating-point underflow). Its support width does not shrink. Posterior concentration, epistemic MI intervals and confidence-set widths are different objects.

### 6. Read the result as an acquisition counterexample

First figure: exact expected target variance and simulated means. The correct acquisition swaps between worlds while all initial predictive uncertainty agrees. Inputs never help because their law is identical under both hypotheses.

Second figure: updated lower/upper **epistemic MI**. The shaded region is the average of the specified prior-sensitive endpoints, not a confidence band or a range across replicates. The bounds alone were initially equal; only applying the distinct channel likelihoods reveals the different responses.

Third figure: confidence-set width, conditional on non-rejection. This is a different target-probability interval, included to connect to the earlier identification framework. See the saved inclusion/rejection rates before interpreting contraction.

### 7. Scientific conclusion and limits

**Established:** the predictive distribution over distributions can be identical while the best acquisition differs. Therefore its scalar epistemic summaries and upper/lower epistemic summaries are not universally sufficient for acquisition choice. The missing object is the relation between unknown parameters and each candidate observation channel.

**Not established:** that MI is an incorrect measure of predictive disagreement; that DAPPr training is defective; that all scalar functions or full-parameter Bayesian methods fail; that our prior-sensitive intervals are confidence bounds; or that a new detector beats another on images.

**Implication for our project:** the useful output is an acquisition-indexed uncertainty response, not merely a richer static uncertainty number. The earlier source-mixture identification analysis supplies precisely the type of observation-channel structure missing from a prediction-only summary. Intervals remain useful for expressing admissible uncertainty, but must be connected to what an acquisition observes.

The construction is stylized: the input is deliberately uninformative, hypotheses and priors are specified, the source-0 reference channel is available, identities are perfect, and acquisition costs are equal. These choices isolate the logical failure of universal sufficiency; they do not estimate how often the failure occurs in natural datasets.