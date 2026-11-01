# What predictive epistemic uncertainty does not characterize: an experiment review

Review of **twelve completed experiments**: the eleven originally requested notebooks and the controlled three-target characterization study. Prepared 30 October 2026; integrated revision 1 November 2026. All results come from saved notebook outputs and workbook artifacts, including experiment 12 executed on 30 October. **No experiments were rerun for this revision.** Numbers below are saved results, not new estimates. Paper recommendations are judgments about this evidence, not formal CVPR requirements or predictions of acceptance.

**Revised research scope.** This paper investigates a limitation of predictive epistemic uncertainty formulations through a controlled finite-cell source-mixture study. Its objective is to characterize conditions under which predictive representations do not determine uncertainty about three specified probability targets. Experiment 12 establishes separate controlled witnesses and sufficiency controls; the empirical objective is now to test these declared conditions in visual settings. Theory and empirical characterization are the contribution; a superior detector, acquisition policy or integrated visual-learning method is **not a requirement**. Acquisitions are experimental probes of reducibility, not necessarily decisions the paper must optimize.

The working thesis, with the scope needed to make it testable, is:

> **Predictive epistemic summaries need not determine the admissible uncertainty about known-source membership, a reference-anchored target probability, or specified candidate-novel membership. Characterizing these targets requires their identification structure and the information available to resolve them.**

**Evidence status:** experiment 12 now establishes controlled same-predictive-law witnesses for **each of the three targets**, with explicit sufficient-information controls. These are separate constructions within declared finite-hypothesis families, not a simultaneous independent decomposition. Membership witnesses use confounded source input laws; the reference-target witness requires reference/deployment mixture mismatch. When the reference target aligns with the prediction, its uncertainty is recoverable. Experiment 4 supports identification floors and acquisition responses; experiment 6 supplies the earlier single-prevalence mechanism. The controlled theoretical connection is therefore complete at this scope. The principal remaining empirical link is a direct, assumption-controlled comparison between predictive representations and three-target ambiguity in visual settings. Robustness and novelty also remain open; a superior acquisition policy is not required.

This revision retains all saved outcomes, including failed practical acquisition predictions. A failure to predict held-out error reduction is not automatically a failure of the characterization thesis; conversely, a positive acquisition result is not proof of that thesis. No general claim about all epistemic formulations, or that MI is an incorrect measure of predictive disagreement, is supported.

## 1. Scope, chronology and evidence provenance

The notebooks did not all test the same hypothesis. Their progression was:

**OOD warnings → within-dataset controls → intervals over predictive disagreement → source-mixture identification → actual DAPPr/Probly comparison → acquisition-insufficiency proof → real-record checks → image-domain pilots → exact three-target characterization with sufficiency controls.**

The first two notebooks were created on 28 October according to local filesystem metadata; their exact execution timestamps are not recorded. Later timestamps come from workbook names and are UTC. For concurrently started image/counterexample runs, start order is administrative rather than a causal research sequence. The substantive transitions below make that explicit.

| Order | Notebook | First available stage | Primary evidence used here |
|---|---|---|---|
| 1 | [Cross-dataset blind spots][n1] | 28 Oct; execution time not recorded | Saved notebook outputs; exported for this review |
| 2 | [Within-dataset blind spots][n2] | 28 Oct; execution time not recorded | Saved notebook outputs; exported for this review |
| 3 | [DAPPr-motivated epistemic sets][n3] | 28 Oct 10:21; repeated 29 Oct 08:55 | `dappr_epistemic_sets_seed0_20261029T085507170686Z` |
| 4 | [Source acquisitions on four datasets][n4] | 29 Oct 15:08; final archive 15:11 | `source_acquisition_20261029T151111775438Z` |
| 5 | [Probly probability-bound comparison][n5] | 29 Oct 18:27 | `probability_bounds_comparison_20261029T182738307854Z` |
| 6 | [Exact acquisition counterexample][n6] | 29 Oct 19:24 | `acquisition_counterexample_20261029T192454214626Z` |
| 7 | [Adult counterexample][n7] | 29 Oct 21:43; later tables 21:44 | Later tables, with earlier executed notebook as execution evidence |
| 8 | [Camelyon17 image acquisition][n8] | 30 Oct 09:20:23; diagnostic rerun 09:24 | `camelyon17_acquisition_20261030T092422959226Z` |
| 9 | [Camelyon17 counterexample][n9] | 30 Oct 09:20:24 | `camelyon17_counterexample_20261030T092024988890Z` |
| 10 | [FMoW counterexample][n10] | 30 Oct 09:49:29 | `fmow_counterexample_20261030T094929883846Z` |
| 11 | [FMoW image acquisition][n11] | 30 Oct 09:49:32 | `fmow_acquisition_20261030T094932286096Z` |
| 12 | [Controlled three-target characterization][n12] | 30 Oct 14:47:11; completed | `three_target_characterization_20261030T144711349004Z` |

**Archive qualifications.** Among the original eleven notebooks, contrary to the expectation of complete workbook exports, both blind-spot notebooks contain `SAVE_RESULTS=False`; their named export directories are absent. Their executed outputs do exist. I exported those existing PNGs and rendered tables into [review assets][extracts], retaining [notebook hashes and zero-based cell/output indices][extractindex]. These are saved-output extracts, not reconstructed raw per-image results. Some displayed tables are rounded or truncated.

The two weighted-MI runs have byte-identical contents in all ten common CSVs. The two source-acquisition runs likewise match in all ten common CSVs. Adult matches in nine of ten common CSVs; only the Monte Carlo Probly check differs. These folders are **not independent experimental replications**. Adult's later folder lacks `executed.ipynb`; the earlier [executed archive][adultexec] is available and its primary result tables match. The earlier Camelyon image run lacks the later drift/containment diagnostics, so the later archive is used. `fmow_acquisition_20261030T092834342530Z` contains setup material only and is not counted as a completed experiment.

## 2. What the different uncertainty objects mean

| Object | Definition or interpretation | What it does not establish |
|---|---|---|
| Predictive MI / dropout EU | Entropy of mean prediction minus mean predictive entropy | ID membership, calibration, or which acquisition is best |
| Weighted-MI interval | Min/max MI over a declared family of weights on fixed dropout predictors | Confidence bounds on “true epistemic uncertainty,” or missing hypotheses |
| MC probability envelope | Coordinate min/max class probability over sampled predictors | Population coverage or source identifiability |
| Source-mixture probability bounds | Event extrema over joint distributions compatible with declared evidence/model constraints | Automatic correctness under source-law shift or arbitrary unseen OOD |
| Beta posterior credible enclosure | A posterior-probability enclosure for population prevalence under a chosen model/prior | A support/identification interval, or an MI interval |
| Finite-hypothesis target support hull (experiment 12) | Target extrema over positive-posterior hypotheses in a declared finite model | A credible interval, confidence interval, or continuous-model identification region |
| Acquisition value | Expected reduction of a specified risk after a specified observation | Agreement with realized or held-out error under model mismatch |

In the source experiments, known-source membership, reference-anchored target probability and candidate-source membership are **different estimands**. Their widths are not an additive decomposition of epistemic uncertainty. In the prevalence counterexamples (6, 7, 9 and 10), both compared states concern the **same population prevalence target**, so differences cannot be explained merely by changing the estimand. Experiment 12 instead fixes each source target within its family and compares evidence states before and after learning U; its finite support hulls are distinguished from the continuous source-compatible regions used in experiment 4.

### The three targets and the characterization being tested

Write C=g(X) for the fixed image cell, S for the source, K for the known-source set, and N for the specified candidate source. The current three-source implementation uses K={0,1} and N=2. The targets are:

- **Known-source membership:** P(S=0 | C=c, S in K). This is which known source generated an example, conditional on known membership; it is not the total probability of being ID.
- **Reference-anchored target probability:** P_ref(Y=1 | C=c), with declared reference source weights. The later image studies use source-and-cell conditional target rates; the earlier simulation uses a simpler declared target mechanism.
- **Candidate-novel membership:** P(S=N | C=c). This concerns a specified candidate distribution; it is not membership in the unrestricted universe of unseen sources.

For each target t_j and evidence e, let M(e) be the declared compatible model set and A_j(e)=[inf over M(e) of t_j, sup over M(e) of t_j]. W_j(e) is its width. These are model-relative probability ranges. Let R(e) be the predictive representation or epistemic summary being evaluated. The research question is whether R(e), with explicitly stated shared auxiliary information, determines the relevant A_j(e), its exact-input identification floor, or its response to a declared additional observation channel.

An exact witness with **the same R but a different A_j** establishes failure of static target-ambiguity characterization. A witness with the same R but different uncertainty responses establishes failure of **reducibility characterization**. These are separate statements. Experiment 6 establishes the second kind for its prevalence target. Experiment 4 establishes three-target identification behavior. Experiment 12 now supplies exact same-R static witnesses for each target separately and enumerates finite and oracle channel responses. Its R is the full second-order law over joint cell/label predictions. Finite positive-likelihood observations can reduce posterior variance without changing support hulls; oracle information can shrink the hulls. Approximate MI matching is empirical evidence, not an exact equality proof or equality of the entire predictive law.

The comparison must declare what R contains and what it omits. A predictive label posterior is not generally designed to recover source identity; the scientifically useful contribution is the precise condition under which interpreting it as a broader epistemic characterization loses relevant information. It would be a strawman to assume every uncertainty formulation promises those source targets. Likewise, bounds computed from additional references cannot be credited to a better representation without identifying the extra information supplied.

OOD also changes meaning: stages 1–3 and 5 largely concern unseen datasets or held-out classes; stages 8 and 11 concern hospital/region novelty with overlapping target classes. Adult is a recorded-stratum study, not a vision OOD benchmark. The source model's candidate is specified; it does not represent every possible unknown distribution.

## 3. Chronological experiment review

### 1 — Cross-dataset blind spots: is low EU necessarily a missed warning?

**Notebook:** [eu_blindspot_mnist_emnist_fashion_blood.ipynb][n1].

**Reason and expected finding.** The initial motivation was to inspect whether MC-dropout epistemic uncertainty warns about unfamiliar inputs. MNIST digits were known classes; EMNIST letters trained a learned eleventh “unknown” output; Fashion and Blood were unexposed datasets. A frozen MNIST encoder and a dropout head made this an inexpensive exploratory check.

**Statement actually tested.** Do EU, entropy and the learned unknown probability distinguish those evaluation groups, and how often do unfamiliar examples receive confident known-digit predictions? Low EU alone is not the failure event: a confident prediction of the **unknown output** can be a successful rejection.

**Main results.** MNIST digit accuracy was 94.35%. Blood had mean EU 0.0067 and EU OOD AUROC **0.0708**, yet unknown-output rejection accuracy was **100%**, unknown-probability AUROC **0.9998**, and the silent known-prediction rate **0%**. Fashion had EU AUROC **0.4446**, unknown-probability AUROC **0.9890**, and silent known predictions on **1.35%** of examples. EMNIST rejection was 87.5%, but EMNIST had been exposed during training.

**What follows.** Low predictive disagreement does not certify familiarity. However, this run does not show that the entire classifier failed on Blood: its explicit unknown output worked. It also does not establish a population-level failure of epistemic uncertainty or an acquisition limitation. Grayscale Blood through a digit encoder is a substantial representation mismatch.

**Transition.** To distinguish genuine withheld-class failures from cross-dataset representation effects and successful unknown rejection, the next experiment trained separate models within each dataset.

**Evidence:** [role-wise outcomes][e1summary], [detection table][e1detect], [uncertainty quadrants][e1fig], [example panels][e1examples]. **Paper role:** historical motivation at most; exclude as central evidence that standard EU or the classifier “fails.”

### 2 — Within-dataset blind spots: can EU miss confident errors after exposure controls?

**Notebook:** [eu_blindspot_within_datasets.ipynb][n2].

**Reason and expected finding.** Each dataset received its own encoder, known classes, exposed unknown classes, and wholly withheld unknown classes. The aim was to test whether silent errors persisted after removing the frozen-digit-encoder confound. Validation-only known-class quantiles set the entropy/EU thresholds.

**Statement actually tested.** Do withheld-class examples receive confident known-class predictions without an EU warning? Secondary diagnostics examined feature perturbations, exact linear-boundary crossings and training duration. These concern sensitivity and representation behavior, not identification or acquisition value.

| Dataset | Known-class test accuracy | Withheld-class EU AUROC | Silent confident known predictions on withheld classes |
|---|---:|---:|---:|
| MNIST | 0.8900 | 0.8090 | 18.88% |
| EMNIST | 0.7550 | 0.6111 | 33.50% |
| Fashion | 0.8912 | 0.6796 | 59.62% |
| Blood | 0.8067 | 0.3883 | 76.83% |

EU flagged only about 8.6–13.2% of the confident known-class error subset, depending on dataset. In the exact linear diagnostic, crossing sufficiently far beyond the deterministic class boundary drove accuracy to zero while EU barely changed for several datasets and decreased for Blood. These are **feature-space** diagnostics; they are not demonstrations that corresponding perturbed natural images exist. The duration study changed MNIST EU AUROC from about 0.257 to 0.820 between short and longer training; Blood moved in the opposite direction. The optional bootstrap-ensemble comparison was explicitly skipped.

**What follows.** This is better evidence that a particular dropout warning rule can miss errors, but it remains a one-seed pilot with representation/training dependence. Fashion used a repartition of the cached official test pool, not an official benchmark evaluation. Row-bootstrap intervals condition on one fitted model and do not replace repeated training/class-split studies. The missing raw export further limits paper readiness.

**Transition.** The next branch asked whether expressing disagreement as an interval, rather than a point score, would improve characterization or detection.

**Evidence:** [role-wise table][e2summary], [detection/per-class outcomes][e2detect], [warning-coverage table][e2warnings], [quadrants][e2fig], [feature attacks][e2attacks], [linear boundary table][e2linear], [boundary plot][e2linearfig], [training-duration table][e2duration]. **Paper role:** optional motivation/appendix after stronger replication; feature attacks and duration sweeps are peripheral to the acquisition argument.

### 3 — Weighted epistemic sets: does an MI interval improve OOD detection?

**Notebook:** [dappr_epistemic_sets.ipynb][n3].

**Reason and expected finding.** The first direct interval proposal replaced a single equal-weight dropout MI with lower/upper MI over weights `w=(1−rho)/T + rho*q`, where q lies in the simplex. The hoped-for benefit was more informative uncertainty characterization and possibly better ID/OOD separation. The saved pilot used rho=0.5 and 20 passes.

**Statement actually tested.** How sensitive is MI to weighting a fixed set of predictors, and does its interval midpoint improve scalar OOD ranking? Here DAPPr supplied training/evaluation workflow conventions. Its actual possibilistic loss was **not** used: training was cross-entropy with dropout. This stage is not a DAPPr reproduction or an empirical replacement of its full method.

| Dataset | Ordinary dropout MI AUROC | Weighted-MI midpoint AUROC |
|---|---:|---:|
| MNIST | 0.8974 | 0.8815 |
| EMNIST | 0.8134 | 0.7943 |
| Fashion | 0.6928 | 0.6675 |
| Blood | 0.4822 | 0.4937 |

There is no consistent improvement. Blood remains near chance. Sixty percent of Blood OOD examples were below the chosen threshold for **all allowed weights**. Wider intervals did not create predictors outside the sampled family. The upper endpoint retained an outer optimization bound: 33/5,400 rows exceeded the requested 1e−5-nat gap, with a maximum around 1.91e−4 nats. Numerical optimization gap is distinct from epistemic width.

The installed Probly sample-MI path ignored supplied weights; the experiment therefore computed weighted MI explicitly using its entropy primitive. This is an implementation audit, not a conceptual failure of the uncertainty definition.

**Transition.** Reweighting predictive samples cannot explain which external information resolves uncertainty. The next experiment returned to the source-mixture question and made observation channels explicit.

**Evidence:** [metrics][e3metrics], [threshold decisions][e3decisions], [per-image epistemic intervals][e3fig], [rho sensitivity][e3rho], [per-row gaps/scores][e3scores], [configuration][e3config]. **Paper role:** a concise negative control in the supplement; exclude claims that the interval midpoint improves detection or solves acquisition choice.

### 4 — Source acquisitions: distinguish finite evidence from non-identification

**Notebook:** [source_acquisition_four_datasets.ipynb][n4].

**Reason and expected finding.** This experiment operationalized “reducible with additional information—but which information?” It sought separate responses for known membership, a reference-anchored target probability and candidate membership. The expectation was that inputs help distinguishable mixtures, labels help the reference target, and identities can resolve ambiguity that inputs cannot.

**Statement actually tested.** Within a declared finite-cell mixture, do exact input laws identify the target, and how do finite observation channels change its compatible interval? Cached image predictions supplied source histograms. Subsequent acquisitions were **simulated categorical observations**, not newly collected images. The model specified 2% uniform-cell smoothing, target rates 0.2/0.8, label-noise rate 0.1, deployment weights 0.45/0.25/0.30, and a known candidate law. Confounding was deliberately constructed by making the candidate law the average of the known laws.

| Dataset | Exact input-only candidate width, confounded | Mean width after 100,000 extra inputs | Mean width after 100,000 source identities |
|---|---:|---:|---:|
| MNIST | 0.6732 | 0.7071 | 0.1280 |
| EMNIST | 0.6758 | 0.7925 | 0.5017 |
| Fashion | 0.6710 | 0.7145 | 0.1273 |
| Blood | 0.6775 | 0.7319 | 0.0820 |

In the image-derived, affinely distinguishable regime, exact-input membership widths collapse numerically to zero. Under constructed confounding they remain positive, and finite identity acquisition crosses the exact input-only candidate floor in all four worlds. Exact rows plus exact weights identify membership. Without target labels, the separate anchored target remains [0,1]; exact target rates identify it.

All 2,880 saved finite intervals included the declared truth; there were no empty sets, and undefined conditioning events were separately recorded. Six paired repeats do not establish precise empirical calibration. Some invariances follow directly from the chosen channel likelihoods and fixed anchor; they should be described as consequences of the model, not discovered universal categories of EU. Reference inputs can affect several targets. Reference/label x-axis budgets are **per known source**, with twice that total acquisition cost.

**Transition.** This study answered identification/acquisition questions but did not implement the user's proposed DAPPr/Probly comparison on actual predictions. That mismatch in purpose motivated stage 5.

**Evidence:** [population/oracle intervals][e4oracle], [acquisition summaries][e4summary], [configuration][e4config], [inclusion and solver checks][e4checks], [source geometry][e4geometry], [candidate endpoints][e4fig], [MNIST curves][e4mnist], [EMNIST curves][e4emnist], [Fashion curves][e4fashion], [Blood curves][e4blood], [theoretical sources][e4sources]. **Paper role:** strong controlled support for the mechanism, explicitly labeled simulation. One representative plot plus all four numerical settings is preferable to presenting this as four independent real-world validations.

### 5 — Actual probability wrappers and DAPPr loss: representation does not create information

**Notebook:** [probly_probability_bounds_four_datasets.ipynb][n5].

**Reason and expected finding.** This stage implemented both requested interpretations of lower/upper probabilities: an envelope of MC predictions and an observation-compatible source-mixture set. It added an actual DAPPr-loss baseline to remove the ambiguity around “workflow motivation only.”

**Statement actually tested.** Do these probability-width representations separate within-dataset ID/OOD classes, and are the source constraints informative on the saved real images? The methods target different events: predictive class probability versus known-class/source membership under a mixture model. Their scalar widths are not automatically interchangeable estimates of epistemic MI.

| Dataset | MC probability-width AUROC | Source-width AUROC | DAPPr original-EU AUROC |
|---|---:|---:|---:|
| MNIST | 0.9306 | 0.5000 | 0.8748 |
| EMNIST | 0.7909 | 0.5000 | 0.7467 |
| Fashion | 0.7916 | 0.5000 | 0.7299 |
| Blood | 0.4486 | 0.5000 | 0.4480 |

The maximum source probability width was **exactly 1 for every test image**. There were 43,516 class-event rows assigned [0,1] because a conditioning denominator could vanish. In EMNIST and Blood an all-candidate model was compatible with the marginal box; in other datasets zero known mass remained possible at particular cells. This is a substantive lack of informativeness, not a successful characterization result disguised by wide intervals.

Tie-correct evaluation matters: the installed Probly ROC implementation could report AUROC 1 for constant scores where the correct value is 0.5. Corrected and raw results are retained. Source inference had candidate-reference access and the unlabeled test histogram; the comparison is not equal-information method superiority. The DAPPr branch is a small-CNN adaptation, not its original benchmark reproduction.

**Transition.** Detector rankings and vacuous bounds did not directly test the central claim. We therefore moved to a counterexample that holds the predictive uncertainty representation fixed and changes what each acquisition reveals.

**Evidence:** [OOD table][e5metrics], [bound summaries][e5bounds], [source-informativeness audit][e5audit], [comparison figure][e5fig], [per-image probability intervals][e5images], [MC sample-count sensitivity][e5mc], [result audit][e5results]. **Paper role:** important negative control/limitations material; exclude any claim of source-width detection improvement or DAPPr's general failure.

### 6 — Exact counterexample: identical predictive uncertainty, opposite useful information

**Notebook:** [acquisition_counterexample.ipynb][n6].

**Reason and expected finding.** The desired claim concerns reducibility, so this experiment removed neural fitting and held the **entire predictive second-order law**, not just one scalar, exactly fixed. It asked whether any rule restricted to that representation can always choose the best acquisition.

**Construction and statement.** Both worlds have target T=P(Y=1), an uninformative constant input, and equal prior weight on Bernoulli(0.2) and Bernoulli(0.8). In world M the unknown quantity is the source proportion, with source target rates 0/1. In world L the source proportion is known 1/2, while both source target rates share the unknown value 0.2/0.8. A representative source identity is informative only in M; a **source-0 reference label** is informative only in L. An unstratified deployment label would help both, so the acquisition definition is essential. Full parameter states are different; only their projection to predictions is identical.

| Quantity | Both worlds / exact result |
|---|---:|
| Initial predictive MI | 0.192744757 nats |
| MI range over prior weights [0.25,0.75] | [0.147044215, 0.192744757] nats |
| Initial target posterior variance | 0.09 |
| Expected variance after one informative acquisition | 0.0576 |
| Informative action value | 0.0324 |
| Worst-world regret lower bound for a representation-only randomized action | 0.0162 |

Exact enumeration and 10,000 paired repetitions agree with the mechanism. Correct Bayesian updates using the actual channel likelihood succeed; the baseline is not artificially frozen. The support interval for target probability and the prior-sensitive MI interval are different objects.

**What follows.** Predictive-representation-only acquisition choice is not universally sufficient. This applies to scalar **and static interval** summaries of the shared predictive law. It does not refute MI as predictive disagreement, show that every scalar function fails, or show that full Bayesian experimental design fails. It also means our own upper/lower epistemic summary is not a complete repair: observation-channel structure is indispensable.

**Transition.** A toy proof establishes possibility, not empirical prevalence or practical relevance. We next tested a near-matched analogue using genuine recorded source/target fields.

**Evidence:** [proof and protocol][e6proof], [identical initial uncertainty][e6initial], [exact action values][e6values], [regret table][e6regret], [risk curves][e6fig], [epistemic intervals][e6intervalfig], [regret plot][e6regretfig]. **Paper role:** strong logical mechanism for loss of reducibility information, subject to a separate novelty audit. Its single-target result must be connected to the three-target thesis; it is not itself a complete three-target characterization. Mathematical validity alone does not establish research novelty.

### 7 — Adult: real recorded fields, matched MI, a horizon-dependent result

**Notebook:** [adult_acquisition_counterexample.ipynb][n7].

**Reason and expected finding.** The cached image datasets lacked a separate recorded domain field. Adult provides workclass and income as distinct recorded variables, enabling a real-record analogue without inventing source identities or target labels.

**Statement actually tested.** In a Beta source-mixture model for unweighted income-above-50K prevalence, can training states with similar MI have opposite useful acquisitions on official held-out records? Private/public workclass defines source strata. Source IDs update the mixture weight; balanced source-stratified labels update income rates. Other census features are excluded. This is controlled field revelation and empirical resampling, not a causal study or population-weighted census estimate.

The search examined 600 states and found 60 qualifying pairs. The selected MI discrepancy was **0.001136%**, predictive-mean difference 0.004389 and variance difference 1.238%. Exact held-out one-step MSE favored identities by **5.0023e−6** in the identity-preferred state and labels by **2.22295e−5** in the label-preferred state: the declared one-step endpoint passed.

**Important horizon qualification.** That does not establish a persistent advantage across budgets. In the saved 4,000-repeat trajectories, the identity-preferred state's mean `identity error − label error` is **+0.0002817 at budget 32** and **+0.0011318 at budget 128**: labels are better at those budgets. Its one-step Monte Carlo contrast is tiny relative to simulation error; the exact enumerated one-step calculation is the primary endpoint. The label-preferred state continues to favor labels. An action's value depends on the horizon as well as the state and channel.

**Transition.** This is a transparent real-record bridge, but it is not computer vision. The next pair of studies introduced genuine hospital domains and actual image features, while preserving the distinction between field-based counterexamples and image prediction.

**Evidence:** [selected states][e7states], [exact held-out endpoint][e7exact], [paired contrasts][e7contrasts], [full search][e7search], [acquisition curves][e7fig], [saved report][e7results]. **Paper role:** useful explanatory supplement; not the main CVPR empirical result and not evidence of a horizon-independent acquisition policy.

### 8 — Camelyon17 images: domain metadata and target-specific bounds under patient shift

**Notebook:** [camelyon17_acquisition.ipynb][n8]. The companion counterexample began almost simultaneously; the added diagnostic run supplies the final image evidence here.

**Reason and expected finding.** Hospital provenance and tumor labels give genuinely distinct source and target fields. The goal was to move the representation/bounds pipeline to a relevant visual domain and ask whether acquisition responses remain informative when reference and evaluation images come from different patients.

**Statement actually tested.** Known hospitals 0/3 and candidate hospital 2 define domain ID/OOD. A custom patient-disjoint split, 18,000 selected images, frozen ImageNet ResNet18, matched CE/DAPPr heads and 30 MC passes form a pilot. Four KMeans cells fitted on known-source fit features define the source model. This is not the official WILDS protocol or an end-to-end DAPPr reproduction.

MC MI OOD AUROC was **0.6897**, MC width **0.6853**, and DAPPr EU **0.5973**. At 2,048 additional source IDs, mean known-membership width decreased **1.000→0.823** and candidate width **0.986→0.721**. Target-label acquisition reduced reference-target width **0.859→0.623**. These changes are responses of different estimands to different channels, not an additive decomposition or demonstrated improvement over neural uncertainty.

**Critical limitation.** Source-cell total-variation differences between reference and evaluation were **0.3115, 0.1995 and 0.3635**. With 2,048 IDs, known-membership intervals contained the held-out empirical target in only **17/20 cell/repeat checks (85%)**. This descriptive diagnostic is not a population coverage estimate, but it shows that narrowing can coexist with missed targets. No set was empty; 30.5% of all queries used the explicit [0,1] denominator policy. Compatibility therefore did not establish source-model correctness.

The evaluation split had only **one, two and one patients** in hospitals 0, 3 and 2. Thousands of image patches do not remove this limitation. In addition, source identities were acquired as marginal source counts, not exploited as paired image/source observations; the resulting intervals are not the best possible use of a richer audit channel.

**Transition.** The image run demonstrates feasibility and exposes model-transfer problems. The companion study asks a narrower question: does the recorded-field acquisition counterexample pass on the same dataset family?

**Evidence:** [OOD metrics][e8metrics], [acquisition table][e8summary], [bound curves][e8fig], [source-law drift][e8drift], [empirical containment][e8coverage], [patient counts][e8groups], [event/budget definitions][e8definitions]. **Paper role:** valuable honest stress test of source-law assumptions; not strong positive evidence of a calibrated acquisition method. Retain the failure if using the positive counterexample from this dataset.

### 9 — Camelyon17 recorded-field counterexample: positive but selected and small-group

**Notebook:** [camelyon17_counterexample.ipynb][n9].

**Reason and expected finding.** This transports the Adult near-match design to genuine hospital/tumor fields, with whole-patient separation. The expected finding was opposite useful source-ID versus target-label acquisitions despite nearly equal predictive MI for the same tumor-prevalence target.

The search found **39 qualifying pairs among 600 states**. Selected means were 0.578622 and 0.578479; MI differed by **0.0170009%**. Exact held-out MSE after one identity/label was **0.04568537/0.04629771** for the identity-preferred state and **0.04633004/0.04577468** for the label-preferred state. Both declared advantages were positive. Saved 2,000-repeat trajectories retain the corresponding ordering through budget 128.

**What follows.** This is a selected real-record analogue under the Beta model, not the exact equality theorem, an estimate of occurrence frequency, or evidence that a neural score fails. It models hospital and tumor fields **without conditioning on image pixels**. The two known-source evaluation pools contain only three patients in total. The initial held-out MSE around 0.046 also shows that similar predictive summaries can coexist with substantial estimation bias. Credible target enclosures here are not source-identification intervals or calibrated upper/lower MI.

**Transition.** A second visual domain was needed to check whether the empirical acquisition story survives a different source/label structure. FMoW provides region and land-use labels with many more location groups.

**Evidence:** [selected states][e9states], [held-out MSE table][e9exact], [acquisition curves][e9fig], [population rates][e9rates], [credible enclosures][e9credible], [patient counts][e9groups]. **Paper role:** supporting single-target reducibility pilot rather than the main three-target characterization. If included, it must be accompanied by the FMoW failure and the image-model limitations.

### 10 — FMoW recorded-field counterexample: a failed prespecified confirmation

**Notebook:** [fmow_counterexample.ipynb][n10].

**Reason and expected finding.** The same protocol was applied to Americas/Europe source strata and the modal known-source training land-use class (class 42) as a binary event. Asia is the image experiment's candidate region; the two-source counterexample uses the known regions. Location groups are separated; pre-2016 records define this pilot rather than the official WILDS temporal task.

The search found **32 qualifying pairs among 600 states**, with selected MI discrepancy **0.0643531%**. However, the held-out comparison contradicted both model-selected preferences:

| Model-selected state | MSE after identity | MSE after target label | Empirically better action |
|---|---:|---:|---|
| Identity-preferred | 0.00019807175 | 0.00017691726 | Label |
| Label-preferred | 0.00002065736 | 0.00002140581 | Identity |

**What follows.** The prespecified endpoint **failed**. The empirical best actions still differ, but relabeling states after observing held-out outcomes would turn a failed directional test into a post-hoc claim. We do not do that. Posterior expected variance reduction is not the same quantity as held-out MSE reduction; the latter depends on the empirical truth and initial bias. MI matching alone controls neither. The initial MSEs also differ substantially between these near-matched states.

No thresholds were relaxed, no alternative pair was selected using held-out outcomes, and this outcome does not invalidate the exact toy theorem. It does invalidate the claim that the current real-record protocol consistently predicts which information will improve held-out estimation.

**Transition.** The companion image study checks which narrower acquisition and detection statements remain supported even when this directional counterexample confirmation fails.

**Evidence:** [selected states][e10states], [exact held-out comparison][e10exact], [acquisition curves][e10fig], [population rates][e10rates], [protocol][e10protocol], [result JSON][e10result]. **Paper role:** boundary/negative result for the secondary practical-risk endpoint if the real-record studies are presented. This failed endpoint does not by itself falsify the three-target characterization thesis. Exclude it as positive confirmation; do not exclude it from the evidence record.

### 11 — FMoW images: more stable cell laws, still broad membership intervals

**Notebook:** [fmow_acquisition.ipynb][n11].

**Reason and expected finding.** This tests the image representation and source-bound pipeline in geographic domains, using recorded metadata and location-disjoint groups. The downloaded mirror has 32 Parquet files and 141,696 rows; the pilot selects 18,000 images and uses the same frozen-encoder/30-epoch/30-pass design as Camelyon.

MC MI OOD AUROC was **0.5737**, MC width **0.5458**, and DAPPr EU **0.5605**. These are weak-to-modest separation results, not a detector superiority story. At 2,048 additional source IDs, known-membership width decreased **0.999→0.884** and candidate width **0.987→0.860**. Target-label acquisition reduced the reference-target width **0.778→0.324**. Additional inputs alone barely changed membership widths in this configuration.

Reference/evaluation source-cell TV differences were **0.0135, 0.0125 and 0.0145**, much smaller than Camelyon's. Every evaluated interval contained its corresponding fixed empirical target, but broad intervals make this compatible with weak informativeness; it is not a population calibration guarantee. No sets were empty, and 11.67% of all queries used the zero-denominator [0,1] policy.

**What follows.** Distinct channels produce distinct target responses in a real-image-derived representation. The strongest contraction is for the reference-anchored target, and much of that behavior follows from the model's explicit channel design. The remaining gap in this visual experiment is that predictive summaries and all three target ranges have not yet been compared on aligned information states to isolate the proposed limitation. Multiple group/training repetitions and alternative cell partitions are also missing. Policy superiority, official temporal-benchmark performance and end-to-end fine-tuning are not requirements of the present thesis.

**Transition.** The unresolved theoretical connection between these visual width studies and the same-predictive-law argument motivated experiment 12. That controlled study establishes the conditions to test next in visual data; it does not retroactively validate these image pilots.

**Evidence:** [OOD metrics][e11metrics], [acquisition summaries][e11summary], [bound curves][e11fig], [source-law drift][e11drift], [empirical containment][e11coverage], [accuracy][e11accuracy], [split counts][e11groups]. **Paper role:** directly relevant pilot evidence for target-specific characterization under a better-behaved source map; needs aligned summary-to-target comparisons and robustness checks to support the full vision claim.

### 12 — Controlled three-target characterization: what can the same predictive law conceal?

**Notebook:** [three_target_characterization.ipynb][n12]. **Status: completed.** [Formal derivations][e12proof] and [executed result summary][e12results].

**Reason and expected finding.** The preceding source studies characterized three-target widths, while the exact counterexample concerned one prevalence target. This experiment closes that controlled theoretical gap by seeking same-predictive-law witnesses for each target and testing conditions where the representation is sufficient. No neural retraining or earlier experiment rerun was needed.

**Statement actually tested.** Can a full second-order law over joint cell/label predictions determine the ambiguity of each declared target and its response to further information? This is a characterization test, not an OOD-ranking contest or a claim of acquisition-policy superiority. Four controlled finite-hypothesis families compare an unresolved state with both possible states after exact structural information U is revealed. All twelve states have the same full predictive law, not merely approximately matching scalar MI. Predictive MI is 0.020135513551 nats; an independent Probly calculation agrees.

**Main results.**

| Family / target | Unresolved support hull | After oracle U=0 | After oracle U=1 |
|---|---|---|---|
| Known-source membership | [0.25, 0.75] | [0.25, 0.25] | [0.75, 0.75] |
| Specified candidate membership | [0.20, 0.60] | [0.20, 0.20] | [0.60, 0.60] |
| Reference target, mismatched mixture | [0.30, 0.70] | [0.50, 0.70] | [0.30, 0.50] |
| Reference target, aligned control | [0.40, 0.60] | [0.40, 0.60] | [0.40, 0.60] |

See the [interval figure](three_target_characterization_20261030T144711349004Z/target_intervals.png), [complete target table](three_target_characterization_20261030T144711349004Z/target_intervals.csv), and [predictive summaries](three_target_characterization_20261030T144711349004Z/predictive_summaries.csv). The membership failures require confounded input laws; the reference-target failure requires reference/deployment mismatch. These are separate witnesses, not an additive decomposition or proof of simultaneous independent changes in all three targets. Bounds computed solely from the same predictive representation remain identical and cannot recover this discarded structural information.

**Reducibility and its measurement.** A single source-identity observation reduces expected known-membership variance by 0.0125 and candidate-membership variance by 0.00666667 in their respective unresolved families, while preserving the entire predictive law for every possible observed identity. These finite observations have positive likelihood under every hypothesis, so support widths remain unchanged. Oracle channels can eliminate hypotheses and shrink support widths. The [response figure](three_target_characterization_20261030T144711349004Z/acquisition_responses.png), [finite-observation table](three_target_characterization_20261030T144711349004Z/finite_acquisition_responses.csv), and [oracle table](three_target_characterization_20261030T144711349004Z/oracle_acquisition_responses.csv) deliberately separate these quantities; oracle information is not a finite equal-cost acquisition.

**Controls and validation.** In the aligned-reference family the target law and deployment-label response are recoverable from the predictive law. In a second control, known affinely independent input laws plus an exact input marginal identify source membership: all [eight LP control checks](three_target_characterization_20261030T144711349004Z/identifiable_source_controls.csv) give point intervals at truth. Six independent [membership LP certificates](three_target_characterization_20261030T144711349004Z/membership_lp_checks.csv) agree with analytic extrema. The notebook executes without errors and 13 relevant unit tests pass. Rational arithmetic verifies the exact deployment-law cancellation; numerical calculations check the resulting quantities.

**Paper consequence.** The controlled formal step now has evidence for all three named targets, with explicit conditions and positive controls. It supports the statement that a predictive representation need not characterize uncertainty about a different declared target. It does not show that predictive MI is incorrect for its own target, establish novelty over sufficiency/identification theory, or establish a trained DAPPr failure. The remaining principal work is image-conditioned, target-matched empirical characterization under estimated source laws, including transfer failures, containment and representation/discretization sensitivity. The paper's central thesis remains characterization; a superior acquisition policy is not a prerequisite.

**Transition and paper role.** Use this as the main controlled theorem-and-witness study, with experiment 4 supplying complementary continuous-set identification results and experiment 6 supplying the earlier acquisition mechanism. Next, test the declared conditions in image-conditioned information states; do not rerun the completed formal step or demand universal failure in the aligned control.

## 4. Reassessing the experiments under the characterization thesis

The criterion is now **whether an experiment isolates or validates a characterization condition**, not whether it improves detection or chooses an optimal action. Historical motivations in section 3 remain historical; this reinterpretation does not retroactively change what those runs tested.

| Experiment | Relevance to the agreed thesis | Recommended use | What not to infer |
|---|---|---|---|
| 1. Cross-dataset blind spots | Low disagreement and successful rejection can coexist; uncertainty and novelty are different | Omit from main results; optional motivation | Blood classifier failure or a three-target characterization theorem |
| 2. Within-dataset blind spots | Particular dropout warning rules miss withheld-class errors | Optional motivating appendix | That error detection establishes source identifiability or reducibility |
| 3. Weighted-MI sets | Static disagreement intervals remain conditional on a fixed predictive family | Supplementary negative control, or a brief bridge to the counterexample | Interval midpoint superiority or a universal failure of interval methods |
| 4. Source acquisitions | **Direct evidence for three-target identification limits and distinct information responses** | **Primary controlled characterization experiment** | Natural-image generalization from simulated draws; three independent/additive epistemic components |
| 5. Probly/DAPPr comparison | Shows how conservative sets and undefined conditioning can make a representation uninformative | Negative control/implementation audit | A successful source-width detector; a general defect in DAPPr |
| 6. Exact counterexample | **Proves that identical predictive representations can hide different reducibility mechanisms** | **Theoretical mechanism**, explicitly scoped to its prevalence target | A completed theorem for all three source targets, or incorrectness of MI |
| 7. Adult | Real-record analogue; useful response depends on state and horizon | Explanatory supplement | A necessary main vision result or a persistent optimal acquisition policy |
| 8. Camelyon image acquisition | **Visual test of target-specific responses and failure of source-law transfer** | Core empirical diagnostic candidate, with group/split limitations | Population calibration or method superiority from shrinking widths |
| 9. Camelyon counterexample | Positive selected analogue of the single-target reducibility mechanism | Supporting pilot, paired with its limitations and FMoW outcome | That a pixel-conditioned predictive representation has been shown insufficient |
| 10. FMoW counterexample | **Separates posterior risk prediction from held-out estimation behavior** | Retain as a boundary result if the real-record studies are included | That its failed directional endpoint falsifies the central characterization thesis, or can be relabeled as a success |
| 11. FMoW image acquisition | **Visual three-target response study with smaller observed source-law drift** | Core empirical diagnostic candidate; strengthen direct summary-to-target comparison | That broad empirical containment proves calibration or useful identification |
| 12. Controlled three-target characterization | **Exact same-law witnesses for each target, channel responses and sufficiency controls** | **Primary theoretical characterization study; include in main paper** | A universal failure claim, simultaneous independent components, natural-image validation, or a trained DAPPr failure |

The main theoretical focus is now **stage 12**, supported by stage 6. The main empirical focus remains **stage 4 and the strengthened diagnostic versions of stages 8 and 11**. Stage 6 supplies the earlier single-target information-loss mechanism. Stages 7, 9 and 10 are supporting single-target studies, not the mandatory endpoint of the paper. They must not replace a direct three-target comparison.

“Exclude” means exclude an unsupported claim or a peripheral branch from the main argument. It does not mean suppress negative evidence. If the paper uses the Camelyon real-record success, the corresponding FMoW failure must remain visible. If it uses the source-bound image curves, it must include their vacuity, conditioning and transfer diagnostics.

The [22 October roadmap][roadmap] remains useful for related-work and source-model concerns, but parts of its method-development agenda are optional under this clarified thesis. This report's characterization scope takes precedence over the earlier demand for a superior acquisition policy.

## 5. What evidence is necessary—and what is sufficient for the stated claims?

### 5.1 The paper narrative, corrected

1. **Define the characterization problem.** Identify the three estimands, the evidence available about each, the predictive summaries being tested and the exact meaning of “fails to reflect.” Distinguish static ambiguity, exact identification limits and response to additional information.
2. **Establish the information-loss mechanism.** Use experiment 12 to show that the same full predictive law can coexist with different support hulls for each source target under its declared conditions. Include the aligned-reference and identifiable-source controls in the central result. Experiment 6 remains the earlier single-prevalence acquisition mechanism; its policy restriction and equal-cost assumptions apply to that result.
3. **Characterize the finite-cell source model.** State when each target is identified, when ambiguity survives exact input laws, and what observation channels can resolve it. Separate population identification, finite evidence, representation coarsening and misspecification.
4. **Validate the characterization on visual data.** Test whether predictive summaries distinguish states with different target-specific ambiguity, identification limits or acquisition responses. **This is the principal incomplete bridge.** The required result is a controlled relationship or counterexample, not a superior acquisition decision or an integrated learning method.
5. **Map the boundary of the claim.** Include settings where the summaries are informative or sufficient, where the proposed bounds are vacuous, and where transfer assumptions fail. Preserve Camelyon containment misses and distinguish FMoW's failed practical-risk prediction from the core information-loss result.

### 5.2 Audit of what is already supported

This table assesses the saved evidence, not the likelihood of conference acceptance. “Partial” means there is relevant evidence but it does not yet establish the complete claim.

| Claim needed for the paper | Evidence now | Assessment | Remaining condition |
|---|---|---|---|
| The three source targets have distinct information requirements | Stage 4 oracle intervals and acquisition sweeps; stages 8/11 visual widths | **Supported within the declared models** | State the estimands, fixed anchor and observation likelihoods; do not present model-imposed invariances as universal discoveries |
| Input information can be insufficient even when its law is known exactly | Constructed-confounding oracle intervals in stage 4 | **Supported under constructed confounding** | Give analytical conditions and identifiable controls; distinguish the finite cell map from raw-image information |
| A predictive representation can hide reducibility information | Stage 6 exact construction; stage 12 exact finite/oracle channel responses | **Supported within the specified models** | Distinguish posterior-variance reduction from support-width reduction and finite observations from oracle information |
| Predictive representations can fail to determine ambiguity about each of the three targets under stated conditions | Stage 12: twelve states with identical full predictive laws, separate target witnesses and positive controls | **Established in the controlled finite-hypothesis model** | Preserve confounding and reference-mismatch conditions; separate witnesses are not a simultaneous decomposition or empirical vision validation |
| The limitation has a visual manifestation beyond a deliberately constructed cell law | Real image maps in Camelyon/FMoW; separate record-field counterexamples | **Partial** | Compare summaries and target ambiguity on aligned visual information states, with fit/evaluation separation and exact-law or justified finite-evidence controls |
| A set representation is automatically informative or fixes the limitation | Stage 5 vacuity; stage 6 identical static MI intervals; stage 12 identical predictive laws and differing target hulls | **Not supported; evidence against this general claim** | Present sets as a diagnostic representation of declared uncertainty, not as a universal repair |
| Finite intervals transfer reliably across real sources/patients | Camelyon containment misses; broad FMoW containment | **Not established** | Either justify transfer and coverage for a restricted model, or make the violation part of the paper's boundary characterization |
| The phenomenon is robust across representations and groups | One principal visual encoder/split per pilot; many resamples of fixed records | **Not established broadly** | Independent group/training repetitions and a prespecified representation/partition check, or explicitly restrict the empirical claim |
| Current action values reliably predict held-out MSE improvement | Adult one-step success with horizon change; Camelyon success; FMoW failure | **Mixed; not a required central claim** | Keep as a secondary diagnostic; no need to improve policy performance to prove characterization insufficiency |
| The contribution is novel relative to existing theory | Prior local roadmap flags related-work overlap | **Open** | A focused novelty/contribution audit; adding datasets or an LP implementation does not resolve this |

**Sufficiency has three different meanings here.** One exact counterexample suffices to refute a universal sufficiency claim under its stated assumptions. The earlier single-prevalence example alone did not cover the three source targets; experiment 12 now supplies separate witnesses for them. These existence constructions still do not estimate how common the phenomenon is. A CVPR-facing empirical account additionally needs a credible connection to visual representations and a justified contribution relative to prior work. There is no fixed dataset count or success rate that guarantees acceptance.

### 5.3 Completion criteria: established formal result and remaining empirical work

The following are scientific completion criteria for the proposed claims. They are not requirements to invent a new model or obtain favorable benchmark numbers.

**A. State a precise, falsifiable limitation.**

Define R: a scalar MI, weighted-MI interval, finite predictor envelope, or full predictive law are different objects. Define the target ranges or response being compared, the common representation and model family, and any auxiliary information supplied. For a claimed static insufficiency result, exhibit equal R with unequal admissible target ranges. For a claimed reducibility result, exhibit equal R with unequal information responses. Do not infer either from low OOD AUROC alone.

For empirical near-matches, fix the matching variables and tolerances without evaluation-target access. Report differences in predictive mean, variance and, where the claim concerns it, the predictive distribution—not just similar scalar MI. A near-match is not the exact theorem. State whether the result is an existence witness or a frequency claim; an existence paper need not estimate prevalence, but must not imply it.

**B. Three-target formal connection — completed in experiment 12; preserve its scope.**

The [propositions][e12proof] and [experiment 12 results][e12results] now give separate same-R witnesses for all three targets, exact channel calculations and positive controls. Six continuous-extension membership LP checks certify analytic endpoints; eight full-rank source controls identify membership. The support-hull result is specific to the declared finite model; the LP checks do not equate its prior with a continuous-source prior. Carry these distinctions into the paper rather than treating this connection as unfinished.

The reference-target positive control passed: when that target equals the deployment label probability, its law and deployment-label response are determined by R. The failure witness instead uses unequal deployment/reference mixtures. This observed boundary belongs in the main statement, not merely a limitation footnote. Sufficiency for the deployment-label channel does not imply sufficiency for every source-stratified observation channel.

Do not compare a Bayesian credible interval to a source-compatible identification interval and attribute the difference to a failure of the summary: their priors, sets and coverage meanings differ. Align the underlying admissible information or explicitly analyze the role of those differences.

**C. Make one direct three-target visual diagnostic—not a new acquisition algorithm.**

Use the existing Camelyon/FMoW maps and recorded metadata to build a prespecified set of information states. Freeze the target definitions and fit the visual map without evaluation labels. On those states, record predictive summaries and the three probability ranges, along with controlled additions of inputs, references, labels and identities. Include an exact-law finite empirical-population control so a large width can be attributed to non-identification or finite evidence rather than left unexplained.

Compare states at matched R, or use a declared controlled perturbation that preserves R while changing target information. If shared predictive summaries are only approximately equal, report the discrepancy and contrast magnitude. Observe all targets, even when the result is no separation or no ambiguity. Selection belongs in training/reference data; evaluation records test the selected relation. Do not search held-out labels for a better example after a failure.

A neural uncertainty baseline must receive the observations it is designed to use, with a declared update rule, if its response to acquisition is being evaluated. Freezing a predictor while updating only our constraints would manufacture a contrast. Alternatively, make an explicitly **static characterization** comparison with identical predictor information and state that the added source metadata are absent from R. The claim then concerns that information restriction, not inferior learning.

**D. Distinguish the causes of width, including cases where the limitation disappears.**

At minimum, the evidence should separate these contrasts:

| Diagnostic contrast | What it resolves | Existing starting point |
|---|---|---|
| Affinely distinguishable versus confounded source laws | Identification versus algebraic ambiguity | Stage 4 geometry; stage 12 confounded witnesses and full-rank LP controls |
| Exact versus estimated input/reference laws | Identification floor versus sampling conservatism | Stage 4; add aligned controls to visual runs |
| No target labels versus known target rates | Whether the declared reference-target ambiguity is actually resolved | Stage 4 oracle target endpoints |
| Limited identities versus known mixture weights | Which membership ambiguity is due to source composition | Stage 4 oracle weights; image acquisition sweeps |
| Stable versus shifted source conditionals | Whether apparent contraction is valid under transfer | Camelyon/FMoW drift and containment diagnostics |
| Prespecified finer/coarser visual cells or another representation | Whether the claimed blind spot is specific to discarded visual information | Not yet adequately evaluated |

A positive result for the thesis can be a correctly explained failure boundary. A stable, identifiable setting in which summaries behave well is a useful control, not an unwanted result. Ordinary correlation between a score and width is descriptive and does not establish sufficiency; poor correlation without controlling the target and evidence is equally inconclusive.

**E. Match the empirical certainty to the claim.**

Report width, nonempty-set status, possible zero denominators and target containment together. Use exact targets in controlled worlds. In real empirical-population studies, label those targets as empirical, and use patient/location-level repetitions for claims about new groups. MC resampling of fixed records is not independent patient or training replication. Camelyon's current one/two/one evaluation-patient allocation cannot support broad hospital claims; use repeated whole-group splits or narrow the claim.

Sharper bounds are needed only if the finite evidence cannot distinguish the hypotheses the paper is testing. They are not required to win a width contest. If the result is that a valid set is vacuous under insufficient information, that can support a clearly scoped characterization—but it cannot be sold as informative uncertainty or practical decision guidance. If source-law assumptions are violated, either provide a justified robustness analysis or present the miss as a limitation rather than a coverage guarantee.

**F. Establish novelty and preserve the complete evidence trail.**

Relate the precise propositions to partial identification, mixture/label-shift estimation, uncertainty about probabilities and experimental-design sufficiency. The existing source lists identify foundations; they do not establish novelty. Freeze the canonical result files, disclose duplicate/same-seed runs, and keep negative cases. The early notebook-only outputs can support historical context, but should not be the sole raw evidence for a main quantitative claim.

### 5.4 Optional extensions—do not treat these as missing thesis requirements

- A learned or optimal acquisition policy, regret improvements over practical policies, or demonstrated annotation savings.
- Beating DAPPr, MC dropout or other methods on OOD AUROC, classification accuracy or a leaderboard.
- End-to-end fine-tuning or a larger encoder, unless needed to rule out a specific representation confound.
- More datasets merely to increase the count. The priority is complementary, interpretable conditions on the existing domains.
- Completing the official FMoW temporal benchmark if the paper is explicitly about regional source mixtures.
- Proving a universally calibrated replacement for existing EU measures, an additive three-component decomposition, or estimating aleatoric uncertainty.
- Noisy-ID and unequal-cost acquisition studies, unless the paper claims robustness or practical utility under those conditions.

Direct estimation and matched-information baselines remain useful **diagnostic controls** when they distinguish sources of ambiguity. A full suite of competing acquisition policies becomes necessary only if the paper adds a policy-performance claim.

### 5.5 Updated work sequence: completed first step and remaining priorities

1. **Completed: controlled same-law/three-target study (experiment 12).** Exact static witnesses, finite/oracle response tables and sufficient-information controls are saved and validated. The reference-target witness requires mixture mismatch; the aligned case is sufficient. No further formal-study execution is pending.
2. **Visual manifestation with exact/finite-law controls.** Use Camelyon and FMoW to test those specific conditions on image-derived cells, then assess source-law transfer separately. Outcome: a direct summary-versus-target/response figure, with uncertainty and failure statuses, rather than another isolated OOD ranking table.
3. **Group and representation robustness.** Repeat the diagnostic over prespecified group splits and a modest partition/representation variation. Outcome: evidence that the visual observation is not a single patient allocation or one coarse map; otherwise explicitly limit the conclusion.
4. **Contribution audit and paper claim freeze.** Decide which conditions are new and which results illustrate established principles. Outcome: a claim-to-theorem/experiment table and an abstract that does not promise unsupported policy or detector gains.

**Next actionable step is item 2.** Experiment 12 already established the relevant scope correction: aligned reference-target uncertainty is captured by R. Visual tests should include that control and the declared failure conditions, not change targets or information after seeing results until a failure appears.

### 5.6 A sufficient evidence package for a defensible draft

The following package would support a **characterization-paper submission narrative**, subject to novelty and the quality of the resulting evidence. It is not a guarantee of acceptance, nor a claim that the missing work has succeeded.

| Paper component | Evidence to use | Completion status |
|---|---|---|
| Problem definition | Three targets, declared R, evidence channels and explicit sufficiency criteria | Formal definitions and scoped criterion available in experiment 12; retain empirical information restrictions |
| Information-loss mechanism | [Exact counterexample][e6proof], [same predictive law][e6initial], [risk curves][e6fig] | Available for prevalence; experiment 12 adds the direct source-target witnesses |
| Three-target characterization | [Oracle intervals][e4oracle], [source geometry][e4geometry], [experiment 12 propositions][e12proof], [same-law target intervals][e12intervals] | **Controlled formal connection completed**, with separate witnesses and aligned-reference control |
| Visual characterization | [Camelyon curves][e8fig], [FMoW curves][e11fig] plus direct summary/target comparisons and exact/finite-law controls | Pilot responses available; required comparison incomplete |
| Conditions and counterconditions | Identifiable controls; [Camelyon drift][e8drift]/[containment][e8coverage]; [FMoW drift][e11drift]/[containment][e11coverage] | Controlled sufficiency controls completed in experiment 12; visual representation/group robustness still missing |
| Supporting reducibility observations | [Adult endpoint][e7exact]/[horizon contrasts][e7contrasts], [Camelyon endpoint][e9exact], [FMoW failed endpoint][e10exact] | Available, mixed, secondary to the three-target thesis |
| Rejected alternatives | [Weighted-MI sensitivity][e3rho], [source-vacuity audit][e5audit] | Available; no superiority claim |
| Scope and novelty | Prior-work comparison and precise list of new conditions/results | Open; no new literature clearance performed here |

**Readiness assessment:** the problem statement, model definitions, scoped three-target same-law propositions, sufficient-information controls, controlled identification results and candid pilot-results section are now supported. The controlled predictive-representation/target-ambiguity connection is complete. It does **not** yet suffice to claim theoretical and empirical characterization in vision: direct image-conditioned comparisons, group/representation robustness and novelty remain open. None requires a superior acquisition algorithm.

### 5.7 Claim language for the draft

Use: **“We characterize conditions under which predictive epistemic summaries do not determine uncertainty about specified source-mixture targets or its response to additional information.”** Specify which conditions are proved and which are observed empirically.

Avoid: **“Existing epistemic uncertainty is wrong,” “three independent types of epistemic uncertainty,” “our bounds fix epistemic uncertainty,”** or an unqualified **“we prove failure on all three targets.”** The supported statement is: **“We construct separate exact same-predictive-law witnesses for each target under declared confounding or reference-mismatch conditions, and identify sufficient-information controls.”** FMoW's failed acquisition-direction test must remain labeled a failure of that secondary endpoint. It cannot be omitted, relabeled as successful prediction, or treated as a general refutation of the characterization thesis.

## 6. Source and artifact index

The theoretical attributions are preserved in the archived [source-mixture sources][e4sources], [exact-counterexample sources][e6sources], [probability-comparison sources][e5sources], [Adult sources][e7sources], [Camelyon sources][e8sources], [FMoW sources][e11sources], and [experiment 12 derivations and source roles][e12proof]. These explain the roles of predictive MI/MC dropout, DAPPr, Probly, mixture equations, Hoeffding concentration, linear-fractional optimization, Beta conjugacy and the datasets. No new literature search or novelty clearance was performed for this retrospective.

[n1]: ../eu_blindspot_mnist_emnist_fashion_blood.ipynb
[n2]: ../eu_blindspot_within_datasets.ipynb
[n3]: ../dappr_epistemic_sets.ipynb
[n4]: ../source_acquisition_four_datasets.ipynb
[n5]: ../probly_probability_bounds_four_datasets.ipynb
[n6]: ../acquisition_counterexample.ipynb
[n7]: ../adult_acquisition_counterexample.ipynb
[n8]: ../camelyon17_acquisition.ipynb
[n9]: ../camelyon17_counterexample.ipynb
[n10]: ../fmow_counterexample.ipynb
[n11]: ../fmow_acquisition.ipynb
[n12]: ../three_target_characterization.ipynb
[e12proof]: ../THREE_TARGET_CHARACTERIZATION.md
[e12results]: three_target_characterization_20261030T144711349004Z/RESULTS.md
[e12intervals]: three_target_characterization_20261030T144711349004Z/target_intervals.csv
[extracts]: experiment_review_assets/README.md
[extractindex]: experiment_review_assets/extracted_outputs_index.csv
[adultexec]: adult_acquisition_20261029T214309638018Z/executed.ipynb
[roadmap]: CVPR_ROADMAP.md
[e1summary]: experiment_review_assets/eu_blindspot_mnist_emnist_fashion_blood/cell_13_output_00.html
[e1detect]: experiment_review_assets/eu_blindspot_mnist_emnist_fashion_blood/cell_15_output_00.html
[e1fig]: experiment_review_assets/eu_blindspot_mnist_emnist_fashion_blood/cell_17_output_00.png
[e1examples]: experiment_review_assets/eu_blindspot_mnist_emnist_fashion_blood/cell_17_output_01.png
[e2summary]: experiment_review_assets/eu_blindspot_within_datasets/cell_13_output_00.html
[e2detect]: experiment_review_assets/eu_blindspot_within_datasets/cell_15_output_00.html
[e2warnings]: experiment_review_assets/eu_blindspot_within_datasets/cell_29_output_00.html
[e2fig]: experiment_review_assets/eu_blindspot_within_datasets/cell_17_output_00.png
[e2attacks]: experiment_review_assets/eu_blindspot_within_datasets/cell_19_output_00.html
[e2linear]: experiment_review_assets/eu_blindspot_within_datasets/cell_21_output_00.html
[e2linearfig]: experiment_review_assets/eu_blindspot_within_datasets/cell_21_output_01.png
[e2duration]: experiment_review_assets/eu_blindspot_within_datasets/cell_23_output_00.html
[e3metrics]: dappr_epistemic_sets_seed0_20261029T085507170686Z/metrics.csv
[e3decisions]: dappr_epistemic_sets_seed0_20261029T085507170686Z/decision_summary.csv
[e3fig]: dappr_epistemic_sets_seed0_20261029T085507170686Z/epistemic_limits.png
[e3rho]: dappr_epistemic_sets_seed0_20261029T085507170686Z/rho_sensitivity.png
[e3scores]: dappr_epistemic_sets_seed0_20261029T085507170686Z/scores.csv
[e3config]: dappr_epistemic_sets_seed0_20261029T085507170686Z/config.json
[e4oracle]: source_acquisition_20261029T151111775438Z/oracle_intervals.csv
[e4summary]: source_acquisition_20261029T151111775438Z/acquisition_summary.csv
[e4config]: source_acquisition_20261029T151111775438Z/config.json
[e4checks]: source_acquisition_20261029T151111775438Z/inclusion_and_solver_checks.csv
[e4geometry]: source_acquisition_20261029T151111775438Z/geometry.csv
[e4fig]: source_acquisition_20261029T151111775438Z/candidate_endpoints.png
[e4mnist]: source_acquisition_20261029T151111775438Z/mnist_acquisition_curves.png
[e4emnist]: source_acquisition_20261029T151111775438Z/emnist_acquisition_curves.png
[e4fashion]: source_acquisition_20261029T151111775438Z/fashion_acquisition_curves.png
[e4blood]: source_acquisition_20261029T151111775438Z/blood_acquisition_curves.png
[e4sources]: source_acquisition_20261029T151111775438Z/SOURCES.md
[e5metrics]: probability_bounds_comparison_20261029T182738307854Z/ood_metrics.csv
[e5bounds]: probability_bounds_comparison_20261029T182738307854Z/bound_summary.csv
[e5audit]: probability_bounds_comparison_20261029T182738307854Z/source_informativeness_audit.csv
[e5fig]: probability_bounds_comparison_20261029T182738307854Z/ood_comparison.png
[e5images]: probability_bounds_comparison_20261029T182738307854Z/image_probability_intervals.png
[e5mc]: probability_bounds_comparison_20261029T182738307854Z/mc_draw_sensitivity.csv
[e5results]: probability_bounds_comparison_20261029T182738307854Z/RESULTS.md
[e5sources]: probability_bounds_comparison_20261029T182738307854Z/SOURCES.md
[e6proof]: acquisition_counterexample_20261029T192454214626Z/PROOF_AND_PROTOCOL.md
[e6initial]: acquisition_counterexample_20261029T192454214626Z/identical_initial_uncertainty.csv
[e6values]: acquisition_counterexample_20261029T192454214626Z/exact_acquisition_values.csv
[e6regret]: acquisition_counterexample_20261029T192454214626Z/score_only_policy_regret.csv
[e6fig]: acquisition_counterexample_20261029T192454214626Z/risk.png
[e6intervalfig]: acquisition_counterexample_20261029T192454214626Z/epistemic_bounds.png
[e6regretfig]: acquisition_counterexample_20261029T192454214626Z/policy_regret.png
[e6sources]: acquisition_counterexample_20261029T192454214626Z/SOURCES.md
[e7states]: adult_acquisition_20261029T214429971737Z/selected_states.csv
[e7exact]: adult_acquisition_20261029T214429971737Z/heldout_exact_one_step.csv
[e7contrasts]: adult_acquisition_20261029T214429971737Z/paired_error_contrasts.csv
[e7search]: adult_acquisition_20261029T214429971737Z/candidate_search.png
[e7fig]: adult_acquisition_20261029T214429971737Z/heldout_acquisition_curves.png
[e7results]: adult_acquisition_20261029T214429971737Z/RESULTS.md
[e7sources]: adult_acquisition_20261029T214429971737Z/SOURCES.md
[e8metrics]: camelyon17_acquisition_20261030T092422959226Z/ood_scores.csv
[e8summary]: camelyon17_acquisition_20261030T092422959226Z/acquisition_summary.csv
[e8fig]: camelyon17_acquisition_20261030T092422959226Z/acquisition_bounds.png
[e8drift]: camelyon17_acquisition_20261030T092422959226Z/source_law_drift.csv
[e8coverage]: camelyon17_acquisition_20261030T092422959226Z/empirical_containment_summary.csv
[e8groups]: camelyon17_acquisition_20261030T092422959226Z/split_counts.csv
[e8definitions]: camelyon17_acquisition_20261030T092422959226Z/bounds_definition.json
[e8sources]: camelyon17_acquisition_20261030T092422959226Z/SOURCES.md
[e9states]: camelyon17_counterexample_20261030T092024988890Z/selected_states.csv
[e9exact]: camelyon17_counterexample_20261030T092024988890Z/heldout_exact_one_step.csv
[e9fig]: camelyon17_counterexample_20261030T092024988890Z/acquisition_curves.png
[e9rates]: camelyon17_counterexample_20261030T092024988890Z/population_rates.csv
[e9credible]: camelyon17_counterexample_20261030T092024988890Z/target_credible_enclosures.csv
[e9groups]: camelyon17_counterexample_20261030T092024988890Z/split_counts.csv
[e10states]: fmow_counterexample_20261030T094929883846Z/selected_states.csv
[e10exact]: fmow_counterexample_20261030T094929883846Z/heldout_exact_one_step.csv
[e10fig]: fmow_counterexample_20261030T094929883846Z/acquisition_curves.png
[e10rates]: fmow_counterexample_20261030T094929883846Z/population_rates.csv
[e10protocol]: fmow_counterexample_20261030T094929883846Z/protocol.json
[e10result]: fmow_counterexample_20261030T094929883846Z/result.json
[e11metrics]: fmow_acquisition_20261030T094932286096Z/ood_scores.csv
[e11summary]: fmow_acquisition_20261030T094932286096Z/acquisition_summary.csv
[e11fig]: fmow_acquisition_20261030T094932286096Z/acquisition_bounds.png
[e11drift]: fmow_acquisition_20261030T094932286096Z/source_law_drift.csv
[e11coverage]: fmow_acquisition_20261030T094932286096Z/empirical_containment_summary.csv
[e11accuracy]: fmow_acquisition_20261030T094932286096Z/accuracy.csv
[e11groups]: fmow_acquisition_20261030T094932286096Z/split_counts.csv
[e11sources]: fmow_acquisition_20261030T094932286096Z/SOURCES.md
