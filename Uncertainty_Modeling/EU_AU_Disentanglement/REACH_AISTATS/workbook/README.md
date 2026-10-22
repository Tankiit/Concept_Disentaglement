# REACH experiment workbook

Current research direction: [CVPR roadmap and ordered next steps](CVPR_ROADMAP.md).
Read this roadmap before planning new experiments; it records the current assessment,
open questions, and evidence needed beyond the completed controls.

Experiments, diagnostics, and preparation studies ordered from oldest to newest.
Dates follow the workbook names or recorded plan dates; nested follow-ups use
their parent workbook date, and entries on the same date are not a precise execution timeline.

| Date | Experiment / study | One-sentence description |
| --- | --- | --- |
| 2026-10-16 | [Data splits](understand_data_splits_2026-10-16/README.md) | Traces known/unknown class assignments, sample splits, and loader behavior using a synthetic dataset. |
| 2026-10-16 | [Feature notebook walkthrough](understand_features_notebook_2026-10-16/README.md) | Maps `features.ipynb` and checks data-pool bookkeeping, with only the initial preparation stage executed. |
| 2026-10-16 | [CIFAR-10 dropout uncertainty](cifar10_dropout_uncertainty_2026-10-16/README.md) | Compares dropout disagreement and predictive entropy for held-out CIFAR-10 classes using frozen ResNet-18 features. |
| 2026-10-16 | [Original script execution](original_scripts_2026-10-16/README.md) | Runs the original example and inspects cached-feature loaders while preserving their original behavior. |
| 2026-10-16 | [REACH first try — plan only](reach_first_try/README.md) | Proposes tracing synthetic REACH intervals and their acquisition responses without recording executed experiments. |
| 2026-10-19 | [MNIST novelty pilot](mnist_reach_novelty_2026-10-19/RESULTS.md) | Trains the original MNIST CNN and studies REACH intervals for digit mixtures containing EMNIST letters or KMNIST novelty. |
| 2026-10-19 | [Fashion and Blood novelty pilot](fashion_blood_reach_2026-10-19/RESULTS.md) | Introduces Fashion-MNIST and BloodMNIST into the MNIST pipeline while retaining EMNIST candidate references. |
| 2026-10-19 | [Focused interval diagnostic](reach_interval_diagnostic_2026-10-19/README.md) | Dissects one MNIST–EMNIST cell interval using exact-probability controls, reference bounds, and denominator diagnostics. |
| 2026-10-19 | [Interval diagnostic follow-up](reach_interval_diagnostic_2026-10-19/followup/REPORT.md) | Investigates candidate mismatch and coherent joint-mass intervals using finite-pool simulations for letters, Fashion, and Blood. |
| 2026-10-20 | [Boundary stress and partitions](boundary_reach_2026-10-20/RESULTS.md) | Moves MNIST features across a linear decision boundary and compares dropout behavior with partition-dependent unexplained-mass bounds. |
| 2026-10-20 | [Fashion/Blood partition comparison](fashion_blood_partitions_2026-10-20/RESULTS.md) | Compares frozen prediction, margin, support, and combined partitions for global candidate prevalence and source-model compatibility. |
| 2026-10-21 | [Original width visualizations](reach_width_plots_2026-10-21/README.md) | Plots the five saved legacy REACH widths across Fashion/Blood acquisition budgets without recomputing intervals. |
| 2026-10-21 | [Controlled MNIST–EMNIST diagnosis](mnist_emnist_controlled_2026-10-21/RESULTS.md) | Compares original, oracle, and revised per-cell intervals on the same saved MNIST–EMNIST run. |
| 2026-10-21 | [Repeated acquisition validation](mnist_emnist_controlled_2026-10-21/repeated/RESULTS.md) | Tests original and revised per-cell intervals across 100 paired finite-pool draws at two acquisition budgets. |
| 2026-10-21 | [Three epistemic targets](epistemic_targets_2026-10-21/RESULTS.md) | Studies input, target-label, and source-identity acquisitions with two aggregated digit sources and distinguishable versus confounded candidates. |
| 2026-10-21 | [Epistemic targets with Fashion/Blood](epistemic_targets_2026-10-21/fashion_blood/RESULTS.md) | Repeats the three-target acquisition controls with matching Fashion and Blood candidate laws treated as exact. |
| 2026-10-21 | [Individual digits and held-out digits](digit_sources_2026-10-21/RESULTS.md) | Restores ten individual digit sources for reference controls and trains a separate 0–5 classifier to test held-out digits 6–9. |
| 2026-10-21 | [Boundary disagreement versus REACH](boundary_epistemic_2026-10-21/RESULTS.md) | Compares dropout disagreement, class intervals, label intervals, and source-model rejection on identical boundary-crossing trajectories. |
| 2026-10-21 | [Candidate camouflage](camouflage_2026-10-21/RESULTS.md) | Gradually blends the EMNIST candidate law into a known-digit mixture to distinguish finite-evidence uncertainty from exact nonidentifiability. |
| 2026-10-21 | [Source identities under camouflage](camouflage_identities_2026-10-21/RESULTS.md) | Adds reliable source-identity observations at 99% and 100% camouflage to measure which epistemic widths shrink with additional evidence. |
| 2026-10-21 | [Reference acquisition under camouflage](camouflage_references_2026-10-21/RESULTS.md) | Holds identities fixed and increases digit references, candidate references, or both to isolate how source-law estimation affects the three epistemic widths. |
| 2026-10-21 | [Target-label acquisition under camouflage](camouflage_labels_2026-10-21/RESULTS.md) | Varies synthetic target-label budgets with other evidence fixed and uses exact-rate controls to isolate remaining source-law uncertainty in W_label. |
| 2026-10-21 | [Candidate-law misspecification](candidate_mismatch_2026-10-21/RESULTS.md) | Deploys Fashion or Blood with matching or wrong candidate references to measure rejection, coverage failures, and narrow-but-wrong membership intervals. |
| 2026-10-22 | [Contribution and theory audit](contribution_audit_2026-10-22/AUDIT.md) | Compares REACH with related methods, reconciles target definitions, and develops an explicit compatible-but-wrong counterexample and proof plan. |
| 2026-10-22 | [Graded candidate mismatch](graded_mismatch_2026-10-22/RESULTS.md) | Keeps Fashion deployment fixed while varying the declared candidate law to locate coverage failures before rejection. |
| 2026-10-22 | [Candidate-law sensitivity baseline](candidate_sensitivity_2026-10-22/RESULTS.md) | Expands candidate coordinate bounds at fixed error budgets to measure recovery of coverage and the loss of interval informativeness. |
| 2026-10-22 | [Coupled TV sensitivity](candidate_tv_2026-10-22/RESULTS.md) | Compares a total-variation candidate region with coordinate boxes and documents an independent-audit protocol for error-budget justification. |
| 2026-10-22 | [Candidate image audit versus direct estimation](candidate_audit_2026-10-22/RESULTS.md) | Uses disjoint cached image evidence to compare an audit-derived TV bound with direct candidate-law estimation, finding the latter more informative. |
| 2026-10-22 | [Visual acquisition — protocol only](visual_acquisition_2026-10-22/PROTOCOL.md) | Specifies equal-cost image annotation comparisons, real parity targets, direct estimation baselines and a second representation while exposing shared label information. |

Each linked report records its assumptions and limitations; the experiments use
different targets and evidence settings, so their widths are not automatically comparable.

Supporting material: [next experiments and progress](next-to-dos.md) ·
[reference manuscript](AISTATS2027_REACH_.pdf).
