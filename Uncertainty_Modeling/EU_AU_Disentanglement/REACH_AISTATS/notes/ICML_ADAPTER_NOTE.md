# WRN adaptation and uncertainty for an ICML study

Research note, 24 October 2026. The proposed experiment asks whether adaptation
changes the relationship between feature-space support, classifier confidence,
and model disagreement along controlled input transformations. It does not assume
that a transformed ID image is semantically OOD, or that uncertainty must increase
with transformation strength.

## Current baseline

Use CIFAR-10 with a frozen pretrained WRN-40-2. Compare a Gaussian mixture density
score on its features, bootstrap-head mutual information (EU proxy), predictive
entropy, MSP and energy. The Gaussian baseline uses training-only PCA and
classwise covariance shrinkage. It is DDU-style density estimation, not a full
DDU reproduction: the pretrained backbone was not trained using DDU's spectral
normalization recipe. Bootstrap heads share one frozen representation, so their
EU omits backbone uncertainty.

Create paired paths by mixing two same-class images, mixing different-class
images, and adding Gaussian noise. Transform pixels before WRN normalization and
feature extraction. Keep original image IDs, partner IDs, source classes and
strengths. Both endpoints of an image-mixing path are ID images; a nonmonotonic
uncertainty curve is plausible. Do not confuse sampling unchanged ID/OOD rows
into one pool with synthesizing a new input by interpolation.

## Adapter comparison

Freeze the backbone and compare:

1. Original features z = f(x).
2. Nonlinear residual bottleneck: z' = z + B sigma(Az).
3. Low-rank linear residual: z' = z + (alpha/r) BAz.

The third is a LoRA-like update of an identity map. Standard LoRA applies the
low-rank update to an existing frozen matrix W: W' = W + (alpha/r) BA. Removing
the activation is not an equivalent rewrite of a nonlinear MLP. Match parameter
budgets and report rank, scaling, activation and initialization. Initialize the
residual at zero if identity behavior at the start is desired.

Placement determines what can change. Adapting only the classification head
changes predictions and head uncertainty but leaves density on the original
features unchanged. A feature adapter changes the space in which density is
fitted; refit the density model and recalibrate thresholds for every adapted
representation. A globally invertible linear feature map with an exactly
transformed full Gaussian model changes log density only by a constant Jacobian
term, preserving rankings. PCA, regularization or nonlinear maps can break this
invariance; improvements must not be attributed to low rank alone.

The training objective is still undecided. Training clean-versus-transformed
classification produces a detector trained with synthetic outliers, not an
identified EU estimator. The objective must be specified before implementing or
claiming the adapter method.

## Controls and falsifiable questions

Does adaptation improve detection on transformation families excluded from
adapter training, at thresholds calibrated on clean ID validation rows? Does
clean accuracy remain acceptable? Do density and EU disagree on the same paired
examples, and does that disagreement persist across seeds, density estimators,
head capacity and transformation choices?

Separate adapter training, density/head fitting as required by the protocol,
threshold calibration and final evaluation. Reserve transformation families or
parameters for testing; test on additional real shifts before generalizing to
wild OOD. Include frozen-backbone and classifier-only adaptation controls. Do not
train a score to increase with severity and then present increasing severity
curves on those same transformations as independent evidence.

Report paired trajectories, rejection at a fixed clean-ID false-alarm target,
clean accuracy and transformed-versus-clean AUROC. This AUROC is a distribution
shift diagnostic, not verified semantic OOD performance. Bootstrap complete
original/transformed pairs; repeat model and data seeds separately. A single
adapter does not by itself provide an ensemble-based EU estimate.

## Tinker decision

The current WRN experiment belongs in PyTorch. Tinker's documented hosted model
catalog does not include WRN, and its custom-loss API operates on token
log-probabilities. Replacing an MLP with a low-rank adapter does not make an
arbitrary WRN architecture executable on Tinker. A Tinker extension would use a
supported vision-language model and preserve the evaluation question, while
changing the backbone, output interface and potentially the uncertainty estimator.
This is a separate experiment, not a drop-in backend substitution.

## Evidence and unfinished work

Executed: a real-CIFAR smoke run with 200 fitting images, 50 calibration images,
50 evaluation images, 50 partner images, five bootstrap heads and one seed. Seven
WRN comparison tests passed. This verifies the pipeline; it does not establish a
method ranking or a paper contribution. Full replication, full DDU training,
adapter training and wild OOD evaluation have not been executed.

The relationship to the CVPR evaluation project and the ICML margin/uncertainty
project remains a framing decision. This note records a possible ICML study;
it does not establish novelty or decide paper ownership.

## Sources

- [DDU](https://arxiv.org/abs/2102.11582)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [Tinker model catalog](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)
- [Tinker custom losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/custom/)

Tinker compatibility was checked during this discussion on 24 October 2026;
recheck before planning a hosted experiment.
