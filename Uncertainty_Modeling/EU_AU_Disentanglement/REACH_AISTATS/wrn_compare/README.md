# CIFAR-10 transformations on frozen WRN features

This compares a Gaussian feature-density score, bootstrap-head EU, ensemble entropy,
mean member entropy, and the original WRN's entropy/MSP/energy on the same inputs.

The backbone is the `pytorch-ood` CIFAR-10 pretrained WRN-40-2 (`cifar10-pt`).
It is frozen and has seen all ten CIFAR-10 classes. This is a transformation study,
not a held-out-class study. No wild OOD dataset is used.

## Run

From the REACH directory, with NumPy, SciPy, pandas, matplotlib, scikit-learn,
PyTorch, torchvision and pytorch-ood installed:

```sh
python -m wrn_compare.run --data-root /path/to/data --smoke --out results/wrn_smoke
python -m wrn_compare.run --data-root /path/to/data --out results/wrn_compare
```

Use `--download` only if you want torchvision/pytorch-ood to download missing data
and weights. An existing checkpoint can be supplied using `--checkpoint PATH`.
Default device selection is CUDA, then MPS, then CPU. The full default uses three
seeds, 500 fitting examples per class, 100 examples per class in each test pool,
five bootstrap heads and five transformation strengths.

For notebook controls:

```python
from wrn_compare.run import Config, run_experiment
config = Config(data_root='/path/to/data', out='results/my_comparison',
                seeds=(0, 1, 2), strengths=(0., .25, .5, .75, 1.),
                fit_per_class=500, test_per_class=100, members=5)
summary = run_experiment(config)
```

## What changes

Pixels are transformed in [0,1] before checkpoint-specific normalization and WRN
feature extraction. The same original images, mixing partners and noise directions
are used across strengths. No transformed features enter density or head fitting.

- `same_class_mix`: blend two different images from the same CIFAR class.
- `cross_class_mix`: blend images from different CIFAR classes.
- `gaussian_noise`: add a fixed Gaussian direction with standard deviation
  `strength * noise_std`, then clip pixels to [0,1].

Both mixing endpoints are ID images. A midpoint may be ambiguous or unfamiliar;
no strength is assigned an unverified semantic OOD label. AUROC is explicitly
**transformed-versus-clean discrimination**. Source-label agreement on changed
images is descriptive, not accuracy against a justified transformed-image label.

## Fairness and interpretation

Training features fit PCA, classwise shrunk Gaussian covariances, feature scaling,
and stratified bootstrap logistic heads. The three test pools (calibration,
evaluation, mixing partners) are disjoint by CIFAR image index. Each partner is
used once within a transformation. Calibration uses clean ID scores only, and
thresholds stay fixed across transformations. The density score uses empirical
class priors, PCA-32 and Ledoit-Wolf shrinkage.

This is a **DDU-style density baseline**, not a reproduction of DDU: no DDU
spectral-normalization training was performed. Head-bootstrap mutual information
is conditional on one fixed representation and does not measure backbone
uncertainty. Entropy and mean member entropy are proxies, not measured true AU.

The density-minus-EU AUROC interval resamples complete original/transformed pairs,
conditional on the fitted models and calibration. Repeated seeds vary selections,
partners, noise and bootstrap heads; they do not vary pretrained backbone training.
No tuning or winner selection should be based on the smoke run.

## Outputs

- `clean_features_seed*.npz`: actual WRN features and original indices/labels.
- `pool_assignments.csv`: auditable pool membership.
- `rows.csv`: aligned scores, normalized threshold distances, predictions and pair IDs.
- `summary_per_seed.csv`, `summary.csv`: rejection rates and shift AUROCs; seed SD
  is blank for a single-seed run because it is undefined.
- `density_minus_EU.csv`: paired AUROC differences and bootstrap intervals.
- `clean_accuracy.csv`: held-out clean accuracy of WRN and the new heads.
- `comparison.pdf/png`, `transformed_examples.pdf/png`: curves and actual images.
- `manifest.json`: completion/failure status, exact configuration, code hashes,
  checkpoint hash, versions and limitations.

Use a new output directory for each configuration to avoid mixing old artifacts.
The smoke configuration runs only one seed, 200 fitting images, 50 calibration
images, 50 evaluation images and 50 partner images. Its purpose is integration
verification, not evidence that a method wins.

## Adapter extension (not trained or implemented yet)

An adapter can map frozen WRN features z to g(z). First define its learning
objective; the input transformations alone do not define one. Compare density and
head-bootstrap EU on both original and adapted features. Refit every feature-space
detector and recalibrate its threshold after adapter fitting. Use separate
transformation families/parameters for adapter training and final testing.

A clean-versus-transformed classifier is a learned detector trained with synthetic
outliers; its output should not be relabeled as epistemic uncertainty.

References: [DDU paper](https://arxiv.org/abs/2102.11582),
[WRN checkpoint example](https://pytorch-ood.readthedocs.io/en/stable/auto_examples/detectors/gram.html).
