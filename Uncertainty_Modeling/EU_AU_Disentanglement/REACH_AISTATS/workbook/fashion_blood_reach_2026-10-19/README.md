# Fashion-MNIST and BloodMNIST REACH extension — 2026-10-19

This extends the MNIST/letters/KMNIST pilot using the exact saved digit CNN,
PCA/LDA partition, MNIST references, EMNIST letter references, and MNIST log/eval
rows. No training, threshold tuning, or partition fitting is performed here.
The original REACH source modules remain unchanged.

## Roles and protocol

- Known source classes: MNIST digits 0–9.
- Declared candidate novelty: EMNIST Letters references from the prior pilot.
- Undeclared deployment novelty: Fashion-MNIST or BloodMNIST, separately.
- Evaluation source/class labels: used only for controlled sampling and scoring.

Fashion-MNIST contributes its official 10,000 test images. BloodMNIST contributes
its official 3,421 test images, across eight blood-cell categories. The full
BloodMNIST archive is downloaded, but training/validation images are unused.
Each new test split is divided into seeded, stratified, disjoint log/evaluation
pools. Simulated deployment draws sample with replacement from the log pool.
Within the novelty source, classes are equally weighted during simulation;
held-out OOD diagnostics use the natural evaluation class counts. These are
different weighting conventions, especially for imbalanced BloodMNIST.

Inputs are 28x28 grayscale divided by 255. BloodMNIST RGB is converted with
Pillow's RGB-to-L conversion, without inversion or contrast normalization.
Removing color does not remove all acquisition differences: bright microscopy
backgrounds and textures may be easy for a digit representation to distinguish.
This is an OOD methodology experiment, not a medical classification evaluation.

## Grid and interpretation

For each new dataset: novelty mass 5% or 20%, three simulation seeds, seven
unique budget configurations, for 84 original `real.audit.run_cell()` calls.
Each returns 11 cell rows, for 924 records.

Baseline: n=1000 deployment observations, m=200 references/digit,
r=50 references/letter class. One factor changes at a time:

- n: 250 / 1000 / 4000.
- m: 50 / 200 / 1000.
- r: 10 / 50 / 200.

As in the prior pilot, internal comparison baselines use 20 bootstrap draws
and five histogram-resampling members; these are not deep neural ensembles.
There is one frozen CNN, not multiple training seeds. Reference and log draws
are paired by seed where the original engine supports it.

Both new novelty laws differ from the EMNIST candidate. REACH intervals concern
the declared mixture family; coverage against the actual novel source is a
stress-test result, not a claimed guarantee. Solver failures, infeasible sets,
and missing bounds must be reported separately from finite interval coverage.
Entropy and feature-distance AUROC are separate OOD diagnostics, not REACH widths.

## Provenance and reproduction

Fashion-MNIST: pinned `zalando-datasets/fashion_mnist` mirror revision with
SHA-256 verification. BloodMNIST: official MedMNIST 28x28 archive from
https://zenodo.org/records/10519652, verified against the project's MD5 and
also recorded with SHA-256. BloodMNIST is CC BY 4.0; dataset information and
attribution: https://medmnist.com/ and https://doi.org/10.1038/s41597-022-01721-8.
Full URLs, digests, and preprocessing are in data_provenance.json.

From REACH_AISTATS, using reach-aistats:

```bash
python workbook/fashion_blood_reach_2026-10-19/prepare_data.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python workbook/fashion_blood_reach_2026-10-19/run_extension.py
python workbook/fashion_blood_reach_2026-10-19/summarize.py
```

The extension imports the architecture from the prior workbook and loads its
saved model/partition; keep that folder available. Config.json records their
checksums. MNIST cell IDs and entropy scores are checked against the previous
run before new REACH computations. Images reside in cache/mnist_reach.
