# MNIST REACH novelty pilot — 2026-10-19

## Question and scope

With a frozen digit representation and partition, how do deployment data,
digit references, and letter references change REACH intervals? What happens
when deployment novelty is KMNIST but the declared novelty reference is EMNIST?

This is an exploratory first pilot on MNIST, EMNIST Letters, and KMNIST.
Fashion-MNIST and medical datasets are extensions, not included in this run.
The separate CIFAR/CLIP notebook reproduction is not being executed here.

## Data and information access

`prepare_data.py` downloads image Parquet files from pinned Hugging Face
revisions, validates their SHA-256 hashes, and records URLs, shapes, and labels
in data_provenance.json. No dataset-provided code is executed.

All inputs are 28x28 grayscale scaled to [0,1]. The raw letter mirror has the
EMNIST transposed orientation; the experiment transposes letter image axes
before encoding. `input_examples.png` shows raw downloaded inputs, while
`preprocessed_examples.png` shows the orientation used in the experiment.

| Pool | Rows | Used for |
| --- | ---: | --- |
| MNIST training subset | 30,000 | Train the CNN for five epochs |
| MNIST partition development | 10,000 | Fit the frozen PCA/LDA cell partition |
| MNIST reference | 20,000 | Draw labelled digit reference budgets |
| EMNIST Letters training subset | 5,200 (200/class) | Nested letter references: 10/50/200 per letter class |
| MNIST test | 10,000 | Stratified disjoint deployment/evaluation halves |
| EMNIST Letters test | 20,800 | Stratified disjoint deployment/evaluation halves |
| KMNIST test | 10,000 | Stratified disjoint deployment/evaluation halves |

The CNN sees only digit labels. Letter labels organize an equal-weight
candidate novelty source; they do not train the encoder or partition. KMNIST
labels organize controlled simulation and evaluation, never reference fitting.
Unknown status refers to information withheld from the model, not visual style.

Within the log pool, simulated observations are sampled with replacement.
A sample budget therefore counts observations, not necessarily distinct images.
The exact source/index assignments are in pool_assignments.csv.

## Frozen model and REACH adapter

A small two-convolution CNN maps images to 64 features and ten digit logits.
After training it is frozen. The original `real.head.LDAHead` fits PCA-16 and
LDA on the independent development features, defining 10 known-class cells
plus an outlier cell using the development 99th-percentile distance threshold.
The CNN classification head and the LDA partition are distinct components.

The workbook constructs contexts for the original `real.audit.run_cell()`:
10 digit sources plus one pooled novelty source. For every scenario the
novelty reference distribution is from EMNIST Letters. Its 26 within-source
classes are equally weighted. KMNIST deployment novelty similarly uses equal
weights over its ten classes. Digital class prevalences are either uniform
or Dirichlet(1), paired by simulation seed across budgets and novelty arms.

Original `real.audit.run_cell`, reference estimators, feasible-set solvers,
and width definitions are used. Only the uniform control temporarily replaces
`true_weights` inside this workbook process; original source files are unchanged.
The exact L2 feasible set is the current audit default.

## Acquisition design

Six scenarios: digits-only uniform, digits-only shifted prevalence, letters
at 5% and 20% novelty, and KMNIST at 5% and 20% novelty. Every scenario has
three simulation seeds and seven unique budget settings:

- Baseline: n=1000 deployment observations, m=200 references/digit,
  r=50 references/letter class.
- n intervention: 250 or 4000, with m/r fixed.
- m intervention: 50 or 1000, with n/r fixed.
- r intervention: 10 or 200, with n/m fixed.

This gives 126 calls to run_cell and 1,386 cell-level records (11 cells/call).
There is one encoder-training seed, not three independently trained encoders.
No test-based model or threshold selection is performed. Baselines inside
run_cell use only 20 bootstrap draws and five resampled histogram members
for this pilot; these are not deep neural-network ensemble members.

## Interpretation

`W_cls` uses plug-in conditionals; `W_joint` also varies the known conditional
entries. `W_unk_joint` allows uncertainty in novelty and known entries.
`W_label` is the audit's fixed-anchor width over conditional entries; it is
not an aleatoric uncertainty estimate or the binary-target prototype's exact
label-channel construction. These widths are not additive components.

Truth/coverage in reach_rows.csv refer to the finite empirical deployment
source pools used for simulation. This is a controlled diagnostic, not a
population coverage proof. Even EMNIST train/test empirical conditionals can
differ. For KMNIST the declared novelty source is mismatched: reported coverage
is descriptive and the model's guarantee is not assumed. Incompatibility is
failure of the declared mixture fit, not identification of the type of shift.
A larger semantic difference does not imply stronger cell-space separation.

Evaluation images are held out for known-digit accuracy, entropy/distance OOD
AUROC, and outlier-cell fractions. These are separate from the cell-level REACH
coverage analysis. No MC-dropout score is computed in this pilot.

## Reproduction

Activate `reach-aistats`, then from REACH_AISTATS:

```bash
python workbook/mnist_reach_novelty_2026-10-19/prepare_data.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python workbook/mnist_reach_novelty_2026-10-19/run_reach.py
```

The existing environment also needs pyarrow, installed during the prior
workbook work. Configurations, checkpoints, fitted partition, pool assignments,
raw scores, per-cell REACH records, warnings, and summary plots are saved here.
The data themselves reside in cache/mnist_reach.

## Completed results

All 126 calls completed. Read [RESULTS.md](RESULTS.md) for the outcome and
a worked cell calculation. See [width_response_grid.png](width_response_grid.png)
for acquisition effects and [pool_summary.csv](pool_summary.csv) for exact pool
sizes (class-wise halving can differ by one row from a global half).
Run `python workbook/mnist_reach_novelty_2026-10-19/summarize.py` to regenerate
the checked report and plots from saved results.
