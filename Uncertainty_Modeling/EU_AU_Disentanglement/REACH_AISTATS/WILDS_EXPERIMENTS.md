# Camelyon17 and FMoW acquisition experiments

Open `camelyon17_acquisition.ipynb`, `camelyon17_counterexample.ipynb`,
`fmow_acquisition.ipynb`, or `fmow_counterexample.ipynb`. Each contains the
protocol, theory, citations, interpretation and a goal before every code cell.
Every run saves outputs below `workbook/<notebook-name>_<UTC>/`.

## Two distinct experiments

**Image acquisition:** frozen ImageNet ResNet18 features; matched cross-entropy
and actual DAPPr-loss heads; actual Probly MC sampling and probability envelopes;
finite-cell source-mixture lower/upper probabilities; four acquisition sweeps.
The events are explicitly different: class predictions, known-source membership,
candidate-source membership, and a reference-anchored target probability.
Their widths are not additive and are not all estimates of predictive MI.

**Real-record counterexample:** source identity and target labels update a
Beta source-mixture model for the same population prevalence. Search 600 states
for a prespecified near-match in MI, then evaluate opposite acquisition values
on held-out records. Save a negative result if no match or no held-out reversal
is found. This counterexample does not condition on pixels or claim empirical
failure of DAPPr. Credible target enclosures are distinguished from identification
bounds and intervals of epistemic MI. The exact toy proof remains in
`acquisition_counterexample.ipynb`.

## Data and splits

Camelyon known hospitals are 0 and 3; candidate hospital is 2. Patients are
partitioned as whole groups. FMoW known regions are Americas and Europe;
candidate region is Asia. Rounded spatial locations form whole groups.
Both use a custom deterministic 60/20/20 fit/reference/evaluation split;
**these are not official WILDS benchmark results**. Original split fields remain
in the audit. FMoW observations from 2016 onward are excluded from this initial
regional experiment, not reported as a completed temporal experiment.

The image pilot caps each role/source at 2,000 randomly selected images, without
class balancing. This changes the empirical mixture and is documented; the
counterexample uses all retained records in its roles. Confidence statements are
conditional on empirical resampling and model assumptions. Patch count is not
patient count. In particular a single held-out patient cannot support broad
hospital-population conclusions. Reference-to-deployment source-law invariance
must be checked; empty sets and undefined conditioning denominators are saved.

## Run

From this directory, use a Python with `nbformat`, `nbclient`, and
`jupyter_client` for the runner. The kernel defaults to the existing
`reach-aistats` interpreter; `--python` can override it. Kernel dependencies:
`numpy scipy pandas matplotlib torch torchvision probly pyarrow scikit-learn
ipykernel`. Exact installed versions are saved per run.

```bash
python scripts/run_wilds_notebook.py camelyon17 acquisition
python scripts/run_wilds_notebook.py camelyon17 counterexample
python scripts/run_wilds_notebook.py fmow acquisition --data-root /home/mtleao/projects/Concept_Disentaglement/fMoW
python scripts/run_wilds_notebook.py fmow counterexample --data-root /home/mtleao/projects/Concept_Disentaglement/fMoW
```

The image runner uses `data/wilds/resnet18-f37072fd.pth`; supply `--weights`
for another local copy of those torchvision ImageNet1K V1 weights. It records the
hash and uses CUDA when accessible. The counterexample does not require a GPU.

FMoW data must be one of:

- An official `fmow_v1.1/` folder with `rgb_metadata.csv`,
  `country_code_mapping.csv` and `images/rgb_img_<index>.png`.
- A local copy of the complete parquet files from
  [danielz01/fMoW](https://huggingface.co/datasets/danielz01/fMoW), WILDS config,
  preferably pinned to `8a27a7988a20341352b85bf0e65e88a6c1337f1c`.
  The adapter reads shards recursively. The full mirror has 16 train, 3 id_val,
  3 id_test, 5 val and 5 test shards. Keep domain and location fields intact.

The FMoW mirror requires access approval/authentication. The original CodaLab
endpoint failed TLS certificate verification on 2026-10-30. No authentication
or certificate checks were bypassed. The user supplied the complete FMoW mirror
at `/home/mtleao/projects/Concept_Disentaglement/fMoW`; both real-data notebooks
have now run. See `WILDS_RESULTS.md` for the negative counterexample endpoint and
image results. The input folder's `local_manifest.json` records local hashes,
file sizes and row counts.

Camelyon was obtained through the public pinned mirror documented in the
notebooks. `scripts/download_camelyon_mirror.py` downloads all 21 shards, with
atomic finalization; the adapter rejects incomplete splits.

## Reproducibility and review

`protocol.json`, `record_index.csv.gz`, `split_counts.csv`, selected-image IDs,
initial/acquired record IDs, posterior states or model checkpoints, probability
endpoints, figures and result-dependent reports are retained. The runner saves
`executed.ipynb`, execution status and an SHA-256 artifact manifest even on a
failed execution. The main notebook also receives outputs.

Build notebook sources using `python scripts/build_wilds_notebooks.py` only when
you intentionally want to regenerate them; rebuilding clears main-notebook
outputs, while timestamped executed notebooks remain intact.

Run meaningful formula and leakage checks:

```bash
python -m unittest tests.test_wilds_acquisition tests.test_beta_acquisition tests.test_source_acquisition
```

No claim of improved OOD performance, identification, calibration or acquisition
choice should be made merely because a run completes. Read `RESULTS.md`, failure
statuses, the held-out endpoint and source-law audit first.
