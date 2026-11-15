# Copy and run the novel input example

The main example is **wrn_comparison.ipynb**, backed by **wrn_compare/**. It uses
real CIFAR-10 images and frozen WRN features. **novel_variations.py** is the older
finite-cell probability-law toy; it is not the image/EU experiment.

## Where the code lives

Until copied, all files live in the Codex task's separate REACH working folder.
Your main repository has not been modified. The transfer ZIP contains only the
new experiment, its note/tests, and a small saved smoke result.

Extract the ZIP, then from a terminal run:

```sh
python /path/to/reach_novel_example/install_into_reach.py \
    /path/to/your/REACH_AISTATS
```

Replace both paths with your actual paths. The destination must already exist.
The installer adds new files, skips byte-identical files, and refuses to overwrite
any differing existing file. It does not change your main.py, models, existing
data modules, original notebook, or environment. You can alternatively copy the
listed experiment files manually.

## Files to add

```text
REACH_AISTATS/
  wrn_compare/                   experiment implementation
  wrn_comparison.ipynb           notebook controls and results
  novel_variations.py            optional finite-cell toy
  requirements-novel.txt         dependency list
  NOVEL_EXAMPLE_README.md        these instructions
  notes/ICML_ADAPTER_NOTE.md     research note
  tests/test_wrn_compare.py      small array-based tests
  tests/test_novel_variations.py toy regression tests
  results/wrn_smoke/             saved one-seed demonstration
```

The finite-cell toy imports your existing `data.worlds.World` and `make_world`.
The WRN experiment does not depend on your REACH solver or data.worlds modules.

## Run the real CIFAR example

Use the environment in which you normally run PyTorch. Install missing dependencies
from `requirements-novel.txt` as needed, then launch Jupyter from REACH_AISTATS.
Open `wrn_comparison.ipynb`. By default it displays the copied smoke result.
For a new run, set:

```python
RUN_EXPERIMENT = True
config = Config(
    data_root='/path/to/data',  # contains cifar-10-batches-py/
    out=str(ROOT / 'results/wrn_my_run'),
    download=False,            # True permits missing data/weights to download
    fit_per_class=500,
    test_per_class=100,
    seeds=(0, 1, 2),
    strengths=(0., .25, .5, .75, 1.),
    members=5,
    bootstrap=200,
)
```

On the original computer, CIFAR-10 was found under `/Users/tanmoy/data`, and the
checkpoint was found in the normal PyTorch hub cache. These are local paths, not
files included in the ZIP. On another machine, supply your own data/checkpoint
paths or set `download=True`. The loader defaults to the cached `cifar10-pt`
WRN-40-2 weights; arbitrary WRN architectures/checkpoints are not interchangeable.

Restart the notebook kernel after copying/updating code if old imports are loaded.
To verify which copy Python imports:

```python
import wrn_compare.run as wrn_runner
print(wrn_runner.__file__)
```

The printed path should be inside your main REACH_AISTATS folder.

A command-line smoke check, run from that folder:

```sh
python -m wrn_compare.run --data-root /path/to/data --smoke --out results/wrn_my_smoke
python -m pytest -q tests/test_wrn_compare.py
```

## Change the novel-input construction

Edit `config.kinds`, `config.strengths`, and `config.noise_std`. Available kinds:
`same_class_mix`, `cross_class_mix`, `gaussian_noise`. Mixing is in RGB pixel
space; WRN features are recomputed after transformation. Strength zero is the
original image. Mixing strength one is the partner ID image. Noise has pixel
standard deviation `strength * noise_std`, followed by clipping.

The example fits density and bootstrap heads on clean training data and fixes
thresholds from a separate clean calibration pool. It records actual scores and
EU proxies rather than inventing noisy logits. Transformation severity is not
a semantic OOD label. No adapter is trained yet.

## Optional older finite-cell toy

From the same folder:

```python
from novel_variations import run
result = run(kind='interpolate', novelty=0.25, delta=0.1)
```

That example samples discrete cells and synthetic logits. `novelty` moves a source
law away from the known convex hull; `delta` changes source prevalence. Do not
interpret its artificial scores as WRN EU.

See `notes/ICML_ADAPTER_NOTE.md` for the research framing and `wrn_compare/README.md`
for the statistical protocol. The saved one-seed smoke run is not a full experiment.
