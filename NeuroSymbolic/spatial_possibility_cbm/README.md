# spatial-possibility-cbm

Patch-grounded possibilistic concept composition for LAST-ViT: compose
hierarchical concepts (e.g. `bird = head ∧ wing ∧ tail`) where each
sub-concept fires only when a **spatially adjacent pair of gated patches**
both hold — not scattered singletons — via pySDD, with literal weights
derived from possibility theory (π, specificity width W). This gives an
**exact possibility-interval WMC** `[lower, upper]` for the composed
concept — cross-validated against PASTA (an independent PASP credal-
semantics solver) to floating-point precision, and against an independent
brute-force 2ⁿ enumeration.

**The differentiator is local geometry, not just uncertainty.** Related
"logic + concept bottleneck" work (NPC, DCR, LENs) and related "credal +
tractable circuit" work (Credal SDDs, Credal SPNs) both lack any spatial
grounding at all — see the novelty-sweep notes referenced below. This
package's `combine="contiguity"` default (see `concept_composition_sdd.py`)
is what makes "local geometry" a concrete, checkable Boolean formula rather
than a description.

This is the vision/symbolic-composition sibling to a separate NLP-domain
credal concept bottleneck model line of work — see that project's own repo
for the (deliberately different) ensemble/variational mechanism used there.


## Status (be honest with yourself before trusting numbers)

| Component | Status |
|---|---|
| `concept_composition_sdd.py` — SDD contiguity composition + two-point possibility-interval WMC | Validated 3 ways: brute-force enumeration (`tests/`), PASTA independent solver, and hand-checked example |
| `clip_part_prototypes.py` — OpenCLIP-based part-prototype scorer | Real implementation, but it is no longer the main Derm7pt path. The supervised same-space LAST-ViT probe is the evaluated scorer. |
| `dps_transform.py` (consonant possibility π, specificity width W) | Bundled and used by the legacy CLIP ladder. The newer real probe interval runners derive `[N,Π]` directly from perturbed probe-score ensembles. |
| Real LAST-ViT patch tokens | **Complete.** `ViT_190k.pth` was loaded directly through its exact torchvision ViT-B/16 architecture. Derm7pt train/valid/test and CUB train/test caches contain distinct `(196,768)` patch tokens. |
| Derm7pt supervised probe | **Complete.** Seven unit-norm `(768,)` directions fitted on train mean-pooled features and evaluated on held-out test patches. Mean per-concept test Spearman ρ is `0.415`. |
| Learned rationale gate | Runs on real Derm7pt, but **fails the faithfulness controls**. A size/connectivity-matched random-region baseline is more sufficient for every concept. Do not use this as a positive result. |
| Probe-derived real intervals | **Executed on Derm7pt and CUB.** Derm7pt is degenerate under the current centered-max transform; CUB has modest AND signal (`AUC=0.601`) but compressed point scores. |

## Install

```bash
pip install -e .                        # core (numpy, pysdd) — SDD composition only
pip install -e ".[openclip]"            # + torch, open_clip_torch, transformers, pillow — for real OpenCLIP scoring
pip install -e ".[crosscheck]"          # + clingo — for the PASTA cross-check
pip install git+https://github.com/damianoazzolini/pasta   # the pastasolver CLI itself (not on PyPI)
pip install -e ".[dev]"                 # + pytest
```

## Quick start

```python
from spatial_possibility_cbm import ConceptComposer, SubConcept, PatchEvidence
import numpy as np

subconcepts = [
    SubConcept(name="head", evidence=PatchEvidence(pi=np.array([0.9, 0.2]), W=np.array([0.1, 0.4]))),
    SubConcept(name="wing", evidence=PatchEvidence(pi=np.array([0.8, 0.1]), W=np.array([0.2, 0.45]))),
    SubConcept(name="tail", evidence=PatchEvidence(pi=np.array([0.6, 0.3]), W=np.array([0.3, 0.4]))),
]
# Patch indices 0 and 1 are adjacent on the default 14x14 grid (same row),
# so combine="contiguity" (the default) is satisfiable here. On a real
# image, patch indices come from your actual patch grid — see
# clip_part_prototypes.py's grid-alignment note before assuming this holds.
n_vars = sum(len(sc.candidate_patches()) for sc in subconcepts)
composer = ConceptComposer(num_vars_hint=max(n_vars, 1))
concept = composer.build_concept(subconcepts)
lo, hi = composer.wmc_interval(concept, subconcepts)
print(f"bird concept: [{lo:.4f}, {hi:.4f}]")
```

### Wiring in your real dps_transform.py

```python
from spatial_possibility_cbm import compose_bird_concept
from dps_transform import consonant_possibility_from_samples  # your file

def my_ensemble_fn(patch_tokens, subconcept_query):
    samples = ...  # your possibilistic-CAM ensemble, scored against subconcept_query
    return consonant_possibility_from_samples(samples)  # -> (pi, W)

concept_node, (lo, hi) = compose_bird_concept(real_last_vit_patch_tokens, ensemble_fn=my_ensemble_fn)
```

## Legacy OpenCLIP single-image path

```bash
# 1. Sanity-check the wiring anywhere (no network, no GPU, no real data needed):
python examples/run_real_image.py --mock

# 2. With a LAST-ViT checkout and OpenCLIP dependencies:
pip install -e ".[openclip]"
python examples/run_real_image.py --real --repo-root /path/to/LAST-ViT --idx 2
```

This optional legacy run prints the possibility-WMC interval for one CUB image and a
gating diagnostic — what fraction of each sub-concept's gated patches
actually fall inside the ground-truth box. **Look at this before anything
else.** If the interval is degenerate (always ~[0,0] or ~[1,1] regardless
of image) or gated patches never overlap the box, that's a signal to
revisit `w_gate` or the `weight_interval()` convention before investing
time elsewhere — see `concept_composition_sdd.py`'s open design questions.

## Real feature caches

The released checkpoint is stored outside version control at:

```text
.cache/lastvit/weights/ViT_190k.pth
```

Feature extraction uses the true encoder output after the final layer norm:
one CLS token and 196 distinct patch tokens. Every image is guarded by:

```python
assert patches.shape == (196, 768)
assert np.ptp(patches, axis=0).max() > 1e-6
```

Reproduce Derm7pt extraction:

```bash
python scripts/extract_features_lastvit.py \
  --dataset derm7pt \
  --split all \
  --weights lastvit \
  --data-root /Users/tanmoy/research/data/derm7pt
```

This produces 413 train, 203 validation, and 395 test caches under:

```text
.cache/vit_features/derm7pt/lastvit_190k/
```

Reproduce CUB extraction:

```bash
python scripts/extract_features_lastvit.py \
  --dataset cub \
  --split both \
  --weights lastvit \
  --data-root /Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011
```

This produces 5,994 train and 5,794 test caches under
`.cache/vit_features/cub/lastvit_190k/` (about 6.2 GB).

## Supervised Derm7pt probe

`ProbeConceptScorer` avoids both split leakage and patch-label broadcasting:
it fits logistic directions on one pooled train vector per image and applies
those directions to test patch tokens.

```bash
python scripts/run_probe_scorer_derm7pt.py
```

The fitted probe is `.cache/probes/derm7pt_lastvit_mean.npz`:

```text
W             (7, 768)
b             (7,)
concept_names (7,)
pooling       scalar "mean"
```

Held-out test Spearman correlations:

| Concept | ρ |
|---|---:|
| pigment network | 0.501 |
| streaks | 0.471 |
| pigmentation | 0.407 |
| regression structures | 0.294 |
| dots and globules | 0.192 |
| blue-whitish veil | 0.528 |
| vascular structures | 0.513 |

Full results are in `results/derm7pt_probe_scorer.json`.

## Rationale audit: negative result

The real rationale runner uses a 16-unit hidden gate, dropout `0.5`,
AdamW weight decay `1e-4`, a constant-then-cosine learning-rate schedule,
and a hard-mask repair that keeps the largest connected component and grows
it contiguously to a minimum of 10 patches. Hyperparameters were selected
using the 203 validation images, followed by a train+validation refit.

```bash
python scripts/run_train_rationale_derm7pt.py \
  --tune-epochs 75 \
  --epochs 200 \
  --out results/derm7pt_rationale_audit_tuned.json
```

The repair eliminates empty masks and yields test mean gated `ρ=0.444`.
However, comprehensiveness remains negative and the masks remain fragmented
before largest-component postprocessing.

The required random control uses 20 BFS-grown connected regions per image,
each with exactly the learned mask's size:

```bash
python scripts/random_region_baseline_derm7pt.py --K 20
```

Mean absolute sufficiency gap (lower is better):

```text
learned connected rationale: 0.191
matched random regions:      0.099
```

Random regions outperform the learned rationale for all seven concepts.
The learned rationale is therefore not supported as a faithful explanation.
See `results/derm7pt_random_region_baseline.json`.

## Real probe-to-interval experiments

### Derm7pt weighted composition

```bash
python scripts/run_probe_interval_derm7pt.py
```

Pipeline:

```text
real held-out patches
→ train-fitted probe directions
→ 20 perturbed max-patch scores
→ per-concept [N, Π]
→ weighted seven-point composition
```

Weights are `[2,1,1,1,1,2,2]`, with normalized threshold `0.30`.
This run is a negative diagnostic:

| Metric | Value |
|---|---:|
| Spearman ρ with seven-point score | 0.047 |
| Melanoma AUC | 0.460 |
| Mean interval width | 0.0758 |
| Empirical target coverage | 3.29% |
| Decisions | 395 positive, 0 possible, 0 negative |

The current probes encode any non-absent finding, including typical/regular
states, whereas the clinical score emphasizes atypical/irregular findings.
Also, centering patch scores and then taking their maximum forces scores above
`0.5`, making the `0.30` readout degenerate. See
`results/derm7pt_probe_intervals.json`.

### CUB AND composition

```bash
python scripts/run_probe_interval_cub.py
```

Three non-trivial CUB attributes are used: plain head pattern, rounded wings,
and notched tail. Probes fit on official train pooled features and are applied
to official test patches; their intervals are composed with outer AND.

| Metric | Value |
|---|---:|
| AND point AUC | 0.601 |
| Positive test conjunctions | 166 / 5,794 |
| Mean interval width | 0.0851 |
| Point-score standard deviation | 0.0113 |

Max-evidence patch bird-box hit rates are 76.0%, 79.4%, and 82.0%, compared
with a mean bird-box area fraction of 52.1%. This establishes preferential
object localization, not correct anatomical-part localization. See
`results/cub_probe_intervals.json`.

## Testing

```bash
pytest tests/ -v
```

`tests/test_concept_composition.py` runs without any external solver — it
includes an independent brute-force enumeration check (2^n literal
assignments, no pysdd WMC call at all) as a second ground truth beyond
PASTA. `tests/test_pasta_crosscheck.py` auto-skips if `pastasolver` isn't
on PATH.

## Known open design questions (not bugs — real unresolved decisions)

See the "Open design questions" comment block at the bottom of
`concept_composition_sdd.py` for the full list. Headline items:

1. **The real interval transformation is not calibrated.** The current
   centered-max ensemble compresses real scores and is degenerate on Derm7pt.
   Transformation and thresholds must be selected on validation data.
2. **No negation, by design.** This is what keeps `wmc_interval()` exact
   rather than approximate. Don't add a NOT without re-deriving the
   interval math (see the docstring — you'd need real constrained
   optimization over the credal set, à la PASOCS/PASTA, instead).
3. **`weight_interval()`'s π ± W/2 band is invented for this package** —
   not yet reconciled against your AAAI paper's exact Theorem 1
   specificity-width construction.
4. **CUB box hits are object-level only.** They do not establish correct
   head/wing/tail localization; use CUB part coordinates for that claim.

## Relationship to Neural Probabilistic Circuits (NPC)

NPC's knowledge-injected circuit (Chen et al., 2025) is recoverable as a
restricted instance of this framework: applying the standard weighted-CNF
encoding of a categorical rule mixture (Chavira & Darwiche, 2008) to NPC's
flat rule-selector structure gives exactly our composition with
`W ≡ 0` (see `tests/test_concept_composition.py::test_zero_width_collapses_to_point_wmc`)
and a single-level vtree. This package generalizes along both the
uncertainty axis (point → credal interval) and the structural axis
(flat → arbitrary nested Boolean hierarchy).
