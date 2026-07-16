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
| `clip_part_prototypes.py` — OpenCLIP-based part-prototype scorer | **Real implementation** (not stubs). Untested end-to-end here (no network access to huggingface.co in this sandbox) — `--mock` mode confirms the wiring/control-flow; run `--real` on your machine before trusting numbers. |
| `dps_transform.py` (consonant possibility π, specificity width W) | **Not bundled.** Bring your own from your earlier session outputs — this package's `ensemble_fn` injection points are designed to plug it in without modification. |
| Real LAST-ViT patch tokens | Not yet run against a real image — see your `LAST_ViT_walkthrough.ipynb` for the loading code (`common.py`'s `build_model`, `EVAL_TF`, etc.) |

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

## Run on a real image (start here)

```bash
# 1. Sanity-check the wiring anywhere (no network, no GPU, no real data needed):
python examples/run_real_image.py --mock

# 2. On your own machine, with your LAST-ViT repo, network access for OpenCLIP,
#    and your real dps_transform.py importable:
pip install -e ".[openclip]"
python examples/run_real_image.py --real --repo-root /path/to/LAST-ViT --idx 2
```

The real run prints the possibility-WMC interval for one CUB image AND a
gating diagnostic — what fraction of each sub-concept's gated patches
actually fall inside the ground-truth box. **Look at this before anything
else.** If the interval is degenerate (always ~[0,0] or ~[1,1] regardless
of image) or gated patches never overlap the box, that's a signal to
revisit `w_gate` or the `weight_interval()` convention before investing
time elsewhere — see `concept_composition_sdd.py`'s open design questions.

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

1. **Combine rule is OR-only.** "At least K of N" and spatial-contiguity
   constraints are still TODO — see the file for why both stay compatible
   with the exact two-point WMC (monotonicity), while contiguity needs a
   bit more care.
2. **No negation, by design.** This is what keeps `wmc_interval()` exact
   rather than approximate. Don't add a NOT without re-deriving the
   interval math (see the docstring — you'd need real constrained
   optimization over the credal set, à la PASOCS/PASTA, instead).
3. **`weight_interval()`'s π ± W/2 band is invented for this package** —
   not yet reconciled against your AAAI paper's exact Theorem 1
   specificity-width construction.
4. **CLIP/LAST-ViT grid alignment is asserted, not verified** — confirm
   both pipelines use the same resize/crop before trusting patch index
   correspondence.

## Relationship to Neural Probabilistic Circuits (NPC)

NPC's knowledge-injected circuit (Chen et al., 2025) is recoverable as a
restricted instance of this framework: applying the standard weighted-CNF
encoding of a categorical rule mixture (Chavira & Darwiche, 2008) to NPC's
flat rule-selector structure gives exactly our composition with
`W ≡ 0` (see `tests/test_concept_composition.py::test_zero_width_collapses_to_point_wmc`)
and a single-level vtree. This package generalizes along both the
uncertainty axis (point → credal interval) and the structural axis
(flat → arbitrary nested Boolean hierarchy).
