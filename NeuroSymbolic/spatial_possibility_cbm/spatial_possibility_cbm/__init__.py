"""
spatial_possibility_cbm — patch-grounded possibilistic/credal concept composition
for LAST-ViT, compiled via pySDD and cross-validated against PASP credal
semantics (PASTA).

Public API:
    ConceptComposer, SubConcept, PatchEvidence, get_patch_evidence,
    compose_bird_concept        — spatial_possibility_cbm.concept_composition_sdd
    ClipConceptDirectionScorer,
    compose_bird_concept_from_clip — spatial_possibility_cbm.clip_part_prototypes

NOT bundled (bring your own, from your existing session outputs):
    patch_uq.py / the possibilistic-CAM ensemble of your choice

Bundled locally in this repo:
    dps_transform.py            — consonant_possibility_from_samples

Optional heavier dependencies (install as needed, not required for the
core SDD/credal-WMC machinery):
    pysdd          — required for concept_composition_sdd
    transformers,
    torch, pillow  — required only for clip_part_prototypes' real CLIP path
"""

from spatial_possibility_cbm.concept_composition_sdd import (
    ConceptComposer,
    SubConcept,
    PatchEvidence,
    get_patch_evidence,
    compose_bird_concept,
)
from spatial_possibility_cbm.learnable_span import LearnableSpanCBM, adjacency_pairs

__version__ = "0.1.0"

__all__ = [
    "ConceptComposer",
    "SubConcept",
    "PatchEvidence",
    "get_patch_evidence",
    "compose_bird_concept",
    "LearnableSpanCBM",
    "adjacency_pairs",
]
