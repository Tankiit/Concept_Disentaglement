"""
clip_part_prototypes.py
------------------------
Real (non-random) `ensemble_fn` for concept_composition_sdd.py's
get_patch_evidence(), scoring LAST-ViT's 14x14 patch grid against CLIP text
prototypes for head/wing/tail.

WHY A SEPARATE VISION TOWER: LAST-ViT's patch tokens are not in the same
embedding space as CLIP text embeddings, so they can't be cosine-scored
against each other. This file runs CLIP's OWN vision tower on the same
image (same 16px patch / 224px / 14x14 grid geometry) to get CLIP patch
embeddings that a CLIP text prototype can actually be compared to. Patch
index i still refers to the same spatial location as in the LAST-ViT grid,
as long as both pipelines use the same resize/crop (256 -> center-crop 224)
before patchifying — TODO: confirm this against common.py's exact transform,
don't assume it.

NOT SMOKE-TESTED AGAINST REAL CLIP WEIGHTS: this sandbox can't reach
huggingface.co (same constraint noted for attention_possibilistic_gradcam_hf.py
in the session notes). The __main__ block below smoke-tests the *interface*
(shapes, control flow, SDD wiring) with a mocked CLIP scorer, not real
CLIP. Run against real `openai/clip-vit-base-patch16` on your machine before
trusting any numbers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np

# from dps_transform import consonant_possibility_from_samples  # TODO: wire in (real file, not the stub below)
from spatial_possibility_cbm.concept_composition_sdd import ConceptComposer, SubConcept, PatchEvidence, compose_bird_concept


# --------------------------------------------------------------------------
# CLIP loading + encoding — TODO: run for real on your machine
# --------------------------------------------------------------------------

def load_clip(device: str = "cpu"):
    """
    Loads real CLIP (openai/clip-vit-base-patch16 — ViT-B/16, same 16px/224px
    patchify as LAST-ViT, so the grid-alignment assumption at least starts
    from matching patch geometry). Needs `pip install -e ".[clip]"` and
    network access to huggingface.co — this sandbox has neither, so this
    function is written correctly but UNTESTED here. Run on your machine.
    """
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return model, processor


def encode_clip_patch_embeddings(model, processor, pil_image, device: str = "cpu") -> np.ndarray:
    """
    Returns (196, proj_dim) CLIP patch embeddings for one image, in the
    SAME shared space as encode_clip_text_prototype's output.

    IMPORTANT — two things NOT verified by this function, only asserted:
    1. Patch ORDER matches LAST-ViT's row-major 14x14 order. CLIP's ViT
       patchifies row-major too (standard ViT position-embedding convention),
       so this should hold, but a transposed grid would silently produce
       plausible-looking, wrong composed concepts — no error, just wrong
       numbers. Worth a one-time manual check (e.g. compare a known-position
       patch's embedding neighborhood) before trusting downstream results.
    2. CLIP's processor does its OWN resize/crop, which may not exactly
       match common.py's "256 -> center-crop 224" pipeline. If they diverge
       even slightly, patch index i in CLIP-space and patch index i in
       LAST-ViT-space stop referring to the same spatial location. Confirm
       processor.image_processor's resize/crop params against C.EVAL_TF /
       C.SHOW_TF before trusting patch-index correspondence.
    """
    import torch
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_out = model.vision_model(inputs["pixel_values"], output_hidden_states=False)
        patch_tokens = vision_out.last_hidden_state[:, 1:, :]   # drop CLS, keep 196 patches
        patch_embeds = model.visual_projection(patch_tokens)    # linear proj is per-token, safe
    return patch_embeds.squeeze(0).cpu().numpy()   # (196, proj_dim)


def encode_clip_text_prototype(model, processor, text: str, device: str = "cpu") -> np.ndarray:
    """Returns (proj_dim,) CLIP text embedding for one prompt string."""
    import torch
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embed = model.get_text_features(**inputs)
    return text_embed.squeeze(0).cpu().numpy()   # (proj_dim,)


PART_PROMPT_TEMPLATES = {
    # TODO: these are placeholder prompts — CUB birds have fine part vocab
    # (bill/crown/nape/throat/breast/etc.) that may score better than the
    # coarse head/wing/tail split. Consider ensembling multiple prompt
    # templates per part rather than one string each.
    "head": "a photo of a bird's head",
    "wing": "a photo of a bird's wing",
    "tail": "a photo of a bird's tail",
}


# --------------------------------------------------------------------------
# Concept-direction perturbation ensemble (mirrors
# concept_direction_possibilistic_cam.py's approach, applied to a CLIP text
# direction instead of the CLS-derived direction)
# --------------------------------------------------------------------------

@dataclass
class ClipConceptDirectionScorer:
    clip_patch_embeds: np.ndarray      # (196, proj_dim), precomputed once per image
    M: int = 20                        # TODO: untuned, same status as concept_direction_possibilistic_cam.py
    noise_scale: float = 0.05          # TODO: untuned — run the sanity check (raw cosine range) first
    temp: float = 10.0                 # TODO: untuned sigmoid sharpening temperature
    dps_fn: Callable = None            # consonant_possibility_from_samples, injected so this file
                                        # doesn't hard-depend on dps_transform.py being importable here

    def __call__(self, _last_vit_patch_tokens_unused, subconcept_query: str):
        """
        Matches concept_composition_sdd.get_patch_evidence's ensemble_fn
        signature: (patch_tokens, subconcept_query) -> (pi, W). The LAST-ViT
        patch_tokens argument is intentionally unused here — see module
        docstring for why CLIP needs its own embeddings instead.
        """
        text_embed = subconcept_query  # caller passes the already-encoded CLIP text vector
        rng = np.random.default_rng()
        text_norm = text_embed / (np.linalg.norm(text_embed) + 1e-8)

        # M perturbed copies of the text direction
        noise = rng.normal(0, self.noise_scale, size=(self.M, len(text_norm)))
        perturbed = text_norm[None, :] + noise
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True) + 1e-8   # (M, dim)

        patches_norm = self.clip_patch_embeds / (
            np.linalg.norm(self.clip_patch_embeds, axis=1, keepdims=True) + 1e-8
        )  # (196, dim)

        cos = perturbed @ patches_norm.T                     # (M, 196)
        scores = 1 / (1 + np.exp(-self.temp * cos))           # sigmoid-sharpened, (M, 196)

        # each patch's M ensemble scores -> consonant_possibility_from_samples
        # expects samples shaped (num_patches, M) per the batch-shared-denominator fix
        samples = scores.T                                     # (196, M)
        pi, W = self.dps_fn(samples)
        return pi, W


def _dps_stub_for_interface_smoke_test(samples: np.ndarray):
    """
    NOT the real dps_transform.consonant_possibility_from_samples — a
    shape-compatible stand-in so this file's control flow is testable
    without that module present. DO NOT use for any real number.
    """
    # crude possibility proxy: normalized max-minus-spread per row, shared
    # denominator across the whole batch (mirrors the real fix's spirit only)
    row_range = samples.max(axis=1) - samples.min(axis=1)
    denom = row_range.max() + 1e-8
    pi = samples.mean(axis=1)
    W = row_range / denom
    return pi, W


# --------------------------------------------------------------------------
# Wiring: build the ensemble_fn per sub-concept and hand off to
# concept_composition_sdd.compose_bird_concept
# --------------------------------------------------------------------------

def compose_bird_concept_from_clip(clip_patch_embeds: np.ndarray, text_embeds: dict[str, np.ndarray],
                                     dps_fn: Callable = None):
    """
    clip_patch_embeds: (196, proj_dim) from encode_clip_patch_embeddings()
    text_embeds: {"head": vec, "wing": vec, "tail": vec} from encode_clip_text_prototype()
    dps_fn: consonant_possibility_from_samples (or the smoke-test stub)

    Returns (subconcepts, concept_node, (lo, hi)) — subconcepts is exposed
    so callers can inspect sc.candidate_patches() per part (e.g. to check
    gated patches against a ground-truth box — see run_real_image.py).
    """
    dps_fn = dps_fn or _dps_stub_for_interface_smoke_test
    scorer = ClipConceptDirectionScorer(clip_patch_embeds=clip_patch_embeds, dps_fn=dps_fn)

    # compose_bird_concept() (in concept_composition_sdd.py) calls
    # get_patch_evidence(patch_tokens, subconcept_query=name, ensemble_fn=...)
    # per sub-concept name. We need subconcept_query to be the ENCODED text
    # vector, not the bare string, so wrap ensemble_fn to look up the right
    # precomputed embedding per part name.
    def ensemble_fn(patch_tokens, subconcept_name):
        return scorer(patch_tokens, text_embeds[subconcept_name])

    # fake_patch_tokens here is a placeholder for the LAST-ViT tensor that
    # get_patch_evidence's signature expects but this scorer ignores —
    # pass the real LAST-ViT patch_tokens from your notebook for clarity
    # even though CLIP scoring doesn't use them.
    dummy_last_vit_tokens = np.zeros((clip_patch_embeds.shape[0], 1))
    return compose_bird_concept(dummy_last_vit_tokens, ensemble_fn=ensemble_fn)


if __name__ == "__main__":
    # INTERFACE smoke test only — mocked CLIP embeddings, not real CLIP.
    rng = np.random.default_rng(0)
    proj_dim = 512
    mock_clip_patches = rng.normal(0, 1, size=(196, proj_dim))
    mock_text_embeds = {name: rng.normal(0, 1, size=proj_dim) for name in PART_PROMPT_TEMPLATES}

    subconcepts, concept_node, (lo, hi) = compose_bird_concept_from_clip(mock_clip_patches, mock_text_embeds)
    print(f"[interface smoke test, NOT real numbers] WMC interval: [{lo:.4f}, {hi:.4f}]")
