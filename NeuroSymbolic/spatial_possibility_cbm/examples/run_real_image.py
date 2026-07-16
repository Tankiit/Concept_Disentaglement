"""
examples/run_real_image.py
----------------------------
End-to-end, first real run: one CUB image -> CLIP part-prototype scoring
(head/wing/tail) -> your real dps_transform.py -> spatial_possibility_cbm's
SDD composition -> a possibility-interval WMC for "bird", plus a sanity
check of whether the gated evidence patches actually overlap the
ground-truth box.

Two modes:

  --mock   Runs ANYWHERE (this sandbox included). Fake image, fake CLIP
           embeddings, fake dps_transform stub. Verifies the wiring/control
           flow only — the numbers mean nothing. Use this FIRST to confirm
           nothing crashes before burning time on the real path.

  --real   Runs on YOUR machine only. Needs:
             1. Your LAST-ViT repo on disk (for common.py / CUB loading)
             2. `pip install -e ".[clip]"` (torch, transformers, pillow)
             3. Network access to huggingface.co (this sandbox has none)
             4. Your own dps_transform.py importable (not bundled — see README)

Usage:
    python examples/run_real_image.py --mock
    python examples/run_real_image.py --real --repo-root /path/to/LAST-ViT --idx 2

WHAT TO LOOK AT FIRST on a real run: the printed WMC interval, and the
per-sub-concept "gated patches" vs "GT box overlap" lines. If the interval
is degenerate ([0,0] or [~1,~1] regardless of image) or gated patches never
overlap the box, that's the signal to revisit w_gate / the weight_interval()
convention BEFORE investing more time elsewhere — see the open design
questions in concept_composition_sdd.py.
"""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

import numpy as np

from spatial_possibility_cbm.clip_part_prototypes import (
    load_clip, encode_clip_patch_embeddings, encode_clip_text_prototype,
    compose_bird_concept_from_clip, PART_PROMPT_TEMPLATES,
    _dps_stub_for_interface_smoke_test,
)


def run_mock():
    print("=== MOCK RUN — interface/control-flow check only, numbers are meaningless ===\n")
    rng = np.random.default_rng(0)
    proj_dim = 512
    mock_clip_patches = rng.normal(0, 1, size=(196, proj_dim))
    mock_text_embeds = {name: rng.normal(0, 1, size=proj_dim) for name in PART_PROMPT_TEMPLATES}

    subconcepts, concept_node, (lo, hi) = compose_bird_concept_from_clip(
        mock_clip_patches, mock_text_embeds, dps_fn=_dps_stub_for_interface_smoke_test,
    )

    # fake a GT patch set just to exercise the overlap-reporting code path
    fake_gt_patch_set = set(rng.choice(196, size=20, replace=False).tolist())
    report_gating(subconcepts, fake_gt_patch_set)
    print("\nMock run completed with no errors. Wiring looks sound — now try --real.")


def run_real(repo_root: str, idx: int, seed: int, weights_filename: str, device: str | None):
    import torch
    from PIL import Image

    # 1. Load one real CUB image via your repo's common.py, exactly as the notebook does.
    sys.path.insert(0, str(Path(repo_root) / "visual_clarity" / "src"))
    import common as C  # noqa: E402  (your repo's module, path-injected above)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    items = C.load_split(test_only=True)
    random.seed(seed)
    random.shuffle(items)
    img_path, box = items[idx]
    print(f"image: {Path(img_path).stem}  (idx={idx}, seed={seed})")

    pil = Image.open(img_path).convert("RGB")
    ow, oh = pil.size
    gt_patch_set = set(C.bbox_patch_set(box, ow, oh))  # ground-truth patch indices, for the sanity check
    print(f"GT box covers {len(gt_patch_set)} / {C.GRID * C.GRID} patches")

    # 2. Real CLIP scoring — separate vision tower from LAST-ViT, see
    #    clip_part_prototypes.py's module docstring for why.
    print("\nloading CLIP (openai/clip-vit-base-patch16)...")
    clip_model, clip_processor = load_clip(device)
    clip_patch_embeds = encode_clip_patch_embeddings(clip_model, clip_processor, pil, device)
    print(f"CLIP patch embeddings: {clip_patch_embeds.shape}")

    text_embeds = {
        name: encode_clip_text_prototype(clip_model, clip_processor, prompt, device)
        for name, prompt in PART_PROMPT_TEMPLATES.items()
    }

    # 3. Real dps_transform.py — NOT bundled, must be importable on your machine.
    try:
        from dps_transform import consonant_possibility_from_samples as dps_fn
        print("using your real dps_transform.consonant_possibility_from_samples")
    except ImportError:
        print("WARNING: dps_transform.py not found on your PYTHONPATH — "
              "falling back to the smoke-test stub. Numbers below are NOT real. "
              "Put dps_transform.py somewhere importable and re-run.")
        dps_fn = _dps_stub_for_interface_smoke_test

    # 4. SDD composition.
    print()
    subconcepts, concept_node, (lo, hi) = compose_bird_concept_from_clip(
        clip_patch_embeds, text_embeds, dps_fn=dps_fn,
    )
    print(f"\n[{Path(img_path).stem}] bird concept possibility-WMC interval: [{lo:.4f}, {hi:.4f}]")

    # 5. Sanity check: does the evidence the concept actually fired on
    #    overlap the ground-truth bird box, or is it gating in background?
    report_gating(subconcepts, gt_patch_set)


def report_gating(subconcepts, gt_patch_set: set[int]):
    print("\n--- gating diagnostics (does evidence land on the object?) ---")
    for sc in subconcepts:
        candidates = set(int(i) for i in sc.candidate_patches())
        overlap = candidates & gt_patch_set
        frac = len(overlap) / len(candidates) if candidates else float("nan")
        print(f"  {sc.name:6s}: {len(candidates):3d} gated patches, "
              f"{len(overlap):3d} overlap GT box ({frac:.0%})" if candidates
              else f"  {sc.name:6s}: 0 gated patches — sub-concept is UNSATISFIABLE "
                   f"(check w_gate={sc.w_gate}, may be too strict)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="Run anywhere with fake data (interface check).")
    mode.add_argument("--real", action="store_true", help="Run for real — needs your machine, see module docstring.")
    parser.add_argument("--repo-root", type=str, default=".",
                         help="Path to your LAST-ViT repo root (contains visual_clarity/src/common.py).")
    parser.add_argument("--idx", type=int, default=2, help="Which CUB test-split image (matches the notebook's default).")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed (matches the notebook's default).")
    parser.add_argument("--weights", type=str, default="ViT_190k.pth",
                         help="LAST-ViT weights filename (unused by CLIP scoring itself, kept for future GT-overlay use).")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu | mps (default: auto-detect).")
    args = parser.parse_args()

    if args.mock:
        run_mock()
    else:
        run_real(args.repo_root, args.idx, args.seed, args.weights, args.device)
