from __future__ import annotations

"""
extract_features_lastvit.py

Feature extraction wired to YOUR real common.py (from LAST_ViT_walkthrough.ipynb),
not a generic HF ViT. LAST-ViT is C.build_model(device, weights) and returns a
dict with "cls_token" and "patch_tokens" -- so we read those directly.

Caches per image:  cls (d,) | mean (d,) | patches (GRID*GRID, d)
Explains the ACTUAL classifier (LAST-ViT), which is what a CBM must do.

Run on your machine (needs the LAST-ViT repo + weights on disk; this sandbox
has neither). Point WEIGHTS at ViT_190k.pth for the trained model, or None for
the stock-ViT baseline.
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


# --- make YOUR common.py importable (mirrors the notebook's resolver) --------
def import_common(repo_root: Path | None = None):
    root = repo_root or Path.cwd()
    if root.name != "LAST-ViT":
        root = next(
            (p for p in [root, *root.parents] if (p / "visual_clarity").exists()),
            None,
        )
    if root is None or not (root / "visual_clarity" / "src" / "common.py").exists():
        raise FileNotFoundError(
            "Could not find LAST-ViT/visual_clarity/src/common.py; pass "
            "--repo-root /path/to/LAST-ViT"
        )
    sys.path.insert(0, str(root / "visual_clarity" / "src"))
    import common as C

    return C


def standalone_common():
    """Minimal LAST-ViT-compatible API for the released torchvision checkpoint."""
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    class TokenViT(torch.nn.Module):
        def __init__(self, checkpoint_path):
            super().__init__()
            self.model = vit_b_16(weights=None)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("model", checkpoint)
            state = {
                (key.removeprefix("model.")): value for key, value in state.items()
            }
            self.model.load_state_dict(state, strict=True)

        def forward(self, x):
            # torchvision's public forward discards patch tokens. Reproduce its
            # encoder path and retain all 197 normalized output tokens.
            x = self.model._process_input(x)
            cls = self.model.class_token.expand(x.shape[0], -1, -1)
            tokens = self.model.encoder(torch.cat([cls, x], dim=1))
            return {"cls_token": tokens[:, 0], "patch_tokens": tokens[:, 1:]}

    def build_model(device, weights):
        if not weights:
            raise ValueError("The standalone loader requires --weights /path/to/checkpoint")
        return TokenViT(weights).to(device)

    return SimpleNamespace(
        GRID=14,
        EVAL_TF=ViT_B_16_Weights.IMAGENET1K_V1.transforms(),
        build_model=build_model,
    )


def _key(model_tag, split, item_id):
    h = hashlib.sha1(f"{model_tag}|{split}|{item_id}".encode()).hexdigest()[:16]
    return f"{h}.npz"


def extract_cub(C, model, device, split_items, model_tag, split, cache_root: Path):
    """
    split_items: [(image_path, box), ...] from C.load_split(...)
    Uses C.EVAL_TF for the model input -> same preprocessing as the notebook,
    so patch-index correspondence with the GT box logic is guaranteed.
    """
    cache_dir = cache_root / model_tag / split
    cache_dir.mkdir(parents=True, exist_ok=True)

    for img_path, _box in split_items:
        iid = Path(img_path).stem
        f = cache_dir / _key(model_tag, split, iid)
        if f.exists():
            continue
        pil = Image.open(img_path).convert("RGB")
        x = C.EVAL_TF(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)  # dict: cls_token, patch_tokens
        cls = out["cls_token"].squeeze(0).cpu().numpy()  # (d,)
        patches = out["patch_tokens"].squeeze(0).cpu().numpy()  # (GRID*GRID, d)
        assert patches.shape[0] == C.GRID * C.GRID, (
            f"patch count {patches.shape[0]} != {C.GRID}^2 -- grid mismatch"
        )
        np.savez_compressed(
            f,
            cls=cls.astype(np.float32),
            mean=patches.mean(0).astype(np.float32),
            patches=patches.astype(np.float32),
        )
    return cache_dir


def load_derm7pt(data_root: Path, split: str):
    """Return ``(stable item id, image path)`` pairs for a Derm7pt split."""
    split_file = {
        "train": "train_indexes.csv",
        "valid": "valid_indexes.csv",
        "test": "test_indexes.csv",
    }[split]
    meta_file = data_root / "meta" / "meta.csv"
    index_file = data_root / "meta" / split_file
    if not meta_file.is_file() or not index_file.is_file():
        raise FileNotFoundError(
            f"Expected {meta_file} and {index_file} (set --data-root)."
        )

    with meta_file.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with index_file.open(newline="", encoding="utf-8-sig") as f:
        index_rows = list(csv.DictReader(f))

    items = []
    for index_row in index_rows:
        # The official Derm7pt split files contain zero-based meta.csv indexes
        # (train includes 0 and test reaches len(meta)-1).
        raw_index = index_row.get("indexes", next(iter(index_row.values())))
        iloc = int(raw_index)
        if not 0 <= iloc < len(rows):
            raise IndexError(f"Derm7pt row index {iloc} is out of range")
        row = rows[iloc]
        relpath = row["derm"]
        image_path = data_root / "images" / relpath
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing Derm7pt image: {image_path}")
        # Include the relative path so the key remains unique even if case_num is absent.
        item_id = f"{row.get('case_num', iloc + 1)}::{relpath}"
        items.append((item_id, image_path))
    return items


def extract_derm7pt(C, model, device, items, model_tag, split, cache_root: Path):
    cache_dir = cache_root / "derm7pt" / model_tag / split
    cache_dir.mkdir(parents=True, exist_ok=True)

    for item_id, image_path in items:
        cache_file = cache_dir / _key(model_tag, split, item_id)
        if cache_file.exists():
            continue
        x = C.EVAL_TF(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
        cls = out["cls_token"].squeeze(0).detach().cpu().numpy()
        patches = out["patch_tokens"].squeeze(0).detach().cpu().numpy()
        assert patches.ndim == 2 and patches.shape[0] == C.GRID * C.GRID, (
            f"{image_path}: patch shape {patches.shape} does not contain "
            f"{C.GRID}^2 patch tokens; fix out['patch_tokens']"
        )
        spread = float(np.ptp(patches, axis=0).max())
        assert spread > 1e-6, (
            f"{image_path}: patch rows are identical (max feature spread={spread:g}); "
            "out['patch_tokens'] appears to be a tiled pooled vector"
        )
        np.savez_compressed(
            cache_file,
            cls=cls.astype(np.float32),
            mean=patches.mean(0).astype(np.float32),
            patches=patches.astype(np.float32),
        )
    return cache_dir


def load_pooled_by_id(cache_dir, item_ids, model_tag, split, pooling="mean"):
    return np.stack(
        [np.load(cache_dir / _key(model_tag, split, iid))[pooling] for iid in item_ids]
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        default=None,
        help="ViT_190k.pth for LAST-ViT; omit for stock-ViT baseline",
    )
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--dataset", choices=["derm7pt", "cub"], default="derm7pt")
    ap.add_argument(
        "--split", choices=["train", "valid", "test", "both", "all"], default="both"
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/tanmoy/research/data/derm7pt"),
    )
    ap.add_argument("--cache", type=Path, default=Path(".cache/vit_features"))
    args = ap.parse_args()

    try:
        C = import_common(args.repo_root)
    except FileNotFoundError:
        C = standalone_common()
        print("LAST-ViT source not found; using exact torchvision ViT-B/16 token path")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if args.weights == "lastvit":
        weights_dir = getattr(C, "WEIGHTS_DIR", Path(".cache/lastvit/weights"))
        weights = str(weights_dir / "ViT_190k.pth")
    else:
        weights = args.weights
    model = C.build_model(device, weights).eval()
    tag = "lastvit_190k" if weights else "stock_vit"

    if args.dataset == "cub":
        if args.split == "train":
            ap.error("CUB extraction currently exposes only the test split")
        items = C.load_split(test_only=True)
        cdir = extract_cub(C, model, device, items, tag, "test", args.cache)
        print("cached ->", cdir, "| items", len(items), "| tag", tag, "| grid", C.GRID)
    else:
        if args.split == "both":
            splits = ("train", "test")
        elif args.split == "all":
            splits = ("train", "valid", "test")
        else:
            splits = (args.split,)
        for split in splits:
            items = load_derm7pt(args.data_root, split)
            cdir = extract_derm7pt(C, model, device, items, tag, split, args.cache)
            print("cached ->", cdir, "| items", len(items), "| tag", tag, "| grid", C.GRID)
