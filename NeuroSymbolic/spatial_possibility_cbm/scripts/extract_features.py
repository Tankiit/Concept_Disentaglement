from __future__ import annotations

"""
extract_features.py

ONE feature-extraction pass over a dataset split. Per image, caches:
  cls     : (d,)      CLS token           -> probe pooling option 1
  mean    : (d,)      mean of patch tokens -> probe pooling option 2
  patches : (196, d)  full patch grid      -> per-patch scoring (the spatial part)

Both pooling options cached so you can compare held-out probe AUC (Q2: keep both).
Resumable: skips items already on disk. Cache key includes model tag + split.

Backends:
  --backend hf    : transformers ViTModel (default). Reads hidden states; sdpa is
                    fine here (eager was only needed for the ATTENTION ensemble).
  --backend timm  : timm.create_model(...).forward_features -> strip prefix tokens.

Point --model at your LAST-ViT checkpoint to explain the ACTUAL classifier;
a base ViT-B/16 is a stand-in that loads with no local checkpoint.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Backends: each returns (cls (d,), patches (n_patches, d)) for a batch of images
# ---------------------------------------------------------------------------
class HFViT:
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", device="cpu"):
        import torch
        from transformers import AutoImageProcessor, ViTModel

        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        self.model = ViTModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @property
    def dim(self):
        return self.model.config.hidden_size

    def encode(self, pil_images):
        import torch

        px = self.proc(images=pil_images, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            out = self.model(pixel_values=px).last_hidden_state  # (B, 1+P, d)
        cls = out[:, 0, :]  # (B, d)
        patches = out[:, 1:, :]  # (B, P, d)
        return cls.cpu().numpy(), patches.cpu().numpy()


class TimmViT:
    def __init__(self, model_name="vit_base_patch16_224", device="cpu"):
        import timm
        import torch

        self.torch = torch
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
        self.n_prefix = getattr(self.model, "num_prefix_tokens", 1)  # cls (+reg) tokens
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.tf = timm.data.create_transform(**cfg)
        self.device = device

    @property
    def dim(self):
        return self.model.num_features

    def encode(self, pil_images):
        import torch

        x = torch.stack([self.tf(im) for im in pil_images]).to(self.device)
        with torch.no_grad():
            tokens = self.model.forward_features(x)  # (B, prefix+P, d)
        cls = tokens[:, 0, :]
        patches = tokens[:, self.n_prefix:, :]  # strip cls/reg tokens
        return cls.cpu().numpy(), patches.cpu().numpy()


# ---------------------------------------------------------------------------
def _key(model_tag, split, item_id):
    h = hashlib.sha1(f"{model_tag}|{split}|{item_id}".encode()).hexdigest()[:16]
    return f"{h}.npz"


def extract_split(backend, items, model_tag, split, cache_root: Path, batch_size=16):
    """
    items: iterable of (item_id, PIL.Image). Returns the cache dir.
    Caches cls/mean/patches per item; skips existing.
    """
    cache_dir = cache_root / model_tag.replace("/", "_") / split
    cache_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        (iid, im)
        for (iid, im) in items
        if not (cache_dir / _key(model_tag, split, iid)).exists()
    ]
    for i in range(0, len(pending), batch_size):
        chunk = pending[i : i + batch_size]
        cls, patches = backend.encode([im for _, im in chunk])  # (b,d), (b,P,d)
        for j, (iid, _) in enumerate(chunk):
            np.savez_compressed(
                cache_dir / _key(model_tag, split, iid),
                cls=cls[j].astype(np.float32),
                mean=patches[j].mean(axis=0).astype(np.float32),  # pooling opt 2
                patches=patches[j].astype(np.float32),  # spatial grid
            )
    return cache_dir


def load_pooled(cache_dir: Path, item_ids, pooling="mean"):
    """Stack pooled features for probe fitting. pooling in {'cls','mean'}."""
    feats = []
    for iid in item_ids:
        # NB: key needs the same model_tag/split as extract -> caller passes matching dir
        f = next(cache_dir.glob("*.npz"))  # placeholder; real caller keys by iid
        feats.append(np.load(f)[pooling])
    return np.stack(feats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "timm"], default="hf")
    ap.add_argument("--model", default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cache", type=Path, default=Path(".cache/vit_features"))
    # TODO(tanmoy): wire your derm7pt loader here -> yields (item_id, PIL image)
    args = ap.parse_args()
    backend = HFViT(args.model) if args.backend == "hf" else TimmViT(args.model)
    print(f"backend={args.backend} dim={backend.dim}")
    # items = your_loader(args.split)
    # cdir = extract_split(backend, items, args.model, args.split, args.cache)
    # print("cached to", cdir)
