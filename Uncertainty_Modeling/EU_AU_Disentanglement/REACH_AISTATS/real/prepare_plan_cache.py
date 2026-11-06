"""Regenerate the plan's CLIP ViT-B/16 CIFAR-10 cache, with resumable shards.

Run from REACH_AISTATS using the reach-aistats Python environment:
    python real/prepare_plan_cache.py
The OpenAI checkpoint and standard open_clip evaluation transform are used.
This is a fresh extraction, not a reproduction of an unavailable original cache.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache/torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def validate(X, y, expected_y):
    assert X.shape == (len(expected_y), 512), X.shape
    assert np.array_equal(y, expected_y), "Label/order mismatch"
    assert np.isfinite(X).all(), "Nonfinite features"
    assert np.allclose(np.linalg.norm(X.astype(np.float32), axis=1), 1, atol=1e-3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "cache/features")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--raw-cache", type=Path, help="Raw CIFAR-10 NPZ: x_train/y_train/x_test/y_test")
    p.add_argument("--limit", type=int, help="Smoke test only; requires a separate output directory")
    args = p.parse_args()
    if args.limit and args.out.resolve() == (ROOT / "cache/features").resolve():
        p.error("Use --out cache/smoke_features with --limit")
    if args.batch_size < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        p.error("Batch size, threads and limit must be positive")
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    print("Loading OpenAI CLIP ViT-B/16 (512 dimensions)", flush=True)
    model, _, transform = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", device=args.device, force_quick_gelu=True,
        cache_dir=str(ROOT / "cache/weights"))
    model.eval()
    weight_hash = hashlib.sha256()
    for key, tensor in sorted(model.visual.state_dict().items()):
        weight_hash.update(key.encode())
        weight_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    spec = dict(model="CLIP ViT-B/16", pretrained="openai", dimensions=512,
                visual_state_sha256=weight_hash.hexdigest(),
                transform=repr(transform), normalization="L2", storage="float16",
                inference_dtype="float32", activation="QuickGELU", device=args.device, limit=args.limit,
                dataset="CIFAR10", order="source split order",
                raw_cache_sha256=sha256(args.raw_cache) if args.raw_cache else None,
                versions={k: importlib.metadata.version(k) for k in
                          ("torch", "torchvision", "open_clip_torch", "numpy")})
    spec_path = args.out / "clip_b16_extraction.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != spec:
            raise RuntimeError("Extraction specification changed; use a new output directory")
    else:
        if any((args.out / n).exists() for n in
               ("features.pt", "cifar10_train.npz", "cifar10_test.npz")):
            raise RuntimeError("Refusing to replace caches without matching provenance")
        atomic_json(spec_path, spec)
    bundle, artifacts = {}, {}
    start_time = time.monotonic()
    for split in ("train", "test"):
        if args.raw_cache:
            with np.load(args.raw_cache, allow_pickle=False) as raw:
                images, labels = raw[f"x_{split}"], raw[f"y_{split}"]
            size = 50000 if split == "train" else 10000
            assert images.shape == (size, 32, 32, 3) and images.dtype == np.uint8
            assert labels.shape == (size,) and np.issubdtype(labels.dtype, np.integer)
            assert np.array_equal(np.bincount(labels, minlength=10), np.full(10, size // 10))
            get_image = lambda i: Image.fromarray(images[i])
        else:
            ds = CIFAR10(str(ROOT / "cache/cifar10"), train=split == "train", download=True)
            size, labels = len(ds), np.asarray(ds.targets)
            get_image = lambda i: ds[i][0]
        n = min(size, args.limit) if args.limit else size
        expected_y = np.asarray(labels[:n], dtype=np.int64)
        shards = args.out / "clip_b16_shards" / split
        shards.mkdir(parents=True, exist_ok=True)
        parts = []
        for start in range(0, n, 256):
            stop = min(start + 256, n)
            path = shards / f"{start:06d}_{stop:06d}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as b:
                    X, y = b["X"], b["y"]
                validate(X, y, expected_y[start:stop])
            else:
                batches = []
                with torch.inference_mode():
                    for first in range(start, stop, args.batch_size):
                        x = torch.stack([transform(get_image(i)) for i in
                                         range(first, min(first + args.batch_size, stop))])
                        z = F.normalize(model.encode_image(x.to(args.device)).float(), dim=-1)
                        batches.append(z.cpu().numpy().astype(np.float16))
                X, y = np.concatenate(batches), expected_y[start:stop]
                validate(X, y, expected_y[start:stop])
                tmp = path.with_suffix(".tmp")
                with tmp.open("wb") as f:
                    np.savez(f, X=X, y=y)
                tmp.replace(path)
            parts.append(X)
            print(f"{split}: {stop}/{n}; elapsed {time.monotonic()-start_time:.1f}s", flush=True)
        X = np.concatenate(parts)
        validate(X, expected_y, expected_y)
        name = f"cifar10_{split}"
        path = args.out / f"{name}.npz"
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.savez(f, X=X, y=expected_y)
        tmp.replace(path)
        bundle[name] = dict(X=torch.from_numpy(X), y=torch.from_numpy(expected_y))
        artifacts[path.name] = dict(sha256=sha256(path), shape=list(X.shape),
                                    class_counts=np.bincount(expected_y, minlength=10).tolist())
    path = args.out / "features.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(bundle, tmp)
    tmp.replace(path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    for name in bundle:
        assert torch.equal(loaded[name]["X"], bundle[name]["X"])
        assert torch.equal(loaded[name]["y"], bundle[name]["y"])
    artifacts[path.name] = dict(sha256=sha256(path))
    atomic_json(args.out / "clip_b16_manifest.json", dict(
        specification=spec, artifacts=artifacts, complete=not bool(args.limit),
        script_sha256=sha256(Path(__file__)),
        note="Fresh extraction; original collaborator cache/preprocessing unavailable."))
    print("Cache exported and validated:", args.out, flush=True)


if __name__ == "__main__":
    main()
