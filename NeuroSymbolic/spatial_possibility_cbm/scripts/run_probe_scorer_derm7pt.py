"""Fit Derm7pt probes on train pooled features and evaluate test patches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spatial_possibility_cbm.probe_scorer import ProbeConceptScorer


CONCEPTS = [
    "pigment_network",
    "streaks",
    "pigmentation",
    "regression_structures",
    "dots_and_globules",
    "blue_whitish_veil",
    "vascular_structures",
]


def cache_key(model_tag: str, split: str, item_id: str) -> str:
    digest = hashlib.sha1(f"{model_tag}|{split}|{item_id}".encode()).hexdigest()[:16]
    return f"{digest}.npz"


def present(value: object) -> int:
    value = str(value).strip().lower()
    return int(value not in {"absent", "none", "nan", ""} and "absent" not in value)


def load_split(
    data_root: Path,
    cache_root: Path,
    model_tag: str,
    split: str,
    pooling: str,
    include_patches: bool = False,
):
    with (data_root / "meta" / "meta.csv").open(newline="", encoding="utf-8-sig") as f:
        meta = list(csv.DictReader(f))
    split_file = "valid_indexes.csv" if split in {"valid", "val"} else f"{split}_indexes.csv"
    with (data_root / "meta" / split_file).open(
        newline="", encoding="utf-8-sig"
    ) as f:
        indexes = [int(row["indexes"]) for row in csv.DictReader(f)]

    pooled, patches, labels, item_ids = [], [], [], []
    for index in indexes:
        row = meta[index]  # Derm7pt indexes are zero-based.
        item_id = f"{row.get('case_num', index + 1)}::{row['derm']}"
        item_ids.append(item_id)
        path = cache_root / split / cache_key(model_tag, split, item_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature cache: {path}")
        with np.load(path) as cached:
            pooled.append(cached[pooling])
            if include_patches:
                patch = cached["patches"]
                assert patch.shape[0] == 196
                assert np.ptp(patch, axis=0).max() > 1e-6
                patches.append(patch)
        labels.append([present(row[name]) for name in CONCEPTS])
    return np.stack(pooled), patches, np.asarray(labels, dtype=np.int64), item_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/Users/tanmoy/research/data/derm7pt"))
    parser.add_argument("--cache-root", type=Path, default=Path(".cache/vit_features/derm7pt/lastvit_190k"))
    parser.add_argument("--model-tag", default="lastvit_190k")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("results/derm7pt_probe_scorer.json"))
    parser.add_argument("--probe-out", type=Path, default=Path(".cache/probes/derm7pt_lastvit_mean.npz"))
    args = parser.parse_args()

    train_x, _, train_y, train_ids = load_split(
        args.data_root, args.cache_root, args.model_tag, "train", args.pooling
    )
    _, test_patches, test_y, test_ids = load_split(
        args.data_root, args.cache_root, args.model_tag, "test", args.pooling,
        include_patches=True,
    )
    overlap = set(train_ids) & set(test_ids)
    assert not overlap, f"train/test leakage: {len(overlap)} overlapping item IDs"
    scorer = ProbeConceptScorer(CONCEPTS).fit(train_x, train_y, l2=args.l2)

    predictions = np.empty((len(test_patches), len(CONCEPTS)), dtype=np.float32)
    for i, patch_tokens in enumerate(test_patches):
        cosines = scorer.concept_cosines(patch_tokens)
        predictions[i] = [float(cosines[name].max()) for name in CONCEPTS]

    results = {
        "contract": "fit=train pooled image features; apply=test patch features; reduce=max",
        "pooling": args.pooling,
        "n_train": int(len(train_x)),
        "n_test": int(len(test_patches)),
        "concepts": {},
    }
    for c, name in enumerate(CONCEPTS):
        rho, p = spearmanr(predictions[:, c], test_y[:, c])
        results["concepts"][name] = {
            "rho": float(rho),
            "p": float(p),
            "test_positive": int(test_y[:, c].sum()),
            "train_positive": int(train_y[:, c].sum()),
        }

    args.probe_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.probe_out,
        W=scorer.W,
        b=scorer.b,
        concept_names=np.asarray(CONCEPTS),
        pooling=args.pooling,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"probe -> {args.probe_out}")
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
