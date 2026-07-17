"""Run real probe-derived possibilistic intervals on held-out Derm7pt images."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from interval_composition import compose_weighted, interval_from_ensemble
from run_probe_scorer_derm7pt import CONCEPTS, cache_key


# Classical seven-point checklist weights in the repository's concept order.
DERM7PT_WEIGHTS = np.asarray([2, 1, 1, 1, 1, 2, 2], dtype=np.float32)


def load_test(data_root: Path, cache_root: Path, model_tag: str):
    with (data_root / "meta" / "meta.csv").open(newline="", encoding="utf-8-sig") as f:
        meta = list(csv.DictReader(f))
    with (data_root / "meta" / "test_indexes.csv").open(
        newline="", encoding="utf-8-sig"
    ) as f:
        indexes = [int(row["indexes"]) for row in csv.DictReader(f)]

    patches, rows = [], []
    for index in indexes:
        row = meta[index]
        item_id = f"{row.get('case_num', index + 1)}::{row['derm']}"
        path = cache_root / "test" / cache_key(model_tag, "test", item_id)
        with np.load(path) as cached:
            patch = cached["patches"]
        assert patch.shape == (196, 768), patch.shape
        assert np.ptp(patch, axis=0).max() > 1e-6
        patches.append(patch)
        rows.append(row)
    return np.stack(patches), rows


def concept_samples(
    normalized_patches: torch.Tensor,
    direction: np.ndarray,
    concept_name: str,
    M: int,
    noise: float,
    temp: float,
) -> np.ndarray:
    seed = int(hashlib.sha1(concept_name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    directions = direction[None, :] + rng.normal(
        0.0, noise, size=(M, direction.size)
    ).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8
    d = torch.from_numpy(directions).to(normalized_patches.device)
    # N x M x P: each ensemble direction scores every real patch.
    cosines = torch.einsum("npd,md->nmp", normalized_patches, d)
    probabilities = torch.sigmoid(
        temp * (cosines - cosines.mean(dim=2, keepdim=True))
    )
    # The real probe headline uses max-patch image readout.
    return probabilities.max(dim=2).values.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/Users/tanmoy/research/data/derm7pt"))
    ap.add_argument("--cache-root", type=Path, default=Path(".cache/vit_features/derm7pt/lastvit_190k"))
    ap.add_argument("--probe", type=Path, default=Path(".cache/probes/derm7pt_lastvit_mean.npz"))
    ap.add_argument("--model-tag", default="lastvit_190k")
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--temp", type=float, default=10.0)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/derm7pt_probe_intervals.json"))
    args = ap.parse_args()

    patches, rows = load_test(args.data_root, args.cache_root, args.model_tag)
    with np.load(args.probe) as probe:
        W = probe["W"].astype(np.float32)
        names = probe["concept_names"].tolist()
    assert names == CONCEPTS
    assert W.shape == (7, 768)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    x = torch.from_numpy(patches).to(device)
    x = x / (torch.linalg.vector_norm(x, dim=2, keepdim=True) + 1e-8)

    per_concept_samples = []
    for c, name in enumerate(CONCEPTS):
        samples = concept_samples(x, W[c], name, args.M, args.noise, args.temp)
        assert samples.shape == (len(rows), args.M)
        per_concept_samples.append(samples)
    samples = np.stack(per_concept_samples, axis=1)  # N,C,M

    composed = []
    concept_intervals = np.empty((len(rows), len(CONCEPTS), 2), dtype=np.float32)
    for i in range(len(rows)):
        intervals = []
        for c in range(len(CONCEPTS)):
            interval = interval_from_ensemble(samples[i, c])
            intervals.append(interval)
            concept_intervals[i, c] = interval
        composed.append(
            compose_weighted(intervals, DERM7PT_WEIGHTS, threshold=args.threshold)
        )

    point = np.asarray([value["point"] for value in composed])
    lower = np.asarray([value["N"] for value in composed])
    upper = np.asarray([value["Pi"] for value in composed])
    width = upper - lower
    decisions = [value["decision"] for value in composed]
    seven_score = np.asarray([float(row["seven_point_score"]) for row in rows])
    normalized_score = np.clip(seven_score / DERM7PT_WEIGHTS.sum(), 0.0, 1.0)
    melanoma = np.asarray(
        ["melanoma" in row["diagnosis"].strip().lower() for row in rows],
        dtype=np.int64,
    )
    rho, rho_p = spearmanr(point, seven_score)
    auc = roc_auc_score(melanoma, point)

    result = {
        "contract": (
            "real held-out LAST-ViT patches -> train-fitted probe directions -> "
            "M perturbed max-patch scores -> [min,max] -> weighted composition"
        ),
        "n_test": len(rows),
        "M": args.M,
        "noise": args.noise,
        "temperature": args.temp,
        "concept_names": CONCEPTS,
        "weights": DERM7PT_WEIGHTS.tolist(),
        "normalized_threshold": args.threshold,
        "metrics": {
            "point_vs_seven_point_spearman_rho": float(rho),
            "point_vs_seven_point_spearman_p": float(rho_p),
            "point_melanoma_auc": float(auc),
            "point_mean": float(point.mean()),
            "point_std": float(point.std()),
            "point_min": float(point.min()),
            "point_max": float(point.max()),
            "mean_interval_width": float(width.mean()),
            "median_interval_width": float(np.median(width)),
            "empirical_target_coverage": float(
                ((normalized_score >= lower) & (normalized_score <= upper)).mean()
            ),
            "melanoma_count": int(melanoma.sum()),
        },
        "decisions": {
            name: int(decisions.count(name))
            for name in ("positive", "possible", "negative")
        },
        "per_concept": {
            name: {
                "mean_N": float(concept_intervals[:, c, 0].mean()),
                "mean_Pi": float(concept_intervals[:, c, 1].mean()),
                "mean_width": float(
                    (concept_intervals[:, c, 1] - concept_intervals[:, c, 0]).mean()
                ),
            }
            for c, name in enumerate(CONCEPTS)
        },
        "caveat": (
            "The fitted probes encode any non-absent criterion state, whereas the "
            "clinical seven-point score distinguishes atypical/irregular states. "
            "Intervals are ensemble-derived and uncalibrated."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
