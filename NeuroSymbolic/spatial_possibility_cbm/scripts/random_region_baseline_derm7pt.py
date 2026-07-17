"""Size- and connectivity-matched random-region rationale baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_probe_scorer_derm7pt import CONCEPTS, load_split
from train_rationale import GRID, RationaleHead


def neighbors(index: int):
    r, c = divmod(index, GRID)
    if r > 0:
        yield index - GRID
    if r + 1 < GRID:
        yield index + GRID
    if c > 0:
        yield index - 1
    if c + 1 < GRID:
        yield index + 1


def random_connected_region(size: int, rng: np.random.Generator) -> np.ndarray:
    """Grow a uniformly seeded blob by randomly expanding its current frontier."""
    region = {int(rng.integers(GRID * GRID))}
    frontier = {q for p in region for q in neighbors(p) if q not in region}
    while len(region) < size:
        candidates = tuple(frontier)
        q = candidates[int(rng.integers(len(candidates)))]
        frontier.remove(q)
        region.add(q)
        frontier.update(n for n in neighbors(q) if n not in region)
    mask = np.zeros(GRID * GRID, dtype=np.float32)
    mask[list(region)] = 1.0
    return mask


def component_count(mask: np.ndarray) -> int:
    unseen = set(np.flatnonzero(mask).tolist())
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            for q in neighbors(stack.pop()):
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/Users/tanmoy/research/data/derm7pt"))
    ap.add_argument("--cache-root", type=Path, default=Path(".cache/vit_features/derm7pt/lastvit_190k"))
    ap.add_argument("--head", type=Path, default=Path(".cache/rationales/derm7pt_rationale_head_tuned.pt"))
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--hidden-dim", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--min-hard-mass", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/derm7pt_random_region_baseline.json"))
    args = ap.parse_args()

    _, patches, labels, item_ids = load_split(
        args.data_root, args.cache_root, "lastvit_190k", "test", "mean",
        include_patches=True,
    )
    X = torch.as_tensor(np.stack(patches), dtype=torch.float32)
    checkpoint = torch.load(args.head, map_location="cpu", weights_only=False)
    relevance_weight = checkpoint["state_dict"]["relevance.weight"].numpy()
    relevance_bias = checkpoint["state_dict"]["relevance.bias"].numpy()
    head = RationaleHead(
        X.shape[-1], relevance_weight, relevance_bias,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        min_hard_mass=args.min_hard_mass,
    )
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.eval()

    with torch.no_grad():
        learned = head._gate(X, hard=True)
        relevance = head.relevance(X)
        full_score = torch.sigmoid(relevance.mean(dim=1))
        learned_score = torch.sigmoid(head.concept_from_region(X, learned))

    sizes = learned.sum(dim=1).to(torch.int64).numpy()
    learned_np = learned.numpy().astype(bool)
    learned_components = np.asarray([component_count(mask) for mask in learned_np])
    assert np.all(learned_components == 1), learned_components
    assert np.all(sizes >= head.min_hard_patches), sizes

    rng = np.random.default_rng(args.seed)
    random_masks = np.empty((args.K, len(X), GRID * GRID), dtype=np.float32)
    for k in range(args.K):
        for i, size in enumerate(sizes):
            random_masks[k, i] = random_connected_region(int(size), rng)
            assert int(random_masks[k, i].sum()) == int(size)
            assert component_count(random_masks[k, i]) == 1

    random_masks_t = torch.from_numpy(random_masks)
    with torch.no_grad():
        numerator = torch.einsum("knp,npc->knc", random_masks_t, relevance)
        denominator = random_masks_t.sum(dim=2).unsqueeze(-1)
        random_score = torch.sigmoid(numerator / denominator)

    learned_suff = full_score - learned_score
    random_suff = full_score.unsqueeze(0) - random_score
    learned_abs = learned_suff.abs()
    random_abs = random_suff.abs()
    result = {
        "contract": (
            "K connected random regions per image; exact learned-mask size; "
            "same frozen concept readout"
        ),
        "K": args.K,
        "seed": args.seed,
        "n_test": len(item_ids),
        "learned_mask": {
            "size_mean": float(sizes.mean()),
            "size_median": float(np.median(sizes)),
            "size_min": int(sizes.min()),
            "size_max": int(sizes.max()),
            "empty": int((sizes == 0).sum()),
            "non_single_component": int((learned_components != 1).sum()),
        },
        "concepts": {},
    }
    for c, name in enumerate(CONCEPTS):
        learned_mean = float(learned_suff[:, c].mean())
        random_mean = float(random_suff[:, :, c].mean())
        learned_abs_mean = float(learned_abs[:, c].mean())
        random_abs_mean = float(random_abs[:, :, c].mean())
        result["concepts"][name] = {
            "sufficiency_learned": learned_mean,
            "sufficiency_random_mean": random_mean,
            "random_minus_learned": random_mean - learned_mean,
            "absolute_sufficiency_learned": learned_abs_mean,
            "absolute_sufficiency_random_mean": random_abs_mean,
            "absolute_random_minus_learned": random_abs_mean - learned_abs_mean,
        }
    result["means"] = {
        key: float(np.mean([metrics[key] for metrics in result["concepts"].values()]))
        for key in (
            "sufficiency_learned",
            "sufficiency_random_mean",
            "random_minus_learned",
            "absolute_sufficiency_learned",
            "absolute_sufficiency_random_mean",
            "absolute_random_minus_learned",
        )
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
