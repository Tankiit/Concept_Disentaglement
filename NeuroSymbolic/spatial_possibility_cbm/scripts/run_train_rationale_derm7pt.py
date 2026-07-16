"""Train and audit the shared Derm7pt rationale gate on cached LAST-ViT tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_probe_scorer_derm7pt import CONCEPTS, load_split
from train_rationale import RationaleHead, evaluate, train


def connected_components(mask: np.ndarray, grid: int = 14) -> int:
    active = set(np.flatnonzero(mask).tolist())
    count = 0
    while active:
        count += 1
        stack = [active.pop()]
        while stack:
            p = stack.pop()
            r, c = divmod(p, grid)
            for q in ((r - 1) * grid + c, (r + 1) * grid + c, r * grid + c - 1, r * grid + c + 1):
                qr, qc = divmod(q, grid)
                if 0 <= qr < grid and 0 <= qc < grid and abs(qr - r) + abs(qc - c) == 1 and q in active:
                    active.remove(q)
                    stack.append(q)
    return count


def mask_audit(head, patches, batch_size=32):
    masks = []
    head.eval()
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            x = torch.as_tensor(patches[i:i + batch_size], dtype=torch.float32)
            masks.append(head._gate(x, hard=True).numpy().astype(bool))
    masks = np.concatenate(masks)
    masses = masks.mean(axis=1)
    components = np.asarray([connected_components(m) for m in masks])
    return {
        "selected_fraction_mean": float(masses.mean()),
        "selected_fraction_median": float(np.median(masses)),
        "empty_masks": int((masses == 0).sum()),
        "full_masks": int((masses == 1).sum()),
        "components_mean": float(components.mean()),
        "multi_component_masks": int((components > 1).sum()),
    }


def summarize(metrics):
    return {
        key: float(np.mean([value[key] for value in metrics.values()]))
        for key in ("gated_rho", "comprehensiveness", "sufficiency")
    }


def validation_score(metrics, masks):
    means = summarize(metrics)
    fragmentation = masks["multi_component_masks"] / max(masks["n_images"], 1)
    # Selection uses validation only. Reward prediction and faithfulness while
    # mildly penalizing fragmented explanations.
    return (
        means["gated_rho"]
        + means["comprehensiveness"]
        - abs(means["sufficiency"])
        - 0.1 * fragmentation
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/Users/tanmoy/research/data/derm7pt"))
    ap.add_argument("--cache-root", type=Path, default=Path(".cache/vit_features/derm7pt/lastvit_190k"))
    ap.add_argument("--probe", type=Path, default=Path(".cache/probes/derm7pt_lastvit_mean.npz"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scheduler-start", type=int, default=25)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--hidden-dims", default="16,32,64")
    ap.add_argument("--dropouts", default="0.2,0.5")
    ap.add_argument("--weight-decays", default="1e-4,1e-3")
    ap.add_argument("--tune-epochs", type=int, default=75)
    ap.add_argument("--min-hard-mass", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/derm7pt_rationale_audit.json"))
    ap.add_argument("--head-out", type=Path, default=Path(".cache/rationales/derm7pt_rationale_head.pt"))
    args = ap.parse_args()

    _, train_patches, train_y, train_ids = load_split(
        args.data_root, args.cache_root, "lastvit_190k", "train", "mean", include_patches=True
    )
    _, test_patches, test_y, test_ids = load_split(
        args.data_root, args.cache_root, "lastvit_190k", "test", "mean", include_patches=True
    )
    _, valid_patches, valid_y, valid_ids = load_split(
        args.data_root, args.cache_root, "lastvit_190k", "valid", "mean",
        include_patches=True,
    )
    assert not (set(train_ids) & set(valid_ids)), "train/valid item overlap"
    assert not (set(train_ids) & set(test_ids)), "train/test item overlap"
    assert not (set(valid_ids) & set(test_ids)), "valid/test item overlap"
    train_patches = np.stack(train_patches)
    valid_patches = np.stack(valid_patches)
    test_patches = np.stack(test_patches)
    with np.load(args.probe) as probe:
        W, b = probe["W"], probe["b"]
        assert W.shape == (len(CONCEPTS), train_patches.shape[-1])

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    trials = []
    configs = [
        (hidden_dim, dropout, weight_decay)
        for hidden_dim in map(int, args.hidden_dims.split(","))
        for dropout in map(float, args.dropouts.split(","))
        for weight_decay in map(float, args.weight_decays.split(","))
    ]
    for trial_index, (hidden_dim, dropout, weight_decay) in enumerate(configs):
        torch.manual_seed(0)
        np.random.seed(0)
        candidate = RationaleHead(
            train_patches.shape[-1], W, b, hidden_dim=hidden_dim,
            dropout=dropout, min_hard_mass=args.min_hard_mass,
        )
        candidate = train(
            candidate, train_patches, train_y, epochs=args.tune_epochs, lr=args.lr,
            batch_size=args.batch_size, device=device,
            scheduler_start=args.scheduler_start, min_lr=args.min_lr,
            weight_decay=weight_decay,
        )
        valid_metrics = evaluate(candidate, valid_patches, valid_y)
        valid_masks = mask_audit(candidate, valid_patches)
        valid_masks["n_images"] = len(valid_ids)
        trial = {
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "valid_means": summarize(valid_metrics),
            "valid_mask_audit": valid_masks,
        }
        trial["selection_score"] = validation_score(valid_metrics, valid_masks)
        trials.append(trial)
        print(f"trial {trial_index + 1}/{len(configs)} -> {trial}")

    best = max(trials, key=lambda trial: trial["selection_score"])
    torch.manual_seed(0)
    np.random.seed(0)
    head = RationaleHead(
        train_patches.shape[-1], W, b, hidden_dim=best["hidden_dim"],
        dropout=best["dropout"], min_hard_mass=args.min_hard_mass,
    )
    final_patches = np.concatenate([train_patches, valid_patches], axis=0)
    final_y = np.concatenate([train_y, valid_y], axis=0)
    head = train(
        head, final_patches, final_y, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device=device,
        scheduler_start=args.scheduler_start, min_lr=args.min_lr,
        weight_decay=best["weight_decay"],
    )

    train_metrics = evaluate(head, train_patches, train_y)
    valid_metrics = evaluate(head, valid_patches, valid_y)
    test_metrics = evaluate(head, test_patches, test_y)
    report = {
        "contract": (
            "hyperparameters selected on valid only; final gate refit on "
            "train+valid; test used once for final audit"
        ),
        "epochs": args.epochs,
        "tune_epochs": args.tune_epochs,
        "best_hyperparameters": {
            key: best[key] for key in ("hidden_dim", "dropout", "weight_decay")
        },
        "validation_trials": trials,
        "learning_rate": {
            "initial": args.lr,
            "schedule": "constant_then_cosine",
            "scheduler_start": args.scheduler_start,
            "minimum": args.min_lr,
        },
        "n_train": len(train_ids),
        "n_valid": len(valid_ids),
        "n_test": len(test_ids),
        "train": {CONCEPTS[i]: v for i, v in train_metrics.items()},
        "valid_after_refit": {CONCEPTS[i]: v for i, v in valid_metrics.items()},
        "test": {CONCEPTS[i]: v for i, v in test_metrics.items()},
        "train_mask_audit": mask_audit(head, train_patches),
        "valid_mask_audit_after_refit": mask_audit(head, valid_patches),
        "test_mask_audit": mask_audit(head, test_patches),
    }
    report["test_means"] = summarize(test_metrics)
    args.head_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "concept_names": CONCEPTS}, args.head_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"head -> {args.head_out}")
    print(f"audit -> {args.out}")


if __name__ == "__main__":
    main()
