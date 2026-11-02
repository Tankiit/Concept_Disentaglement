import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# Keep this file beside main.py in REACH_AISTATS, or import it from that folder.
from data.worlds import World, make_world

# Edit these options and call run(**OPTIONS) again. No model training required.
OPTIONS = dict(kind="interpolate", novelty=0.25, anchor_mix=0.5, delta=0.1,
               score_mode="known_surprisal", n_samples=2800, batch_size=256,
               seed=7, calibration_seed=13, calibration_size=512,
               target_fpr=0.05)


def make_novel_world(kind="interpolate", novelty=0.5, anchor_mix=0.5,
                     delta=0.1, custom_q=None):
    """Vary the NOVEL SOURCE LAW, keeping the two known laws and targets fixed.

    interpolate: q = (1-novelty)*q_hull + novelty*q_separated.
    q_hull = anchor_mix*a0 + (1-anchor_mix)*a1.
    novelty=0 is confounded; novelty=1 is separated. Intermediate values
    control departure from the hull, not the proportion of novel rows (delta).
    near_known0/1 anchor the same interpolation at that known source.
    custom_q is a probability vector over the existing four cells.
    These are distributions over cells, not image blends or new semantic classes.
    """
    if not np.isfinite(novelty) or not 0 <= novelty <= 1:
        raise ValueError("novelty must be in [0,1]")
    if not np.isfinite(anchor_mix) or not 0 <= anchor_mix <= 1:
        raise ValueError("anchor_mix must be in [0,1]")
    if not np.isfinite(delta) or not 0 <= delta < 1:
        raise ValueError("delta must be in [0,1)")
    if custom_q is not None and kind != "custom":
        raise ValueError("custom_q is used only with kind='custom'")
    base = make_world("separated", delta=delta)
    a0, a1, q_out = base.conditionals
    if kind == "separated":
        q = q_out
    elif kind == "confounded":
        q = anchor_mix*a0 + (1-anchor_mix)*a1
    elif kind in ("interpolate", "near_known0", "near_known1"):
        mix = {"near_known0": 1., "near_known1": 0.}.get(kind, anchor_mix)
        q = (1-novelty)*(mix*a0 + (1-mix)*a1) + novelty*q_out
    elif kind == "custom":
        q = np.asarray(custom_q, dtype=float)
        if (q.shape != q_out.shape or not np.isfinite(q).all()
                or np.any(q < 0) or not np.isclose(q.sum(), 1)):
            raise ValueError("custom_q must be a finite four-cell probability vector")
    elif kind == "absent":
        return make_world("absent")
    else:
        raise ValueError("kind must be separated, confounded, interpolate, "
                         "near_known0, near_known1, custom, or absent")
    regions = tuple('novel_region_extra' if label == 'off_support' and q[i] > 0 else label
                    for i, label in enumerate(base.regions))
    return World(f"{kind}:novelty={novelty:g}", np.vstack([a0, a1, q]),
                 base.weights.copy(), base.theta.copy(), regions)


def categorize_detection(
    logits,
    ood_scores,
    known_val_scores,
    is_ood,
    class_ids,
    target_fpr=0.05,
):
    """
    Larger ood_scores = more OOD-like.

    logits:           [N, C] classifier logits
    ood_scores:       [N] novelty scores
    known_val_scores: [M] scores on known validation data
    is_ood:           [N] ground-truth OOD indicators
    class_ids:        [N] original class IDs
    """
    logits = np.asarray(logits)
    scores = np.asarray(ood_scores).reshape(-1)
    val = np.asarray(known_val_scores).reshape(-1)
    labels = np.asarray(is_ood).astype(bool)
    classes = np.asarray(class_ids)

    assert logits.ndim == 2 and logits.shape[1] >= 2
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0,1)")
    assert len(logits) == len(scores) == len(labels) == len(classes)
    assert len(val) > 0
    assert np.isfinite(logits).all()
    assert np.isfinite(scores).all()
    assert np.isfinite(val).all()

    # Calibrate the OOD rejection boundary using known validation data.
    threshold = np.quantile(val, 1 - target_fpr, method="higher")

    # Standardize distance using known validation score dispersion.
    scale = max(
        np.quantile(val, 0.75) - np.quantile(val, 0.25),
        1e-8,
    )

    detector_depth = (scores - threshold) / scale

    # Classifier logit geometry.
    sorted_logits = np.sort(logits, axis=1)
    logit_margin = sorted_logits[:, -1] - sorted_logits[:, -2]
    predicted_class = logits.argmax(axis=1)

    # Descriptive bands, not statistical guarantees.
    category = np.select(
        [
            detector_depth < -1,
            (detector_depth >= -1) & (detector_depth <= 0),
            (detector_depth > 0) & (detector_depth < 1),
            detector_depth >= 1,
        ],
        [
            "strongly_accepted",
            "weakly_accepted",
            "weakly_rejected",
            "strongly_rejected",
        ],
        default="unclassified",
    )

    df = pd.DataFrame(
        {
            "class_id": classes,
            "is_ood": labels,
            "predicted_class": predicted_class,
            "logit_margin": logit_margin,
            "ood_score": scores,
            "detector_depth": detector_depth,
            "category": category,
            "detected": scores > threshold,
        }
    )

    df["outcome"] = np.select(
        [
            labels & df["detected"],
            labels & ~df["detected"],
            ~labels & df["detected"],
            ~labels & ~df["detected"],
        ],
        [
            "correct_rejection",
            "missed_ood",
            "false_alarm",
            "correct_acceptance",
        ],
        default="unknown",
    )

    return df, threshold


class _DictDataset(Dataset):
    def __init__(self, arrays):
        self.arrays = arrays
        self.n = len(arrays["cell_id"])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.arrays.items()}


def _sample_world_rows(world, n_samples=2400, seed=7, known_only=False,
                       score_mode="known_surprisal", logit_noise=0.12, score_noise=0.08):
    """Preserve source IDs and common random draws across novelty variations.

    known_surprisal: -log p_known(cell), fixed as q changes (default).
    oracle_ratio: log q(cell)-log p_known(cell), uses the TRUE novel law;
                  this changes the detector with q and is an oracle diagnostic.
    Neither score is an ensemble-based epistemic uncertainty estimate.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if any(not np.isfinite(v) or v < 0 for v in (logit_noise, score_noise)):
        raise ValueError("Noise scales must be finite and nonnegative")
    if score_mode not in ("known_surprisal", "oracle_ratio"):
        raise ValueError("Unknown score_mode")
    rngs = [np.random.default_rng(np.random.SeedSequence([int(seed), 505, i]))
            for i in range(4)]
    weights = world.weights.copy()
    known_prior = weights[:2]/weights[:2].sum()
    if known_only:
        weights = np.r_[known_prior, 0.]
    src = np.searchsorted(np.cumsum(weights), rngs[0].random(n_samples), side="right").clip(max=2)
    u = rngs[1].random(n_samples)
    cdf = np.cumsum(world.conditionals[src], axis=1)
    cell_id = (u[:, None] >= cdf).sum(1).clip(max=world.conditionals.shape[1]-1)
    eps = 1e-8
    logits_known = np.log(world.conditionals[:2, cell_id].T + eps)
    logits_known += rngs[2].normal(0, logit_noise, size=logits_known.shape)
    p_known = known_prior @ world.conditionals[:2]
    ood_score = -np.log(p_known[cell_id] + eps)
    if score_mode == "oracle_ratio":
        ood_score += np.log(world.conditionals[2, cell_id] + eps)
    ood_score += rngs[3].normal(0, score_noise, size=n_samples)
    return {
        "logits": torch.as_tensor(logits_known, dtype=torch.float32),
        "ood_score": torch.as_tensor(ood_score, dtype=torch.float32),
        "is_ood": torch.as_tensor(src == 2, dtype=torch.int64),
        "class_id": torch.as_tensor(src, dtype=torch.int64),  # source, NOT cell
        "source_id": torch.as_tensor(src, dtype=torch.int64),
        "cell_id": torch.as_tensor(cell_id, dtype=torch.int64),
        "row_id": torch.arange(n_samples),
    }


def _make_world_dataloader(world, n_samples=2400, batch_size=256, seed=7,
                           shuffle=False, score_mode="known_surprisal",
                           logit_noise=0.12, score_noise=0.08):
    arrays = _sample_world_rows(world, n_samples, seed, score_mode=score_mode,
                                logit_noise=logit_noise, score_noise=score_noise)
    return DataLoader(_DictDataset(arrays), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def _collect_from_loader(loader):
    logits, scores, labels, classes = [], [], [], []
    for batch in loader:
        logits.append(batch["logits"].numpy())
        scores.append(batch["ood_score"].numpy())
        labels.append(batch["is_ood"].numpy().astype(bool))
        classes.append(batch["class_id"].numpy())

    return (
        np.vstack(logits),
        np.concatenate(scores),
        np.concatenate(labels),
        np.concatenate(classes),
    )


def _build_known_val_scores(world, m=600, seed=11, score_mode="known_surprisal",
                            score_noise=0.08):
    """m independent known examples scored by the SAME rule/noise as evaluation.

    Sampling weights follow the conditional known-source deployment proportions.
    This returns m per-example scores, not one value per possible cell.
    """
    return _sample_world_rows(world, m, seed, known_only=True,
                              score_mode=score_mode, score_noise=score_noise)["ood_score"].numpy()


def _display(obj, n=10):
    """Notebook-like display fallback for scripts."""
    try:
        from IPython.display import display as ipy_display

        ipy_display(obj)
    except Exception:
        if isinstance(obj, pd.DataFrame):
            print(obj.head(n).to_string())
        else:
            print(obj)


def run(kind="interpolate", novelty=0.25, anchor_mix=0.5, delta=0.1,
        custom_q=None, score_mode="known_surprisal", n_samples=2800,
        batch_size=256, seed=7, calibration_seed=13, calibration_size=512,
        target_fpr=0.05, logit_noise=0.12, score_noise=0.08, show=True):
    if seed == calibration_seed:
        raise ValueError("Use distinct evaluation and calibration seeds")
    world = make_novel_world(kind, novelty, anchor_mix, delta, custom_q)
    loader = _make_world_dataloader(world, n_samples, batch_size, seed,
                                    score_mode=score_mode, logit_noise=logit_noise,
                                    score_noise=score_noise)
    all_logits, all_scores, all_is_ood, all_class_ids = _collect_from_loader(loader)
    known_val_scores = _build_known_val_scores(world, calibration_size, calibration_seed,
                                              score_mode, score_noise)

    # These arrays are from the SAME split and aligned by row order.
    df, threshold = categorize_detection(
        logits=all_logits,
        ood_scores=all_scores,
        known_val_scores=known_val_scores,
        is_ood=all_is_ood,
        class_ids=all_class_ids,
        target_fpr=target_fpr,
    )

    df["cell_id"] = loader.dataset.arrays["cell_id"].numpy()  # shuffle=False above
    print(f"Calibrated threshold (target FPR={target_fpr:.1%}): {threshold:.4f}")
    print(f"Score: {score_mode}; q={world.conditionals[2]}; novel prevalence={world.weights[2]:g}")
    print(f"World: {world.name}; regions={list(world.regions)}")
    _display(df.head())

    # One novel source exists; stratify its detections by INPUT CELL, not class.
    per_class = (
        df[df.is_ood]
        .groupby("cell_id")
        .agg(
            n=("detected", "size"),
            detection_rate=("detected", "mean"),
            mean_depth=("detector_depth", "mean"),
            median_logit_margin=("logit_margin", "median"),
        )
        .sort_values("detection_rate")
    )

    print("\nNovel-source detection by input cell:")
    _display(per_class)

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, group in df.groupby("outcome"):
        ax.scatter(
            group["logit_margin"],
            group["detector_depth"],
            label=label,
            alpha=0.5,
            s=14,
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Classifier logit margin")
    ax.set_ylabel("Normalized detector depth")
    ax.set_title("REACH: Input-level novelty detection")
    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return dict(world=world, loader=loader, rows=df, threshold=threshold,
                per_cell=per_class, figure=fig, known_val_scores=known_val_scores)


if __name__ == "__main__":
    run(**OPTIONS)
