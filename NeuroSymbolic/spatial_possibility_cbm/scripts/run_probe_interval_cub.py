"""Real CUB probe intervals for a non-trivial head AND wing AND tail query."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from interval_composition import compose_and, interval_from_ensemble
from run_probe_interval_derm7pt import concept_samples
from run_probe_scorer_derm7pt import cache_key
from spatial_possibility_cbm.probe_scorer import ProbeConceptScorer


ATTRIBUTES = {
    102: "plain_head_pattern",
    213: "rounded_wings",
    76: "notched_tail",
}


def load_metadata(root: Path):
    images = {
        int(line.split(maxsplit=1)[0]): line.split(maxsplit=1)[1]
        for line in (root / "images.txt").read_text().splitlines()
    }
    split = {
        int(line.split()[0]): int(line.split()[1])
        for line in (root / "train_test_split.txt").read_text().splitlines()
    }
    boxes = {
        int(fields[0]): tuple(map(float, fields[1:]))
        for fields in (
            line.split() for line in (root / "bounding_boxes.txt").read_text().splitlines()
        )
    }
    labels = {image_id: {attribute_id: 0 for attribute_id in ATTRIBUTES} for image_id in images}
    with (root / "attributes" / "image_attribute_labels.txt").open() as f:
        for line in f:
            fields = line.split()
            image_id, attribute_id, present = map(int, fields[:3])
            if attribute_id in ATTRIBUTES:
                labels[image_id][attribute_id] = present
    return images, split, boxes, labels


def load_features(
    root, cache_root, model_tag, images, split_flags, split_name,
    include_patches=False,
):
    want_train = split_name == "train"
    ids, pooled, patches = [], [], []
    for image_id in sorted(images):
        if bool(split_flags[image_id]) != want_train:
            continue
        item_id = f"{image_id}::{images[image_id]}"
        path = cache_root / split_name / cache_key(model_tag, split_name, item_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as cached:
            pooled.append(cached["mean"])
            if include_patches:
                patch = cached["patches"]
                assert patch.shape == (196, 768)
                assert np.ptp(patch, axis=0).max() > 1e-6
                patches.append(patch)
        ids.append(image_id)
    return ids, np.stack(pooled), np.stack(patches) if patches else None


def load_patch_batch(cache_root, model_tag, images, image_ids):
    batch = []
    for image_id in image_ids:
        item_id = f"{image_id}::{images[image_id]}"
        path = cache_root / "test" / cache_key(model_tag, "test", item_id)
        with np.load(path) as cached:
            patch = cached["patches"]
        assert patch.shape == (196, 768)
        assert np.ptp(patch, axis=0).max() > 1e-6
        batch.append(patch)
    return np.stack(batch)


def transformed_box(box, image_size):
    x, y, width, height = box
    original_width, original_height = image_size
    scale = 256.0 / min(original_width, original_height)
    resized_width, resized_height = original_width * scale, original_height * scale
    crop_left, crop_top = (resized_width - 224.0) / 2, (resized_height - 224.0) / 2
    return (
        x * scale - crop_left,
        y * scale - crop_top,
        (x + width) * scale - crop_left,
        (y + height) * scale - crop_top,
    )


def patch_in_box(patch_index, box):
    row, col = divmod(int(patch_index), 14)
    center_x, center_y = (col + 0.5) * 16, (row + 0.5) * 16
    x1, y1, x2, y2 = box
    return x1 <= center_x <= x2 and y1 <= center_y <= y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-root", type=Path, default=Path("/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011"))
    ap.add_argument("--cache-root", type=Path, default=Path(".cache/vit_features/cub/lastvit_190k"))
    ap.add_argument("--model-tag", default="lastvit_190k")
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--temp", type=float, default=10.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/cub_probe_intervals.json"))
    ap.add_argument("--probe-out", type=Path, default=Path(".cache/probes/cub_lastvit_head_wing_tail.npz"))
    args = ap.parse_args()

    images, split, boxes, label_map = load_metadata(args.cub_root)
    train_ids, train_x, _ = load_features(
        args.cub_root, args.cache_root, args.model_tag, images, split, "train"
    )
    test_ids, _, _ = load_features(
        args.cub_root, args.cache_root, args.model_tag, images, split, "test"
    )
    attribute_ids = list(ATTRIBUTES)
    concept_names = list(ATTRIBUTES.values())
    train_y = np.asarray(
        [[label_map[i][a] for a in attribute_ids] for i in train_ids], dtype=np.int64
    )
    test_y = np.asarray(
        [[label_map[i][a] for a in attribute_ids] for i in test_ids], dtype=np.int64
    )
    scorer = ProbeConceptScorer(concept_names).fit(train_x, train_y)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    concept_intervals = np.empty((len(test_ids), 3, 2), dtype=np.float32)
    composed = []
    box_hits = {name: [] for name in concept_names}
    chance = []
    from PIL import Image
    for start in range(0, len(test_ids), 64):
        batch_ids = test_ids[start : start + 64]
        x = torch.from_numpy(
            load_patch_batch(args.cache_root, args.model_tag, images, batch_ids)
        ).to(device)
        normalized = x / (torch.linalg.vector_norm(x, dim=2, keepdim=True) + 1e-8)
        samples_by_concept, raw_cosines = [], []
        for c, name in enumerate(concept_names):
            samples_by_concept.append(
                concept_samples(
                    normalized, scorer.W[c], name, args.M, args.noise, args.temp
                )
            )
            direction = torch.from_numpy(scorer.W[c]).to(device)
            raw_cosines.append(
                torch.einsum("npd,d->np", normalized, direction).cpu().numpy()
            )
        samples = np.stack(samples_by_concept, axis=1)
        raw_cosines = np.stack(raw_cosines, axis=1)
        for local_i, image_id in enumerate(batch_ids):
            global_i = start + local_i
            intervals = []
            for c in range(3):
                interval = interval_from_ensemble(samples[local_i, c])
                intervals.append(interval)
                concept_intervals[global_i, c] = interval
            composed.append(compose_and(intervals))
            with Image.open(args.cub_root / "images" / images[image_id]) as image:
                box = transformed_box(boxes[image_id], image.size)
            x1, y1, x2, y2 = box
            chance.append(
                max(0.0, min(224.0, x2) - max(0.0, x1))
                * max(0.0, min(224.0, y2) - max(0.0, y1))
                / (224.0 * 224.0)
            )
            for c, name in enumerate(concept_names):
                box_hits[name].append(
                    patch_in_box(raw_cosines[local_i, c].argmax(), box)
                )
        if start % 640 == 0:
            print(f"test intervals: {min(start + len(batch_ids), len(test_ids))}/{len(test_ids)}")

    point = np.asarray([value["point"] for value in composed])
    width = np.asarray([value["width"] for value in composed])
    conjunction = test_y.all(axis=1).astype(np.int64)

    result = {
        "contract": (
            "train pooled CUB features -> three attribute probes -> real test patches "
            "-> perturbed max-patch intervals -> outer AND"
        ),
        "attributes": {
            name: attribute_id for attribute_id, name in ATTRIBUTES.items()
        },
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "M": args.M,
        "noise": args.noise,
        "temperature": args.temp,
        "positive_conjunction_test": int(conjunction.sum()),
        "metrics": {
            "and_point_auc": float(roc_auc_score(conjunction, point)),
            "and_point_mean": float(point.mean()),
            "and_point_std": float(point.std()),
            "mean_interval_width": float(width.mean()),
            "mean_box_area_fraction": float(np.mean(chance)),
        },
        "per_concept": {},
    }
    for c, name in enumerate(concept_names):
        result["per_concept"][name] = {
            "train_positive": int(train_y[:, c].sum()),
            "test_positive": int(test_y[:, c].sum()),
            "point_auc": float(
                roc_auc_score(test_y[:, c], concept_intervals[:, c].mean(axis=1))
            ),
            "mean_interval_width": float(
                (concept_intervals[:, c, 1] - concept_intervals[:, c, 0]).mean()
            ),
            "argmax_patch_box_hit_rate": float(np.mean(box_hits[name])),
        }
    args.probe_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.probe_out, W=scorer.W, b=scorer.b,
        concept_names=np.asarray(concept_names), attribute_ids=np.asarray(attribute_ids),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"probe -> {args.probe_out}")
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
