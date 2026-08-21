"""Steps 0-1: CUB part masks and masked DINOv2 extraction.

Mask the pixels, then run the backbone. Selecting tokens after a ViT forward
pass does not restrict information because self-attention has already mixed it.
See ``MASKING.md`` at the repository root for the experimental rationale.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
import json

import numpy as np


_CUB_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "experiments" / "configs" / "datasets" / "cub.json"
)
with _CUB_CONFIG.open(encoding="utf-8") as _handle:
    _CUB_METADATA = json.load(_handle)

# Backward-compatible aliases; their values come from dataset metadata, not code.
PART_NAMES = {int(key): value for key, value in _CUB_METADATA["part_names"].items()}
PART_TO_GROUP = dict(_CUB_METADATA["part_to_group"])
CONCEPT_PREFIX_TO_GROUP = dict(_CUB_METADATA["concept_prefix_to_group"])
GLOBAL = str(_CUB_METADATA["global_group"])


def _normalise_part_names(part_names=None):
    """Normalise JSON string keys while retaining the legacy defaults."""
    source = PART_NAMES if part_names is None else part_names
    return {int(part_id): name for part_id, name in source.items()}


def load_part_locs(cub_root: Path, part_names=None):
    """Return ``(locations, sizes, image_paths)`` keyed by integer image ID.

    Locations are ``{image_id: {part_name: (x, y, visible)}}`` in original
    image coordinates. Sizes are original ``(width, height)`` tuples.
    """
    from PIL import Image

    cub_root = Path(cub_root)
    part_names = _normalise_part_names(part_names)
    image_paths: dict[int, Path] = {}
    with (cub_root / "images.txt").open(encoding="utf-8") as handle:
        for line in handle:
            image_id, relative_path = line.rstrip().split(maxsplit=1)
            image_paths[int(image_id)] = cub_root / "images" / relative_path

    locs: dict[int, dict[str, tuple[float, float, bool]]] = {
        image_id: {} for image_id in image_paths
    }
    with (cub_root / "parts" / "part_locs.txt").open(encoding="utf-8") as handle:
        for line in handle:
            image_id, part_id, x, y, visible = line.split()
            locs[int(image_id)][part_names[int(part_id)]] = (
                float(x), float(y), bool(int(visible))
            )

    sizes = {}
    for image_id, path in image_paths.items():
        with Image.open(path) as image:
            sizes[image_id] = image.size
    return locs, sizes, image_paths


def part_masks(locs, sizes, part_names=None, image_size=224, radius=32):
    """Create per-image ``(15, S, S)`` boolean landmark-disc masks.

    Coordinates are rescaled independently in x/y for a plain square resize;
    invisible or missing landmarks produce an empty mask.
    """
    if image_size <= 0 or radius < 0:
        raise ValueError("image_size must be positive and radius non-negative")
    part_names = _normalise_part_names(part_names)
    yy, xx = np.ogrid[:image_size, :image_size]
    result = {}
    for image_id, image_locs in locs.items():
        width, height = sizes[image_id]
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid original size for image {image_id}: {(width, height)}")
        masks = np.zeros((len(part_names), image_size, image_size), dtype=bool)
        for part_id, part_name in part_names.items():
            landmark = image_locs.get(part_name)
            if landmark is None or not landmark[2]:
                continue
            x = landmark[0] * image_size / width
            y = landmark[1] * image_size / height
            masks[part_id - 1] = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        result[image_id] = masks
    return result


def group_masks(
    part_mask, part_names=None, part_to_group=None, global_group=GLOBAL,
    image_size=224,
):
    """Union left/right parts into groups and add an all-ones GLOBAL mask."""
    part_names = _normalise_part_names(part_names)
    part_to_group = PART_TO_GROUP if part_to_group is None else part_to_group
    part_mask = np.asarray(part_mask, dtype=bool)
    expected = (len(part_names), image_size, image_size)
    if part_mask.shape != expected:
        raise ValueError(f"expected part mask shape {expected}, got {part_mask.shape}")
    grouped = {
        group: np.zeros((image_size, image_size), dtype=bool)
        for group in sorted(set(part_to_group.values()))
    }
    for part_id, part_name in part_names.items():
        grouped[part_to_group[part_name]] |= part_mask[part_id - 1]
    grouped[global_group] = np.ones((image_size, image_size), dtype=bool)
    return grouped


def visibility_report(locs, part_names=None) -> dict:
    """Return per-part counts and drop rates for absent/invisible landmarks."""
    part_names = _normalise_part_names(part_names)
    n_images = len(locs)
    report = {}
    for part_name in part_names.values():
        visible = sum(
            int(part_name in image_locs and image_locs[part_name][2])
            for image_locs in locs.values()
        )
        report[part_name] = {
            "visible": visible,
            "total": n_images,
            "drop_rate": 1.0 - visible / n_images if n_images else float("nan"),
        }
    return report


class LandmarkGroupMaskSequence(Sequence):
    """Lazily construct one group's mask for each selected image.

    This avoids materialising ``N x groups x S x S`` masks for full-dataset
    extraction. Only the masks needed by the current backbone batch exist.
    """

    def __init__(
        self, image_ids, locs, sizes, group, part_names, part_to_group,
        global_group="GLOBAL", image_size=224, radius=32,
    ):
        self.image_ids = list(image_ids)
        self.locs = locs
        self.sizes = sizes
        self.group = group
        self.part_names = _normalise_part_names(part_names)
        self.part_to_group = part_to_group
        self.global_group = global_group
        self.image_size = image_size
        self.radius = radius

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if self.group == self.global_group:
            return np.ones((self.image_size, self.image_size), dtype=bool)
        image_id = self.image_ids[index]
        width, height = self.sizes[image_id]
        yy, xx = np.ogrid[:self.image_size, :self.image_size]
        mask = np.zeros((self.image_size, self.image_size), dtype=bool)
        for part_name, landmark in self.locs[image_id].items():
            if self.part_to_group.get(part_name) != self.group or not landmark[2]:
                continue
            x = landmark[0] * self.image_size / width
            y = landmark[1] * self.image_size / height
            mask |= (xx - x) ** 2 + (yy - y) ** 2 <= self.radius ** 2
        return mask


def _mask_for(masks, group, index):
    value = masks[group]
    if isinstance(value, np.ndarray) and value.ndim == 2:
        return value
    if isinstance(value, Mapping):
        return value[index]
    return value[index]


def extract_masked(
    image_paths, masks, model_name="facebook/dinov2-base", fill="zero",
    batch_size=64, device="mps", out: Path | None = None, image_size=224,
):
    """Run one DINOv2 pass per group, always masking pixels first.

    ``masks`` maps group names to either an ``(N,S,S)`` array/sequence or a
    mapping keyed by image index. Features are returned and optionally saved as
    a compressed NPZ. MPS is the default device.
    """
    import torch

    if fill not in {"zero", "blur", "mean"}:
        raise ValueError("fill must be one of: zero, blur, mean")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; pass device='cpu' explicitly")

    from PIL import Image, ImageFilter
    from tqdm.auto import tqdm
    from transformers import AutoImageProcessor, AutoModel

    image_paths = [Path(path) for path in image_paths]
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    results = {}

    for group in tqdm(sorted(masks), desc="DINOv2 groups", unit="group"):
        group_features = []
        batches = range(0, len(image_paths), batch_size)
        for start in tqdm(batches, desc=group, unit="batch", leave=False):
            prepared = []
            stop = min(start + batch_size, len(image_paths))
            for index in range(start, stop):
                with Image.open(image_paths[index]) as source:
                    image = source.convert("RGB").resize(
                        (image_size, image_size), Image.Resampling.BICUBIC
                    )
                pixels = np.asarray(image).copy()
                mask = np.asarray(_mask_for(masks, group, index), dtype=bool)
                if mask.shape != (image_size, image_size):
                    raise ValueError(
                        f"{group}[{index}] has mask shape {mask.shape}, "
                        f"expected {(image_size, image_size)}"
                    )
                if fill == "zero":
                    replacement = 0
                elif fill == "mean":
                    replacement = pixels.reshape(-1, 3).mean(axis=0)
                else:
                    replacement = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=16)))
                pixels[~mask] = replacement if np.isscalar(replacement) or replacement.ndim == 1 else replacement[~mask]
                prepared.append(Image.fromarray(pixels.astype(np.uint8)))

            inputs = processor(
                images=prepared, return_tensors="pt", do_resize=False,
                do_center_crop=False,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                cls = model(**inputs).last_hidden_state[:, 0].float().cpu().numpy()
            group_features.append(cls)
        hidden_size = int(model.config.hidden_size)
        results[group] = (
            np.concatenate(group_features).astype(np.float32, copy=False)
            if group_features else np.empty((0, hidden_size), dtype=np.float32)
        )

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **results)
    return results


def containment_poset(group_masks: dict) -> tuple[np.ndarray, list]:
    """Return ``contains[j, i] = (M_i subseteq M_j)``."""
    names = sorted(group_masks)
    arrays = {name: np.asarray(group_masks[name], dtype=bool) for name in names}
    contains = np.zeros((len(names), len(names)), dtype=bool)
    for j, outer in enumerate(names):
        for i, inner in enumerate(names):
            contains[j, i] = not np.any(arrays[inner] & ~arrays[outer])
    return contains, names


def predicted_orbit_dim(poset, names, concept_to_group, embedding_size=8):
    """Count allowed concept-pair terms implied by mask containment."""
    del embedding_size  # Decoder-specific scaling is deliberately not assumed.
    idx = {name: i for i, name in enumerate(names)}
    concepts = sorted(concept_to_group)
    within = glob = 0
    for concept_j in concepts:
        group_j = concept_to_group[concept_j]
        for concept_i in concepts:
            group_i = concept_to_group[concept_i]
            if not poset[idx[group_j], idx[group_i]]:
                continue
            if group_j == GLOBAL:
                glob += 1
            else:
                within += 1
    total = within + glob
    return {
        "within_group": within, "global_rows": glob, "total": total,
        "unmasked_baseline": len(concepts) ** 2,
        "reduction": len(concepts) ** 2 / max(total, 1),
    }
