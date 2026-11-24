"""Verify the distributed checkpoint and cache, with an optional stratified image check."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tiny_imagenet_notebook_source"))
import tiny_imagenet_acquisition as original

MODEL = ROOT / "tiny_imagenet_model"
SEED = 20261111
COLAB_DATA = Path("/content/tiny_reach_cvpr/data/tiny-imagenet-200")


def check_images(model, image_root, subset, features, predictions):
    # Four images per source in each split. Preserve Colab paths for the noise seed.
    sample = subset.groupby("source", sort=False).head(4)
    assert len(sample) == 12
    transform = original._make_transform(False)
    images = []
    for row in sample.itertuples():
        with Image.open(image_root / row.relative_path) as image:
            corrupted = original.corrupt(image.convert("RGB"), row.source, SEED + 1009,
                                         str(COLAB_DATA / row.relative_path))
            images.append(transform(corrupted))
    with torch.autocast("cpu", dtype=torch.float16):
        recomputed = original.resnet_features(model, torch.stack(images))
        logits = model.fc(recomputed)
    positions = sample.position.to_numpy(copy=True)
    similarity = torch.nn.functional.cosine_similarity(recomputed.float(), features[positions])
    assert float(similarity.min()) > 0.999
    np.testing.assert_array_equal(logits.argmax(1).numpy(), predictions[positions])
    print(f"Image check: {len(positions)} matching top-1 predictions, minimum feature cosine {similarity.min():.8f}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, help="extracted Tiny ImageNet root for the optional image check")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "provenance/tiny_model_cache.json").read_text())
    for entry in manifest["artifacts"]:
        with (ROOT / entry["file"]).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
    ckpt = torch.load(MODEL / "resnet18_scratch_seed20261111_epochs40.pt", map_location="cpu", weights_only=True)
    assert ckpt["seed"] == SEED and ckpt["completed_epochs"] == ckpt["total_epochs"] == 40
    model = original.resnet18_64(200).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    records = ROOT / "TinyImageNet_REACH_Results_v2full"
    selection = pd.read_csv(records / "reconstructed_image_selection.csv.gz")
    classes = selection[["class_id", "wnid"]].drop_duplicates().sort_values("class_id")
    assert classes.wnid.tolist() == ckpt["classes"]
    with np.load(MODEL / "frozen_tiny_seed20261111_epochs40.npz", allow_pickle=False) as cache:
        assert int(cache["seed"]) == SEED and int(cache["holdout_per_class"]) == ckpt["holdout_per_class"] == 100
        np.testing.assert_array_equal(cache["sources_names"], original.SOURCES)
        np.testing.assert_allclose(cache["source_weights"], [0.45, 0.25, 0.30])
        for split, n in (("fit", 20000), ("eval", 10000)):
            rows = selection.loc[selection.role == split].sort_values("position")
            np.testing.assert_array_equal(rows.position, np.arange(n))
            np.testing.assert_array_equal(rows.source, cache[split + "_source"])
            np.testing.assert_array_equal(rows.class_id, cache[split + "_label"])
            features = torch.from_numpy(cache[split + "_features"])
            assert features.shape == (n, 512)
            predicted = cache[split + "_pred"]
            scores = cache[split + "_scores"]
            assert scores.shape == (n, 200)
            np.testing.assert_array_equal(scores.argmax(1), predicted)
            np.testing.assert_array_equal(predicted != rows.class_id, cache[split + "_error"])
            # The original CUDA autocast head used half-precision operands and logits.
            logits = torch.nn.functional.linear(features, model.fc.weight.half().float(), model.fc.bias.half().float())
            np.testing.assert_array_equal(logits.half().argmax(1).numpy(), predicted)
            print(f"{split}: checkpoint top-1 predictions and cache labels, errors and sources agree for all {n} images")
            if args.images:
                check_images(model, args.images, rows, features, predicted)


if __name__ == "__main__":
    main()
