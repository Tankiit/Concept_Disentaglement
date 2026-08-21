"""Build landmark masks and extract features from pixel-masked images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.cub_masking import (
    LandmarkGroupMaskSequence,
    extract_masked,
    load_part_locs,
    visibility_report,
)
from .configured import ConfiguredExperiment


class MaskedBackboneExtractionExperiment(ConfiguredExperiment):
    """Dataset-configured pixel masking and backbone feature extraction."""

    def run(self):
        dataset_root = Path(self.config["dataset"]["root"])
        masking = self.config.get("masking", {})
        extraction = self.config.get("extraction", {})
        image_size = int(masking.get("image_size", 224))
        radius = int(masking.get("radius", 32))

        locs, sizes, image_paths = load_part_locs(
            dataset_root, part_names=self.dataset["part_names"]
        )
        ordered_ids = sorted(image_paths)
        split = self.config.get("dataset", {}).get("split")
        if split is not None:
            split_file = dataset_root / "train_test_split.txt"
            split_flags = {}
            with split_file.open(encoding="utf-8") as handle:
                for line in handle:
                    image_id, is_train = line.split()
                    split_flags[int(image_id)] = bool(int(is_train))
            if split not in {"train", "test"}:
                raise ValueError("dataset.split must be 'train' or 'test'")
            want_train = split == "train"
            ordered_ids = [
                image_id for image_id in ordered_ids
                if split_flags[image_id] == want_train
            ]
        max_images = extraction.get("max_images")
        if max_images is not None:
            max_images = int(max_images)
            if max_images <= 0:
                raise ValueError("extraction.max_images must be positive")
            ordered_ids = ordered_ids[:max_images]
        selected_locs = {image_id: locs[image_id] for image_id in ordered_ids}
        groups = sorted(set(self.dataset["part_to_group"].values()))
        global_group = self.dataset["global_group"]
        masks = {
            group: LandmarkGroupMaskSequence(
                ordered_ids, locs, sizes, group, self.dataset["part_names"],
                self.dataset["part_to_group"], global_group, image_size, radius,
            )
            for group in groups + [global_group]
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        report = visibility_report(selected_locs, self.dataset["part_names"])
        report_path = self.output_dir / "visibility_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        feature_path = self.output_dir / extraction.get(
            "feature_file", "masked_backbone_features.npz"
        )
        features = extract_masked(
            [image_paths[image_id] for image_id in ordered_ids], masks,
            model_name=extraction.get("base_model", "facebook/dinov2-base"),
            fill=masking.get("fill", "zero"),
            batch_size=int(extraction.get("batch_size", 64)),
            device=self.device, out=feature_path, image_size=image_size,
        )
        return {
            "features": features,
            "feature_path": feature_path,
            "visibility_report": report,
            "visibility_path": report_path,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = MaskedBackboneExtractionExperiment(args.config).run()
    print(f"features: {result['feature_path']}")
    print(f"visibility: {result['visibility_path']}")


if __name__ == "__main__":
    main()
