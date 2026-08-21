# CUB masking protocol

## Mask the pixels, then run the backbone

Selecting patch tokens after a ViT forward pass does not restrict information:
self-attention has already mixed every patch into every token. The experiment
must resize each image directly to 224 x 224, apply the group mask to those
pixels, and only then run DINOv2.

The landmark coordinates are in each original image's coordinate system. Scale
them with `x224 = x * 224 / W` and `y224 = y * 224 / H`; do not let the image
processor introduce a center crop. Sweep the landmark-disc radius and report
part visibility/drop rates. Missing landmarks should be dropped from the
corresponding concept's training rather than replaced with full or blank images.

Begin with zero fill. Blur can leak the low-frequency colour and size evidence
used by global concepts. As a sanity check, compare masked and full-image CLS
features: their cosine similarity should be clearly below one.

## Configuration and execution

Dataset anatomy and concept-prefix mappings live under
`src/gauge/experiments/configs/datasets/`. Experiment YAML files live under
`configs/experiments/`; `cub_masked_dinov2.yaml` inherits shared runtime and
training settings from `shared_defaults.yaml`.

From the `CEM` directory, run:

```bash
python -m gauge.experiments.masked_backbone_extraction \
  --config configs/experiments/cub_masked_dinov2.yaml
```

To add a dataset, create `<dataset-name>.json` or YAML with `part_names`,
`part_to_group`, `concept_prefix_to_group`, and `global_group`, then set
`dataset.name` in an experiment config.
