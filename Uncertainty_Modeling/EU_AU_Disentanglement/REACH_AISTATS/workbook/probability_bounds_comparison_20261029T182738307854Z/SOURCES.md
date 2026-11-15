### Explicit sources and adaptation map

| Source | Used here |
|---|---|
| [DAPPr repository](https://github.com/MaxwellYaoNi/DAPPr), [train.py](https://github.com/MaxwellYaoNi/DAPPr/blob/main/train.py) | AdamW, warmup/cosine scheduling, ID-validation checkpoint selection, ID-positive confidence/OOD AUPR convention. Architecture, datasets and schedule length are explicitly adapted. |
| [DAPPr.py](https://github.com/MaxwellYaoNi/DAPPr/blob/main/DAPPr.py), [paper](https://arxiv.org/abs/2605.00600) | Actual DAPPr loss formula and original score K/sum(alpha), implemented below. No claim that our interval score is this scalar. |
| [Probly README](https://github.com/pwhofman/probly), [representer interface](https://github.com/pwhofman/probly/blob/main/src/probly/representer/_representer.py) | Actual `dropout`, `representer`, `Representation`, custom `quantify` dispatch and OOD evaluation API. Installed version is recorded; source snapshots are saved. |
| Gal & Ghahramani (2016), [Dropout as a Bayesian Approximation](https://proceedings.mlr.press/v48/gal16.html) | MC-dropout sampled predictors. This does not supply coverage guarantees for sample min/max. |
| Hüllermeier & Waegeman (2021), [uncertainty concepts](https://doi.org/10.1007/s10994-021-05946-3) | Sets of candidate probability distributions describe incomplete knowledge. |
| Lipton et al. (2018), [label-shift estimation](https://proceedings.mlr.press/v80/lipton18a.html) | Frozen source-conditionals and mixture moment equation; identification depends on source distinguishability. |
| Hoeffding (1963), [bounded-variable concentration](https://doi.org/10.1080/01621459.1963.10500830) | Coordinate confidence boxes for known-source reference rows and deployment histogram, conditional on sampling/model assumptions. |
| Charnes & Cooper (1962), [linear-fractional programming](https://doi.org/10.1002/nav.3800090303) | Exact ratio extrema over the declared joint-mass polytope, subject to numerical tolerances. |
| [MNIST](https://yann.lecun.com/exdb/mnist/), [EMNIST](https://arxiv.org/abs/1702.05373), [Fashion-MNIST](https://arxiv.org/abs/1708.07747), [MedMNIST v2](https://doi.org/10.1038/s41597-022-01721-8) | Dataset provenance. Fashion uses the previously cached official-test repartition and is not an official benchmark result. |

Local mathematical foundation: `real/source_acquisition.py` and the previous `source_acquisition_four_datasets.ipynb`. Here known sources are individual ID classes, rather than the previous two groups. The candidate comprises all within-dataset OOD classes, mixed with equal class weight. This method has candidate-reference access; dropout and DAPPr training do not. Therefore the comparison is not an equal-information superiority test.