### Sources and their exact roles

These are all external papers/theories used in this notebook. The mathematical derivations below are written out; a citation does not substitute for an assumption.

| Key | Source | What we use; what we do not claim |
|---|---|---|
| D | Ni et al., [Possibilistic Predictive Uncertainty for Deep Learning](https://arxiv.org/abs/2605.00600) | Workflow motivation only; no reproduction or comparison of its reported scores. |
| E | Hüllermeier & Waegeman (2021), [Aleatoric and epistemic uncertainty](https://doi.org/10.1007/s10994-021-05946-3) | Sets of possible distributions express incomplete knowledge. No entropy decomposition is estimated here. |
| L | Lipton et al. (2018), [Detecting and Correcting for Label Shift with Black Box Predictors](https://proceedings.mlr.press/v80/lipton18a.html) | Fixed source-conditionals and the mixture moment equation. Our set inference allows non-identification. |
| H | Hoeffding (1963), [Probability Inequalities for Sums of Bounded Random Variables](https://doi.org/10.1080/01621459.1963.10500830) ([accessible paper](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)) | Fixed-sample coordinate concentration plus an explicitly counted union bound. Not an anytime confidence sequence. |
| C | Charnes & Cooper (1962), [Programming with Linear Fractional Functionals](https://doi.org/10.1002/nav.3800090303) | Transform each probability-ratio extremization into a linear program; the transformation is derived below. |
| M | LeCun et al., [MNIST dataset](https://yann.lecun.com/exdb/mnist/) | Cached digit images. Official site returned a gateway error during source checking; no new download or unseen content is relied on. |
| EL | Cohen et al. (2017), [EMNIST](https://arxiv.org/abs/1702.05373), [NIST dataset page](https://www.nist.gov/itl/products-and-services/emnist-dataset) | Cached EMNIST Letters images, transposed as in the prior pilot. |
| F | Xiao et al. (2017), [Fashion-MNIST](https://arxiv.org/abs/1708.07747) | Cached clothing images; the pilot repartitioned the official test pool. Not an official benchmark protocol. |
| B | Yang et al. (2023), [MedMNIST v2](https://doi.org/10.1038/s41597-022-01721-8) | Cached BloodMNIST images, converted to grayscale in the prior pilot. No clinical interpretation. |

Local foundations: `paper/sections/setup.tex` and `paper/sections/theory.tex` supply the source-mixture question and identification arguments. This notebook uses **coordinate boxes and joint-mass LPs**, not that draft's Euclidean MMD ball. It preserves reference uncertainty jointly and does not import an unproved contraction rate from the draft.

The paper fetches were read-only. The run exports this source list, the notebook, helper source, input hashes, and configuration for auditing.