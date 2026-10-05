"""Frozen CLIP features -> PCA-32 -> equal-prior Gaussian LDA head -> finite cells.

Cells are the LDA-predicted known class plus one outlier cell (Mahalanobis distance to the
nearest class mean above the dev-pool 99th percentile). The class-conditionals of the REACH
mixture are then P(cell | class): confusion columns of a frozen head. Under pure label shift
those columns do not move, which is the frozen-conditionals assumption the ball needs; under
corruption (CIFAR-10-C) they do, and the mixture-incompatibility diagnostic should notice.

Equal priors keep the partition independent of the training prevalence. Nothing is fine-tuned.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

FEATURES = Path(__file__).resolve().parents[1]/'cache'/'features'
CORRUPTIONS = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
               'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
               'elastic_transform', 'pixelate', 'jpeg_compression')


def load(name, root=FEATURES):
    with np.load(Path(root)/f'{name}.npz', allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@dataclass
class LDAHead:
    mean: np.ndarray        # PCA centring
    components: np.ndarray  # 32 x 512
    class_means: np.ndarray # K x 32
    precision: np.ndarray   # 32 x 32 pooled within-class inverse covariance
    threshold: float        # outlier cut on min Mahalanobis distance
    classes: np.ndarray     # original class ids of the K known classes

    @classmethod
    def fit(cls, X, y, classes, dim=32, outlier_quantile=.99):
        X = np.asarray(X, dtype=np.float64)
        keep = np.isin(y, classes)
        X, y = X[keep], y[keep]
        mean = X.mean(0)
        _, _, vt = np.linalg.svd(X-mean, full_matrices=False)
        Z = (X-mean)@vt[:dim].T
        means = np.stack([Z[y == c].mean(0) for c in classes])
        resid = Z-means[np.searchsorted(classes, y)]
        cov = resid.T@resid/(len(Z)-len(classes))
        head = cls(mean, vt[:dim], means, np.linalg.inv(cov), 0., np.asarray(classes))
        head.threshold = float(np.quantile(head._distances(X).min(1), outlier_quantile))
        return head

    def _distances(self, X):
        Z = (np.asarray(X, dtype=np.float64)-self.mean)@self.components.T
        diff = Z[:, None, :]-self.class_means[None]
        return np.einsum('nkd,de,nke->nk', diff, self.precision, diff)

    @property
    def cells(self):
        return len(self.classes)+1

    def cell(self, X):
        """Predicted known-class index, or len(classes) for the outlier cell."""
        out = []
        for start in range(0, len(X), 4096):
            d = self._distances(X[start:start+4096])
            pred = d.argmin(1)
            pred[d.min(1) > self.threshold] = len(self.classes)
            out.append(pred)
        return np.concatenate(out)


def conditionals(cells, y, classes, n_cells):
    """Rows P(cell | class) for each class in `classes` (empirical histograms)."""
    return np.stack([np.bincount(cells[y == c], minlength=n_cells)/max((y == c).sum(), 1)
                     for c in classes])
