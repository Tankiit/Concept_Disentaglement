"""Statistical components independent of WRN downloads and image extraction."""
import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def stratified_indices(labels, per_class, seed):
    labels = np.asarray(labels)
    if not isinstance(per_class, int) or per_class < 1:
        raise ValueError('per_class must be a positive integer')
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(labels):
        ix = np.flatnonzero(labels == c)
        if len(ix) < per_class:
            raise ValueError(f'class {c}: insufficient rows')
        out.extend(rng.permutation(ix)[:per_class])
    return np.asarray(out, dtype=int)


def test_pools(labels, per_class, seed):
    """Disjoint, class-balanced calibration, evaluation and partner rows."""
    chosen = stratified_indices(labels, 3*per_class, seed)
    by_class = chosen.reshape(-1, 3*per_class)
    return {name: by_class[:, j*per_class:(j+1)*per_class].reshape(-1)
            for j, name in enumerate(('calibration', 'evaluation', 'partner'))}


def pair_indices(labels, base, partner, kind, seed):
    """One-to-one partner assignment; no clean calibration rows are reused."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels[base])
    if kind not in ('same_class_mix', 'cross_class_mix'):
        raise ValueError(kind)
    cycle = rng.permutation(classes)
    mapping = dict(zip(cycle, np.roll(cycle, -1)))
    if kind == 'cross_class_mix' and len(classes) < 2:
        raise ValueError('Cross-class mixing requires two classes')
    result = np.empty(len(base), dtype=int)
    for c in classes:
        slots = np.flatnonzero(labels[base] == c)
        other = c if kind == 'same_class_mix' else mapping[c]
        candidates = partner[labels[partner] == other]
        if len(candidates) != len(slots):
            raise ValueError('Balanced partner and evaluation pools required')
        result[slots] = rng.permutation(candidates)
    return result


def transform_images(images, strength, kind, partner=None, noise=None, noise_std=.3):
    """Transform RGB pixels in [0,1], BEFORE WRN normalization and extraction.

    Mixing ends at another ID image at strength=1. Noise uses a fixed direction
    across strengths, with clipping; strength is not a semantic OOD label.
    """
    x = np.asarray(images, dtype=np.float32)
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength must be in [0,1]')
    if x.ndim != 4 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('Expected finite NHWC images in [0,1]')
    if kind in ('same_class_mix', 'cross_class_mix'):
        other = np.asarray(partner, dtype=np.float32)
        if other.shape != x.shape or not np.isfinite(other).all() or other.min() < 0 or other.max() > 1:
            raise ValueError('Invalid partner images')
        return (1-strength)*x + strength*other
    if kind == 'gaussian_noise':
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape != x.shape or not np.isfinite(noise).all() or not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError('Invalid noise or scale')
        return np.clip(x+strength*noise_std*noise, 0, 1)
    raise ValueError(kind)


class GaussianDensity:
    """DDU-style Gaussian mixture on frozen features, NOT full DDU training.

    PCA is fitted on training features only. Classwise Ledoit-Wolf covariance
    shrinkage plus a small ridge permits small-sample checks without singularities.
    The optional PCA and shrinkage are explicit departures from the DDU recipe.
    """
    def __init__(self, pca_dim=32):
        self.pca_dim = pca_dim

    def fit(self, features, labels):
        x, y = np.asarray(features, float), np.asarray(labels)
        if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all():
            raise ValueError('Invalid features/labels')
        self.pca = None
        if self.pca_dim is not None:
            if self.pca_dim < 1:
                raise ValueError('pca_dim must be positive or None')
            self.pca = PCA(n_components=min(self.pca_dim, x.shape[1], len(x)-1), svd_solver='full')
            x = self.pca.fit_transform(x)
        self.classes = np.unique(y)
        self.models = []
        for c in self.classes:
            xc = x[y == c]
            if len(xc) < 2:
                raise ValueError('At least two density-training examples per class required')
            est = LedoitWolf().fit(xc)
            cov = est.covariance_ + np.eye(x.shape[1])*max(np.trace(est.covariance_)/x.shape[1]*1e-6, 1e-8)
            chol = np.linalg.cholesky(cov)
            self.models.append((est.location_, np.linalg.inv(chol),
                                2*np.log(np.diag(chol)).sum(), np.log(len(xc)/len(x))))
        return self

    def score(self, features):
        x = np.asarray(features, float)
        if self.pca is not None:
            x = self.pca.transform(x)
        logs = []
        for mean, inv_chol, logdet, logprior in self.models:
            whitened = (x-mean)@inv_chol.T
            logs.append(logprior-.5*((whitened**2).sum(1)+logdet+x.shape[1]*np.log(2*np.pi)))
        return -logsumexp(np.stack(logs, axis=1), axis=1)


class BootstrapHeads:
    """Stratified bootstrap linear heads on ONE frozen WRN, not deep ensembles.

    EU is conditional on this representation. It omits backbone uncertainty.
    C is fixed before evaluation. All members use training-only standardization.
    """
    def __init__(self, members=5, seed=0, C=1.):
        if members < 2:
            raise ValueError('EU comparison requires at least two bootstrap heads')
        self.members, self.seed, self.C = members, seed, C

    def fit(self, features, labels):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        x, y = np.asarray(features), np.asarray(labels)
        self.scaler = StandardScaler().fit(x)
        x = self.scaler.transform(x)
        self.classes = np.unique(y)
        rng = np.random.default_rng(self.seed)
        self.heads = []
        for member in range(self.members):
            ix = np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum(), replace=True)
                                 for c in self.classes])
            head = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed+member)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                head.fit(x[ix], y[ix])
            self.heads.append(head)
        return self

    def probabilities(self, features):
        x = self.scaler.transform(features)
        return np.stack([head.predict_proba(x) for head in self.heads])


def entropy(p):
    p = np.asarray(p)
    return -(p*np.log(np.clip(p, 1e-15, 1))).sum(-1)


def score_bundle(features, logits, density, heads):
    p = heads.probabilities(features)
    mean = p.mean(0)
    tu, au = entropy(mean), entropy(p).mean(0)
    logits = np.asarray(logits, float)
    wrn_prob = softmax(logits, axis=1)
    scores = dict(density=density.score(features), EU=np.maximum(tu-au, 0),
                  ensemble_entropy=tu, mean_member_entropy=au,
                  WRN_entropy=entropy(wrn_prob), MSP=1-wrn_prob.max(1),
                  energy=-logsumexp(logits, axis=1))
    if not all(np.isfinite(s).all() for s in scores.values()):
        raise RuntimeError('Nonfinite detector score')
    return scores, heads.classes[mean.argmax(1)]


def thresholds_from_known(scores, fpr=.05):
    if not 0 < fpr < 1:
        raise ValueError('fpr must be in (0,1)')
    return {name: float(np.quantile(values, 1-fpr, method='higher'))
            for name, values in scores.items()}


def shift_auc(clean, changed):
    """Discrimination of transformed vs clean inputs, NOT semantic OOD AUROC."""
    return float(roc_auc_score(np.r_[np.zeros(len(clean)), np.ones(len(changed))],
                               np.r_[clean, changed]))


def paired_auc_difference(clean_a, shifted_a, clean_b, shifted_b, seed=0, repeats=200):
    """Bootstrap original-image pairs; fitted models/calibration remain fixed."""
    n = len(clean_a)
    arrays = [np.asarray(x) for x in (clean_a, shifted_a, clean_b, shifted_b)]
    if any(x.shape != (n,) for x in arrays) or n < 2:
        raise ValueError('Need aligned clean/transformed pairs')
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        ix = rng.integers(n, size=n)
        a, b, c, d = [x[ix] for x in arrays]
        values.append(shift_auc(a, b)-shift_auc(c, d))
    return dict(difference=shift_auc(arrays[0], arrays[1])-shift_auc(arrays[2], arrays[3]),
                lower=float(np.quantile(values, .025)), upper=float(np.quantile(values, .975)))
