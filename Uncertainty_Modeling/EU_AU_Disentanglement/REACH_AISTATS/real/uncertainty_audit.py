"""Shared, testable controls for exploratory uncertainty experiments."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def mc_probs(head, x, T=30):
    """Activate dropout only; restore every module's original mode afterwards."""
    modes = {m: m.training for m in head.modules()}
    try:
        head.eval()
        for m in head.modules():
            if isinstance(m, torch.nn.modules.dropout._DropoutNd):
                m.train()
        with torch.no_grad():
            return torch.stack([head(x).softmax(-1) for _ in range(T)]).cpu().numpy()
    finally:
        for m, mode in modes.items():
            m.training = mode


def uncertainty(p):
    p = np.asarray(p, dtype=np.float64)
    entropy = lambda a: -(a*np.log(np.clip(a, 1e-12, None))).sum(-1)
    tu, au = entropy(p.mean(0)), entropy(p).mean(0)
    return dict(tu=tu, au=au, eu=tu-au)


def class_vs_known(is_novel, labels, class_id):
    """Inputs must already exclude exposed unknowns (KUC)."""
    is_novel = np.asarray(is_novel, dtype=bool)
    selected = ~is_novel | (np.asarray(labels) == class_id)
    return selected, is_novel[selected].astype(int)


def low_eu_subset(is_novel, eu, threshold):
    is_novel = np.asarray(is_novel, dtype=bool)
    selected = ~is_novel | (np.asarray(eu) < threshold)
    return selected, is_novel[selected].astype(int)


def cluster_mean_difference(values, positive, row_ids, n_boot=500, seed=0):
    """Positive-minus-negative mean; resample source images, retaining their repeats."""
    values, positive = np.asarray(values), np.asarray(positive, dtype=bool)
    ids, inverse = np.unique(row_ids, return_inverse=True)
    if not positive.any() or positive.all():
        raise ValueError('Both groups are required')
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        weights = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))[inverse]
        if weights[positive].sum() and weights[~positive].sum():
            diffs.append(np.average(values[positive], weights=weights[positive]) -
                         np.average(values[~positive], weights=weights[~positive]))
    if not diffs:
        raise ValueError('No valid bootstrap draws')
    return dict(difference=float(values[positive].mean()-values[~positive].mean()),
                lower=float(np.quantile(diffs,.025)),upper=float(np.quantile(diffs,.975)))


def paired_auc_difference(y, score, baseline, n_boot=1000, seed=0, pairs=False):
    """Stratified paired-score bootstrap; optional clean/attack row pairing.

    Conditions on trained heads and the evaluated class inventory. This is not
    uncertainty across training seeds or unseen class inventories. With pairs=True,
    the positive and negative rows must be corresponding images in the same order.
    """
    y = np.asarray(y, dtype=int)
    score, baseline = np.asarray(score), np.asarray(baseline)
    neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if not len(neg) or not len(pos):
        raise ValueError('Both classes are required')
    if pairs and len(pos) != len(neg):
        raise ValueError('Paired conditions must have the same number of rows')
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(len(neg), size=len(neg))
        j = i if pairs else rng.integers(len(pos), size=len(pos))
        ix = np.r_[neg[i], pos[j]]
        diffs.append(roc_auc_score(y[ix], score[ix])-roc_auc_score(y[ix], baseline[ix]))
    return dict(difference=float(roc_auc_score(y, score)-roc_auc_score(y, baseline)),
                lower=float(np.quantile(diffs, .025)), upper=float(np.quantile(diffs, .975)))


def pgd_l2(head, x, y, eps, steps=20, alpha=None):
    """Deterministic-head PGD in an actual Euclidean ball around x; no sphere step."""
    if eps < 0 or steps < 1:
        raise ValueError('Require eps >= 0 and steps >= 1')
    modes = {m: m.training for m in head.modules()}
    try:
        head.eval()
        delta = torch.zeros_like(x)
        alpha = eps/4 if alpha is None else alpha
        for _ in range(steps):
            xa = (x.detach()+delta).requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(head(xa), y)
            grad, = torch.autograd.grad(loss, xa)
            with torch.no_grad():
                delta += alpha*grad/grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
                factor = (eps/delta.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)
                delta *= factor
        return x.detach()+delta, delta
    finally:
        for m, mode in modes.items():
            m.training = mode


def linear_boundary_path(weight, bias, x, labels, multiple):
    """Move correctly classified rows toward their nearest linear decision plane.

    The first crossing is exact at multiple=1 (up to ties). Beyond it competitors
    can switch. Rows initially misclassified are returned unchanged and flagged.
    No assertion that perturbed features correspond to realizable images is made.
    """
    w, b, x, labels = map(np.asarray, (weight, bias, x, labels))
    logits = x@w.T+b
    correct = logits.argmax(1) == labels
    diffs = w[labels, None, :]-w[None, :, :]
    norm = np.linalg.norm(diffs, axis=2)
    gap = logits[np.arange(len(x)), labels, None]-logits
    distances = np.full_like(gap, np.inf, dtype=float)
    np.divide(gap, norm, out=distances, where=norm > 1e-12)
    distances[np.arange(len(x)), labels] = np.inf
    rival = distances.argmin(1)
    distance = distances[np.arange(len(x)), rival]
    if np.any(~np.isfinite(distance[correct])):
        raise ValueError('No finite rival boundary')
    direction = diffs[np.arange(len(x)), rival]/norm[np.arange(len(x)), rival, None].clip(1e-12)
    delta = -multiple*np.where(correct, distance, 0)[:, None]*direction
    return x+delta, dict(correct=correct, rival=rival, distance=distance, delta=delta)
