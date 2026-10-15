"""Disjoint real-feature pools: dev, labelled reference, unlabelled log, evaluation.

The guarantee the downstream intervals need is that the pools are disjoint and
independent. dev and reference are disjoint prefixes of one seeded train
permutation (real/audit.py: first 10k images head-fit the frozen head). log and
eval are contiguous halves of the 10k CIFAR-10 test pool, split on image index,
so disjointness holds by construction and cannot silently fail.

A CIFAR-10-C file is severity-block aligned to the test pool (row s*N_test + j
is test image j at severity s, per modal_extract.py), so a corrupted log reuses
the log half's exact indices and still never touches eval. The alignment itself
is checkable: the corruption file's labels must agree with the test labels on
every shared row, and the loader refuses a file where they do not — the one
disjointness-adjacent assumption that arrays alone can verify.
"""
import numpy as np

from real.head import load, FEATURES

DEV = 10_000  # head-fitting prefix of the train permutation, matching real/audit.py


def split_test(root=None, frac=.5):
    """Contiguous halves of the test pool by image index: (log_idx, eval_idx).

    Contiguous is enough: the CIFAR-10 test pool is not class-sorted (per-class
    counts land within ~±13 of 500 in each half), and the log half doubles as
    the index base for corruption blocks.
    """
    n = len(load('cifar10_test', FEATURES if root is None else root)['y'])
    h = int(round(frac*n))
    return np.arange(h), np.arange(h, n)


def check_disjoint(pools):
    """Raise unless log and eval (and dev and reference) share no image index.

    Train-side and test-side pools are disjoint by dataset. This checks the
    bookkeeping we own; for foreign bundles nothing here replaces the
    independence argument in data/cache.py.
    """
    for a, b in (('log', 'eval'), ('dev', 'reference')):
        if len(np.intersect1d(pools[a]['index'], pools[b]['index'], assume_unique=True)):
            raise ValueError(f'{a} and {b} pools overlap on '
                             f'{len(np.intersect1d(pools[a]["index"], pools[b]["index"]))} images')


def load_pools(root=None, m_per_class=None, seed=0, frac=.5, corruption=None, severity=None):
    """Return dev/reference/log/eval pools; log and eval never share an image.

    root: features directory (default cache/features). m_per_class: optional
    labelled-reference budget, stratified over the 10 classes. corruption:
    CIFAR-10-C name; the log then draws its rows from that file at the given
    severity, index-aligned to the log half, while eval stays clean.
    """
    root = FEATURES if root is None else root
    train, test = load('cifar10_train', root), load('cifar10_test', root)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train['y']))
    dev_idx, ref_idx = perm[:DEV], perm[DEV:]
    if m_per_class is not None:
        ref_idx = _stratify(train['y'], ref_idx, m_per_class, rng)
    log_idx, eval_idx = split_test(root, frac)
    if corruption is None:
        log_X, log_y, log_pool = test['X'][log_idx], test['y'][log_idx], 'cifar10_test'
    else:
        log_X, log_y = _corruption_rows(corruption, severity, log_idx, root)
        log_pool = f'cifar10c_{corruption}'
    pools = dict(
        dev=dict(X=train['X'][dev_idx], y=train['y'][dev_idx], index=dev_idx, pool='cifar10_train'),
        reference=dict(X=train['X'][ref_idx], y=train['y'][ref_idx], index=ref_idx, pool='cifar10_train'),
        log=dict(X=log_X, y=log_y, index=log_idx, pool=log_pool),
        eval=dict(X=test['X'][eval_idx], y=test['y'][eval_idx], index=eval_idx, pool='cifar10_test'),
    )
    check_disjoint(pools)
    return pools


def _stratify(y, idx, m, rng):
    """m per class from the candidate indices idx, returned in the same index space
    as idx (train rows), sorted for a target-blind row order (train/cache.py: row
    order must not use outcomes)."""
    counts = {c: int((y[idx] == c).sum()) for c in np.unique(y[idx])}
    short = [c for c, n in counts.items() if n < m]
    if short:
        raise ValueError(f'class(es) {short} have fewer than m={m} labelled images')
    return np.sort(np.concatenate([rng.choice(idx[y[idx] == c], m, replace=False)
                                   for c in sorted(counts)]))


def _corruption_rows(corruption, severity, log_idx, root):
    """Rows of one CIFAR-10-C severity block at the log half's indices.

    The label-agreement check is what makes the block alignment trustworthy:
    5000 chance agreements over 10 classes are impossible, so a mismatch means
    the file is not the alignment modal_extract.py wrote.
    """
    if severity is None:
        raise ValueError('corruption logs need a severity in 1..5')
    test_y = load('cifar10_test', root)['y']
    pool = load(f'cifar10c_{corruption}', root)
    n = len(test_y)
    if 'severity' not in pool or len(pool['y']) % n:
        raise ValueError(f'cifar10c_{corruption} is not severity-block aligned to the test pool')
    rows = int(severity-1)*n + log_idx
    if not np.array_equal(pool['y'][rows], test_y[log_idx]):
        raise ValueError(f'cifar10c_{corruption} labels disagree with the test pool '
                         'at the log half\'s indices; refusing an unaligned file')
    return pool['X'][rows], pool['y'][rows]
