"""wide_resnet50_2 OSS example: KKC/KUC/UUC (oss) or plain fixed-class OOD worlds.

Reuse of example.py machinery with two world modes over the same world-dict
interface: --mode oss keeps the DynamicOSS known-unknown budget; --mode plain
fixes the known classes and pools everything else as OOD (same dataset or a
second dataset, e.g. CIFAR-10 vs MNIST). Head input dim is wired to the
feature width, so 2048-d wide_resnet50_2 features work (example.py's default
MLP width 512 only fits resnet18 by coincidence). Budget tags use sklearn
exact kNN (faiss segfaults under torch here), and extraction, head training
and EU evaluation all run on config.device (mps/cuda/cpu).
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from probly.representation.credal_set.array import ArrayProbabilityIntervalsCredalSet
from probly.plot import plot_credal_set

from example import (extract_features_timm, make_oss, group_idx, make_loader,
                     Head, ETFHead, eu_bald, auroc)
from example import load_cached, prep

OOD_LABEL_OFFSET = 1000   # second-dataset OOD labels are shifted to stay disjoint


def features_path(model_name, dataset):
    """Cache file for frozen features; cifar10 keeps example.py's c10test name."""
    tag = 'c10test' if dataset == 'cifar10' else f'{dataset}test'
    return Path('cache/features')/f'timm_{model_name}_{tag}.npz'


def _tv_datasets():
    from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, SVHN
    return {
        'cifar10': lambda root, dl: (CIFAR10(root, train=False, download=dl), False),
        'cifar100': lambda root, dl: (CIFAR100(root, train=False, download=dl), False),
        'mnist': lambda root, dl: (MNIST(root or 'cache/mnist', train=False, download=dl), True),
        'fashionmnist': lambda root, dl: (FashionMNIST(root or 'cache/fmnist', train=False, download=dl), True),
        'svhn': lambda root, dl: (SVHN(root or 'cache/svhn', split='test', download=dl), False),
    }


SUPPORTED_DATASETS = tuple(sorted(_tv_datasets()))


def load_test_dataset(name, root=None, download=True):
    """Torchvision test split by name -> (dataset, grayscale_flag). Fails loudly."""
    table = _tv_datasets()
    if name not in table:
        raise ValueError(f'unsupported dataset {name!r}; supported: {SUPPORTED_DATASETS}')
    return table[name](root, download)


def dataset_pixels(ds, ix):
    """NHWC float [0,1] pixels for row indices, across torchvision layouts.

    Handles (N, H, W, 3) numpy (CIFAR), (N, H, W) grayscale tensors (MNIST/
    FashionMNIST) and (N, C, H, W) arrays (SVHN); falls back to __getitem__
    for datasets without a .data array.
    """
    ix = np.asarray(ix, dtype=int)
    data = getattr(ds, 'data', None)
    if data is None:
        return np.stack([np.asarray(ds[int(i)][0].convert('RGB'), np.float32)/255. for i in ix])
    arr = np.asarray(data)[ix]
    if arr.ndim == 3:                                   # (N, H, W) grayscale
        arr = np.repeat(arr[..., None], 3, axis=3)
    elif arr.ndim == 4 and arr.shape[1] in (1, 3):      # (N, C, H, W)
        arr = arr.transpose(0, 2, 3, 1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
    return arr.astype(np.float32)/255.


def extract_features(model_name='wide_resnet50_2', dataset='cifar10', n_per_class=None,
                     batch_size=256, device='cpu', root='cache/cifar10'):
    """Frozen timm features for any supported dataset; cifar10 keeps example.py's cache."""
    if dataset == 'cifar10':
        return extract_features_timm(model_name, n_per_class=n_per_class, root=root, device=device)
    path = features_path(model_name, dataset)
    if path.exists():
        return load_cached(path)
    import timm
    ds, _ = load_test_dataset(dataset, root)
    targets = getattr(ds, 'targets', None)
    if targets is None:
        targets = ds.labels                             # SVHN
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    tfm = timm.data.create_transform(**timm.data.resolve_model_data_config(model))
    feats, labels = [], []
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            chunk = range(start, min(start+batch_size, len(ds)))
            x = torch.stack([tfm(ds[j][0].convert('RGB')) for j in chunk]).to(device)
            feats.append(model(x).cpu().numpy())
            labels.extend(int(targets[j]) for j in chunk)
    X = np.vstack(feats).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y)
    print(f'timm {model_name} {dataset}: X {X.shape}, classes {np.unique(y).tolist()} -> {path}')
    return X, y


def stratified_split(rows_by_class, train_size=.7, val_size=.2, seed=0):
    """Per-class row split with every class present in train, val and test."""
    rng = np.random.default_rng(seed)
    parts = {name: [] for name in ('train', 'val', 'test')}
    for c, rows in rows_by_class:
        rows = np.sort(np.asarray(rows, dtype=int))
        perm = rng.permutation(len(rows))
        n_train, n_val = int(round(train_size*len(rows))), int(round(val_size*len(rows)))
        if n_train < 1 or n_val < 1 or len(rows)-n_train-n_val < 1:
            raise ValueError(f'class {c}: insufficient rows for a {train_size:g}/{val_size:g} split')
        parts['train'].append(rows[perm[:n_train]])
        parts['val'].append(rows[perm[n_train:n_train+n_val]])
        parts['test'].append(rows[perm[n_train+n_val:]])
    return {name: np.sort(np.concatenate(chunks)) for name, chunks in parts.items()}


def make_plain_ood(id_path, ood_path=None, known=None, n_known=None, seed=0,
                   train_size=.7, val_size=.2, ood_label_offset=OOD_LABEL_OFFSET):
    """Plain OOD world: fixed known classes, one OOD pool, no known-unknown split.

    OOD rows (held-out classes or a second feature cache with offset labels)
    all go to the test split, so val stays known-only for calibration. The
    returned dict mirrors make_oss's keys, with kuc and uuc_val empty.
    """
    X, y = load_cached(id_path)
    classes = [int(c) for c in np.unique(y)]
    if known is not None:
        known = sorted({int(c) for c in known})
        if not known or set(known)-set(classes):
            raise ValueError(f'known must be a nonempty subset of {classes}')
    else:
        # Cross-dataset OOD defaults to every ID class known; held-out mode splits half.
        default_n = len(classes) if ood_path is not None else max(1, len(classes)//2)
        n = n_known if n_known is not None else default_n
        rng = np.random.default_rng(seed)
        known = sorted(int(c) for c in rng.choice(classes, size=min(n, len(classes)), replace=False))
    if ood_path is None:
        novel = [c for c in classes if c not in known]
        if not novel:
            raise ValueError('Every class is known and no ood_path given; plain mode needs OOD rows')
        X_ood = y_ood = None
    else:
        X_ood, y_ood_raw = load_cached(ood_path)
        y_ood = y_ood_raw.astype(np.int64) + ood_label_offset
        if set(y_ood.tolist()) & set(classes):
            raise ValueError('OOD labels collide with ID labels; raise ood_label_offset')
        novel = sorted({int(c) for c in np.unique(y_ood)})
    idx = stratified_split([(c, np.flatnonzero(y == c)) for c in known], train_size, val_size, seed)
    if X_ood is not None:
        ood_rows = np.arange(len(y), len(y)+len(y_ood))
        X, y = np.vstack([X, X_ood]), np.concatenate([y, y_ood])
    else:
        ood_rows = np.concatenate([np.flatnonzero(y == c) for c in novel])
    idx['test'] = np.sort(np.concatenate([idx['test'], np.sort(ood_rows)]))
    return dict(X=X, y=y, oss=None, idx=idx, kkc=known, kuc=[], uuc=novel,
                uuc_val=[], uuc_test=novel, n_kkc=len(known), n_kuc=0, n_uuc=len(novel),
                budget=dict(n_kkc=len(known), kuc=0, uuc_val=0, uuc_test=len(novel)),
                mode='plain', ood_path=str(ood_path) if ood_path else None)


def build_head(name, d_in, k):
    """Head wired to the actual feature width; example.py's mlp default is 512."""
    if name == 'mlp':
        return Head(d=d_in, k=k)
    if name == 'etf':
        return ETFHead(d_in=d_in, k=k)
    raise ValueError(name)


def train_head(head, loader, lut, epochs=40, lr=1e-3, seed=0, device='cpu'):
    """example.train with features and labels moved to device; per-epoch losses out."""
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    torch.manual_seed(seed)
    losses = []
    for _ in range(epochs):
        head.train()
        total, n = 0.0, 0
        for b in loader:
            x, y = b['X'].to(device), lut[b['y']].to(device)
            opt.zero_grad()
            loss = F.cross_entropy(head(x), y)
            loss.backward()
            opt.step()
            total += float(loss.detach())*len(y)
            n += len(y)
        losses.append(total/n)
    return losses


def credal_intervals(stacked):
    """Per-class probability intervals from T sampled predictors (probly credal set).

    stacked: (T, N, K) softmax samples; returns lower, upper, width (N, K) with
    width = upper - lower, i.e. the envelope of the sampled predictives.
    """
    intervals = ArrayProbabilityIntervalsCredalSet(lower_bounds=stacked.min(axis=0),
                                                   upper_bounds=stacked.max(axis=0))
    return intervals.lower(), intervals.upper(), intervals.width()


def bald_and_credal_width(stacked):
    """EU and mean credal width from ONE (T, N, K) stack of sampled predictives.

    EU is BALD, exactly example.eu_bald's expression, so margins, EU and the
    credal envelope all describe the same sampled heads. Note the asymmetry the
    envelope brings: width sees only per-class extrema, BALD all T predictions
    (29x(1,0)+1x(0,1) and 15x/15x share width 1 but not BALD).
    """
    mean = stacked.mean(0)
    eu = (-(mean * mean.clamp_min(1e-12).log()).sum(-1)
          + (stacked * stacked.clamp_min(1e-12).log()).sum(-1).mean(0))
    _, _, width = credal_intervals(stacked.detach().cpu().numpy())
    return eu, torch.from_numpy(width.mean(-1))


def auroc_fast(known, ood):
    """Exact AUROC via Mann-Whitney midranks; identical to example.auroc's value.

    example.auroc scores all n_k x n_o pairs in a Python loop, which is minutes
    per call at tercile sizes inside a bootstrap; this is O(n log n) with the
    same tie handling (ties count 1/2).
    """
    from scipy.stats import rankdata
    known, ood = np.asarray(known, float), np.asarray(ood, float)
    if len(known) == 0 or len(ood) == 0:
        raise ValueError('auroc needs both classes')
    ranks = rankdata(np.concatenate([known, ood]))
    n_k, n_o = len(known), len(ood)
    r_ood = ranks[n_k:].sum()
    return float((r_ood - n_o*(n_o + 1)/2)/(n_k*n_o))


def rank_disagreement(a, b, tol=.1):
    """Where two scores rank the same examples differently.

    Percentile ranks per example; 'frac' is the share whose ranks differ by more
    than tol, 'worst' the indices by decreasing gap. Aggregate means can hide
    these, so per-example comparison precedes any 'the scores agree' claim.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 1
    rank = lambda v: (np.argsort(np.argsort(v)) + 1)/len(v)
    gap = np.abs(rank(a)-rank(b))
    return dict(frac=float((gap > tol).mean()), max_gap=float(gap.max()),
                worst=np.argsort(-gap)[:10], n=len(a), tol=tol)


def plot_credal_examples(stacks, out_path, class_labels=None, series_labels=None):
    """probly-native credal-set plot for chosen examples.

    stacks: dict name -> (T, K) sampled predictives for ONE example each. Every
    example becomes a ProbabilityIntervalsCredalSet (per-class min/max envelope);
    probly overlays the batch on one axes (radar for K >= 4 classes, ternary
    simplex for K = 3). Returns the envelope bounds for tests/records.
    """
    import matplotlib
    matplotlib.use('Agg')
    lowers, uppers = [], []
    for stack in stacks.values():
        lo, up, _ = credal_intervals(np.asarray(stack)[:, None, :])
        lowers.append(lo[0])
        uppers.append(up[0])
    credal_set = ArrayProbabilityIntervalsCredalSet(lower_bounds=np.asarray(lowers),
                                                    upper_bounds=np.asarray(uppers))
    ax = plot_credal_set(credal_set, labels=class_labels,
                         series_labels=series_labels or list(stacks))
    ax.figure.savefig(out_path, dpi=150, bbox_inches='tight')
    return dict(lower=np.asarray(lowers), upper=np.asarray(uppers))


def sampling_stability(head, X, is_ood, Ts=(10, 30, 100), repeats=20, device='cpu'):
    """Dropout-sampling variation with head, rows and preprocessing all fixed.

    Every repeat redraws only the dropout masks, so spread across repeats is
    sampling variation and nothing else. Envelope width over nested pass sets
    can only grow with T (min down, max up), so cw scales are T-dependent.
    """
    x = torch.from_numpy(np.ascontiguousarray(X, np.float32)).to(device)
    known, ood = ~is_ood, is_ood
    out = {}
    for t in Ts:
        eus, cws, eu_means, cw_means = [], [], [], []
        for _ in range(repeats):
            with torch.no_grad():
                stacked = torch.stack([F.softmax(head(x), dim=1) for _ in range(t)])
            eu, cw = bald_and_credal_width(stacked)
            eus.append(auroc_fast(eu.cpu().numpy()[known], eu.cpu().numpy()[ood]))
            cws.append(auroc_fast(cw.numpy()[known], cw.numpy()[ood]))
            eu_means.append(float(eu.mean()))
            cw_means.append(float(cw.mean()))
        out[int(t)] = dict(eu_auroc_mean=float(np.mean(eus)), eu_auroc_std=float(np.std(eus)),
                           cw_auroc_mean=float(np.mean(cws)), cw_auroc_std=float(np.std(cws)),
                           eu_mean=float(np.mean(eu_means)), cw_mean=float(np.mean(cw_means)),
                           repeats=repeats)
    return out


def choose_control_partners(start_rows, labels, n_same, seed=0):
    """Control partners with declared class relation: first n_same same-class, rest different.

    Raises when the requested relation has no eligible candidate, so a silently
    mixed control arm cannot happen.
    """
    rng = np.random.default_rng(seed)
    partners, relation = [], []
    for i, s in enumerate(start_rows):
        want_same = i < n_same
        pool = np.flatnonzero(labels == labels[s]) if want_same else np.flatnonzero(labels != labels[s])
        pool = pool[pool != s]
        pool = np.setdiff1d(pool, np.asarray(partners, dtype=int), assume_unique=False)
        if len(pool) == 0:
            raise ValueError(f'no eligible {"same-class" if want_same else "different-class"} partner for start {s}')
        partners.append(int(rng.choice(pool)))
        relation.append('same_class' if want_same else 'diff_class')
    return np.asarray(partners, dtype=int), np.asarray(relation)


def stratified_tercile_auroc(eu, cw, hardness, is_ood, n_boot=500, seed=0):
    """AUROC within hardness terciles and a bootstrap CI on width - EU per tercile.

    Hardness must come from outside the scored heads (here: frozen-feature kNN
    distance). Tercile edges are hardness quantiles over the pooled rows and
    stay fixed across resamples; each bootstrap redraws known and OOD rows
    within a tercile, so the interval covers row resampling with edges fixed.
    """
    eu, cw = np.asarray(eu, float), np.asarray(cw, float)
    hardness, is_ood = np.asarray(hardness, float), np.asarray(is_ood, bool)
    assert eu.shape == cw.shape == hardness.shape == is_ood.shape and len(eu) > 2
    edges = np.quantile(hardness, [1/3, 2/3])
    groups = np.digitize(hardness, edges)
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(3):
        sel = np.flatnonzero(groups == g)
        known, ood = sel[~is_ood[sel]], sel[is_ood[sel]]
        if len(known) == 0 or len(ood) == 0:
            rows.append(dict(tercile=int(g), n=int(len(sel)), n_known=int(len(known)),
                             n_ood=int(len(ood)),
                             hardness_range=[float(hardness[sel].min()), float(hardness[sel].max())],
                             skipped='tercile lacks a class: hardness separates classes completely'))
            continue
        a_eu, a_cw = auroc_fast(eu[known], eu[ood]), auroc_fast(cw[known], cw[ood])
        diffs = []
        for _ in range(n_boot):
            k = rng.choice(known, len(known), replace=True)
            o = rng.choice(ood, len(ood), replace=True)
            diffs.append(auroc_fast(cw[k], cw[o])-auroc_fast(eu[k], eu[o]))
        rows.append(dict(tercile=int(g), n=int(len(sel)), n_known=int(len(known)),
                         n_ood=int(len(ood)),
                         hardness_range=[float(hardness[sel].min()), float(hardness[sel].max())],
                         auroc_eu=a_eu, auroc_cw=a_cw, diff=a_cw-a_eu,
                         ci_lo=float(np.quantile(diffs, .025)),
                         ci_hi=float(np.quantile(diffs, .975))))
    return rows


def divergent_hardness_concentration(eu, cw, hardness, tol=.1):
    """Do rank-divergent rows concentrate in the hardest hardness tercile?"""
    eu, cw, hardness = np.asarray(eu, float), np.asarray(cw, float), np.asarray(hardness, float)
    rank = lambda v: (np.argsort(np.argsort(v)) + 1)/len(v)
    div = np.abs(rank(eu)-rank(cw)) > tol
    hard = hardness >= np.quantile(hardness, 2/3)
    return dict(n_divergent=int(div.sum()),
                frac_in_hard_tercile=float(div[hard].sum()/max(int(div.sum()), 1)),
                base_rate_hard_tercile=float(hard.mean()))


def eval_head(head, loader, lut, T=30, device='cpu'):
    """example.evaluate with features moved to device; EU and credal width on cpu."""
    head.eval()
    tot, n_lab, eu_all, cw_all = 0.0, 0, [], []
    with torch.no_grad():
        for b in loader:
            x, y = b['X'].to(device), lut[b['y']].to(device)
            if (y >= 0).any():                     # loss only for rows the head knows
                keep = y >= 0
                tot += float(F.cross_entropy(head(x[keep]), y[keep], reduction='sum'))
                n_lab += int(keep.sum())
            head.train()                           # dropout stochastic for the samples
            eu, cw = bald_and_credal_width(torch.stack([F.softmax(head(x), dim=1)
                                                        for _ in range(T)]))
            eu_all.append(eu.cpu())
            cw_all.append(cw)
    eu, cw = torch.cat(eu_all), torch.cat(cw_all)
    return (tot/n_lab if n_lab else float('nan')), eu, cw


def mix_path(x0, x1, t):
    """Pixel-space interpolation in [0,1]; t=0 is x0, t=1 is x1 (scalar or per-row)."""
    a, b = np.asarray(x0, np.float32), np.asarray(x1, np.float32)
    if a.shape != b.shape or a.ndim != 4:
        raise ValueError('Expected equal-shape NHWC images')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or a.min() < 0 or a.max() > 1 \
            or b.min() < 0 or b.max() > 1:
        raise ValueError('Images must be finite and in [0,1]')
    tt = np.asarray(t, np.float32)*np.ones(len(a), np.float32)[:, None, None, None]
    if np.any((tt < 0) | (tt > 1)):
        raise ValueError('t must lie in [0,1]')
    return (1-tt)*a + tt*b


def path_metrics(features, true_labels, head, reference, k=5, T=30):
    """Position vs agreement at fixed feature points: EU, class margins, kNN distance.

    EU is dropout BALD (T stochastic passes); margins come from the dropout-averaged
    predictive (top1-top2, and true-class minus runner-up where a label exists).
    dist_k / dist_norm are the minimum over known classes of the k-th NN distance,
    raw and divided by that class's calibrated radius tau, matching the budget
    report's near/far scale (cutoff 1.0).
    """
    device = next(head.parameters()).device
    x = torch.from_numpy(np.ascontiguousarray(features, np.float32)).to(device)
    head.train()                              # dropout stochastic, same passes for EU and margins
    with torch.no_grad():
        stacked = torch.stack([F.softmax(head(x), dim=1) for _ in range(T)])
        eu, credal_width = bald_and_credal_width(stacked)
        eu = eu.cpu().numpy()
    p = stacked.mean(0).cpu().numpy()
    top2 = np.sort(p, axis=1)[:, -2:]
    labels = np.asarray(true_labels, dtype=float)
    valid = np.isfinite(labels)
    margin_true = np.full(len(p), np.nan)
    if valid.any():
        pv, lv = p[valid], labels[valid].astype(int)
        others = pv.copy()
        others[np.arange(len(pv)), lv] = -np.inf
        margin_true[valid] = pv[np.arange(len(pv)), lv]-others.max(1)
    Z_by_class, tau, _ = reference
    dist_k = np.min([knn_kth_dist(Zc, features, k) for Zc in Z_by_class], axis=0)
    dist_norm = np.min([knn_kth_dist(Zc, features, k)/t for Zc, t in zip(Z_by_class, tau)], axis=0)
    return dict(eu=eu, margin_top12=top2[:, 1]-top2[:, 0], margin_true=margin_true,
                credal_width=credal_width.numpy(),
                dist_k=dist_k, dist_norm=dist_norm, pred_class=p.argmax(1))


def trace_paths(start_px, partner_px, t_grid, extract_batch, head, reference,
                true_labels, k=5, T=30):
    """Walk every start image toward its partner; one metrics row per step.

    true_labels are the START images' labels and stay attached along the path
    (they are NaN when the start itself is not from a known class).
    """
    t_grid = tuple(float(t) for t in t_grid)
    rows = []
    for t in t_grid:
        m = path_metrics(extract_batch(mix_path(start_px, partner_px, t)),
                         true_labels, head, reference, k=k, T=T)
        for i in range(len(start_px)):
            rows.append(dict(path_id=i, t=t, eu=float(m['eu'][i]),
                             margin_top12=float(m['margin_top12'][i]),
                             margin_true=float(m['margin_true'][i]),
                             credal_width=float(m['credal_width'][i]),
                             dist_k=float(m['dist_k'][i]), dist_norm=float(m['dist_norm'][i]),
                             pred_class=int(m['pred_class'][i])))
    return rows


def write_path_outputs(rows, out_dir):
    """Per-step CSV, per-(kind, t) summary JSON, and an EU/margin/distance figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out/'wrn50_paths.csv', index=False)
    summary = {kind: {float(t): {m: float(g[m].mean()) for m in
                                 ('eu', 'margin_top12', 'margin_true', 'credal_width',
                                  'dist_k', 'dist_norm')}
                      for t, g in part.groupby('t')}
               for kind, part in df.groupby('kind')}
    per_pair = {f'{kind}/{rel}': [{'path_id': int(r.path_id), 'eu': float(r.eu),
                                   'margin_top12': float(r.margin_top12),
                                   'credal_width': float(r.credal_width),
                                   'dist_norm': float(r.dist_norm)}
                                  for r in part[part.t == part.t.max()].itertuples()]
                for (kind, rel), part in df.groupby(['kind', 'partner_relation'])}
    (out/'wrn50_paths_summary.json').write_text(json.dumps(summary | dict(per_pair_final=per_pair),
                                                           indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for kind, part in df.groupby('kind'):
        means = part.groupby('t')[['eu', 'margin_top12', 'dist_norm']].mean()
        axes[0].plot(means.index, means['eu'], marker='o', label=kind)
        axes[1].plot(means.index, means['dist_norm'], marker='o', label=kind+' dist_norm')
        axes[1].plot(means.index, means['margin_top12'], marker='s', ls='--', alpha=.7,
                     label=kind+' margin')
    axes[0].set_ylabel('EU (nats)')
    axes[1].set_ylabel('dist_norm / margin')
    for ax in axes:
        ax.set_xlabel('path fraction t')
        ax.legend(fontsize=7)
    axes[1].axhline(1., color='gray', lw=.8)
    fig.tight_layout()
    fig.savefig(out/'wrn50_paths.png', dpi=150)
    plt.close(fig)
    return dict(summary=summary, rows=len(df))


def knn_kth_dist(Z_ref, Q, k, exclude_self=False):
    """Euclidean distance to the k-th nearest reference row, IndexFlatL2 semantics.

    exclude_self drops the self-match at distance 0, i.e. asks for k+1 neighbors.
    sklearn brute-force exact search gives the same numbers as faiss' IndexFlatL2
    without the libomp clash (torch + faiss segfault, OMP error #15, in this
    environment); this also matches the kNN metrics used in eu_blindspot.ipynb.
    """
    Z_ref, Q = prep(Z_ref), prep(Q)
    n_q = k+1 if exclude_self else k
    assert 1 <= k <= len(Z_ref) and len(Z_ref) >= n_q, f'reference has {len(Z_ref)} rows, need {n_q}'
    nn = NearestNeighbors(n_neighbors=n_q, algorithm='brute').fit(Z_ref)
    return nn.kneighbors(Q)[0][:, -1].astype(np.float64)


def class_distances_np(Z_by_class, Z, k, tau=None):
    Dm = np.stack([knn_kth_dist(Zc, Z, k) for Zc in Z_by_class], axis=1)
    return Dm if tau is None else Dm/tau[None, :]


def fit_knn_reference(world, q=.95, k=5):
    """Per-class kkc-train reference rows, k for calibration, and q-quantile radii tau."""
    X, y = world['X'], world['y']
    train_idx = world['idx']['train']
    train_kkc_rows = train_idx[np.isin(y[train_idx], world['kkc'])]
    Z_train = prep(X[train_kkc_rows])
    lut = {c: i for i, c in enumerate(world['kkc'])}
    y_train_idx = np.array([lut[c] for c in y[train_kkc_rows]], dtype=np.int64)
    Z_by_class = [Z_train[y_train_idx == c] for c in range(len(world['kkc']))]
    assert all(len(Zc) > 0 for Zc in Z_by_class), 'empty kkc class after split'
    k_eff = min(k, max(1, min(len(Zc) for Zc in Z_by_class)-1))
    tau = np.array([np.quantile(knn_kth_dist(Zc, Zc, k_eff, exclude_self=True), q)
                    for Zc in Z_by_class])
    return Z_by_class, tau, k_eff


def tag_and_report_budget_np(world, k=5, q=.95, save_path='results/oss_metrics/budget_info_wrn50.json'):
    """Same fields and output as example.tag_and_report_budget, with exact numpy kNN.

    Writes the near/far budget JSON (plus this module's mode) and returns it.
    """
    X, y = world['X'], world['y']
    kkc, kuc = world['kkc'], set(world['kuc'])
    uuc_val, uuc_test = set(world['uuc_val']), set(world['uuc_test'])
    novel_classes = sorted(list(kuc | set(world['uuc'])))
    Z_by_class, tau, k_eff = fit_knn_reference(world, q=q, k=k)
    novel_info, med_norm_dists = [], []
    for c in novel_classes:
        Z_c = prep(X[y == c])
        d_norm = class_distances_np(Z_by_class, Z_c, k_eff, tau).min(axis=1)
        d_raw = class_distances_np(Z_by_class, Z_c, k_eff, None).min(axis=1)
        grp = 'kuc' if c in kuc else 'uuc'
        split = 'train/val/test' if c in kuc else ('val' if c in uuc_val else 'test')
        med_norm = float(np.median(d_norm))
        med_norm_dists.append(med_norm)
        novel_info.append({'class': int(c), 'group': grp, 'split': split,
                           'n_samples': int(len(Z_c)),
                           'mean_norm_dist': float(np.mean(d_norm)), 'median_norm_dist': med_norm,
                           'mean_raw_dist': float(np.mean(d_raw)),
                           'median_raw_dist': float(np.median(d_raw))})
    cutoff = 1.0 if any(item['median_norm_dist'] <= 1.0 for item in novel_info) \
        else float(np.median(med_norm_dists))
    for item in novel_info:
        item['tag'] = 'near' if item['median_norm_dist'] <= cutoff else 'far'
    print('\n'+'='*68)
    print('OSSIM CLASS INVENTORY & BUDGET SUMMARY')
    print('='*68)
    print(f"Counts: n_kkc = {world['n_kkc']}, n_kuc = {world['n_kuc']}, n_uuc = {world['n_uuc']}")
    print(f"  - KKC classes ({len(kkc)}): {kkc}")
    print(f"  - KUC classes ({len(world['kuc'])}): {world['kuc']}")
    print(f"  - UUC classes ({len(world['uuc'])}): {world['uuc']}")
    print(f"      in val  ({len(world['uuc_val'])}): {world['uuc_val']}")
    print(f"      in test ({len(world['uuc_test'])}): {world['uuc_test']}")
    print('-'*68)
    print(f"{'Class':>6s}  {'Group':>5s}  {'Split':>14s}  {'Norm Dist (mean)':>16s}  {'Raw Dist':>10s}  {'Tag':>5s}")
    print('-'*68)
    for it in novel_info:
        print(f"{it['class']:6d}  {it['group']:>5s}  {it['split']:>14s}  "
              f"{it['mean_norm_dist']:16.3f}  {it['mean_raw_dist']:10.3f}  {it['tag']:>5s}")
    print('='*68+'\n')
    out_dict = {
        'n_kkc': world['n_kkc'], 'n_kuc': world['n_kuc'], 'n_uuc': world['n_uuc'],
        'n_uuc_val': len(world['uuc_val']), 'n_uuc_test': len(world['uuc_test']),
        'classes': dict(kkc=kkc, kuc=world['kuc'], uuc=world['uuc'],
                        uuc_val=world['uuc_val'], uuc_test=world['uuc_test']),
        'novel_classes': novel_info,
        'tau': [float(t) for t in tau], 'mode': world.get('mode'),
    }
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out_dict, indent=2))
        print(f'Budget and novel class info saved to: {path}')
    return out_dict


@dataclass
class Config:
    mode: str = 'oss'                    # 'oss' keeps the KKC/KUC/UUC budget; 'plain' fixes known classes
    model: str = 'wide_resnet50_2'
    id_dataset: str = 'cifar10'
    ood_dataset: str | None = None       # e.g. 'mnist': second-dataset OOD cache for plain mode
    id_cache: str | None = None          # explicit feature cache; extracted when absent
    ood_cache: str | None = None
    known: tuple | None = None           # plain mode: explicit ID class ids
    n_known: int | None = None           # plain mode: seeded pick size when --known absent
    kuc: int | None = None               # oss mode budget knobs (None = example.py defaults)
    uuc_val: int | None = None
    uuc_test: int | None = None
    seed: int = 0
    epochs: int = 40
    lr: float = 1e-3
    head: str = 'mlp'
    device: str = 'cpu'                  # compute device for extraction, head training and EU eval
    n_per_class: int | None = None       # balanced extraction subsample (speed knob)
    n_paths: int = 16                    # paths mode: number of start/partner image pairs
    t_grid: tuple = (0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.)
    path_k: int = 5                      # paths mode: k for the kNN distance to kkc-train rows
    T: int = 30                          # paths mode: stochastic dropout passes for EU/margins
    stability: bool = False              # also rerun eval with fresh dropout draws at T=10/30/100
    data_root: str = 'cache/cifar10'     # where CIFAR-10 images live (or should download)
    out_dir: str = 'results/oss_metrics'

    def validate(self):
        if self.mode not in ('oss', 'plain'):
            raise ValueError(self.mode)
        if self.known is not None and self.n_known is not None:
            raise ValueError('Give known or n_known, not both')
        if self.mode == 'oss' and self.known is not None:
            raise ValueError('known is a plain-mode knob')


def build_world(config):
    """Extract (or load cached) features and build the mode's world dict."""
    if config.id_cache:
        id_path = Path(config.id_cache)
    else:
        extract_features(config.model, config.id_dataset, config.n_per_class,
                         device=config.device, root=config.data_root)
        id_path = features_path(config.model, config.id_dataset)
    if config.mode == 'oss':
        world = make_oss(str(id_path), kuc=config.kuc, uuc_val=config.uuc_val,
                         uuc_test=config.uuc_test, seed=config.seed)
    else:
        if config.ood_cache:
            ood_path: Path | None = Path(config.ood_cache)
        elif config.ood_dataset:
            extract_features(config.model, config.ood_dataset, config.n_per_class,
                             device=config.device, root=config.data_root)
            ood_path = features_path(config.model, config.ood_dataset)
        else:
            ood_path = None
        world = make_plain_ood(id_path, ood_path=ood_path, known=config.known,
                               n_known=config.n_known, seed=config.seed)
    return world


def train_world_head(world, config):
    """Train the head on the world's kkc train rows; returns (head, lut)."""
    X, y = world['X'], world['y']
    lut = torch.full((int(y.max())+1,), -1, dtype=torch.long)
    for k, c in enumerate(world['kkc']):
        lut[c] = k
    train_ld = make_loader(X, y, group_idx(world, 'train', 'kkc'), batch_size=256,
                           shuffle=True, seed=config.seed)
    head = build_head(config.head, d_in=X.shape[1], k=len(world['kkc'])).to(config.device)
    losses = train_head(head, train_ld, lut, epochs=config.epochs, lr=config.lr,
                        seed=config.seed, device=config.device)
    print('loss per 5 epochs:', [round(l, 4) for l in losses[::5]], '->', round(losses[-1], 4))
    return head, lut, losses[-1]


def run_paths(config):
    """EU vs position along controlled pixel paths from known images.

    Each start image is a held-out known test row; partners are MNIST images
    (cifar_to_mnist) or other CIFAR rows (cifar_to_cifar control). At every t the
    mixed pixels are re-extracted through the backbone, then EU, class margins and
    the tau-normalized kNN distance to kkc-train features are recorded. Low EU far
    from known support flags confident extrapolation rather than familiarity.
    """
    config.validate()
    out = Path(config.out_dir); out.mkdir(parents=True, exist_ok=True)
    world = build_world(config)
    head, lut, _ = train_world_head(world, config)
    reference = fit_knn_reference(world, q=.95, k=config.path_k)

    import timm
    ds_id, _ = load_test_dataset(config.id_dataset, config.data_root)
    ds_ood, _ = load_test_dataset(config.ood_dataset)
    model = timm.create_model(config.model, pretrained=True, num_classes=0).to(config.device).eval()
    data_cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(data_cfg['mean'], device=config.device)[None, :, None, None]
    std = torch.tensor(data_cfg['std'], device=config.device)[None, :, None, None]

    def extract_batch(pixels):
        x = torch.from_numpy(np.ascontiguousarray(pixels.transpose(0, 3, 1, 2))).to(config.device)
        x = F.interpolate(x, size=data_cfg['input_size'][-2:], mode='bicubic', align_corners=False)
        with torch.no_grad():
            return model((x-mean)/std).cpu().numpy()

    starts = group_idx(world, 'test', 'kkc')[:config.n_paths]
    start_px = dataset_pixels(ds_id, starts)
    start_hw = start_px.shape[1:3]
    start_labels = world['y'][starts]

    def partner_pixels(j):
        img = ds_ood[j][0]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if img.size != (start_hw[1], start_hw[0]):
            img = img.resize((start_hw[1], start_hw[0]))
        return np.asarray(img, np.float32)/255.
    ood_px = np.stack([partner_pixels(j) for j in range(config.n_paths)])
    candidates = group_idx(world, 'test', 'kkc')
    control_rows, control_relation = choose_control_partners(
        starts, world['y'][candidates], n_same=config.n_paths//2, seed=config.seed)
    control_rows = candidates[control_rows]
    control_px = dataset_pixels(ds_id, control_rows)
    rows = []
    for kind, partner_px, relations in (('cifar_to_mnist', ood_px, ['ood']*config.n_paths),
                                        ('cifar_to_cifar', control_px, control_relation)):
        print(f'paths: {kind} ({", ".join(sorted(set(relations)))})', flush=True)
        rows.extend(dict(kind=kind, partner_relation=relations[r['path_id']], **r)
                    for r in trace_paths(start_px, partner_px, config.t_grid, extract_batch,
                                         head, reference, true_labels=start_labels,
                                         k=config.path_k, T=config.T))
    result = write_path_outputs(rows, out)
    for kind, per_t in result['summary'].items():
        line = '  '.join(f"t={t:g} eu={v['eu']:.3f} d={v['dist_norm']:.2f} m={v['margin_top12']:.3f}"
                         for t, v in sorted(per_t.items()))
        print(f'{kind}:\n  {line}')
    return result


def run(config):
    config.validate()
    out = Path(config.out_dir); out.mkdir(parents=True, exist_ok=True)
    world = build_world(config)
    tag_and_report_budget_np(world, k=5, q=.95, save_path=str(out/'budget_info_wrn50.json'))

    X, y = world['X'], world['y']
    classes = world['kkc']
    head, lut, final_loss = train_world_head(world, config)
    groups = [g for g in ('kkc', 'kuc', 'uuc') if len(world[g]) > 0]
    val = {g: make_loader(X, y, group_idx(world, 'val', g), batch_size=512) for g in groups}
    test = {g: make_loader(X, y, group_idx(world, 'test', g), batch_size=512) for g in groups}

    table = {}
    print(f"\n{'split':6s} {'group':5s} {'loss':>7s} {'mean EU (nats)':>15s} "
          f"{'mean credal w':>13s} {'rows':>6s}")
    for split, parts in (('val', val), ('test', test)):
        for g, ld in parts.items():
            if not len(ld.dataset):
                continue
            loss, eu, cw = eval_head(head, ld, lut, device=config.device)
            table[(split, g)] = (eu, cw)
            print(f"{split:6s} {g:5s} {loss:7.4f} {float(eu.mean()):15.4f} "
                  f"{float(cw.mean()):13.4f} {len(eu):6d}")

    aurocs = {}
    for split in ('val', 'test'):
        if (split, 'kkc') not in table:
            continue
        for novel_grp in ('kuc', 'uuc'):
            if (split, novel_grp) in table and len(table[(split, novel_grp)][0]) > 0:
                for score_ix, suffix, label in ((0, '', 'EU'), (1, '_cw', 'credal width')):
                    value = auroc_fast(table[(split, 'kkc')][score_ix].numpy(),
                                       table[(split, novel_grp)][score_ix].numpy())
                    aurocs[f'{split}_kkc_vs_{novel_grp}{suffix}'] = value
                    print(f'{split}: {label} AUROC kkc vs {novel_grp} = {value:.3f}')
    if ('test', 'kkc') in table and ('test', 'uuc') in table and len(table[('test', 'uuc')][0]):
        test_eu = torch.cat([table[('test', 'kkc')][0], table[('test', 'uuc')][0]]).numpy()
        test_cw = torch.cat([table[('test', 'kkc')][1], table[('test', 'uuc')][1]]).numpy()
        test_ood = np.r_[np.zeros(len(table[('test', 'kkc')][0])),
                         np.ones(len(table[('test', 'uuc')][0]))].astype(bool)
        disagreement = rank_disagreement(test_eu, test_cw)
        result_rank = dict(frac=disagreement['frac'], max_gap=disagreement['max_gap'],
                           n_divergent=int(disagreement['frac']*disagreement['n']),
                           worst=list(int(i) for i in disagreement['worst'][:5]))
        # hardness = frozen-feature kNN distance, outside the scored heads
        X_test_all = np.concatenate([X[group_idx(world, 'test', 'kkc')],
                                     X[group_idx(world, 'test', 'uuc')]])
        Z_by_class, tau, k_eff = fit_knn_reference(world, q=.95, k=config.path_k)
        hardness = np.min([knn_kth_dist(Zc, X_test_all, k_eff)/t
                           for Zc, t in zip(Z_by_class, tau)], axis=0)
        terciles = stratified_tercile_auroc(test_eu, test_cw, hardness, test_ood, seed=config.seed)
        concentration = divergent_hardness_concentration(test_eu, test_cw, hardness)
        for r in terciles:
            if 'skipped' in r:
                print(f"tercile {r['tercile']}: skipped ({r['skipped']})")
                continue
            print(f"tercile {r['tercile']} (hardness {r['hardness_range'][0]:.2f}"
                  f"-{r['hardness_range'][1]:.2f}): EU {r['auroc_eu']:.3f} width {r['auroc_cw']:.3f} "
                  f"diff {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
        print(f"divergent rows in hardest tercile: {concentration['frac_in_hard_tercile']:.2f} "
              f"(base rate {concentration['base_rate_hard_tercile']:.2f})")
        (out/'wrn50_terciles.json').write_text(json.dumps(
            dict(terciles=terciles, concentration=concentration), indent=2))
        print(f"test rows where EU and credal-width percentile ranks differ >10%: "
              f"{result_rank['n_divergent']}/{disagreement['n']} "
              f"(max gap {result_rank['max_gap']:.2f})")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        for flag, name, marker in ((False, 'known', 'o'), (True, 'OOD', '^')):
            ax.scatter(test_eu[test_ood == flag], test_cw[test_ood == flag], s=8, alpha=.5,
                       label=name, marker=marker)
        ax.set_xlabel('EU (nats)'); ax.set_ylabel('mean credal width')
        ax.legend(); fig.tight_layout()
        fig.savefig(out/'wrn50_scatter.png', dpi=150); plt.close(fig)
        # probly-native credal-set views: envelope of T sampled predictives per example
        n_known = len(table[('test', 'kkc')][0])
        def stack_for(row_ix):
            x_row = torch.from_numpy(X_test_all[int(row_ix)][None].astype(np.float32)).to(config.device)
            head.train()
            with torch.no_grad():
                return torch.stack([F.softmax(head(x_row), dim=1)
                                    for _ in range(config.T)]).cpu().numpy()
        i_low = int(np.argmin(table[('test', 'kkc')][1].numpy()))
        i_div = next((int(i) for i in disagreement['worst'] if int(i) < n_known), i_low)
        i_ood = int(np.argmax(table[('test', 'uuc')][1].numpy())) + n_known
        picks = {'known_min_width': stack_for(i_low), 'known_rank_divergent': stack_for(i_div),
                 'ood_max_width': stack_for(i_ood)}
        plot_credal_examples(picks, out/'wrn50_credal_set.png',
                             class_labels=[str(c) for c in classes])
        print('credal-set example plot: wrn50_credal_set.png '
              '(known_min_width, known_rank_divergent, ood_max_width)')
    else:
        result_rank = None
    if config.stability and ('test', 'kkc') in table and ('test', 'uuc') in table:
        stability = sampling_stability(
            head, np.concatenate([X[group_idx(world, 'test', 'kkc')],
                                  X[group_idx(world, 'test', 'uuc')]]),
            is_ood=np.r_[np.zeros(len(group_idx(world, 'test', 'kkc'))),
                         np.ones(len(group_idx(world, 'test', 'uuc')))].astype(bool),
            device=config.device)
        (out/'wrn50_stability.json').write_text(json.dumps(stability, indent=2))
        print('dropout-sampling stability (fixed head+rows):',
              json.dumps({t: {k: round(v, 4) for k, v in s.items()} for t, s in stability.items()}))
    else:
        stability = None

    result = dict(mode=config.mode, model=config.model, device=config.device,
                  world_keys=sorted(world), auroc=dict(
                      kkc_vs_kuc=aurocs.get('test_kkc_vs_kuc'),
                      kkc_vs_uuc=aurocs.get('test_kkc_vs_uuc'),
                      kkc_vs_kuc_cw=aurocs.get('test_kkc_vs_kuc_cw'),
                      kkc_vs_uuc_cw=aurocs.get('test_kkc_vs_uuc_cw'),
                      **{k: v for k, v in aurocs.items() if k.startswith('val_')}),
                  mean_eu={f'{split}_{g}': float(eu.mean())
                           for (split, g), (eu, cw) in table.items()},
                  mean_credal_width={f'{split}_{g}': float(cw.mean())
                                     for (split, g), (eu, cw) in table.items()},
                  rank_disagreement=result_rank, stability=stability,
                  terciles=terciles if ('test', 'kkc') in table and ('test', 'uuc') in table
                  and len(table[('test', 'uuc')][0]) else None,
                  final_loss=final_loss)
    (out/'wrn50_eval.json').write_text(json.dumps(
        result | dict(id_dataset=config.id_dataset, ood_dataset=config.ood_dataset,
                      epochs=config.epochs, seed=config.seed,
                      n_known_classes=len(classes), all_aurocs=aurocs), indent=2))
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('oss', 'plain'), default='oss')
    p.add_argument('--model', default='wide_resnet50_2', help='timm backbone')
    p.add_argument('--id-dataset', default='cifar10', help='timm extraction source for ID features')
    p.add_argument('--ood-dataset', default='mnist', help='plain mode: second dataset as OOD')
    p.add_argument('--id-cache'), p.add_argument('--ood-cache')
    p.add_argument('--known', type=int, nargs='+', default=None, help='plain mode: explicit ID class ids')
    p.add_argument('--n-known', type=int, default=None)
    p.add_argument('--kuc', type=int, default=None)
    p.add_argument('--uuc-val', type=int, default=None)
    p.add_argument('--uuc-test', type=int, default=None)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--head', choices=('mlp', 'etf'), default='mlp')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu', help='backbone extraction device (mps/cuda/cpu)')
    p.add_argument('--paths', action='store_true',
                   help='trace EU/margin/distance along CIFAR->MNIST and CIFAR->CIFAR pixel paths')
    p.add_argument('--n-paths', type=int, default=16)
    p.add_argument('--stability', action='store_true',
                   help='rerun eval with fresh dropout draws at T=10/30/100 (fixed head+rows)')
    p.add_argument('--n-per-class', type=int, default=None)
    p.add_argument('--data-root', default='cache/cifar10', help='CIFAR-10 image folder or download target')
    p.add_argument('--out-dir', default='results/oss_metrics')
    args = p.parse_args(argv)
    return args.paths, Config(**{k: v for k, v in vars(args).items() if k != 'paths'})


def main():
    go_paths, config = parse_args()
    run_paths(config) if go_paths else run(config)


if __name__ == '__main__':
    main()
