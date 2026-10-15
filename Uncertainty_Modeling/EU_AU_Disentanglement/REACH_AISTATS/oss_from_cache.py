# oss_from_cache.py -- part 1: load, split, inspect, recover row indices
import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset
from pytorch_ood.dataset.ossim import DynamicOSS


def load_cached(path):
    d = np.load(path, allow_pickle=False)
    X = d["X"].astype(np.float32)          # float16 -> float32 before any PCA/Gram
    y = d["y"].astype(np.int64)
    assert X.ndim == 2 and len(X) == len(y), (X.shape, y.shape)
    print(f"{path}: X {X.shape}, classes {np.unique(y).tolist()}, "
          f"mean feature norm {np.linalg.norm(X, axis=1).mean():.3f}")
    return X, y


def make_oss(path, kuc=0, uuc_val=2, uuc_test=5,
             train_size=0.7, val_size=0.2, test_size=0.1, seed=0):
    X, y = load_cached(path)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    oss = DynamicOSS(ds, train_size=train_size, val_size=val_size,
                     test_size=test_size, kuc=kuc,
                     uuc_val=uuc_val, uuc_test=uuc_test, seed=seed)

    kkc, kuc_, uuc = set(oss.kkc), set(oss.kuc), set(oss.uuc)
    print(f"seed={seed}\n  kkc={sorted(kkc)}\n  kuc={sorted(kuc_)}\n  uuc={sorted(uuc)}")

    # --- sanity checks: fail loudly, do not assume ---
    assert kkc.isdisjoint(uuc), "known and unknown classes overlap"
    assert kkc.isdisjoint(kuc_), "kkc and kuc overlap"
    # TODO: decide whether uuc_val and uuc_test should be disjoint and assert it

    # --- recover ROW indices (valid only because the dataset is a TensorDataset) ---
    parts = {
        "train": oss.train_dataset(in_dist=True, out_dist=True),
        "val":   oss.val_dataset(in_dist=True, out_dist=True),
        "test":  oss.test_dataset(in_dist=True, out_dist=True),
    }
    idx = {}
    for name, sub in parts.items():
        assert isinstance(sub, Subset), f"{name}: expected Subset, got {type(sub)}"
        idx[name] = np.asarray(sub.indices)
        cls = np.unique(y[idx[name]])
        print(f"  {name}: {len(idx[name])} rows, classes {cls.tolist()}")

    # no row should appear in two splits
    assert not (set(idx["train"]) & set(idx["val"])), "train/val leak"
    assert not (set(idx["train"]) & set(idx["test"])), "train/test leak"
    assert not (set(idx["val"]) & set(idx["test"])), "val/test leak"

    return dict(X=X, y=y, oss=oss, idx=idx, kkc=kkc, kuc=kuc_, uuc=uuc)


class _DictDataset(torch.utils.data.Dataset):
    """Rows as plain dicts so default_collate batches come out as key -> value dicts."""

    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors['X'])

    def __getitem__(self, i):
        return {k: t[i] for k, t in self.tensors.items()}


def make_loader(X, y, indices=None, batch_size=256, shuffle=False, seed=0, **extra):
    """Simple DataLoader whose batches are plain dicts: keys -> values.

    Every batch has 'X' (b, d) float32, 'y' (b,) int64, and 'index' (b,) holding each
    row's id in the ORIGINAL arrays, so any batch value is traceable to its source row.
    indices: row ids to load (e.g. one of make_oss's splits); default all rows.
    extra:   per-row arrays included under their keyword name (group tags, cells, ...);
             float arrays keep float32, bool/int arrays become int64.
    """
    idx = np.arange(len(X)) if indices is None else np.asarray(indices)
    tensors = {'X': torch.from_numpy(np.asarray(X, dtype=np.float32)[idx]),
               'y': torch.from_numpy(np.asarray(y, dtype=np.int64)[idx]),
               'index': torch.from_numpy(np.asarray(idx, dtype=np.int64))}
    for name, arr in extra.items():
        arr = np.asarray(arr)[idx]
        if arr.ndim != 1 or len(arr) != len(idx):
            raise ValueError(f'extra[{name!r}] must be a 1-d per-row array')
        dtype = np.int64 if arr.dtype == bool or np.issubdtype(arr.dtype, np.integer) else np.float32
        tensors[name] = torch.from_numpy(arr.astype(dtype))
    return torch.utils.data.DataLoader(_DictDataset(tensors), batch_size=batch_size,
                                       shuffle=shuffle and len(idx) > 0,   # RandomSampler rejects empty
                                       generator=torch.Generator().manual_seed(seed) if shuffle else None)

GROUP_CODES = {'kkc': 0, 'kuc': 1, 'uuc': 2}


def group_loaders(path='cache/features/cifar10_test.npz', batch_size=256, shuffle=True,
                  seed=0, splits=('train', 'val', 'test'), **oss_kwargs):
    """OSSIM class-group loaders: out[split][group] for group in kkc/kuc/uuc.

    Groups partition every row exactly once (kkc known-known, kuc known-unknown,
    uuc unknown-unknown); each batch is a make_loader dict batch with a 'group' key
    carrying the numeric code in GROUP_CODES. With the default class budget
    (kuc=0, uuc_val=2, uuc_test=5 on 10 classes) the kuc loaders have no rows.
    """
    world = make_oss(path, seed=seed, **oss_kwargs)
    code = np.full(len(world['y']), -1, dtype=np.int64)
    for name, classes_ in (('kkc', world['kkc']), ('kuc', world['kuc']), ('uuc', world['uuc'])):
        code[np.isin(world['y'], list(classes_))] = GROUP_CODES[name]
    assert (code >= 0).all(), 'every row must belong to exactly one OSSIM group'
    out = {}
    for split in splits:
        out[split] = {}
        split_code = code[world['idx'][split]]
        for name in GROUP_CODES:
            sel = np.asarray(world['idx'][split])[split_code == GROUP_CODES[name]]
            out[split][name] = make_loader(world['X'], world['y'], indices=sel,
                                           batch_size=batch_size, shuffle=shuffle, seed=seed,
                                           group=np.full(len(world['y']), GROUP_CODES[name]))
    return out

# TODO (yours): build_log(), group tags (centre / boundary / novel), PCA fit on labelled rows only, saving masks to disk
