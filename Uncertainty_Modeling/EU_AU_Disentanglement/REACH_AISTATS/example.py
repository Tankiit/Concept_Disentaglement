import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Subset, DataLoader
from pytorch_ood.dataset.ossim import DynamicOSS

from probly.method import dropout
from probly.quantification import quantify
from probly.evaluation.ood import evaluate_ood


def load_cached(path):
    d = np.load(path, allow_pickle=False)
    X = d["X"].astype(np.float32)          # float16 -> float32 before any PCA/Gram
    y = d["y"].astype(np.int64)
    assert X.ndim == 2 and len(X) == len(y), (X.shape, y.shape)
    print(f"{path}: X {X.shape}, classes {np.unique(y).tolist()}, "
          f"mean feature norm {np.linalg.norm(X, axis=1).mean():.3f}")
    return X, y


def extract_features_timm(model_name="resnet18", n_per_class=None, root="cache/cifar10",
                          batch_size=256, device="cpu"):
    """Frozen timm backbone -> pooled features for the CIFAR-10 test split.

    The local counterpart of real/modal_extract.py's CLIP extraction: raw images
    in, an (N, d) float32 X + int64 y NPZ out, in the same cache format make_oss
    consumes. Weights come from the HF hub on first use; the NPZ is written once
    and reused, so later runs skip the backbone entirely.
    """
    import timm
    from pathlib import Path
    from torchvision.datasets import CIFAR10

    out = Path("cache/features") / f"timm_{model_name}_c10test.npz"
    if out.exists():
        return load_cached(out)
    ds = CIFAR10(root, train=False, download=True)
    take, counts = [], {}
    for i, t in enumerate(ds.targets):
        if n_per_class is None or counts.get(t, 0) < n_per_class:
            take.append(i)
            counts[t] = counts.get(t, 0) + 1
    take = np.asarray(take)

    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    tfm = timm.data.create_transform(**timm.data.resolve_model_data_config(model))
    feats, labels = [], []
    with torch.no_grad():
        for i in range(0, len(take), batch_size):
            chunk = take[i:i + batch_size]
            x = torch.stack([tfm(ds[int(j)][0]) for j in chunk]).to(device)
            feats.append(model(x).cpu().numpy())
            labels.extend(int(ds.targets[int(j)]) for j in chunk)
    X, y = np.vstack(feats).astype(np.float32), np.asarray(labels, dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, y=y)
    print(f"timm {model_name}: X {X.shape}, classes {np.unique(y).tolist()}, "
          f"mean feature norm {np.linalg.norm(X, axis=1).mean():.3f} -> {out}")
    return X, y


def make_oss(path, kuc=0, uuc_val=2, uuc_test=5,
             train_size=0.7, val_size=0.2, test_size=0.1, seed=0):
    X, y = load_cached(path)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    oss = DynamicOSS(ds, train_size=train_size, val_size=val_size,
                     test_size=test_size, kuc=kuc,
                     uuc_val=uuc_val, uuc_test=uuc_test, seed=seed)

    kkc, kuc_, uuc = set(oss.kkc), set(oss.kuc), set(oss.uuc)
    parts = {"train": oss.train_dataset(), "val": oss.val_dataset(), "test": oss.test_dataset()}
    idx = {k: np.asarray(v.indices) for k, v in parts.items()}   # row ids into X/y
    return dict(X=X, y=y, idx=idx, kkc=sorted(kkc), kuc=sorted(kuc_), uuc=sorted(uuc))


class DictRows(torch.utils.data.Dataset):
    """Rows as dicts so batches come out as key -> value."""

    def __init__(self, X, y, idx):
        self.X, self.y, self.idx = X, y, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return {"X": self.X[j], "y": self.y[j], "index": j}


def make_loader(X, y, indices, batch_size=256, shuffle=False, seed=0):
    g = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(DictRows(X, y, indices), batch_size=batch_size,
                      shuffle=shuffle and len(indices) > 0, generator=g)


# --- training: measure loss + EU (BALD under MC-dropout) ------------------------
# EU = H(mean_t p_t) - mean_t H(p_t): high where the model has seen no data
# (expected on uuc rows), low on seen support. Dropout-EU is an approximation,
# but it is the one that responds to data, which is the point here.

class Head(nn.Module):
    def __init__(self, d=512, hidden=128, k=3, p=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Dropout(p), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


def train(head, loader, lut, epochs=40, lr=1e-3, seed=0):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    torch.manual_seed(seed)
    losses = []
    for epoch in range(epochs):
        head.train()
        total, n = 0.0, 0
        for b in loader:
            y = lut[b["y"]]
            opt.zero_grad()
            loss = F.cross_entropy(head(b["X"]), y)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        losses.append(total / n)
    return losses


def eu_bald(head, X, T=30):
    head.train()                                   # keep dropout stochastic
    with torch.no_grad():
        p = torch.stack([F.softmax(head(X), dim=1) for _ in range(T)])
    mean = p.mean(0)
    return -(mean * mean.clamp_min(1e-12).log()).sum(-1) \
        + (p * p.clamp_min(1e-12).log()).sum(-1).mean(0)


def evaluate(head, loader, lut, T=30):
    head.eval()
    tot, n_lab, eu = 0.0, 0, []
    with torch.no_grad():
        for b in loader:
            y = lut[b["y"]]
            if (y >= 0).any():                     # loss only for rows the head knows
                keep = y >= 0
                tot += float(F.cross_entropy(head(b["X"][keep]), y[keep], reduction="sum"))
                n_lab += int(keep.sum())
            eu.append(eu_bald(head, b["X"], T))    # EU needs no labels at all
    eu = torch.cat(eu)
    return (tot / n_lab if n_lab else float("nan")), eu


def group_idx(world, split, group):
    """Rows of one split whose labels are in the named class group."""
    sel = world["idx"][split]
    return sel[np.isin(world["y"][sel], list(world[group]))]


def auroc(eu_known, eu_unknown):
    win = sum(int((o > k).sum()) for k in eu_known for o in eu_unknown)
    tie = sum(int((o == k).sum()) for k in eu_known for o in eu_unknown)
    return (win + 0.5 * tie) / (len(eu_known) * len(eu_unknown))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="loss + EU on OSSIM groups over cached or timm features")
    ap.add_argument("--source", choices=("cache", "timm"), default="cache",
                    help="cached CLIP features, or extract fresh with a timm backbone")
    ap.add_argument("--model", default="resnet18", help="timm model name (with --source timm)")
    ap.add_argument("--n-per-class", type=int, default=None,
                    help="balanced per-class subsample before extraction (speed knob)")
    args = ap.parse_args()

    if args.source == "timm":
        extract_features_timm(args.model, args.n_per_class)
        path = f"cache/features/timm_{args.model}_c10test.npz"
    else:
        path = "cache/features/cifar10_test.npz"

    world = make_oss(path, kuc=0, uuc_val=2, uuc_test=5, seed=0)
    X, y, idx = world["X"], world["y"], world["idx"]
    classes = world["kkc"]
    lut = torch.full((1024,), -1, dtype=torch.long)
    for k, c in enumerate(classes):
        lut[c] = k

    train_ld = make_loader(X, y, group_idx(world, "train", "kkc"), batch_size=256, shuffle=True, seed=0)
    val = {g: make_loader(X, y, group_idx(world, "val", g), batch_size=512) for g in ("kkc", "uuc")}
    test = {g: make_loader(X, y, group_idx(world, "test", g), batch_size=512) for g in ("kkc", "uuc")}

    head = Head(k=len(classes))
    losses = train(head, train_ld, lut, epochs=40, seed=0)
    print("loss per 5 epochs:", [round(l, 4) for l in losses[::5]], "->", round(losses[-1], 4))

    print(f"\n{'split':6s} {'group':5s} {'loss':>7s} {'mean EU (nats)':>15s} {'rows':>6s}")
    table = {}
    for split, parts in (("val", val), ("test", test)):
        for g, ld in parts.items():
            if not len(ld.dataset):
                continue
            loss, eu = evaluate(head, ld, lut)
            table[(split, g)] = eu
            print(f"{split:6s} {g:5s} {loss:7.4f} {float(eu.mean()):15.4f} {len(eu):6d}")

    for split in ("val", "test"):
        print(f"{split}: EU AUROC kkc vs uuc = {auroc(table[(split, 'kkc')], table[(split, 'uuc')]):.3f}")
