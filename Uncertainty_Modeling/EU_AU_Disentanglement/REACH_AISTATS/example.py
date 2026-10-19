import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Subset, DataLoader
from pytorch_ood.dataset.ossim import DynamicOSS
from tqdm import tqdm

from probly.method import dropout
from probly.quantification import quantify
from probly.evaluation.ood import evaluate_ood

from metrics import *

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import faiss

def prep(Z):
    Z = np.ascontiguousarray(Z, dtype=np.float32)   # faiss needs float32, C-contiguous
    return Z

def fit_class_indices(Z_lab, y_idx, K):
    d = Z_lab.shape[1]
    indices = []
    for c in range(K):
        Zc = prep(Z_lab[y_idx == c])
        assert len(Zc) > 0, f"class {c} empty"
        index = faiss.IndexFlatL2(d)        # exact search; TODO: keep flat, see hint 1
        index.add(Zc)
        indices.append(index)
    return indices

def kth_dist(index, Q, k, exclude_self=False):
    n_q = k + 1 if exclude_self else k
    assert index.ntotal >= n_q, "class has fewer rows than k (+1)"
    D, I = index.search(prep(Q), n_q)       # D is SQUARED L2
    assert (I >= 0).all(), "missing neighbours returned"
    return np.sqrt(D[:, -1])                # TODO: keep sqrt; tau must use the same scale
    # TODO: duplicates give distance 0 for the self match AND for the copy; check on a toy case

def class_distances(indices, Z, k, tau=None):
    Dm = np.stack([kth_dist(ix, Z, k) for ix in indices], axis=1)
    return Dm if tau is None else Dm / tau[None, :]

def calibrate(indices, Z_lab, y_idx, k, q):
    tau = np.zeros(len(indices))
    for c, ix in enumerate(indices):
        Zc = prep(Z_lab[y_idx == c])
        tau[c] = np.quantile(kth_dist(ix, Zc, k, exclude_self=True), q)   # TODO: q
    return tau


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


def choose_budget(n_classes, kuc=None, uuc_val=None, uuc_test=None):
    """Select a valid budget with enough known classes and non-empty kuc.

    Adapts to class inventory:
      - Inventory >= 100 (e.g. CIFAR-100): n_kkc = 50, kuc = 10, uuc = 20 (uuc_val=10, uuc_test=10)
      - Inventory >= 35:                  n_kkc = 20, kuc = 5,  uuc = 10 (uuc_val=5,  uuc_test=5)
      - Inventory >= 10 (e.g. CIFAR-10):  n_kkc = 5,  kuc = 2,  uuc = 3  (uuc_val=1,  uuc_test=2)
    """
    if kuc is not None and uuc_val is not None and uuc_test is not None:
        n_uuc = uuc_val + uuc_test
        n_kkc = n_classes - (kuc + n_uuc)
        assert n_kkc > 0, f"Budget exceeds total classes ({n_classes}): kuc={kuc}, uuc={n_uuc}"
        return n_kkc, kuc, uuc_val, uuc_test

    if n_classes >= 100:
        return 50, 10, 10, 10
    elif n_classes >= 35:
        return 20, 5, 5, 5
    elif n_classes >= 10:
        return 5, 2, 1, 2
    else:
        kuc_c = max(1, n_classes // 4)
        u_val = max(1, n_classes // 5)
        u_test = max(1, n_classes // 5)
        n_kkc = n_classes - (kuc_c + u_val + u_test)
        assert n_kkc > 0
        return n_kkc, kuc_c, u_val, u_test


def make_oss(path, kuc=None, uuc_val=None, uuc_test=None,
             train_size=0.7, val_size=0.2, test_size=0.1, seed=0):
    X, y = load_cached(path)
    n_classes = len(np.unique(y))
    n_kkc, kuc_b, uuc_val_b, uuc_test_b = choose_budget(n_classes, kuc=kuc, uuc_val=uuc_val, uuc_test=uuc_test)

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    oss = DynamicOSS(ds, train_size=train_size, val_size=val_size,
                     test_size=test_size, kuc=kuc_b,
                     uuc_val=uuc_val_b, uuc_test=uuc_test_b, seed=seed)

    kkc = sorted([int(c) for c in oss.kkc])
    kuc_ = sorted([int(c) for c in oss.kuc])
    uuc = sorted([int(c) for c in oss.uuc])
    uuc_val_classes = sorted([int(c) for c in oss.classes["val"]["uuc"]])
    uuc_test_classes = sorted([int(c) for c in oss.classes["test"]["uuc"]])

    parts = {"train": oss.train_dataset(), "val": oss.val_dataset(), "test": oss.test_dataset()}
    idx = {k: np.asarray(v.indices) for k, v in parts.items()}   # row ids into X/y
    return dict(
        X=X, y=y, oss=oss, idx=idx,
        kkc=kkc, kuc=kuc_, uuc=uuc,
        uuc_val=uuc_val_classes, uuc_test=uuc_test_classes,
        n_kkc=len(kkc), n_kuc=len(kuc_), n_uuc=len(uuc),
        budget=dict(n_kkc=len(kkc), kuc=len(kuc_), uuc_val=len(uuc_val_classes), uuc_test=len(uuc_test_classes))
    )


def tag_and_report_budget(world, k=5, q=0.95, save_path="results/oss_metrics/budget_info.json"):
    """Compute near/far tags for novel classes, print summary, and save to JSON."""
    X, y = world["X"], world["y"]
    kkc = world["kkc"]
    kuc = set(world["kuc"])
    uuc_val = set(world["uuc_val"])
    uuc_test = set(world["uuc_test"])
    novel_classes = sorted(list(kuc | set(world["uuc"])))

    # Fit class indices on KKC training rows
    train_idx = world["idx"]["train"]
    train_y = y[train_idx]
    train_kkc_rows = train_idx[np.isin(train_y, kkc)]

    Z_train = prep(X[train_kkc_rows])
    y_train_raw = y[train_kkc_rows]
    lut = {c: i for i, c in enumerate(kkc)}
    y_train_idx = np.array([lut[c] for c in y_train_raw], dtype=np.int64)

    K = len(kkc)
    indices = fit_class_indices(Z_train, y_train_idx, K)

    # Calibrate tau with safe k
    min_samples = min(np.sum(y_train_idx == c) for c in range(K))
    k_eff = min(k, max(1, min_samples - 1))
    tau = calibrate(indices, Z_train, y_train_idx, k=k_eff, q=q)

    novel_info = []
    med_norm_dists = []
    for c in novel_classes:
        Z_c = prep(X[y == c])
        d_norm = class_distances(indices, Z_c, k=k_eff, tau=tau).min(axis=1)
        d_raw = class_distances(indices, Z_c, k=k_eff, tau=None).min(axis=1)

        if c in kuc:
            grp = "kuc"
            split = "train/val/test"
        elif c in uuc_val:
            grp = "uuc"
            split = "val"
        else:
            grp = "uuc"
            split = "test"

        med_norm = float(np.median(d_norm))
        med_norm_dists.append(med_norm)

        novel_info.append({
            "class": int(c),
            "group": grp,
            "split": split,
            "n_samples": int(len(Z_c)),
            "mean_norm_dist": float(np.mean(d_norm)),
            "median_norm_dist": med_norm,
            "mean_raw_dist": float(np.mean(d_raw)),
            "median_raw_dist": float(np.median(d_raw)),
        })

    # Near vs far tag:
    # Under quantile calibration, points with normalized distance <= 1.0 lie within the calibrated support radius
    cutoff = 1.0 if any(item["median_norm_dist"] <= 1.0 for item in novel_info) else float(np.median(med_norm_dists))
    for item in novel_info:
        item["tag"] = "near" if item["median_norm_dist"] <= cutoff else "far"

    # Print summary
    print("\n" + "=" * 68)
    print("OSSIM CLASS INVENTORY & BUDGET SUMMARY")
    print("=" * 68)
    print(f"Counts: n_kkc = {world['n_kkc']}, n_kuc = {world['n_kuc']}, n_uuc = {world['n_uuc']}")
    print(f"  - KKC classes ({len(world['kkc'])}): {world['kkc']}")
    print(f"  - KUC classes ({len(world['kuc'])}): {world['kuc']}")
    print(f"  - UUC classes ({len(world['uuc'])}): {world['uuc']}")
    print(f"      in val  ({len(world['uuc_val'])}): {world['uuc_val']}")
    print(f"      in test ({len(world['uuc_test'])}): {world['uuc_test']}")
    print("-" * 68)
    print(f"{'Class':>6s}  {'Group':>5s}  {'Split':>14s}  {'Norm Dist (mean)':>16s}  {'Raw Dist':>10s}  {'Tag':>5s}")
    print("-" * 68)
    for it in novel_info:
        print(f"{it['class']:6d}  {it['group']:>5s}  {it['split']:>14s}  {it['mean_norm_dist']:16.3f}  {it['mean_raw_dist']:10.3f}  {it['tag']:>5s}")
    print("=" * 68 + "\n")

    # Save to JSON
    out_dict = {
        "n_kkc": world["n_kkc"],
        "n_kuc": world["n_kuc"],
        "n_uuc": world["n_uuc"],
        "n_uuc_val": len(world["uuc_val"]),
        "n_uuc_test": len(world["uuc_test"]),
        "classes": {
            "kkc": world["kkc"],
            "kuc": world["kuc"],
            "uuc": world["uuc"],
            "uuc_val": world["uuc_val"],
            "uuc_test": world["uuc_test"],
        },
        "novel_classes": novel_info,
        "tau": [float(t) for t in tau],
    }
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(out_dict, f, indent=2)
        print(f"Budget and novel class info saved to: {save_path}")

    return out_dict


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


def create_etf(k, d):
    """Fixed ETF readout: k unit prototypes, maximally separated (neural collapse)."""
    I = np.eye(k)
    C = np.ones((k, k)) / k
    U, _, _ = np.linalg.svd(I - C)
    W = U[:, :k - 1] * np.sqrt(k / (k - 1))
    if d > k - 1:
        W = np.hstack([W, np.zeros((k, d - (k - 1)))])
    return torch.tensor(W.T, dtype=torch.float32)          # (d, k)


class ETFHead(nn.Module):
    """Project frozen features to the ETF space; the classifier itself is FIXED.

    Trainable part: the linear projector only (no ReLU -- ETF prototypes need
    signed coordinates). Logits = scale * normalised(z) @ ETF, so training pulls
    same-class features onto their prototype. Note: dropout-BALD is the WRONG EU
    estimator here (it tracks boundary jitter of collapsed points, not novelty)
    -- run it to see the contrast with Head, and reach for the kNN/prototype
    distances for ETF-side uncertainty.
    """

    def __init__(self, d_in=512, d_etf=32, k=3, scale=10., p=0.5):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_etf), nn.Dropout(p))
        self.register_buffer("etf", create_etf(k, d_etf))
        self.scale = scale

    def forward(self, x):
        z = F.normalize(self.proj(x), dim=1)
        return self.scale * (z @ self.etf)


def train(head, loader, lut, epochs=40, lr=1e-3, seed=0):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    torch.manual_seed(seed)
    losses = []
    for _ in range(epochs):
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





#
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="loss + EU on OSSIM groups over cached or timm features")
    ap.add_argument("--source", choices=("cache", "timm"), default="cache",
                    help="cached CLIP features, or extract fresh with a timm backbone")
    ap.add_argument("--model", default="resnet18", help="timm model name (with --source timm)")
    ap.add_argument("--n-per-class", type=int, default=None,
                    help="balanced per-class subsample before extraction (speed knob)")
    ap.add_argument("--epochs", type=int, default=40, help="number of epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    ap.add_argument("--head", choices=("mlp", "etf"), default="mlp",
                    help="mlp: dropout BALD separates known/novel; etf: fixed readout, EU inverts (contrast)")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--kuc", type=int, default=None, help="number of known-unknown classes")
    ap.add_argument("--uuc-val", type=int, default=None, help="number of UUC classes in val")
    ap.add_argument("--uuc-test", type=int, default=None, help="number of UUC classes in test")
    ap.add_argument("--save-budget", default="results/oss_metrics/budget_info.json",
                    help="path to save class budget and novel tags JSON")
    args = ap.parse_args()

    if args.source == "timm":
        extract_features_timm(args.model, args.n_per_class)
        path = f"cache/features/timm_{args.model}_c10test.npz"
    else:
        path = "cache/features/cifar10_test.npz"

    world = make_oss(path, kuc=args.kuc, uuc_val=args.uuc_val, uuc_test=args.uuc_test, seed=args.seed)

    # Print and save class lists, UUC in val vs test, and near/far tags per novel class
    tag_and_report_budget(world, k=5, q=0.95, save_path=args.save_budget)

    X, y, idx = world["X"], world["y"], world["idx"]
    classes = world["kkc"]
    lut = torch.full((1024,), -1, dtype=torch.long)
    for k, c in enumerate(classes):
        lut[c] = k

    train_ld = make_loader(X, y, group_idx(world, "train", "kkc"), batch_size=256, shuffle=True, seed=args.seed)
    groups = [g for g in ("kkc", "kuc", "uuc") if len(world[g]) > 0]
    val = {g: make_loader(X, y, group_idx(world, "val", g), batch_size=512) for g in groups}
    test = {g: make_loader(X, y, group_idx(world, "test", g), batch_size=512) for g in groups}

    # train on kkc: loss should fall; then EU separates known (low) from novel (high)
    head = ETFHead(d_in=X.shape[1], k=len(classes)) if args.head == "etf" else Head(k=len(classes))
    losses = train(head, train_ld, lut, epochs=args.epochs, lr=args.lr, seed=args.seed)
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
        if (split, "kkc") in table:
            for novel_grp in ("kuc", "uuc"):
                if (split, novel_grp) in table and len(table[(split, novel_grp)]) > 0:
                    print(f"{split}: EU AUROC kkc vs {novel_grp} = {auroc(table[(split, 'kkc')], table[(split, novel_grp)]):.3f}")