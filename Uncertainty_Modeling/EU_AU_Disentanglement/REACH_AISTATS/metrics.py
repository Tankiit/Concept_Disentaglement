# metrics.py -- declare everything up front; fill as you go
import json
from pathlib import Path

import numpy as np

SECTIONS = {
    "data":      ["n_rows", "feat_dim", "feat_norm_mean", "classes_present"],
    "split":     ["kkc", "kuc", "uuc", "n_kkc", "rows_per_split",
                  "classes_per_split", "uuc_in_val", "uuc_in_test", "leak_free"],
    "head":      ["n_members", "recipe", "acc_known_per_member", "acc_known_ens"],
    "scores":    ["MI", "AU", "MSP"],              # each: per group {known,kuc,uuc} -> mean, ci, n
    "separate":  ["auroc_known_vs_kuc", "auroc_known_vs_uuc"],   # per score
    "knn":       ["slot_acc_loo", "tau_c", "frac_outside_known", "frac_outside_kuc", "frac_outside_uuc"],
    "diversity": ["mi_scale_seed", "mi_scale_bootstrap"],
}

MIN_GROUP = 200  # report() flags groups below this


def new_metrics():
    return {s: {k: None for k in keys} for s, keys in SECTIONS.items()}


def _entropy(p):
    """Entropy in nats over the last axis; safe for zeros."""
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


def fill_data_and_split(M, X, y, out):
    """Everything here is free: no model needed."""
    M["data"].update(
        n_rows=len(X), feat_dim=X.shape[1],
        feat_norm_mean=float(np.linalg.norm(X.astype(np.float32), axis=1).mean()),
        classes_present=sorted(np.unique(y).tolist()))
    kkc, kuc, uuc = set(out["kkc"]), set(out["kuc"]), set(out["uuc"])
    per = {n: dict(zip(*[a.tolist() for a in np.unique(y[ix], return_counts=True)]))
           for n, ix in out["idx"].items()}
    M["split"].update(
        kkc=sorted(kkc), kuc=sorted(kuc), uuc=sorted(uuc), n_kkc=len(kkc),
        rows_per_split={n: len(ix) for n, ix in out["idx"].items()},
        classes_per_split=per,
        uuc_in_val=sorted(set(per["val"]) & uuc),
        uuc_in_test=sorted(set(per["test"]) & uuc),
        leak_free=all(not (set(out["idx"][a]) & set(out["idx"][b]))
                      for a, b in [("train", "val"), ("train", "test"), ("val", "test")]))
    return M


def fill_scores(M, P, masks):
    """P: (members, N, K) probs on the eval rows. masks: group -> boolean over N.

    mean p over members; H = entropy(mean p); AU = mean member entropy;
    MI = H - AU (epistemic, BALD; asserted >= -1e-8); MSP = max of mean p.
    Per group store {mean, ci over MEMBERS, n}: MI's spread is a leave-one-member-
    out jackknife; AU and MSP get the direct per-member spread. Returns the
    per-row scores {name: (N,) array} for fill_separation.
    """
    P = np.asarray(P, dtype=np.float64)
    T = len(P)
    mean_p = P.mean(0)
    AU = _entropy(P).mean(0)
    MI = _entropy(mean_p) - AU
    assert MI.min() >= -1e-8, f"MI below zero by {float(MI.min()):.2e}"
    MSP = mean_p.max(-1)
    rows = {"MI": MI, "AU": AU, "MSP": MSP}

    def member_values(name):
        if name == "AU":
            return [_entropy(P[t])[m].mean() for t in range(T)]
        if name == "MSP":
            return [P[t][m].max(-1).mean() for t in range(T)]
        out = []
        for t in range(T):                          # MI: jackknife over members
            Pk = np.delete(P, t, axis=0)
            out.append(float((_entropy(Pk.mean(0)) - _entropy(Pk).mean(0))[m].mean()))
        return out

    for name, s in rows.items():
        M["scores"][name] = {}
        for g, m in masks.items():
            if not m.any():
                M["scores"][name][g] = {"mean": None, "ci": None, "n": 0}
                continue
            per_member = member_values(name)
            lo, hi = np.quantile(per_member, [.025, .975])
            M["scores"][name][g] = {"mean": float(s[m].mean()),
                                    "ci": (float(lo), float(hi)), "n": int(m.sum())}
    return rows


def fill_separation(M, scores, masks):
    """AUROC known-vs-kuc and known-vs-uuc for each score.

    Sign convention: higher = more uncertain, same for every score; probly's
    evaluate_ood shares it, so its auroc is used directly. Groups with no rows
    leave an empty dict behind rather than being dropped.
    """
    from probly.evaluation.ood import evaluate_ood
    for other in ("kuc", "uuc"):
        key = f"auroc_known_vs_{other}"
        M["separate"][key] = {}
        if masks["known"].any() and masks.get(other, np.zeros(1, bool)).any():
            for name, s in scores.items():
                s_use = 1 - s if name == "MSP" else s   # MSP is confidence; flip to uncertainty
                M["separate"][key][name] = float(
                    evaluate_ood(s_use[masks["known"]], s_use[masks[other]])["auroc"])
    return M


def to_json(M, path=None):
    """JSON-safe dump of the metrics dict (numpy scalars -> python, tuples -> arrays).

    path None: return the JSON string; otherwise write it (parents created).
    """
    def cv(o):
        if isinstance(o, dict):
            return {str(k): cv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [cv(v) for v in o]
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    data = cv(M)
    if path is None:
        return json.dumps(data, indent=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print("metrics written to", path)
    return data


def report(M):
    have, missing = [], []
    for s, d in M.items():
        for k, v in d.items():
            (missing if v is None else have).append(f"{s}.{k}")
    print(f"HAVE    ({len(have)}):", *have, sep="\n  ")
    print(f"MISSING ({len(missing)}):", *missing, sep="\n  ")
    mi = M["scores"].get("MI") or {}
    sizes = {g: v["n"] for g, v in mi.items() if isinstance(v, dict)}
    if sizes:
        print("group sizes:", sizes)
        small = {g: n for g, n in sizes.items() if 0 < n < MIN_GROUP}
        if small:
            print(f"SMALL GROUPS (n < {MIN_GROUP}):", small)


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from example import Head, group_idx, make_loader, make_oss, train

    world = make_oss("cache/features/cifar10_test.npz", kuc=0, uuc_val=2, uuc_test=5, seed=0)
    X, y = world["X"], world["y"]
    classes = world["kkc"]
    lut = torch.full((1024,), -1, dtype=torch.long)
    for k, c in enumerate(classes):
        lut[c] = k
    train_ld = make_loader(X, y, group_idx(world, "train", "kkc"),
                           batch_size=256, shuffle=True, seed=0)

    MEMBERS = 5
    probs, accs = [], []
    for s in range(MEMBERS):
        head = Head(k=len(classes))
        train(head, train_ld, lut, epochs=40, seed=s)
        head.eval()
        rows, ys = [], []
        with torch.no_grad():
            for b in make_loader(X, y, world["idx"]["test"], batch_size=512):
                rows.append(F.softmax(head(b["X"]), dim=1).numpy())
                ys.append(b["y"].numpy())
        P_t = np.vstack(rows)
        y_eval = np.concatenate(ys)
        known = np.isin(y_eval, classes)
        y_idx = lut[torch.as_tensor(y_eval)].numpy()     # raw ids -> head class indices
        accs.append(float((P_t.argmax(1)[known] == y_idx[known]).mean()))
        probs.append(P_t)
    P = np.stack(probs)                              # (members, N, K)

    y_eval = y[world["idx"]["test"]]
    masks = {"known": np.isin(y_eval, classes),
             "kuc": np.isin(y_eval, world["kuc"]),
             "uuc": np.isin(y_eval, world["uuc"])}

    M = new_metrics()
    fill_data_and_split(M, X, y, world)
    ens = (P.mean(0).argmax(1)[masks["known"]] ==
           lut[torch.as_tensor(y_eval)].numpy()[masks["known"]]).mean()
    M["head"].update(n_members=MEMBERS, recipe="deep ensemble: 40 epochs Adam per seed",
                     acc_known_per_member=[round(a, 4) for a in accs],
                     acc_known_ens=float(ens))
    rows = fill_scores(M, P, masks)
    fill_separation(M, rows, masks)
    report(M)
    to_json(M, "results/oss_metrics/metrics.json")

    print("\nknown-class accuracy: members", M["head"]["acc_known_per_member"],
          "| ensemble", round(M["head"]["acc_known_ens"], 4))
    for name in ("MI", "AU", "MSP"):
        means = {g: (None if v["mean"] is None else round(v["mean"], 4))
                 for g, v in M["scores"][name].items()}
        print(f"{name:4s} mean per group: {means}")
    for key, per in M["separate"].items():
        print(f"{key}: " + (", ".join(f"{n}={a:.3f}" for n, a in per.items()) or "no pairs"))


def fill_knn(M, X_lab, y_lab, eval_pools, k=10, q=0.95):
    """kNN slot metrics via exact L2 search (sklearn; faiss's threadpool deadlocks
    after torch/cvxpy load in the full suite, and these pools are small).

    slot_acc_loo: leave-one-out global kNN vote (k+1 search, self dropped, majority).
    tau_c: per-class q-quantile of the k-th-neighbour distance within the class
    (self excluded). frac_outside_<group>: fraction of that pool whose nearest-class
    calibrated distance (min over classes of d_k / tau_c) exceeds 1.
    """
    from sklearn.neighbors import NearestNeighbors
    X_lab = np.asarray(X_lab, dtype=np.float32)
    classes = [int(c) for c in np.unique(y_lab)]
    gnn = NearestNeighbors(n_neighbors=k + 1).fit(X_lab)
    _, Ig = gnn.kneighbors(X_lab, k + 1)               # self at distance 0 dropped
    votes = y_lab[Ig[:, 1:]]
    pred = np.array([np.bincount(v).argmax() for v in votes])
    taus, class_nn = {}, {}
    for c in classes:
        rows = X_lab[y_lab == c]
        nn = NearestNeighbors(n_neighbors=k + 1).fit(rows)
        Dc, _ = nn.kneighbors(rows, k + 1)             # self (distance 0) dropped
        taus[c] = float(np.quantile(Dc[:, -1], q))
        class_nn[c] = NearestNeighbors(n_neighbors=k).fit(rows)

    M['knn']['slot_acc_loo'] = float((pred == y_lab).mean())
    M['knn']['tau_c'] = taus
    for g, (Xe, ye) in eval_pools.items():
        Q = np.asarray(Xe, dtype=np.float32)
        per_class = np.stack([class_nn[c].kneighbors(Q, k)[0][:, -1] / taus[c]
                              for c in classes], axis=1)
        M['knn'][f'frac_outside_{g}'] = float((per_class.min(axis=1) > 1.0).mean())
    return M


def fill_diversity(M, stacks):
    """stacks: {name: (members, N, K) probability stack}. MI scale = spread of the
    per-row ensemble MI (TU - AU) across rows, plus the member count for provenance."""
    for name, P in stacks.items():
        P = np.asarray(P, dtype=np.float64)
        mean = P.mean(0)
        tu = -(mean * np.log(np.clip(mean, 1e-12, None))).sum(-1)
        au = -(P * np.log(np.clip(P, 1e-12, None))).sum(-1).mean(0)
        mi = tu - au
        M['diversity'][f'mi_scale_{name}'] = {
            'mean': float(mi.mean()), 'std': float(mi.std()),
            'max': float(mi.max()), 'n_members': int(P.shape[0])}
    return M
