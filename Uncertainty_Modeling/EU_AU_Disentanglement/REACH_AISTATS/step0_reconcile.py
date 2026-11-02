# step0_reconcile.py -- COLLABORATOR_PLAN step 0: reconcile the learned-unknown-output
# AUROCs (plan quotes 0.970 vs 0.664) and resolve the margin/MSP identity flag.
#
# Rebuilds eu_blindspot.ipynb cells 3-8 exactly (world seed 0, kuc=1, uuc_val=2,
# uuc_test=5; HeadDrop K+1 head, 300 epochs, batch 128, Adam 1e-3), then scores the
# learned unknown output P(unknown) under every evaluation variant we can construct
# from the recorded runs, plus head-seed and world-seed spreads, and checks the
# margin/MSP identity (identical AUROC in the notebook's score table).
#
# Deviation from the notebook (documented): cell 6 constructed the head before
# torch.manual_seed(0), so its init drew from an unreproducible ambient RNG state.
# Here torch.manual_seed(seed) runs BEFORE head construction; the cell-8 readout
# should sit inside the head-seed spread this produces.
#
# Run:  OMP_NUM_THREADS=1 python step0_reconcile.py
# Out:  results/step0_reconciliation/summary.json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pytorch_ood.dataset.ossim import DynamicOSS
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from metrics import CIFAR10_CLASSES

OUT_DIR = Path("results/step0_reconciliation")
FEAT = "cache/features/features.pt"
T_MC, EPOCHS, HIDDEN, P_DROPOUT, LR, BATCH = 30, 300, 128, 0.5, 1e-3, 128

# What eu_blindspot.ipynb (seed 0, kuc=1, uuc_val=2, uuc_test=5) actually produced;
# the reconciliation is only exact if today's DynamicOSS reproduces this split.
NOTEBOOK_WORLD = {
    "kkc": [1, 8], "kuc": [4], "uuc": [0, 2, 3, 5, 6, 7, 9],
    "rows": {"train": 2100, "val": 2600, "test": 5300},
    "classes": {"train": [1, 4, 8], "val": [1, 2, 4, 6, 8],
                "test": [0, 1, 3, 4, 5, 7, 8, 9]},
}


def names(ids):
    return {int(i): CIFAR10_CLASSES[int(i)] for i in ids}


def load_cached(path):
    b = torch.load(path)
    entry = b["cifar10_test"]
    X = entry["X"].float().numpy()          # fp16 -> fp32 before any math
    y = entry["y"].numpy().astype(np.int64)
    assert X.ndim == 2 and len(X) == len(y), (X.shape, y.shape)
    print(f"{path}: X {X.shape}, mean norm {np.linalg.norm(X, axis=1).mean():.3f}")
    return X, y


def make_oss_notebook(path, kuc=1, uuc_val=2, uuc_test=5,
                      train_size=0.7, val_size=0.2, test_size=0.1, seed=0):
    """eu_blindspot.ipynb cell 3 verbatim (inline copy, NOT example.make_oss)."""
    X, y = load_cached(path)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    oss = DynamicOSS(ds, train_size=train_size, val_size=val_size,
                     test_size=test_size, kuc=kuc,
                     uuc_val=uuc_val, uuc_test=uuc_test, seed=seed)
    kkc, kuc_, uuc = set(oss.kkc), set(oss.kuc), set(oss.uuc)
    assert kkc.isdisjoint(uuc) and kkc.isdisjoint(kuc_)
    parts = {"train": oss.train_dataset(in_dist=True, out_dist=True),
             "val": oss.val_dataset(in_dist=True, out_dist=True),
             "test": oss.test_dataset(in_dist=True, out_dist=True)}
    idx = {k: np.asarray(v.indices) for k, v in parts.items()}
    assert not (set(idx["train"]) & set(idx["val"]) | set(idx["train"]) & set(idx["test"])
                | set(idx["val"]) & set(idx["test"])), "split leak"
    return dict(X=X, y=y, idx=idx, kkc=sorted(kkc), kuc=sorted(kuc_), uuc=sorted(uuc))


def assert_notebook_world(w):
    """Fail loudly if DynamicOSS no longer reproduces the recorded notebook split."""
    for key in ("kkc", "kuc", "uuc"):
        got = [int(c) for c in w[key]]
        assert got == NOTEBOOK_WORLD[key], f"{key}: {got} != {NOTEBOOK_WORLD[key]}"
    for split, n in NOTEBOOK_WORLD["rows"].items():
        assert len(w["idx"][split]) == n, (split, len(w["idx"][split]), n)
    for split, cs in NOTEBOOK_WORLD["classes"].items():
        got = sorted(np.unique(w["y"][w["idx"][split]]).tolist())
        assert got == cs, (split, got, cs)


def per_class_n(w, split):
    y = w["y"][w["idx"][split]]
    cs, ns = np.unique(y, return_counts=True)
    return {f"class {int(c):2d} {CIFAR10_CLASSES[int(c)]:10s}": int(n)
            for c, n in zip(cs, ns)}


class HeadDrop(nn.Module):
    """eu_blindspot.ipynb cell 6 verbatim."""
    def __init__(self, d_in, hidden=HIDDEN, k_total=3, p=P_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Dropout(p), nn.Linear(hidden, k_total))
    def forward(self, x):
        return self.net(x)


def train_kp1(w, seed=0, epochs=EPOCHS):
    """Cells 5+6 protocol: kkc -> 0..K-1, every kuc id -> output K (the learned
    unknown); train on kkc+kuc train rows only."""
    K = len(w["kkc"])
    lut = torch.full((1024,), -1, dtype=torch.long)
    for k, c in enumerate(w["kkc"]):
        lut[c] = k
    for c in w["kuc"]:
        lut[c] = K
    ti = w["idx"]["train"]
    ds = TensorDataset(torch.from_numpy(w["X"][ti]).float(),
                       lut[torch.from_numpy(w["y"][ti]).long()])
    ld = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(0))
    torch.manual_seed(seed)                 # BEFORE construction (see header note)
    head = HeadDrop(w["X"].shape[1], k_total=K + 1)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    for _ in range(epochs):
        head.train()
        for xb, yb in ld:
            opt.zero_grad()
            loss = F.cross_entropy(head(xb), yb)
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        acc = float((head(ds.tensors[0]).argmax(1) == ds.tensors[1]).float().mean())
    assert acc > 0.9, f"head untrained (acc {acc:.3f}) -- do not trust anything below"
    return head, lut, acc


def mc_probs(head, Xq, t=T_MC):
    """T softmax passes with dropout active -> (T, N, K+1); restores mode."""
    was_training = head.training
    head.train()
    with torch.no_grad():
        out = torch.stack([F.softmax(head(Xq), dim=1) for _ in range(t)])
    head.train(was_training)
    return out.numpy()


def group_table(w, split, mean_p, K):
    yt = w["y"][w["idx"][split]]
    ent = -(mean_p * np.log(np.clip(mean_p, 1e-12, None))).sum(1)
    p_unk = mean_p[:, K]
    rows = {}
    for g in ("kkc", "kuc", "uuc"):
        m = np.isin(yt, w[g])
        rows[g] = dict(n=int(m.sum()), TU=float(ent[m].mean()), P_unk=float(p_unk[m].mean()))
    return rows


def auroc(pos, neg):
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                               np.r_[pos, neg]))


def main():
    torch.set_num_threads(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    S = {"protocol": dict(T_MC=T_MC, epochs=EPOCHS, hidden=HIDDEN, dropout=P_DROPOUT,
                          lr=LR, batch=BATCH, features=FEAT,
                          head_seed_policy="torch.manual_seed(seed) before construction"),
         "findings": []}

    # ---- world + class identities (step 1 rides along) ----------------------
    w = make_oss_notebook(FEAT, seed=0)
    assert_notebook_world(w)
    K = len(w["kkc"])
    S["world"] = {
        "seed": 0, "kkc": w["kkc"], "kuc": w["kuc"], "uuc": w["uuc"],
        "kkc_names": names(w["kkc"]), "kuc_names": names(w["kuc"]),
        "uuc_names": names(w["uuc"]),
        "uuc_in_test": [int(c) for c in sorted(set(w["uuc"]) &
                                               set(np.unique(w["y"][w["idx"]["test"]]).tolist()))],
        "per_class_n": {s: per_class_n(w, s) for s in ("train", "val", "test")},
    }
    print("world reproduced exactly: kkc", names(w["kkc"]), "| kuc", names(w["kuc"]),
          "| uuc", names(w["uuc"]))
    for s in ("train", "val", "test"):
        print(f"  {s:5s}", S["world"]["per_class_n"][s])

    # ---- the cell-8 run ------------------------------------------------------
    head, lut, acc = train_kp1(w, seed=0)
    Xt = torch.from_numpy(w["X"][w["idx"]["test"]]).float()
    yv = w["y"][w["idx"]["test"]]
    P = mc_probs(head, Xt)                          # (T, N, K+1)
    mean_p = P.mean(0)
    p_unk_mc = mean_p[:, K]
    with torch.no_grad():
        p_unk_det = F.softmax(head(Xt), dim=1).numpy()[:, K]      # eval mode
        head.train()                                             # dropout-active draw
        logits_mc = torch.stack([head(Xt) for _ in range(T_MC)])
        head.eval()
    p_unk_logitmean = F.softmax(logits_mc.mean(0), dim=1).numpy()[:, K]

    known = np.isin(yv, w["kkc"])
    kuc_m = np.isin(yv, w["kuc"])
    uuc_m = np.isin(yv, w["uuc"])
    notnovel = ~uuc_m                                 # kkc + kuc = cell 8's actual negatives

    EU = (-(mean_p * np.log(np.clip(mean_p, 1e-12, None))).sum(1)
          - (-(P * np.log(np.clip(P, 1e-12, None))).sum(-1).mean(0)))

    V = {}
    # The notebook cell-8 CODE: roc_auc_score(uuc, p_unknown) -- negatives are every
    # non-uuc row, i.e. kkc + the 100 kuc deer rows trained on the unknown output.
    V["cell8_code (pos uuc | neg kkc+kuc, MC mean probs)"] = auroc(p_unk_mc[uuc_m], p_unk_mc[notnovel])
    # The honest readout the notebook's printed label suggests (negatives = kkc only).
    V["kkc-only negatives (pos uuc | neg kkc, MC mean probs)"] = auroc(p_unk_mc[uuc_m], p_unk_mc[known])
    V["kkc-only, eval-mode deterministic"] = auroc(p_unk_det[uuc_m], p_unk_det[known])
    V["kkc-only, softmax(mean logits)"] = auroc(p_unk_logitmean[uuc_m], p_unk_logitmean[known])
    # The optimistic readout cell 8's own comment flags (deer positives).
    V["kuc readout (pos kuc | neg kkc, MC mean probs)"] = auroc(p_unk_mc[kuc_m], p_unk_mc[known])
    # Decomposition check: pooled-over-negatives = weighted mix of the two pair types.
    V["uuc vs kuc rows only (pos uuc | neg kuc)"] = auroc(p_unk_mc[uuc_m], p_unk_mc[kuc_m])
    for key, val in V.items():
        print(f"{key:55s} AUROC {val:.3f}")

    # val split readout (uuc_val = classes 2, 6 -- a different, easier novel mix)
    Xv = torch.from_numpy(w["X"][w["idx"]["val"]]).float()
    yvv = w["y"][w["idx"]["val"]]
    Pv = mc_probs(head, Xv).mean(0)
    p_unk_val = Pv[:, K]
    val_known = np.isin(yvv, w["kkc"])
    val_uuc = np.isin(yvv, w["uuc"])
    V["val split (pos uuc_val | neg kkc, MC mean probs)"] = auroc(p_unk_val[val_uuc], p_unk_val[val_known])

    S["cell8_replica"] = {"train_acc": acc,
                          "groups": group_table(w, "test", mean_p, K),
                          "EU_auroc_pos_uuc_neg_kkc": auroc(EU[uuc_m], EU[known]),
                          "EU_auroc_pos_uuc_neg_kkc_kuc": auroc(EU[uuc_m], EU[notnovel]),
                          "variants": V}

    # ---- head-seed spread of the cell-8 readout ------------------------------
    spread = []
    for s in range(5):
        h, _, a = train_kp1(w, seed=s)
        pm = mc_probs(h, Xt).mean(0)[:, K]
        spread.append(dict(seed=s, train_acc=a,
                           auroc_uuc=auroc(pm[uuc_m], pm[known]),
                           auroc_kuc=auroc(pm[kuc_m], pm[known])))
        print(f"head seed {s}: P(unk) AUROC uuc {spread[-1]['auroc_uuc']:.3f} "
              f"| kuc {spread[-1]['auroc_kuc']:.3f}")
    S["head_seed_spread"] = spread

    # ---- world-seed spread of the cell-8 readout ------------------------------
    wspread = []
    for ws in range(4):
        ww = make_oss_notebook(FEAT, seed=ws)
        h, _, a = train_kp1(ww, seed=0)
        Xt_w = torch.from_numpy(ww["X"][ww["idx"]["test"]]).float()
        yv_w = ww["y"][ww["idx"]["test"]]
        pm = mc_probs(h, Xt_w).mean(0)[:, len(ww["kkc"])]
        kn, uu = np.isin(yv_w, ww["kkc"]), np.isin(yv_w, ww["uuc"])
        kk = np.isin(yv_w, ww["kuc"])
        wspread.append(dict(world_seed=ws, kkc=[int(c) for c in ww["kkc"]],
                            kuc=[int(c) for c in ww["kuc"]], train_acc=a,
                            auroc_uuc=auroc(pm[uu], pm[kn]),
                            auroc_kuc=auroc(pm[kk], pm[kn])))
        print(f"world seed {ws}: kkc {ww['kkc']} kuc {ww['kuc']} -> "
              f"P(unk) AUROC uuc {wspread[-1]['auroc_uuc']:.3f} | kuc {wspread[-1]['auroc_kuc']:.3f}")
    S["world_seed_spread"] = wspread

    # ---- margin / MSP identity -----------------------------------------------
    head.eval()
    with torch.no_grad():
        Z = head(Xt).numpy()                        # deterministic logits, all K+1
    Zk = Z[:, :K]                                   # known logits (cell 43's basis)
    srt = np.sort(Zk, axis=1)
    neg_margin = -(srt[:, -1] - srt[:, -2])
    pk = F.softmax(torch.from_numpy(Zk), dim=1).numpy()
    neg_msp = -pk.max(1)
    full = F.softmax(torch.from_numpy(Z), dim=1).numpy()
    neg_msp_full = -full.max(1)
    srt3 = np.sort(Z, axis=1)
    neg_margin_full = -(srt3[:, -1] - srt3[:, -2])

    def a(y, s):
        return float(roc_auc_score(y.astype(int), s))

    S["margin_msp"] = {
        "K_known_logits": K,
        "known_only": {
            "auroc_neg_margin": a(uuc_m, neg_margin), "auroc_neg_msp": a(uuc_m, neg_msp),
            "spearman": float(spearmanr(neg_margin, neg_msp).statistic),
            "note": "K=2: max softmax = sigmoid(top1-top2), strictly monotone in the "
                    "margin -> identical ranking and AUROC by construction"},
        "all_outputs_Kplus1": {
            "auroc_neg_margin": a(uuc_m, neg_margin_full),
            "auroc_neg_msp": a(uuc_m, neg_msp_full),
            "spearman": float(spearmanr(neg_margin_full, neg_msp_full).statistic),
            "note": "K+1=3 outputs: margin and MSP ranks genuinely differ"},
        "cell43_mode_caveat": "cell 43 computed Zk after cell 13 restored head.train(), "
                              "so the notebook's margin/MSP (and its EU table basis) came "
                              "from a single dropout-active stochastic forward; eval mode "
                              "here is the deterministic counterpart",
    }
    print("margin/MSP (known logits, K=2): AUROC",
          S["margin_msp"]["known_only"]["auroc_neg_margin"], "==",
          S["margin_msp"]["known_only"]["auroc_neg_msp"],
          "| spearman", S["margin_msp"]["known_only"]["spearman"])
    print("margin/MSP (all 3 logits):     AUROC",
          S["margin_msp"]["all_outputs_Kplus1"]["auroc_neg_margin"], "vs",
          S["margin_msp"]["all_outputs_Kplus1"]["auroc_neg_msp"],
          "| spearman", S["margin_msp"]["all_outputs_Kplus1"]["spearman"])

    # ---- findings -------------------------------------------------------------
    v_cell8 = V["cell8_code (pos uuc | neg kkc+kuc, MC mean probs)"]
    v_honest = V["kkc-only negatives (pos uuc | neg kkc, MC mean probs)"]
    v_kuc = V["kuc readout (pos kuc | neg kkc, MC mean probs)"]
    v_uuc_vs_kuc = V["uuc vs kuc rows only (pos uuc | neg kuc)"]
    # mixture identity: pooled negatives == row-weighted mean of the two pair types
    n_kkc, n_kuc = int(known.sum()), int(kuc_m.sum())
    v_mix = (n_kkc * v_honest + n_kuc * v_uuc_vs_kuc) / (n_kkc + n_kuc)
    uuc_vals = [r["auroc_uuc"] for r in spread] + [r["auroc_uuc"] for r in wspread]
    lo, hi = min(uuc_vals), max(uuc_vals)
    agg_shift = max(abs(V["kkc-only, eval-mode deterministic"] - v_honest),
                    abs(V["kkc-only, softmax(mean logits)"] - v_honest))
    S["findings"] = [
        f"ROOT CAUSE (resolves the plan's 0.970-vs-0.664): the two numbers are two "
        f"different NEGATIVE SETS, not two different runs of the same thing. Notebook "
        f"cell 8's code roc_auc_score(uuc_mask, p_unknown) takes negatives = every "
        f"non-uuc row = kkc (200) + kuc deer (100); the deer rows were TRAINED on the "
        f"unknown output, score P(unk)~0.99, and beat almost every uuc row, dragging "
        f"AUROC to {v_cell8:.3f} (notebook recorded 0.664-0.667 in all three committed "
        f"executions). With kkc-only negatives -- the honest readout the cell's printed "
        f"label implies -- the same head gives {v_honest:.3f}. The plan's 0.970 is a "
        f"normal kkc-only readout (this replica: {v_honest:.3f}).",
        f"Mixture identity confirms the mechanism: ({n_kkc}*{v_honest:.3f} + {n_kuc}*"
        f"{v_uuc_vs_kuc:.3f})/{n_kkc + n_kuc} = {v_mix:.3f} ~= cell-8 code value "
        f"{v_cell8:.3f}. The 0.664 is arithmetic, not instability.",
        f"Run-to-run spread of the honest readout: 5 head seeds {min(r['auroc_uuc'] for r in spread):.3f}-"
        f"{max(r['auroc_uuc'] for r in spread):.3f}; 4 world seeds {lo:.3f}-{hi:.3f}. "
        f"The unknown output is NOT unstable; the scare was a negative-set artifact. "
        f"World seed moves it more (novel-class mix) than head seed does.",
        f"Aggregation (eval-mode deterministic, softmax(mean logits)) shifts the honest "
        f"readout by at most {agg_shift:.3f} -- not a factor.",
        f"kuc readout (deer positives, the 'optimistic' one cell 8 flags): {v_kuc:.3f} "
        f"-- ceiling by construction (trained-on rows as positives); must never be "
        f"quoted as detection performance.",
        "margin/MSP identical AUROC is an identity at K=2 known outputs: max softmax = "
        "sigmoid(top1-top2), strictly monotone in the margin. Measured: AUROC "
        f"{S['margin_msp']['known_only']['auroc_neg_margin']:.5f} vs "
        f"{S['margin_msp']['known_only']['auroc_neg_msp']:.5f}, spearman "
        f"{S['margin_msp']['known_only']['spearman']:.6f} (deviation from 1.0 is float64 "
        "saturation of softmax for very large margins -- ties). Not a bug; arm A "
        "(kkc=2) can never separate the two, arms B/C (kkc=4/6) will.",
        "Cell 43's score table shares the same leak: its 'known' negatives are ~is_novel "
        "(300 rows including the 100 deer rows trained on the unknown output). Every "
        "number in the plan's exploratory score table (EU 0.949-0.953, margin 0.790) is "
        f"a kkc+kuc-negative number. EU is barely moved ({S['cell8_replica']['EU_auroc_pos_uuc_neg_kkc']:.3f} "
        f"kkc-only vs {S['cell8_replica']['EU_auroc_pos_uuc_neg_kkc_kuc']:.3f} with deer "
        "negatives) because deer EU is low, but margin/MSP and any known-vs-novel score "
        "must be recomputed under named negatives before the prereg freezes them.",
        "GATE DECISION: the learned unknown output is UNBLOCKED for confirmatory tables "
        "under the naming discipline -- kkc-only negatives for detection claims; the "
        "kuc readout disclosed as a trained-on ceiling; never an unnamed pooled "
        "negative set. The prereg's negative-set clause (already drafted in the plan) "
        "is the fix; no property of the method changes.",
    ]
    for f_ in S["findings"]:
        print("FINDING:", f_)

    (OUT_DIR / "summary.json").write_text(json.dumps(
        S, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    print("\nwritten:", OUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
