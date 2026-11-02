# epistemic_sets.py -- meeting 1 Nov 2026 (Tanmoy side): epistemic uncertainty as a SET.
#
# Contract from the collaborator meeting:
#   * extend the RepresentationBuilder base class with an epistemic-set representer
#     that outputs a list/set of contours (nested families of class sets);
#   * derive usable scalar scores from the interval: lower, upper, midpoint ("average"),
#     median-of-samples, widths, contour cardinality, and learned maps (logistic
#     regression / MLP over interval-valued features) -- the CredalAdapter
#     ("cradle adapter" in the meeting notes; post-hoc trainer, no backbone changes);
#   * run ID/OOD open-set simulation with pytorch_ood (DynamicOSS: kkc / kuc / uuc)
#     on cached features, laptop-scale;
#   * export every per-row value and every comparison in tabular form (pandas CSV).
#
# The RepresentationBuilder base class lives in the collaborator's codebase (the paper
# being replicated); it is not importable here, so this module declares the same
# minimal contract locally. If the upstream base becomes available, EpistemicSetRepresenter
# should be re-based onto it -- build() is the whole interface.
#
# Standing caveat (COLLABORATOR_PLAN 24 Oct): all AUROCs here are scalar-score AUROCs;
# envelopes are not membership intervals and there is no interval-valued AUROC claim.
# Negative discipline (step-0 finding): every AUROC names its negative set -- kkc test
# rows unless stated; the kuc readout is disclosed as trained-on whenever shown.
#
# Run:  OMP_NUM_THREADS=1 python epistemic_sets.py
# Out:  results/epistemic_sets/{rows.csv, summary.csv, per_class.csv, manifest.json,
#                               comparison.png}
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from example import make_oss, load_cached
from example_wrn50 import auroc_fast
from metrics import CIFAR10_CLASSES, cifar10_names

OUT_DIR = Path("results/epistemic_sets")
DEFAULT_ALPHAS = (0.25, 0.5, 0.75)      # contour grid: alpha-cuts of the interval family


# ---------------------------------------------------------------------------
# 1. Representation: epistemic uncertainty as a set of probability vectors
# ---------------------------------------------------------------------------
class RepresentationBuilder:
    """Minimal local declaration of the collaborator base class.

    build(stacked) turns a (T, N, K) stack of sampled predictives into a
    representation of epistemic uncertainty; the epistemic-set variant returns
    per-row probability intervals plus their contour family.
    """

    def build(self, stacked):
        raise NotImplementedError


@dataclass
class Contour:
    """One alpha-cut of a row's interval family: nested class sets."""
    alpha: float
    optimistic: np.ndarray      # (K,) bool: classes NOT excluded ({c: upper_c >= alpha})
    pessimistic: np.ndarray     # (K,) bool: classes CERTAINLY above ({c: lower_c >= alpha})

    @property
    def card_optimistic(self):
        return int(self.optimistic.sum())

    @property
    def card_pessimistic(self):
        return int(self.pessimistic.sum())


@dataclass
class EpistemicSet:
    """Per-row set-valued epistemic state over K classes (the credal envelope)."""
    lower: np.ndarray            # (N, K) min over the T sampled predictives
    upper: np.ndarray            # (N, K) max over the T sampled predictives
    mean: np.ndarray             # (N, K) mean predictive
    median: np.ndarray           # (N, K) median predictive (a "score from the set")
    n_samples: int

    @property
    def width(self):
        return self.upper - self.lower

    def contours(self, row, alphas=DEFAULT_ALPHAS):
        """Nested alpha-cut family for one row: pessimistic_alpha subsets
        optimistic_alpha, and both shrink as alpha grows."""
        return [Contour(alpha=a,
                        optimistic=self.upper[row] >= a,
                        pessimistic=self.lower[row] >= a)
                for a in sorted(alphas)]


class EpistemicSetRepresenter(RepresentationBuilder):
    """stacked (T, N, K) softmax samples -> EpistemicSet (envelope + contours).

    Same envelope semantics as example_wrn50.credal_intervals (probly's
    ArrayProbabilityIntervalsCredalSet): lower/upper are per-class extrema over
    the T sampled predictives; width = upper - lower is T-dependent (nesting:
    width over a pass subset <= width over the full set).
    """

    def build(self, stacked):
        stacked = np.asarray(stacked, dtype=np.float64)
        assert stacked.ndim == 3, stacked.shape
        return EpistemicSet(lower=stacked.min(0), upper=stacked.max(0),
                            mean=stacked.mean(0), median=np.median(stacked, 0),
                            n_samples=int(stacked.shape[0]))


# ---------------------------------------------------------------------------
# 2. Scalar scores derived from the set
# ---------------------------------------------------------------------------
def set_scores(bundle, k_unknown=None, alphas=DEFAULT_ALPHAS):
    """Interval -> scalar novelty scores, all higher = more novel-looking.

    k_unknown: index of the learned-unknown output of a K+1 head. When given,
    the *_novel scores are read off that column; otherwise they are computed on
    the residual 1 - max known-class probability (bounds propagate accordingly).
    """
    if k_unknown is not None:
        lo, hi, med = (bundle.lower[:, k_unknown], bundle.upper[:, k_unknown],
                       bundle.median[:, k_unknown])
    else:
        k_lo = bundle.lower.max(1)                       # max_c lower_c
        k_hi = bundle.upper.max(1)                       # max_c upper_c
        lo, hi, med = 1 - k_hi, 1 - k_lo, 1 - bundle.median.max(1)
    width = bundle.width
    cards = np.stack([[c.card_optimistic for c in bundle.contours(i, alphas)]
                      for i in range(len(lo))])          # (N, A) optimistic cardinality
    return {
        "set_lower": lo,
        "set_upper": hi,
        "set_mid": (lo + hi) / 2,                        # "average of the interval"
        "set_median": med,                               # median of the sampled set
        "width_novel": hi - lo,
        "width_mean": width.mean(1),
        "card_mean": cards.mean(1),                      # how many classes the set straddles
    }


FEATURE_NAMES = ["set_lower", "set_upper", "set_mid", "set_median",
                 "width_novel", "width_mean", "card_mean"]


def interval_features(scores):
    """The set-derived scores as an (N, F) feature matrix for learned adapters."""
    return np.stack([scores[n] for n in FEATURE_NAMES], axis=1)


class CredalAdapter:
    """Post-hoc adapter over interval-valued features (the meeting's "cradle adapter").

    kind='logistic' or 'mlp'. fit() takes ONLY rows the caller passes -- the
    disclosed val split (kkc negatives + uuc_val positives); it never sees test
    rows. decision_function returns a higher = more novel score.
    """

    def __init__(self, kind="logistic", seed=0):
        assert kind in ("logistic", "mlp")
        self.kind, self.seed = kind, seed

    def fit(self, feats, y):
        X = torch.as_tensor(np.asarray(feats, dtype=np.float32))
        y = torch.as_tensor(np.asarray(y, dtype=np.float32))
        torch.manual_seed(self.seed)
        d = X.shape[1]
        self.model = (nn.Linear(d, 1) if self.kind == "logistic"
                      else nn.Sequential(nn.Linear(d, 16), nn.ReLU(),
                                         nn.Linear(16, 1)))
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=1e-4)
        for _ in range(300):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self.model(X).squeeze(-1), y)
            loss.backward()
            opt.step()
        return self

    def decision_function(self, feats):
        X = torch.as_tensor(np.asarray(feats, dtype=np.float32))
        with torch.no_grad():
            return self.model(X).squeeze(-1).numpy()

    def predict_proba(self, feats):
        """P(novel) in [0, 1] -- what calibration readouts must use (sigmoid of the
        raw decision function, which is a logit and unbounded)."""
        from scipy.special import expit
        return expit(self.decision_function(feats))


# ---------------------------------------------------------------------------
# 3. The K+1 head (eu_blindspot cell-6 protocol; canonical home now here)
# ---------------------------------------------------------------------------
class HeadDrop(nn.Module):
    """MLP 512-128-ReLU-Dropout(p)-(K+1); dropout inside enables the MC passes."""

    def __init__(self, d_in, hidden=128, k_total=3, p=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Dropout(p), nn.Linear(hidden, k_total))

    def forward(self, x):
        return self.net(x)


def train_kp1(world, seed=0, epochs=300, lr=1e-3, batch=128, hidden=128, p=0.5):
    """kkc -> outputs 0..K-1, every kuc id -> output K (learned unknown);
    trained on kkc+kuc train rows only."""
    K = len(world["kkc"])
    lut = torch.full((1024,), -1, dtype=torch.long)
    for k, c in enumerate(world["kkc"]):
        lut[c] = k
    for c in world["kuc"]:
        lut[c] = K
    ti = world["idx"]["train"]
    ds = TensorDataset(torch.from_numpy(world["X"][ti]).float(),
                       lut[torch.from_numpy(world["y"][ti]).long()])
    ld = DataLoader(ds, batch_size=batch, shuffle=True,
                    generator=torch.Generator().manual_seed(0))
    torch.manual_seed(seed)
    head = HeadDrop(world["X"].shape[1], hidden=hidden, k_total=K + 1, p=p)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        head.train()
        for xb, yb in ld:
            opt.zero_grad()
            F.cross_entropy(head(xb), yb).backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        acc = float((head(ds.tensors[0]).argmax(1) == ds.tensors[1]).float().mean())
    assert acc > 0.9, f"head untrained (acc {acc:.3f})"
    return head, lut, acc


def mc_stack(head, X, T=30):
    """(T, N, K+1) softmax passes with dropout active; restores mode."""
    was_training = head.training
    head.train()
    torch.manual_seed(0)                       # fixed mask sequence per call
    with torch.no_grad():
        out = torch.stack([F.softmax(head(X), dim=1) for _ in range(T)]).numpy()
    head.train(was_training)
    return out


# ---------------------------------------------------------------------------
# 4. Evaluation with named negative sets + tabular export
# ---------------------------------------------------------------------------
def single_scores(stacked, k_unknown, Z=None):
    """Single-valued baselines from the same sampled heads (higher = more novel).

    Z: deterministic eval-mode raw logits (N, K+1). Energy and MaxLogit are only
    defined on raw logits -- on log-probabilities logsumexp is identically 0, so
    they are skipped (with a note) when Z is not supplied.
    """
    mean = stacked.mean(0)
    tu = -(mean * np.log(np.clip(mean, 1e-12, None))).sum(1)
    au = -(stacked * np.log(np.clip(stacked, 1e-12, None))).sum(-1).mean(0)
    logits = np.log(np.clip(mean, 1e-12, None))
    top2 = np.sort(logits, axis=1)
    out = {
        "EU_bald": tu - au,
        "TU": tu,
        "AU": au,
        "neg_margin": -(top2[:, -1] - top2[:, -2]),
        "neg_msp": -mean.max(1),
        "P_unknown": mean[:, k_unknown],
    }
    if Z is not None:
        lse = (np.log(np.exp(Z - Z.max(1, keepdims=True)).sum(1)) + Z.max(1))
        out["neg_energy"] = -lse              # free energy; higher = more novel
        out["neg_maxlogit"] = -Z.max(1)
    return out


def per_class_aurocs(score, y, uuc_classes, known_mask):
    out = {}
    for c in uuc_classes:
        m = y == c
        if m.sum():
            out[int(c)] = auroc_fast(score[known_mask], score[m])
    return out


def ece(conf, correct, n_bins=15):
    """Equal-mass-binned expected calibration error."""
    order = np.argsort(conf)
    edges = np.array_split(order, n_bins)
    total = 0.0
    for ix in edges:
        if len(ix):
            total += len(ix) / len(conf) * abs(conf[ix].mean() - correct[ix].mean())
    return float(total)


def nll(conf, correct, eps=1e-12):
    conf = np.nan_to_num(np.asarray(conf, dtype=np.float64), nan=0.5,
                         neginf=eps, posinf=1 - eps)
    p = np.clip(conf, eps, 1 - eps)
    return float(-(correct * np.log(p) + (1 - correct) * np.log(1 - p)).mean())


def run_world(world_seed, T=30, epochs=300, kuc=1, uuc_val=2, uuc_test=5,
              feat_path="cache/features/cifar10_test.npz", alphas=DEFAULT_ALPHAS):
    """One DynamicOSS world: head, epistemic sets, derived + learned + baseline
    scores, evaluated on the test split under the named-negative discipline."""
    world = make_oss(feat_path, kuc=kuc, uuc_val=uuc_val, uuc_test=uuc_test,
                     seed=world_seed)
    names = cifar10_names(range(10))
    head, lut, acc = train_kp1(world, seed=0, epochs=epochs)
    K = len(world["kkc"])
    k_unknown = K

    Xte = torch.from_numpy(world["X"][world["idx"]["test"]]).float()
    yte = world["y"][world["idx"]["test"]]
    stacked = mc_stack(head, Xte, T=T)
    with torch.no_grad():
        Z = head(Xte).numpy()                     # deterministic eval-mode raw logits
    bundle = EpistemicSetRepresenter().build(stacked)
    scores = set_scores(bundle, k_unknown=k_unknown, alphas=alphas)
    scores.update(single_scores(stacked, k_unknown, Z=Z))

    known = np.isin(yte, world["kkc"])
    kuc_m = np.isin(yte, world["kuc"])
    uuc_m = np.isin(yte, world["uuc"])
    uuc_test_classes = sorted(set(yte[uuc_m].tolist()))
    pred_ok = stacked.mean(0).argmax(1) == lut[torch.as_tensor(yte)].numpy()
    correct = pred_ok[known]

    # CredalAdapter: fit on the val split only -- kkc val negatives + uuc_val
    # positives (disclosed: uuc_val is a training signal here, never a threshold
    # source; test rows unseen). Calibration uses adapter PROBABILITIES.
    Xva = torch.from_numpy(world["X"][world["idx"]["val"]]).float()
    yva = world["y"][world["idx"]["val"]]
    stacked_val = mc_stack(head, Xva, T=T)
    bundle_val = EpistemicSetRepresenter().build(stacked_val)
    scores_val = set_scores(bundle_val, k_unknown=k_unknown, alphas=alphas)
    Ftr = interval_features(scores_val)
    ytr = (~np.isin(yva, world["kkc"] + world["kuc"])).astype(int)   # uuc_val = 1
    cal = {"msp_ece": ece(stacked.mean(0).max(1)[known], correct),
           "msp_nll": nll(stacked.mean(0).max(1)[known], correct)}
    for kind in ("logistic", "mlp"):
        ad = CredalAdapter(kind=kind).fit(Ftr, ytr)
        scores[f"credal_adapter_{kind}"] = ad.decision_function(interval_features(scores))
        adapter_conf = 1 - ad.predict_proba(interval_features(scores))[known]
        cal[f"adapter_{kind}_ece"] = ece(adapter_conf, correct)
        cal[f"adapter_{kind}_nll"] = nll(adapter_conf, correct)

    per_method, per_class = {}, {}
    for name, s in scores.items():
        per_method[name] = {
            "auroc_pos_uuc_neg_kkc": auroc_fast(s[known], s[uuc_m]),
            "auroc_pos_kuc_neg_kkc_DISCLOSED_trained_on": auroc_fast(s[known], s[kuc_m]),
        }
        pc = per_class_aurocs(s, yte, uuc_test_classes, known)
        per_method[name]["macro_auroc"] = float(np.mean(list(pc.values())))
        per_method[name]["worst_class_auroc"] = float(min(pc.values()))
        per_class[name] = {names[c]: v for c, v in pc.items()}

    rows = pd.DataFrame({
        "world_seed": world_seed, "row": np.arange(len(yte)),
        "label": yte,
        "class_name": [names[int(c)] for c in yte],
        "group": np.where(known, "kkc", np.where(kuc_m, "kuc", "uuc")),
        "known_correct": np.where(known, pred_ok, np.nan),
        **{k: v for k, v in scores.items()},
    })
    meta = dict(world_seed=world_seed, kkc=[names[c] for c in world["kkc"]],
                kuc=[names[c] for c in world["kuc"]],
                uuc_test=[names[c] for c in uuc_test_classes],
                head_train_acc=acc, T=T, epochs=epochs, n_test=int(len(yte)))
    return rows, per_method, per_class, cal, meta


def main(world_seeds=range(5), out_dir=OUT_DIR):
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, summaries, per_class_frames, cals, metas = [], [], [], [], []
    for ws in world_seeds:
        rows, per_method, per_class, cal, meta = run_world(world_seed=ws)
        all_rows.append(rows)
        cals.append(cal)
        metas.append(meta)
        for name, m in per_method.items():
            summaries.append(pd.DataFrame({"world_seed": ws, "method": name, **m},
                                          index=[0]))
            pc = per_class[name]
            per_class_frames.append(pd.DataFrame(
                {"world_seed": ws, "method": name,
                 "uuc_class": list(pc), "auroc": list(pc)}))
        print(f"world seed {ws}: kkc {meta['kkc']} kuc {meta['kuc']} "
              f"uuc_test {meta['uuc_test']} done", flush=True)

    rows_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.concat(summaries, ignore_index=True)
    per_class_df = pd.concat(per_class_frames, ignore_index=True)
    rows_df.to_csv(out_dir / "rows.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    per_class_df.to_csv(out_dir / "per_class.csv", index=False)

    agg = (summary_df.groupby("method")
           .agg(mean_auroc=("auroc_pos_uuc_neg_kkc", "mean"),
                sd_auroc=("auroc_pos_uuc_neg_kkc", "std"),
                mean_macro=("macro_auroc", "mean"),
                mean_worst=("worst_class_auroc", "mean"))
           .sort_values("mean_auroc", ascending=False).round(4))
    print("\n=== pooled AUROC (pos uuc | neg kkc), mean over world seeds ===")
    print(agg.to_string())

    cal_df = pd.DataFrame(cals).round(4)
    print("\n=== calibration on known test rows (mean over world seeds) ===")
    print(cal_df.mean().to_string())

    src = Path(__file__).read_bytes()
    manifest = dict(script_sha256=hashlib.sha256(src).hexdigest(),
                    world_seeds=list(world_seeds), alphas=list(DEFAULT_ALPHAS),
                    negative_set="kkc test rows (named per step-0 finding)",
                    adapter_fit="val split only: kkc val negatives + uuc_val positives",
                    caveat="scalar-score AUROC only; envelopes are not membership intervals",
                    worlds=metas)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    agg.to_csv(out_dir / "summary_agg.csv")

    # comparison figure: pooled AUROC mean +- sd per method, set-derived highlighted
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = agg.sort_values("mean_auroc")
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(order) + 1.5))
    colors = ["#1f77b4" if m.startswith(("set_", "width_", "card_", "credal_")) else "#999999"
              for m in order.index]
    ax.errorbar(order["mean_auroc"], range(len(order)),
                xerr=order["sd_auroc"].fillna(0), fmt="o", color="k", ms=3, zorder=2)
    ax.scatter(order["mean_auroc"], range(len(order)), c=colors, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order.index)
    ax.set_xlabel("AUROC (pos uuc | neg kkc), mean ± sd over 5 world seeds")
    ax.set_title("Epistemic-set readouts (blue) vs single-score baselines (grey)")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=150)
    print("\nwritten:", out_dir)


if __name__ == "__main__":
    main()
