# credal_head.py -- relative-likelihood credal head: baseline arm for the epistemic-set line.
#
# Question this arm answers (COLLABORATOR_PLAN 1 Nov evening addendum, open item):
#   does a *relative-likelihood* credal predictor -- {h : L(h)/L(h_ML) >= tau} built from
#   the ensemble, the DRE-free stand-in locked OUT of the AISTATS paper and IN scope for
#   the RL paper -- support any claim on the CVPR epistemic-set data that the sampled
#   envelope (EpistemicSetRepresenter min/max, i.e. width_novel / W_unk) cannot?
#
# Construction (row-level, ensemble-based):
#   hypotheses h = predictive distributions on the K+1 simplex of one row; log-likelihood
#   under log-pooling of the T member posteriors p_t,
#       log L(h) = sum_t (1/T) sum_y p_t(y) log h(y) = sum_y p_bar(y) log h(y),
#   is maximized at h_ML = p_bar (the ensemble mean posterior), and
#       log L(h) - log L(h_ML) = -KL(p_bar || h),
#   so the relative-likelihood set at level tau is the KL ball
#       C_tau(row) = {h : KL(p_bar || h) <= r},  r = -log tau.
#   Per-class extrema over C_tau: putting h(y0) = q and rescaling the rest of p_bar gives
#   the minimizer of KL subject to h(y0) = q, with KL = D_ber(p_bar(y0) || q) (binary KL);
#   the bounds q_lo/q_hi invert D_ber(., .) = r by bisection. Closed forms at the edges:
#   p_bar(y0) = 0 -> (0, 1 - tau);  p_bar(y0) = 1 -> (tau, 1).
#
# COLLAPSE THEOREM (load-bearing disclosure): log-likelihood pooling is linear in the
# member posteriors, so C_tau depends on the ensemble ONLY through p_bar. This arm
# carries zero member-disagreement information; its width is a deterministic single-peaked
# function g_tau(p_bar(unk)). PRED below is written against that collapse. The
# disagreement-aware credal set on the same stack already exists -- it is the min/max
# envelope (width_novel), so the two arms are exactly the likelihood-only vs hull-only
# halves of the comparison.
#
# SYMMETRY (found by a unit test before any scoring; see PRED_AMENDMENT): the binary KL
# obeys D(p||q) = D(1-p||1-q), so the unknown-slot width g_tau(p_bar(unk)) is symmetric
# about 1/2 -- floor 1 - tau at BOTH ends, peak sqrt(1 - tau^2) at 1/2. The arm is an
# ambiguity readout of the mean unknown probability: a certainly-novel row and a
# certainly-known row get the same width.
#
# Scalar choice (width_from_bounds): primary = unknown-slot gap hi[K] - lo[K], directly
# comparable to width_novel (same slot); secondary = mean per-class gap, comparable to
# width_mean. Max-class gap is not slot-comparable and is not used.
#
# Discipline (per the 1 Nov addendum + step-0/1 rules):
#   * tau in {0.90, 0.95} FIXED BY CONVENTION a priori -- no calibration on any rows, so
#     "no uuc rows touch the likelihood threshold calibration" holds by construction
#     (unit-tested: the bounds are a pure function of (stacked, tau)).
#   * negative set named: kkc test rows. kuc readouts disclosed as trained-on.
#   * matched-AU bins: interior edges from kkc rows only; every row is assigned by those
#     edges (outer edges infinite), disclosed here.
#   * PRED is frozen below and written to results/credal_head/pred.json BEFORE any score
#     is computed; the driver aborts if that file already exists with different content.
#   * shared per-row scores are verified against the delivered
#     results/epistemic_sets/rows.csv (same seeds, same protocol) before analysis.
#
# Run:  OMP_NUM_THREADS=1 python credal_head.py
# Out:  results/credal_head/{pred.json, pred_amendment.json, rows.csv, summary.csv,
#                            summary_agg.csv, matched_au.csv, near_far.csv,
#                            correlations.csv, manifest.json, verdict.json}
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from example import make_oss
from example_wrn50 import auroc_fast
from epistemic_sets import (DEFAULT_ALPHAS, EpistemicSetRepresenter, mc_stack,
                            per_class_aurocs, set_scores, single_scores, train_kp1)
from metrics import cifar10_names

OUT_DIR = Path("results/credal_head")
DELIVERED_ROWS = Path("results/epistemic_sets/rows.csv")
TAUS = (0.90, 0.95)          # fixed by convention; assert no code path calibrates them
PRIMARY_TAU = 0.90
FEAT_PATH = "cache/features/cifar10_test.npz"
WORLD_SEEDS = tuple(range(5))
NEAR_FAR_K, NEAR_FAR_Q = 5, 0.95   # example.tag_and_report_budget recipe

# ---------------------------------------------------------------------------
# PRED -- frozen before any scoring on this line (the hint's rule: if you cannot
# write a differing prediction, the arm is not informative). Reference numbers are
# the delivered epistemic_sets aggregates (mean over the same 5 world seeds).
# ---------------------------------------------------------------------------
PRED = {
    "written_before_scoring": True,
    "construction": "KL ball around p_bar, r = -log tau (see module docstring); "
                    "ensemble enters only through p_bar (collapse theorem)",
    "P1_direction": {
        "claim": "mean credal_width_unk on uuc > on kkc, ratio >= 5 "
                 "(theory: kkc rows have p_bar(unk) ~ 1e-5..1e-3 -> width ~ 0.005-0.02; "
                 "uuc rows mid-band -> width ~ 0.15-0.35 at tau=0.90)",
        "ratio_band": [10, 60],
    },
    "P2_collapse_corollary": {
        "claim": "pooled AUROC(credal_width_unk, tau=0.90) < AUROC(P_unknown)=0.946 "
                 "and < AUROC(set_median)=0.950 by >= 0.005 each (mean over seeds), "
                 "because g_tau is single-peaked in p_bar(unk) and discards "
                 "disagreement; confident-novel rows (p_bar(unk)->1) shrink toward 0.13",
        "auroc_band_t090": [0.85, 0.95],
    },
    "P3_vs_envelope": {
        "claim": "|AUROC(cred90) - AUROC(width_novel=0.916)| <= 0.04: statistically "
                 "indistinguishable from the hull width -- adds nothing the envelope "
                 "does not have; a beat of EITHER bound is informative and gets escalated"
    },
    "P4_dependence": {
        "claim": "Spearman(cred90, P_unknown) >= 0.6 over all test rows, and "
                 "Spearman(cred90, EU_bald) < Spearman(width_novel, EU_bald): the arm "
                 "couples to the mean posterior, not to disagreement"
    },
    "P5_near_far_crossover": {
        "claim": "with near/far tags per example.tag_and_report_budget recipe: "
                 "far-tagged uuc rows have HIGHER median P_unknown but LOWER-or-equal "
                 "median credal_width_unk than near-tagged rows -- the distinctive "
                 "mean-collapse signature (g falls as p_bar(unk) -> 1)"
    },
    "P6_tau_invariance": {
        "claim": "|AUROC(tau=0.95) - AUROC(tau=0.90)| <= 0.02"
    },
    "P7_matched_au": {
        "claim": "within AU deciles (kkc-only edges), mean per-bin AUROC(cred90) < "
                 "mean per-bin AUROC(P_unknown); both remain > 0.7 mean per bin"
    },
    "verdict_rule": "if P2 holds -> the credal-head arm supports no claim beyond the "
                    "existing set readouts (question-1 answer: nothing; keep as a "
                    "baseline row). If P2 fails -> informative, escalate to the "
                    "CVPR-line decision.",
}

# Amendment made BEFORE any real-data scoring of this arm. PRED above stays verbatim
# and every original item is still scored as written.
PRED_AMENDMENT = {
    "date": "2026-11-09",
    "made_before_real_data_scoring": True,
    "trigger": "unit test test_credal_head.py::test_tiny_mass_sits_at_the_floor (synthetic "
               "inputs only) showed the P1 rationale computed the small-p_bar bound with "
               "the KL direction reversed (D(q||p) instead of D(p||q)); no real-data "
               "output of this arm existed when this amendment was written",
    "corrected_arithmetic": [
        "g_tau(p) = q_hi - q_lo satisfies g(p) = g(1 - p) exactly (binary-KL symmetry); "
        "floor 1 - tau at p -> 0 AND p -> 1; peak sqrt(1 - tau^2) at p = 0.5 (0.436 at "
        "tau=0.90, 0.312 at tau=0.95); strictly increasing on [0, 0.5]",
        "consequence: credal_width_unk is strictly decreasing in |p_bar(unk) - 0.5|, so "
        "AUROC(credal_width_unk) == AUROC(-|P_unknown - 0.5|) -- an identity, checked as "
        "implementation sanity, not a prediction. A certainly-novel row (p_bar(unk) -> 1) "
        "gets the same width as a certainly-known row (p_bar(unk) -> 0)",
    ],
    "P1_status": "kept verbatim and scored as written; unattainable by construction "
                 "(ratio of means <= peak/floor = 4.36 at tau=0.90), so it will FAIL",
    "P1_amended": {"claim": "mean credal_width_unk(uuc) > mean(kkc); kkc mean in "
                            "[0.100, 0.120] (at the 1 - tau floor); ratio in [1.2, 4.36]"},
    "P2_note": "parenthetical corrected: confident-novel rows shrink toward the 1 - tau "
               "floor (0.10), the same width as confident-known rows -- not 0.13. The P2 "
               "inequality claims are unchanged",
    "unchanged": "P2-P7 claims and the verdict rule. A low-side P3 failure (arm worse "
                 "than the envelope) is not an escalation trigger",
}

# Post-run analysis-code fixes (Rule 3: bugs only, logged). No PRED item or score changes.
ANALYSIS_FIXES = [
    "2026-11-09, after the first full run: the amendment's identity check compared "
    "against the float32 P_unknown column, whose |p - 1/2| has ~10-20% ties per world; "
    "ties moved the check by up to 2.5e-4 (flagged FAIL). Switched to the head's own "
    "float64 p_bar (p_bar_unk_f64 column): max abs diff 0.0. All scores, AUROCs and "
    "every PRED verdict were identical across the two runs (bitwise-deterministic stack).",
]

# ---------------------------------------------------------------------------
# 1. The credal head: relative-likelihood set -> per-class lower/upper bounds
# ---------------------------------------------------------------------------
def _binary_kl(p, q):
    """D_ber(p || q) = p log(p/q) + (1-p) log((1-p)/(1-q)); 0 log 0 = 0. Vectorized."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-300, 1.0 - 1e-16)
    t1 = np.where(p > 0, p * (np.log(np.maximum(p, 1e-300)) - np.log(q)), 0.0)
    t2 = np.where(p < 1, (1 - p) * (np.log(1 - np.minimum(p, 1 - 1e-16))
                                    - np.log1p(-q)), 0.0)
    return t1 + t2


def _invert_bkl_upper(p0, r, iters=100):
    """q_hi: root of D_ber(p0 || q) = r on (p0, 1); D is 0 at q=p0 and inf at q=1."""
    lo = np.array(p0, dtype=np.float64, copy=True)
    hi = np.ones_like(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        side = _binary_kl(p0, mid) < r          # still below r -> root is above
        lo = np.where(side, mid, lo)
        hi = np.where(side, hi, mid)
    return 0.5 * (lo + hi)


def _invert_bkl_lower(p0, r, iters=100):
    """q_lo: root of D_ber(p0 || q) = r on (0, p0); D is inf at q=0 and 0 at q=p0."""
    hi = np.array(p0, dtype=np.float64, copy=True)
    lo = np.zeros_like(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        side = _binary_kl(p0, mid) > r          # mid too far below p0 -> root is above
        lo = np.where(side, mid, lo)
        hi = np.where(side, hi, mid)
    return 0.5 * (lo + hi)


def kl_ball_bounds(p_bar, tau):
    """Per-class extrema of the relative-likelihood set {h : KL(p_bar || h) <= -log tau}.

    p_bar: (N, K) mean posteriors (rows on the simplex, possibly with exact 0/1).
    Returns (p_lo, p_hi), both (N, K), with p_lo <= p_bar <= p_hi asserted.
    Exact edges: p_bar=0 -> (0, 1 - tau); p_bar=1 -> (tau, 1).
    """
    assert 0.0 < tau < 1.0, "tau is a likelihood ratio in (0, 1), fixed by convention"
    p_bar = np.asarray(p_bar, dtype=np.float64)
    r = -np.log(tau)
    lo = np.zeros_like(p_bar)
    hi = np.ones_like(p_bar)
    interior = (p_bar > 0) & (p_bar < 1)
    if interior.any():
        p0 = p_bar[interior]
        hi[interior] = _invert_bkl_upper(p0, r)
        lo[interior] = _invert_bkl_lower(p0, r)
    # exact closed forms where the binary KL has a one-sided infinite/zero branch
    hi = np.where(p_bar <= 0, 1.0 - tau, hi)
    hi = np.where(p_bar >= 1, 1.0, hi)
    lo = np.where(p_bar >= 1, tau, lo)
    lo = np.where(p_bar <= 0, 0.0, lo)
    # p_bar values that are merely tiny but nonzero got an exact-0/1 overwrite only at
    # the true edges; the interior path handles everything in (0, 1)
    assert (lo <= p_bar + 1e-12).all() and (p_bar <= hi + 1e-12).all()
    assert (lo >= 0).all() and (hi <= 1).all() and (hi - lo >= 0).all()
    return lo, hi


def credal_head_width(ens, tau, k_unknown=-1):
    """Relative-likelihood credal head from the ensemble stack.

    ens: (T, N, K) member posteriors (or raw logits -- softmaxed per member first).
    k_unknown: learned-unknown output of the K+1 head (last column, train_kp1 convention).
    Returns dict with p_bar, per-class bounds, and the width scalars. Pure function of
    (ens, tau): no calibration, no side channels (the tau hygiene assertion).
    """
    ens = np.asarray(ens, dtype=np.float64)
    if not np.allclose(ens.sum(-1), 1.0, atol=1e-4):       # logits in -> probs
        e = ens - ens.max(-1, keepdims=True)
        ens = np.exp(e) / np.exp(e).sum(-1, keepdims=True)
    assert ens.ndim == 3 and (ens >= 0).all()
    p_bar = ens.mean(0)
    lo, hi = kl_ball_bounds(p_bar, tau)
    out = {"p_bar": p_bar, f"p_lo_t{int(round(tau*100)):03d}": lo,
           f"p_hi_t{int(round(tau*100)):03d}": hi}
    out.update(width_from_bounds(lo, hi, k_unknown=k_unknown))
    return out


def width_from_bounds(p_lo, p_hi, k_unknown=None):
    """Bounds -> width scalars. Primary: unknown-slot gap (slot-matched to width_novel).
    Secondary: mean per-class gap (matched to width_mean). Justification in the module
    docstring; max-class gap is deliberately not used (not slot-comparable)."""
    p_lo, p_hi = np.asarray(p_lo), np.asarray(p_hi)
    gap = p_hi - p_lo
    out = {"credal_width_mean": gap.mean(1)}
    if k_unknown is not None:
        out["credal_width_unk"] = gap[:, k_unknown]
    return out


# ---------------------------------------------------------------------------
# 2. Near/far tags (example.tag_and_report_budget recipe, exact L2 via sklearn --
#    same values as faiss IndexFlatL2, avoids the faiss/torch threadpool deadlock)
# ---------------------------------------------------------------------------
def near_far_tags(world, k=NEAR_FAR_K, q=NEAR_FAR_Q):
    """Tag every novel class (kuc + uuc) near/far by median normalized kNN distance to
    the kkc train classes. tau_c: q-quantile of the k-th-neighbour distance within each
    kkc train class (self excluded). cutoff: 1.0 if any median <= 1.0 else the median
    of the medians (verbatim rule from example.tag_and_report_budget)."""
    from sklearn.neighbors import NearestNeighbors
    X, y, kkc = world["X"].astype(np.float32), world["y"], world["kkc"]
    tr = world["idx"]["train"]
    tr = tr[np.isin(y[tr], kkc)]
    taus, class_nn = {}, {}
    for c in kkc:
        rows = X[tr][y[tr] == c]
        nn = NearestNeighbors(n_neighbors=k + 1).fit(rows)
        D, _ = nn.kneighbors(rows, k + 1)                  # self (dist 0) dropped
        taus[c] = float(np.quantile(D[:, -1], q))
        class_nn[c] = NearestNeighbors(n_neighbors=k).fit(rows)
    novel = sorted(set(world["kuc"]) | set(world["uuc"]))
    meds = {}
    for c in novel:
        D = np.stack([class_nn[a].kneighbors(X[y == c], k)[0][:, -1] / taus[a]
                      for a in kkc], axis=1)
        meds[c] = float(np.median(D.min(axis=1)))
    cutoff = 1.0 if any(m <= 1.0 for m in meds.values()) else float(np.median(list(meds.values())))
    tag = {c: ("near" if meds[c] <= cutoff else "far") for c in novel}
    return tag, meds, cutoff


# ---------------------------------------------------------------------------
# 3. Driver: same worlds/protocol as epistemic_sets, plus the credal-head arm
# ---------------------------------------------------------------------------
ARMS_FOR_TABLE = ("credal_width_unk_t090", "credal_width_unk_t095",
                  "credal_width_mean_t090", "width_novel", "P_unknown",
                  "set_median", "EU_bald", "TU", "neg_msp")
MATCHED_ARMS = ("credal_width_unk_t090", "credal_width_unk_t095", "width_novel",
                "P_unknown", "EU_bald", "TU")


def au_bins(au, known_mask, n_bins=10):
    """AU bin per row. Interior quantile edges come from the known (kkc) rows ONLY;
    novel rows are assignees, never edge sources. Outer edges are infinite, so every
    row lands in 0..n_bins-1."""
    au = np.asarray(au, dtype=np.float64)
    edges = np.quantile(au[np.asarray(known_mask, bool)],
                        np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, au, side="right")


def run_world(world_seed, T=30, epochs=300):
    """One world: rebuild the delivered stack (verified below), add the credal arm."""
    world = make_oss(FEAT_PATH, kuc=1, uuc_val=2, uuc_test=5, seed=world_seed)
    names = cifar10_names(range(10))
    head, lut, acc = train_kp1(world, seed=0, epochs=epochs)
    K = len(world["kkc"])
    Xte = torch.from_numpy(world["X"][world["idx"]["test"]]).float()
    yte = world["y"][world["idx"]["test"]]
    stacked = mc_stack(head, Xte, T=T)
    with torch.no_grad():
        Z = head(Xte).numpy()
    bundle = EpistemicSetRepresenter().build(stacked)
    scores = set_scores(bundle, k_unknown=K, alphas=DEFAULT_ALPHAS)
    scores.update(single_scores(stacked, K, Z=Z))

    cred = {int(round(tau * 100)): credal_head_width(stacked, tau, k_unknown=K)
            for tau in TAUS}
    scores["p_bar_unk_f64"] = cred[90]["p_bar"][:, K]   # P_unknown is the float32 copy
    for pct, c in cred.items():
        scores[f"credal_width_unk_t{pct:03d}"] = c["credal_width_unk"]
        scores[f"credal_width_mean_t{pct:03d}"] = c["credal_width_mean"]
        scores[f"credal_p_lo_unk_t{pct:03d}"] = c[f"p_lo_t{pct:03d}"][:, K]
        scores[f"credal_p_hi_unk_t{pct:03d}"] = c[f"p_hi_t{pct:03d}"][:, K]

    known = np.isin(yte, world["kkc"])
    kuc_m = np.isin(yte, world["kuc"])
    uuc_m = np.isin(yte, world["uuc"])
    uuc_classes = sorted(set(yte[uuc_m].tolist()))
    tag, meds, cutoff = near_far_tags(world)
    row_tag = np.array([tag.get(int(c), "kkc") for c in yte])

    rows = pd.DataFrame({
        "world_seed": world_seed, "row": np.arange(len(yte)), "label": yte,
        "class_name": [names[int(c)] for c in yte],
        "group": np.where(known, "kkc", np.where(kuc_m, "kuc", "uuc")),
        "novel_tag": np.where(known, "kkc", row_tag),
        **{k: v for k, v in scores.items()},
    })

    per_method = {}
    for name in ARMS_FOR_TABLE:
        s = scores[name]
        pc = per_class_aurocs(s, yte, uuc_classes, known)
        per_method[name] = {
            "auroc_pos_uuc_neg_kkc": auroc_fast(s[known], s[uuc_m]),
            "auroc_pos_kuc_neg_kkc_DISCLOSED_trained_on": auroc_fast(s[known], s[kuc_m]),
            "macro_auroc": float(np.mean(list(pc.values()))),
            "worst_class_auroc": float(min(pc.values())),
        }

    # matched-AU: interior decile edges from kkc rows only; every row assigned (outer inf)
    bins = au_bins(scores["AU"], known)
    matched = []
    for b in range(10):
        m = bins == b
        if m.sum() and known[m].any() and uuc_m[m].any():
            matched.append({"world_seed": world_seed, "bin": b,
                            "n_kkc": int(known[m].sum()), "n_uuc": int(uuc_m[m].sum()),
                            **{a: auroc_fast(scores[a][m & known], scores[a][m & uuc_m])
                               for a in MATCHED_ARMS}})

    from scipy.stats import spearmanr
    cors = {"world_seed": world_seed,
            "spearman_cred90_vs_p_unknown": float(spearmanr(
                scores["credal_width_unk_t090"], scores["P_unknown"])[0]),
            "spearman_cred90_vs_eu_bald": float(spearmanr(
                scores["credal_width_unk_t090"], scores["EU_bald"])[0]),
            "spearman_width_novel_vs_eu_bald": float(spearmanr(
                scores["width_novel"], scores["EU_bald"])[0])}

    near_far = [{"world_seed": world_seed, "class": int(c), "name": names[int(c)],
                 "group": "kuc" if c in set(world["kuc"]) else "uuc",
                 "median_norm_dist": meds[c], "tag": tag[c]}
                for c in sorted(meds)]

    meta = dict(world_seed=world_seed, kkc=[names[c] for c in world["kkc"]],
                kuc=[names[c] for c in world["kuc"]],
                uuc_test=[names[c] for c in uuc_classes], head_train_acc=acc,
                near_far_cutoff=cutoff, T=T, epochs=epochs)
    return rows, per_method, matched, cors, near_far, meta


def verify_against_delivered(rows_df):
    """Same seeds and protocol must reproduce the delivered epistemic_sets rows; any
    drift means the comparison targets are not the same objects. Both sides go through
    a CSV round trip so float32 columns compare at their serialized precision."""
    import io
    buf = io.StringIO()
    rows_df.to_csv(buf, index=False)
    buf.seek(0)
    got = pd.read_csv(buf)
    ref = pd.read_csv(DELIVERED_ROWS)
    ref = ref[ref.world_seed.isin(got.world_seed.unique())]
    keys = ["world_seed", "row"]
    m = got.merge(ref, on=keys, suffixes=("", "_ref"))
    assert len(m) == len(ref) == len(got), "row sets differ from the delivered run"
    for c in ("label", "class_name", "group"):
        assert (m[c] == m[f"{c}_ref"]).all(), f"identity column {c} differs"
    numeric = [c for c in ref.columns
               if c in got.columns and c not in keys + ["label", "class_name", "group"]
               and pd.api.types.is_numeric_dtype(ref[c])]
    worst = 0.0
    for c in numeric:
        a, b = m[c].to_numpy(float), m[f"{c}_ref"].to_numpy(float)
        assert np.allclose(a, b, rtol=1e-5, atol=1e-9, equal_nan=True), \
            f"drift vs delivered rows in {c} (max abs diff {np.nanmax(np.abs(a - b)):.2e})"
        worst = max(worst, float(np.nanmax(np.abs(a - b))))
    return worst


def evaluate_pred(summary_df, matched_df, cors_df, rows_df):
    """Score PRED items; the verdict rule decides whether the arm is informative."""
    def auroc(arm):
        return float(summary_df[summary_df.method == arm]["auroc_pos_uuc_neg_kkc"].mean())

    uuc = rows_df[rows_df.group == "uuc"]
    kkc = rows_df[rows_df.group == "kkc"]
    ratio = float(uuc["credal_width_unk_t090"].mean() / kkc["credal_width_unk_t090"].mean())
    cred90, cred95 = auroc("credal_width_unk_t090"), auroc("credal_width_unk_t095")
    p_unk, set_med, w_nov = auroc("P_unknown"), auroc("set_median"), auroc("width_novel")

    # matched AU: mean over bins within a world, then mean over worlds (unweighted bins)
    per_world = matched_df.groupby("world_seed")[list(MATCHED_ARMS)].mean()
    mean_bin = per_world.mean()
    worst_bin = matched_df[list(MATCHED_ARMS)].min()

    near, far = uuc[uuc.novel_tag == "near"], uuc[uuc.novel_tag == "far"]
    p5 = {"n_near_rows": int(len(near)), "n_far_rows": int(len(far))}
    if len(near) and len(far):
        p5.update(med_p_unknown_near=float(near.P_unknown.median()),
                  med_p_unknown_far=float(far.P_unknown.median()),
                  med_cred90_near=float(near.credal_width_unk_t090.median()),
                  med_cred90_far=float(far.credal_width_unk_t090.median()))
        p5["pass"] = bool(p5["med_p_unknown_far"] >= p5["med_p_unknown_near"]
                          and p5["med_cred90_far"] <= p5["med_cred90_near"])
    else:
        p5["pass"] = None                         # untestable: one tag absent on uuc rows

    s_pu = float(cors_df.spearman_cred90_vs_p_unknown.mean())
    s_ce = float(cors_df.spearman_cred90_vs_eu_bald.mean())
    s_we = float(cors_df.spearman_width_novel_vs_eu_bald.mean())
    kkc_mean = float(kkc["credal_width_unk_t090"].mean())

    # amendment identity (implementation sanity): against the head's own float64 p_bar.
    # The float32 P_unknown column has ~10-20% tied values after |p - 1/2|, which moved
    # the first run's check by up to 2.5e-4 (ties count 1/2 in AUROC).
    ident = []
    for ws in sorted(rows_df.world_seed.unique()):
        k_w, u_w = kkc[kkc.world_seed == ws], uuc[uuc.world_seed == ws]
        amb = lambda d: -np.abs(d.p_bar_unk_f64.to_numpy(np.float64) - 0.5)
        ident.append((auroc_fast(k_w.credal_width_unk_t090, u_w.credal_width_unk_t090),
                      auroc_fast(amb(k_w), amb(u_w))))
    ident = np.array(ident)
    v = {
        "P1_ratio_uuc_over_kkc": {"value": ratio, "pass": bool(ratio >= 5),
                                  "pred_band": PRED["P1_direction"]["ratio_band"],
                                  "note": "original, scored as written; unattainable "
                                          "(see PRED_AMENDMENT)"},
        "P1_amended": {"ratio": ratio, "kkc_mean": kkc_mean,
                       "pass": bool(1.2 <= ratio <= 4.36 and 0.100 <= kkc_mean <= 0.120)},
        "identity_ambiguity_auroc": {
            "cred90": float(ident[:, 0].mean()), "neg_abs_p_bar_minus_half":
            float(ident[:, 1].mean()), "max_abs_diff": float(np.abs(ident[:, 0]
                                                                    - ident[:, 1]).max()),
            "pass": bool(np.abs(ident[:, 0] - ident[:, 1]).max() <= 1e-4)},
        "P2a_below_p_unknown": {"value": cred90, "ref": p_unk,
                                "pass": bool(cred90 < p_unk - 0.005)},
        "P2b_below_set_median": {"value": cred90, "ref": set_med,
                                 "pass": bool(cred90 < set_med - 0.005)},
        "P3_within_envelope_band": {"value": cred90, "ref_width_novel": w_nov,
                                    "abs_gap": abs(cred90 - w_nov),
                                    "pass": bool(abs(cred90 - w_nov) <= 0.04)},
        "P4_dependence": {"spearman_cred90_p_unknown": s_pu,
                          "spearman_cred90_eu_bald": s_ce,
                          "spearman_width_novel_eu_bald": s_we,
                          "pass": bool(s_pu >= 0.6 and s_ce < s_we)},
        "P5_near_far_crossover": p5,
        "P6_tau_invariance": {"t090": cred90, "t095": cred95,
                              "abs_gap": abs(cred95 - cred90),
                              "pass": bool(abs(cred95 - cred90) <= 0.02)},
        "P7_matched_au": {"mean_bin": {a: float(mean_bin[a]) for a in MATCHED_ARMS},
                          "worst_bin": {a: float(worst_bin[a]) for a in MATCHED_ARMS},
                          "pass": bool(mean_bin["credal_width_unk_t090"]
                                       < mean_bin["P_unknown"]
                                       and mean_bin["credal_width_unk_t090"] > 0.7
                                       and mean_bin["P_unknown"] > 0.7)},
    }
    v["overall"] = ("not informative beyond existing readouts -- keep as baseline row"
                    if v["P2a_below_p_unknown"]["pass"] and v["P2b_below_set_median"]["pass"]
                    else "INFORMATIVE: beats an existing readout -- escalate to the "
                         "CVPR-line decision")
    return v


def main(world_seeds=WORLD_SEEDS, out_dir=OUT_DIR):
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "pred.json"
    if pred_path.exists():                       # PRED must be the frozen first word
        assert json.loads(pred_path.read_text()) == PRED, \
            "results already exist under a different PRED; inspect before overwriting"
    else:
        pred_path.write_text(json.dumps(PRED, indent=2))
        print("PRED frozen to", pred_path, "-- before any scoring")
    amend_path = out_dir / "pred_amendment.json"
    if amend_path.exists():
        assert json.loads(amend_path.read_text()) == PRED_AMENDMENT, \
            "results already exist under a different PRED_AMENDMENT"
    else:
        amend_path.write_text(json.dumps(PRED_AMENDMENT, indent=2))
        print("PRED_AMENDMENT written to", amend_path, "-- before any scoring")

    all_rows, summaries, matched, cors, near_far, metas = [], [], [], [], [], []
    drift = 0.0
    for ws in world_seeds:
        rows, per_method, m, c, nf, meta = run_world(world_seed=ws)
        drift = max(drift, verify_against_delivered(rows))
        all_rows.append(rows)
        summaries.append(pd.DataFrame(
            [{"world_seed": ws, "method": name, **vals} for name, vals in per_method.items()]))
        matched.extend(m)
        cors.append(c)
        near_far.extend(nf)
        metas.append(meta)
        print(f"world seed {ws}: kkc {meta['kkc']} uuc_test {meta['uuc_test']} done "
              f"(near/far cutoff {meta['near_far_cutoff']:.2f})", flush=True)
    print(f"verified against delivered rows: max abs diff {drift:.2e}")

    rows_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.concat(summaries, ignore_index=True)
    matched_df = pd.DataFrame(matched)
    cors_df = pd.DataFrame(cors)
    near_far_df = pd.DataFrame(near_far)
    rows_df.to_csv(out_dir / "rows.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    matched_df.to_csv(out_dir / "matched_au.csv", index=False)
    cors_df.to_csv(out_dir / "correlations.csv", index=False)
    near_far_df.to_csv(out_dir / "near_far.csv", index=False)

    agg = (summary_df.groupby("method")
           .agg(mean_auroc=("auroc_pos_uuc_neg_kkc", "mean"),
                sd_auroc=("auroc_pos_uuc_neg_kkc", "std"),
                mean_worst=("worst_class_auroc", "mean"))
           .sort_values("mean_auroc", ascending=False).round(4))
    agg.to_csv(out_dir / "summary_agg.csv")
    print("\n=== pooled AUROC (pos uuc | neg kkc), mean over world seeds ===")
    print(agg.to_string())

    verdict = evaluate_pred(summary_df, matched_df, cors_df, rows_df)
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n=== PRED verdicts ===")
    for key, item in verdict.items():
        if key != "overall":
            print(f"{key}: {'PASS' if item.get('pass') else 'FAIL'}  {item}")
    print("OVERALL:", verdict["overall"])

    src = Path(__file__).read_bytes()
    manifest = dict(
        script_sha256=hashlib.sha256(src).hexdigest(),
        world_seeds=list(world_seeds), taus=list(TAUS), primary_tau=PRIMARY_TAU,
        tau_source="fixed by relative-likelihood convention; NOT calibrated on any rows",
        negative_set="kkc test rows (named per step-0 finding)",
        matched_au="interior decile edges from kkc rows only; all rows assigned (outer inf)",
        near_far=f"example.tag_and_report_budget recipe, k={NEAR_FAR_K}, q={NEAR_FAR_Q}, "
                 "exact L2 via sklearn (same values as faiss IndexFlatL2)",
        collapse_theorem="log-pooling likelihood is linear in member posteriors; the "
                         "credal set depends on the ensemble only through p_bar",
        pred_amendment="pred_amendment.json (2026-11-09, before any real-data scoring): "
                       "P1 rationale had the KL direction reversed; P1 still scored as "
                       "written, P1_amended added, unknown-slot width symmetric about 1/2",
        analysis_fixes=ANALYSIS_FIXES,
        delivered_check=f"shared columns reproduced vs {DELIVERED_ROWS}, max abs diff "
                        f"{drift:.2e}",
        caveat="scalar-score AUROC only; bounds are not membership intervals",
        worlds=metas)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nwritten:", out_dir)


if __name__ == "__main__":
    main()
