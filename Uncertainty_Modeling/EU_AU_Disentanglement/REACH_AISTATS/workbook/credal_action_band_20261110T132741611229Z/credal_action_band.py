# credal_action_band.py -- sensitivity band on the ch.8.4 one-step action values.
#
# Robustness layer on the real-record counterexample test (proposal 10 Nov 2026):
# is the belief-based channel preference (identity vs stratified label) an artifact
# of one prior, or does it hold across every prior compatible with the evidence?
#
#   * Replace the single Beta posterior (uniform Beta(1,1) prior + frozen counts)
#     with a SET of priors; recompute the one-step squared-error Bayes values
#     (real/beta_acquisition.action_values, sha256-pinned) over the set; band = [lo, hi]
#     per channel.
#   * Verdict at a state: ROBUST if the two channel bands are disjoint (one channel
#     wins everywhere), FRAGILE if they overlap. Confirmed/failed comes from the
#     frozen protocol's held-out rule (preferred action's expected squared error is
#     lower), never from the new computation.
#   * Primary family width is tied to the ch.7 reference-to-evaluation total-variation
#     distances of each dataset (not tuned on held-out error); a fixed 16x
#     prior-strength family and an evidence-discount family are disclosures.
#
# Why: it splits the FMoW failure into prior sensitivity (fragile -> report bands)
# vs model misspecification (robust and still wrong -> the Beta channel model itself
# fails on FMoW), which the point posterior cannot distinguish.
#
# PRED below is frozen and written to the run dir BEFORE any band is computed
# (credal_head.py precedent). Run from REACH_AISTATS on branch reach-mateus-workbook:
#
#   OMP_NUM_THREADS=1 python real/credal_action_band.py [--out DIR]
#   Out (default): workbook/credal_action_band_<UTC ts>Z/{pred.json, bands.csv,
#                  verdicts.csv, table_2x2.csv, candidate_sweep.csv, radius.csv,
#                  manifest.json, credal_action_band.py}
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from real.beta_acquisition import action_values

ROOT = Path(__file__).resolve().parent.parent        # REACH_AISTATS/

# ---------------------------------------------------------------------------
# Frozen inputs (workbook branch e309a46). Every file is sha256-verified before
# anything is computed; a mismatch aborts the run.
# ---------------------------------------------------------------------------
BETA_ACQUISITION_SHA = "d87c3b1335e409fdba8aeed149570bb1621f68fafc2bbc57162fcb3c1a49067b"

DATASETS = {
    "camelyon17": {
        "run": "workbook/camelyon17_counterexample_20261030T092024988890Z",
        "acquisition_run": "workbook/camelyon17_acquisition_20261030T092023425816Z",
        "sha256": {
            "selected_states.csv": "49f6a415185c5ec8c705a13cff64a526711d63dedfd42b62c5e559c0c5d9583c",
            "heldout_exact_one_step.csv": "c68fd38b94877b353fdfd3ff9d1b0ea2ded2db79cf932554613d9412ef3d9f79",
            "protocol.json": "5c7b49af18bd16b9725efdea67bae6fdc9a633390f9bf627262901c57ea0662d",
            "result.json": "1db33b3e2cf7291ec90b7de61cb14c5b8e4f3435d5d7ece1d3533691d27d6786",
            "candidate_states.csv": "5b772e72a34823dcd2cb9b50c530974a3672b4036f4d6e9a27cab070ae775c4d",
            "../camelyon17_acquisition_20261030T092023425816Z/source_cell_law_audit.csv":
                "315c69ad83dc979a1192503c4d12a4988d64bb277b4289cae8e9b6a770c7bc7c",
        },
        # tutorial ch.7, listed in dataset source order; audit sources are ids 0,1,2
        "tv_tutorial": [("hospital 0", 0.3115), ("hospital 3", 0.1995), ("hospital 2", 0.3625)],
    },
    "fmow": {
        "run": "workbook/fmow_counterexample_20261030T094929883846Z",
        "acquisition_run": "workbook/fmow_acquisition_20261030T094932286096Z",
        "sha256": {
            "selected_states.csv": "8bd95506d806615161a6815b45bac0ab4f24b0c14cf6dbf3cb9998dd88086d01",
            "heldout_exact_one_step.csv": "20e01c73daa4454bc13f6ba330ecc9896de5ab8a9f783eab69b1473f953f7855",
            "protocol.json": "5c7b49af18bd16b9725efdea67bae6fdc9a633390f9bf627262901c57ea0662d",
            "result.json": "2a65966130bd43069cd66e74f5955df1a59ffc7e39473ca6784aadda5966542d",
            "candidate_states.csv": "9d8ee71764aa81e09be8fd17592c448ef863bb2967b9b098130ea27d950bcdbb",
            "../fmow_acquisition_20261030T094932286096Z/source_cell_law_audit.csv":
                "361b9409af7f9ed1bd7c07a0646b03b32020dfa7916bf8cdba6eb3b9871684a5",
        },
        # tutorial ch.7 (FMoW containment section); audit sources 0,1,2 = Americas, Europe, Asia
        "tv_tutorial": [("Americas", 0.0135), ("Europe", 0.0125), ("Asia", 0.0145)],
    },
}

BASE_PRIOR = 1.0          # uniform Beta(1,1) per component, as archived (shapes = counts + 1)
STRESS_CS = (0.25, 0.5, 1.0, 2.0, 4.0)   # fixed 16x prior-strength range (disclosure family)
TV_BALL_GRID = 9
DISCOUNT_GRID = 5
CHANNELS = ("source_id", "reference_label")
PRIMARY = "tv_ball"

# ---------------------------------------------------------------------------
# PRED -- frozen before any band is computed. The FMoW call is ours.
# ---------------------------------------------------------------------------
PRED = {
    "written": "2026-11-10, before any credal band was computed",
    "question": "Is the belief-based channel preference an artifact of one prior, "
                "or does it hold across every prior compatible with the evidence?",
    "families": {
        "tv_ball (PRIMARY)": "per-component prior Beta(2m, 2(1-m)) with m in a "
                             "[0.5-TV, 0.5+TV] ball (9-point grid, endpoints and 0.5 "
                             "included, clipped to [0.05, 0.95]), independent across "
                             "(W, Theta0, Theta1); TV = max reference-to-evaluation "
                             "total-variation distance over sources from the ch.7 audit",
        "stress_fixed_strength": "per-component prior Beta(c, c), c in {0.25, 0.5, 1, 2, 4} "
                                 "(a 16x strength range; fixed, not data-derived)",
        "extra_evidence_discount": "counts tempered per component by lambda in "
                                   "[1-TV, 1] (5-point grid), base prior Beta(1,1); "
                                   "beyond the proposal, answers the 'robust only by "
                                   "sample size' objection",
    },
    "verdict_rule": "robust iff the [min,max] bands of the two channel one-step values "
                    "are disjoint under the PRIMARY family; winner = point-argmax "
                    "channel; confirmed iff the frozen protocol's per-state advantage "
                    "(ESE of other action minus ESE of preferred action) is > 0. "
                    "Prior families use only the ch.7 TV distances and fixed constants; "
                    "never tuned on held-out error.",
    "states": {
        "camelyon17": {
            "390": {"pred": "robust", "winner": "source_id", "recorded_heldout": "confirmed"},
            "566": {"pred": "robust", "winner": "reference_label", "recorded_heldout": "confirmed"},
        },
        "fmow": {
            "287": {"pred": "robust", "winner": "source_id", "recorded_heldout": "failed"},
            "561": {"pred": "robust", "winner": "reference_label", "recorded_heldout": "failed"},
        },
    },
    "quantitative": {
        "P1_all_robust_primary": True,
        "P2_all_robust_stress_16x": True,
        "P3_primary_separation_ge_10x": True,
        "P4_margin_robust_at_protocol_action_margin_2": True,
        "P5_extra_discount_family_robust": True,
        "P6_break_even_prior_pseudo_counts_exceed_data": True,
    },
    "rationale": "Point value ratios are 45x-470x in all four states; every family "
                 "perturbs O(1) prior pseudo-counts against n=8..512 observations, so "
                 "values move by O(1) factors and disjointness should survive. FMoW's "
                 "primary band is nearly a point (TV 0.0145 is tiny), so its robustness "
                 "is only meaningfully stressed by the 16x family. If P1-P4 hold, the "
                 "FMoW held-out failure is NOT prior sensitivity: the preference is "
                 "robust across priors yet empirically wrong, so the Beta source-mixture "
                 "channel model itself is misspecified on FMoW -- the stronger finding. "
                 "CAMELYON17 would then be a robust-and-confirmed success at ~25x larger "
                 "measured TV, dissociating measured law shift from where the belief-based "
                 "preference fails.",
    "falsified_if": "any selected state is fragile under the primary family, or any "
                    "quantitative P1-P6 item fails",
    "exploratory_no_verdict": [
        "candidate-state fragility rates (does the band ever say fragile: test teeth)",
        "break-even prior strength k* (how much prior weight would flip a preference)",
    ],
}

# Dated amendment, written before re-scoring (credal_head precedent): P6's original
# criterion compared k* against max(n_identity, n_labels), but the adversarial corner
# of the strength family attacks ONE component: W for identity_preferred states
# (prior weight vs n_identity), Theta0/Theta1 for label_preferred (vs n_labels).
# P6 is scored as written (FAIL); P6_amended scores the intended per-component read.
PRED_AMENDMENT = {
    "dated": "2026-11-10, after the first run, before re-scoring",
    "P6_as_written": "k_star > max(n_identity, n_labels_per_source) for all four states",
    "P6_amended": "k_star > n of the adversarial component (n_identity for "
                  "identity_preferred states, n_labels_per_source for label_preferred)",
    "why": "The break-even flip is driven by prior weight on the WINNING channel's "
           "governing component; comparing it to the larger, irrelevant component's "
           "sample size was an operationalisation error, discovered when P6 failed "
           "while the rationale's per-component arithmetic (k* ~ (sqrt(ratio)-1)*(n+1)/2) "
           "clearly held for every state.",
}

ANALYSIS_FIXES = [
    "2026-11-10: two reporting bugs fixed after the first (crashed) invocation, before "
    "any output file was written: the candidate-sweep dict iteration and an inverted "
    "cap-check in break_even_strength that returned inf exactly when bands DO touch. "
    "The four selected states' bands printed by that invocation were identical to the "
    "completed run's; pred.json predates both (frozen 13:27:41Z).",
]


# ---------------------------------------------------------------------------
# frozen-input loading + verification
# ---------------------------------------------------------------------------
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen():
    assert sha256(ROOT / "real/beta_acquisition.py") == BETA_ACQUISITION_SHA, \
        "real/beta_acquisition.py differs from the archived snapshot; refusing to run"
    for name, spec in DATASETS.items():
        for rel, want in spec["sha256"].items():
            p = (ROOT / spec["run"] / rel).resolve()
            assert sha256(p) == want, f"frozen input changed: {p}"
    print("frozen inputs verified (sha256, workbook e309a46)")


def load_selected(dataset):
    """Selected states + held-out errors + protocol constants, from frozen CSVs."""
    spec = DATASETS[dataset]
    sel = pd.read_csv(ROOT / spec["run"] / "selected_states.csv")
    held = pd.read_csv(ROOT / spec["run"] / "heldout_exact_one_step.csv")
    proto = json.loads((ROOT / spec["run"] / "protocol.json").read_text())
    result = json.loads((ROOT / spec["run"] / "result.json").read_text())
    states = {}
    for _, r in sel.iterrows():
        sid = int(r.state_id)
        shapes = np.array(json.loads(r.shapes), dtype=float)
        counts = shapes - BASE_PRIOR
        assert counts.min() >= 0, f"state {sid}: shapes below the uniform prior floor"
        assert counts[0].sum() == r.n_identity, f"state {sid}: W counts != n_identity"
        for j in (1, 2):
            assert counts[j].sum() == r.n_labels_per_source, \
                f"state {sid}: Theta counts != n_labels_per_source"
        vi, vl = action_values(shapes)
        assert np.isclose(vi, r.value_identity, rtol=1e-9) and \
               np.isclose(vl, r.value_label, rtol=1e-9), \
            f"state {sid}: action_values does not reproduce the recorded values"
        h = held[held.state_id == sid].set_index("action")["expected_squared_error"]
        role = r.role
        preferred = "source_id" if role == "identity_preferred" else "reference_label"
        other = "reference_label" if preferred == "source_id" else "source_id"
        states[sid] = dict(shapes=shapes, counts=counts, role=role,
                           preferred=preferred, point=(float(vi), float(vl)),
                           ese={a: float(h[a]) for a in CHANNELS},
                           heldout_advantage=float(h[other] - h[preferred]),
                           confirmed=bool(h[other] - h[preferred] > 0),
                           n_identity=int(r.n_identity),
                           n_labels=int(r.n_labels_per_source))
    # cross-check against the frozen run-level result.json
    ident = [s for s in states.values() if s["preferred"] == "source_id"][0]
    labl = [s for s in states.values() if s["preferred"] == "reference_label"][0]
    assert np.isclose(ident["heldout_advantage"], result["identity_advantage"], atol=1e-12)
    assert np.isclose(labl["heldout_advantage"], result["label_advantage"], atol=1e-12)
    assert result["confirmed"] == all(s["confirmed"] for s in states.values())
    margin = float(proto["action_margin"])
    return states, margin


def tv_distances(dataset):
    """Reference-to-evaluation TV per source from the ch.7 cell-law audit, verified
    against the tutorial's reported values (audit source ids 0,1,2 are in the
    tutorial's listing order). Returns (per_source, labels, tv_max)."""
    spec = DATASETS[dataset]
    df = pd.read_csv(ROOT / spec["acquisition_run"] / "source_cell_law_audit.csv")
    piv = df.pivot_table(index=["source", "cell"], columns="role",
                         values="probability").reset_index()
    per_source = {int(s): float(0.5 * np.abs(g["reference"] - g["evaluation"]).sum())
                  for s, g in piv.groupby("source")}
    labels = {}
    for src_id, (label, want) in zip(sorted(per_source), spec["tv_tutorial"]):
        got = per_source[src_id]
        assert abs(got - want) < 2e-4, \
            f"{dataset}: TV({label}, audit source {src_id}) = {got:.4f} but tutorial reports {want}"
        labels[src_id] = label
    return per_source, labels, max(per_source.values())


# ---------------------------------------------------------------------------
# prior families -> credal bands
# ---------------------------------------------------------------------------
def family_tv_ball(counts, tv, grid=TV_BALL_GRID):
    """Prior set: per-component Beta(2m, 2(1-m)), m in [0.5-tv, 0.5+tv], independent
    across components (product grid)."""
    lo, hi = max(0.05, 0.5 - tv), min(0.95, 0.5 + tv)
    ms = np.linspace(lo, hi, grid)
    M = np.stack(np.meshgrid(ms, ms, ms, indexing="ij"), -1).reshape(-1, 3)  # (F,3)
    A, B = 2 * M, 2 * (1 - M)
    return counts[None] + np.stack([A, B], axis=-1)                          # (F,3,2)


def family_fixed_strength(counts, cs=STRESS_CS):
    """Prior set: per-component Beta(c, c), c in cs, independent across components."""
    C = np.stack(np.meshgrid(*[np.array(cs)] * 3, indexing="ij"), -1).reshape(-1, 3)
    return counts[None] + np.stack([C, C], axis=-1)


def family_evidence_discount(counts, tv, grid=DISCOUNT_GRID):
    """Prior Beta(1,1) fixed; data counts tempered by lambda in [1-tv, 1] per component."""
    lams = np.linspace(max(0.0, 1.0 - tv), 1.0, grid)
    L = np.stack(np.meshgrid(*[lams] * 3, indexing="ij"), -1).reshape(-1, 3)
    return BASE_PRIOR + L[..., None] * counts[None]


def credal_action_band(family):
    """One-step value band per channel over a (F,3,2) family of posteriors."""
    vi, vl = action_values(family)                    # vectorized over leading dim
    return {"source_id": (float(vi.min()), float(vi.max())),
            "reference_label": (float(vl.min()), float(vl.max()))}


def verdict(bands, point, margin):
    """robust iff the channel bands are disjoint; winner by point value."""
    (ilo, ihi), (llo, lhi) = bands["source_id"], bands["reference_label"]
    disjoint = (ihi < llo) or (lhi < ilo)
    winner = "source_id" if point[0] >= point[1] else "reference_label"
    loser = "reference_label" if winner == "source_id" else "source_id"
    lose_hi = bands[loser][1]
    separation = float("inf") if lose_hi == 0 else bands[winner][0] / lose_hi
    return dict(verdict="robust" if disjoint else "fragile", winner=winner,
                separation=separation, margin_robust=bool(separation >= margin))


def break_even_strength(counts, point, cap=2.0 ** 24):
    """Exploratory: smallest k such that the product family c in {1/k, 1, k} per
    component makes the bands touch (log2-bisected)."""
    def overlap(k):
        fam = family_fixed_strength(counts, cs=(1.0 / k, 1.0, k))
        b = credal_action_band(fam)
        (ilo, ihi), (llo, lhi) = b["source_id"], b["reference_label"]
        return not ((ihi < llo) or (lhi < ilo))
    if not overlap(cap):
        return float("inf")                   # bands never touch, even at cap
    lo, hi = 1.0, cap                      # overlap(hi) is True or unreachable
    while np.log2(hi / lo) > 0.05:
        mid = np.sqrt(lo * hi)
        if overlap(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def build_families(counts, tv):
    return {"tv_ball": family_tv_ball(counts, tv),
            "stress_fixed_strength": family_fixed_strength(counts),
            "extra_evidence_discount": family_evidence_discount(counts, tv)}


def main(out_dir=None):
    out = Path(out_dir) if out_dir else ROOT / "workbook" / (
        "credal_action_band_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "Z")
    out.mkdir(parents=True, exist_ok=True)

    pred_path = out / "pred.json"
    if pred_path.exists():                 # PRED must be the frozen first word
        assert json.loads(pred_path.read_text()) == PRED, \
            "results exist under a different PRED; inspect before overwriting"
    else:
        pred_path.write_text(json.dumps(PRED, indent=2))
        print("PRED frozen to", pred_path, "-- before any band is computed")
    amend_path = out / "pred_amendment.json"
    if amend_path.exists():
        assert json.loads(amend_path.read_text()) == PRED_AMENDMENT, \
            "results exist under a different PRED_AMENDMENT"
    else:
        amend_path.write_text(json.dumps(PRED_AMENDMENT, indent=2))
        print("PRED_AMENDMENT written to", amend_path)

    verify_frozen()

    band_rows, verdict_rows, radius_rows = [], [], []
    fragility = {"tv_ball": [], "stress_fixed_strength": []}
    pred_score, table, margins = {}, {}, {}
    for dataset, spec in DATASETS.items():
        states, margin = load_selected(dataset)
        margins[dataset] = margin
        per_source, labels, tv_max = tv_distances(dataset)
        print(f"\n{dataset}: TV per source "
              f"{ {labels[k]: round(v, 4) for k, v in per_source.items()} } -> "
              f"family radius {tv_max:.4f}; protocol action_margin {margin}")
        table[dataset] = dict(tv={labels[k]: v for k, v in per_source.items()},
                              tv_max=tv_max, states={})
        for sid, st in states.items():
            families = build_families(st["counts"], tv_max)
            for n, f in families.items():          # base posterior must be in-family
                assert np.any(np.all(f == st["shapes"], axis=(1, 2))), \
                    f"state {sid}: base posterior not in family {n}"
            rows_v = {}
            for name, fam in families.items():
                b = credal_action_band(fam)
                v = verdict(b, st["point"], margin)
                rows_v[name] = dict(bands=b, verdict=v)
                for i, ch in enumerate(CHANNELS):
                    band_rows.append(dict(dataset=dataset, state_id=sid, family=name,
                                          channel=ch, lo=b[ch][0], hi=b[ch][1],
                                          point=st["point"][i]))
                short = {"source_id": "id", "reference_label": "lab"}
                verdict_rows.append(dict(
                    dataset=dataset, state_id=sid, role=st["role"], family=name,
                    **{f"v_{short[c]}_{x}": b[c][i] for c in CHANNELS
                       for i, x in ((0, "lo"), (1, "hi"))},
                    point_v_id=st["point"][0], point_v_lab=st["point"][1],
                    **v, heldout_advantage=st["heldout_advantage"],
                    confirmed=st["confirmed"]))
            primary = rows_v[PRIMARY]["verdict"]
            table[dataset]["states"][sid] = dict(
                role=st["role"], preferred=st["preferred"],
                point=dict(zip(CHANNELS, st["point"])),
                n_identity=st["n_identity"], n_labels=st["n_labels"], tv_max=tv_max,
                bands={n: r["bands"] for n, r in rows_v.items()},
                verdict={n: r["verdict"]["verdict"] for n, r in rows_v.items()},
                separation={n: r["verdict"]["separation"] for n, r in rows_v.items()},
                heldout_advantage=st["heldout_advantage"], confirmed=st["confirmed"])
            k = break_even_strength(st["counts"], st["point"])
            n_adv = st["n_identity"] if st["preferred"] == "source_id" else st["n_labels"]
            radius_rows.append(dict(dataset=dataset, state_id=sid, role=st["role"],
                                    k_star=k, n_identity=st["n_identity"],
                                    n_labels_per_source=st["n_labels"],
                                    adversarial_component=("W" if st["preferred"] == "source_id"
                                                           else "Theta0/Theta1"),
                                    n_adversarial=n_adv,
                                    k_star_over_n_adversarial=(float(k) / n_adv
                                                               if np.isfinite(k) else None),
                                    k_star_exceeds_data=bool(
                                        k > max(st["n_identity"], st["n_labels"])),   # P6 as written
                                    k_star_exceeds_adversarial_n=bool(k > n_adv)))     # P6 amended
            print(f"  state {sid} ({st['role']}): primary {primary['verdict']}, "
                  f"winner {primary['winner']}, separation {primary['separation']:.1f}x; "
                  f"stress {rows_v['stress_fixed_strength']['verdict']['verdict']} "
                  f"({rows_v['stress_fixed_strength']['verdict']['separation']:.1f}x); "
                  f"discount {rows_v['extra_evidence_discount']['verdict']['verdict']}; "
                  f"held-out {'confirmed' if st['confirmed'] else 'FAILED'} "
                  f"(advantage {st['heldout_advantage']:+.2e}); k* "
                  f"{'>' if np.isinf(k) else f'{k:.0f}'}")
            p = PRED["states"][dataset][str(sid)]
            pred_score[f"{dataset}:{sid}"] = dict(
                pred_verdict=p["pred"], got_verdict=primary["verdict"],
                pred_winner=p["winner"], got_winner=primary["winner"],
                recorded_heldout=p["recorded_heldout"],
                got_heldout="confirmed" if st["confirmed"] else "failed",
                passed=bool(p["pred"] == primary["verdict"]
                            and p["winner"] == primary["winner"]
                            and p["recorded_heldout"] ==
                            ("confirmed" if st["confirmed"] else "failed")))

        # exploratory: candidate sweep (test teeth; no held-out exists there)
        cand = pd.read_csv(ROOT / spec["run"] / "candidate_states.csv")
        for _, r in cand.iterrows():
            shapes = np.array(json.loads(r.shapes), dtype=float)
            counts = shapes - BASE_PRIOR
            if counts.min() < 0:
                continue
            vi, vl = action_values(shapes)
            point = (float(vi), float(vl))
            ratio = (max(point) / min(point)) if min(point) > 0 else float("inf")
            for fam_name in fragility:
                fam = (family_tv_ball(counts, tv_max) if fam_name == "tv_ball"
                       else family_fixed_strength(counts))
                v = verdict(credal_action_band(fam), point, margin)
                fragility[fam_name].append((dataset, int(r.state_id), ratio,
                                            v["verdict"]))

    # 2x2 table under the PRIMARY family
    cells = {f"{rv}&{cf}": [] for rv in ("robust", "fragile")
             for cf in ("confirmed", "failed")}
    for dataset in DATASETS:
        for sid, st in table[dataset]["states"].items():
            cells[f"{st['verdict'][PRIMARY]}&"
                  f"{'confirmed' if st['confirmed'] else 'failed'}"].append(
                  f"{dataset}:{sid}")

    # quantitative PRED items
    q = {
        "P1_all_robust_primary": all(r["verdict"] == "robust" for r in verdict_rows
                                     if r["family"] == PRIMARY),
        "P2_all_robust_stress_16x": all(r["verdict"] == "robust" for r in verdict_rows
                                        if r["family"] == "stress_fixed_strength"),
        "P3_primary_separation_ge_10x": all(r["separation"] >= 10.0
                                            for r in verdict_rows
                                            if r["family"] == PRIMARY),
        "P4_margin_robust_at_protocol_action_margin_2": all(r["margin_robust"]
                                                            for r in verdict_rows
                                                            if r["family"] == PRIMARY),
        "P5_extra_discount_family_robust": all(r["verdict"] == "robust"
                                               for r in verdict_rows
                                               if r["family"] == "extra_evidence_discount"),
        "P6_break_even_prior_pseudo_counts_exceed_data": all(
            r["k_star_exceeds_data"] for r in radius_rows),
        "P6_amended_break_even_exceeds_adversarial_component_n": all(
            r["k_star_exceeds_adversarial_n"] for r in radius_rows),
    }
    # predicted values come from PRED, or from the dated amendment for P6_amended
    q_pass = {k: {"predicted": bool(PRED["quantitative"].get(k, True)), "got": bool(v),
                  "pass": bool(PRED["quantitative"].get(k, True) == bool(v))}
              for k, v in q.items()}

    pd.DataFrame(band_rows).to_csv(out / "bands.csv", index=False)
    pd.DataFrame(verdict_rows).to_csv(out / "verdicts.csv", index=False)
    pd.DataFrame([dict(cell=cell, states=";".join(ids)) for cell, ids in cells.items()]
                 ).to_csv(out / "table_2x2.csv", index=False)
    pd.DataFrame([dict(dataset=d, state_id=s, family=f, point_ratio=r, verdict=v)
                  for f, rows in fragility.items() for d, s, r, v in rows]
                 ).to_csv(out / "candidate_sweep.csv", index=False)
    pd.DataFrame(radius_rows).to_csv(out / "radius.csv", index=False)

    manifest = dict(
        script_sha256=sha256(Path(__file__)),
        branch="reach-mateus-workbook @ " + subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True).stdout.strip(),
        frozen_inputs={d: dict(s["sha256"],
                               **{"beta_acquisition.py": BETA_ACQUISITION_SHA})
                       for d, s in DATASETS.items()},
        tv_distances={d: dict(table[d]["tv"], max_used=table[d]["tv_max"])
                      for d in DATASETS},
        action_margin=margins,
        families=PRED["families"],
        pred_amendment=PRED_AMENDMENT,
        analysis_fixes=ANALYSIS_FIXES,
        pred_scorecard=pred_score,
        quantitative=q_pass,
        overall_as_written=("ALL PRED PASS" if
                            all(v["passed"] for v in pred_score.values()) and
                            all(v["pass"] for k, v in q_pass.items()
                                if k != "P6_amended_break_even_exceeds_adversarial_component_n")
                            else "PRED FAILED (P6 as written; see pred_amendment.json)"),
        overall_with_amendment=(
            "ALL PRED PASS under P6_amended" if
            all(v["passed"] for v in pred_score.values()) and
            all(v["pass"] for k, v in q_pass.items()
                if k != "P6_break_even_prior_pseudo_counts_exceed_data")
            else "PRED FAILED even under P6_amended"),
        candidate_fragility={f: {
            "n": len(rows),
            "fragile": sum(1 for _, _, _, v in rows if v == "fragile"),
            "n_ratio_ge_2": sum(1 for _, _, r, _ in rows if r >= 2),
            "fragile_among_ratio_ge_2": sum(1 for _, _, r, v in rows
                                            if r >= 2 and v == "fragile")}
            for f, rows in fragility.items()},
        results=table,
        python=sys.version.split()[0],
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(Path(__file__), out / "credal_action_band.py")   # code snapshot

    print("\n=== 2x2 (primary family) ===")
    for cell, ids in cells.items():
        print(f"  {cell:20s} {ids if ids else '-'}")
    print("\n=== candidate sweep (exploratory: can the band say fragile?) ===")
    for f, rows in fragility.items():
        frag = sum(1 for _, _, _, v in rows if v == "fragile")
        n2 = sum(1 for _, _, r, _ in rows if r >= 2)
        frag2 = sum(1 for _, _, r, v in rows if r >= 2 and v == "fragile")
        print(f"  {f:22s} fragile {frag}/{len(rows)} overall; "
              f"{frag2}/{n2} among point-ratio>=2 states")
    print("\nPRED:", manifest["overall_as_written"])
    print("PRED (with dated amendment):", manifest["overall_with_amendment"])
    for k, v in pred_score.items():
        print(f"  {k}: {'PASS' if v['passed'] else 'FAIL'}")
    for k, v in q_pass.items():
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} (got {v['got']})")
    print("\nwritten:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="run dir (default workbook/<ts>)")
    args = ap.parse_args()
    main(args.out)
