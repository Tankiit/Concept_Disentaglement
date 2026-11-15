# gap_by_verdict.py -- does the credal sensitivity band mark WHERE the channel pick
# fails? (the "sets tell you when not to trust the pick" claim; collaborator 10 Nov)
#
# Design (circularity guards, all satisfied by construction):
#   * PICK   = per-channel Bayesian pick: argmax of the point action values V
#              (expected per-channel information gain; the fair baseline, not scalar BALD).
#   * BEST   = argmin of exact held-out one-step squared error (frozen empirical
#              population; validated to 1e-9 against heldout_exact_one_step.csv).
#   * GAP    = E(pick) - min(E_id, E_lab) >= 0, from real/acquisition_gap.py output.
#   * VERDICT= robust/fragile under the prior families frozen in credal_action_band.py
#              (PRED frozen 2026-11-10 MORNING, before any gap existed): primary
#              tv_ball (TV-tied), disclosure stress_fixed_strength (16x).
#   The verdicts never used held-out errors; the gaps never used the prior families.
#
# Run from REACH_AISTATS on branch reach-mateus-workbook:
#   OMP_NUM_THREADS=1 python -m real.gap_by_verdict [--out DIR]
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
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
GAPS = ROOT / "workbook/acquisition_gap_20261110T165255140652Z/gaps.csv"
SWEEP = ROOT / "workbook/credal_action_band_20261110T132741611229Z/candidate_sweep.csv"
SHA = {
    "gaps.csv": None,          # filled at run start from disk, recorded in manifest
    "candidate_sweep.csv": None,
}
PRIMARY = "tv_ball"
ALREADY_SEEN = ("decisive (action_margin >= 2) vs not, from the acquisition-gap run's "
                "exploratory block: 33.4% vs 37.0% wrong -- weak concentration; this "
                "context informs PRED but no gap-by-verdict number was seen before it.")

# ---------------------------------------------------------------------------
# PRED -- frozen before the join below was ever executed.
# ---------------------------------------------------------------------------
PRED = {
    "written": "2026-11-10, before any gap-by-verdict computation",
    "claim_under_test": "the acquisition gap is concentrated in states the sensitivity "
                        "band marks fragile (sets flag when the pick is not to be trusted)",
    "rationale": "Fragile states are mostly near-ties of the point values, and the "
                 "already-seen decisiveness split showed only weak concentration "
                 "(33.4% decisive vs 37.0% indecisive wrong). The band adds "
                 "prior-overlap on top of the point margin, so I predict the direction "
                 "holds but weakly: the band is a weak trust marker, not a reliable one.",
    "P1_fragile_wrongrate_gt_robust_tv_ball_pooled": True,
    "P2_concentration_weak_lt_15_points_tv_ball_pooled": True,
    "P3_auc_verdict_predicts_gap_positive_in_[0.50,0.65]_tv_ball_pooled": True,
    "P4_mean_gap_fragile_gt_robust_either_family": True,
    "P5_direction_holds_under_stress_family": True,
    "P6_fragile_wrongrate_le_0.50_any_family_pooled": True,
    "falsified_if": "fragile wrong-rate <= robust (no concentration), or concentration "
                    ">= 15 points with AUC >= 0.75 (then the band is a strong marker "
                    "and the claim is bigger than predicted)",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def auc_of_marker(marker, positive):
    """Rank-AUC of a binary marker predicting a binary outcome."""
    m, p = np.asarray(marker, bool), np.asarray(positive, bool)
    n1, n0 = p.sum(), (~p).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(np.where(m, 1.0, 0.0)).rank().to_numpy()
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(out_dir=None):
    out = Path(out_dir) if out_dir else ROOT / "workbook" / (
        "gap_by_verdict_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "Z")
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "pred.json"
    if pred_path.exists():
        assert json.loads(pred_path.read_text()) == PRED
    else:
        pred_path.write_text(json.dumps(PRED, indent=2))
        print("PRED frozen to", pred_path, "-- before any gap-by-verdict computation")
    SHA["gaps.csv"] = sha(GAPS)
    SHA["candidate_sweep.csv"] = sha(SWEEP)

    G = pd.read_csv(GAPS)[["dataset", "state_id", "gap_model_pick", "initial_err",
                           "decisive"]]
    V = pd.read_csv(SWEEP)[["dataset", "state_id", "family", "verdict", "point_ratio"]]
    M = G.merge(V, on=["dataset", "state_id"], validate="one_to_many")  # 2 families/state
    assert len(M) == len(G) * 2, "join lost or duplicated states"
    M["wrong"] = M.gap_model_pick > 1e-9
    M["rel_gap"] = M.gap_model_pick / M.initial_err.clip(1e-18)

    rows, verdict = [], {}
    for fam in (PRIMARY, "stress_fixed_strength"):
        sub = M[M.family == fam]
        rob, fra = sub[~sub.verdict.eq("fragile")], sub[sub.verdict.eq("fragile")]
        wr = dict(robust=float(rob.wrong.mean()), fragile=float(fra.wrong.mean()),
                  n_robust=len(rob), n_fragile=len(fra),
                  diff_points=100 * (fra.wrong.mean() - rob.wrong.mean()))
        gap = dict(mean_gap_robust=float(rob.gap_model_pick.mean()),
                   mean_gap_fragile=float(fra.gap_model_pick.mean()),
                   median_rel_robust=float(rob[rob.wrong].rel_gap.median()) if rob.wrong.any() else 0.0,
                   median_rel_fragile=float(fra[fra.wrong].rel_gap.median()) if fra.wrong.any() else 0.0)
        try:
            u = mannwhitneyu(rob.gap_model_pick, fra.gap_model_pick, alternative="less")
            p_one_sided = float(u.pvalue)      # H1: robust gaps stochastically smaller
        except ValueError:
            p_one_sided = float("nan")
        auc = auc_of_marker(sub.verdict.eq("fragile"), sub.wrong)
        verdict[fam] = dict(wrong_rates=wr, gaps=gap, auc_fragile_predicts_wrong=auc,
                            mannwhitney_p_robust_smaller=p_one_sided)
        for name, g in list(sub.groupby("dataset")) + [("POOLED", sub)]:
            r, f = g[~g.verdict.eq("fragile")], g[g.verdict.eq("fragile")]
            rows.append(dict(family=fam, dataset=name, n_fragile=len(f), n_robust=len(r),
                             wrong_robust=float(r.wrong.mean()),
                             wrong_fragile=float(f.wrong.mean()) if len(f) else float("nan"),
                             mean_gap_robust=float(r.gap_model_pick.mean()),
                             mean_gap_fragile=float(f.gap_model_pick.mean()) if len(f) else float("nan")))
        print(f"\n[{fam}] pooled: wrong robust {wr['robust']:.3f} (n={wr['n_robust']}) vs "
              f"fragile {wr['fragile']:.3f} (n={wr['n_fragile']}) | diff "
              f"{wr['diff_points']:+.1f} pts | AUC(fragile->wrong) {auc:.3f} | "
              f"Mann-Whitney p (robust smaller) {p_one_sided:.2e}")
        print(f"    mean gap robust {gap['mean_gap_robust']:.2e} vs fragile "
              f"{gap['mean_gap_fragile']:.2e}")

    prim = verdict[PRIMARY]
    stress = verdict["stress_fixed_strength"]
    got = {
        "P1_fragile_wrongrate_gt_robust_tv_ball_pooled":
            bool(prim["wrong_rates"]["fragile"] > prim["wrong_rates"]["robust"]),
        "P2_concentration_weak_lt_15_points_tv_ball_pooled":
            bool(prim["wrong_rates"]["diff_points"] < 15.0),
        "P3_auc_verdict_predicts_gap_positive_in_[0.50,0.65]_tv_ball_pooled":
            bool(0.50 <= prim["auc_fragile_predicts_wrong"] <= 0.65),
        "P4_mean_gap_fragile_gt_robust_either_family":
            bool((prim["gaps"]["mean_gap_fragile"] > prim["gaps"]["mean_gap_robust"])
                 or (stress["gaps"]["mean_gap_fragile"] > stress["gaps"]["mean_gap_robust"])),
        "P5_direction_holds_under_stress_family":
            bool(stress["wrong_rates"]["fragile"] > stress["wrong_rates"]["robust"]),
        "P6_fragile_wrongrate_le_0.50_any_family_pooled":
            bool(prim["wrong_rates"]["fragile"] <= 0.50
                 and stress["wrong_rates"]["fragile"] <= 0.50),
    }
    passed = {k: bool(got[k]) == PRED[k] for k in got}

    pd.DataFrame(rows).to_csv(out / "gap_by_verdict.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, fam in zip(axes, (PRIMARY, "stress_fixed_strength")):
        sub = M[M.family == fam]
        for v, color in (("robust", "#4c72b0"), ("fragile", "#c44e52")):
            g = sub[sub.verdict == v]
            wrong = float(g.wrong.mean())
            ax.bar(v, wrong, color=color)
            ax.text(v, wrong + 0.01, f"{wrong:.2f}\n(n={len(g)})", ha="center", fontsize=8)
        ax.set_ylim(0, 0.6); ax.set_title(fam)
        ax.set_ylabel("share of states with wrong pick")
    fig.suptitle("Wrong-pick rate by sensitivity-band verdict (pick = per-channel "
                 "Bayesian values; best = held-out only)", y=1.03)
    fig.tight_layout()
    fig.savefig(out / "gap_by_verdict.png", dpi=150, bbox_inches="tight")

    manifest = dict(script_sha256=sha(Path(__file__)),
                    branch=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip(),
                    inputs=SHA, results=verdict, got=got, passed=passed,
                    overall=("ALL PRED PASS" if all(passed.values()) else "PRED FAILED"),
                    already_seen_context=ALREADY_SEEN,
                    circularity="verdicts: prior families frozen before any gap; gaps: "
                                "held-out errors only; pick: point action values",
                    python=sys.version.split()[0])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(Path(__file__), out / "gap_by_verdict.py")
    print("\n== PRED ==")
    for k, ok in passed.items():
        print(f"  {k}: {'PASS' if ok else 'FAIL'} (got {got[k]})")
    print("  overall:", manifest["overall"])
    print("\nwritten:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    main(ap.parse_args().out)
