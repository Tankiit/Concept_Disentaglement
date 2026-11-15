# band_beyond_budget.py -- the two remaining checks between the band result and the
# sentence "the band tells you when not to trust the pick" (collaborator, 10 Nov night).
#
#   1. Expected cost by verdict on CAMELYON17 alone: mean gap per state (wrong rate
#      times cost-given-wrong), robust vs fragile. If flat, the claim is frequency-only.
#   2. Band beyond budget: (a) which of the 30 budget cells contain BOTH verdicts, and
#      the wrong-rate difference inside those mixed cells; (b) out-of-fold logistic
#      wrong ~ BUDGET_FEATURES (frozen below) vs the same + verdict: the lift.
#
# PRED frozen before any of these numbers were computed. Gaps/evaluator reused from
# real/gap_margin_check (validated against the frozen held-out rows).
# Run from REACH_AISTATS on branch reach-mateus-workbook:
#   OMP_NUM_THREADS=1 python -m real.band_beyond_budget [--out DIR]
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from real.gap_margin_check import RUNS, errors, population_cells, state_arrays

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "workbook/credal_action_band_20261110T132741611229Z/candidate_sweep.csv"
BUDGET_FEATURES = ["n_identity", "n_labels_per_source",
                   "log1p_n_identity", "log1p_n_labels"]      # FROZEN before fitting

# ---------------------------------------------------------------------------
# PRED -- frozen 2026-11-10, before any cost/mixed-cell/regression number.
# ---------------------------------------------------------------------------
PRED = {
    "written": "2026-11-10, before any computation",
    "rationale": "Fragile should concentrate in low-budget cells (n_identity=0 makes W "
                 "posterior = prior, so the prior family moves it directly); a few "
                 "cells straddle the threshold. Within cells, fragile repeats are the "
                 "diffuse ones, plausibly also the error-prone ones, so a positive but "
                 "smaller difference should survive. The collaborator's arithmetic "
                 "(0.56 x 1.5e-3 vs 0.30 x 3.3e-3) says expected cost is flat; I expect "
                 "that confirmed, making the claim frequency-only.",
    "Q1_expected_cost_ratio_fragile_over_robust_in_[0.8,1.3]_tvball": True,
    "Q2_mixed_cells_of_30_in_[2,6]_tvball": True,
    "Q3_within_mixed_cells_diff_points_in_[3,25]_tvball": True,
    "Q4_verdict_auc_lift_over_budget_in_[0.00,0.08]_tvball": True,
    "Q5_ge80pct_fragile_in_cells_with_n_identity_le_32": True,
    "paper_rule": "if Q1 flat and Q3 holds: the licensed sentence is about error "
                  "FREQUENCY (the band marks where the pick is more often wrong, "
                  "beyond budget size), not expected cost; if Q3 fails, the band is "
                  "budget size restated and the claim does not survive the redundancy "
                  "check.",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def camelyon_frame():
    spec = RUNS["camelyon17"]
    cells = population_cells(spec)
    N = int(sum(c["n0"] + c["n1"] for c in cells.values()))
    w = {s: (cells[s]["n0"] + cells[s]["n1"]) / N for s in (0, 1)}
    rates = {s: cells[s]["n1"] / (cells[s]["n0"] + cells[s]["n1"]) for s in (0, 1)}
    T = sum(w[s] * rates[s] for s in (0, 1))
    cand, a, b = state_arrays(spec)
    e_id, e_lab = errors(a, b, w, rates, T)
    pick = np.where(cand.value_identity >= cand.value_label, 0, 1)
    cand = cand.assign(
        gap=np.where(pick == 0, e_id, e_lab) - np.minimum(e_id, e_lab),
        e_identity=e_id, e_label=e_lab,
        log1p_n_identity=np.log1p(cand.n_identity),
        log1p_n_labels=np.log1p(cand.n_labels_per_source))
    cand["wrong"] = cand.gap > 1e-9
    # evaluator re-validation against the frozen rows (3 states' errors suffice)
    held = pd.read_csv(ROOT / spec["run"] / "heldout_exact_one_step.csv")
    idx = cand.set_index("state_id")
    for _, r in held.iterrows():
        want = float(r.expected_squared_error)
        got = float(idx.loc[int(r.state_id), "e_identity" if r.action == "source_id"
                            else "e_label"])
        assert abs(got - want) <= 1e-9, "evaluator drift"
    sweep = pd.read_csv(SWEEP)
    v = sweep[sweep.dataset == "camelyon17"][["state_id", "family", "verdict"]]
    return cand.merge(v, on="state_id")


def oof_auc(df, features, target="wrong"):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    p = np.empty(len(df))
    y = df[target].to_numpy().astype(int)
    X = df[features].to_numpy()
    for tr, te in skf.split(X, y):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=1000, random_state=0))])
        pipe.fit(X[tr], y[tr])
        p[te] = pipe.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p)), float(log_loss(y, p))


def main(out_dir=None):
    out = Path(out_dir) if out_dir else ROOT / "workbook" / (
        "band_beyond_budget_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "Z")
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "pred.json"
    if pred_path.exists():
        assert json.loads(pred_path.read_text()) == PRED
    else:
        pred_path.write_text(json.dumps(PRED, indent=2))
        print("PRED frozen to", pred_path, "-- before any computation")

    D = camelyon_frame()
    res = {}
    for fam in ("tv_ball", "stress_fixed_strength"):
        d = D[D.family == fam].copy()
        frag = d.verdict.eq("fragile")
        rob, fr = d[~frag], d[frag]

        # ---- 1. expected cost by verdict (CAMELYON17 alone) --------------------
        cost = dict(
            wrong_robust=float(rob.wrong.mean()), wrong_fragile=float(fr.wrong.mean()),
            mean_gap_given_wrong_robust=float(rob[rob.wrong].gap.mean()),
            mean_gap_given_wrong_fragile=float(fr[fr.wrong].gap.mean()),
            expected_cost_robust=float(rob.gap.mean()),
            expected_cost_fragile=float(fr.gap.mean()))
        cost["ratio_fragile_over_robust"] = (cost["expected_cost_fragile"]
                                             / cost["expected_cost_robust"])
        res[fam] = dict(cost=cost)

        # ---- 2a. mixed budget cells --------------------------------------------
        cell = d.groupby(["n_identity", "n_labels_per_source"]).verdict
        per_cell = d.groupby(["n_identity", "n_labels_per_source"]).agg(
            n=("verdict", "size"), n_fragile=("verdict", lambda s: (s == "fragile").sum()),
            wrong=("wrong", "mean")).reset_index()
        per_cell["mixed"] = (per_cell.n_fragile > 0) & (per_cell.n_fragile < per_cell.n)
        mixed = per_cell[per_cell.mixed]
        d["cell_mixed"] = d.set_index(["n_identity", "n_labels_per_source"]).index.isin(
            mixed.set_index(["n_identity", "n_labels_per_source"]).index)
        dm = d[d.cell_mixed]
        within = float(dm[dm.verdict == "fragile"].wrong.mean()
                       - dm[dm.verdict == "robust"].wrong.mean()) if len(dm) else float("nan")
        frag_all = d[frag]
        res[fam]["cells"] = dict(
            n_cells=len(per_cell), n_mixed=int(per_cell.mixed.sum()),
            n_all_fragile=int(((per_cell.n_fragile == per_cell.n)).sum()),
            fragile_in_low_budget=float((frag_all.n_identity <= 32).mean()),
            within_mixed_diff_points=100 * within,
            mixed_cell_list=[f"id={r.n_identity}/lab={r.n_labels_per_source}: {r.n_fragile}/{r.n} "
                             f"fragile, wrong {r.wrong:.2f}" for r in mixed.itertuples()])

        # ---- 2b. budget-only vs budget+verdict lift -----------------------------
        base = d[BUDGET_FEATURES].to_numpy()
        withv = np.column_stack([base, frag.to_numpy().astype(int)])
        dfb = pd.DataFrame(withv, columns=BUDGET_FEATURES + ["fragile"]).assign(wrong=d.wrong.to_numpy())
        auc_b, ll_b = oof_auc(dfb, BUDGET_FEATURES)
        auc_v, ll_v = oof_auc(dfb, BUDGET_FEATURES + ["fragile"])
        auc_only, ll_only = oof_auc(dfb, ["fragile"])
        res[fam]["regression"] = dict(
            auc_budget_only=auc_b, auc_budget_plus_verdict=auc_v, auc_verdict_only=auc_only,
            auc_lift=auc_v - auc_b, logloss_budget_only=ll_b,
            logloss_budget_plus_verdict=ll_v, logloss_lift=ll_v - ll_b)

    c = res["tv_ball"]
    got = {
        "Q1_expected_cost_ratio_fragile_over_robust_in_[0.8,1.3]_tvball":
            bool(0.8 <= c["cost"]["ratio_fragile_over_robust"] <= 1.3),
        "Q2_mixed_cells_of_30_in_[2,6]_tvball": bool(2 <= c["cells"]["n_mixed"] <= 6),
        "Q3_within_mixed_cells_diff_points_in_[3,25]_tvball":
            bool(3 <= c["cells"]["within_mixed_diff_points"] <= 25),
        "Q4_verdict_auc_lift_over_budget_in_[0.00,0.08]_tvball":
            bool(0.0 <= c["regression"]["auc_lift"] <= 0.08),
        "Q5_ge80pct_fragile_in_cells_with_n_identity_le_32":
            bool(c["cells"]["fragile_in_low_budget"] >= 0.8),
    }
    passed = {k: bool(got[k]) == PRED[k] for k in got}

    pd.DataFrame([{**{"family": fam}, **res[fam]["cost"]} for fam in res]
                 ).to_csv(out / "cost_by_verdict.csv", index=False)
    pd.DataFrame([dict(family=fam, **{k: (v if not isinstance(v, list) else "; ".join(v))
                                      for k, v in res[fam]["cells"].items()})
                  for fam in res]).to_csv(out / "cells.csv", index=False)
    pd.DataFrame([dict(family=fam, **res[fam]["regression"]) for fam in res]
                 ).to_csv(out / "regression_lift.csv", index=False)

    manifest = dict(script_sha256=sha(Path(__file__)),
                    branch=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip(),
                    inputs={"candidate_sweep.csv": sha(SWEEP)} |
                            {f"camelyon17/{f}": sha(ROOT / RUNS['camelyon17']['run'] / f)
                             for f in ("candidate_states.csv", "population_rates.csv",
                                       "heldout_exact_one_step.csv")},
                    budget_features=BUDGET_FEATURES, results=res, got=got, passed=passed,
                    overall=("ALL PRED PASS" if all(passed.values()) else "PRED FAILED"),
                    paper_rule=PRED["paper_rule"], python=sys.version.split()[0])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(Path(__file__), out / "band_beyond_budget.py")

    for fam in res:
        r = res[fam]
        print(f"\n[{fam}] expected cost/state: robust {r['cost']['expected_cost_robust']:.2e} "
              f"vs fragile {r['cost']['expected_cost_fragile']:.2e} "
              f"(ratio {r['cost']['ratio_fragile_over_robust']:.2f})")
        print(f"  cost given wrong: robust {r['cost']['mean_gap_given_wrong_robust']:.2e} "
              f"vs fragile {r['cost']['mean_gap_given_wrong_fragile']:.2e}")
        print(f"  cells: {r['cells']['n_mixed']} mixed of {r['cells']['n_cells']} | "
              f"within-mixed diff {r['cells']['within_mixed_diff_points']:+.1f} pts | "
              f"fragile in low-budget cells {r['cells']['fragile_in_low_budget']:.2f}")
        for s in r["cells"]["mixed_cell_list"]:
            print(f"    {s}")
        g = r["regression"]
        print(f"  AUC: budget-only {g['auc_budget_only']:.3f} + verdict "
              f"{g['auc_budget_plus_verdict']:.3f} (lift {g['auc_lift']:+.3f}) | "
              f"verdict-only {g['auc_verdict_only']:.3f} | logloss lift "
              f"{g['logloss_lift']:+.4f}")
    print("\n== PRED ==")
    for k, ok in passed.items():
        print(f"  {k}: {'PASS' if ok else 'FAIL'} (got {got[k]})")
    print("  overall:", manifest["overall"])
    print("\nwritten:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    main(ap.parse_args().out)
