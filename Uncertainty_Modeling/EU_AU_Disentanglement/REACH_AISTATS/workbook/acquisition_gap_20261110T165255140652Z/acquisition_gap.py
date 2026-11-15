# acquisition_gap.py -- how often does a belief-only pick of the acquisition channel
# differ from the best channel, and by how much? (the prevalence instrument behind
# contribution 1 of the acquisition paper; 600 information states per dataset)
#
# Picks compared (all rules FROZEN in PRED below before any gap was computed):
#   belief_only : out-of-fold logistic regression on z-scored [mean, variance, MI]
#                 (5-fold CV, random_state 0); p == 0.5 -> source_id.
#   budget_only : majority best-channel of the TRAINING-fold states in the same
#                 (n_identity, n_labels_per_source) cell; empty/tie -> source_id.
#   model_pick  : argmax of the recorded point action values (per-channel bounds);
#                 tie -> source_id.
#   best        : argmin of the exact held-out one-step squared error (enumeration
#                 over the frozen held-out population counts).
# Gap(s, pick) = E(s, pick) - min(E_identity, E_label) >= 0.
# Practitioner threshold (frozen): relative gap >= 10% of the state's initial
# (no-further-acquisition) squared error.
#
# Evaluator discipline: the exact one-step error reproduces the frozen protocol's
# heldout_exact_one_step.csv (all 8 rows) to <= 1e-9 absolute BEFORE any gap is
# computed; a mismatch aborts the run.
#
# Run from REACH_AISTATS on branch reach-mateus-workbook:
#   OMP_NUM_THREADS=1 python -m real.acquisition_gap [--out DIR]
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from real.beta_acquisition import action_values, moments

ROOT = Path(__file__).resolve().parent.parent
BETA_SHA = "d87c3b1335e409fdba8aeed149570bb1621f68fafc2bbc57162fcb3c1a49067b"
DATASETS = {
    "camelyon17": dict(run="workbook/camelyon17_counterexample_20261030T092024988890Z",
                       sha={"candidate_states.csv": "5b772e72a34823dcd2cb9b50c530974a3672b4036f4d6e9a27cab070ae775c4d",
                            "population_rates.csv": "b72bdb9fde2d11a87c7d1a2b1cc320a256e504885e50fa9138b0175db8bc92a0",
                            "heldout_exact_one_step.csv": "c68fd38b94877b353fdfd3ff9d1b0ea2ded2db79cf932554613d9412ef3d9f79"}),
    "fmow": dict(run="workbook/fmow_counterexample_20261030T094929883846Z",
                 sha={"candidate_states.csv": "9d8ee71764aa81e09be8fd17592c448ef863bb2967b9b098130ea27d950bcdbb",
                      "population_rates.csv": "f161559d9ae404c362afbbbc56c4eb9cbd121ae94e290298f9075cea92b8b741",
                      "heldout_exact_one_step.csv": "20e01c73daa4454bc13f6ba330ecc9896de5ab8a9f783eab69b1473f953f7855"}),
}
TIE = "source_id"          # frozen tie-break for every rule
REL_THRESHOLD = 0.10       # frozen practitioner threshold
CHANNELS = ("source_id", "reference_label")

# ---------------------------------------------------------------------------
# PRED -- frozen 2026-11-10, before any held-out error or gap was computed for the
# 600-state grid. Point estimates and the ranges we would consider consistent.
# ---------------------------------------------------------------------------
PRED = {
    "written": "2026-11-10, before any gap computation",
    "rationale": "Best channel should be mostly a function of the budget composition "
                 "(n_identity, n_labels), which the scalar summary (mean/variance/MI of "
                 "T) does not encode -- the witness shows exact non-identifiability. So "
                 "a fitted summary-only rule should retain a substantial gap, budget "
                 "knowledge should be near-sufficient, and the shape-based point values "
                 "(which encode budgets through the posterior) should be mostly right, "
                 "with FMoW's selected-state failure possibly visible as a higher "
                 "model-pick error there.",
    "P1_belief_only_pos_frac": {"point": 0.35, "range": [0.25, 0.50]},
    "P2_model_pick_pos_frac": {"point": 0.15, "range": [0.05, 0.30]},
    "P3_budget_only_pos_frac": {"point": 0.02, "range": [0.0, 0.05]},
    "P4_median_rel_gap_belief_ge_0.03": True,
    "P5_frac_rel_gap_ge_0.10_belief_ge_0.15": True,
    "P6_cells_unanimous_ge_0.90_per_dataset": True,
    "P7_model_pos_frac_diff_between_datasets_lt_0.20": True,
    "practitioner_threshold": "relative gap >= 10% of the state's initial squared error",
    "falsified_if": "belief-only gap is indistinguishable from budget-only (summary as "
                    "good as knowing what you acquired), or model-pick error exceeds "
                    "belief-only",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_frozen():
    assert sha(ROOT / "real/beta_acquisition.py") == BETA_SHA, "beta_acquisition.py drifted"
    for spec in DATASETS.values():
        for f, want in spec["sha"].items():
            assert sha(ROOT / spec["run"] / f) == want, f"frozen input changed: {f}"


def heldout_population(spec):
    df = pd.read_csv(ROOT / spec["run"] / "population_rates.csv")
    h = df[df.split == "heldout"]
    n = h.set_index("source")["n"].astype(float)
    r = h.set_index("source")["target_rate"].astype(float)
    return {int(s): dict(n=n[s], rate=r[s]) for s in h.source}, float(n.sum())


def t_star(pop, N):
    return sum(p["n"] * p["rate"] for p in pop.values()) / N


def exact_errors(shapes_list, pop, N, T):
    """Exact one-step expected squared error per channel, the frozen protocol's
    definition (wilds_counterexample.py): source_id draws the source with the held-out
    source weight; reference_label is STRATIFIED with equal weight 1/2 per source and
    the held-out target rate of that source."""
    out = np.empty((len(shapes_list), 2))
    for i, sh in enumerate(shapes_list):
        e_id = e_lab = 0.0
        for s, p in pop.items():
            up = sh.copy(); up[0, 1 - s] += 1               # observe source s
            e_id += (p["n"] / N) * (moments(up)[0] - T) ** 2
            for y, py in ((1, p["rate"]), (0, 1 - p["rate"])):
                up = sh.copy(); up[1 + s, 0 if y else 1] += 1
                e_lab += 0.5 * py * (moments(up)[0] - T) ** 2
        out[i] = (e_id, e_lab)
    return out


ANALYSIS_FIXES = [
    "2026-11-10: first invocation used population weights n_s/N for the label channel; "
    "the built-in validation against heldout_exact_one_step.csv failed before any gap "
    "was computed. The frozen protocol (wilds_counterexample.py lines 97-102) stratifies "
    "labels with weight 1/2 per source; corrected to that definition. PRED was already "
    "on disk and is unchanged.",
    "2026-11-10: the budget_only cell-majority was inverted in the first completed run "
    "(picked source_id when label was the training majority), giving a worse-than-chance "
    "0.794 positive-gap fraction that flagged the bug. Rule as frozen in PRED is "
    "unchanged; implementation corrected. belief_only and model_pick were unaffected "
    "(identical numbers across the two runs).",
]


def validate_evaluator(spec, pop, N, T):
    held = pd.read_csv(ROOT / spec["run"] / "heldout_exact_one_step.csv")
    cand = pd.read_csv(ROOT / spec["run"] / "candidate_states.csv").set_index("state_id")
    shapes = [np.array(json.loads(cand.loc[int(r.state_id)].shapes), float)
              for _, r in held.iterrows()]
    got = exact_errors(shapes, pop, N, T)
    want = held.pivot(index="state_id", columns="action", values="expected_squared_error")
    for k, (_, r) in enumerate(held.iterrows()):
        w = want.loc[int(r.state_id)]
        assert abs(got[k, 0] - w["source_id"]) <= 1e-9, \
            f"evaluator mismatch (identity) state {r.state_id}: {got[k, 0]} vs {w['source_id']}"
        assert abs(got[k, 1] - w["reference_label"]) <= 1e-9, \
            f"evaluator mismatch (label) state {r.state_id}"
    init = held.groupby("state_id").initial_squared_error.first()
    for sid in init.index:
        m = moments(np.array(json.loads(cand.loc[int(sid)].shapes), float))[0]
        assert abs((m - T) ** 2 - init.loc[sid]) <= 1e-9, f"initial-error mismatch {sid}"


def out_of_fold_picks(cand):
    """belief_only (logistic CV), budget_only (cell majority CV); frozen rules."""
    X = cand[["mean", "variance", "mi"]].to_numpy()
    best = np.where(cand.e_identity <= cand.e_label, 0, 1)     # 0 = source_id
    cell = (cand.n_identity.astype(str) + "/" + cand.n_labels_per_source.astype(str)).to_numpy()
    belief = np.empty(len(cand), int)
    budget = np.empty(len(cand), int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, best):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=1000, random_state=0))])
        pipe.fit(X[tr], best[tr])
        p = pipe.predict_proba(X[te])[:, list(pipe.classes_).index(0)]
        belief[te] = np.where(p >= 0.5, 0, 1)                  # p==0.5 -> source_id
        maj = pd.Series(best[tr]).groupby(cell[tr]).mean()     # share of train states with label best
        for i in te:
            share_label = maj.get(cell[i], 0.5)
            budget[i] = 1 if share_label > 0.5 else 0          # tie/empty -> source_id
    return belief, budget


def summarise(gaps, rel, label):
    pos = gaps > 1e-9
    return dict(pick=label, n=len(gaps), pos_frac=float(pos.mean()),
                median_gap=float(np.median(gaps)), median_rel_gap=float(np.median(rel[pos])) if pos.any() else 0.0,
                p90_rel_gap=float(np.percentile(rel[pos], 90)) if pos.any() else 0.0,
                max_rel_gap=float(rel[pos].max()) if pos.any() else 0.0,
                frac_rel_ge_threshold=float((rel >= REL_THRESHOLD).mean()))


def main(out_dir=None):
    out = Path(out_dir) if out_dir else ROOT / "workbook" / (
        "acquisition_gap_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "Z")
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "pred.json"
    if pred_path.exists():
        assert json.loads(pred_path.read_text()) == PRED
    else:
        pred_path.write_text(json.dumps(PRED, indent=2))
        print("PRED frozen to", pred_path, "-- before any gap computation")

    verify_frozen()
    all_rows, summaries, score = [], [], {}
    for name, spec in DATASETS.items():
        pop, N = heldout_population(spec)
        T = t_star(pop, N)
        validate_evaluator(spec, pop, N, T)
        print(f"{name}: evaluator reproduces the frozen held-out rows exactly "
              f"(T* = {T:.6f}, N = {N:.0f})")
        cand = pd.read_csv(ROOT / spec["run"] / "candidate_states.csv")
        shapes = [np.array(json.loads(s), float) for s in cand.shapes]
        E = exact_errors(shapes, pop, N, T)
        vi, vl = action_values(np.stack(shapes))
        assert np.allclose(vi, cand.value_identity, rtol=1e-9)
        assert np.allclose(vl, cand.value_label, rtol=1e-9)
        cand["e_identity"], cand["e_label"] = E[:, 0], E[:, 1]
        cand["initial_err"] = (cand["mean"] - T) ** 2
        belief, budget = out_of_fold_picks(cand)
        model = np.where(cand.value_identity >= cand.value_label, 0, 1)  # tie -> source_id
        best = np.where(cand.e_identity <= cand.e_label, 0, 1)
        tie = cand.e_identity == cand.e_label

        cell_key = cand.n_identity.astype(str) + "/" + cand.n_labels_per_source.astype(str)
        unanimous = (cand.groupby(cell_key).apply(
            lambda g: g.e_identity.eq(g.e_label).all()
            or g.e_identity.lt(g.e_label).all() or g.e_identity.gt(g.e_label).all(),
            include_groups=False))
        for pick, col in (("belief_only", belief), ("budget_only", budget), ("model_pick", model)):
            cand[f"gap_{pick}"] = np.where(col == 0, cand.e_identity, cand.e_label) \
                                  - np.minimum(cand.e_identity, cand.e_label)
        cand["dataset"], cand["belief_pick"], cand["budget_pick"] = name, belief, budget
        cand["model_pick"], cand["best"], cand["tie"] = model, best, tie
        all_rows.append(cand)

        for pick in ("belief_only", "budget_only", "model_pick"):
            rel = cand[f"gap_{pick}"].to_numpy() / cand.initial_err.to_numpy().clip(1e-18)
            s = summarise(cand[f"gap_{pick}"].to_numpy(), rel, pick)
            s["dataset"] = name
            summaries.append(s)
        score[name] = dict(
            cells_unanimous=float(unanimous.mean()),
            belief_pos_frac=float((cand.gap_belief_only > 1e-9).mean()),
            budget_pos_frac=float((cand.gap_budget_only > 1e-9).mean()),
            model_pos_frac=float((cand.gap_model_pick > 1e-9).mean()),
            median_rel_gap_belief=float(np.median(
                (cand.gap_belief_only / cand.initial_err)[cand.gap_belief_only > 1e-9])) if (cand.gap_belief_only > 1e-9).any() else 0.0,
            frac_rel_ge_thr_belief=float(((cand.gap_belief_only / cand.initial_err)
                                          >= REL_THRESHOLD).mean()),
            exact_ties=int(tie.sum()))
        print(f"  {name}: belief-only pos {score[name]['belief_pos_frac']:.3f} | "
              f"budget-only {score[name]['budget_pos_frac']:.3f} | model "
              f"{score[name]['model_pos_frac']:.3f} | cells unanimous "
              f"{score[name]['cells_unanimous']:.2f}")

    G = pd.concat(all_rows, ignore_index=True)
    # exploratory (not in PRED): is the model pick reliable where it is decisive? Uses
    # the frozen protocol's own qualifying criterion, action_margin >= 2.
    vmax = G[["value_identity", "value_label"]].max(1)
    vmin = G[["value_identity", "value_label"]].min(1)
    G["action_margin"] = np.where(vmin > 0, vmax / vmin.clip(lower=1e-300), np.inf)
    G["decisive"] = G.action_margin >= 2.0
    expl = {}
    for name, g in list(G.groupby("dataset")) + [("POOLED", G)]:
        d, u = g[g.decisive], g[~g.decisive]
        expl[name] = dict(
            n_decisive=int(len(d)), n_indecisive=int(len(u)),
            model_pos_frac_decisive=float((d.gap_model_pick > 1e-9).mean()),
            model_pos_frac_indecisive=float((u.gap_model_pick > 1e-9).mean()) if len(u) else None,
            belief_pos_frac_decisive=float((d.gap_belief_only > 1e-9).mean()),
            budget_pos_frac_decisive=float((d.gap_budget_only > 1e-9).mean()))
    G.to_csv(out / "gaps.csv", index=False)
    pooled = {}
    for pick in ("belief_only", "budget_only", "model_pick"):
        rel = (G[f"gap_{pick}"] / G.initial_err.clip(1e-18)).to_numpy()
        pooled[pick] = summarise(G[f"gap_{pick}"].to_numpy(), rel, pick)
    S = pd.DataFrame(summaries + [{**p, "dataset": "POOLED"} for p in pooled.values()])
    S.to_csv(out / "summary.csv", index=False)

    verdict = {
        "P1_belief_only_pos_frac": pooled["belief_only"]["pos_frac"],
        "P2_model_pick_pos_frac": pooled["model_pick"]["pos_frac"],
        "P3_budget_only_pos_frac": pooled["budget_only"]["pos_frac"],
        "P4_median_rel_gap_belief_ge_0.03": pooled["belief_only"]["median_rel_gap"] >= 0.03,
        "P5_frac_rel_gap_ge_0.10_belief_ge_0.15":
            pooled["belief_only"]["frac_rel_ge_threshold"] >= 0.15,
        "P6_cells_unanimous_ge_0.90_per_dataset":
            all(v["cells_unanimous"] >= 0.90 for v in score.values()),
        "P7_model_pos_frac_diff_between_datasets_lt_0.20":
            abs(score["camelyon17"]["model_pos_frac"] - score["fmow"]["model_pos_frac"]) < 0.20,
    }
    passed = {
        "P1": PRED["P1_belief_only_pos_frac"]["range"][0] <= verdict["P1_belief_only_pos_frac"]
              <= PRED["P1_belief_only_pos_frac"]["range"][1],
        "P2": PRED["P2_model_pick_pos_frac"]["range"][0] <= verdict["P2_model_pick_pos_frac"]
              <= PRED["P2_model_pick_pos_frac"]["range"][1],
        "P3": PRED["P3_budget_only_pos_frac"]["range"][0] <= verdict["P3_budget_only_pos_frac"]
              <= PRED["P3_budget_only_pos_frac"]["range"][1],
        "P4": verdict["P4_median_rel_gap_belief_ge_0.03"] == PRED["P4_median_rel_gap_belief_ge_0.03"],
        "P5": verdict["P5_frac_rel_gap_ge_0.10_belief_ge_0.15"] == PRED["P5_frac_rel_gap_ge_0.10_belief_ge_0.15"],
        "P6": verdict["P6_cells_unanimous_ge_0.90_per_dataset"] == PRED["P6_cells_unanimous_ge_0.90_per_dataset"],
        "P7": verdict["P7_model_pos_frac_diff_between_datasets_lt_0.20"]
              == PRED["P7_model_pos_frac_diff_between_datasets_lt_0.20"],
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    colors = {"belief_only": "#c44e52", "budget_only": "#4c72b0", "model_pick": "#55a868"}
    for ax, name in zip(axes, ["camelyon17", "fmow"]):
        g = G[G.dataset == name]
        for pick in ("belief_only", "budget_only", "model_pick"):
            rel = np.sort((g[f"gap_{pick}"] / g.initial_err.clip(1e-18)).to_numpy())
            ax.plot(rel, np.arange(1, len(rel) + 1) / len(rel), label=pick,
                    color=colors[pick], lw=1.8 if pick == "belief_only" else 1.2)
        ax.axvline(REL_THRESHOLD, color="k", ls=":", lw=0.8)
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_xlabel("gap / initial squared error (symlog)")
        ax.set_title(name)
    axes[0].set_ylabel("ECDF over 600 states")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Acquisition gap: penalty of each pick vs the best channel "
                 f"(dotted = practitioner threshold {REL_THRESHOLD:.0%})", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "acquisition_gap.png", dpi=150, bbox_inches="tight")

    manifest = dict(script_sha256=sha(Path(__file__)),
                    branch=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip(),
                    inputs={d: {f: sha(ROOT / s["run"] / f) for f in s["sha"]} for d, s in DATASETS.items()},
                    per_dataset=score, pooled=pooled,
                    verdict=verdict, passed=passed,
                    overall=("ALL PRED PASS" if all(passed.values()) else "PRED FAILED"),
                    analysis_fixes=ANALYSIS_FIXES,
                    exploratory_model_pick_by_action_margin=expl,
                    tie_rule=TIE, rel_threshold=REL_THRESHOLD,
                    evaluator="exact enumeration over frozen held-out counts; validated "
                              "against all 8 heldout_exact_one_step rows to 1e-9",
                    python=sys.version.split()[0])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(Path(__file__), out / "acquisition_gap.py")

    print("\n== pooled ==")
    for p in pooled.values():
        print(f"  {p['pick']:12s} pos {p['pos_frac']:.3f} | median gap {p['median_gap']:.2e} "
              f"| median rel {p['median_rel_gap']:.3f} | p90 rel {p['p90_rel_gap']:.3f} "
              f"| max rel {p['max_rel_gap']:.3f} | rel>=10%: {p['frac_rel_ge_threshold']:.3f}")
    print("\n== exploratory: model pick by action margin (protocol criterion >= 2) ==")
    for name, e in expl.items():
        print(f"  {name:11s} decisive n={e['n_decisive']:3d}: model wrong "
              f"{e['model_pos_frac_decisive']:.3f} (belief {e['belief_pos_frac_decisive']:.3f}, "
              f"budget {e['budget_pos_frac_decisive']:.3f}) | indecisive n={e['n_indecisive']:3d}: "
              f"model wrong {e['model_pos_frac_indecisive']}")
    print("\n== PRED ==")
    for (k, ok), (vk, vv) in zip(passed.items(), verdict.items()):
        print(f"  {k}: {'PASS' if ok else 'FAIL'}  ({vk} = {vv})")
    print("  overall:", manifest["overall"])
    print("\nwritten:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    main(ap.parse_args().out)
