# gap_margin_check.py -- the three checks on the gap-by-verdict result (collaborator,
# 10 Nov evening): (1) are the gaps above estimation noise, or are fragile states just
# near-ties where the choice does not matter? (2) honest intervals under shared held-out
# rows and shared state structure; (3) CAMELYON17 as the result, FMoW uninformative
# under the primary family.
#
# Noise scale: per-state bootstrap SE of (E_id - E_lab), resampling the underlying
# held-out ROWS (multinomial over the four (source, outcome) cells). A pick counts as
# wrong at margin k only if gap > k * SE_state. The same row bootstrap gives a
# percentile CI for the robust-vs-fragile wrong-rate difference (point 2, population
# side); a budget-cell cluster bootstrap addresses state-set dependence (point 2,
# state side).
#
# PRED below was frozen before any SE, margin table, or interval was computed.
# Run from REACH_AISTATS on branch reach-mateus-workbook:
#   OMP_NUM_THREADS=1 python -m real.gap_margin_check [--out DIR]
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

ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    "camelyon17": dict(run="workbook/camelyon17_counterexample_20261030T092024988890Z"),
    "fmow": dict(run="workbook/fmow_counterexample_20261030T094929883846Z"),
}
SWEEP = ROOT / "workbook/credal_action_band_20261110T132741611229Z/candidate_sweep.csv"
HELD = "heldout_exact_one_step.csv"
CAND = "candidate_states.csv"
POP = "population_rates.csv"
KS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
B = 2000
FAMILIES = ("tv_ball", "stress_fixed_strength")

# ---------------------------------------------------------------------------
# PRED -- frozen 2026-11-10, before any noise scale or margin table was computed.
# ---------------------------------------------------------------------------
PRED = {
    "written": "2026-11-10, before any SE / margin / interval computation",
    "question": "At a 2-SE margin, does the 26-point CAMELYON17 robust-vs-fragile "
                "wrong-rate difference survive, shrink, or vanish?",
    "prediction": "SURVIVES (at most mild shrink). Rationale: the held-out rates are "
                  "estimated from ~35k rows (SE(r0) ~ 0.0023, SE(w) ~ 0.0026) and the "
                  "T*-dependent parts of E_id - E_lab largely cancel, so per-state "
                  "SE(dE) should sit well below the typical gap (~1e-3); if so, "
                  "fragile wrongs are NOT sub-noise and the dull near-tie explanation "
                  "is mostly rejected. Risk to this prediction: if SE ~ 5e-4 the 2-SE "
                  "margin bites and the difference shrinks.",
    "Q1_camelyon_tvball_diff_at_k2_points_in_[12,26]": True,
    "Q2_rowboot_CI_at_k2_excludes_0_camelyon": True,
    "Q3_fragile_wrongrate_at_k2_ge_0.45_camelyon": True,
    "Q4_dull_rejected_fragile_rate_at_k1_ge_0.50_camelyon": True,
    "Q5_cellcluster_CI_at_k0_excludes_0_camelyon": True,
    "Q6_cond_gap_median_ratio_fragile_over_robust_in_[0.7,1.5]_camelyon": True,
    "Q7_fmow_stress_diff_at_k2_lt_10_points": True,
    "paper_rule": "if Q1 and Q2 hold: report CAMELYON17 alone as the result (primary "
                  "family), FMoW as uninformative; if Q2 fails: report as suggestive "
                  "with the near-tie caveat leading; if Q1 fails: the band marks "
                  "where the choice does not matter -- publish as that negative.",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def population_cells(spec):
    """Held-out rows as (source, outcome) cell counts, reconstructed exactly from
    population_rates.csv: n_s and rate_s give n_{s,1} = n_s * rate_s (integers)."""
    df = pd.read_csv(ROOT / spec["run"] / POP)
    h = df[df.split == "heldout"].set_index("source")
    cells = {}
    for s in h.index:
        n, r = float(h.loc[s, "n"]), float(h.loc[s, "target_rate"])
        n1 = n * r
        assert abs(n1 - round(n1)) < 1e-3, "rate*n not integral; row reconstruction fails"
        cells[int(s)] = dict(n0=n - round(n1), n1=round(n1))
    return cells


def state_arrays(spec):
    cand = pd.read_csv(ROOT / spec["run"] / CAND)
    ab = np.array([json.loads(s) for s in cand.shapes], dtype=float)     # (S,3,2)
    return cand, ab[:, :, 0], ab[:, :, 1]                                  # a, b


def errors(a, b, w, rates, T):
    """Vectorized exact one-step errors (the frozen evaluator's definition):
    identity weighted by w_s, label stratified 1/2 per source at rate r_s."""
    m = a / (a + b)
    n0, n1, n2 = (a + b)[:, 0], (a + b)[:, 1], (a + b)[:, 2]
    e_id = np.zeros(len(a))
    e_lab = np.zeros(len(a))
    for s in (0, 1):
        m0s = (a[:, 0] + s) / (n0 + 1)                    # observe source s (s=1 -> a+1)
        mean_id = (1 - m0s) * m[:, 1] + m0s * m[:, 2]
        e_id += w[s] * (mean_id - T) ** 2
        ms = s + 1                                        # Theta component index
        n_ms = (a + b)[:, ms]
        for y in (0, 1):
            m_y = (a[:, ms] + y) / (n_ms + 1)
            mean_y = (1 - m[:, 0]) * (m_y if ms == 1 else m[:, 1]) \
                     + m[:, 0] * (m_y if ms == 2 else m[:, 2])
            e_lab += 0.5 * ((rates[s] if y else 1 - rates[s]) * (mean_y - T) ** 2)
    return e_id, e_lab


def validate(spec, a, b, w, rates, T):
    held = pd.read_csv(ROOT / spec["run"] / HELD)
    cand = pd.read_csv(ROOT / spec["run"] / CAND).set_index("state_id")
    e_id, e_lab = errors(a, b, w, rates, T)
    for _, r in held.iterrows():
        i = cand.index.get_loc(int(r.state_id))
        want = float(r.expected_squared_error)
        got = e_id[i] if r.action == "source_id" else e_lab[i]
        assert abs(got - want) <= 1e-9, f"evaluator drift state {r.state_id} {r.action}"
    m = a / (a + b)
    mean_T = (1 - m[:, 0]) * m[:, 1] + m[:, 0] * m[:, 2]
    for sid, ie in held.groupby("state_id").initial_squared_error.first().items():
        assert abs((mean_T[cand.index.get_loc(int(sid))] - T) ** 2 - ie) <= 1e-9


def main(out_dir=None):
    out = Path(out_dir) if out_dir else ROOT / "workbook" / (
        "gap_margin_check_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "Z")
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "pred.json"
    if pred_path.exists():
        assert json.loads(pred_path.read_text()) == PRED
    else:
        pred_path.write_text(json.dumps(PRED, indent=2))
        print("PRED frozen to", pred_path, "-- before any noise/margin computation")

    sweep = pd.read_csv(SWEEP)[["dataset", "state_id", "family", "verdict"]]
    rng = np.random.default_rng(0)
    results, rows = {}, []
    for name, spec in RUNS.items():
        cells = population_cells(spec)
        N = int(sum(c["n0"] + c["n1"] for c in cells.values()))
        p_cells = np.array([cells[s][k] / N for s in (0, 1) for k in ("n1", "n0")])  # order (s,y)
        cand, a, b = state_arrays(spec)
        w0 = (cells[0]["n0"] + cells[0]["n1"]) / N
        w = {0: w0, 1: 1 - w0}
        rates = {s: cells[s]["n1"] / (cells[s]["n0"] + cells[s]["n1"]) for s in (0, 1)}
        T = sum(w[s] * rates[s] for s in (0, 1))
        validate(spec, a, b, w, rates, T)
        e_id, e_lab = errors(a, b, w, rates, T)
        print(f"{name}: evaluator validated; T* = {T:.6f}, N = {N}")

        pick = np.where(cand.value_identity >= cand.value_label, 0, 1)
        gap = np.where(pick == 0, e_id, e_lab) - np.minimum(e_id, e_lab)
        dE = e_id - e_lab

        # row bootstrap: multinomial over the four (source, outcome) cells
        cnt = np.array([cells[s][k] for s in (0, 1) for k in ("n1", "n0")], float)
        boots = rng.multinomial(N, cnt / N, size=B).astype(float)          # (B,4)
        dE_b = np.empty((B, len(a)))
        gap_b = np.empty((B, len(a)))
        for i, nb in enumerate(boots):
            n_s = {s: nb[2 * s] + nb[2 * s + 1] for s in (0, 1)}
            wb = {s: n_s[s] / N for s in (0, 1)}
            rb = {s: nb[2 * s] / n_s[s] for s in (0, 1)}
            Tb = sum(wb[s] * rb[s] for s in (0, 1))
            ei, el = errors(a, b, wb, rb, Tb)
            dE_b[i] = ei - el
            gap_b[i] = np.where(pick == 0, ei, el) - np.minimum(ei, el)
        se = dE_b.std(0)
        print(f"  per-state SE(dE): median {np.median(se):.2e}, p90 {np.percentile(se, 90):.2e}"
              f" | gap median (wrong states) {np.median(gap[gap > 1e-9]):.2e}")

        v = sweep[(sweep.dataset == name)][["state_id", "family", "verdict"]]
        res_ds = {}
        for fam in FAMILIES:
            verd = v[v.family == fam].set_index("state_id").verdict
            frag = (verd.reindex(cand.state_id.to_numpy()).to_numpy() == "fragile")
            for k in KS:
                thr = k * se
                wrong = gap > thr
                wrong_b = gap_b > thr                                 # (B,S), same thr
                wr = wrong[~frag].mean(); wf = wrong[frag].mean() if frag.any() else np.nan
                diff_b = wrong_b[:, frag].mean(1) - wrong_b[:, ~frag].mean(1)
                lo, hi = np.percentile(diff_b, [2.5, 97.5])
                rows.append(dict(dataset=name, family=fam, k=k,
                                 n_fragile=int(frag.sum()), n_robust=int((~frag).sum()),
                                 wrong_robust=float(wr), wrong_fragile=float(wf),
                                 diff_points=100 * (wf - wr),
                                 ci_lo_points=100 * lo, ci_hi_points=100 * hi))
                if (name, fam) == ("camelyon17", "tv_ball"):
                    res_ds[f"k={k}"] = dict(wrong_robust=float(wr), wrong_fragile=float(wf),
                                            diff_points=100 * (wf - wr),
                                            ci=(100 * lo, 100 * hi))
                if k == 2.0 and fam == "stress_fixed_strength" and name == "fmow":
                    res_ds["fmow_stress_k2_diff"] = 100 * (wf - wr)
        # conditional gap stats (k=0 wrongs) + cell-cluster bootstrap at k=0
        frag = (sweep[(sweep.dataset == name) & (sweep.family == "tv_ball")]
                .set_index("state_id").verdict.reindex(cand.state_id.to_numpy()).to_numpy()
                == "fragile")
        wmask = gap > 1e-9
        res_ds["cond_gap_median_fragile"] = float(np.median(gap[wmask & frag])) if (wmask & frag).any() else 0.0
        res_ds["cond_gap_median_robust"] = float(np.median(gap[wmask & ~frag]))
        res_ds["se_median"] = float(np.median(se))
        cell = (cand.n_identity.astype(str) + "/" + cand.n_labels_per_source.astype(str)).to_numpy()
        ucells = np.unique(cell)
        diffs = []
        for _ in range(B):
            pickc = rng.choice(ucells, size=len(ucells), replace=True)
            idx = np.concatenate([np.flatnonzero(cell == c) for c in pickc])
            f = frag[idx]
            d = (wmask[idx][f].mean() - wmask[idx][~f].mean()) if f.any() and (~f).any() else 0.0
            diffs.append(d)
        res_ds["cellcluster_ci_k0"] = tuple(100 * np.percentile(diffs, [2.5, 97.5]))
        results[name] = res_ds

    pd.DataFrame(rows).to_csv(out / "margin_table.csv", index=False)
    c = results["camelyon17"]
    got = {
        "Q1_camelyon_tvball_diff_at_k2_points_in_[12,26]":
            bool(12 <= c["k=2.0"]["diff_points"] <= 26),
        "Q2_rowboot_CI_at_k2_excludes_0_camelyon": bool(c["k=2.0"]["ci"][0] > 0),
        "Q3_fragile_wrongrate_at_k2_ge_0.45_camelyon":
            bool(c["k=2.0"]["wrong_fragile"] >= 0.45),
        "Q4_dull_rejected_fragile_rate_at_k1_ge_0.50_camelyon":
            bool(c["k=1.0"]["wrong_fragile"] >= 0.50),
        "Q5_cellcluster_CI_at_k0_excludes_0_camelyon": bool(c["cellcluster_ci_k0"][0] > 0),
        "Q6_cond_gap_median_ratio_fragile_over_robust_in_[0.7,1.5]_camelyon":
            bool(0.7 <= c["cond_gap_median_fragile"] / max(c["cond_gap_median_robust"], 1e-18) <= 1.5),
        "Q7_fmow_stress_diff_at_k2_lt_10_points":
            bool(abs(results["fmow"].get("fmow_stress_k2_diff", 99)) < 10),
    }
    passed = {k: bool(got[k]) == PRED[k] for k in got}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for fam, color in (("tv_ball", "#c44e52"), ("stress_fixed_strength", "#4c72b0")):
        t = tab[(tab.dataset == "camelyon17") & (tab.family == fam)].sort_values("k")
        ax.plot(t.k, t.diff_points, "o-", label=f"{fam}", color=color)
        ax.fill_between(t.k, t.ci_lo_points, t.ci_hi_points, alpha=0.15, color=color)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("margin k (pick wrong only if gap > k * bootstrap SE of the "
                  "channel-error difference)")
    ax.set_ylabel("fragile - robust wrong rate (points)")
    ax.set_title("CAMELYON17: does the band result survive noise margins? "
                 "(shaded = row-bootstrap 95% CI)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "margin_check.png", dpi=150)

    manifest = dict(script_sha256=sha(Path(__file__)),
                    branch=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip(),
                    inputs={"candidate_sweep.csv": sha(SWEEP)} |
                            {f"{d}/{f}": sha(ROOT / s["run"] / f) for d, s in RUNS.items()
                             for f in (CAND, POP, HELD)},
                    results=results, got=got, passed=passed,
                    overall=("ALL PRED PASS" if all(passed.values()) else "PRED FAILED"),
                    bootstrap=dict(B=B, unit="held-out rows (multinomial over 4 cells) "
                                  "+ budget-cell cluster for state dependence",
                                  caveat="state-set dependence beyond the shared "
                                         "population is addressed only by the cluster "
                                         "bootstrap"),
                    paper_rule=PRED["paper_rule"],
                    python=sys.version.split()[0])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(Path(__file__), out / "gap_margin_check.py")

    print("\n== CAMELYON17, tv_ball: wrong rate robust / fragile by margin k ==")
    for k in KS:
        r = c[f"k={k}"]
        print(f"  k={k:.1f}: {r['wrong_robust']:.3f} / {r['wrong_fragile']:.3f} "
              f"(diff {r['diff_points']:+.1f} pts, CI [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}])")
    print(f"  SE(dE) median {c['se_median']:.2e}; cond-gap median fragile "
          f"{c['cond_gap_median_fragile']:.2e} vs robust {c['cond_gap_median_robust']:.2e}")
    print(f"  cell-cluster CI at k=0: [{c['cellcluster_ci_k0'][0]:+.1f}, "
          f"{c['cellcluster_ci_k0'][1]:+.1f}] pts")
    print("\n== PRED ==")
    for k, ok in passed.items():
        print(f"  {k}: {'PASS' if ok else 'FAIL'} (got {got[k]})")
    print("  overall:", manifest["overall"])
    print("\nwritten:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    main(ap.parse_args().out)
