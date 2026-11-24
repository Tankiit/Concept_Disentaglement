"""Replay the saved Tiny ImageNet input arm and vary one constraint at a time."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent / "TinyImageNet_REACH_Results_v2full"
sys.path.insert(0, str(OUT.parent / "tiny_imagenet_notebook_source"))
import tiny_four_channel_executed as pilot


def summarize(intervals):
    keys = ["regime", "target", "arm", "budget"]
    by_setting = intervals.groupby(keys)
    table = by_setting.agg(
        mean_lower=("lower", "mean"), mean_upper=("upper", "mean"),
        mean_width=("width", "mean"), min_width=("width", "min"),
        max_width=("width", "max"), n=("repeat", "count"),
        contained=("contains_truth", "sum"),
        denominator_fallbacks=("status", lambda s: (s == "undefined_denominator_policy").sum()),
    )
    saved = pd.read_csv(RESULTS / "summary.csv").set_index(keys).sort_index()
    assert saved.index.equals(table.index)
    cols = ["mean_width", "min_width", "max_width", "n"]
    np.testing.assert_allclose(table[cols], saved[cols], atol=1e-10)
    np.testing.assert_allclose(table.contained / table.n, saved.containment_fraction)
    assert intervals.status.isin(["ok", "undefined_denominator_policy"]).all()
    np.testing.assert_array_equal(intervals.contains_truth,
                                  (intervals.lower <= intervals.truth + 1e-8)
                                  & (intervals.truth <= intervals.upper + 1e-8))
    assert len(intervals) == 720 and (table.n == 6).all()
    table.to_csv(OUT / "all_settings.csv")


def replay(audit, saved, benchmarks):
    fit = np.array(audit["fit_source_laws"])
    weights = np.array(audit["generating_mixture_weights"])
    c = audit["probe_cell"]
    alpha = audit["alpha_fixed_setting"]
    base = audit["baseline"]
    seed = 20261029
    d = fit.shape[1]
    num = np.zeros(3*d)
    num[c] = 1
    den = num.copy()
    den[d+c] = 1
    rows = []
    for regime in ["image_derived", "confounded"]:
        A = fit.copy()
        if regime == "confounded":
            A[2] = A[:2].mean(axis=0)
        p = weights @ A
        # Exact input marginals identify the two independent effective weights.
        # In the confounded law w2=t, w0=.60-t/2 and w1=.40-t/2.
        if regime == "image_derived":
            assert np.linalg.matrix_rank((A[:2]-A[2]).T) == 2
            endpoint_weights = [weights]
        else:
            effective = weights[:2] + weights[2]/2
            endpoint_weights = [np.r_[effective-t/2, t] for t in (0, 2*effective.min())]
        endpoints = [pilot.true_targets(A, np.array(audit["fit_error_rates"]), w, c)
                     for w in endpoint_weights]
        for target in ["known_membership", "candidate_membership"]:
            vals = [v[target] for v in endpoints]
            benchmark = benchmarks.loc[(regime, target)]
            np.testing.assert_allclose([min(vals), max(vals)], [benchmark.lower, benchmark.upper], atol=1e-8)
        for rep in range(audit["repeats"]):
            rng = np.random.default_rng(pilot.seed_for(seed, regime, rep, "baseline"))
            inputs = rng.multinomial(base["inputs"], p)
            rng.multinomial(base["identities"], weights)
            refs = [rng.multinomial(base["references_per_known_source"], row) for row in A[:2]]
            ref_boxes = [pilot.frequency_box(counts, alpha/8) for counts in refs]
            ref_lo, ref_hi = (np.vstack([ref_boxes[0][i], ref_boxes[1][i], A[2]]) for i in (0, 1))
            stream = np.random.default_rng(pilot.seed_for(seed, regime, rep, "inputs", "stream"))
            prev = 0
            for budget in audit["budgets"]:
                if budget > prev:
                    inputs += stream.multinomial(budget - prev, p)
                prev = budget
                marginal_box = pilot.frequency_box(inputs, alpha/4)
                for refs_kind, row_box in [("sampled", (ref_lo, ref_hi)), ("exact", (A, A))]:
                    for p_kind, p_box in [("sampled", marginal_box), ("exact", (p, p))]:
                        constraints = pilot.joint_constraints(*row_box, *p_box)
                        lo, hi, status = pilot.sharp_ratio(*constraints, num, den)
                        # The transformed LP imposes den @ y = 1. Bypassing only
                        # the zero-denominator screen projects over positive mass.
                        pos_lo, pos_hi, pos_status = pilot.sharp_ratio(*constraints, num, den, eps=-np.inf)
                        assert pos_status == "ok" and status in ("ok", "undefined_denominator_policy")
                        if refs_kind == p_kind == "sampled":
                            original = saved.loc[(regime, rep, budget)]
                            np.testing.assert_allclose([lo, hi], [original.lower, original.upper], atol=1e-8, rtol=0)
                            assert status == original.status
                        if refs_kind == p_kind == "exact":
                            benchmark = benchmarks.loc[(regime, "known_membership")]
                            np.testing.assert_allclose([lo, hi], [benchmark.lower, benchmark.upper], atol=1e-8)
                        rows.append(dict(regime=regime, repeat=rep, budget=budget,
                                         references=refs_kind, marginal=p_kind, status=status,
                                         lower=lo, upper=hi, width=hi-lo,
                                         positive_lower=pos_lo, positive_upper=pos_hi,
                                         positive_width=pos_hi-pos_lo))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    intervals = pd.read_csv(RESULTS / "intervals.csv")
    summarize(intervals)
    benchmarks = pd.read_csv(RESULTS / "exact_identification_benchmarks.csv").set_index(["regime", "target"])
    inputs = intervals.query("target == 'known_membership' and arm == 'inputs'")
    result = replay(json.loads((RESULTS / "audit.json").read_text()), inputs.set_index(["regime", "repeat", "budget"]), benchmarks)
    result.to_csv(OUT / "known_membership_diagnostics.csv", index=False)
    print(result.query("budget == 100000").groupby(["regime", "references", "marginal"])
          .agg(width=("width", "mean"), positive_width=("positive_width", "mean"),
               fallbacks=("status", lambda s: (s != "ok").sum())).to_string())
