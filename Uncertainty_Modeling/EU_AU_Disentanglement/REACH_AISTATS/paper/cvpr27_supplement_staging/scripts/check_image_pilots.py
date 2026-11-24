"""Replay Appendix D/F pilot bounds using recorded fields, cell assignments and run-specific LP code."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PILOTS = {
    "CAMELYON17": "camelyon17_acquisition_20261030T092422959226Z",
    "FMoW": "fmow_acquisition_20261030T094932286096Z",
}
KEY = ["repeat", "action", "budget", "cell", "target"]
GROUP = ["action", "budget", "target"]
ACTIONS = ["inputs", "reference_inputs", "source_id", "target_labels"]
TARGETS = ["known_membership", "candidate_membership", "reference_label"]


def check(dataset, run_id):
    run = ROOT / "workbook" / run_id
    spec = importlib.util.spec_from_file_location("archived_bounds", run / "source_acquisition.py")
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    definition = json.loads((run / "bounds_definition.json").read_text())
    alpha = definition["alpha_per_box"]
    ref = pd.read_csv(run / "reference_cells.csv").set_index("record_id", verify_integrity=True)
    ev = pd.read_csv(run / "evaluation_cells.csv").set_index("record_id", verify_integrity=True)
    np.testing.assert_array_equal(ref.binary_target, ref.target == definition["event_class"])
    truth = {}
    for cell in range(4):
        inside = ev[ev.cell == cell]
        truth["candidate_membership", cell] = (inside.source == 2).mean()
        truth["known_membership", cell] = (inside.loc[inside.source < 2, "source"] == 0).mean()
        source = ref[ref.source < 2]
        mass = (source.cell == cell).groupby(source.source).mean()
        event_mass = ((source.cell == cell) & (source.binary_target == 1)).groupby(source.source).mean()
        truth["reference_label", cell] = event_mass.sum() / mass.sum()

    def box(stream, size, categories):
        return solver.frequency_box(np.bincount(stream[:size], minlength=categories), alpha)

    rows = []
    for stream in json.loads((run / "acquisition_record_ids.json").read_text()):
        references = [ref.loc[ids, "cell"].to_numpy() for ids in stream["reference"]]
        inputs = ev.loc[stream["inputs"], "cell"].to_numpy()
        identities = ev.loc[stream["identities"], "source"].to_numpy()
        labels = {key: ref.loc[ids, "binary_target"].to_numpy(dtype=int) for key, ids in stream["labels"].items()}
        for action in ACTIONS:
            for budget in (0, 32, 128, 512, 2048):
                nref = 128 + (budget // 3 if action == "reference_inputs" else 0)
                lo, hi = np.array([box(s, nref, 4) for s in references]).transpose(1, 0, 2)
                margin = box(inputs, 128 + (budget if action == "inputs" else 0), 4)
                weights = box(identities, 32 + (budget if action == "source_id" else 0), 3)
                nlab = 16 + (budget // 8 if action == "target_labels" else 0)
                for cell in range(4):
                    bounds = solver.membership_bounds(lo, hi, margin, weights, cell)
                    tl, th = np.array([box(labels[f"{s}:{cell}"], nlab, 2) for s in (0, 1)])[:, :, 1].T
                    bounds["reference_label"] = solver.anchored_label_bounds(lo, hi, tl, th, cell)
                    for target, value in bounds.items():
                        t = truth[target, cell]
                        lower, upper = value["lower"], value["upper"]
                        rows.append(dict(repeat=stream["repeat"], action=action, budget=budget, cell=cell,
                                         target=target, lower=lower, upper=upper, width=upper-lower,
                                         status=value["status"], empirical_target=t,
                                         contains=lower-1e-8 <= t <= upper+1e-8))
    replay = pd.DataFrame(rows).set_index(KEY, verify_integrity=True).sort_index()
    assert len(replay) == 1200
    saved = pd.read_csv(run / "acquisition_intervals.csv").set_index(KEY, verify_integrity=True).sort_index()
    pd.testing.assert_index_equal(replay.index, saved.index)
    np.testing.assert_allclose(replay[["lower","upper","width"]], saved[["lower","upper","width"]], atol=1e-8, rtol=0)
    np.testing.assert_array_equal(replay.status, saved.status)
    diagnostics = pd.read_csv(run / "empirical_target_diagnostics.csv").set_index(KEY).sort_index()
    pd.testing.assert_index_equal(replay.index, diagnostics.index)
    np.testing.assert_allclose(replay.empirical_target, diagnostics.empirical_target, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(replay.contains, diagnostics.contains_empirical_target)
    fallback = replay.status == "undefined_denominator_policy"
    np.testing.assert_equal(replay.loc[fallback, "lower"].to_numpy(), 0)
    np.testing.assert_equal(replay.loc[fallback, "upper"].to_numpy(), 1)
    assert replay.status.isin(["ok", "undefined_denominator_policy"]).all()
    groups = replay.assign(fallback=fallback).groupby(GROUP)
    stats = groups.agg(mean_lower=("lower","mean"), mean_upper=("upper","mean"), mean_width=("width","mean"),
                       n=("status","size"), undefined=("fallback","sum"), contained=("contains","sum"))
    assert (stats.n == 20).all()
    plot_data = pd.read_csv(ROOT / "figure_data" / f"{dataset.lower()}_pilot_summary.csv").set_index(GROUP).sort_index()
    pd.testing.assert_index_equal(stats.index, plot_data.index)
    np.testing.assert_allclose(stats.mean_width, plot_data.mean_width, atol=1e-8, rtol=0)
    np.testing.assert_allclose(stats.undefined / stats.n, plot_data.undefined_fraction)
    np.testing.assert_array_equal(plot_data.empty_fraction, np.zeros(len(plot_data)))
    archived = pd.read_csv(run / "empirical_containment_summary.csv").set_index(GROUP).sort_index()
    pd.testing.assert_index_equal(stats.index, archived.index)
    np.testing.assert_allclose(stats.contained/stats.n, archived.empirical_containment)
    np.testing.assert_array_equal(stats.n, archived.valid_intervals)
    endpoint = stats.xs(2048, level="budget").reindex([(a,t) for t in TARGETS for a in ACTIONS]).reset_index()
    endpoint.insert(0,"dataset",dataset)
    report = dict(dataset=dataset, run_id=run_id, interval_rows=len(replay), empty=int((replay.status=="empty").sum()),
                  denominator_fallbacks=int(fallback.sum()), known_id_contained=int(stats.loc[("source_id",2048,"known_membership"),"contained"]),
                  max_abs_endpoint_difference=float((saved[["lower","upper"]]-replay[["lower","upper"]]).abs().max().max()))
    return endpoint, report


if __name__ == "__main__":
    outputs = [check(dataset, run_id) for dataset, run_id in PILOTS.items()]
    pd.concat([r[0] for r in outputs]).to_csv(ROOT / "table_data" / "visual_endpoints_2048.csv",index=False)
    report = [r[1] for r in outputs]
    (ROOT / "provenance" / "pilot_replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
