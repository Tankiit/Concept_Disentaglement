"""Reproduce the fit-only selection and Appendix E.4 from the archived search records."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "analysis_outputs"
DATASETS = {"CAMELYON17": "camelyon17", "FMoW": "fmow"}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def relative_gap(a, b):
    return abs(a - b) / max((a + b) / 2, 1e-15)


def replay_selection(states, protocol):
    margin = protocol["action_margin"]
    identity = [s for s in states if s["value_identity"] >= margin * s["value_label"]]
    label = [s for s in states if s["value_label"] >= margin * s["value_identity"]]
    matches = []
    for i in identity:
        for j in label:
            mi = relative_gap(i["mi"], j["mi"])
            mean = abs(i["mean"] - j["mean"])
            var = relative_gap(i["variance"], j["variance"])
            if (mi <= protocol["mi_relative_tolerance"] and mean <= protocol["mean_tolerance"]
                    and var <= protocol["variance_relative_tolerance"]):
                matches.append(dict(identity_state=i["state_id"], label_state=j["state_id"],
                                    mi_relative_difference=mi, mean_difference=mean,
                                    variance_relative_difference=var))
    return sorted(matches, key=lambda m: (m["mi_relative_difference"], m["identity_state"], m["label_state"]))


def one_step_risks(shapes, w, rates):
    # Rows of shapes parameterize W, Theta_0 and Theta_1, respectively.
    truth = (1 - w) * rates[0] + w * rates[1]
    means = [a / (a + b) for a, b in shapes]

    def expected_error(index, probability):
        a, b = shapes[index]
        error = 0.0
        for outcome, mass in ((0, 1 - probability), (1, probability)):
            updated = means.copy()
            updated[index] = (a + outcome) / (a + b + 1)
            weight, rate0, rate1 = updated
            error += mass * ((1 - weight) * rate0 + weight * rate1 - truth) ** 2
        return error

    return expected_error(0, w), (expected_error(1, rates[0]) + expected_error(2, rates[1])) / 2


def analyze(dataset, directory):
    records = OUTPUT / "search_records" / directory
    protocol = json.loads((records / "protocol.json").read_text())
    states = read_csv(records / "candidate_states.csv")
    for state in states:
        state["state_id"] = int(state["state_id"])
        for key in ("mi", "variance", "mean", "value_identity", "value_label"):
            state[key] = float(state[key])
    budgets = Counter((int(s["n_identity"]), int(s["n_labels_per_source"])) for s in states)
    expected = {(i, l): protocol["state_repeats"] for i in protocol["identity_budgets"]
                for l in protocol["label_budgets"]}
    if budgets != expected or sorted(s["state_id"] for s in states) != list(range(len(states))):
        raise ValueError(f"{dataset}: incomplete candidate search")

    matches = replay_selection(states, protocol)
    archived = read_csv(records / "qualifying_pairs.csv")
    if len(matches) != len(archived) or any(
        any(value != float(row[key]) for key, value in match.items())
        for match, row in zip(matches, archived)
    ):
        raise ValueError(f"{dataset}: fit-only replay differs from the archived pair log")
    selected = {s["role"]: int(s["state_id"]) for s in read_csv(records / "selected_states.csv")}
    if (matches[0]["identity_state"], matches[0]["label_state"]) != (
        selected["identity_preferred"], selected["label_preferred"]
    ):
        raise ValueError(f"{dataset}: smallest-gap pair differs from the archived selection")

    pool = sorted((r for r in read_csv(records / "population_rates.csv") if r["split"] == "heldout"),
                  key=lambda r: int(r["source"]))
    if [r["source"] for r in pool] != ["0", "1"]:
        raise ValueError(f"{dataset}: expected both held-out sources")
    w = int(pool[1]["n"]) / sum(int(r["n"]) for r in pool)
    rates = [float(r["target_rate"]) for r in pool]
    risks = {s["state_id"]: one_step_risks(json.loads(s["shapes"]), w, rates) for s in states}
    reference = {(int(r["state_id"]), r["action"]): float(r["expected_squared_error"])
                 for r in read_csv(records / "heldout_exact_one_step.csv")}
    differences = []
    for sid in selected.values():
        for action, risk in zip(("source_id", "reference_label"), risks[sid]):
            saved = reference[sid, action]
            if not math.isclose(risk, saved, abs_tol=1e-16, rel_tol=1e-13):
                raise ValueError(f"{dataset}, state {sid}: {action} risk differs from archive")
            differences.append(abs(risk - saved))

    log, advantages = [], []
    for rank, match in enumerate(matches, 1):
        i, j = match["identity_state"], match["label_state"]
        pair = dict(dataset=dataset, rank=rank, selected=int(rank == 1), identity_state=i, label_state=j)
        log.append({**pair, **match})
        ei_i, el_i = risks[i]
        ei_j, el_j = risks[j]
        di, dl = el_i - ei_i, ei_j - el_j
        advantages.append(dict(**pair, E_I_i=ei_i, E_L_i=el_i, E_I_j=ei_j, E_L_j=el_j,
                               identity_advantage=di, label_advantage=dl, both_confirmed=int(di > 0 and dl > 0)))
    summary = dict(dataset=dataset, states=len(states), qualifying_pairs=len(log),
                   identity_positive=sum(r["identity_advantage"] > 0 for r in advantages),
                   label_positive=sum(r["label_advantage"] > 0 for r in advantages),
                   both_positive=sum(r["both_confirmed"] for r in advantages),
                   selected_risks_checked=len(differences), max_abs_risk_difference=max(differences))
    return log, advantages, summary


if __name__ == "__main__":
    logs, pairs, summaries = [], [], []
    for dataset, directory in DATASETS.items():
        log, advantages, summary = analyze(dataset, directory)
        logs.extend(log)
        pairs.extend(advantages)
        summaries.append(summary)
    write_csv(OUTPUT / "pair_selection_log.csv", logs)
    write_csv(OUTPUT / "pairwise_heldout_advantages.csv", pairs)
    write_csv(OUTPUT / "pairwise_confirmation_summary.csv", summaries)
    for summary in summaries:
        print(summary)
