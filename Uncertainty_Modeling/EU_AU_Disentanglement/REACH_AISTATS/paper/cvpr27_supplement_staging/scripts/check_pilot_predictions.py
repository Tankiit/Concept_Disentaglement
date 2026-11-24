"""Check Tables 14–16 from the pilot cell assignments, frozen heads and recorded scores."""
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from probly.method import dropout
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from check_image_pilots import ROOT, PILOTS


def check(run):
    selected = pd.read_csv(run / "selected_images.csv")
    assert selected.groupby(["source","role"]).size().tolist() == [2000]*9
    ref = pd.read_csv(run / "reference_cells.csv")
    ev = pd.read_csv(run / "evaluation_cells.csv")
    def cell_law(frame):
        return pd.crosstab(frame.source, frame.cell, normalize="index")
    drift = (cell_law(ref)-cell_law(ev)).abs().sum(axis=1)/2
    np.testing.assert_allclose(drift, pd.read_csv(run / "source_law_drift.csv").sort_values("source").empirical_total_variation)
    with np.load(run / "features.npz") as features:
        np.testing.assert_array_equal(selected.record_id, features["record_ids"])
        mask = selected.role.eq("evaluation") & ~selected.temporal_ood
        x = torch.from_numpy(features["features"][mask])
    np.testing.assert_array_equal(selected.loc[mask,"record_id"], ev.record_id)
    scaler = torch.load(run / "feature_scaler.pt", map_location="cpu", weights_only=True)
    x = (x - scaler["mean"]) / scaler["std"]
    accuracy = pd.read_csv(run / "accuracy.csv").set_index(["method","source"])
    for method in ["ce", "dappr"]:
        model = dropout(nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.Linear(128,int(selected.target.max())+1)),
                        p=0.25, predictor_type="logit_classifier")
        model.load_state_dict(torch.load(run / f"{method}_head.pt", map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            pred = model(x).argmax(1).numpy()
        for source, records in ev.groupby("source"):
            saved = accuracy.loc[(method, source)]
            assert len(records) == saved.n and (pred[records.index] == records.target).mean() == saved.accuracy
    scores = pd.read_csv(run / "image_scores.csv")
    np.testing.assert_array_equal(scores.record_id, ev.record_id)
    y = scores.source.eq(2)
    for row in pd.read_csv(run / "ood_scores.csv").itertuples():
        score = scores[row.method]
        fpr, tpr, _ = roc_curve(y,score)
        np.testing.assert_allclose([roc_auc_score(y,score),average_precision_score(y,score),fpr[tpr>=.95][0]],
                                   [row.auroc,row.aupr_ood,row.fpr95],atol=1e-12,rtol=0)
    print(run.name, "verified 3 source drifts, 6 source accuracies and 4 OOD score rows")


if __name__ == "__main__":
    for run_id in PILOTS.values():
        check(ROOT / "workbook" / run_id)
