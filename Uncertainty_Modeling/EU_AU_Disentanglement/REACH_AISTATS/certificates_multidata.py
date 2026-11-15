# certificates_multidata.py -- confirmatory run of the certificate pipeline on the
# frozen dataset grid of real/certificates_prereg.py (10 Nov 2026 gate).
#
# Refuses to run unless the prereg's sha256 matches results/certificates_multidata/
# prereg.sha256. Protocol, pool rule, budgets and predictions live in the prereg file;
# this script only executes and scores them.
#
#   OMP_NUM_THREADS=1 python certificates_multidata.py [--datasets a,b,c] [--score]
#   Out: results/certificates_multidata/{worlds_<name>.csv, per_world.csv, summary.csv,
#        verdict.json, manifest.json, certificates_multidata.png}
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

import epistemic_sets as es
from certificate_audit import certificates
from credal_head import near_far_tags

OUT = Path("results/certificates_multidata")
PREREG = Path("real/certificates_prereg.py")
POOL_DIR = Path("cache/features")
MEDMNIST = Path.home() / ".medmnist"
POOL_N, POOL_SEED = 10_000, 0
WORLD_SEEDS = range(5)

DATASETS = {
    "mnist":          dict(kind="cache", src="timm_wide_resnet50_2_mnisttest.npz"),
    "fashionmnist":   dict(kind="cache", src="timm_wide_resnet50_2_fashionmnisttest.npz"),
    "svhn":           dict(kind="cache", src="timm_wide_resnet50_2_svhntest.npz"),
    "cifar10":        dict(kind="cache", src="timm_wide_resnet50_2_c10test.npz"),
    "cifar100":       dict(kind="cache", src="timm_wide_resnet50_2_cifar100test.npz"),
    "bloodmnist":     dict(kind="medmnist", src="bloodmnist.npz"),
    "pathmnist":      dict(kind="medmnist", src="pathmnist.npz"),
    "organamnist":    dict(kind="medmnist", src="organamnist.npz"),
    "organcmnist":    dict(kind="medmnist", src="organcmnist.npz"),
    "organsmnist":    dict(kind="medmnist", src="organsmnist.npz"),
    "emnist_letters": dict(kind="emnist"),
    "eurosat":        dict(kind="eurosat"),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_prereg():
    want = (OUT / "prereg.sha256").read_text().split()[0]
    got = sha(PREREG)
    assert got == want, f"prereg changed since freeze ({got[:12]} != {want[:12]}); refusing to run"
    return got


def auroc(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def stratified_indices(y, total=POOL_N, seed=POOL_SEED):
    """Per-class proportional draw to `total` rows (largest-remainder), frozen rule."""
    y = np.asarray(y)
    if len(y) <= total:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    quota = counts / counts.sum() * total
    base = np.floor(quota).astype(int)
    order = np.argsort(-(quota - base), kind="stable")
    base[order[: total - base.sum()]] += 1
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), size=n, replace=False)
                                   for c, n in zip(classes, base)]))


def extract(get_pil, n):
    """example_wrn50.extract_features recipe, verbatim, over an index -> PIL getter."""
    import timm
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model("wide_resnet50_2", pretrained=True, num_classes=0).to(device).eval()
    tfm = timm.data.create_transform(**timm.data.resolve_model_data_config(model))
    feats = []
    with torch.no_grad():
        for start in range(0, n, 256):
            x = torch.stack([tfm(get_pil(j).convert("RGB"))
                             for j in range(start, min(start + 256, n))]).to(device)
            feats.append(model(x).cpu().numpy())
    return np.vstack(feats).astype(np.float32), device


def build_pool(name):
    """Frozen pool rule -> cache/features/certpool_<name>.npz (L2-normalized X, y)."""
    from PIL import Image
    path = POOL_DIR / f"certpool_{name}.npz"
    if path.exists():
        return path
    spec, note = DATASETS[name], {}
    if spec["kind"] == "cache":
        d = np.load(POOL_DIR / spec["src"])
        X, y = d["X"].astype(np.float32), d["y"].astype(np.int64)
        idx = stratified_indices(y)
        X, y = X[idx], y[idx]
        note["source"] = spec["src"]
    else:
        if spec["kind"] == "medmnist":
            z = np.load(MEDMNIST / spec["src"])
            if len(z["test_labels"]) >= POOL_N:
                imgs, labs = z["test_images"], z["test_labels"]
                note["splits"] = "test"
            else:
                imgs = np.concatenate([z["train_images"], z["val_images"], z["test_images"]])
                labs = np.concatenate([z["train_labels"], z["val_labels"], z["test_labels"]])
                note["splits"] = "train+val+test"
            labs = labs.reshape(-1).astype(np.int64)
            idx = stratified_indices(labs)
            getter = lambda j: Image.fromarray(imgs[idx[j]])
        elif spec["kind"] == "emnist":
            from torchvision.datasets import EMNIST
            ds = EMNIST("/Users/tanmoy/data", split="letters", train=False, download=False)
            imgs = ds.data.numpy().transpose(0, 2, 1)             # EMNIST stores transposed
            labs = ds.targets.numpy().astype(np.int64) - 1        # 1..26 -> 0..25
            idx = stratified_indices(labs)
            getter = lambda j: Image.fromarray(np.ascontiguousarray(imgs[idx[j]]))
            note["splits"] = "test (letters)"
        elif spec["kind"] == "eurosat":
            from torchvision.datasets import EuroSAT
            ds = EuroSAT("/Users/tanmoy/data/eurosat", download=False)
            labs = np.asarray(ds.targets, dtype=np.int64)
            idx = stratified_indices(labs)
            getter = lambda j: ds[int(idx[j])][0]
            note["splits"] = "all (no official split)"
        y = labs[idx]
        X, device = extract(getter, len(idx))
        note["extraction_device"] = device
    X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    np.savez(path, X=X.astype(np.float32), y=y, src_index=idx)
    print(f"pool {name}: X {X.shape}, {len(np.unique(y))} classes -> {path} {note}", flush=True)
    return path


def budget(n_classes):
    if n_classes >= 10:
        return 1, 2, 5
    return {9: (1, 2, 4), 8: (1, 2, 3)}[n_classes]


def run_dataset(name):
    pool = build_pool(name)
    y_all = np.load(pool)["y"]
    n_classes = len(np.unique(y_all))
    kuc, uuc_val, uuc_test = budget(n_classes)
    recs = []
    for ws in WORLD_SEEDS:
        world = es.make_oss(str(pool), kuc=kuc, uuc_val=uuc_val, uuc_test=uuc_test, seed=ws)
        K = len(world["kkc"])
        rec = dict(dataset=name, world_seed=ws, n_classes=n_classes, n_kkc=K,
                   kuc=kuc, uuc_val=uuc_val, uuc_test=uuc_test, head_failed=False)
        try:
            head, lut, acc = es.train_kp1(world, seed=0, epochs=300)
        except AssertionError as e:                     # head's own trained-check failed
            rec.update(head_failed=True, head_error=str(e))
            recs.append(rec)
            print(f"{name} world {ws}: HEAD FAILED ({e})", flush=True)
            continue
        Xte = torch.from_numpy(world["X"][world["idx"]["test"]]).float()
        yte = world["y"][world["idx"]["test"]]
        S = es.mc_stack(head, Xte, T=30)
        mean = S.mean(0)
        pb, lo, hi = mean[:, K], S.min(0)[:, K], S.max(0)[:, K]
        known, uuc = np.isin(yte, world["kkc"]), np.isin(yte, world["uuc"])
        certs = certificates(lo, hi)
        tags, _, cutoff = near_far_tags(world)
        row_tag = np.array([tags.get(int(c), "kkc") for c in yte])
        near, far = uuc & (row_tag == "near"), uuc & (row_tag == "far")
        tu = -(mean * np.log(np.clip(mean, 1e-12, None))).sum(1)
        au = -(S * np.log(np.clip(S, 1e-12, None))).sum(-1).mean(0)
        sc = {"P_unknown": pb, "set_median": np.median(S, 0)[:, K], "EU_bald": tu - au,
              "neg_msp": -mean.max(1), "width_mean": (S.max(0) - S.min(0)).mean(1)}
        absorbed = uuc & (pb <= 0.01)

        def rate(mask, val):
            return float((certs[mask] == val).mean()) if mask.sum() else float("nan")

        rec.update(
            head_train_acc=acc, n_known_test=int(known.sum()), n_uuc_test=int(uuc.sum()),
            false_novel_certs=int((certs[known] == "cert_novel").sum()),
            known_certknown=rate(known, "cert_known"),
            cov_uuc=rate(uuc, "cert_novel"), uuc_certknown=rate(uuc, "cert_known"),
            n_near=int(near.sum()), n_far=int(far.sum()),
            cov_near=rate(near, "cert_novel"), cov_far=rate(far, "cert_novel"),
            certknown_near=rate(near, "cert_known"), certknown_far=rate(far, "cert_known"),
            absorbed_share=float(absorbed.sum() / max(uuc.sum(), 1)),
            n_absorbed=int(absorbed.sum()),
            absorbed_certknown=rate(absorbed, "cert_known"),
            **{f"auroc_{k}": auroc(v[known], v[uuc]) for k, v in sc.items()},
        )
        rec["width_minus_bald"] = rec["auroc_width_mean"] - rec["auroc_EU_bald"]
        recs.append(rec)
        print(f"{name} world {ws}: K={K} false-certs {rec['false_novel_certs']}/"
              f"{rec['n_known_test']} | cov uuc {rec['cov_uuc']:.3f} (far {rec['cov_far']:.3f}"
              f" / near {rec['cov_near']:.3f}) | absorbed {rec['absorbed_share']:.3f} -> "
              f"known-cert {rec['absorbed_certknown']:.3f} | AUROC P_unk "
              f"{rec['auroc_P_unknown']:.3f} BALD {rec['auroc_EU_bald']:.3f} "
              f"width-BALD {rec['width_minus_bald']:+.4f}", flush=True)
    pd.DataFrame(recs).to_csv(OUT / f"worlds_{name}.csv", index=False)


def score(prereg_sha):
    frames = [pd.read_csv(p) for p in sorted(OUT.glob("worlds_*.csv"))]
    W = pd.concat(frames, ignore_index=True)
    W.to_csv(OUT / "per_world.csv", index=False)
    ok = W[~W.head_failed.astype(bool)]
    both = ok[(ok.n_near > 0) & (ok.n_far > 0)]
    absd = ok[ok.n_absorbed > 0]
    import importlib
    PRED = importlib.import_module("real.certificates_prereg").PRED
    got = {
        "P1_gate_zero_false_certificates_everywhere": bool((ok.false_novel_certs == 0).all()),
        "P2_coverage_escapes_cifar10_band": bool(((ok.cov_uuc < 0.10) | (ok.cov_uuc > 0.80)).any()),
        "P3_far_gt_near_certified_every_world": bool((both.cov_far > both.cov_near).all()),
        "P4_absorbed_certknown_ge_0.95_every_world": bool((absd.absorbed_certknown >= 0.95).all()),
        "P5_width_minus_bald_le_0.01_every_world": bool((ok.width_minus_bald <= 0.01).all()),
    }
    detail = {
        "worlds_run": int(len(ok)), "heads_failed": int(W.head_failed.astype(bool).sum()),
        "datasets": sorted(W.dataset.unique().tolist()),
        "P1_total_false_certs": int(ok.false_novel_certs.sum()),
        "P1_known_rows_checked": int(ok.n_known_test.sum()),
        "P1_violations": ok[ok.false_novel_certs > 0][["dataset", "world_seed", "false_novel_certs",
                                                      "n_known_test"]].to_dict("records"),
        "P3_worlds_with_both_tags": int(len(both)),
        "P3_violations": both[~(both.cov_far > both.cov_near)][["dataset", "world_seed", "cov_far",
                                                                "cov_near"]].to_dict("records"),
        "P4_worlds_with_absorbed": int(len(absd)),
        "P4_violations": absd[absd.absorbed_certknown < 0.95][["dataset", "world_seed",
                                                               "absorbed_certknown",
                                                               "n_absorbed"]].to_dict("records"),
        "P5_violations": ok[ok.width_minus_bald > 0.01][["dataset", "world_seed",
                                                         "width_minus_bald"]].to_dict("records"),
    }
    verdict = dict(prereg_sha256=prereg_sha, predicted=PRED,
                   got=got, passed={k: bool(got[k] == PRED[k]) for k in got},
                   GATE=("PASS" if got["P1_gate_zero_false_certificates_everywhere"] else "FAIL"),
                   detail=detail)
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))

    num = ["false_novel_certs", "n_known_test", "known_certknown", "cov_uuc", "cov_far",
           "cov_near", "certknown_near", "certknown_far", "absorbed_share",
           "absorbed_certknown", "auroc_P_unknown", "auroc_set_median", "auroc_EU_bald",
           "width_minus_bald", "n_kkc"]
    summ = ok.groupby("dataset")[num].agg(["mean", "min", "max"])
    summ.to_csv(OUT / "summary.csv")
    show = ok.groupby("dataset").agg(
        worlds=("world_seed", "count"), kkc=("n_kkc", "first"),
        false_certs=("false_novel_certs", "sum"), known_rows=("n_known_test", "sum"),
        cov_uuc=("cov_uuc", "mean"), cov_far=("cov_far", "mean"), cov_near=("cov_near", "mean"),
        near_certknown=("certknown_near", "mean"), absorbed=("absorbed_share", "mean"),
        absorbed_known=("absorbed_certknown", "mean"), auroc_Punk=("auroc_P_unknown", "mean"),
        auroc_bald=("auroc_EU_bald", "mean"), width_bald=("width_minus_bald", "max"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n== per-dataset summary (means over worlds; false_certs/known_rows are totals) ==")
    print(show.round(3).to_string())
    print("\n== PREREG VERDICT ==")
    for k in got:
        print(f"  {k:46s} predicted {PRED[k]} got {got[k]} -> {'PASS' if verdict['passed'][k] else 'FAIL'}")
    print(f"  GATE (P1 soundness): {verdict['GATE']}  "
          f"[{detail['P1_total_false_certs']} false certificates over "
          f"{detail['P1_known_rows_checked']} known test rows, {detail['worlds_run']} worlds, "
          f"{len(detail['datasets'])} datasets]")
    for k in ("P1_violations", "P3_violations", "P4_violations", "P5_violations"):
        if detail[k]:
            print(f"  {k}: {detail[k]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = list(show.index)
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(ds) * 3 + 1.2))
    ylab, ypos = [], []
    for i, d in enumerate(ds):
        g = ok[ok.dataset == d]
        known_cn = float((g.false_novel_certs / g.n_known_test).mean())
        for j, (lab, ck, cn) in enumerate([("known", g.known_certknown.mean(), known_cn),
                                           ("uuc near", g.certknown_near.mean(), g.cov_near.mean()),
                                           ("uuc far", g.certknown_far.mean(), g.cov_far.mean())]):
            y0 = i * 3.6 + j
            if np.isnan(ck):
                continue
            ax.barh(y0, ck, color="#4c72b0"); ax.barh(y0, 1 - ck - cn, left=ck, color="#c9c9c9")
            ax.barh(y0, cn, left=1 - cn, color="#c44e52")
            ylab.append(f"{d}: {lab}"); ypos.append(y0)
    ax.set_yticks(ypos); ax.set_yticklabels(ylab, fontsize=7); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_title("Certificates across datasets (mean over 5 worlds): "
                 "certainly known | undecided | certainly novel", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "certificates_multidata.png", dpi=150)

    manifest = dict(prereg_sha256=prereg_sha, script_sha256=sha(Path(__file__)),
                    epistemic_sets_sha256=sha("epistemic_sets.py"),
                    pools={p.stem: sha(p) for p in sorted(POOL_DIR.glob("certpool_*.npz"))},
                    python=sys.version.split()[0], torch=torch.__version__,
                    note="certificates are relative to the T=30 sampled envelope, not to truth")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nwritten:", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--score", action="store_true", help="aggregate + score only")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    h = check_prereg()
    torch.set_num_threads(1)
    if not args.score:
        for name in args.datasets.split(","):
            run_dataset(name)
    score(h)
