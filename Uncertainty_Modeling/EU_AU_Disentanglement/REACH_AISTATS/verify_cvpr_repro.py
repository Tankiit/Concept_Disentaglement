# verify_cvpr_repro.py -- end-to-end reproduction check for the CVPR-line experiments.
#
# Re-runs every CVPR experiment into a scratch directory and compares against the
# recorded results/ outputs. Nothing in results/ is written or overwritten.
#
# Run from REACH_AISTATS with the project environment (see REPRODUCE_CVPR.md):
#   OMP_NUM_THREADS=1 python verify_cvpr_repro.py [--out DIR] [--skip-suite] [--wrn50]
#
# Verdicts per experiment:
#   IDENTICAL  -- re-run output is byte-identical to the recorded file
#   ~TOLERANCE -- bytes differ; numeric content matches within atol 1e-9 / rtol 1e-5
#   FAIL       -- content differs beyond tolerance (or the run crashed)
# The WRN50 plain-mode arm (--wrn50) is head-training nondeterministic by record
# (COLLABORATOR_PLAN 24-25 Oct) and is therefore tolerance-checked at 0.02.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPECTED_SUITE_PASSED = 147          # tests/ count as of 2026-11-10

# Inputs whose bytes the recorded results depend on. A mismatch here means the
# experiments will not regenerate the recorded numbers even if they run.
PINNED_SHA256 = {
    "cache/features/features.pt":        "814295a90760102c0c70d0a7f3c975b698801990b7027189ae1c2053180d54d3",
    "cache/features/cifar10_test.npz":   "8514442826f981f32da8c811006994d70eb410db54f1c5c6e8bc192189ff2156",
    "cache/features/cifar10_train.npz":  "c23e78a38dee3a5f2e0d727144d646bc7b21a50f733f824e9d279e3163cb3367",
    "cache/features/timm_wide_resnet50_2_c10test.npz":        "1a61991960c19728575dd255357a18f3564d66a67b81daff87141f3c798890fc",
    "cache/features/timm_wide_resnet50_2_mnisttest.npz":      "ba731096eb9fd7cacaf5a8ee92a78ee5f1680ec970c670110dadcd99f08142a8",
    "cache/features/timm_wide_resnet50_2_fashionmnisttest.npz": "fa2453f516a034d41103a3d2d7c87d18f030033397cfa1dcf39324b497526d01",
    "cache/features/timm_wide_resnet50_2_svhntest.npz":       "34d2aeb059ee5f36c2cfc7f7f6e808533f64964e32e15d056c83c5f46de400c8",
    "cache/features/timm_wide_resnet50_2_cifar100test.npz":   "5aabef1dd2acfbb43465ab073f9af715e84b79668a0bdf430b7964fc412179ba",
    "epistemic_sets.py":  "b36c4647b965026749df94d9a8676d06c37daf0dc844227f7eced83a953dd91d",
    "credal_head.py":     "e5dc36807953486a5bf4708461249571bc8371f6481d6975eb395a95f0bba730",
    "step0_reconcile.py": "8a89391fc9789c1b01816d53551f5f553711cc1b39d7efb3c5845bab46893075",
    "metrics.py":         "5dfa658ffde37d880b5610fecc1ea790e853dc1dcb0c3ddbf30e9b8282aee14d",
    "example.py":         "db9e69fe83a00b5a87472d035e5542fae142a4e867f5205a1ae8a6dd3f727b24",
    "example_wrn50.py":   "e3db4c15b551c6cf5000e3e78916e3c2643a2bb8ec31d8cfc2e5c9a79a25857e",
}

# Headline experiments: (label, runner lambda(out_dir), files to compare against results/<label>/).
BYTE_EXPERIMENTS = [
    ("epistemic_sets",
     lambda out: __import__("epistemic_sets").main(out_dir=out),
     ["rows.csv", "summary.csv", "per_class.csv", "summary_agg.csv", "manifest.json"]),
    ("credal_head",
     lambda out: __import__("credal_head").main(out_dir=out),
     ["rows.csv", "summary.csv", "matched_au.csv", "correlations.csv", "near_far.csv",
      "summary_agg.csv", "verdict.json", "pred.json", "pred_amendment.json", "manifest.json"]),
    ("step0_reconciliation",
     lambda out: _run_step0(out),
     ["summary.json"]),
]

# WRN50 plain-mode arm (recorded with --device mps; head-training nondeterministic).
WRN50_RUNS = {  # results dir -> CLI args
    "oss_metrics":          ["--id-dataset", "cifar10", "--ood-dataset", "mnist"],
    "oss_metrics_fmnist":   ["--id-dataset", "cifar10", "--ood-dataset", "fashionmnist"],
    "oss_metrics_svhn":     ["--id-dataset", "cifar10", "--ood-dataset", "svhn"],
    "oss_metrics_cifar100": ["--id-dataset", "cifar10", "--ood-dataset", "cifar100"],
    "oss_metrics_svhn_id":  ["--id-dataset", "svhn", "--ood-dataset", "cifar10"],
    "oss_metrics_svhn_id_mnist": ["--id-dataset", "svhn", "--ood-dataset", "mnist"],
}
WRN50_TOL = 0.02   # headline AUROCs moved <=0.013 across recorded-vs-rerun; diagnostics are looser


def _run_step0(out_dir):
    import step0_reconcile
    step0_reconcile.OUT_DIR = Path(out_dir)
    step0_reconcile.main()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def env_report():
    print("== environment ==")
    print("python", sys.version.split()[0])
    for mod in ("torch", "pytorch_ood", "sklearn", "pandas", "numpy", "scipy", "probly",
                "timm", "matplotlib"):
        try:
            m = __import__(mod)
            print(f"  {mod} {getattr(m, '__version__', '?')}")
        except Exception:
            print(f"  {mod} MISSING")
    print("Recorded environment: python 3.12.4, torch 2.10.0, pytorch_ood 0.2.1, "
          "numpy 2.2.6, pandas 2.3.2, scipy 1.16.2, sklearn 1.5.1, probly 0.9.1, "
          "timm 1.0.21 (conda env torch-multimodal)")


def check_inputs():
    print("\n== pinned inputs ==")
    ok = True
    for rel, want in PINNED_SHA256.items():
        p = HERE / rel
        if not p.exists():
            print(f"  MISSING  {rel}")
            ok = False
            continue
        got = sha256(p)
        status = "OK      " if got == want else "HASH DIFF"
        if got != want:
            ok = False
        print(f"  {status} {rel}")
    return ok


def run_suite():
    print("\n== test suite ==")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                       cwd=HERE, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()
    print("  " + (tail[-2] if len(tail) >= 2 else tail[-1] if tail else "no output"))
    return f"{EXPECTED_SUITE_PASSED} passed" in r.stdout


def compare_file(ref, new, atol=1e-9, rtol=1e-5):
    """IDENTICAL | (ok, maxdiff) tolerance verdict | FAIL."""
    if sha256(ref) == sha256(new):
        return "IDENTICAL", None
    suffix = ref.suffix
    try:
        if suffix == ".json":
            a = json.loads(ref.read_text()); b = json.loads(new.read_text())
            diffs = []
            def walk(x, y, p=""):
                if isinstance(x, dict) and isinstance(y, dict):
                    for k in set(x) | set(y):
                        if k in ("device", "world_keys"):
                            continue            # environment-relative fields
                        walk(x.get(k), y.get(k), f"{p}.{k}")
                elif isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                        and not isinstance(x, bool):
                    d = abs(float(x) - float(y))
                    if d > atol + rtol * abs(float(x)):
                        diffs.append((p, x, y, d))
                elif x != y:
                    diffs.append((p, x, y, None))
            walk(a, b)
            if not diffs:
                return "~TOLERANCE", 0.0
            worst = max((d for *_, d in diffs if d is not None), default=None)
            return "FAIL", (worst, diffs[:3])
        if suffix == ".csv":
            a = pd.read_csv(ref); b = pd.read_csv(new)
            if a.shape != b.shape or list(a.columns) != list(b.columns):
                return "FAIL", ("shape/columns", a.shape, b.shape)
            worst = 0.0
            for c in a.columns:
                if pd.api.types.is_numeric_dtype(a[c]):
                    delta = float(np.nanmax(np.abs(
                        a[c].to_numpy(float) - b[c].to_numpy(float))))
                    worst = max(worst, delta)
                elif not (a[c].astype(str) == b[c].astype(str)).all():
                    return "FAIL", (f"non-numeric column {c} differs",)
            if worst <= atol + rtol:
                return "~TOLERANCE", worst
            return "FAIL", (worst,)
    except Exception as e:  # unreadable / unexpected format
        return "FAIL", (f"{type(e).__name__}: {e}",)
    return "FAIL", ("binary files differ",)


def run_byte_experiments(out_root):
    print("\n== headline experiments (byte-level) ==")
    ok = True
    for label, runner, files in BYTE_EXPERIMENTS:
        out = Path(out_root) / label
        print(f"-- {label}")
        runner(out)
        for f in files:
            ref = HERE / "results" / label / f
            new = out / f
            if not new.exists():
                print(f"   MISSING   {f}")
                ok = False
                continue
            verdict, detail = compare_file(ref, new)
            print(f"   {verdict:10s} {f}" + (f"  (max diff {detail[0]:.2e})" if detail else ""))
            if verdict == "FAIL":
                ok = False
                if detail:
                    print(f"              worst: {detail[1] if len(detail) > 1 else detail}")
    return ok


def run_wrn50(out_root):
    print("\n== WRN50 plain-mode arm (tolerance; head-training nondeterministic) ==")
    ok = True
    out_root = Path(out_root)
    for name, args in WRN50_RUNS.items():
        out = out_root / "wrn50" / name
        print(f"-- {name} ({' '.join(args)})")
        r = subprocess.run([sys.executable, "example_wrn50.py", "--mode", "plain",
                            "--device", "mps", *args, "--out-dir", str(out)],
                           cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            print("   RUN FAILED"); print("   " + r.stderr.strip().splitlines()[-1][:200])
            ok = False
            continue
        for f in ("wrn50_eval.json", "wrn50_terciles.json"):
            verdict, detail = compare_file(HERE / "results" / name / f, out / f,
                                           atol=WRN50_TOL, rtol=0.0)
            note = ""
            if verdict == "FAIL" and detail:
                note = f"  worst {detail[0]}"
            print(f"   {verdict:10s} {f}{note}")
            if verdict == "FAIL":
                ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="scratch dir (default: tempdir)")
    ap.add_argument("--skip-suite", action="store_true")
    ap.add_argument("--wrn50", action="store_true", help="also re-run the WRN50 arm (~10 min)")
    ap.add_argument("--skip-runs", action="store_true",
                    help="only check pinned inputs + recorded outputs exist")
    args = ap.parse_args()

    os.chdir(HERE)                       # experiment scripts use relative paths
    print("verify_cvpr_repro -- CVPR experiment reproduction check")
    env_report()
    results = {"inputs": check_inputs()}
    if not args.skip_suite:
        results["suite"] = run_suite()
    if not args.skip_runs:
        out_root = args.out or tempfile.mkdtemp(prefix="reach_repro_")
        print(f"\nscratch output dir: {out_root}")
        results["headline"] = run_byte_experiments(out_root)
        if args.wrn50:
            results["wrn50"] = run_wrn50(out_root)
        else:
            print("\n(WRN50 arm skipped; add --wrn50 to include it)")

    print("\n== SUMMARY ==")
    for k, v in results.items():
        print(f"  {k:10s} {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
