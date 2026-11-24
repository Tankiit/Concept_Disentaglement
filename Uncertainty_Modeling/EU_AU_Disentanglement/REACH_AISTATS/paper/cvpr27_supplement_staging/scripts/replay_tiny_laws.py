"""Run the saved Tiny ImageNet acquisition code from the audited cell laws or the original feature cache."""
import argparse
from contextlib import nullcontext
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tiny_imagenet_notebook_source"))
import tiny_four_channel_executed as primary

records = ROOT / "TinyImageNet_REACH_Results_v2full"
audit = json.loads((records / "audit.json").read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--from-cache", action="store_true", help="refit all four cells from the 40-epoch feature cache")
args = parser.parse_args()
if args.from_cache:
    cache = ROOT / "tiny_imagenet_model/frozen_tiny_seed20261111_epochs40.npz"
    world_source = nullcontext()
else:
    cache = None
    world = tuple(np.asarray(audit[k]) for k in ("fit_source_laws","fit_error_rates","eval_source_laws","eval_error_rates"))
    world_source = patch.object(primary, "build_world_from_cache", return_value=(*world, audit.copy()))

with tempfile.TemporaryDirectory() as tmp:
    with world_source:
        primary.run_experiment(cache, tmp, cells=audit["cells"], repeats=audit["repeats"], budgets=audit["budgets"],
                               w=audit["generating_mixture_weights"], seed=20261029, delta=audit["alpha_fixed_setting"],
                               baseline_inputs=audit["baseline"]["inputs"],
                               baseline_references=audit["baseline"]["references_per_known_source"],
                               baseline_labels=audit["baseline"]["labels_per_known_source_at_probe"])
    if args.from_cache and json.loads((Path(tmp) / "audit.json").read_text()) != audit:
        raise ValueError("Refitting cells from the cache did not reproduce the archived audit")
    for name in ("intervals", "summary", "reference_eval_transfer", "exact_identification_benchmarks"):
        actual = pd.read_csv(Path(tmp) / f"{name}.csv")
        saved = pd.read_csv(records / f"{name}.csv")
        pd.testing.assert_frame_equal(actual, saved, check_exact=False, rtol=0, atol=1e-8)
    print("Verified all four numeric exports, including 720 labeled interval rows.")
