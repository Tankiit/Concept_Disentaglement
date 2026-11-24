"""Acquire the six pinned upstream image files and prepare the classifier's input arrays."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("folder", type=Path, help="directory for source files and the classifier image cache")
args = parser.parse_args()
args.folder.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[1]
for name, entry in json.loads((root / "provenance/controlled_classifier_raw_inputs.json").read_text()).items():
    path = args.folder / name
    if not path.exists():
        urllib.request.urlretrieve(entry["url"], path)
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
            raise ValueError(f"{name} differs from the logged input hash")
    if path.suffix == ".parquet":
        table = pd.read_parquet(path)
        col = "image" if "image" in table else "img"
        images = np.stack([np.array(Image.open(io.BytesIO(v["bytes"])).convert("L")) for v in table[col]])
        np.savez_compressed(path.with_suffix(".npz"), X=images, y=table.label.to_numpy(dtype=np.int64))
