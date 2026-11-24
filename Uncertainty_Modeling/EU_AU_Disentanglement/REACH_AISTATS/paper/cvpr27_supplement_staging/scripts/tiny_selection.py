"""Reconstruct image selections using the class order and path strings from the recorded Colab run."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tiny_imagenet_notebook_source"))
from tiny_imagenet_acquisition import deterministic_source, parse_source_weights, split_manifest

RECORDED_ROOT = Path("/content/tiny_reach_cvpr/data/tiny-imagenet-200")
SEED = 20261111


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_root", type=Path, help="extracted tiny-imagenet-200 directory")
    args = parser.parse_args()
    train, fit, evaluation, classes = split_manifest(args.image_root, SEED)
    weights = parse_source_weights("0.45,0.25,0.30")
    folder = ROOT / "TinyImageNet_REACH_Results_v2full"
    audit = json.loads((folder / "audit.json").read_text())
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["role","position","relative_path","class_id","wnid","source"])
    for role, examples in (("train",train),("fit",fit),("eval",evaluation)):
        sources = []
        for i, (path, label) in enumerate(examples):
            rel = Path(path).relative_to(args.image_root)
            source = 0 if role == "train" else deterministic_source(SEED+1009, str(RECORDED_ROOT/rel), weights)
            sources.append(source)
            writer.writerow([role,i,str(rel),label,classes[label],source])
        if role != "train":
            np.testing.assert_array_equal(np.bincount(sources), np.sum(audit[role+"_source_cell_count"], axis=1))
        print(role,len(examples),np.bincount(sources))
    (folder / "reconstructed_image_selection.csv.gz").write_bytes(gzip.compress(output.getvalue().encode(),mtime=0))


if __name__=="__main__":
    main()
