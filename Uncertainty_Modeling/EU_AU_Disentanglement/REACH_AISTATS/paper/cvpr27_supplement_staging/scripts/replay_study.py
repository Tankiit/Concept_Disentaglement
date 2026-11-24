"""Execute a frozen notebook with its matching implementation in a new working directory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "controlled_classifier": ("dappr_epistemic_sets_seed0_20261029T085507170686Z", "dappr_epistemic_sets"),
    "controlled": ("source_acquisition_20261029T151111775438Z", "source_acquisition_four_datasets"),
    "constant": ("acquisition_counterexample_20261029T192454214626Z", "acquisition_counterexample"),
    "three_target": ("three_target_characterization_20261030T144711349004Z", "three_target_characterization"),
    "camelyon17_pilot": ("camelyon17_acquisition_20261030T092422959226Z", "camelyon17_acquisition"),
    "fmow_pilot": ("fmow_acquisition_20261030T094932286096Z", "fmow_acquisition"),
    "camelyon17_search": ("camelyon17_counterexample_20261030T092024988890Z", "camelyon17_counterexample"),
    "fmow_search": ("fmow_counterexample_20261030T094929883846Z", "fmow_counterexample"),
    "informative": ("informative_cell_verification_20261117", None),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=RUNS)
    parser.add_argument("work", type=Path, help="new directory for all replay outputs")
    parser.add_argument("--image-cache", type=Path, help="directory of six prepared digit, clothing and blood image arrays")
    parser.add_argument("--data-root", type=Path, help="local WILDS mirror")
    parser.add_argument("--weights", type=Path, help="ImageNet ResNet18 V1 weights for a pilot")
    args = parser.parse_args()
    if args.study.endswith(("_pilot","_search")) and not (args.data_root and args.data_root.is_dir()):
        parser.error("supply --data-root for this WILDS study")
    if args.study.endswith("_pilot") and not (args.weights and args.weights.is_file()):
        parser.error("supply --weights for an image pilot")
    run_id, notebook = RUNS[args.study]
    run = ROOT / "workbook" / run_id
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    (work / "workbook").mkdir()
    env = dict(os.environ, MPLBACKEND="Agg")
    if args.data_root:
        env["WILDS_DATA_ROOT"] = str(args.data_root.resolve())
    if args.weights:
        env["WILDS_ENCODER_WEIGHTS"] = str(args.weights.resolve())
    if args.study == "informative":
        shutil.copytree(run / "source", work, dirs_exist_ok=True)
        subprocess.run([sys.executable, "scripts/verify_informative_cell.py"], cwd=work, env=env, check=True)
        return
    for src in run.glob("*.py"):
        sub = "tests" if src.name.startswith("test_") else "scripts" if src.name.startswith("build_") else "real"
        (work / sub).mkdir(exist_ok=True)
        shutil.copyfile(src, work / sub / src.name)
    if args.study == "three_target":
        shutil.copyfile(run / "THREE_TARGET_CHARACTERIZATION.md", work / "THREE_TARGET_CHARACTERIZATION.md")
    if args.study == "controlled_classifier":
        if not args.image_cache:
            parser.error("supply --image-cache for the upstream classifier")
        cache = work / "cache" / "mnist_reach"
        cache.mkdir(parents=True)
        for name in json.loads((ROOT / "provenance/controlled_classifier_image_cache.json").read_text()):
            shutil.copyfile(args.image_cache / name, cache / name)
    if args.study == "controlled":
        for item in json.loads((run / "input_manifest.json").read_text()):
            rel = Path(item["path"])
            (work / rel).parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(ROOT / rel, work / rel)
    # Keep code, markdown and ordering used by the executed notebook, starting with clean outputs.
    nb = json.loads((run / "executed.ipynb").read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell.update(outputs=[], execution_count=None)
    path = work / (notebook + ".ipynb")
    path.write_text(json.dumps(nb))
    # The seven archived notebooks contain plain Python code cells. Executing them
    # in order also propagates exceptions to the command's exit status.
    source = "\n\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
    program = work / (notebook + ".py")
    program.write_text(source + "\n")
    subprocess.run([sys.executable, program.name], cwd=work, env=env, check=True)


if __name__ == "__main__":
    main()
