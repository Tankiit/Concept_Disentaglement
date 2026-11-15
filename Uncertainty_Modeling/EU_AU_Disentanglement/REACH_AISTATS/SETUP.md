# REACH_AISTATS environment

From this folder, create and activate the environment:

```bash
conda env create -f environment.yml
conda activate reach-aistats
python main.py --help
```

If `reach-aistats` already exists on this machine, activate it directly.
The environment includes the core scientific packages, CVXPY/CLARABEL,
PyTorch/torchvision, PyTorch-OOD, Probly, timm, OpenCLIP, Modal, and ipykernel.
Python 3.13 supports the pinned Probly 0.9.0 release. The existing
`requirements.txt` describes only the minimal core dependencies.

In your IDE, select the `reach-aistats` Python interpreter or notebook kernel.
On this machine the interpreter is:
`/home/mtleao/.conda/envs/reach-aistats/bin/python`.

Run the numerical regression tests:

```bash
python -m unittest discover -s tests -p test_reach.py
```

Validated on Linux: all 28 numerical regression tests pass, required external
imports succeed, `pip check` passes, and PyTorch executes on the NVIDIA GPU
with CUDA 13.0. GPU execution on other machines depends on their hardware and
driver support.

Run commands from this folder so local modules and relative data paths resolve.
The complete `python main.py test` suite also includes tests requiring cached
CIFAR feature files under `cache/features/` (including `cifar10_test.npz` and
corruption features). These data are not installed by Conda. Feature extraction
may download datasets and pretrained weights; Modal jobs require a configured
Modal account.
