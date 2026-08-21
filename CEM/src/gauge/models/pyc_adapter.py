from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..core import Activations
from ..group.cem import CEMGauge, CBMPartialGauge


# ============================================================================
# 1. Introspection - run this FIRST
# ============================================================================


def probe_bottleneck(bottleneck: nn.Module) -> dict:
    """Report the facts the gauge derivation depends on.

    The core empirical fact is whether the final scorer is shared across concepts
    or applied independently to each concept. This determines whether the CEM
    gauge is the shared-group case (dim = m(m-2)) or the per-concept product case
    (dim = k * m * (m-2)).
    """
    params = [{"name": n, "shape": tuple(p.shape)} for n, p in bottleneck.named_parameters()]
    candidates = [
        p for p in params if p["name"].endswith("weight") and len(p["shape"]) in (1, 2)
    ]

    # Heuristic pinning for the current PyC family: keep the exact layer path in the
    # result after the first real probe, and then hard-code that path. This keeps the
    # adapter honest without shipping a brittle guess.
    scorer_shape = None
    scorer_is_shared = None
    implied_gauge_dim = None

    if candidates:
        scorer_shape = candidates[-1]["shape"]
        scorer_is_shared = len(scorer_shape) == 1 or (len(scorer_shape) == 2 and scorer_shape[0] == 1)
        implied_gauge_dim = None if scorer_shape == () else (scorer_shape[-1] // 2) * ((scorer_shape[-1] // 2) - 2) if scorer_is_shared else None

    return {
        "params": params,
        "scorer_candidates": candidates,
        "scorer_shape": scorer_shape,
        "scorer_is_shared": scorer_is_shared,
        "implied_gauge_dim": implied_gauge_dim,
    }


def extract_scoring_covectors(bottleneck: nn.Module, path: str):
    """Pull w_plus, w_minus from the CEM scoring layer.

    path: dotted path to the scoring weight, from probe_bottleneck.
    Returns two (m,) arrays: W_s = [w+, w-] in R^{1 x 2m}, split in half.
    """
    obj = bottleneck
    for part in path.split("."):
        obj = getattr(obj, part)
    W = obj.detach().cpu().numpy()
    if W.ndim == 2 and W.shape[0] != 1:
        raise ValueError(
            f"scoring weight shape {W.shape}: leading dim != 1 means the scorer is PER-CONCEPT. "
            "The admissible set becomes a product of k independent groups, dim = k*m*(m-2), "
            "and group/cem.py needs a variant. This is a finding, not a bug - record it."
        )
    W = W.reshape(-1)
    assert W.size % 2 == 0, f"scorer length {W.size} is not 2m"
    m = W.size // 2
    return W[:m].copy(), W[m:].copy()


# ============================================================================
# 2. One forward pass -> Activations
# ============================================================================


@torch.no_grad()
def to_activations(model: nn.Module, loader, *, arch: str, device="cuda",
                   scorer_path: str | None = None,
                   cache: str | None = None):
    """Run the backbone ONCE, return (Activations, GaugeGroup).

    arch: "cem" | "cbm" | "residual"

    This adapter is deliberately conservative: the real implementation should do
    the actual PyC hooking and then save the extracted bundle for later reuse.
    """
    if cache is not None:
        try:
            z = np.load(cache, allow_pickle=True)
            act = Activations(**{k: z[k] for k in z.files})
            return act, _group_for(act, arch, scorer_path, model)
        except FileNotFoundError:
            pass

    raise NotImplementedError(
        "Wire to your PyC model:\n"
        "  1. probe_bottleneck(model.bottleneck) -> find the scorer path\n"
        "  2. hook the bottleneck, run the loader once\n"
        "  3. stack c_plus, c_minus, c_prob, c_repr, y_logit, y_true, c_true\n"
        "  4. grab head_W from the task predictor's first Linear\n"
        "  5. np.savez(cache, **fields)\n"
        "  6. assert_equivalent on a random admissible A before trusting it"
    )


def _group_for(act, arch, scorer_path, model):
    if arch == "cem":
        if scorer_path is None:
            raise ValueError("scorer_path is required for CEM gauge construction")
        wp, wm = extract_scoring_covectors(model.bottleneck, scorer_path)
        return CEMGauge(wp, wm)
    if arch in ("cbm", "residual"):
        return CBMPartialGauge(act.supervised)
    raise ValueError(arch)
