from __future__ import annotations

"""
interval_composition.py

Composition WITHOUT a circuit: logical operators are evaluated directly on the
credal envelope [N_c, Pi_c] per concept.

Two instantiations of ONE operator family, chosen by domain ground truth:
  CUB      : clean AND   -> bird = head AND wing AND tail (true conjunction)
  derm7pt  : weighted sum -> composed interval = sum_c w_c * [N_c, Pi_c],
             threshold applied ONLY at readout.

Design commitments:
  - W is DERIVED from the ensemble, not trained. Interval is exact-given-ensemble,
    uncalibrated. Reported, not validated.
  - Composition bounds are OUTER (contain the true credal composition); tightness
    under concept dependence is deferred.
  - Point readout predicts the target; WIDTH readout is reported, not validated.
"""

import json
import os

import numpy as np


def interval_from_ensemble(pi_samples: np.ndarray) -> tuple[float, float]:
    """Return [N, Pi] from one concept's ensemble scores."""
    vals = np.asarray(pi_samples, dtype=float)
    if vals.ndim != 1 or vals.size == 0:
        raise ValueError("pi_samples must be a non-empty 1D array")
    pi = float(np.max(vals))
    nec = float(1.0 - np.max(1.0 - vals))  # equals min(vals)
    return nec, pi


def consonance_check(N: float, Pi: float, band_lo: float, band_hi: float, tol: float = 1e-3) -> bool:
    """Check that [N,Pi] matches an external band [band_lo, band_hi]."""
    ok = abs(N - band_lo) < tol and abs(Pi - band_hi) < tol
    if not ok:
        print(
            f"!! consonance mismatch: [N,Pi]=[{N:.3f},{Pi:.3f}] "
            f"band=[{band_lo:.3f},{band_hi:.3f}] -- distribution not consonant?"
        )
    return ok


def compose_and(intervals: list[tuple[float, float]]) -> dict:
    """Conjunction under outer-envelope composition."""
    if not intervals:
        raise ValueError("intervals must be non-empty")
    N = min(n for n, _ in intervals)
    Pi = min(p for _, p in intervals)
    return {"N": N, "Pi": Pi, "point": 0.5 * (N + Pi), "width": Pi - N}


def compose_weighted(
    intervals: list[tuple[float, float]],
    weights: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """Weighted interval composition with optional imprecise 3-way decision."""
    if not intervals:
        raise ValueError("intervals must be non-empty")
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size != len(intervals):
        raise ValueError("weights must be a 1D array with same length as intervals")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")

    N = float(sum(wc * n for wc, (n, _) in zip(w, intervals)))
    Pi = float(sum(wc * p for wc, (_, p) in zip(w, intervals)))
    denom = float(w.sum()) + 1e-8
    Nn, Pin = N / denom, Pi / denom

    out = {
        "N": Nn,
        "Pi": Pin,
        "point": 0.5 * (Nn + Pin),
        "width": Pin - Nn,
        "raw_sum_interval": (N, Pi),
    }
    if threshold is not None:
        out["decision"] = (
            "positive" if Nn >= threshold else "possible" if Pin >= threshold else "negative"
        )
    return out


def _weights_from_env(var_name: str, expected_len: int) -> np.ndarray | None:
    """Read weights from env var as JSON list, e.g. DERM7PT_WEIGHTS='[2,1,...]'"""
    raw = os.getenv(var_name)
    if not raw:
        return None
    try:
        values = json.loads(raw)
        arr = np.asarray(values, dtype=float)
    except Exception as exc:
        raise ValueError(f"Could not parse {var_name} as JSON list: {exc}") from exc
    if arr.ndim != 1 or arr.size != expected_len:
        raise ValueError(f"{var_name} must contain exactly {expected_len} weights")
    return arr


if __name__ == "__main__":
    np.random.seed(0)

    # CUB-style conjunction smoke
    ens = {
        c: np.clip(np.random.rand(20) * 0.6 + off, 0, 1)
        for c, off in [("head", 0.3), ("wing", 0.35), ("tail", 0.1)]
    }
    ivals = {c: interval_from_ensemble(v) for c, v in ens.items()}
    for c, (n, p) in ivals.items():
        assert n <= p, f"{c}: N>Pi"
        consonance_check(n, p, float(ens[c].min()), float(ens[c].max()))
    print("[ok] per-concept intervals:", {c: (round(n, 2), round(p, 2)) for c, (n, p) in ivals.items()})

    andr = compose_and(list(ivals.values()))
    print(
        f"[ok] CUB AND: N={andr['N']:.3f} Pi={andr['Pi']:.3f} "
        f"point={andr['point']:.3f} width={andr['width']:.3f}"
    )
    assert andr["N"] <= andr["point"] <= andr["Pi"]

    # derm-style weighted composition smoke (no hardcoded placeholder weights)
    ens7 = [np.clip(np.random.rand(20) * 0.5 + 0.2, 0, 1) for _ in range(7)]
    iv7 = [interval_from_ensemble(v) for v in ens7]
    weights = _weights_from_env("DERM7PT_WEIGHTS", expected_len=7)
    if weights is None:
        print("[skip] weighted smoke: set DERM7PT_WEIGHTS='[w1,...,w7]' to run this check")
    else:
        wr = compose_weighted(iv7, weights, threshold=0.35)
        print(
            f"[ok] derm WEIGHTED: N={wr['N']:.3f} Pi={wr['Pi']:.3f} "
            f"width={wr['width']:.3f} decision={wr['decision']}"
        )
        assert wr["decision"] in {"positive", "possible", "negative"}

    print("\nSMOKE OK: endpoint composition works; weighted mode expects explicit weights.")
