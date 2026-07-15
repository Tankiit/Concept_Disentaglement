"""Local DPS transform helpers.

This repository expects `consonant_possibility_from_samples(samples)` to map a
patch-by-sample score matrix to a per-patch possibility estimate `pi` and a
specificity width `W`.

The implementation below is deterministic, vectorized, and intentionally small:
it treats the per-patch sample envelope as the support interval and uses the
row-wise maximum as the possibility score. Width is the normalized sample range.
"""

from __future__ import annotations

import numpy as np


def consonant_possibility_from_samples(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-patch possibility `pi` and specificity width `W`.

    Parameters
    ----------
    samples:
        Array of shape `(num_patches, num_samples)`.

    Returns
    -------
    (pi, W):
        `pi` is the row-wise maximum, clipped to `[0, 1]`.
        `W` is the row-wise range normalized by the maximum range in the batch.
    """

    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array of shape (num_patches, num_samples); got {arr.shape}")
    if arr.shape[1] == 0:
        raise ValueError("samples must contain at least one ensemble draw")

    row_min = arr.min(axis=1)
    row_max = arr.max(axis=1)
    row_range = row_max - row_min

    pi = np.clip(row_max, 0.0, 1.0)
    scale = float(row_range.max())
    if scale < 1e-12:
        W = np.zeros_like(pi, dtype=np.float32)
    else:
        W = row_range / (scale + 1e-12)

    return pi.astype(np.float32), W.astype(np.float32)
