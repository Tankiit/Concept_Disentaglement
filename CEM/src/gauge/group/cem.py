from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


D_CEM: Dict[str, str] = {
    "A": "A'w+ = w+",
    "A'": "A'w- = w-",
}


@dataclass
class CEMGauge:
    """Gauge element for the shared scoring layer of a CEM.

    For a CEM where the task scorer is shared across concepts, the admissible
    group is parameterized by a single pair (w_plus, w_minus) and has dimension
    m(m-2). When the scorer is per-concept, the admissible set becomes a product
    of k copies and the dimension is k * m * (m - 2).
    """

    w_plus: np.ndarray
    w_minus: np.ndarray
    name: str = "CEMGauge"

    def __post_init__(self) -> None:
        self.w_plus = np.asarray(self.w_plus, dtype=float)
        self.w_minus = np.asarray(self.w_minus, dtype=float)
        if self.w_plus.shape != self.w_minus.shape:
            raise ValueError("w_plus and w_minus must have the same shape")

    @property
    def m(self) -> int:
        return self.w_plus.size

    @property
    def implied_dim(self) -> int:
        return self.m * (self.m - 2)


@dataclass
class CBMPartial:
    """Plain CBM variant with k' of k supervised concepts, to be extended later."""

    k: int
    k_supervised: int
    weights: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.k_supervised > self.k:
            raise ValueError("k_supervised must be <= k")

    @property
    def supervised_fraction(self) -> float:
        return self.k_supervised / self.k if self.k else 0.0


@dataclass
class CBMPartialGauge:
    """Gauge for partial supervision where only k' of the k concepts are supervised."""

    supervised_mask: Optional[np.ndarray] = None
    k_supervised: Optional[int] = None

    def __post_init__(self) -> None:
        if self.supervised_mask is not None:
            self.supervised_mask = np.asarray(self.supervised_mask)
        if self.k_supervised is not None and self.supervised_mask is not None:
            if int(self.k_supervised) != int(np.sum(self.supervised_mask)):
                raise ValueError("k_supervised does not match the supervised mask")

    @property
    def n_supervised(self) -> int:
        if self.supervised_mask is None:
            return int(self.k_supervised or 0)
        return int(np.sum(self.supervised_mask))


# A compact block-form reference for the CEM operator used in the theory section.
# The matrix structure is a 2 x 2 block form with a supervised and unsupervised split.
def cem_block_form(dim: int, r: int = 2) -> np.ndarray:
    """Return a simple block-form reference matrix for a CEM operator."""

    if r != 2:
        raise ValueError("This reference implementation currently assumes r = 2")

    blocks = np.zeros((dim, dim))
    for i in range(dim):
        blocks[i, i] = 1.0
    return blocks
