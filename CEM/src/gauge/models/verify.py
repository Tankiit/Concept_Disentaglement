from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..core import Activation


def assert_equivalent(lhs: Sequence[Activation], rhs: Sequence[Activation], *, atol: float = 1e-8, rtol: float = 1e-5) -> None:
    """Assert that two activation collections are equivalent up to numerical tolerance."""

    if len(lhs) != len(rhs):
        raise AssertionError(f"Activation counts differ: {len(lhs)} != {len(rhs)}")

    for left, right in zip(lhs, rhs):
        if left.values.shape != right.values.shape:
            raise AssertionError(
                f"Shape mismatch: {left.values.shape} != {right.values.shape}"
            )
        if not np.allclose(left.values, right.values, atol=atol, rtol=rtol):
            raise AssertionError("Activations are not equivalent within tolerance")
