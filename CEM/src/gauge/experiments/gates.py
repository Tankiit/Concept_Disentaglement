from __future__ import annotations

from typing import Sequence

import numpy as np


def g1_gram_spectrum(matrix: np.ndarray) -> np.ndarray:
    """Return the Gram-spectrum diagnostic for a matrix."""

    matrix = np.asarray(matrix)
    gram = matrix.T @ matrix
    eigenvalues = np.linalg.eigvalsh(gram)
    return np.sort(np.abs(eigenvalues))[::-1]
