from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def ctl(x: np.ndarray) -> float:
    """Concept-level transfer measure placeholder."""
    return float(np.mean(np.abs(x)))


def icl(x: np.ndarray) -> float:
    """Inter-concept localization placeholder."""
    return float(np.std(x))


def probe(x: np.ndarray) -> float:
    """Probe-score style diagnostic placeholder."""
    return float(np.mean(x))


def nec_ncc(x: np.ndarray) -> float:
    """Necessity-normalized consistency placeholder."""
    return float(np.corrcoef(x, np.arange(len(x)))[0, 1]) if len(x) > 1 else 0.0


def ois(x: np.ndarray) -> float:
    """Out-of-support invariance score placeholder."""
    return float(np.var(x))


def rho_moment(x: np.ndarray) -> float:
    """Moment-based rho metric placeholder."""
    return float(np.mean(np.square(x)))


def rho_knn(x: np.ndarray) -> float:
    """KNN-style rho metric placeholder."""
    return float(np.mean(np.abs(x)))


REGISTRY: Dict[str, Callable[[np.ndarray], float]] = {
    "ctl": ctl,
    "icl": icl,
    "probe": probe,
    "nec_ncc": nec_ncc,
    "ois": ois,
    "rho_moment": rho_moment,
    "rho_knn": rho_knn,
}
