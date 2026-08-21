from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class InvarianceAuditResult:
    """Container for invariance audit results reported in a table."""

    metrics: Dict[str, float]

    def as_dict(self) -> Dict[str, float]:
        return dict(self.metrics)


def x1_invariance_audit(values: Iterable[float]) -> InvarianceAuditResult:
    """Placeholder invariance audit used to summarize the Table 1 style report."""

    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return InvarianceAuditResult({"mean": 0.0, "std": 0.0})
    return InvarianceAuditResult({
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    })
