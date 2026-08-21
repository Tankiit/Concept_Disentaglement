from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class SUB:
    """Per-concept substitution benchmark container.

    This stores a concept-wise substitution accuracy score dictionary and a
    convenience method for computing overall accuracy.
    """

    scores: Dict[str, float]

    @property
    def mean(self) -> float:
        return float(np.mean(list(self.scores.values()))) if self.scores else 0.0

    def as_dict(self) -> Dict[str, float]:
        return dict(self.scores)
