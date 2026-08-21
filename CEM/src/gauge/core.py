from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Activation:
    """Container for activation tensors with a simple interface for concept analysis."""

    values: np.ndarray
    label: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.values.shape


@dataclass
class Activations:
    """Bundle of concept and task activations produced by a single forward pass."""

    c_plus: Optional[np.ndarray] = None
    c_minus: Optional[np.ndarray] = None
    c_prob: Optional[np.ndarray] = None
    c_repr: Optional[np.ndarray] = None
    y_logit: Optional[np.ndarray] = None
    y_true: Optional[np.ndarray] = None
    c_true: Optional[np.ndarray] = None
    supervised: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        for key, value in self.__dict__.items():
            if key != "metadata":
                yield key, value

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__iter__() if v is not None}


@dataclass
class Measure:
    """A scalar diagnostic computed from activations or groups of activations."""

    name: str
    value: float
    metadata: Dict[str, object] = field(default_factory=dict)


class GaugeGroup:
    """A collection of activations with helper logic for model-preserving transforms."""

    def __init__(self, activations: Sequence[Activation]):
        self.activations = list(activations)

    def __len__(self) -> int:
        return len(self.activations)

    def __iter__(self):
        return iter(self.activations)

    def add(self, activation: Activation) -> None:
        self.activations.append(activation)

    def summary(self) -> Dict[str, object]:
        return {
            "count": len(self.activations),
            "shapes": [tuple(a.shape) for a in self.activations],
        }


def spread(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Compute the empirical spread of a tensor along an axis."""

    x = np.asarray(x)
    if axis is None:
        return np.std(x)
    return np.std(x, axis=axis)
