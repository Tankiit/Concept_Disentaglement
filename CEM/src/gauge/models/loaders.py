from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ..core import Activation


def load_tensorboard_metrics(log_dir: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load scalar and tensor summaries from TensorBoard event files.

    The return value maps tags to ordered records containing ``step``,
    ``wall_time``, and ``value``. Tensor values are NumPy arrays; scalar values
    are floats. TensorBoard remains an optional dependency.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboard.util import tensor_util
    except ImportError as exc:
        raise ImportError(
            "TensorBoard metrics loading requires `pip install tensorboard`."
        ) from exc

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    metrics: Dict[str, List[Dict[str, Any]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        metrics[tag] = [
            {"step": event.step, "wall_time": event.wall_time, "value": float(event.value)}
            for event in accumulator.Scalars(tag)
        ]
    for tag in accumulator.Tags().get("tensors", []):
        metrics[tag] = [
            {
                "step": event.step,
                "wall_time": event.wall_time,
                "value": np.asarray(tensor_util.make_ndarray(event.tensor_proto)),
            }
            for event in accumulator.Tensors(tag)
        ]
    return metrics


def load_checkpoint_activations(path: str | Path) -> List[Activation]:
    """Load activations from a checkpoint-style artifact.

    This is a lightweight placeholder intended to be replaced by a PyTorch-backed
    implementation when a real checkpoint format is available.
    """

    checkpoint = np.load(path, allow_pickle=True)
    arrays = checkpoint.get("activations", checkpoint)
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    activations: List[Activation] = []
    for idx, array in enumerate(arrays):
        activations.append(Activation(values=np.asarray(array), label=f"layer_{idx}"))
    return activations
