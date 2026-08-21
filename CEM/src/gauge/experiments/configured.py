"""Shared foundation for configuration-driven experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .config import load_dataset_config, load_experiment_config


class ConfiguredExperiment(ABC):
    """Experiment that resolves inherited settings and dataset metadata."""

    def __init__(self, config: str | Path | dict[str, Any]):
        self.config = (
            load_experiment_config(config)
            if isinstance(config, (str, Path))
            else dict(config)
        )
        dataset_name = self.config.get("dataset", {}).get("name")
        if not dataset_name:
            raise ValueError("experiment config requires dataset.name")
        config_dir = self.config.get("dataset", {}).get("config_dir")
        self.dataset = load_dataset_config(dataset_name, config_dir)

    @property
    def device(self) -> str:
        return str(self.config.get("runtime", {}).get("device", "mps"))

    @property
    def output_dir(self) -> Path:
        return Path(self.config.get("output", {}).get("dir", "results"))

    def training_parameters(self) -> dict[str, Any]:
        """Expose shared downstream-training settings to later experiments."""
        return dict(self.config.get("training", {}))

    @abstractmethod
    def run(self):
        raise NotImplementedError
