"""VisualDLLogger - logs metrics to VisualDL (Paddle's native visualization tool).

Analogous to TensorBoardLogger in PyTorch Lightning.
"""

import os
from typing import Any, Optional

from ocean.loggers.base import Logger


class VisualDLLogger(Logger):
    """Log metrics to VisualDL for visualization.

    Args:
        save_dir: Directory to save logs.
        name: Experiment name. Default: ``'lightning_logs'``.
        version: Experiment version. Auto-incremented if None.
        prefix: Prefix for metric keys.
    """

    def __init__(
        self,
        save_dir: str,
        name: str = "lightning_logs",
        version: Optional[str] = None,
        prefix: str = "",
    ) -> None:
        self._save_dir = save_dir
        self._name = name
        self._version = version
        self._prefix = prefix
        self._experiment = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    def root_dir(self) -> str:
        return self._save_dir

    @property
    def log_dir(self) -> str:
        return os.path.join(self._save_dir, self._name, f"version_{self.version}")

    @property
    def experiment(self) -> Any:
        if self._experiment is None:
            self._experiment = self._create_experiment()
        return self._experiment

    def _create_experiment(self) -> Any:
        """Create a VisualDL LogWriter."""
        try:
            from visualdl import LogWriter
            return LogWriter(logdir=self.log_dir)
        except ImportError:
            # VisualDL not installed - use a dummy
            class _DummyWriter:
                def add_scalar(self, *args, **kwargs): ...
                def close(self): ...
            return _DummyWriter()

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            key = f"{self._prefix}/{k}" if self._prefix else k
            if hasattr(v, "item"):
                v = v.item()
            self.experiment.add_scalar(key, float(v), step or 0)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        try:
            import yaml
            hparams_path = os.path.join(self.log_dir, "hparams.yaml")
            os.makedirs(os.path.dirname(hparams_path), exist_ok=True)
            with open(hparams_path, "w") as f:
                yaml.dump(params, f)
        except ImportError:
            pass

    def save(self) -> None:
        pass

    def finalize(self, status: str) -> None:
        if self._experiment is not None:
            try:
                self._experiment.close()
            except Exception:
                pass

    def _get_next_version(self) -> str:
        version_dir = os.path.join(self._save_dir, self._name)
        if not os.path.exists(version_dir):
            return "0"
        existing = [d for d in os.listdir(version_dir) if d.startswith("version_")]
        versions = []
        for d in existing:
            try:
                versions.append(int(d.replace("version_", "")))
            except ValueError:
                continue
        return str(max(versions) + 1) if versions else "0"
