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
        name: Experiment name. Default: ``'ocean_logs'``.
        prefix: Prefix for metric keys (DiffSinger-style, no version dirs).
    """

    def __init__(
        self,
        save_dir: str,
        name: str = "ocean_logs",
        prefix: str = "",
    ) -> None:
        self._save_dir = save_dir
        self._name = name
        self._prefix = prefix
        self._experiment = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "0"  # DiffSinger-style: no version dirs

    @property
    def root_dir(self) -> str:
        return self._save_dir

    @property
    def log_dir(self) -> str:
        return os.path.join(self._save_dir, self._name)

    @property
    def experiment(self) -> Any:
        if self._experiment is None:
            self._experiment = self._create_experiment()
        return self._experiment

    def _create_experiment(self) -> Any:
        """Create a VisualDL LogWriter (DiffSinger-style: no version dirs)."""
        try:
            from visualdl import LogWriter

            logdir = self.log_dir
            os.makedirs(logdir, exist_ok=True)
            return LogWriter(logdir=logdir)
        except ImportError:

            class _DummyWriter:
                def add_scalar(self, *args, **kwargs): ...
                def close(self): ...

            return _DummyWriter()

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        if step is None:
            return
        for k, v in metrics.items():
            key = f"{self._prefix}/{k}" if self._prefix else k
            if hasattr(v, "item"):
                v = v.item()
            self.experiment.add_scalar(key, float(v), step)

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
