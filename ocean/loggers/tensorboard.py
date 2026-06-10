"""TensorBoard logger for PaddlePaddle (using VisualDL as backend).

PaddlePaddle doesn't have native TensorBoard support.
This logger bridges the gap by writing TensorBoard-format events
via the `tensorboardX` or `visualdl` package.
"""

import os
from typing import Any, Optional

from ocean.loggers.base import Logger


class TensorBoardLogger(Logger):
    """Log metrics in TensorBoard-format using VisualDL or tensorboardX.

    Args:
        save_dir: Directory to save logs.
        name: Experiment name.
        version: Experiment version.
        sub_dir: Subdirectory within log_dir.
        prefix: Prefix for metric keys.
    """

    def __init__(
        self,
        save_dir: str,
        name: str = "lightning_logs",
        version: Optional[str] = None,
        sub_dir: Optional[str] = None,
        prefix: str = "",
    ) -> None:
        self._save_dir = save_dir
        self._name = name
        self._version = version
        self._sub_dir = sub_dir
        self._prefix = prefix
        self._experiment = None
        self._backend = self._detect_backend()

    @staticmethod
    def _detect_backend() -> str:
        try:
            from tensorboardX import SummaryWriter
            return "tensorboardX"
        except ImportError:
            try:
                from visualdl import LogWriter
                return "visualdl"
            except ImportError:
                return "none"

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    def log_dir(self) -> str:
        base = os.path.join(self._save_dir, self._name, f"version_{self.version}")
        if self._sub_dir:
            base = os.path.join(base, self._sub_dir)
        return base

    @property
    def experiment(self) -> Any:
        if self._experiment is None:
            self._experiment = self._create_writer()
        return self._experiment

    def _create_writer(self) -> Any:
        os.makedirs(self.log_dir, exist_ok=True)
        if self._backend == "tensorboardX":
            from tensorboardX import SummaryWriter
            return SummaryWriter(logdir=self.log_dir)
        elif self._backend == "visualdl":
            from visualdl import LogWriter
            return LogWriter(logdir=self.log_dir)
        else:
            class _DummyWriter:
                def add_scalar(self, *args, **kwargs): ...
                def add_histogram(self, *args, **kwargs): ...
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
            path = os.path.join(self.log_dir, "hparams.yaml")
            with open(path, "w") as f:
                yaml.dump(params, f)
        except ImportError:
            pass

    def finalize(self, status: str) -> None:
        try:
            self.experiment.close()
        except Exception:
            pass

    def _get_next_version(self) -> str:
        base = os.path.join(self._save_dir, self._name)
        if not os.path.exists(base):
            return "0"
        existing = [d for d in os.listdir(base) if d.startswith("version_")]
        versions = []
        for d in existing:
            try:
                versions.append(int(d.replace("version_", "")))
            except ValueError:
                continue
        return str(max(versions) + 1) if versions else "0"
