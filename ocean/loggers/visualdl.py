"""VisualDLLogger - logs metrics to VisualDL (Paddle's native visualization tool).

Lightning-compatible: auto-versioning, save_dir/name/version_N/ structure.
"""

import os
from typing import Any, Optional

from ocean.loggers.base import Logger


class VisualDLLogger(Logger):
    """Log metrics to VisualDL for visualization.

    Lightning-compatible with auto-versioning.

    Args:
        save_dir: Directory to save logs.
        name: Experiment name. Default: ``'ocean_logs'``.
        version: Experiment version. Auto-incremented if None.
        prefix: Prefix for metric keys.
    """

    def __init__(
        self,
        save_dir: str,
        name: str = "ocean_logs",
        version: Optional[Any] = None,
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
    def version(self) -> Any:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    def root_dir(self) -> str:
        return self._save_dir

    @property
    def log_dir(self) -> str:
        ver = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self._save_dir, self._name, ver)

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

    def _get_next_version(self) -> int:
        """Scan log dir for existing version_N dirs and auto-increment."""
        version_dir = os.path.join(self._save_dir, self._name)
        if not os.path.exists(version_dir):
            return 0
        existing_versions = []
        for d in os.listdir(version_dir):
            dp = os.path.join(version_dir, d)
            if os.path.isdir(dp) and d.startswith("version_"):
                try:
                    existing_versions.append(int(d.replace("version_", "")))
                except ValueError:
                    continue
        return max(existing_versions) + 1 if existing_versions else 0
