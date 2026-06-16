"""CSVLogger - logs metrics to CSV files."""

import csv
import os
from typing import Optional

from ocean.loggers.base import Logger


class CSVLogger(Logger):
    """Log metrics to a CSV file.

    Args:
        root_dir: Root directory for logs.
        name: Experiment name. Default: ``'ocean_logs'``.
        version: Experiment version. Auto-incremented if None.
        prefix: Prefix for metric keys.
        flush_logs_every_n_steps: Flush to disk every N steps.
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        root_dir: str,
        name: str = "ocean_logs",
        version: Optional[str] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ) -> None:
        self._root_dir = root_dir
        self._name = name
        self._version = version
        self._prefix = prefix
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

        self._metrics: list[dict[str, float]] = []
        self._metrics_keys: list[str] = []
        self._experiment: Optional["_ExperimentWriter"] = None

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
        return self._root_dir

    @property
    def log_dir(self) -> str:
        return os.path.join(self._root_dir, self._name, f"version_{self.version}")

    @property
    def experiment(self) -> "_ExperimentWriter":
        if self._experiment is None:
            self._experiment = _ExperimentWriter(self.log_dir)
        return self._experiment

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        if step is None:
            step = len(self._metrics)
        prefixed = {}
        for k, v in metrics.items():
            key = f"{self._prefix}{self.LOGGER_JOIN_CHAR}{k}" if self._prefix else k
            if hasattr(v, "item"):
                v = v.item()
            prefixed[key] = float(v)
        prefixed["step"] = step
        self._metrics.append(prefixed)
        self.experiment.log_metrics(prefixed)

        if len(self._metrics) % self._flush_logs_every_n_steps == 0:
            self.save()

    def save(self) -> None:
        self.experiment.save()

    def finalize(self, status: str) -> None:
        self.save()

    def _get_next_version(self) -> str:
        version_dir = os.path.join(self._root_dir, self._name)
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


class _ExperimentWriter:
    """Internal class for writing metrics to a CSV file."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_file = os.path.join(log_dir, "metrics.csv")
        self.metrics: list[dict[str, float]] = []
        self.metrics_keys: list[str] = []

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.append(metrics)

    def save(self) -> None:
        if not self.metrics:
            return
        all_keys = set()
        for m in self.metrics:
            all_keys.update(m.keys())
        new_keys = [k for k in all_keys if k not in self.metrics_keys]
        self.metrics_keys.extend(new_keys)

        file_exists = os.path.exists(self.metrics_file)
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step"] + [k for k in self.metrics_keys if k != "step"])
            if not file_exists:
                writer.writeheader()
            for m in self.metrics:
                writer.writerow(m)
        self.metrics = []
