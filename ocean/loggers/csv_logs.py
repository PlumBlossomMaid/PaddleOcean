"""CSVLogger - logs metrics to CSV files.

Uses ``@rank_zero_only`` and ``@rank_zero_experiment`` to ensure
only rank 0 writes CSV files.
"""

import csv
import os
from typing import Any, Optional

from ocean.loggers.base import Logger
from ocean.utils.rank_zero import rank_zero_experiment, rank_zero_only


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
    @rank_zero_experiment
    def experiment(self) -> "_ExperimentWriter":
        if self._experiment is None:
            self._experiment = _ExperimentWriter(self.log_dir)
        return self._experiment

    @rank_zero_only
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

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Write the hyperparameters next to the metrics.

        The base class treats this as a no-op, so the default logger used to
        drop everything handed to it — including whatever the Trainer logs
        automatically at the start of a run.
        """
        if not params:
            return
        from ocean.core.saving import save_hparams_to_yaml

        os.makedirs(self.log_dir, exist_ok=True)
        save_hparams_to_yaml(dict(params), os.path.join(self.log_dir, "hparams.yaml"))

    @rank_zero_only
    def save(self) -> None:
        self.experiment.save()

    @rank_zero_only
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
        new_keys = self._record_new_keys()
        file_exists = os.path.exists(self.metrics_file)

        # A metric seen for the first time widens the header. Appending under
        # the old header would write rows with more fields than it declares —
        # the file stops being the CSV it claims to be — so it is rewritten.
        if new_keys and file_exists:
            self._rewrite_with_new_header()

        with open(self.metrics_file, "a" if file_exists else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            for m in self.metrics:
                writer.writerow(m)
        self.metrics = []

    @property
    def fieldnames(self) -> list[str]:
        """Column order: ``step`` first, then the rest sorted.

        Sorted rather than "whatever the set iterated as": string hashing is
        randomised per process, so the same run wrote its columns in a
        different order every time.
        """
        return ["step"] + sorted(k for k in self.metrics_keys if k != "step")

    def _record_new_keys(self) -> list[str]:
        current = set().union(*self.metrics) if self.metrics else set()
        new_keys = sorted(current - set(self.metrics_keys))
        self.metrics_keys.extend(new_keys)
        return new_keys

    def _rewrite_with_new_header(self) -> None:
        with open(self.metrics_file, newline="") as f:
            rows = list(csv.DictReader(f))
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
