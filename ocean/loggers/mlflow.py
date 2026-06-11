"""MLFlowLogger - logs metrics to MLflow."""

import os
import re
import time
from typing import Any, Mapping, Optional

from ocean.loggers.base import Logger


class MLFlowLogger(Logger):
    """Log metrics to MLflow.

    Args:
        experiment_name: Name of the experiment.
        run_name: Name of the run.
        tracking_uri: MLflow tracking server URI.
        tags: Tags for the run.
        save_dir: Directory for MLflow logs.
        log_model: If True/all, log model checkpoints.
        prefix: Prefix for metric keys.
        artifact_location: Location for artifacts.
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[dict[str, Any]] = None,
        save_dir: Optional[str] = "./mlruns",
        log_model: bool = False,
        prefix: str = "",
        artifact_location: Optional[str] = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self._tags = tags or {}
        self._save_dir = save_dir
        self._log_model = log_model
        self._prefix = prefix
        self._artifact_location = artifact_location
        self._experiment = None
        self._run_id = None
        self._experiment_id = None

    @property
    def experiment(self) -> Any:
        if self._experiment is None:
            self._experiment = self._create_experiment()
        return self._experiment

    def _create_experiment(self) -> Any:
        try:
            import mlflow
            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)
            experiment = mlflow.set_experiment(self._experiment_name)
            self._experiment_id = experiment.experiment_id

            run = mlflow.start_run(run_name=self._run_name, tags=self._tags)
            self._run_id = run.info.run_id
            return mlflow
        except ImportError:
            class _DummyMlflow:
                def log_metric(self, *a, **kw): ...
                def log_param(self, *a, **kw): ...
                def log_batch(self, *a, **kw): ...
                def end_run(self, *a, **kw): ...
            return _DummyMlflow()

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    @property
    def experiment_id(self) -> Optional[str]:
        return self._experiment_id

    @property
    def name(self) -> str:
        return self._run_name or self._experiment_name

    @property
    def version(self) -> str:
        return self._run_id or ""

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        try:
            import mlflow
            params = self._flatten_dict(params)
            batch = []
            for k, v in params.items():
                v_str = str(v)[:250]  # MLflow 250 char limit
                batch.append(mlflow.entities.Param(k, v_str))
                if len(batch) >= 100:
                    self.experiment.log_batch(run_id=self._run_id, params=batch)
                    batch = []
            if batch:
                self.experiment.log_batch(run_id=self._run_id, params=batch)
        except Exception:
            pass

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        try:
            import mlflow
            prefixed = {}
            for k, v in metrics.items():
                key = f"{self._prefix}{self.LOGGER_JOIN_CHAR}{k}" if self._prefix else k
                if isinstance(v, str):
                    continue
                clean_key = re.sub(r"[^a-zA-Z0-9_/\ .-]", "", key)
                if clean_key != key:
                    key = clean_key
                if hasattr(v, "item"):
                    v = v.item()
                prefixed[key] = float(v)

            metrics_list = [
                mlflow.entities.Metric(k, v, int(time.time() * 1000), step or 0)
                for k, v in prefixed.items()
            ]
            self.experiment.log_batch(run_id=self._run_id, metrics=metrics_list)
        except Exception:
            pass

    def finalize(self, status: str = "success") -> None:
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLFlowLogger._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
