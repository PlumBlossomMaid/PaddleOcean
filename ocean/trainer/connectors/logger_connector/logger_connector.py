"""Logger connector: owns metric snapshots and dispatches to loggers.

Metric *storage and reduction* live in the per-stage ``_ResultCollection``
(``trainer._results``); this connector keeps flat snapshot dicts
(``callback_metrics`` / ``logged_metrics`` / ``progress_bar_metrics``) refreshed
from the active collection so callbacks and the progress bar can read them
synchronously, and forwards logged values to the configured loggers.
"""

from __future__ import annotations

from typing import Any, Optional

from ocean.loggers.base import Logger
from ocean.loggers.csv_logs import CSVLogger


class _LoggerConnector:
    """Manages logging: logger config, metric snapshots, logger dispatch."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self._callback_metrics: dict[str, float] = {}
        self._logged_metrics: dict[str, float] = {}
        self._progress_bar_metrics: dict[str, float] = {}

    def on_trainer_init(self, logger: Any, log_every_n_steps: int) -> None:
        self.configure_logger(logger)
        self.trainer.log_every_n_steps = log_every_n_steps

    def configure_logger(self, logger: Any) -> None:
        """Configure logger from bool/None/Logger/list."""
        if logger is False:
            self.trainer.loggers = []
        elif logger is True or logger is None:
            self.trainer.loggers = [CSVLogger(root_dir=self.trainer.default_root_dir or ".")]
        elif isinstance(logger, Logger):
            self.trainer.loggers = [logger]
        elif isinstance(logger, (list, tuple)):
            self.trainer.loggers = list(logger)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Dispatch a metric dict to each logger (loggers filter by rank internally)."""
        for lg in getattr(self.trainer, "loggers", None) or []:
            if hasattr(lg, "log_metrics"):
                lg.log_metrics(metrics, step)

    def log_hyperparams(self, params: Optional[dict[str, Any]] = None) -> None:
        """Dispatch hyperparameters to each logger.

        Called once near the start of fit so every backend records the model's
        hparams. Backends without their dependency installed degrade silently
        rather than fail the run (see per-logger fallback semantics).
        """
        if not params:
            return
        for lg in getattr(self.trainer, "loggers", None) or []:
            if hasattr(lg, "log_hyperparams"):
                lg.log_hyperparams(params)

    def finalize(self, status: str) -> None:
        """Ask every logger to finalize its run with the given status.

        ``status`` is ``"success"`` on clean completion of an entry-point call
        (fit/validate/test/predict) and ``"failed"`` when an exception bubbled
        out. Releases writers / closes runs (Wandb finish, MLflow end_run,
        TensorBoard/VisualDL writer close, CSV flush).
        """
        for lg in getattr(self.trainer, "loggers", None) or []:
            if hasattr(lg, "finalize"):
                lg.finalize(status)

    # ------------------------------------------------------------------
    # Snapshot refresh from the active result collection
    # ------------------------------------------------------------------
    def update_metrics(self, on_step: bool) -> None:
        """Refresh snapshot dicts from ``trainer._results`` for the given view."""
        results = getattr(self.trainer, "_results", None)
        if results is None:
            return
        m = results.metrics(on_step)
        self._logged_metrics.update(m["log"])
        self._callback_metrics.update(m["callback"])
        self._progress_bar_metrics.update(m["pbar"])

    def reset_results(self, fx: Optional[str] = None) -> None:
        """Reset the active result collection (optionally only entries under ``fx``)."""
        results = getattr(self.trainer, "_results", None)
        if results is not None:
            results.reset(fx=fx)

    def reset_validation_metrics(self) -> None:
        """Drop eval-stage metric state so it can't leak into training log flushes.

        With per-stage collections the training collection is a *separate* object,
        so this only clears the current (eval) collection and the stale snapshot
        entries it produced — training accumulation is never touched.
        """
        results = getattr(self.trainer, "_results", None)
        if results is not None and not getattr(results, "training", True):
            results.reset()

    def reset_metrics(self) -> None:
        self._callback_metrics = {}
        self._logged_metrics = {}
        self._progress_bar_metrics = {}

    @property
    def callback_metrics(self) -> dict[str, float]:
        return self._callback_metrics

    @property
    def logged_metrics(self) -> dict[str, float]:
        return self._logged_metrics

    @property
    def progress_bar_metrics(self) -> dict[str, float]:
        return self._progress_bar_metrics

    def teardown(self) -> None:
        pass
