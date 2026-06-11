"""WandbLogger - logs metrics to Weights & Biases."""

from typing import Any, Mapping, Optional

from ocean.loggers.base import Logger


class WandbLogger(Logger):
    """Log metrics to Weights & Biases.

    Args:
        name: Experiment name.
        save_dir: Directory to save logs.
        version: Experiment version.
        project: W&B project name.
        offline: If True, run in offline mode.
        log_model: If True/all, log model checkpoints.
        prefix: Prefix for metric keys.
        **kwargs: Additional kwargs to wandb.init().
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: str = ".",
        version: Optional[str] = None,
        project: Optional[str] = None,
        offline: bool = False,
        log_model: bool = False,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        self._name = name
        self._save_dir = save_dir
        self._version = version
        self._project = project
        self._offline = offline
        self._log_model = log_model
        self._prefix = prefix
        self._kwargs = kwargs
        self._experiment = None
        self._logged_model_time = None

    @property
    def experiment(self) -> Any:
        if self._experiment is None:
            self._experiment = self._create_experiment()
        return self._experiment

    def _create_experiment(self) -> Any:
        try:
            import wandb
            # Check if wandb.run already exists
            run = wandb.run
            if run is not None:
                return run
            init_kwargs = {"dir": self._save_dir, "project": self._project, "name": self._name, "reinit": True}
            if self._offline:
                init_kwargs["mode"] = "offline"
            init_kwargs.update(self._kwargs)
            return wandb.init(**init_kwargs)
        except ImportError:
            class _DummyExperiment:
                def log(self, *args, **kwargs): ...
                def config(self): return type("Config", (), {"update": lambda *a, **kw: None})()
                def finish(self): ...
                def watch(self, *args, **kwargs): ...
            return _DummyExperiment()

    @property
    def name(self) -> str:
        if self._experiment is not None:
            return getattr(self._experiment, "name", None) or self._name or ""
        return self._name or ""

    @property
    def version(self) -> str:
        if self._experiment is not None:
            return getattr(self._experiment, "id", None) or self._version or ""
        return self._version or ""

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        try:
            prefixed = {}
            for k, v in metrics.items():
                key = f"{self._prefix}{self.LOGGER_JOIN_CHAR}{k}" if self._prefix else k
                if hasattr(v, "item"):
                    v = v.item()
                prefixed[key] = float(v)
            log_kwargs = {"step": step} if step is not None else {}
            self.experiment.log(prefixed, **log_kwargs)
        except Exception:
            pass

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        try:
            self.experiment.config.update(params, allow_val_change=True)
        except Exception:
            pass

    def finalize(self, status: str) -> None:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish(exit_code=0 if status == "success" else 1)
        except Exception:
            pass

    def watch(self, model: Any, log: str = "gradients", log_freq: int = 100) -> None:
        try:
            self.experiment.watch(model, log=log, log_freq=log_freq)
        except Exception:
            pass

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state
