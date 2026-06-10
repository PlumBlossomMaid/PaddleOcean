"""CometLogger - logs metrics to CometML."""

from typing import Any, Mapping, Optional

from ocean.loggers.base import Logger


class CometLogger(Logger):
    """Log metrics to CometML.

    Args:
        api_key: Comet API key.
        workspace: Comet workspace name.
        project: Comet project name.
        experiment_key: Existing experiment key to resume.
        mode: 'get_or_create', 'get', or 'create'.
        online: If True, log online.
        prefix: Prefix for metric keys.
        **kwargs: Additional kwargs to comet_ml.ExperimentConfig.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        experiment_key: Optional[str] = None,
        mode: Optional[str] = None,
        online: Optional[bool] = None,
        prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key
        self._workspace = workspace
        self._project = project
        self._experiment_key = experiment_key
        self._mode = mode
        self._online = online
        self._prefix = prefix or ""
        self._kwargs = kwargs
        self._experiment = None

    @property
    def experiment(self) -> Any:
        if self._experiment is None:
            self._experiment = self._create_experiment()
        return self._experiment

    def _create_experiment(self) -> Any:
        try:
            import comet_ml
            config = comet_ml.ExperimentConfig(
                api_key=self._api_key,
                workspace=self._workspace,
                project=self._project,
            )
            for k, v in self._kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            experiment = comet_ml.Experiment(config=config)
            if self._experiment_key:
                experiment.set_experiment_key(self._experiment_key)
            return experiment
        except ImportError:
            class _DummyExperiment:
                def log_metrics(self, *args, **kwargs): ...
                def log_parameters(self, *args, **kwargs): ...
                def __internal_api__log_metrics__(self, *args, **kwargs): ...
                def __internal_api__log_parameters__(self, *args, **kwargs): ...
                def end(self): ...
            return _DummyExperiment()

    @property
    def name(self) -> str:
        return self._project or "comet_logs"

    @property
    def version(self) -> str:
        if self._experiment is not None:
            return getattr(self._experiment, "get_key", lambda: "")()
        return ""

    @property
    def save_dir(self) -> Optional[str]:
        return None

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        try:
            self.experiment.__internal_api__log_parameters__(
                parameters=params,
                framework="paddle-ocean",
                flatten_nested=True,
                source="manual",
            )
        except Exception:
            pass

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        try:
            m = {}
            for k, v in metrics.items():
                if hasattr(v, "item"):
                    v = v.item()
                m[k] = float(v)
            epoch = m.pop("epoch", None) if "epoch" in m else None
            self.experiment.__internal_api__log_metrics__(
                m,
                step=step,
                epoch=epoch,
                prefix=self._prefix,
                framework="paddle-ocean",
            )
        except Exception:
            pass

    def finalize(self, status: str) -> None:
        try:
            self.experiment.end()
        except Exception:
            pass

    def log_graph(self, model: Any, input_array: Any = None) -> None:
        pass
