"""OceanOptimizer and optimizer initialization utilities."""

import warnings
from typing import Any, Optional

import paddle

from ocean.utils import MisconfigurationException


class OceanOptimizer:
    """Wrapper around a Paddle optimizer that adds hooks for the training loop.

    Wraps a Paddle optimizer with training-loop step hooks.
    """

    def __init__(self, optimizer: paddle.optimizer.Optimizer) -> None:
        self._optimizer = optimizer
        self._on_before_step = lambda: None
        self._on_after_step = lambda: None
        # Set by the trainer so manual-mode ``step()`` routes the update through
        # the precision plugin (AMP GradScaler.step/update). None → raw step.
        self._precision_plugin: Optional[Any] = None

    @property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        return self._optimizer

    def step(self, closure: Optional[Any] = None) -> None:
        self._on_before_step()
        if self._precision_plugin is not None:
            # Mirrors Lightning's LightningOptimizer.step → strategy.optimizer_step,
            # so AMP scaling is unwound (scaler.step + scaler.update) on manual steps.
            self._precision_plugin.optimizer_step(self._optimizer)
        elif closure is not None:
            self._optimizer.step(closure)
        else:
            self._optimizer.step()
        self._on_after_step()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._optimizer, name)


def init_optimizers_and_lr_schedulers(model: Any) -> tuple[list, list]:
    """Call configure_optimizers() on the model and parse the result.

    Returns:
        Tuple of (optimizers, lr_scheduler_configs).
    """
    result = model.configure_optimizers()
    if result is None:
        return [], []

    optimizers = []
    lr_schedulers = []

    if isinstance(result, paddle.optimizer.Optimizer):
        optimizers = [result]
    elif isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, paddle.optimizer.Optimizer):
                optimizers.append(item)
            elif isinstance(item, dict):
                opt = item.get("optimizer")
                if opt is not None:
                    optimizers.append(opt)
                sch = item.get("lr_scheduler")
                if sch is not None:
                    lr_schedulers.append({
                        "scheduler": sch,
                        "interval": item.get("interval", "epoch"),
                        "frequency": item.get("frequency", 1),
                        "monitor": item.get("monitor"),
                    })
            elif isinstance(item, (list, tuple)):
                if item and isinstance(item[0], paddle.optimizer.Optimizer):
                    optimizers.extend(item)
    elif isinstance(result, dict):
        opt = result.get("optimizer")
        if opt is not None:
            optimizers = [opt]
        sch_or_cfg = result.get("lr_scheduler")
        if sch_or_cfg is not None:
            if isinstance(sch_or_cfg, dict):
                cfg = sch_or_cfg
                scheduler = cfg.get("scheduler", cfg)
                interval = cfg.get("interval", "epoch")
                frequency = cfg.get("frequency", 1)
                monitor = cfg.get("monitor")
            else:
                scheduler = sch_or_cfg
                interval = result.get("interval", "epoch")
                frequency = result.get("frequency", 1)
                monitor = result.get("monitor")
            lr_schedulers = [
                {
                    "scheduler": scheduler,
                    "interval": interval,
                    "frequency": frequency,
                    "monitor": monitor,
                }
            ]

    _validate_schedulers(lr_schedulers)
    _warn_unbound_schedulers(optimizers, lr_schedulers)
    return optimizers, lr_schedulers


def _validate_schedulers(lr_schedulers: list) -> None:
    """Reject misconfigurations that would otherwise fail silently at step time.

    A ``ReduceOnPlateau`` scheduler calls ``step(metric)`` and needs a monitored
    metric; without one it has nothing to react to. Fail fast rather than
    passing ``None`` to ``step`` at runtime.
    """
    plateau_type = None
    try:
        plateau_type = paddle.optimizer.lr.ReduceOnPlateau
    except AttributeError:  # older paddle without ReduceOnPlateau
        plateau_type = None
    for cfg in lr_schedulers:
        scheduler = cfg["scheduler"]
        is_plateau = plateau_type is not None and isinstance(scheduler, plateau_type)
        if is_plateau and cfg.get("monitor") is None:
            raise MisconfigurationException(
                "The lr scheduler dict must include a `monitor` when a "
                "`ReduceOnPlateau` scheduler is used. For example: "
                "{'optimizer': optimizer, 'lr_scheduler': "
                "{'scheduler': scheduler, 'monitor': 'your_loss'}}"
            )
        interval = cfg.get("interval", "epoch")
        if interval not in ("epoch", "step"):
            raise MisconfigurationException(
                f'The "interval" key in lr scheduler dict must be "step" or "epoch" but is "{interval}"'
            )


def _warn_unbound_schedulers(optimizers: list, lr_schedulers: list) -> None:
    """Warn when an LR scheduler is not bound to any optimizer's ``learning_rate``.

    In PaddlePaddle (where the scheduler holds a reference to the optimizer and
    writes into its param groups), PaddlePaddle stores the schedule *inside* the
    optimizer: it must be created as ``optimizer(learning_rate=scheduler, ...)``.
    If that binding is missing, ``scheduler.step()`` silently has no effect on the
    learning rate, so surface it early.
    """
    for cfg in lr_schedulers:
        scheduler = cfg["scheduler"]
        bound = any(getattr(opt, "_learning_rate", None) is scheduler for opt in optimizers)
        if not bound:
            warnings.warn(
                "LR scheduler is not bound to any optimizer's learning_rate. In PaddlePaddle "
                "the scheduler must be passed to the optimizer as "
                "`paddle.optimizer.X(learning_rate=scheduler, ...)`; otherwise scheduler.step() "
                "has no effect on the learning rate.",
                UserWarning,
                stacklevel=3,
            )
