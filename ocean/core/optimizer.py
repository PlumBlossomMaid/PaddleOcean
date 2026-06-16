"""OceanOptimizer and optimizer initialization utilities."""

from typing import Any, Optional

import paddle


class OceanOptimizer:
    """Wrapper around a Paddle optimizer that adds hooks for the training loop.

    Analogous to paddleOceanOptimizer in paddleOcean.
    """

    def __init__(self, optimizer: paddle.optimizer.Optimizer) -> None:
        self._optimizer = optimizer
        self._on_before_step = lambda: None
        self._on_after_step = lambda: None

    @property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        return self._optimizer

    def step(self, closure: Optional[Any] = None) -> None:
        self._on_before_step()
        if closure is not None:
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
        sch = result.get("lr_scheduler")
        if sch is not None:
            lr_schedulers = [
                {
                    "scheduler": sch,
                    "interval": result.get("interval", "epoch"),
                    "frequency": result.get("frequency", 1),
                    "monitor": result.get("monitor"),
                }
            ]

    return optimizers, lr_schedulers
