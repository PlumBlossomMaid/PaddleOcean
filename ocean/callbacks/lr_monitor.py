"""LearningRateMonitor callback - logs learning rates during training."""

from typing import Any

from ocean.callbacks.callback import Callback


class LearningRateMonitor(Callback):
    """Monitor and log learning rates during training.

    Args:
        logging_interval: Log at ``'step'`` or ``'epoch'``. Defaults to ``'step'``.
    """

    def __init__(self, logging_interval: str = "step") -> None:
        self.logging_interval = logging_interval

    def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.logging_interval != "step":
            return
        self._record_lr(trainer, pl_module)

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if self.logging_interval != "epoch":
            return
        self._record_lr(trainer, pl_module)

    def _record_lr(self, trainer: Any, pl_module: Any) -> None:
        optimizers = trainer._optimizers
        for i, opt_wrapper in enumerate(optimizers):
            opt = opt_wrapper._optimizer
            lr = getattr(opt, "_learning_rate", None)
            if lr is not None:
                actual_lr = lr.numpy() if hasattr(lr, "numpy") else lr
                if hasattr(actual_lr, "__iter__"):
                    for j, lr_val in enumerate(actual_lr):
                        pl_module.log(f"lr_{i}_group_{j}", lr_val)
                else:
                    pl_module.log(f"lr_{i}", actual_lr)
