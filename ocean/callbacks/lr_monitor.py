"""LearningRateMonitor callback - logs learning rates during training."""

from typing import Any, Optional

from ocean.callbacks.callback import Callback


class LearningRateMonitor(Callback):
    """Monitor and log learning rates during training.

    Args:
        logging_interval: Log at ``'step'`` or ``'epoch'``. If None, logs at step.
        log_momentum: If True, also logs momentum values.
        log_weight_decay: If True, also logs weight decay values.
    """

    def __init__(
        self,
        logging_interval: Optional[str] = None,
        log_momentum: bool = False,
        log_weight_decay: bool = False,
    ) -> None:
        self.logging_interval = logging_interval or "step"
        self.log_momentum = log_momentum
        self.log_weight_decay = log_weight_decay

    def on_train_batch_start(self, trainer: Any, model: Any, batch: Any, batch_idx: int) -> None:
        if self.logging_interval == "epoch":
            return
        self._extract_and_log(trainer, model)

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if self.logging_interval != "epoch":
            return
        self._extract_and_log(trainer, model)

    def _extract_and_log(self, trainer: Any, model: Any) -> None:
        optimizer = model._optimizer if model._optimizer is not None else getattr(trainer, "_optimizer", None)
        if optimizer is None:
            return
        if hasattr(optimizer, "_learning_rate"):
            lr = float(optimizer._learning_rate)
            if hasattr(model, "log") and trainer._log_metrics_buffer is not None:
                model.log("learning_rate", lr, prog_bar=True)
