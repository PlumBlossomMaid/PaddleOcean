"""LR finder callback - learning rate range test."""

from typing import Any, Optional

import paddle

from ocean.callbacks.callback import Callback


class LRFinder(Callback):
    """Learning rate range test.

    Runs the model for a small number of steps while linearly/exponentially
    increasing the learning rate, and records the loss at each step.

    Args:
        min_lr: Minimum learning rate.
        max_lr: Maximum learning rate.
        num_training_steps: Number of steps for the test.
        mode: 'exponential' or 'linear'.
        early_stop_threshold: Stop when loss exceeds this threshold * best_loss.
    """

    def __init__(
        self,
        min_lr: float = 1e-8,
        max_lr: float = 1.0,
        num_training_steps: int = 100,
        mode: str = "exponential",
        early_stop_threshold: float = 4.0,
    ) -> None:
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_training_steps = num_training_steps
        self.mode = mode
        self.early_stop_threshold = early_stop_threshold
        self.results: list[tuple[float, float]] = []  # (lr, loss)
        self.optimal_lr: Optional[float] = None
        self._step = 0
        self._best_loss = float("inf")

    def on_train_batch_start(self, trainer: Any, model: Any, batch: Any, batch_idx: int) -> None:
        if self._step >= self.num_training_steps:
            return

        # Compute current LR
        if self.mode == "exponential":
            ratio = self._step / max(self.num_training_steps - 1, 1)
            lr = self.min_lr * (self.max_lr / self.min_lr) ** ratio
        else:
            ratio = self._step / max(self.num_training_steps - 1, 1)
            lr = self.min_lr + (self.max_lr - self.min_lr) * ratio

        # Set LR
        self._set_lr(model, lr)
        self._step += 1

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        loss = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif isinstance(outputs, paddle.Tensor):
            loss = outputs

        if loss is not None and self._step > 0:
            loss_val = float(loss.item())
            self._best_loss = min(self._best_loss, loss_val)

            # Current LR
            lr = self._get_lr(model)
            self.results.append((lr, loss_val))

            # Early stop
            if loss_val > self.early_stop_threshold * self._best_loss and len(self.results) > 5:
                # Suggest LR as 1/10 of the point of steepest descent
                if len(self.results) >= 3:
                    grads = [self.results[i+1][1] - self.results[i][1] for i in range(len(self.results)-1)]
                    steepest = grads.index(min(grads))
                    self.optimal_lr = self.results[steepest][0] * 0.1
                else:
                    self.optimal_lr = lr * 0.1

    def _set_lr(self, model: Any, lr: float) -> None:
        opt = model._optimizer
        if opt is not None and hasattr(opt, "_learning_rate"):
            opt._learning_rate = lr

    def _get_lr(self, model: Any) -> float:
        opt = model._optimizer
        if opt is not None and hasattr(opt, "_learning_rate"):
            return float(opt._learning_rate)
        return 0.001
