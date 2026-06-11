"""Spike detection callback - detects loss spikes during training."""

from typing import Any

import paddle

from ocean.callbacks.callback import Callback


class SpikeDetection(Callback):
    """Detect loss spikes (sudden large increases) during training.

    Args:
        threshold_multiplier: Multiplier of rolling mean to trigger alert.
        window_size: Number of steps for rolling statistics.
        verbose: If True, print spike warnings.
    """

    def __init__(
        self,
        threshold_multiplier: float = 5.0,
        window_size: int = 10,
        verbose: bool = True,
    ) -> None:
        self.threshold_multiplier = threshold_multiplier
        self.window_size = window_size
        self.verbose = verbose
        self._loss_history: list[float] = []

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        loss = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif isinstance(outputs, paddle.Tensor):
            loss = outputs

        if loss is not None:
            loss_val = float(loss.item())
            self._loss_history.append(loss_val)
            if len(self._loss_history) > self.window_size:
                self._loss_history.pop(0)

            if len(self._loss_history) >= self.window_size:
                mean = sum(self._loss_history) / len(self._loss_history)
                if loss_val > mean * self.threshold_multiplier:
                    if self.verbose:
                        print(f"Spike detected: loss={loss_val:.4f} (mean={mean:.4f}, "
                              f"threshold={mean*self.threshold_multiplier:.4f})")
