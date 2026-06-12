"""Stochastic Weight Averaging callback.

Saves a running average of model weights during the last phase of training,
then swaps the averaged weights in at the end for improved generalization.

Follows Lightning's StochasticWeightAveraging pattern.
"""

from __future__ import annotations

from typing import Any

import paddle

from ocean.callbacks.callback import Callback


class StochasticWeightAveraging(Callback):
    """Stochastic Weight Averaging.

    Args:
        swa_epoch_start: epoch to start averaging from (0.0-1.0 = fraction, >1 = absolute)
        annealing_epochs: number of epochs for learning rate annealing
        swa_lrs: learning rate(s) for SWA phase
    """

    def __init__(
        self,
        swa_epoch_start: float | int = 0.8,
        annealing_epochs: int = 10,
        swa_lrs: float = 1e-3,
    ):
        super().__init__()
        self.swa_epoch_start = swa_epoch_start
        self.annealing_epochs = annealing_epochs
        self.swa_lrs = swa_lrs
        self._swa_weights: list[paddle.Tensor] | None = None
        self._nb_averaged: int = 0
        self._has_started: bool = False

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        self._swa_weights = None
        self._nb_averaged = 0
        self._has_started = False

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        current_epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs

        # Determine when to start SWA
        if isinstance(self.swa_epoch_start, float):
            start_epoch = int(self.swa_epoch_start * max_epochs)
        else:
            start_epoch = self.swa_epoch_start

        if current_epoch >= start_epoch and not self._has_started:
            self._has_started = True
            # Initialize SWA weights from current model
            self._swa_weights = [p.clone().detach() for p in pl_module.parameters() if p.trainable]
            self._nb_averaged = 1

    def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if not self._has_started:
            return
        # Update running average
        if self._swa_weights is not None:
            params = [p for p in pl_module.parameters() if p.trainable]
            if len(params) != len(self._swa_weights):
                return
            self._nb_averaged += 1
            n = self._nb_averaged
            for i, p in enumerate(params):
                self._swa_weights[i] = self._swa_weights[i] + (p.detach() - self._swa_weights[i]) / n

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if not self._has_started:
            return
        # Log SWA info
        pl_module.log("swa/n_averaged", self._nb_averaged)

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """Swap model weights with the averaged version."""
        if self._swa_weights is None or self._nb_averaged < 2:
            return
        params = [p for p in pl_module.parameters() if p.trainable]
        for i, p in enumerate(params):
            if i < len(self._swa_weights):
                p.set_value(self._swa_weights[i])
