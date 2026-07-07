"""_FitLoop - orchestrates training across epochs.

Counts epochs and delegates each epoch's batch loop to :class:`_TrainingEpochLoop`.
The nested structure mirrors the reference design::

    for epoch in epochs:          # _FitLoop
        for batch in train_dl:    # _TrainingEpochLoop
            optimizer step        # _AutomaticOptimization / _ManualOptimization
"""

from typing import Any, Optional

from ocean.loops.loop import _Loop
from ocean.loops.training_epoch_loop import _TrainingEpochLoop
from ocean.trainer.call import _call_callback_hooks, _call_module_hook


class _FitLoop(_Loop):
    """Top-level training loop that iterates over epochs."""

    def __init__(self, trainer: Any, min_epochs: int = 0, max_epochs: Optional[int] = None) -> None:
        super().__init__(trainer)
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs or 1000
        self.epoch_loop = _TrainingEpochLoop(trainer)

    @property
    def done(self) -> bool:
        trainer = self.trainer
        if trainer.current_epoch >= self.max_epochs:
            return True
        if trainer.should_stop and trainer.current_epoch >= self.min_epochs:
            return True
        return False

    @property
    def max_batches(self) -> int:
        """Number of training batches per epoch (cached once at run() start)."""
        return self.epoch_loop.max_batches

    def run(self) -> None:
        trainer = self.trainer
        train_loader = getattr(trainer, "train_dataloader", None)
        if train_loader is None:
            return

        # Cache the batch count once for the whole run (drives num_training_batches).
        try:
            self.epoch_loop._max_batches = len(train_loader)
        except TypeError:
            self.epoch_loop._max_batches = 0

        _call_module_hook(trainer, "on_train_start")
        _call_callback_hooks(trainer, "on_train_start")

        while not self.done:
            _call_module_hook(trainer, "on_train_epoch_start")
            _call_callback_hooks(trainer, "on_train_epoch_start")

            self.epoch_loop.run()

            trainer._compute_epoch_metrics()
            _call_module_hook(trainer, "on_train_epoch_end")
            _call_callback_hooks(trainer, "on_train_epoch_end")

            # Step epoch-interval LR schedulers at epoch end.
            self.epoch_loop._update_lr_schedulers("epoch")

            trainer.current_epoch += 1

            # Restart handling complete — clear the flag (cascades to child loops).
            self.restarting = False

            if trainer._should_stop():
                break

        _call_module_hook(trainer, "on_train_end")
        _call_callback_hooks(trainer, "on_train_end")

    def teardown(self) -> None:
        self.epoch_loop.teardown()

    def load_state_dict(self, state_dict: dict) -> None:
        # Legacy checkpoints stored ``batch_progress`` directly on the fit loop,
        # before the batch loop moved into the epoch loop. Route it to its new home.
        if "batch_progress" in state_dict and "epoch_loop" not in state_dict:
            self.epoch_loop.batch_progress.load_state_dict(state_dict["batch_progress"])
            self._restarting = True
            self._resuming_from_checkpoint = True
            self.epoch_loop._restarting = True
            self.epoch_loop._resuming_from_checkpoint = True
            return
        super().load_state_dict(state_dict)
