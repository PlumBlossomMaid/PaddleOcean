"""_FitLoop - orchestrates training across epochs.

Counts epochs and delegates each epoch's batch loop to :class:`_TrainingEpochLoop`.
The nested structure is the natural fit for this division of labour::

    for epoch in epochs:          # _FitLoop
        for batch in train_dl:    # _TrainingEpochLoop
            optimizer step        # _AutomaticOptimization / _ManualOptimization
"""

from typing import Any, Optional

from ocean.loops.loop import _Loop
from ocean.loops.training_epoch_loop import _TrainingEpochLoop
from ocean.trainer.call import _call_callback_hooks, _call_module_hook
from ocean.trainer.connectors.logger_connector.result import _ResultCollection


class _FitLoop(_Loop):
    """Top-level training loop that iterates over epochs."""

    def __init__(self, trainer: Any, min_epochs: int = 0, max_epochs: Optional[int] = None) -> None:
        super().__init__(trainer)
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs or 1000
        self.epoch_loop = _TrainingEpochLoop(trainer)
        # Loader-state pipeline: the combined loader wraps the raw train dataloader
        # so it can carry and restore per-loader state dicts at checkpoint time.
        # ``_combined_loader_states_to_load`` is staged by
        # ``_CheckpointConnector.restore`` and consumed once on the next ``run()``,
        # only when actually restarting. With plain Paddle loaders these lists stay
        # empty — the pipeline is a no-op and behaviour is unchanged, but it is the
        # opt-in point for a loader that chooses to be stateful.
        self._combined_loader = None
        self._combined_loader_states_to_load: list = []

    def _setup_combined_loader(self, train_loader: Any) -> None:
        """Wrap the raw train loader in a CombinedLoader.

        For a single train loader this is the trivial sequential wrapper. The
        wrapper is what ``_TrainingEpochLoop._is_resumable_loader`` probes for
        statefulness, so building it up-front keeps that probe valid even before the
        first epoch runs.
        """
        from ocean.utils.combined_loader import CombinedLoader

        self._combined_loader = (
            train_loader
            if isinstance(train_loader, CombinedLoader)
            else CombinedLoader(train_loader, mode="sequential")
        )

    def _load_combined_loader_states(self) -> None:
        """Apply staged loader state only when restarting and state was staged.

        A fresh (non-resume) run, or a plain non-stateful loader, fails the guard
        and nothing is seeded — the epoch runs from batch 0. After the staged list
        is consumed it is released so it can only be applied once.
        """
        states = self._combined_loader_states_to_load
        if not self.restarting or not states or self._combined_loader is None:
            return
        self._combined_loader._load_state_dicts(states)
        self._combined_loader_states_to_load = []

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

        # Cache the limit-adjusted batch count once for the whole run. Using the
        # resolved limit (not the raw len) makes both int and fractional
        # limit_train_batches (e.g. 0.5) actually cap training, and drives
        # num_training_batches.
        try:
            effective_train_batches = trainer._resolve_limit(train_loader, trainer.limit_train_batches)
        except TypeError:
            effective_train_batches = 0
        self.epoch_loop._max_batches = effective_train_batches

        # The validation schedule keys off the same limit-adjusted count (so
        # val_check_interval=1.0 lands on the effective last batch of each epoch).
        trainer._setup_val_check_batch(effective_train_batches)

        # Mark the running stage so trainer.training and metric fx-keying are
        # correct during the training loop.
        trainer.training = True

        # Training metrics accumulate in a dedicated collection for the whole fit.
        # A mid-epoch validation swaps in its own collection, so it can never clear
        # the training accumulation gathered earlier in the epoch.
        trainer._results = _ResultCollection(training=True, fork_names=False)

        _call_module_hook(trainer, "on_train_start")
        _call_callback_hooks(trainer, "on_train_start")

        while not self.done:
            # Optionally rebuild the training dataloader from the datamodule.
            self._reload_train_dataloader_if_needed()

            # Wrap the (possibly just-rebuilt) train loader for the combined-loader
            # pipeline. ``_load_combined_loader_states`` is a guarded no-op except on
            # the restarting epoch where staged restore state is applied once; on a
            # reload epoch the wrapper switches to the fresh loader and nothing is
            # reseeded, matching how a plain non-stateful loader re-yields from 0.
            self._setup_combined_loader(trainer.train_dataloader)
            self._load_combined_loader_states()

            # Fresh accumulation each epoch (weighted means are per-epoch).
            trainer._results = _ResultCollection(training=True, fork_names=False)

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

    def _reload_train_dataloader_if_needed(self) -> None:
        """Rebuild the training dataloader every ``reload_dataloaders_every_n_epochs``.

        Only possible when a datamodule is attached (a raw dataloader has no source
        to reload from). Re-runs ``setup('fit')`` so datasets can change per epoch,
        then re-caches the effective batch count.
        """
        trainer = self.trainer
        n = getattr(trainer, "reload_dataloaders_every_n_epochs", 0)
        datamodule = getattr(trainer, "datamodule", None)
        if not n or datamodule is None:
            return
        if trainer.current_epoch == 0 or trainer.current_epoch % n != 0:
            return

        datamodule.setup("fit")
        trainer.train_dataloader = datamodule.train_dataloader()
        try:
            self.epoch_loop._max_batches = trainer._resolve_limit(trainer.train_dataloader, trainer.limit_train_batches)
        except TypeError:
            self.epoch_loop._max_batches = 0

    def teardown(self) -> None:
        self.epoch_loop.teardown()

    def load_state_dict(self, state_dict: dict) -> None:
        # Legacy checkpoints stored ``batch_progress`` directly on the fit loop,
        # before the batch loop moved into the epoch loop. Route it to its new home.
        if "batch_progress" in state_dict and "epoch_loop" not in state_dict:
            self.epoch_loop.batch_progress.load_state_dict(state_dict["batch_progress"])
            # Use the property setter so the restart flag cascades to child loops
            # (matches what super().load_state_dict does after the loop.py fix).
            self.restarting = True
            self._resuming_from_checkpoint = True
            self.epoch_loop._resuming_from_checkpoint = True
            return
        super().load_state_dict(state_dict)
