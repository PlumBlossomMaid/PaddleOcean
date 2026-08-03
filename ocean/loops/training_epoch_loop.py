"""_TrainingEpochLoop - processes all batches in one training epoch.

Owns the per-batch loop: batch hooks, gradient-accumulation decision, per-step
LR-scheduler stepping, mid-epoch validation, and metric logging. Delegates the
actual forward/backward/step to the automatic or manual optimization sub-loop.

Iterates the DataLoader directly (rather than through a prefetching fetcher) to
avoid PaddlePaddle shared-memory issues; ``is_last_batch`` is derived from the
cached batch count instead.
"""

from typing import Any

import paddle

from ocean.loops.loop import _Loop
from ocean.loops.optimization import _AutomaticOptimization, _ManualOptimization
from ocean.loops.progress import _BatchProgress, _SchedulerProgress
from ocean.trainer.call import _call_callback_hooks, _call_module_hook
from ocean.trainer.connectors.logger_connector.result import _ResultCollection


class _TrainingEpochLoop(_Loop):
    """Processes all batches in a single training epoch."""

    def __init__(self, trainer: Any) -> None:
        super().__init__(trainer)
        self.batch_progress = _BatchProgress()
        self.scheduler_progress = _SchedulerProgress()
        self.automatic_optimization = _AutomaticOptimization(trainer)
        self.manual_optimization = _ManualOptimization(trainer)
        # Number of training batches per epoch, cached by _FitLoop at run() start.
        self._max_batches: int = 0

    @property
    def max_batches(self) -> int:
        return self._max_batches

    # ------------------------------------------------------------------
    # Accumulation predicates
    # ------------------------------------------------------------------
    def _accumulated_batches_reached(self) -> bool:
        return self.batch_progress.current.ready % max(1, self.trainer.accumulate_grad_batches) == 0

    def _num_ready_batches_reached(self) -> bool:
        return self.batch_progress.is_last_batch

    def _should_accumulate(self) -> bool:
        """True when gradients should keep accumulating (no optimizer step yet)."""
        return not self._accumulated_batches_reached() and not self._num_ready_batches_reached()

    def _is_resumable_loader(self) -> bool:
        """Whether every train loader in the combined loader is stateful.

        A loader is considered stateful here when it exposes a ``state_dict``
        method (the ability to persist and restore its own iteration state). Only
        then is a mid-epoch resume meaningful: such a loader re-yields the
        already-skipped batches on restart, so its ``batch_progress`` is worth
        preserving. Plain Paddle loaders are not stateful and re-yield from batch 0,
        so reporting any preserved mid-epoch progress would be wrong.
        """
        loader = getattr(self.trainer.fit_loop, "_combined_loader", None)
        if loader is None:
            return False
        flattened = getattr(loader, "flattened", None)
        if not flattened:
            return False
        return all(hasattr(ld, "state_dict") for ld in flattened)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        trainer = self.trainer
        model = trainer._model
        train_loader = getattr(trainer.fit_loop, "_combined_loader", None) or getattr(trainer, "train_dataloader", None)
        if train_loader is None:
            return

        # On restart, preserve the epoch's batch_progress only when the loader is
        # stateful, so a resumed dataloader that genuinely skips already-processed
        # batches keeps counting from where it stopped. A plain (non-stateful) loader
        # is rebuilt fresh from the checkpoint and re-yields from batch 0, so it is
        # reset to zero — preserving a mid-epoch count here would describe progress
        # the data does not actually have. The epoch then runs plainly from batch 0.
        if self.restarting and self._is_resumable_loader():
            self.batch_progress.reset_on_restart()
        else:
            self.batch_progress.reset_on_run()

        device = trainer._resolve_device()
        max_batches = self._max_batches

        # The loader is iterated directly (no prefetch fetcher): on resume with a
        # plain non-stateful loader it re-yields from batch 0, so the epoch is
        # honestly reprocessed and batch_idx starts at 0. A ``_Stateful`` loader
        # (future: a Paddle ``StatefulDataLoader`` equivalent) would resume mid-epoch
        # because its ``__iter__`` yields the already-skipped batches, in which case
        # batch_idx still from 0 is correct since the iterator itself advanced.
        pending_grads = False
        last_batch_idx = -1
        for batch_idx, batch in enumerate(iter(train_loader)):
            # ``max_batches`` is already limit-adjusted and may be inf for a
            # loader with no length. Comparing directly (rather than guarding on
            # truthiness first) is what makes a cap of 0 mean "no batches"
            # instead of "no cap".
            if batch_idx >= max_batches:
                break

            last_batch_idx = batch_idx
            self.batch_progress.increment_ready()
            is_last = batch_idx == max_batches - 1
            self.batch_progress.update_last_batch(is_last)

            batch = trainer._move_to_device(batch, device)

            # Attach the batch so epoch-mean metrics are batch-size weighted.
            if trainer._results is not None:
                trainer._results.batch = batch
                trainer._results.batch_size = None

            _call_callback_hooks(trainer, "on_train_batch_start", batch, batch_idx)
            skip_flag = model.on_train_batch_start(batch, batch_idx)
            if skip_flag == -1:
                # Keep completed in step with ready so progress bookkeeping stays consistent.
                self.batch_progress.increment_completed()
                continue

            self.batch_progress.increment_started()

            # Accumulation decision is owned here; the optimization sub-loop reacts to it.
            should_step = not self._should_accumulate()

            kwargs = {"batch": batch, "batch_idx": batch_idx}
            with trainer.profiler.profile("run_training_batch"):
                if model.automatic_optimization:
                    result = self.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs, should_step)
                else:
                    result = self.manual_optimization.run(kwargs)

            self.batch_progress.increment_processed()

            # Advance the global step + schedulers + logging on a real optimizer step.
            if model.automatic_optimization:
                pending_grads = not should_step
                if should_step:
                    trainer._dataloader_step += 1
                    self._update_lr_schedulers("step")
                    self._maybe_log_metrics()
            else:
                # Manual mode: the user drives optimizer.step(); count each batch as a step.
                trainer._dataloader_step += 1
                self._maybe_log_metrics()

            model.on_train_batch_end(result, batch, batch_idx)
            _call_callback_hooks(trainer, "on_train_batch_end", result, batch, batch_idx)
            self.batch_progress.increment_completed()

            if trainer._should_check_val_step(batch_idx):
                self._run_validation()

            if 0 < trainer.max_steps <= trainer.dataloader_step:
                trainer.should_stop = True
                break

            if is_last:
                break
        else:
            # Reached only when the loader ran dry on its own (no break): every
            # stopping condition above either closed the accumulation window via
            # ``is_last`` or is deliberately cutting the run short. A sized epoch
            # closes on ``is_last``, but a streaming loader has no known last
            # batch, so without this flush the gradients accumulated by the
            # trailing batches would be dropped on the floor.
            self._flush_pending_gradients(pending_grads, last_batch_idx)

    def _flush_pending_gradients(self, pending_grads: bool, last_batch_idx: int) -> None:
        """Step the optimizer for a partially-filled accumulation window."""
        trainer = self.trainer
        model = trainer._model
        if pending_grads and model.automatic_optimization:
            self.automatic_optimization.optimizer_step(trainer.optimizers[0]._optimizer, last_batch_idx)
            trainer._dataloader_step += 1
            self._update_lr_schedulers("step")
            self._maybe_log_metrics()

    # ------------------------------------------------------------------
    # LR schedulers / logging
    # ------------------------------------------------------------------
    def _update_lr_schedulers(self, interval: str) -> None:
        """Step LR schedulers registered for ``interval`` (``"step"`` or ``"epoch"``).

        Schedulers are only auto-stepped in automatic optimization; in manual mode
        the user owns scheduler stepping.
        """
        trainer = self.trainer
        model = trainer._model
        if not model.automatic_optimization:
            return
        for sched_cfg in trainer._lr_schedulers:
            if sched_cfg.get("interval", "epoch") != interval:
                continue
            scheduler = sched_cfg["scheduler"]
            monitor = sched_cfg.get("monitor")
            # Monitored values are read from callback_metrics (reference behavior),
            # which — unlike logged_metrics — includes metrics logged with logger=False.
            metric = trainer.callback_metrics.get(monitor) if monitor else None
            self.scheduler_progress.increment_ready()
            model.lr_scheduler_step(scheduler, metric)
            self.scheduler_progress.increment_completed()

    def _maybe_log_metrics(self) -> None:
        trainer = self.trainer
        step = trainer.dataloader_step
        if step > 0 and step % max(1, trainer.log_every_n_steps) == 0:
            trainer._logger_connector.log_metrics(trainer.logged_metrics, step)

    # ------------------------------------------------------------------
    # Mid-epoch validation
    # ------------------------------------------------------------------
    def _run_validation(self) -> None:
        trainer = self.trainer
        model = trainer._model
        val_loader = getattr(trainer, "val_dataloaders", None)
        if not val_loader:
            return

        # Validation logs into its OWN collection; the training collection is set
        # aside untouched and restored afterwards. This is the core F9 fix — a
        # mid-epoch val can no longer clear the epoch's training accumulation.
        train_results = trainer._results
        trainer._results = _ResultCollection(training=False, fork_names=False)
        # Reflect the running stage so trainer.validating and fx-keying are correct.
        trainer.validating = True

        model.on_validation_model_eval()
        _call_module_hook(trainer, "on_validation_start")
        _call_callback_hooks(trainer, "on_validation_start")
        _call_module_hook(trainer, "on_validation_epoch_start")
        _call_callback_hooks(trainer, "on_validation_epoch_start")

        device = trainer._resolve_device()
        val_limit = getattr(trainer, "limit_val_batches", 1.0)
        try:
            for dataloader in val_loader if isinstance(val_loader, (list, tuple)) else [val_loader]:
                max_val_batches = trainer._resolve_limit(dataloader, val_limit, stage="val")
                with paddle.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        if batch_idx >= max_val_batches:
                            break
                        batch = trainer._move_to_device(batch, device)
                        trainer._results.batch = batch
                        trainer._results.batch_size = None
                        _call_callback_hooks(trainer, "on_validation_batch_start", batch, batch_idx, dataloader_idx=0)
                        model.on_validation_batch_start(batch, batch_idx)
                        with trainer.profiler.profile("[EvaluationLoop].validation_step"):
                            result = model.validation_step(batch, batch_idx)
                        model.on_validation_batch_end(result, batch, batch_idx)
                        _call_callback_hooks(
                            trainer, "on_validation_batch_end", result, batch, batch_idx, dataloader_idx=0
                        )

            trainer._compute_epoch_metrics()
            _call_module_hook(trainer, "on_validation_epoch_end")
            _call_callback_hooks(trainer, "on_validation_epoch_end")
            _call_module_hook(trainer, "on_validation_end")
            _call_callback_hooks(trainer, "on_validation_end")
        finally:
            # Discard the eval collection and restore the untouched training one.
            trainer._logger_connector.reset_validation_metrics()
            trainer._results = train_results
            trainer.training = True  # back to the training stage
            model.on_validation_model_train()
            # Restart the wall-clock window for time-based validation scheduling.
            if trainer._val_check_time_interval is not None:
                import time

                trainer._last_val_time = time.monotonic()

    def teardown(self) -> None:
        pass
