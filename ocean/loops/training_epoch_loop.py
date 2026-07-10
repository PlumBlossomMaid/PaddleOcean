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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        trainer = self.trainer
        model = trainer._model
        train_loader = getattr(trainer, "train_dataloader", None)
        if train_loader is None:
            return

        if self.restarting:
            self.batch_progress.reset_on_restart()
        else:
            self.batch_progress.reset_on_run()

        device = trainer._resolve_device()
        max_batches = self._max_batches

        # On restart, skip the batches this epoch had already consumed before the
        # checkpoint. Lightning tracks this by advancing a data-fetcher's `fetched`
        # counter by `batch_progress.current.ready` in `on_run_start`; without a
        # fetcher here, the equivalent is to drain that many batches from the
        # loader before entering the main loop. `enumerate(..., start=...)` keeps
        # batch_idx continuous (matching Lightning's `batch_idx = self.batch_idx + 1`)
        # so hooks, logging, and mid-epoch validation see a true index.
        skip_batches = self.batch_progress.current.ready
        loader_iter = iter(train_loader)
        if skip_batches > 0:
            for _ in range(skip_batches):
                if next(loader_iter, None) is None:
                    break

        for batch_idx, batch in enumerate(loader_iter, start=skip_batches):
            if trainer._should_limit_batches(batch_idx, "train"):
                break
            if max_batches and batch_idx >= max_batches:
                break

            self.batch_progress.increment_ready()
            is_last = bool(max_batches) and batch_idx == max_batches - 1
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
            if model.automatic_optimization:
                result = self.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs, should_step)
            else:
                result = self.manual_optimization.run(kwargs)

            self.batch_progress.increment_processed()

            # Advance the global step + schedulers + logging on a real optimizer step.
            if model.automatic_optimization:
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
                max_val_batches = trainer._resolve_limit(dataloader, val_limit)
                with paddle.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        if max_val_batches and batch_idx >= max_val_batches:
                            break
                        batch = trainer._move_to_device(batch, device)
                        trainer._results.batch = batch
                        trainer._results.batch_size = None
                        _call_callback_hooks(trainer, "on_validation_batch_start", batch, batch_idx, dataloader_idx=0)
                        model.on_validation_batch_start(batch, batch_idx)
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
