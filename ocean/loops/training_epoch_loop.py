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

        for batch_idx, batch in enumerate(iter(train_loader)):
            if trainer._should_limit_batches(batch_idx, "train"):
                break
            if max_batches and batch_idx >= max_batches:
                break

            self.batch_progress.increment_ready()
            is_last = bool(max_batches) and batch_idx == max_batches - 1
            self.batch_progress.update_last_batch(is_last)

            batch = trainer._move_to_device(batch, device)

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
            metric = trainer.logged_metrics.get(monitor) if monitor else None
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

        model.on_validation_model_eval()
        _call_module_hook(trainer, "on_validation_start")
        _call_callback_hooks(trainer, "on_validation_start")
        _call_module_hook(trainer, "on_validation_epoch_start")
        _call_callback_hooks(trainer, "on_validation_epoch_start")

        device = trainer._resolve_device()
        for dataloader in val_loader if isinstance(val_loader, (list, tuple)) else [val_loader]:
            with paddle.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if trainer._should_limit_batches(batch_idx, "val"):
                        break
                    batch = trainer._move_to_device(batch, device)
                    _call_callback_hooks(trainer, "on_validation_batch_start", batch, batch_idx, dataloader_idx=0)
                    model.on_validation_batch_start(batch, batch_idx)
                    result = model.validation_step(batch, batch_idx)
                    model.on_validation_batch_end(result, batch, batch_idx)
                    _call_callback_hooks(trainer, "on_validation_batch_end", result, batch, batch_idx, dataloader_idx=0)

        trainer._compute_epoch_metrics()
        _call_module_hook(trainer, "on_validation_epoch_end")
        _call_callback_hooks(trainer, "on_validation_epoch_end")
        _call_module_hook(trainer, "on_validation_end")
        _call_callback_hooks(trainer, "on_validation_end")
        # Clear val/test metrics so they don't leak into training log flushes.
        trainer._logger_connector.reset_validation_metrics()
        model.on_validation_model_train()

    def teardown(self) -> None:
        pass
