"""_TrainingEpochLoop - processes all batches in one epoch."""

from typing import Any, Optional

import paddle

from ocean.loops.loop import _Loop
from ocean.loops.progress import _BatchProgress, _SchedulerProgress
from ocean.loops.fetchers import _PrefetchDataFetcher
from ocean.loops.optimization import _AutomaticOptimization, _ManualOptimization
from ocean.loops.evaluation_loop import _EvaluationLoop
from ocean.trainer.states import TrainerFn, RunningStage


class _TrainingEpochLoop(_Loop):
    """Processes all batches in a single training epoch.

    Delegates per-batch work to automatic or manual optimization sub-loops.
    Runs validation at configured intervals.
    """

    def __init__(self, trainer: Any, min_steps: Optional[int] = None, max_steps: int = -1) -> None:
        super().__init__(trainer)
        self.batch_progress = _BatchProgress()
        self.scheduler_progress = _SchedulerProgress()
        self.automatic_optimization = _AutomaticOptimization(trainer)
        self.manual_optimization = _ManualOptimization(trainer)
        self.val_loop = _EvaluationLoop(trainer, TrainerFn.FITTING, RunningStage.VALIDATING, verbose=False)
        self.min_steps = min_steps
        self.max_steps = max_steps
        self._data_fetcher = _PrefetchDataFetcher()

    @property
    def done(self) -> bool:
        if self.trainer.should_stop:
            return True
        return self.global_step >= self.max_steps if self.max_steps > 0 else False

    @property
    def global_step(self) -> int:
        return self.trainer.global_step

    def run(self, data_fetcher: Any) -> None:
        self._data_fetcher = data_fetcher
        while not self.done:
            try:
                batch = next(self._data_fetcher)
                batch_idx = self.batch_progress.current.ready
                is_last = self._data_fetcher.done

                self._process_batch(batch, batch_idx, is_last)

                if is_last:
                    break
            except StopIteration:
                break

    def _process_batch(self, batch: Any, batch_idx: int, is_last: bool) -> None:
        trainer = self.trainer
        model = trainer._model
        device = trainer._resolve_device()

        self.batch_progress.increment_ready()
        batch = trainer._move_to_device(batch, device)

        # On train batch start hook
        skip = model.on_train_batch_start(batch, batch_idx)
        if skip == -1:
            return

        self.batch_progress.increment_started()

        # Training step via optimization sub-loop
        kwargs = {"batch": batch, "batch_idx": batch_idx}
        if model.automatic_optimization:
            result = self.automatic_optimization.run(trainer._optimizer, batch_idx, kwargs)
        else:
            result = self.manual_optimization.run(kwargs)

        self.batch_progress.increment_processed()

        # On train batch end hook
        model.on_train_batch_end(result, batch, batch_idx)

        self.batch_progress.increment_completed()
        self.batch_progress.update_last_batch(is_last)

    def on_advance_end(self, data_fetcher: Any) -> None:
        """Run validation if needed, update LR schedulers."""
        if self._should_check_val():
            self.val_loop.run()

    def _should_check_val(self) -> bool:
        return False  # Validation is managed by FitLoop

    def teardown(self) -> None:
        self._data_fetcher.teardown()
