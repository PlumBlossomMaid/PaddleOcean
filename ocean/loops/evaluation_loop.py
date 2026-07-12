"""_EvaluationLoop - runs validation or test loops."""

import inspect
from typing import Any

import paddle

from ocean.loops.fetchers import _DataFetcher
from ocean.loops.loop import _Loop
from ocean.loops.progress import _BatchProgress
from ocean.trainer.call import _call_callback_hooks, _call_module_hook
from ocean.trainer.connectors.logger_connector.result import _ResultCollection
from ocean.trainer.states import RunningStage, TrainerFn


class _EvaluationLoop(_Loop):
    """Runs validation or test evaluation across all dataloaders."""

    def __init__(
        self,
        trainer: Any,
        trainer_fn: TrainerFn,
        stage: RunningStage,
        verbose: bool = True,
    ) -> None:
        super().__init__(trainer)
        self.trainer_fn = trainer_fn
        self.stage = stage
        self.verbose = verbose
        self.batch_progress = _BatchProgress()
        self._data_fetcher = _DataFetcher()
        self._outputs: list[dict[str, float]] = []

    @property
    def skip(self) -> bool:
        return False

    def run(self) -> list[dict[str, float]]:
        trainer = self.trainer
        model = trainer._model

        # Get dataloaders
        if self.stage == RunningStage.VALIDATING:
            dataloaders = getattr(trainer, "val_dataloaders", None) or []
            start_hook = "on_validation_start"
            epoch_start_hook = "on_validation_epoch_start"
            step_method = "validation_step"
            batch_start_hook = "on_validation_batch_start"
            batch_end_hook = "on_validation_batch_end"
            epoch_end_hook = "on_validation_epoch_end"
            end_hook = "on_validation_end"
        elif self.stage == RunningStage.TESTING:
            dataloaders = getattr(trainer, "test_dataloaders", None) or []
            start_hook = "on_test_start"
            epoch_start_hook = "on_test_epoch_start"
            step_method = "test_step"
            batch_start_hook = "on_test_batch_start"
            batch_end_hook = "on_test_batch_end"
            epoch_end_hook = "on_test_epoch_end"
            end_hook = "on_test_end"
        else:
            return []

        if not dataloaders:
            return []

        # Reflect the running stage (VALIDATING / TESTING) on the trainer so
        # trainer.validating/testing and metric fx-keying are correct.
        trainer.state.stage = self.stage

        # Evaluation logs into its own (training=False) collection.
        trainer._results = _ResultCollection(training=False, fork_names=False)

        model.eval()
        _call_module_hook(trainer, start_hook)
        _call_callback_hooks(trainer, start_hook)
        _call_module_hook(trainer, epoch_start_hook)
        _call_callback_hooks(trainer, epoch_start_hook)

        # Resolve the batch cap from limit_{val,test}_batches (supports int and
        # fractional limits); previously this loop ignored the limit entirely.
        mode = "test" if self.stage == RunningStage.TESTING else "val"
        limit = getattr(trainer, f"limit_{mode}_batches", 1.0)

        with paddle.no_grad():
            for dl_idx, dataloader in enumerate(dataloaders):
                max_batches = trainer._resolve_limit(dataloader, limit)
                for batch_idx, batch in enumerate(dataloader):
                    if max_batches and batch_idx >= max_batches:
                        break
                    device = trainer._resolve_device()
                    batch = trainer._move_to_device(batch, device)
                    trainer._results.batch = batch
                    trainer._results.batch_size = None
                    _call_callback_hooks(trainer, batch_start_hook, batch, batch_idx, dl_idx)
                    _call_module_hook(trainer, batch_start_hook, batch, batch_idx, dl_idx)
                    step_fn = getattr(model, step_method)
                    # Check if step_method accepts dataloader_idx (backward compat)
                    sig = inspect.signature(step_fn)
                    if "dataloader_idx" in sig.parameters:
                        result = step_fn(batch, batch_idx, dataloader_idx=dl_idx)
                    else:
                        result = step_fn(batch, batch_idx)
                    _call_module_hook(trainer, batch_end_hook, result, batch, batch_idx, dl_idx)
                    _call_callback_hooks(trainer, batch_end_hook, result, batch, batch_idx, dl_idx)

        trainer._compute_epoch_metrics()
        _call_module_hook(trainer, epoch_end_hook)
        _call_callback_hooks(trainer, epoch_end_hook)
        _call_module_hook(trainer, end_hook)
        _call_callback_hooks(trainer, end_hook)
        model.train()

        # Dispatch logged metrics to the loggers so a standalone validate/test
        # run surfaces results through every backend (during fit, metrics from
        # mid-epoch validation are flushed through the training step path).
        from ocean.trainer.states import TrainerFn

        if trainer.state.fn in (TrainerFn.VALIDATING, TrainerFn.TESTING):
            logged = dict(trainer._log_metrics_on_epoch)
            if logged and trainer.loggers:
                trainer._logger_connector.log_metrics(logged, step=trainer.current_epoch)

        return [dict(trainer._log_metrics_on_epoch)]

    def teardown(self) -> None:
        self._data_fetcher.teardown()
