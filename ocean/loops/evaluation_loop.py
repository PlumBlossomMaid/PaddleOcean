"""_EvaluationLoop - runs validation or test loops."""

from typing import Any, Optional

import paddle

from ocean.loops.loop import _Loop
from ocean.loops.progress import _BatchProgress
from ocean.loops.fetchers import _DataFetcher
from ocean.trainer.states import TrainerFn, RunningStage
from ocean.trainer.call import _call_callback_hooks, _call_lightning_module_hook


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

        model.eval()
        _call_lightning_module_hook(trainer, start_hook)
        _call_callback_hooks(trainer, start_hook)
        _call_lightning_module_hook(trainer, epoch_start_hook)

        with paddle.no_grad():
            for dl_idx, dataloader in enumerate(dataloaders):
                for batch_idx, batch in enumerate(dataloader):
                    device = trainer._resolve_device()
                    batch = trainer._move_to_device(batch, device)
                    _call_lightning_module_hook(trainer, batch_start_hook, batch, batch_idx, dl_idx)
                    result = getattr(model, step_method)(batch, batch_idx)
                    _call_lightning_module_hook(trainer, batch_end_hook, result, batch, batch_idx, dl_idx)

        trainer._compute_epoch_metrics()
        _call_lightning_module_hook(trainer, epoch_end_hook)
        _call_lightning_module_hook(trainer, end_hook)
        _call_callback_hooks(trainer, end_hook)
        model.train()

        return [dict(trainer._log_metrics_on_epoch)]

    def teardown(self) -> None:
        self._data_fetcher.teardown()
