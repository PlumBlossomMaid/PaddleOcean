"""_PredictionLoop - runs prediction across all dataloaders.

Prediction is an inference-only pass: the model is switched to eval mode (so
dropout and batch-norm behave as they do at serving time) and gradients are
disabled for the whole run. The original per-sublayer training flags are
restored afterwards, so ``trainer.predict()`` never leaves the model in a
different mode than it was found in.
"""

import inspect
from typing import Any, Optional

import paddle

from ocean.loops.loop import _Loop
from ocean.trainer.call import _call_callback_hooks, _call_module_hook
from ocean.trainer.states import RunningStage
from ocean.utils.model_helpers import _ModuleMode


class _PredictionLoop(_Loop):
    """Runs prediction across all dataloaders."""

    def __init__(self, trainer: Any) -> None:
        super().__init__(trainer)
        # Set by Trainer.predict(); when False the batches are still run (hooks
        # such as a prediction writer still fire) but nothing is accumulated.
        self.return_predictions: bool = True
        self._predictions: list[list[Any]] = []
        self._module_mode = _ModuleMode()

    @property
    def predictions(self) -> list[Any]:
        """Cached predictions: flat for a single dataloader, nested for several."""
        if not self._predictions:
            return []
        return self._predictions[0] if len(self._predictions) == 1 else self._predictions

    def run(self) -> Optional[list[Any]]:
        trainer = self.trainer
        model = trainer._model
        dataloaders = trainer.predict_dataloaders

        if not dataloaders:
            return []

        # Resolve the per-dataloader batch cap from limit_predict_batches; a
        # total of zero means prediction is disabled and nothing runs at all.
        limit = getattr(trainer, "limit_predict_batches", 1.0)
        max_batches = [trainer._resolve_limit(dl, limit, "predict") for dl in dataloaders]
        if sum(max_batches) == 0:
            return None

        self._predictions = [[] for _ in dataloaders]

        # Reflect the running stage so trainer.predicting is correct inside
        # predict_step, the way the other loops do for their stages.
        trainer.state.stage = RunningStage.PREDICTING

        # eval mode for the whole run, restored on the way out.
        self._module_mode.capture(model)
        _call_module_hook(trainer, "on_predict_model_eval")
        try:
            with paddle.no_grad():
                return self._run(dataloaders, max_batches)
        finally:
            self._module_mode.restore(model)

    def _run(self, dataloaders: list, max_batches: list) -> Optional[list[Any]]:
        trainer = self.trainer

        _call_callback_hooks(trainer, "on_predict_start")
        _call_module_hook(trainer, "on_predict_start")
        # Epoch hooks bracket the whole run, not each dataloader.
        _call_callback_hooks(trainer, "on_predict_epoch_start")
        _call_module_hook(trainer, "on_predict_epoch_start")

        for dl_idx, dataloader in enumerate(dataloaders):
            self._predict_dataloader(dl_idx, dataloader, max_batches[dl_idx])

        _call_callback_hooks(trainer, "on_predict_epoch_end")
        _call_module_hook(trainer, "on_predict_epoch_end")

        results = self.predictions if self.return_predictions else None
        if not self.return_predictions:
            # Release the batches accumulated for the writer hooks.
            self._predictions = []

        _call_callback_hooks(trainer, "on_predict_end")
        _call_module_hook(trainer, "on_predict_end")
        return results

    def _predict_dataloader(self, dl_idx: int, dataloader: Any, max_batches: int) -> None:
        trainer = self.trainer
        model = trainer._model
        device = trainer._resolve_device()
        step_fn = model.predict_step
        params = inspect.signature(step_fn).parameters
        # With return_predictions=False nothing is kept, unless a writer needs the
        # whole set at epoch end — the point of the flag is to not hold every
        # batch in memory.
        keep = self.return_predictions or self._any_writer_on_epoch()

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            batch = trainer._move_to_device(batch, device)

            _call_callback_hooks(trainer, "on_predict_batch_start", batch, batch_idx, dl_idx)
            _call_module_hook(trainer, "on_predict_batch_start", batch, batch_idx, dl_idx)

            # predict_step may be written with any prefix of
            # (batch, batch_idx, dataloader_idx); pass only what it accepts.
            kwargs: dict[str, Any] = {}
            if "batch_idx" in params:
                kwargs["batch_idx"] = batch_idx
            if "dataloader_idx" in params:
                kwargs["dataloader_idx"] = dl_idx
            with trainer.profiler.profile("[PredictionLoop].predict_step"):
                pred = step_fn(batch, **kwargs)

            _call_callback_hooks(trainer, "on_predict_batch_end", pred, batch, batch_idx, dl_idx)
            _call_module_hook(trainer, "on_predict_batch_end", pred, batch, batch_idx, dl_idx)

            if keep:
                self._predictions[dl_idx].append(pred)

    def _any_writer_on_epoch(self) -> bool:
        """Whether a callback consumes the predictions at epoch end."""
        return any(
            getattr(cb, "write_interval", None) == "epoch" for cb in getattr(self.trainer, "callbacks", None) or []
        )

    def teardown(self) -> None:
        self._predictions = []
