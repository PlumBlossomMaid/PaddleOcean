"""_PredictionLoop - runs prediction across all dataloaders."""

from typing import Any

from ocean.loops.loop import _Loop
from ocean.trainer.call import _call_callback_hooks


class _PredictionLoop(_Loop):
    """Runs prediction across all dataloaders."""

    def run(self) -> list[Any]:
        trainer = self.trainer
        model = trainer._model
        dataloaders = trainer.predict_dataloaders

        if not dataloaders:
            return [], []

        _call_callback_hooks(trainer, "on_predict_start")

        predictions = []
        for dl_idx, dataloader in enumerate(dataloaders):
            _call_callback_hooks(trainer, "on_predict_epoch_start")
            for batch_idx, batch in enumerate(dataloader):
                _call_callback_hooks(trainer, "on_predict_batch_start", batch, batch_idx, dl_idx)
                batch = trainer._move_to_device(batch, trainer._resolve_device())
                with trainer.profiler.profile("[PredictionLoop].predict_step"):
                    pred = model.predict_step(batch)
                predictions.append(pred)
                _call_callback_hooks(trainer, "on_predict_batch_end", pred, batch, batch_idx, dl_idx)
            _call_callback_hooks(trainer, "on_predict_epoch_end")

        _call_callback_hooks(trainer, "on_predict_end")
        return predictions
