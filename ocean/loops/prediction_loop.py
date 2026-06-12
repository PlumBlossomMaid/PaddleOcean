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
        for dataloader in dataloaders:
            _call_callback_hooks(trainer, "on_predict_epoch_start")
            for batch in dataloader:
                _call_callback_hooks(trainer, "on_predict_batch_start", batch)
                batch = trainer._move_to_device(batch, trainer._resolve_device())
                pred = model.predict_step(batch)
                predictions.append(pred)
                _call_callback_hooks(trainer, "on_predict_batch_end", batch, pred)
            _call_callback_hooks(trainer, "on_predict_epoch_end")

        _call_callback_hooks(trainer, "on_predict_end")
        return predictions
