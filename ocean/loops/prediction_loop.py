"""_PredictionLoop - runs prediction across all dataloaders."""

from typing import Any, Optional

import paddle

from ocean.loops.loop import _Loop
from ocean.loops.fetchers import _DataFetcher
from ocean.trainer.call import _call_callback_hooks, _call_lightning_module_hook


class _PredictionLoop(_Loop):
    """Runs prediction across all dataloaders."""

    def __init__(self, trainer: Any) -> None:
        super().__init__(trainer)
        self._data_fetcher = _DataFetcher()
        self._predictions: list[Any] = []
        self.return_predictions: bool = True

    @property
    def skip(self) -> bool:
        return False

    def run(self) -> Optional[list[Any]]:
        trainer = self.trainer
        model = trainer._model
        dataloaders = getattr(trainer, "predict_dataloaders", None) or []

        if not dataloaders:
            return []

        model.eval()
        _call_lightning_module_hook(trainer, "on_predict_start")
        _call_callback_hooks(trainer, "on_predict_start")

        predictions = []
        with paddle.no_grad():
            for dl_idx, dataloader in enumerate(dataloaders):
                for batch_idx, batch in enumerate(dataloader):
                    device = trainer._resolve_device()
                    batch = trainer._move_to_device(batch, device)
                    result = model.predict_step(batch, batch_idx, dl_idx)
                    predictions.append(result)

        _call_lightning_module_hook(trainer, "on_predict_end")
        _call_callback_hooks(trainer, "on_predict_end")
        model.train()

        self._predictions = predictions
        return predictions if self.return_predictions else None

    def teardown(self) -> None:
        self._data_fetcher.teardown()
