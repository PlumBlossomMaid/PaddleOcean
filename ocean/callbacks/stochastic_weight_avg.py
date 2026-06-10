"""StochasticWeightAveraging callback - averages model weights for better generalization."""

from typing import Any, Optional

import paddle

from ocean.callbacks.callback import Callback


class StochasticWeightAveraging(Callback):
    """Averaging model weights with SWA for improved generalization.

    Args:
        swa_lrs: Learning rate(s) to use during SWA phase.
        swa_epoch_start: Epoch to start SWA (float=fraction, int=epoch number).
        annealing_epochs: Number of epochs for annealing.
        annealing_strategy: 'cos' or 'linear'.
    """

    def __init__(
        self,
        swa_lrs: float = 1e-3,
        swa_epoch_start: float = 0.8,
        annealing_epochs: int = 10,
        annealing_strategy: str = "cos",
    ) -> None:
        self.swa_lrs = swa_lrs
        self.swa_epoch_start = swa_epoch_start
        self.annealing_epochs = annealing_epochs
        self.annealing_strategy = annealing_strategy
        self._average_model: Optional[Any] = None
        self._n_averaged = 0

    def on_fit_start(self, trainer: Any, model: Any) -> None:
        self._n_averaged = 0

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        num_epochs = trainer.max_epochs if trainer.max_epochs else 1
        start_epoch = int(self.swa_epoch_start * num_epochs) if isinstance(self.swa_epoch_start, float) else self.swa_epoch_start

        if trainer.current_epoch >= start_epoch:
            if self._average_model is None:
                self._average_model = self._copy_model(model)
                self._n_averaged = 1
            else:
                self._update_average(model)
                self._n_averaged += 1

    def on_train_end(self, trainer: Any, model: Any) -> None:
        if self._average_model is not None and self._n_averaged > 0:
            model.set_state_dict(self._average_model.state_dict())

    def _copy_model(self, model: Any) -> Any:
        import copy
        return copy.deepcopy(model)

    def _update_average(self, model: Any) -> None:
        if self._average_model is None:
            return
        n = self._n_averaged + 1
        with paddle.no_grad():
            for avg_param, param in zip(self._average_model.parameters(), model.parameters()):
                avg_param.set_value(avg_param * (n - 1) / n + param * (1 / n))
