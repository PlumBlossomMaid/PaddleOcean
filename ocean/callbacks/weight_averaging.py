"""WeightAveraging callback - averages model weights (simpler than SWA)."""

from typing import Any, Optional

import paddle

from ocean.callbacks.callback import Callback


class WeightAveraging(Callback):
    """Averaging model weights for improved performance.

    Simpler alternative to StochasticWeightAveraging. Applies exponential
    moving average (EMA) or simple averaging during training.

    Args:
        avg_type: 'ema' for exponential moving average, 'simple' for simple average.
        decay: EMA decay rate (only used if avg_type='ema').
        start_epoch: Epoch to start averaging.
    """

    def __init__(self, avg_type: str = "ema", decay: float = 0.995, start_epoch: int = 1) -> None:
        self.avg_type = avg_type
        self.decay = decay
        self.start_epoch = start_epoch
        self._avg_model: Optional[Any] = None
        self._n_averaged = 0

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if trainer.current_epoch < self.start_epoch:
            return
        if self._avg_model is None:
            self._avg_model = self._copy_model(model)
            self._n_averaged = 1
            return

        self._n_averaged += 1
        with paddle.no_grad():
            for avg_param, param in zip(self._avg_model.parameters(), model.parameters()):
                if self.avg_type == "ema":
                    avg_param.set_value(self.decay * avg_param + (1 - self.decay) * param)
                else:
                    n = self._n_averaged
                    avg_param.set_value(avg_param * (n - 1) / n + param * (1 / n))

    def on_train_end(self, trainer: Any, model: Any) -> None:
        if self._avg_model is not None:
            model.set_state_dict(self._avg_model.state_dict())

    def _copy_model(self, model: Any) -> Any:
        import copy
        return copy.deepcopy(model)

    def state_dict(self) -> dict:
        return {
            "n_averaged": self._n_averaged,
            "avg_model_state": self._avg_model.state_dict() if self._avg_model else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._n_averaged = state_dict.get("n_averaged", 0)
        avg_state = state_dict.get("avg_model_state")
        if avg_state is not None and self._avg_model is not None:
            self._avg_model.set_state_dict(avg_state)
