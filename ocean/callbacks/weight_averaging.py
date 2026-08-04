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
        self._avg_state: Optional[dict[str, Any]] = None
        self._n_averaged = 0

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if trainer.current_epoch < self.start_epoch:
            return
        if self._avg_state is None:
            self._avg_state = {k: v.clone() for k, v in model.state_dict().items()}
            self._n_averaged = 1
            return

        self._n_averaged += 1
        parameter_names = {name for name, _ in model.named_parameters()}
        with paddle.no_grad():
            for name, value in model.state_dict().items():
                if name not in self._avg_state:
                    continue
                if name not in parameter_names or not paddle.is_floating_point(value):
                    # Buffers (batch-norm statistics, counters) track the latest
                    # weights rather than being blended.
                    self._avg_state[name] = value.clone()
                    continue
                averaged = self._avg_state[name]
                if self.avg_type == "ema":
                    averaged.set_value(self.decay * averaged + (1 - self.decay) * value)
                else:
                    n = self._n_averaged
                    averaged.set_value(averaged * (n - 1) / n + value * (1 / n))

    def on_train_end(self, trainer: Any, model: Any) -> None:
        if self._avg_state is not None:
            model.set_state_dict(self._avg_state)

    def state_dict(self) -> dict:
        return {"n_averaged": self._n_averaged, "avg_model_state": self._avg_state}

    def load_state_dict(self, state_dict: dict) -> None:
        self._n_averaged = state_dict.get("n_averaged", 0)
        self._avg_state = state_dict.get("avg_model_state")
