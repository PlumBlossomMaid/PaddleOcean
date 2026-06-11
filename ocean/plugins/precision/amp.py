"""Mixed precision (AMP) plugin using PaddlePaddle's native AMP API."""

from typing import Any

import paddle

from ocean.plugins.precision.precision import Precision


class MixedPrecision(Precision):
    """Mixed precision training using PaddlePaddle AMP.

    Supports O1 (automatic mixed precision with white list) and O2 (half precision).
    """

    def __init__(self, precision: str = "16-mixed") -> None:
        super().__init__(precision)
        self._scaler = paddle.amp.GradScaler(init_loss_scaling=2.0**15)
        self._level = "O1" if precision.startswith("16") else "O2"

    def forward_context(self) -> Any:
        dtype = "float16" if self.precision.startswith("16") else "bfloat16"
        return paddle.amp.auto_cast(level=self._level, dtype=dtype)

    def backward(self, tensor: paddle.Tensor, model: Any, *args: Any, **kwargs: Any) -> None:
        scaled = self._scaler.scale(tensor)
        scaled.backward(*args, **kwargs)

    def optimizer_step(self, optimizer: paddle.optimizer.Optimizer, **kwargs: Any) -> Any:
        self._scaler.step(optimizer)
        self._scaler.update()

    def unscale_gradients(self, optimizer: paddle.optimizer.Optimizer) -> None:
        self._scaler.unscale_(optimizer)

    def state_dict(self) -> dict:
        return {"scaler": self._scaler.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        if "scaler" in state_dict:
            self._scaler.load_state_dict(state_dict["scaler"])
