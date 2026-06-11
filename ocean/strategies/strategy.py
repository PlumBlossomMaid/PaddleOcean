"""Strategy base class - manages model, optimizer, device placement, and distributed execution."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import paddle

from ocean.accelerators.accelerator import Accelerator
from ocean.plugins.precision import PrecisionPlugin
from ocean.plugins.precision.precision import Precision


class Strategy(ABC):
    """Base strategy class."""

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ) -> None:
        self._accelerator = accelerator
        self._precision_plugin = precision_plugin or Precision()
        self._model: Optional[paddle.nn.Layer] = None
        self._lightning_module: Optional[Any] = None
        self._optimizers: list = []

    @property
    def accelerator(self) -> Optional[Accelerator]:
        return self._accelerator

    @accelerator.setter
    def accelerator(self, acc: Accelerator) -> None:
        self._accelerator = acc

    @property
    def precision_plugin(self) -> "PrecisionPlugin":
        return self._precision_plugin

    @property
    def model(self) -> Optional[paddle.nn.Layer]:
        return self._model

    @property
    def lightning_module(self) -> Optional[Any]:
        return self._lightning_module

    @property
    def optimizers(self) -> list:
        return self._optimizers

    @optimizers.setter
    def optimizers(self, opts: list) -> None:
        self._optimizers = opts

    @property
    @abstractmethod
    def root_device(self) -> Any: ...

    @property
    @abstractmethod
    def is_global_zero(self) -> bool: ...

    @property
    def global_rank(self) -> int: return 0
    @property
    def local_rank(self) -> int: return 0
    @property
    def world_size(self) -> int: return 1

    def connect(self, model: Any) -> None:
        self._lightning_module = model
        self._model = model

    def setup_environment(self) -> None:
        if self._accelerator:
            self._accelerator.setup_device(self.root_device)

    def setup(self, trainer: Any) -> None:
        if self._accelerator:
            self._accelerator.setup(trainer)
        self._precision_plugin.convert_module(self._model)
        self.model_to_device()
        self.setup_optimizers(trainer)

    def model_to_device(self) -> None:
        if self._model is not None:
            self._model.to(self.root_device)

    def setup_optimizers(self, trainer: Any) -> None:
        from ocean.core.optimizer import init_optimizers_and_lr_schedulers
        opts, _ = init_optimizers_and_lr_schedulers(trainer._model)
        self._optimizers = opts

    def backward(self, closure_loss: Any, *args: Any, **kwargs: Any) -> None:
        self._precision_plugin.pre_backward(closure_loss, self._model)
        self._precision_plugin.backward(closure_loss, self._model, *args, **kwargs)
        self._precision_plugin.post_backward(closure_loss, self._model)

    def optimizer_step(self, optimizer: Any, **kwargs: Any) -> Any:
        return self._precision_plugin.optimizer_step(optimizer, **kwargs)

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        with self._precision_plugin.forward_context():
            return self._lightning_module.training_step(*args, **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        with self._precision_plugin.forward_context():
            return self._lightning_module.validation_step(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        with self._precision_plugin.forward_context():
            return self._lightning_module.test_step(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        with self._precision_plugin.forward_context():
            return self._lightning_module.predict_step(*args, **kwargs)

    def reduce(self, tensor: Any, reduce_op: str = "mean") -> Any:
        return tensor

    def barrier(self, name: Optional[str] = None) -> None: pass
    def broadcast(self, obj: Any, src: int = 0) -> Any: return obj
    def all_gather(self, tensor: Any, group: Any = None, sync_grads: bool = False) -> Any: return tensor
    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool: return decision

    def save_checkpoint(self, checkpoint: dict, filepath: str) -> None:
        if self.is_global_zero:
            paddle.save(checkpoint, filepath)

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        return paddle.load(checkpoint_path)

    def load_model_state_dict(self, checkpoint: dict, strict: bool = True) -> None:
        if "state_dict" in checkpoint and self._model is not None:
            if strict:
                self._model.set_state_dict(checkpoint["state_dict"])
            else:
                self._model.set_dict(checkpoint["state_dict"])

    def teardown(self) -> None:
        if self._model is not None:
            self._model.to(paddle.CPUPlace())
        self._precision_plugin.teardown()
