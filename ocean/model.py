"""ocean.Model - dual-mode model (Keras + Lightning).

Keras mode:
    net = paddle.nn.Sequential(...)
    model = ocean.Model(__model__=net)
    model.compile(optimizer=..., loss=..., metrics=...)
    model.fit(train_loader, val_loader, epochs=10)

Lightning mode:
    class MyModel(ocean.Model):
        def training_step(self, batch, batch_idx): ...
        def configure_optimizers(self): ...

    model = MyModel()
    trainer = ocean.Trainer(max_epochs=10)
    trainer.fit(model, train_loader)
"""

from typing import Any, Callable, Optional, Sequence, Union

import paddle
from paddle import nn


class Model(nn.Layer):
    """Dual-mode model: Keras (via __model__) or Lightning (via hooks).

    Args:
        __model__: Optional bare nn.Layer for Keras mode.
    """

    def __init__(self, __model__: Optional[nn.Layer] = None) -> None:
        super().__init__()

        # --- Keras mode members ---
        self.__model__: Optional[nn.Layer] = __model__
        self._optimizer: Optional[paddle.optimizer.Optimizer] = None
        self._loss_fns: list[Callable] = []
        self._loss_weights: Optional[list[float]] = None
        self._metrics: list[Any] = []
        self._metrics_name_cache: list[str] = []

        # --- Trainer reference ---
        self.__trainer__: Optional["Trainer"] = None  # noqa: F821
        self._trainer: Optional["Trainer"] = None  # noqa: F821

        # --- Internal state ---
        self._current_fx_name: Optional[str] = None
        self._automatic_optimization: bool = True
        self._log_metrics: dict[str, list[float]] = {}
        self._training_step_outputs: list[Any] = []
        self._validation_step_outputs: list[Any] = []
        self._example_input_array: Optional[Any] = None

    @property
    def automatic_optimization(self) -> bool:
        return self._automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, value: bool) -> None:
        self._automatic_optimization = value

    @property
    def current_epoch(self) -> int:
        return self._trainer.current_epoch if self._trainer else 0

    @property
    def dataloader_step(self) -> int:
        return self._trainer.dataloader_step if self._trainer else 0

    @property
    def optimizer_step(self) -> int:
        return self._trainer.optimizer_step if self._trainer else 0

    @property
    def example_input_array(self) -> Any:
        return self._example_input_array

    @example_input_array.setter
    def example_input_array(self, example: Any) -> None:
        self._example_input_array = example

    # ====================================================================
    # Forward
    # ====================================================================

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.__model__ is not None:
            return self.__model__(*args, **kwargs)
        return super().forward(*args, **kwargs)

    # ====================================================================
    # Keras-mode compile
    # ====================================================================

    def compile(
        self,
        optimizer: paddle.optimizer.Optimizer,
        loss: Optional[Union[Callable, list[Callable]]] = None,
        metrics: Optional[Sequence[Any]] = None,
        loss_weights: Optional[list[float]] = None,
    ) -> None:
        if self.__model__ is None:
            raise ValueError("compile() requires __model__. Use ocean.Model(__model__=your_network) for Keras mode.")
        self._optimizer = optimizer
        self._loss_fns = [loss] if callable(loss) else (list(loss) if loss is not None else [])
        self._loss_weights = loss_weights
        self._metrics = list(metrics) if metrics is not None else []
        self._metrics_name_cache = ["loss"] if self._loss_fns else []
        for m in self._metrics:
            name = m.name() if hasattr(m, "name") else m.__class__.__name__
            self._metrics_name_cache.append(name)

    # ====================================================================
    # Lightning hooks (override in subclass)
    # ====================================================================

    def training_step(self, batch: Any, batch_idx: int) -> Union[paddle.Tensor, dict[str, Any], None]:
        if self.__model__ is not None:
            return self._keras_training_step(batch, batch_idx)
        raise NotImplementedError("training_step must be implemented")

    def validation_step(self, batch: Any, batch_idx: int) -> Any: ...
    def test_step(self, batch: Any, batch_idx: int) -> Any: ...

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self.__model__ is not None:
            return self.__model__(batch)
        if isinstance(batch, (list, tuple)):
            return self(*batch) if len(batch) > 1 else self(batch[0])
        return self(batch)

    def configure_optimizers(self) -> Any:
        raise NotImplementedError("configure_optimizers must be implemented")

    # ====================================================================
    # Lifecycle hooks
    # ====================================================================

    def on_fit_start(self) -> None: ...
    def on_fit_end(self) -> None: ...
    def on_train_start(self) -> None: ...
    def on_train_end(self) -> None: ...
    def on_validation_start(self) -> None: ...
    def on_validation_end(self) -> None: ...
    def on_test_start(self) -> None: ...
    def on_test_end(self) -> None: ...
    def on_predict_start(self) -> None: ...
    def on_predict_end(self) -> None: ...
    def on_train_epoch_start(self) -> None: ...
    def on_train_epoch_end(self) -> None: ...
    def on_validation_epoch_start(self) -> None: ...
    def on_validation_epoch_end(self) -> None: ...
    def on_test_epoch_start(self) -> None: ...
    def on_test_epoch_end(self) -> None: ...
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]: ...
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None: ...
    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None: ...
    def on_validation_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None: ...
    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None: ...
    def on_test_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None: ...
    def on_before_backward(self, loss: paddle.Tensor) -> None: ...
    def on_after_backward(self) -> None: ...
    def on_before_optimizer_step(self, optimizer: paddle.optimizer.Optimizer) -> None: ...
    def on_validation_model_eval(self) -> None:
        self.eval()

    def on_validation_model_train(self) -> None:
        self.train()

    def on_test_model_eval(self) -> None:
        self.eval()

    def on_test_model_train(self) -> None:
        self.train()

    # ====================================================================
    # Logging
    # ====================================================================

    def log(
        self,
        name: str,
        value: Union[float, paddle.Tensor],
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: str = "mean",
        batch_size: Optional[int] = None,
    ) -> None:
        trainer = self._trainer
        if trainer is None:
            return
        trainer._log_metric(self, name, value, prog_bar, logger, on_step, on_epoch, reduce_fx, batch_size)

    # ====================================================================
    # Checkpoint save/load
    # ====================================================================

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load state dict (alias for set_state_dict with PyTorch-compatible API).

        Args:
            state_dict: Dictionary mapping parameter names to tensors.
            strict: If True, keys must match exactly.
        """
        if strict:
            self.set_state_dict(state_dict)
        else:
            self.set_dict(state_dict)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint to path.

        Args:
            path: File path to save to.
        """
        state = {"state_dict": self.state_dict()}
        if self._optimizer is not None:
            state["optimizer"] = self._optimizer.state_dict()
        state["epoch"] = self.current_epoch
        state["dataloader_step"] = self.dataloader_step
        state["optimizer_step"] = self.optimizer_step
        paddle.save(state, path)

    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:
        """Load model checkpoint from path.

        Args:
            path: File path to load from.
            strict: Whether to strictly enforce that keys match.
            load_optimizer: If True, also load optimizer state.

        Returns:
            The full checkpoint dictionary.
        """
        checkpoint = paddle.load(path)
        if strict:
            self.set_state_dict(checkpoint["state_dict"])
        else:
            self.set_dict(checkpoint["state_dict"])
        if load_optimizer and "optimizer" in checkpoint and self._optimizer is not None:
            self._optimizer.set_state_dict(checkpoint["optimizer"])
        # Restore training state
        return checkpoint

    # ====================================================================
    # Keras-mode convenience methods
    # ====================================================================

    def fit(
        self,
        train_data: Optional[Any] = None,
        val_data: Optional[Any] = None,
        batch_size: int = 1,
        epochs: int = 1,
        datamodule: Optional[Any] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        trainer = self.__trainer__
        if trainer is None:
            from ocean.trainer import Trainer

            trainer = Trainer(max_epochs=epochs)
            self.__trainer__ = trainer
        trainer.fit(
            self, train_dataloaders=train_data, val_dataloaders=val_data, datamodule=datamodule, ckpt_path=ckpt_path
        )

    def evaluate(self, eval_data: Optional[Any] = None, datamodule: Optional[Any] = None) -> list[dict[str, float]]:
        from ocean.trainer import Trainer

        trainer = self.__trainer__ or Trainer()
        return trainer.validate(self, dataloaders=eval_data, datamodule=datamodule)

    def predict(self, test_data: Optional[Any] = None, datamodule: Optional[Any] = None) -> list[Any]:
        from ocean.trainer import Trainer

        trainer = self.__trainer__ or Trainer()
        return trainer.predict(self, dataloaders=test_data, datamodule=datamodule)

    # ====================================================================
    # Internal: Keras training step
    # ====================================================================

    def _keras_training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
            labels = batch[1] if len(batch) >= 2 else None
        else:
            inputs, labels = batch, None

        outputs = self.__model__(inputs)

        if self._loss_fns:
            loss_values = []
            for i, loss_fn in enumerate(self._loss_fns):
                loss_val = loss_fn(outputs, labels) if labels is not None else loss_fn(outputs)
                if self._loss_weights and i < len(self._loss_weights):
                    loss_val = loss_val * self._loss_weights[i]
                loss_values.append(loss_val)

            # Fix 3: weighted aggregation (not hardcoded add_n)
            total_loss = sum(loss_values)
            # Fix 2: log all losses
            for i, lv in enumerate(loss_values):
                self.log(f"loss_{i}", lv.item(), prog_bar=(i == 0))
            self._update_metrics(outputs, labels)
            return {"loss": total_loss}
        return {"loss": paddle.to_tensor(0.0)}

    def _update_metrics(self, outputs: paddle.Tensor, labels: Optional[paddle.Tensor]) -> None:
        for metric in self._metrics:
            if hasattr(metric, "update"):
                metric.update(outputs, labels)

    def _compute_metrics(self) -> dict[str, float]:
        results = {}
        for i, metric in enumerate(self._metrics):
            if hasattr(metric, "accumulate"):
                val = metric.accumulate()
                results[self._metrics_name_cache[i + 1 if self._loss_fns else i]] = (
                    float(val.item()) if hasattr(val, "item") else float(val)
                )
            if hasattr(metric, "reset"):
                metric.reset()
        return results
