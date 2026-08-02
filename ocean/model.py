"""ocean.Model - dual-mode model (Keras + Ocean).

Keras mode:
    net = paddle.nn.Sequential(...)
    model = ocean.Model(net)
    model.prepare(optimizer=..., loss=..., metrics=...)
    model.fit(train_loader, val_loader, epochs=10)

Ocean mode:
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
    """Dual-mode model: Keras (via model) or Ocean (via hooks).

    Args:
        model: Optional bare nn.Layer for Keras mode.
    """

    def __init__(self, model: Optional[nn.Layer] = None) -> None:
        super().__init__()

        # --- Keras mode members ---
        self.__model__: Optional[nn.Layer] = model
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
        self._compiler_ctx: Optional[dict] = None
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
    def global_step(self) -> int:
        """Number of optimizer steps performed so far.

        Counts only real optimizer steps (how many times the optimizer has been
        stepped), not raw batches seen. With ``accumulate_grad_batches > 1``
        this advances ``1`` per ``accumulate_grad_batches`` batches, staying
        aligned with learning-rate-scheduler / logging cadence. Use
        :attr:`dataloader_step` for raw batch counts.
        """
        return self._trainer.optimizer_step if self._trainer else 0

    @property
    def dataloader_step(self) -> int:
        return self._trainer.dataloader_step if self._trainer else 0

    @property
    def global_rank(self) -> int:
        """Global rank (ocean-compatible)."""
        try:
            import paddle.distributed as dist

            if dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return 0

    @property
    def local_rank(self) -> int:
        """Local rank within a node (ocean-compatible)."""
        try:
            import os

            return int(os.environ.get("PADDLE_LOCAL_RANK", 0))
        except Exception:
            return 0

    @property
    def on_gpu(self) -> bool:
        """Whether the model is on a GPU (ocean-compatible)."""
        return paddle.is_compiled_with_cuda()

    @property
    def logger(self) -> Any:
        """The first logger from the Trainer (ocean-compatible)."""
        if self._trainer and getattr(self._trainer, "loggers", None):
            return self._trainer.loggers[0]
        return None

    @property
    def loggers(self) -> list:
        """All loggers from the Trainer (ocean-compatible)."""
        if self._trainer:
            return getattr(self._trainer, "loggers", [])
        return []

    @property
    def example_input_array(self) -> Any:
        return self._example_input_array

    @example_input_array.setter
    def example_input_array(self, example: Any) -> None:
        self._example_input_array = example

    # ====================================================================
    # JIT Compile (paddle.jit.to_static)
    # ====================================================================

    def compile(self, full_graph: bool = False, input_spec=None) -> "Model":
        """Apply ``paddle.jit.to_static`` to accelerate training.

        Wraps forward and step methods with static graph compilation.
        Call before passing to ``Trainer.fit()``.

        Args:
            full_graph: If True, compile the entire graph. Set True when
                using ``input_spec`` for shape inference.
            input_spec: Optional list of ``InputSpec`` for shape/type
                annotation. Requires ``full_graph=True``.

        Returns:
            self, with compiled methods.
        """
        from ocean.utilities.compile import from_compiled

        return from_compiled(self, full_graph=full_graph, input_spec=input_spec)

    # ====================================================================
    # Forward
    # ====================================================================

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.__model__ is not None:
            return self.__model__(*args, **kwargs)
        return super().forward(*args, **kwargs)

    # ====================================================================
    # Keras-mode: prepare (equivalent to paddle.Model.prepare)
    # ====================================================================

    def prepare(
        self,
        optimizer: paddle.optimizer.Optimizer,
        loss: Optional[Union[Callable, list[Callable]]] = None,
        metrics: Optional[Sequence[Any]] = None,
        loss_weights: Optional[list[float]] = None,
    ) -> None:
        if self.__model__ is None:
            raise ValueError("prepare() requires model. Use ocean.Model(model=your_network) for Keras mode.")
        self._optimizer = optimizer
        self._loss_fns = [loss] if callable(loss) else (list(loss) if loss is not None else [])
        self._loss_weights = loss_weights
        self._metrics = list(metrics) if metrics is not None else []
        self._metrics_name_cache = ["loss"] if self._loss_fns else []
        for m in self._metrics:
            name = m.name() if hasattr(m, "name") else m.__class__.__name__
            # paddle.metric.* .name() returns a list (e.g. ['acc']); flatten to a
            # single hashable string so it can be used as a metric key later.
            if isinstance(name, (list, tuple)):
                name = "_".join(str(n) for n in name) if name else m.__class__.__name__
            self._metrics_name_cache.append(name)

    # ====================================================================
    # Model hooks (override in subclass)
    # ====================================================================

    def training_step(self, batch: Any, batch_idx: int) -> Union[paddle.Tensor, dict[str, Any], None]:
        if self.__model__ is not None:
            return self._keras_training_step(batch, batch_idx)
        raise NotImplementedError("training_step must be implemented")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # Keras mode: fall back to forward + loss so Model.evaluate works out
        # of the box. In Ocean mode the subclass is expected to override this;
        # we keep a no-op default to stay compatible with existing usage that
        # runs validation loops without a custom step.
        if self.__model__ is not None:
            return self._keras_eval_step(batch)
        return None

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self.__model__ is not None:
            return self._keras_eval_step(batch)
        return None

    def predict_step(self, batch: Any, batch_idx: int = 0) -> Any:
        # Route through self.forward so an overridden forward participates in
        # prediction (rather than calling the wrapped network directly).
        if isinstance(batch, (list, tuple)):
            return self(batch[0])
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
    def on_train_epoch_end(self) -> None:
        # Keras mode: report accumulated metrics at the end of each training
        # epoch and reset the accumulators. In Ocean mode this is a no-op so
        # user overrides remain in control (a subclass overriding this hook is
        # expected to call super().on_train_epoch_end() to keep reporting, or to
        # do its own).
        if self.__model__ is not None:
            metrics = self._compute_metrics()
            for name, value in metrics.items():
                self.log(name, value, prog_bar=True)

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

    # ── Additional ocean-aligned hooks ──

    def on_before_zero_grad(self, optimizer: paddle.optimizer.Optimizer) -> None: ...

    def on_predict_model_eval(self) -> None:
        self.eval()

    def on_predict_epoch_start(self) -> None: ...
    def on_predict_epoch_end(self) -> None: ...
    def on_predict_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None: ...
    def on_predict_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None: ...

    def backward(self, loss: paddle.Tensor, *args: Any, **kwargs: Any) -> None:
        """Override to customize backward (ocean-compatible)."""
        loss.backward(*args, **kwargs)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: paddle.optimizer.Optimizer,
        optimizer_closure: Any = None,
    ) -> None:
        """Override to customize the optimizer step.

        Called by the automatic-optimization loop in place of a raw
        ``optimizer.step()``. The default routes through ``trainer.strategy``
        so AMP/GradScaler semantics are preserved. Paddle optimizers don't
        accept a closure, so ``optimizer_closure`` is accepted for signature
        parity but unused here (the forward/backward already ran inline).
        """
        trainer = self._trainer
        if trainer is not None and trainer.strategy is not None:
            trainer.strategy.optimizer_step(optimizer)
        else:
            optimizer.step()

    def optimizer_zero_grad(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: paddle.optimizer.Optimizer,
    ) -> None:
        """Override to customize gradient zeroing (e.g. set-to-none)."""
        optimizer.clear_grad()

    # Backward-compatible aliases for the legacy hook names. They delegate to
    # the canonical names above so user overrides of either form still fire.
    def on_optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: paddle.optimizer.Optimizer,
        optimizer_closure: Any = None,
    ) -> None:
        """Override to customize optimizer step."""
        self.optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def optimizer_clear_grad(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: paddle.optimizer.Optimizer,
    ) -> None:
        self.optimizer_zero_grad(epoch, batch_idx, optimizer)

    def lr_scheduler_step(
        self,
        scheduler: Any,
        metric: Any = None,
    ) -> None:
        """Override to customize LR scheduler step (ocean-compatible).

        PaddlePaddle's ``ReduceOnPlateau`` takes the monitored metric as a
        required argument to ``step()``, unlike other schedulers which take none.
        """
        if isinstance(scheduler, paddle.optimizer.lr.ReduceOnPlateau):
            scheduler.step(metric)
        else:
            scheduler.step()

    def manual_backward(self, loss: paddle.Tensor, *args: Any, **kwargs: Any) -> None:
        """Backward in manual optimization (ocean-compatible).

        Routes through the strategy/precision plugin so AMP GradScaler scaling
        is applied, matching the reference behavior of dispatching the backward
        through the strategy.
        Falls back to a plain ``loss.backward()`` when no trainer/strategy is
        attached (e.g. unit tests calling the model directly).
        """
        trainer = self._trainer
        if trainer is not None and trainer.strategy is not None:
            trainer.strategy.backward(loss, *args, **kwargs)
        else:
            loss.backward(*args, **kwargs)

    def clip_gradients(
        self,
        optimizer: Any,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Clip gradients in manual optimization.

        Call from ``training_step`` after ``manual_backward`` and before
        ``optimizer.step()``. Under AMP the gradients are unscaled first, so the
        clip threshold applies to the true (unscaled) gradient. ``optimizer``
        may be an
        ``OceanOptimizer`` wrapper or a raw Paddle optimizer.

        Args:
            optimizer: The optimizer whose parameters' gradients are clipped.
            gradient_clip_val: Max norm/value; ``None`` or ``<=0`` is a no-op.
            gradient_clip_algorithm: ``"norm"`` (default) or ``"value"``.
        """
        if gradient_clip_val is None or gradient_clip_val <= 0:
            return
        algorithm = (gradient_clip_algorithm or "norm").lower()

        raw_opt = getattr(optimizer, "_optimizer", optimizer)
        trainer = self._trainer
        if trainer is not None and trainer.strategy is not None:
            # Unscale before clipping when AMP GradScaler is active (no-op otherwise).
            trainer.strategy.precision_plugin.unscale_gradients(raw_opt)

        if algorithm == "value":
            paddle.nn.utils.clip_grad_value_(self.parameters(), gradient_clip_val)
        else:
            paddle.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)

    def optimizers(self) -> Any:
        """Return the optimizer(s) being used during training.

        For manual optimization, returns the wrapped OceanOptimizer(s) so that
        ``trainer.optimizer_step`` advances correctly and AMP/GradScaler
        semantics are preserved.  Returns a single optimizer when only one is
        present, or a list when multiple optimizers are configured.
        """
        if self._trainer is None:
            raise RuntimeError("optimizers() called outside of training context")
        opts = self._trainer.optimizers
        if isinstance(opts, list) and len(opts) == 1:
            return opts[0]
        return opts

    def lr_schedulers(self) -> Any:
        """Return the LR scheduler(s) being used during training.

        Returns ``None`` when no schedulers are configured, a single scheduler
        when only one is present, or a list for multiple schedulers.
        """
        if self._trainer is None:
            raise RuntimeError("lr_schedulers() called outside of training context")
        configs = self._trainer._lr_schedulers
        if not configs:
            return None
        schedulers = [cfg["scheduler"] for cfg in configs]
        if len(schedulers) == 1:
            return schedulers[0]
        return schedulers

    def freeze(self) -> None:
        """Freeze all parameters."""
        for p in self.parameters():
            p.stop_gradient = True

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.stop_gradient = False

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print only on rank 0 (ocean-compatible)."""
        if self.global_rank == 0:
            import builtins

            builtins.print(*args, **kwargs)

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
        enable_graph: bool = False,
        batch_size: Optional[int] = None,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        rank_zero_only: bool = False,
        metric_attribute: Optional[str] = None,
    ) -> None:
        trainer = self._trainer
        if trainer is None:
            return
        trainer._log_metric(
            self,
            name,
            value,
            prog_bar,
            logger,
            on_step,
            on_epoch,
            reduce_fx,
            batch_size,
            sync_dist,
            sync_dist_group,
            add_dataloader_idx,
            rank_zero_only,
            metric_attribute,
        )

    def log_dict(
        self,
        dictionary: dict[str, Union[float, paddle.Tensor]],
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: str = "mean",
        enable_graph: bool = False,
        batch_size: Optional[int] = None,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        rank_zero_only: bool = False,
    ) -> None:
        """Log a dictionary of metrics at once.

        Log a dict of scalar metrics in one call.
        """
        for name, value in dictionary.items():
            self.log(
                name,
                value,
                prog_bar=prog_bar,
                logger=logger,
                on_step=on_step,
                on_epoch=on_epoch,
                reduce_fx=reduce_fx,
                enable_graph=enable_graph,
                batch_size=batch_size,
                sync_dist=sync_dist,
                sync_dist_group=sync_dist_group,
                add_dataloader_idx=add_dataloader_idx,
                rank_zero_only=rank_zero_only,
            )

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

    def on_save_checkpoint(self) -> dict[str, Any]:
        """Hook for adding custom state to checkpoint.

        Returns:
            Dictionary of custom state to include in the checkpoint.
        """
        return {}

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Hook for restoring custom state from a checkpoint.

        Receives the full checkpoint dict; the counterpart of
        :meth:`on_save_checkpoint`. Override to restore any extra state that
        was added there. The default is a no-op.
        """

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
        state["optimizer_step"] = self._trainer.optimizer_step if self._trainer else 0
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

    def _keras_eval_step(self, batch: Any) -> dict[str, Any]:
        """Evaluate one batch in Keras mode: forward + (optional) loss/metrics.

        Mirrors the training step's forward path but skips gradient work.
        Loss/metric names match :meth:`_keras_training_step` so an epoch-end
        reduction aggregates consistently across fit and evaluate.
        """
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
            labels = batch[1] if len(batch) >= 2 else None
        else:
            inputs, labels = batch, None

        outputs = self.__model__(inputs)
        result: dict[str, Any] = {}
        if self._loss_fns:
            for i, loss_fn in enumerate(self._loss_fns):
                lv = loss_fn(outputs, labels) if labels is not None else loss_fn(outputs)
                self.log(f"val_loss_{i}", lv.item())
            result["loss"] = sum(
                (loss_fn(outputs, labels) if labels is not None else loss_fn(outputs)) for loss_fn in self._loss_fns
            )
        self._update_metrics(outputs, labels)
        return result

    def _update_metrics(self, outputs: paddle.Tensor, labels: Optional[paddle.Tensor]) -> None:
        for metric in self._metrics:
            if hasattr(metric, "update"):
                metric.update(outputs, labels)

    def _compute_metrics(self) -> dict[str, float]:
        results = {}
        for i, metric in enumerate(self._metrics):
            if hasattr(metric, "accumulate"):
                val = metric.accumulate()
                key = self._metrics_name_cache[i + 1 if self._loss_fns else i]
                if not isinstance(key, str):
                    key = "_".join(str(k) for k in key) if isinstance(key, (list, tuple)) else str(key)
                results[key] = float(val.item()) if hasattr(val, "item") else float(val)
            if hasattr(metric, "reset"):
                metric.reset()
        return results
