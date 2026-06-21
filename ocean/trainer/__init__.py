"""ocean.Trainer - complete training loop engine with connectors, loops, callbacks, loggers."""

from collections import defaultdict
from typing import Any, Optional, Union

import paddle

from ocean.strategies import SingleDeviceStrategy
from ocean.trainer.call import (
    _call_and_handle_interrupt,
    _call_callback_hooks,
    _call_configure_model,
    _call_module_hook,
    _call_setup_hook,
)
from ocean.trainer.connectors import (
    _AcceleratorConnector,
    _CallbackConnector,
    _CheckpointConnector,
    _DataConnector,
    _LoggerConnector,
    _SignalConnector,
)
from ocean.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus


class Trainer:
    """Complete training engine with all ocean components.

    Args:
        max_epochs: Stop after this many epochs. Default: 1000.
        min_epochs: Force at least this many epochs.
        max_steps: Stop after this many global steps. Default: -1.
        accelerator: Device type ('cpu', 'gpu', 'auto').
        strategy: Strategy ('auto', 'single_device', 'ddp').
        devices: Which devices to use.
        precision: '32', '16-mixed', 'bf16-mixed'.
        log_every_n_steps: Log metrics every N training steps.
        limit_train_batches: Fraction (float) or number (int) of train batches.
        limit_val_batches: Fraction (float) or number (int) of val batches.
        limit_test_batches: Fraction (float) or number (int) of test batches.
        limit_predict_batches: Fraction (float) or number (int) of predict batches.
        val_check_interval: How often to validate.
        check_val_every_n_epoch: Validate every N epochs.
        accumulate_grad_batches: Accumulate gradients over N batches.
        gradient_clip_val: Max gradient norm for clipping.
        gradient_clip_algorithm: Clipping algorithm ('norm' or 'value').
        num_sanity_val_steps: Validation steps before training starts.
        fast_dev_run: Run N batches for debugging.
        max_time: Maximum training time. String ("HH:MM:SS", "DD:HH:MM:SS"),
            timedelta, dict, or seconds. Stops training when exceeded.
        overfit_batches: Overfit N batches (int) or fraction (float).
        deterministic: If True, use deterministic algorithms for reproducibility.
            Can be "warn" to warn on non-deterministic ops. Default: None.
        benchmark: If True, enable cudnn benchmark for faster training.
            Incompatible with deterministic=True. Default: None.
        callbacks: List of callbacks.
        logger: Logger or list of loggers, or True/False.
        enable_checkpointing: Enable automatic checkpointing.
        enable_progress_bar: Enable progress bar.
        default_root_dir: Default directory for logs and checkpoints.
        verbose: Console output verbosity (0=silent, 1=normal).
        inference_mode: Use inference mode for eval steps.
        use_distributed_sampler: Auto-add distributed sampler.
        profiler: Profiler to use.
        reload_dataloaders_every_n_epochs: Reload dataloaders every N epochs.
        detect_anomaly: Detect NaN/Inf in backward.
        barebones: Minimal mode (no logging/checkpointing).
    """

    def __init__(
        self,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        accelerator: str = "auto",
        strategy: str = "auto",
        devices: Any = "auto",
        num_nodes: int = 1,
        precision: str = "32",
        log_every_n_steps: int = 50,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        limit_predict_batches: Union[int, float] = 1.0,
        val_check_interval: Union[int, float] = 1.0,
        check_val_every_n_epoch: Optional[int] = 1,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: str = "norm",
        num_sanity_val_steps: Optional[int] = None,
        fast_dev_run: Union[int, bool] = False,
        max_time: Any = None,
        overfit_batches: Union[int, float] = 0.0,
        deterministic: Any = None,
        benchmark: Any = None,
        callbacks: Optional[list] = None,
        logger: Any = None,
        enable_checkpointing: bool = True,
        enable_progress_bar: bool = True,
        enable_model_summary: bool = True,
        default_root_dir: Optional[str] = None,
        verbose: int = 1,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Optional[Any] = None,
        reload_dataloaders_every_n_epochs: int = 0,
        detect_anomaly: bool = False,
        barebones: bool = False,
        sync_batchnorm: bool = False,
        plugins: Optional[Any] = None,
        enable_autolog_hparams: bool = True,
    ) -> None:
        # === Init with defaults ===
        if max_epochs is None:
            max_epochs = 1000 if max_steps <= 0 else 1000
        if num_sanity_val_steps is None:
            num_sanity_val_steps = 2

        self.max_epochs = max_epochs
        self.min_epochs = min_epochs or 0
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.accelerator_flag = accelerator
        self.strategy_flag = strategy
        self.devices = devices
        self.num_nodes = num_nodes
        self.precision_flag = precision
        self.log_every_n_steps = log_every_n_steps
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.limit_test_batches = limit_test_batches
        self.limit_predict_batches = limit_predict_batches
        self.val_check_interval = val_check_interval
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.num_sanity_val_steps = num_sanity_val_steps
        self.fast_dev_run = fast_dev_run
        self.overfit_batches = overfit_batches
        self.enable_model_summary = enable_model_summary
        self.sync_batchnorm = sync_batchnorm
        self.plugins = plugins
        self.enable_autolog_hparams = enable_autolog_hparams
        self.verbose = verbose
        self.inference_mode = inference_mode
        self.use_distributed_sampler = use_distributed_sampler
        self.profiler = profiler
        self.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
        self.detect_anomaly = detect_anomaly
        self.default_root_dir = default_root_dir or "."

        if barebones:
            enable_checkpointing = False
            enable_progress_bar = False
            logger = False

        # === State ===
        self.state = TrainerState()
        self.current_epoch: int = 0
        self._dataloader_step: int = 0
        self._optimizer_step: int = 0
        self._optimizers: list = []
        self._optimizer: Any = None  # kept for backward compat
        self.should_stop: bool = False

        # === Model & Data ===
        self._model: Any = None
        self.datamodule: Any = None
        self.train_dataloader: Any = None
        self.val_dataloaders: list = []
        self.test_dataloaders: list = []
        self.predict_dataloaders: list = []

        # === Debug flags ===
        from ocean.trainer.setup import _init_debugging_flags

        _init_debugging_flags(
            self,
            limit_train_batches,
            limit_val_batches,
            limit_test_batches,
            limit_predict_batches,
            overfit_batches,
            val_check_interval,
            fast_dev_run,
            accumulate_grad_batches,
            detect_anomaly,
        )

        # === Connectors ===
        self._accelerator_connector = _AcceleratorConnector(
            self, accelerator, strategy, devices, precision, deterministic, benchmark
        )
        self._data_connector = _DataConnector(self)
        self._logger_connector = _LoggerConnector(self)
        self._callback_connector = _CallbackConnector(self)
        self._checkpoint_connector = _CheckpointConnector(self)
        self._signal_connector = _SignalConnector(self)

        # === Init connectors ===
        self._data_connector.on_trainer_init(
            val_check_interval, reload_dataloaders_every_n_epochs, check_val_every_n_epoch
        )
        self._logger_connector.on_trainer_init(logger, log_every_n_steps)
        self._callback_connector.on_trainer_init(
            callbacks, enable_checkpointing, enable_progress_bar, self.default_root_dir, max_time
        )

        # === Loops ===
        from ocean.loops.evaluation_loop import _EvaluationLoop
        from ocean.loops.fit_loop import _FitLoop
        from ocean.loops.prediction_loop import _PredictionLoop

        self.fit_loop = _FitLoop(self, min_epochs=self.min_epochs, max_epochs=self.max_epochs)
        self.validate_loop = _EvaluationLoop(self, TrainerFn.VALIDATING, RunningStage.VALIDATING, verbose=verbose)
        self.test_loop = _EvaluationLoop(self, TrainerFn.TESTING, RunningStage.TESTING, verbose=verbose)
        self.predict_loop = _PredictionLoop(self)

        # === Strategy ===
        self.strategy = self._accelerator_connector.strategy

        # === Metrics ===
        self._log_metrics_buffer: dict[str, list[float]] = defaultdict(list)
        self._log_metrics_on_epoch: dict[str, float] = {}

    # ====================================================================
    # Properties
    # ====================================================================

    @property
    def training(self) -> bool:
        return self.state.stage == RunningStage.TRAINING if self.state.stage else False

    @training.setter
    def training(self, val: bool) -> None:
        self.state.stage = RunningStage.TRAINING if val else None

    @property
    def validating(self) -> bool:
        return self.state.stage == RunningStage.VALIDATING if self.state.stage else False

    @validating.setter
    def validating(self, val: bool) -> None:
        self.state.stage = RunningStage.VALIDATING if val else None

    @property
    def testing(self) -> bool:
        return self.state.stage == RunningStage.TESTING if self.state.stage else False

    @testing.setter
    def testing(self, val: bool) -> None:
        self.state.stage = RunningStage.TESTING if val else None

    @property
    def predicting(self) -> bool:
        return self.state.stage == RunningStage.PREDICTING if self.state.stage else False

    @predicting.setter
    def predicting(self, val: bool) -> None:
        self.state.stage = RunningStage.PREDICTING if val else None

    @property
    def loggers(self) -> list:
        return getattr(self, "_loggers", [])

    @loggers.setter
    def loggers(self, val: list) -> None:
        self._loggers = val

    @property
    def callbacks(self) -> list:
        return self._callbacks

    @callbacks.setter
    def callbacks(self, val: list) -> None:
        self._callbacks = val or []

    @property
    def callback_metrics(self) -> dict[str, float]:
        return self._logger_connector.callback_metrics

    @property
    def logged_metrics(self) -> dict[str, float]:
        return self._logger_connector.logged_metrics

    @property
    def progress_bar_metrics(self) -> dict[str, float]:
        return self._logger_connector.progress_bar_metrics

    @property
    def dataloader_step(self) -> int:
        return self._dataloader_step

    @property
    def optimizer_step(self) -> int:
        return self._optimizer_step

    @property
    def is_global_zero(self) -> bool:
        return True

    # ====================================================================
    # Fit
    # ====================================================================

    def fit(
        self,
        model: Any,
        train_dataloaders: Optional[Any] = None,
        val_dataloaders: Optional[Any] = None,
        datamodule: Optional[Any] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        self.state.fn = TrainerFn.FITTING
        self.state.status = TrainerStatus.RUNNING
        _call_and_handle_interrupt(
            self, self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
        )
        self.state.status = TrainerStatus.FINISHED

    def _fit_impl(
        self,
        model: Any,
        train_dataloaders: Optional[Any] = None,
        val_dataloaders: Optional[Any] = None,
        datamodule: Optional[Any] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        self._model = model
        model._trainer = self

        # Verify compile compatibility with strategy
        from ocean.utilities.compile import _verify_strategy_supports_compile

        _verify_strategy_supports_compile(model, self.strategy)

        # Fast dev run
        if self.fast_dev_run:
            n = self.fast_dev_run if isinstance(self.fast_dev_run, int) else 1
            self.limit_train_batches = n
            self.limit_val_batches = n
            self.num_sanity_val_steps = 0

        # Attach data
        self._data_connector.attach_data(model, train_dataloaders, val_dataloaders, datamodule=datamodule)

        # Strategy setup
        self.strategy.connect(model)
        self.strategy.setup_environment()

        # Setup hooks
        _call_setup_hook(self)
        _call_configure_model(self)

        # Device
        device = self._resolve_device()
        model.to(device)

        # Optimizer & Strategy setup
        self._optimizers = self._resolve_optimizers(model)
        if self._optimizers:
            self.strategy._optimizers = [o._optimizer for o in self._optimizers]
            # Set up auto-increment for optimizer steps
            for o in self._optimizers:
                o._on_after_step = lambda: self._advance_optimizer_step()
            self._optimizer = self._optimizers[0]._optimizer  # backward compat

        # Checkpoint restore
        if ckpt_path is not None:
            self._checkpoint_connector.restore(ckpt_path)

        # Attach model callbacks
        self._callback_connector._attach_model_callbacks()

        # Fit start hooks
        _call_module_hook(self, "on_fit_start")
        _call_callback_hooks(self, "on_fit_start")

        # Sanity check
        if self.val_dataloaders and self.num_sanity_val_steps > 0:
            self._sanity_check(model, device)

        # Run training
        self.fit_loop.run()

        # Fit end hooks
        _call_module_hook(self, "on_fit_end")
        _call_callback_hooks(self, "on_fit_end")
        self._teardown()

    def _advance_optimizer_step(self) -> None:
        self._optimizer_step += 1

    # ====================================================================
    # Validate / Test / Predict
    # ====================================================================

    def validate(
        self,
        model: Optional[Any] = None,
        dataloaders: Optional[Any] = None,
        datamodule: Optional[Any] = None,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        if model is not None:
            self._model = model
            model._trainer = self

        if datamodule is not None:
            datamodule.trainer = self
            datamodule.prepare_data()
            datamodule.setup("validate")
            dataloaders = [datamodule.val_dataloader()]

        self.val_dataloaders = [dataloaders] if not isinstance(dataloaders, (list, tuple)) else dataloaders
        self.state.fn = TrainerFn.VALIDATING

        device = self._resolve_device()
        self._model.to(device)

        self._log_metrics_buffer.clear()
        self._log_metrics_on_epoch.clear()
        self.validate_loop.verbose = verbose
        return self.validate_loop.run()

    def test(
        self,
        model: Optional[Any] = None,
        dataloaders: Optional[Any] = None,
        datamodule: Optional[Any] = None,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        if model is not None:
            self._model = model
            model._trainer = self

        if datamodule is not None:
            datamodule.trainer = self
            datamodule.prepare_data()
            datamodule.setup("test")
            dataloaders = [datamodule.test_dataloader()]

        self.test_dataloaders = [dataloaders] if not isinstance(dataloaders, (list, tuple)) else dataloaders
        self.state.fn = TrainerFn.TESTING

        device = self._resolve_device()
        self._model.to(device)

        self._log_metrics_buffer.clear()
        self._log_metrics_on_epoch.clear()
        return self.test_loop.run()

    def predict(
        self,
        model: Optional[Any] = None,
        dataloaders: Optional[Any] = None,
        datamodule: Optional[Any] = None,
        return_predictions: bool = True,
    ) -> Optional[list[Any]]:
        if model is not None:
            self._model = model
            model._trainer = self

        if datamodule is not None:
            datamodule.trainer = self
            datamodule.prepare_data()
            datamodule.setup("predict")
            dataloaders = [datamodule.predict_dataloader()]

        self.predict_dataloaders = [dataloaders] if not isinstance(dataloaders, (list, tuple)) else dataloaders
        self.state.fn = TrainerFn.PREDICTING

        device = self._resolve_device()
        self._model.to(device)

        self.predict_loop.return_predictions = return_predictions
        return self.predict_loop.run()

    # ====================================================================
    # Save / Load checkpoint
    # ====================================================================

    def save_checkpoint(self, filepath: str, weights_only: bool = False) -> None:
        """Save a checkpoint to filepath."""
        checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
        self.strategy.save_checkpoint(checkpoint, filepath)

    # ====================================================================
    # Logging
    # ====================================================================

    def _log_metric(
        self,
        model: Any,
        name: str,
        value: Any,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Any = None,
        on_epoch: Any = None,
        reduce_fx: str = "mean",
        batch_size: Optional[int] = None,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        rank_zero_only: bool = False,
        metric_attribute: Optional[str] = None,
    ) -> None:
        if hasattr(value, "item"):
            value = value.item()

        # Handle sync_dist: reduce across processes
        if sync_dist and hasattr(self.strategy, "reduce"):
            try:
                import paddle

                value = self.strategy.reduce(paddle.to_tensor(value), reduce_op="mean", group=sync_dist_group)
                if hasattr(value, "item"):
                    value = value.item()
            except Exception as e:
                import warnings

                warnings.warn(f"sync_dist reduce failed: {e}")

        # Handle rank_zero_only: skip logging on non-zero ranks
        if rank_zero_only and not self.is_global_zero:
            return

        # Respect on_step/on_epoch flags (ocean-compatible)
        step_log = on_step if on_step is not None else True
        epoch_log = on_epoch if on_epoch is not None else True

        if epoch_log:
            self._log_metrics_buffer[name].append(value)
        if step_log:
            self._logger_connector.log_metric_value(name, float(value), prog_bar=prog_bar)

    def _compute_epoch_metrics(self) -> None:
        """Reduce buffered metrics into epoch-level values."""
        for name, values in self._log_metrics_buffer.items():
            if values:
                self._log_metrics_on_epoch[name] = float(sum(values)) / len(values)
        self._log_metrics_buffer.clear()

    # ====================================================================
    # Internal utilities
    # ====================================================================

    def _resolve_device(self) -> Any:
        """Resolve device consistent with _AcceleratorConnector auto-detection.

        ocean-compatible: auto mode checks CUDA availability.
        """
        if self.accelerator_flag == "auto":
            if paddle.is_compiled_with_cuda():
                return paddle.CUDAPlace(0)
            return paddle.CPUPlace()
        if self.accelerator_flag == "cpu":
            return paddle.CPUPlace()
        if self.accelerator_flag == "gpu":
            if paddle.is_compiled_with_cuda():
                return paddle.CUDAPlace(0)
            raise RuntimeError("CUDA not available")
        return paddle.CPUPlace()

    def _resolve_optimizers(self, model: Any) -> list:
        """Configure all optimizers from the model, wrapped in OceanOptimizer."""
        from ocean.core.optimizer import OceanOptimizer, init_optimizers_and_lr_schedulers

        if model._optimizer is not None:
            return [OceanOptimizer(model._optimizer)]
        opts, _ = init_optimizers_and_lr_schedulers(model)
        return [OceanOptimizer(opt) for opt in opts]

    def _move_to_device(self, batch: Any, device: Any) -> Any:
        if isinstance(batch, paddle.Tensor):
            return batch.to(device)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(b, device) for b in batch)
        if isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        return batch

    def _resolve_limit(self, loader: Any, limit: Union[int, float]) -> int:
        try:
            total = len(loader)
        except (TypeError, AttributeError):
            return int(limit) if isinstance(limit, (int, float)) else 100
        if isinstance(limit, int):
            return min(limit, total)
        return min(int(total * limit), total)

    def _print(self, msg: str) -> None:
        """Print a message to console if verbose."""
        if self.verbose > 0:
            print(msg)

    def _should_check_val(self) -> bool:
        if self.check_val_every_n_epoch is None:
            return False
        return self.current_epoch % self.check_val_every_n_epoch == 0

    def _should_check_val_step(self, step: int) -> bool:
        """Check if validation should run at this training step (ocean-compatible).

        Only triggered when val_check_interval is an int (every N steps).
        """
        val_interval = self.val_check_interval
        if not isinstance(val_interval, int) or val_interval <= 0:
            return False
        return step % val_interval == 0

    def _should_stop(self) -> bool:
        return self.should_stop or (0 < self.max_steps <= self._dataloader_step)

    def _should_limit_batches(self, batch_idx: int, mode: str) -> bool:
        limit = getattr(self, f"limit_{mode}_batches", 1.0)
        if isinstance(limit, int):
            return batch_idx >= limit
        return False

    def _sanity_check(self, model: Any, device: Any) -> None:
        """Run sanity check — aligned with Lightning behavior.

        - Calls ``on_sanity_check_start/end`` (NOT ``on_validation_start/end``).
        - Resets logged metrics before/after so ``val/loss`` does not leak
          into training step-0 log flushes.
        - Does NOT call ``empty_cache()`` (Lightning doesn't either; any
          memory issue is a real leak that must be fixed at the root).
        """
        from ocean.trainer.call import _call_callback_hooks

        # Reset metrics before sanity check (Lightning-style)
        self._logger_connector.reset_validation_metrics()

        self.sanity_checking = True
        model.eval()
        try:
            _call_callback_hooks(self, "on_sanity_check_start")
            for dataloader in self.val_dataloaders:
                count = 0
                with paddle.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        if count >= self.num_sanity_val_steps:
                            break
                        batch = self._move_to_device(batch, device)
                        _call_callback_hooks(self, "on_validation_batch_start", batch, batch_idx, dataloader_idx=0)
                        model.validation_step(batch, batch_idx)
                        _call_callback_hooks(self, "on_validation_batch_end", None, batch, batch_idx, dataloader_idx=0)
                        count += 1
        finally:
            _call_callback_hooks(self, "on_sanity_check_end")
            self.sanity_checking = False
            model.train()  # restore train mode after sanity check
            # Reset metrics again after sanity check (Lightning-style)
            # so val/* metrics don't pollute training step-0 log flushes
            self._logger_connector.reset_validation_metrics()

    def _teardown(self) -> None:
        self.strategy.teardown()
        self._signal_connector.teardown()
        self.fit_loop.teardown()
        if self.datamodule is not None:
            self.datamodule.teardown("fit")
