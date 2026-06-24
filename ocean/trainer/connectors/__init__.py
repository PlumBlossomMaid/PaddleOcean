"""Trainer connectors - data/logger/callback/checkpoint/signal/accelerator."""

# Each connector is in its own section below for organizational clarity.
# They are imported into trainer.py.

from __future__ import annotations

from typing import Any, Optional

import paddle

from ocean.callbacks.callback import Callback
from ocean.callbacks.checkpoint import ModelCheckpoint
from ocean.loggers.base import Logger
from ocean.loggers.csv_logs import CSVLogger

# ====================================================================
# Data Connector
# ====================================================================


class _DataConnector:
    """Manages data sources: dataloaders and DataModules."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def on_trainer_init(
        self,
        val_check_interval: Any,
        reload_dataloaders_every_n_epochs: int,
        check_val_every_n_epoch: Optional[int],
    ) -> None:
        self.trainer.val_check_interval = val_check_interval
        self.trainer.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
        self.trainer.check_val_every_n_epoch = check_val_every_n_epoch

    def prepare_data(self) -> None:
        if self.trainer.datamodule is not None:
            self.trainer.datamodule.prepare_data()

    def attach_data(
        self,
        model: Any,
        train_dataloaders: Optional[Any] = None,
        val_dataloaders: Optional[Any] = None,
        test_dataloaders: Optional[Any] = None,
        predict_dataloaders: Optional[Any] = None,
        datamodule: Optional[Any] = None,
    ) -> None:
        self.trainer.datamodule = datamodule
        if datamodule is not None:
            datamodule.trainer = self.trainer
            datamodule.setup("fit")
            self.trainer.train_dataloader = train_dataloaders or datamodule.train_dataloader()
            self.trainer.val_dataloaders = val_dataloaders or [datamodule.val_dataloader()]
        else:
            self.trainer.train_dataloader = train_dataloaders
            self.trainer.val_dataloaders = [val_dataloaders] if val_dataloaders is not None else []
            self.trainer.test_dataloaders = [test_dataloaders] if test_dataloaders is not None else []
            self.trainer.predict_dataloaders = [predict_dataloaders] if predict_dataloaders is not None else []


# ====================================================================
# Logger Connector
# ====================================================================


class _LoggerConnector:
    """Manages logging: metric tracking, result collection, logger dispatch."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self._callback_metrics: dict[str, float] = {}
        self._logged_metrics: dict[str, float] = {}
        self._progress_bar_metrics: dict[str, float] = {}
        self._metrics_buffer: dict[str, list[float]] = {}
        self._reset_all_metrics()

    def on_trainer_init(self, logger: Any, log_every_n_steps: int) -> None:
        self.configure_logger(logger)
        self.trainer.log_every_n_steps = log_every_n_steps

    def configure_logger(self, logger: Any) -> None:
        """Configure logger from bool/None/Logger/list."""
        if logger is False:
            self.trainer.loggers = []
        elif logger is True or logger is None:
            self.trainer.loggers = [CSVLogger(root_dir=self.trainer.default_root_dir or ".")]
        elif isinstance(logger, Logger):
            self.trainer.loggers = [logger]
        elif isinstance(logger, (list, tuple)):
            self.trainer.loggers = list(logger)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics — delegates to each logger (which filter by rank internally)."""
        for lg in getattr(self.trainer, "loggers", None) or []:
            if hasattr(lg, "log_metrics"):
                lg.log_metrics(metrics, step)

    def reset_validation_metrics(self) -> None:
        """Clear all val/test metrics after validation.

        Clears both ``_logged_metrics`` and ``_metrics_buffer`` so stale
        validation values (e.g. ``loss/val`` which doesn't start with
        ``val/``) do not leak into training step-0 log flushes.
        """
        self._logged_metrics.clear()
        self._metrics_buffer.clear()

    def _reset_all_metrics(self) -> None:
        self._callback_metrics = {}
        self._logged_metrics = {}
        self._progress_bar_metrics = {}

    @property
    def callback_metrics(self) -> dict[str, float]:
        return self._callback_metrics

    @property
    def logged_metrics(self) -> dict[str, float]:
        return self._logged_metrics

    @property
    def progress_bar_metrics(self) -> dict[str, float]:
        return self._progress_bar_metrics

    def update_train_step_metrics(self) -> None:
        """Aggregate step-level metrics."""
        pass

    def update_train_epoch_metrics(self) -> None:
        """Aggregate epoch-level metrics."""
        for name, values in self._metrics_buffer.items():
            if values:
                mean_val = float(sum(values)) / len(values)
                self._callback_metrics[name] = mean_val
                self._logged_metrics[name] = mean_val
        self._metrics_buffer.clear()

    def log_metric_value(self, name: str, value: float, prog_bar: bool = False) -> None:
        self._callback_metrics[name] = value
        self._logged_metrics[name] = value
        if prog_bar:
            self._progress_bar_metrics[name] = value
        if name not in self._metrics_buffer:
            self._metrics_buffer[name] = []
        self._metrics_buffer[name].append(value)

    def teardown(self) -> None:
        pass


# ====================================================================
# Callback Connector
# ====================================================================


class _CallbackConnector:
    """Manages callbacks: attaches default callbacks, manages callback state."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def on_trainer_init(
        self,
        callbacks: Optional[list],
        enable_checkpointing: bool,
        enable_progress_bar: bool,
        default_root_dir: Optional[str],
        max_time: Any = None,
    ) -> None:
        callbacks = callbacks or []
        if enable_checkpointing and not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
            callbacks.append(ModelCheckpoint(dirpath=default_root_dir or "."))
        if enable_progress_bar and not any(cb.__class__.__name__ == "TQDMProgressBar" for cb in callbacks):
            from ocean.callbacks.progress import TQDMProgressBar

            callbacks.append(TQDMProgressBar())
        if max_time is not None and not any(cb.__class__.__name__ == "Timer" for cb in callbacks):
            from ocean.callbacks.timer import Timer

            callbacks.append(Timer(duration=max_time))
        self.trainer.callbacks = callbacks

    def _attach_model_callbacks(self) -> None:
        """Attach callbacks from configure_callbacks() hook."""
        model = self.trainer._model
        if hasattr(model, "configure_callbacks"):
            extra = model.configure_callbacks()
            if extra:
                self.trainer.callbacks.extend(extra)


# ====================================================================
# Checkpoint Connector
# ====================================================================


class _CheckpointConnector:
    """Manages checkpoint loading/resuming."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self._loaded_checkpoint: Optional[dict] = None

    def restore(self, checkpoint_path: str, weights_only: Optional[bool] = None) -> None:
        """Load checkpoint and restore model/optimizer state."""
        ckpt = paddle.load(checkpoint_path)
        self._loaded_checkpoint = ckpt
        model = self.trainer._model

        if "state_dict" in ckpt:
            model.set_state_dict(ckpt["state_dict"])

        if "optimizer_states" in ckpt and not weights_only:
            opt = self.trainer._optimizer
            if opt is not None and ckpt["optimizer_states"]:
                opt.set_state_dict(ckpt["optimizer_states"][0])

        if "epoch" in ckpt:
            self.trainer.current_epoch = ckpt["epoch"]
        if "dataloader_step" in ckpt:
            self.trainer._dataloader_step = ckpt["dataloader_step"]
        if "optimizer_step" in ckpt:
            self.trainer._optimizer_step = ckpt["optimizer_step"]

    def dump_checkpoint(self, weights_only: bool = False) -> dict:
        """Build a complete checkpoint dictionary."""
        model = self.trainer._model
        checkpoint = {
            "epoch": self.trainer.current_epoch,
            "dataloader_step": self.trainer.dataloader_step,
            "optimizer_step": self.trainer.optimizer_step,
            "state_dict": model.state_dict(),
        }
        if not weights_only and self.trainer._optimizer is not None:
            checkpoint["optimizer_states"] = [self.trainer._optimizer.state_dict()]
        return checkpoint


# ====================================================================
# Signal Connector
# ====================================================================


class _SignalConnector:
    """Manages signal handlers for graceful shutdown."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self.received_sigterm = False

    def register_signal_handlers(self) -> None:
        pass

    def teardown(self) -> None:
        pass


# ====================================================================
# Accelerator Connector
# ====================================================================


class _AcceleratorConnector:
    """Resolves accelerator, strategy, devices, and precision.

    Matches Lightning's pattern::
        - Resolve accelerator first (auto-detect available hardware).
        - Parse devices via ``accelerator.parse_devices(devices)``.
        - Build parallel device list via ``accelerator.get_parallel_devices()``.
        - Choose strategy based on device count:
            ``len(parallel_devices) > 1 → DDPStrategy``
            ``len == 1 → SingleDeviceStrategy``
        - Inject accelerator and parallel_devices into the strategy.

    Auto-detects:
    - strategy="auto" → DDP for multi-device, SingleDevice for single
    - accelerator="auto" → CUDA > ROCm > XPU > CustomDevice > CPU
    """

    def __init__(
        self,
        trainer: Any,
        accelerator: str,
        strategy: str,
        devices: Any,
        precision: str,
        deterministic: Any = None,
        benchmark: Any = None,
    ) -> None:
        self.trainer = trainer
        self._accelerator = self._resolve_accelerator(accelerator)
        # Parse devices and build parallel device list
        self._devices_flag = self._accelerator.parse_devices(devices)
        self._parallel_devices = self._accelerator.get_parallel_devices(devices)
        # Choose strategy based on device count (Lightning pattern)
        self._strategy = self._resolve_strategy(strategy, self._parallel_devices)
        # Inject accelerator & devices into strategy (Lightning's _lazy_init_strategy)
        self._strategy.accelerator = self._accelerator
        self._strategy.parallel_devices = self._parallel_devices
        # Resolve precision
        self._precision = self._resolve_precision(precision)
        self._set_flags(deterministic, benchmark)

    @property
    def strategy(self) -> Any:
        return self._strategy

    @staticmethod
    def _resolve_accelerator(accelerator: str) -> Any:
        from ocean.accelerators import (
            CPUAccelerator,
            CUDAAccelerator,
            CustomDeviceAccelerator,
            IPUAccelerator,
            ROCmAccelerator,
            XPUAccelerator,
        )

        if accelerator == "cpu":
            return CPUAccelerator()
        if accelerator == "auto":
            if CUDAAccelerator.is_available():
                return CUDAAccelerator()
            if ROCmAccelerator.is_available():
                return ROCmAccelerator()
            if XPUAccelerator.is_available():
                return XPUAccelerator()
            if CustomDeviceAccelerator.is_available():
                return CustomDeviceAccelerator()
            return CPUAccelerator()
        if accelerator == "gpu":
            if CUDAAccelerator.is_available():
                return CUDAAccelerator()
            raise RuntimeError("GPU requested but CUDA not available")
        if accelerator == "rocm":
            return ROCmAccelerator()
        if accelerator == "xpu":
            return XPUAccelerator()
        if accelerator == "ipu":
            return IPUAccelerator()
        if accelerator == "custom":
            if CustomDeviceAccelerator.is_available():
                return CustomDeviceAccelerator()
            raise RuntimeError("custom accelerator requested but no custom device available")
        raise ValueError(f"Unknown accelerator: {accelerator}")

    @staticmethod
    def _resolve_strategy(strategy: str, parallel_devices: list[Any]) -> Any:
        """Choose strategy based on device count, matching Lightning's pattern.

        Resolution order::
            strategy="auto":
                multi-device  → ``DDPStrategy``
                single-device → ``SingleDeviceStrategy``
            strategy in ("ddp", "ddp_spawn"): ``DDPStrategy``
            strategy="fleet":  ``DDPStrategy`` + fleet init
            strategy="single_device": ``SingleDeviceStrategy``
        """
        from ocean.strategies import SingleDeviceStrategy

        if strategy == "auto":
            # Multi-device → DDP (Lightning pattern)
            if len(parallel_devices) > 1:
                from ocean.strategies.ddp import DDPStrategy

                return DDPStrategy()
            # Already initialized distributed → DDP (external launcher)
            try:
                import paddle.distributed as dist

                if dist.is_initialized():
                    from ocean.strategies.ddp import DDPStrategy

                    return DDPStrategy()
            except Exception:
                pass
            return SingleDeviceStrategy()

        if strategy in ("ddp", "ddp_spawn"):
            from ocean.strategies.ddp import DDPStrategy

            return DDPStrategy()

        if strategy == "fleet":
            from ocean.strategies.ddp import DDPStrategy

            ds = DDPStrategy()
            try:
                import ocean.distributed as odist

                odist.fleet.init(is_collective=True)
            except Exception:
                pass
            return ds

        if strategy == "single_device":
            return SingleDeviceStrategy()

        # If it's already a Strategy instance, use it directly
        from ocean.strategies import Strategy

        if isinstance(strategy, Strategy):
            return strategy

        return SingleDeviceStrategy()

    @staticmethod
    def _resolve_precision(precision: str) -> Any:
        from ocean.plugins import MixedPrecision, Precision

        if precision in ("16", "16-mixed", "bf16", "bf16-mixed"):
            return MixedPrecision(precision)
        if precision == "16-true":
            from ocean.plugins.precision import HalfPrecision

            return HalfPrecision()
        if precision in ("64", "64-true"):
            from ocean.plugins.precision import DoublePrecision

            return DoublePrecision()
        return Precision(precision)

    @staticmethod
    def _set_flags(deterministic: Any = None, benchmark: Any = None) -> None:
        """Set flags for deterministic/benchmark modes.

        Mirrors paddleOcean's _set_torch_flags.
        PaddlePaddle may not support cudnn flags via set_flags on all builds,
        so we use try/except and fall back to environment variables.
        """
        import os

        if deterministic is True or deterministic == "warn":
            if benchmark is None:
                benchmark = False
            elif benchmark:
                import warnings

                warnings.warn("deterministic=True is incompatible with benchmark=True")

            # Set PaddlePaddle deterministic flags
            try:
                paddle.set_flags({"FLAGS_cudnn_deterministic": True})
            except ValueError:
                pass
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        if benchmark is not None:
            try:
                paddle.set_flags({"FLAGS_cudnn_benchmark": benchmark})
            except ValueError:
                pass

        if deterministic is True:
            try:
                paddle.set_flags({"FLAGS_cudnn_deterministic": True})
            except ValueError:
                pass
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif deterministic == "warn":
            try:
                paddle.set_flags({"FLAGS_cudnn_deterministic": True})
            except ValueError:
                pass
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
