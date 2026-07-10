"""Trainer connectors - data/logger/callback/checkpoint/signal/accelerator."""

# Each connector is in its own section below for organizational clarity.
# They are imported into trainer.py.

from __future__ import annotations

from typing import Any, Optional

import paddle

import ocean
from ocean.callbacks.callback import Callback
from ocean.callbacks.checkpoint import ModelCheckpoint
from ocean.callbacks.progress import ProgressBar

# _LoggerConnector now lives in the logger_connector/ subpackage alongside the
# _ResultCollection it delegates metric storage to. Re-exported here so existing
# ``from ocean.trainer.connectors import _LoggerConnector`` imports keep working.
from ocean.trainer.connectors.logger_connector import _LoggerConnector
from ocean.utils import MisconfigurationException

__all__ = [
    "_DataConnector",
    "_LoggerConnector",
    "_CallbackConnector",
    "_CheckpointConnector",
    "_SignalConnector",
    "_AcceleratorConnector",
]

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
        # Validate at construction time so misconfiguration fails fast, rather than
        # silently mis-scheduling (or skipping) validation once fit() runs.
        if check_val_every_n_epoch is not None and not isinstance(check_val_every_n_epoch, int):
            raise MisconfigurationException(
                f"`check_val_every_n_epoch` should be an integer, found {check_val_every_n_epoch!r}."
            )

        if check_val_every_n_epoch is None and isinstance(val_check_interval, float):
            raise MisconfigurationException(
                "`val_check_interval` should be an integer or a time-based duration "
                "(str 'DD:HH:MM:SS', datetime.timedelta, or dict kwargs for timedelta) "
                "when `check_val_every_n_epoch=None`."
            )

        if not isinstance(reload_dataloaders_every_n_epochs, int) or reload_dataloaders_every_n_epochs < 0:
            raise MisconfigurationException(
                f"`reload_dataloaders_every_n_epochs` should be an int >= 0, "
                f"got {reload_dataloaders_every_n_epochs!r}."
            )

        self.trainer.val_check_interval = val_check_interval
        self.trainer.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
        self.trainer.check_val_every_n_epoch = check_val_every_n_epoch

    def prepare_data(self) -> None:
        """Run datamodule.prepare_data() under rank gating + a barrier.

        Downloads/preprocessing must happen on a single process to avoid races
        on shared storage. ``prepare_data_per_node`` (default True) runs it once
        per node (local rank 0); when False it runs only on global rank 0. Other
        ranks wait at the barrier until preparation completes.
        """
        datamodule = self.trainer.datamodule
        if datamodule is None:
            return

        strategy = getattr(self.trainer, "strategy", None)
        local_rank = getattr(strategy, "local_rank", 0) if strategy is not None else 0
        node_rank = getattr(strategy, "node_rank", 0) if strategy is not None else 0
        local_rank_zero = local_rank == 0
        global_rank_zero = local_rank == 0 and node_rank == 0

        per_node = getattr(datamodule, "prepare_data_per_node", True)
        should_prepare = local_rank_zero if per_node else global_rank_zero

        if should_prepare:
            datamodule.prepare_data()

        # Ranks that skipped preparation wait here until it finishes.
        if strategy is not None and hasattr(strategy, "barrier"):
            strategy.barrier("prepare_data")

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
            # prepare_data (download/preprocess, rank-gated) must precede setup.
            self.prepare_data()
            datamodule.setup("fit")
            self.trainer.train_dataloader = train_dataloaders or datamodule.train_dataloader()
            self.trainer.val_dataloaders = val_dataloaders or [datamodule.val_dataloader()]
            # Keep test/predict channels defined for later test()/predict() calls,
            # symmetric with the explicit-dataloaders branch below.
            self.trainer.test_dataloaders = [test_dataloaders] if test_dataloaders is not None else []
            self.trainer.predict_dataloaders = [predict_dataloaders] if predict_dataloaders is not None else []
        else:
            self.trainer.train_dataloader = train_dataloaders
            self.trainer.val_dataloaders = [val_dataloaders] if val_dataloaders is not None else []
            self.trainer.test_dataloaders = [test_dataloaders] if test_dataloaders is not None else []
            self.trainer.predict_dataloaders = [predict_dataloaders] if predict_dataloaders is not None else []


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
        if enable_progress_bar and not any(isinstance(cb, ProgressBar) for cb in callbacks):
            from ocean.callbacks.progress import TQDMProgressBar

            callbacks.append(TQDMProgressBar())
        if max_time is not None and not any(cb.__class__.__name__ == "Timer" for cb in callbacks):
            from ocean.callbacks.timer import Timer

            callbacks.append(Timer(duration=max_time))
        self.trainer.callbacks = self._reorder_callbacks(callbacks)

    def _attach_model_callbacks(self) -> None:
        """Attach callbacks from the model's configure_callbacks() hook.

        A model callback whose type matches (or subclasses) a trainer callback
        replaces it, so a model-provided ModelCheckpoint doesn't coexist with the
        default one. Checkpoint callbacks are then reordered to run last.
        """
        model = self.trainer._model
        if not hasattr(model, "configure_callbacks"):
            return
        model_callbacks = model.configure_callbacks()
        if not model_callbacks:
            return
        if not isinstance(model_callbacks, (list, tuple)):
            model_callbacks = [model_callbacks]

        model_types = {type(c) for c in model_callbacks}
        # A trainer callback is overridden if a model callback is the same type
        # or a subclass of it.
        override_types = set()
        for trainer_cb in self.trainer.callbacks:
            t_type = type(trainer_cb)
            if t_type is Callback:
                continue
            if any(issubclass(m_type, t_type) for m_type in model_types):
                override_types.add(t_type)

        kept = [c for c in self.trainer.callbacks if type(c) not in override_types]
        kept.extend(model_callbacks)
        self.trainer.callbacks = self._reorder_callbacks(kept)

    @staticmethod
    def _reorder_callbacks(callbacks: list) -> list:
        """Order callbacks so tuner callbacks run first and checkpoint callbacks last.

        Checkpoint callbacks must run after callbacks that update metrics (so a
        monitored value is current when a checkpoint is written). Relative order
        within each group is preserved.
        """
        from ocean.callbacks.batch_size_finder import BatchSizeFinder
        from ocean.callbacks.lr_finder import LRFinder
        from ocean.callbacks.on_exception_checkpoint import OnExceptionCheckpoint

        tuner, other, checkpoint = [], [], []
        for cb in callbacks:
            if isinstance(cb, (BatchSizeFinder, LRFinder)):
                tuner.append(cb)
            elif isinstance(cb, (ModelCheckpoint, OnExceptionCheckpoint)):
                checkpoint.append(cb)
            else:
                other.append(cb)
        return tuner + other + checkpoint


# ====================================================================
# Checkpoint Connector
# ====================================================================


class _CheckpointConnector:
    """Manages checkpoint loading/resuming."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self._loaded_checkpoint: Optional[dict] = None

    def restore(self, checkpoint_path: str, weights_only: Optional[bool] = None) -> None:
        """Load checkpoint and restore model/optimizer/loop state."""
        ckpt = paddle.load(checkpoint_path)
        self._loaded_checkpoint = ckpt
        model = self.trainer._model

        if "state_dict" in ckpt:
            model.set_state_dict(ckpt["state_dict"])

        if not weights_only:
            if "optimizer_states" not in ckpt:
                raise KeyError("Trying to restore optimizer state but checkpoint contains only the model.")
            opt = self.trainer.optimizers[0]._optimizer
            if opt is not None and ckpt["optimizer_states"]:
                opt.set_state_dict(ckpt["optimizer_states"][0])

            if "lr_schedulers" not in ckpt:
                raise KeyError(
                    "Trying to restore learning rate scheduler state but checkpoint contains only the model."
                )
            for config, state in zip(self.trainer._lr_schedulers, ckpt["lr_schedulers"]):
                config["scheduler"].set_state_dict(state)

            if "callbacks" in ckpt:
                for cb in self.trainer.callbacks:
                    name = type(cb).__qualname__
                    if name in ckpt["callbacks"] and hasattr(cb, "load_state_dict"):
                        cb.load_state_dict(ckpt["callbacks"][name])

            if self.trainer.strategy is not None:
                pp = self.trainer.strategy.precision_plugin
                pname = type(pp).__qualname__
                if pname in ckpt and hasattr(pp, "load_state_dict"):
                    pp.load_state_dict(ckpt[pname])

            # Restore datamodule state saved under its qualified name (dump side).
            datamodule = self.trainer.datamodule
            if datamodule is not None and hasattr(datamodule, "load_state_dict"):
                dm_key = type(datamodule).__qualname__
                if dm_key in ckpt:
                    datamodule.load_state_dict(ckpt[dm_key])

        if "epoch" in ckpt:
            restored_epoch = ckpt["epoch"]
            # Guard against resuming past the configured training budget.
            max_epochs = getattr(self.trainer, "max_epochs", None)
            if max_epochs is not None and max_epochs != -1 and restored_epoch > max_epochs:
                raise MisconfigurationException(
                    f"You restored a checkpoint with current_epoch={restored_epoch}, but you have "
                    f"set Trainer(max_epochs={max_epochs}). Increase max_epochs to continue training."
                )
            self.trainer.current_epoch = restored_epoch
        if "dataloader_step" in ckpt:
            self.trainer._dataloader_step = ckpt["dataloader_step"]
        if "optimizer_step" in ckpt:
            self.trainer._optimizer_step = ckpt["optimizer_step"]

        if "loops" in ckpt:
            self.trainer.fit_loop.load_state_dict(ckpt["loops"])

        # Restore hparams and any custom state added via on_save_checkpoint(),
        # symmetric with dump_checkpoint().
        if "hparams" in ckpt and hasattr(model, "hparams"):
            model.hparams = ckpt["hparams"]
        if hasattr(model, "on_load_checkpoint"):
            model.on_load_checkpoint(ckpt)

    def dump_checkpoint(self, weights_only: bool = False) -> dict:
        """Build a complete checkpoint dictionary."""
        model = self.trainer._model
        checkpoint = {
            "ocean_version": ocean.__version__,
            "epoch": self.trainer.current_epoch,
            "dataloader_step": self.trainer.dataloader_step,
            "optimizer_step": self.trainer.optimizer_step,
            "state_dict": model.state_dict(),
        }
        if not weights_only and self.trainer.optimizers:
            raw_opt = self.trainer.optimizers[0]._optimizer
            if raw_opt is not None:
                checkpoint["optimizer_states"] = [raw_opt.state_dict()]

        loop_state = self.trainer.fit_loop.state_dict()
        if loop_state:
            checkpoint["loops"] = loop_state

        checkpoint["lr_schedulers"] = [cfg["scheduler"].state_dict() for cfg in self.trainer._lr_schedulers]

        if hasattr(model, "on_save_checkpoint"):
            checkpoint.update(model.on_save_checkpoint())

        if self.trainer.strategy is not None:
            pp = self.trainer.strategy.precision_plugin
            if pp is not None and hasattr(pp, "state_dict"):
                ps = pp.state_dict()
                if ps:
                    checkpoint[type(pp).__qualname__] = ps

        if self.trainer.datamodule is not None:
            if hasattr(self.trainer.datamodule, "state_dict"):
                ds = self.trainer.datamodule.state_dict()
                if ds:
                    checkpoint[type(self.trainer.datamodule).__qualname__] = ds

        if hasattr(model, "hparams") and model.hparams:
            checkpoint["hparams"] = model.hparams

        callback_states = {}
        for cb in self.trainer.callbacks:
            if hasattr(cb, "state_dict"):
                state = cb.state_dict()
                if state:
                    callback_states[type(cb).__qualname__] = state
        if callback_states:
            checkpoint["callbacks"] = callback_states

        return checkpoint


# ====================================================================
# Signal Connector
# ====================================================================


class _SignalConnector:
    """Installs a SIGTERM handler for graceful shutdown.

    On SIGTERM (e.g. cluster preemption or ``kill``) the trainer is asked to stop
    at the next loop boundary — ``should_stop`` is honored by the fit loop — so an
    end-of-epoch checkpoint callback can still run. Original handlers are restored
    on teardown.
    """

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self.received_sigterm = False
        self._original_handlers: dict = {}

    def register_signal_handlers(self) -> None:
        import signal
        import threading

        self.received_sigterm = False
        self._original_handlers = {}
        # Signal handlers can only be installed from the main thread.
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            self._original_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._sigterm_handler)
        except (ValueError, OSError):
            # e.g. unsupported platform / not permitted; leave defaults in place.
            self._original_handlers.pop(signal.SIGTERM, None)

    def _sigterm_handler(self, signum: Any, frame: Any) -> None:
        self.received_sigterm = True
        # Request a graceful stop; the fit loop checks should_stop each epoch.
        self.trainer.should_stop = True

    def teardown(self) -> None:
        import signal

        for signum, handler in self._original_handlers.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError, TypeError):
                pass
        self._original_handlers = {}


# ====================================================================
# Accelerator Connector
# ====================================================================


class _AcceleratorConnector:
    """Resolves accelerator, strategy, devices, and precision.

    Strategy resolution logic::
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
        # Choose strategy based on device count
        self._strategy = self._resolve_strategy(strategy, self._parallel_devices)
        # Inject accelerator & devices into strategy
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
        """Choose strategy based on device count.

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
            # Multi-device → DDP
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
            except Exception as e:
                import warnings

                warnings.warn(
                    f"fleet initialization failed: {e}. Distributed collectives may not work; "
                    "ensure the process was launched with a distributed launcher."
                )
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
        """Set deterministic / benchmark mode flags.

        PaddlePaddle may not support cudnn flags via ``set_flags`` on all builds,
        so each call is guarded; the deterministic workspace config is also set
        via an environment variable.
        """
        import os

        if deterministic is True or deterministic == "warn":
            if benchmark is None:
                benchmark = False
            elif benchmark:
                import warnings

                warnings.warn("deterministic=True is incompatible with benchmark=True")

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
