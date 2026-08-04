"""ocean.Gear - lightweight manual training API for user-driven loops.

Gear provides manual control over training with minimal boilerplate.
Users write their own training loop while Gear handles device placement,
precision, distributed setup, and checkpointing.

Multi-GPU support::
    gear = ocean.Gear(accelerator="gpu", devices=2, strategy="ddp")
    gear.launch()  # spawns processes if needed
    model = gear.setup(model)
    # ... training loop ...
"""

from contextlib import nullcontext
from typing import Any, Optional, Union

import paddle

from ocean.accelerators.accelerator import Accelerator
from ocean.strategies import DDPStrategy, Strategy


class Gear:
    """Lightweight manual training API for user-driven training loops.

    Usage::

        # Single device
        gear = ocean.Gear(accelerator="gpu", devices=1)
        model = paddle.nn.Linear(10, 2)
        model = gear.setup(model)

        # Multi-GPU DDP
        gear = ocean.Gear(accelerator="gpu", devices=2, strategy="ddp")
        gear.launch()
        model = gear.setup(model)

    Args:
        accelerator: Device type (``'auto'``, ``'cpu'``, ``'gpu'``, ``'xpu'``).
        devices: Number of devices or device IDs (``1``, ``2``, ``"0,1"``).
        strategy: Strategy name (``'auto'``, ``'ddp'``, ``'single_device'``).
        precision: Training precision (``'32'``, ``'16-mixed'``, ``'bf16-mixed'``).
        loggers: Optional logger(s).
    """

    def __init__(
        self,
        accelerator: str = "auto",
        devices: Union[str, int, list[int]] = "auto",
        strategy: str = "auto",
        precision: str = "32",
        loggers: Optional[Union[Any, list[Any]]] = None,
    ) -> None:
        self.accelerator_flag = accelerator
        self.devices_flag = devices
        self.strategy_flag = strategy
        self.precision_flag = precision
        self.loggers = [loggers] if loggers is not None and not isinstance(loggers, (list, tuple)) else (loggers or [])
        self._models_setup = 0
        self._launched = False
        self._strategy: Optional[Strategy] = None
        self._accelerator: Optional[Accelerator] = None

        # Resolve accelerator and strategy
        self._resolve()

    @property
    def device(self) -> paddle.CPUPlace:
        strategy = self._strategy or self._resolve_fallback_strategy()
        return strategy.root_device if strategy else paddle.CPUPlace()

    @property
    def strategy(self) -> Optional[Strategy]:
        return self._strategy

    @property
    def logger(self) -> Any:
        """The first configured logger, or None."""
        return self.loggers[0] if self.loggers else None

    @property
    def global_rank(self) -> int:
        return getattr(self._strategy, "global_rank", 0) if self._strategy else 0

    @property
    def local_rank(self) -> int:
        return getattr(self._strategy, "local_rank", 0) if self._strategy else 0

    @property
    def world_size(self) -> int:
        return getattr(self._strategy, "world_size", 1) if self._strategy else 1

    @property
    def is_global_zero(self) -> bool:
        """Whether this is the main process — the one that should write."""
        return getattr(self._strategy, "is_global_zero", True) if self._strategy else True

    # ------------------------------------------------------------------
    # Resolution — accelerator/strategy selection logic
    # ------------------------------------------------------------------

    def _resolve(self) -> None:
        from ocean.trainer.connectors import _AcceleratorConnector

        self._accelerator = _AcceleratorConnector._resolve_accelerator(self.accelerator_flag)
        parallel = self._accelerator.get_parallel_devices(self.devices_flag)
        self._strategy = _AcceleratorConnector._resolve_strategy(self.strategy_flag, parallel)
        self._strategy.accelerator = self._accelerator
        self._strategy.parallel_devices = parallel
        # Same precision resolution the Trainer does. Without this the strategy
        # keeps its default full-precision plugin, so ``precision="16-mixed"``
        # would cast the forward pass and then leave the backward unscaled.
        self._strategy._precision_plugin = _AcceleratorConnector._resolve_precision(self.precision_flag)

    def _resolve_fallback_strategy(self) -> Strategy:
        """Create a minimal strategy for device access."""
        from ocean.strategies.single_device import SingleDeviceStrategy

        return SingleDeviceStrategy(device="cpu")

    # ------------------------------------------------------------------
    # Launch — multi-process entry point
    # ------------------------------------------------------------------

    def launch(self) -> None:
        """Set up the distributed environment.

        In multi-process mode, this would spawn subprocesses. Currently
        sets up the device and distributed environment in the current process
        (intended for use with ``paddle.distributed.launch`` or similar).
        """
        if self._launched:
            return
        self._launched = True

        if self._strategy:
            self._strategy.setup_environment()

    # ------------------------------------------------------------------
    # Setup model/optimizers
    # ------------------------------------------------------------------

    def setup(self, module: paddle.nn.Layer, *optimizers: paddle.optimizer.Optimizer) -> Any:
        """Set up a model (and optional optimizers) for accelerated training.

        Moves model to device and, in DDP mode, wraps it in
        ``paddle.distributed.DataParallel``.

        Args:
            module: The model to set up.
            *optimizers: Optional optimizers.

        Returns:
            Model (and optimizers) ready for training.
        """
        self._models_setup += 1

        if self._strategy:
            self._strategy._model = module
            # Only DDP-wrap on the first call
            if self._models_setup == 1 and isinstance(self._strategy, DDPStrategy):
                self._strategy.model_to_device()
                if self._strategy._is_initialized:
                    module = paddle.distributed.DataParallel(
                        module,
                        find_unused_parameters=getattr(self._strategy, "_find_unused_parameters", False),
                    )
                    self._strategy._model = module
            elif self._strategy:
                module.to(self._strategy.root_device)
        else:
            module.to(paddle.CPUPlace())

        if optimizers:
            # Wrapped so ``optimizer.step()`` goes through the precision plugin;
            # a raw Paddle step under mixed precision would update on unscaled
            # gradients and never call ``GradScaler.update()``.
            wrapped = tuple(self._setup_optimizer(opt) for opt in optimizers)
            return (module, *wrapped)
        return module

    def _setup_optimizer(self, optimizer: paddle.optimizer.Optimizer) -> Any:
        from ocean.core.optimizer import OceanOptimizer

        wrapper = OceanOptimizer(optimizer)
        if self._strategy is not None:
            wrapper._precision_plugin = self._strategy.precision_plugin
        return wrapper

    def setup_dataloaders(self, *dataloaders: paddle.io.DataLoader, move_to_device: bool = True) -> Any:
        """Set up dataloaders for training.

        Args:
            *dataloaders: DataLoaders to set up.
            move_to_device: If True, each batch is moved to the Gear's device as
                it is yielded, so the loop body never has to.

        Returns:
            DataLoader(s) ready for training.
        """
        prepared = tuple(_GearDataLoader(dl, self) if move_to_device else dl for dl in dataloaders)
        if len(prepared) == 1:
            return prepared[0]
        return prepared

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def backward(self, tensor: paddle.Tensor, *args: Any, **kwargs: Any) -> None:
        """Backward pass. Handles precision scaling if needed.

        Args:
            tensor: Loss tensor to backpropagate.
        """
        if self._strategy:
            self._strategy.backward(tensor, *args, **kwargs)
        else:
            tensor.backward(*args, **kwargs)

    def save(self, path: str, state: dict[str, Any]) -> None:
        """Save a checkpoint.

        Args:
            path: File path.
            state: Dictionary containing model/optimizer state.
        """
        if self._strategy and not getattr(self._strategy, "is_global_zero", True):
            return  # Only save on rank 0

        serializable = {}
        for k, v in state.items():
            if isinstance(v, paddle.nn.Layer):
                serializable[k] = v.state_dict()
            elif hasattr(v, "state_dict"):
                serializable[k] = v.state_dict()
            else:
                serializable[k] = v
        paddle.save(serializable, path)

    def load(self, path: str, state: Optional[dict[str, Any]] = None, strict: bool = True) -> dict[str, Any]:
        """Load a checkpoint.

        Args:
            path: File path.
            state: Optional dict mapping keys to objects to restore.
            strict: Strict state dict loading.

        Returns:
            Full checkpoint dictionary.
        """
        checkpoint = paddle.load(path)
        if state is not None:
            for k, v in state.items():
                if k not in checkpoint:
                    if strict:
                        raise KeyError(
                            f"Checkpoint {path!r} has no entry {k!r}. Available keys: {sorted(checkpoint)}. "
                            "Pass `strict=False` to skip missing entries."
                        )
                    continue
                if isinstance(v, paddle.nn.Layer):
                    # set_state_dict warns rather than raises on a mismatch, so
                    # a strict load has to check the keys itself.
                    if strict:
                        _verify_state_keys(v.state_dict(), checkpoint[k], k)
                    v.set_state_dict(checkpoint[k])
                elif hasattr(v, "set_state_dict"):
                    v.set_state_dict(checkpoint[k])
        return checkpoint

    def barrier(self, name: Optional[str] = None) -> None:
        """Barrier for distributed synchronization. No-op in single-process mode."""
        if self._strategy:
            self._strategy.barrier(name)

    def seed_everything(self, seed: int = 42, verbose: bool = True) -> int:
        """Set global random seed.

        Args:
            seed: Random seed.
            verbose: If True, prints the seed.

        Returns:
            The seed used.
        """
        import random

        import numpy as np

        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if verbose:
            print(f"Global seed set to {seed}")
        return seed

    def log(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """Send a single metric to every configured logger."""
        self.log_dict({name: value}, step=step)

    def log_dict(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Send a dict of metrics to every configured logger.

        ``Gear(loggers=...)`` used to be accepted and then never read — there was
        no way to get anything into them.
        """
        if not self.loggers:
            return
        scalars = {k: (v.item() if isinstance(v, paddle.Tensor) else v) for k, v in metrics.items()}
        for logger in self.loggers:
            logger.log_metrics(scalars, step)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print only on the main process."""
        if not self._strategy or getattr(self._strategy, "is_global_zero", True):
            print(*args, **kwargs)

    def to_device(self, obj: Any) -> Any:
        """Move a tensor/model to the Gear's device."""
        device = self.device
        if isinstance(obj, paddle.nn.Layer):
            return obj.to(device)
        if isinstance(obj, paddle.Tensor):
            return obj.to(device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.to_device(item) for item in obj)
        if isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}
        return obj

    def autocast(self) -> Any:
        """Return the forward context for the configured precision.

        Delegates to the precision plugin, so the cast used here and the scaling
        applied by ``backward()`` can never disagree about the precision.
        """
        if self._strategy is not None:
            return self._strategy.precision_plugin.forward_context()
        # No strategy: a no-op context, so grads are NOT disabled — using
        # `paddle.no_grad()` here would silently break backward under fp32.
        return nullcontext()


def _verify_state_keys(current: dict, loaded: dict, name: str) -> None:
    """Raise when a checkpoint entry does not match the object being restored."""
    missing = sorted(set(current) - set(loaded))
    unexpected = sorted(set(loaded) - set(current))
    mismatched = [k for k in set(current) & set(loaded) if tuple(current[k].shape) != tuple(loaded[k].shape)]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"State for {name!r} does not match the checkpoint. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}. Shape mismatch: {sorted(mismatched)}. "
            "Pass `strict=False` to load anyway."
        )


class _GearDataLoader:
    """Wraps a dataloader so each batch arrives on the Gear's device.

    ``setup_dataloaders(move_to_device=True)`` used to return the loader
    untouched, so the flag promised a device transfer that never happened and
    the loop body still had to do it by hand.
    """

    def __init__(self, dataloader: Any, gear: "Gear") -> None:
        self._dataloader = dataloader
        self._gear = gear

    def __iter__(self) -> Any:
        for batch in self._dataloader:
            yield self._gear.to_device(batch)

    def __len__(self) -> int:
        return len(self._dataloader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataloader, name)
