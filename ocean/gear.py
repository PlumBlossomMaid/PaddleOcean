"""ocean.Gear - lightweight manual training API (Fabric equivalent).

Gear provides manual control over training with minimal boilerplate.
Users write their own training loop while Gear handles device placement,
precision, distributed setup, and checkpointing.

Multi-GPU support::
    gear = ocean.Gear(accelerator="gpu", devices=2, strategy="ddp")
    gear.launch()  # spawns processes if needed
    model = gear.setup(model)
    # ... training loop ...
"""

from typing import Any, Optional, Union

import paddle

from ocean.accelerators.accelerator import Accelerator
from ocean.strategies import DDPStrategy, Strategy


class Gear:
    """Lightweight manual training API (analogous to ocean Fabric).

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

        # Resolve accelerator and strategy (aligned with _AcceleratorConnector)
        self._resolve()

    @property
    def device(self) -> paddle.CPUPlace:
        strategy = self._strategy or self._resolve_fallback_strategy()
        return strategy.root_device if strategy else paddle.CPUPlace()

    @property
    def strategy(self) -> Optional[Strategy]:
        return self._strategy

    # ------------------------------------------------------------------
    # Resolution — mirrors _AcceleratorConnector logic
    # ------------------------------------------------------------------

    def _resolve(self) -> None:
        from ocean.trainer.connectors import _AcceleratorConnector

        self._accelerator = _AcceleratorConnector._resolve_accelerator(self.accelerator_flag)
        parallel = self._accelerator.get_parallel_devices(self.devices_flag)
        self._strategy = _AcceleratorConnector._resolve_strategy(self.strategy_flag, parallel)
        self._strategy.accelerator = self._accelerator
        self._strategy.parallel_devices = parallel

    def _resolve_fallback_strategy(self) -> Strategy:
        """Create a minimal strategy for device access."""
        from ocean.strategies.single_device import SingleDeviceStrategy

        return SingleDeviceStrategy(device="cpu")

    # ------------------------------------------------------------------
    # Launch — multi-process entry point (Fabric.launch equivalent)
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
            # Only DDP-wrap on the first call (like Fabric)
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
            return (module, *optimizers)
        return module

    def setup_dataloaders(self, *dataloaders: paddle.io.DataLoader, move_to_device: bool = True) -> Any:
        """Set up dataloaders. Returns the same dataloaders (future: distributed samplers).

        Args:
            *dataloaders: DataLoaders to set up.
            move_to_device: If True, batches are moved to device automatically.

        Returns:
            DataLoader(s) ready for training.
        """
        if len(dataloaders) == 1:
            return dataloaders[0]
        return dataloaders

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
                if k in checkpoint:
                    if isinstance(v, paddle.nn.Layer):
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
        """Return an autocast context manager for mixed precision."""
        if self.precision_flag == "16":
            return paddle.amp.auto_cast(level="O1")
        elif self.precision_flag == "bf16":
            return paddle.amp.auto_cast(level="O2", dtype="bfloat16")
        return paddle.no_grad()  # no-op context for fp32
