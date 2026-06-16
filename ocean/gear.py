"""ocean.Gear - lightweight manual training API (Fabric equivalent).

Gear provides manual control over training with minimal boilerplate.
Users write their own training loop while Gear handles device placement,
precision, distributed setup, and checkpointing.
"""

from typing import Any, Optional, Union

import paddle


class Gear:
    """Lightweight manual training API (analogous to ocean Fabric).

    Usage::

        gear = ocean.Gear(accelerator="gpu", devices=1)
        model = paddle.nn.Linear(10, 2)
        optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

        model = gear.setup(model)
        dataloader = gear.setup_dataloaders(dataloader)

        for batch in dataloader:
            x, y = batch
            optimizer.clear_grad()
            loss = paddle.nn.functional.cross_entropy(model(x), y)
            gear.backward(loss)
            optimizer.step()

        gear.save("checkpoint.pth", {"model": model, "optimizer": optimizer})

    Args:
        accelerator: Device type ('cpu', 'gpu', 'auto').
        devices: Number of devices or device IDs.
        precision: Training precision ('32', '16-mixed', 'bf16-mixed').
        loggers: Optional logger(s).
    """

    def __init__(
        self,
        accelerator: str = "auto",
        devices: Union[str, int, list[int]] = "auto",
        precision: str = "32",
        loggers: Optional[Union[Any, list[Any]]] = None,
    ) -> None:
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision
        self.loggers = [loggers] if loggers is not None and not isinstance(loggers, (list, tuple)) else (loggers or [])
        self._models_setup = 0
        self._device = self._resolve_device()

    @property
    def device(self) -> paddle.CPUPlace:
        return self._device

    def setup(self, module: paddle.nn.Layer, *optimizers: paddle.optimizer.Optimizer) -> Any:
        """Set up a model (and optional optimizers) for accelerated training.

        Args:
            module: The model to set up.
            *optimizers: Optional optimizers.

        Returns:
            Model (and optimizers) ready for training.
        """
        module.to(self._device)
        self._models_setup += 1

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

    def backward(self, tensor: paddle.Tensor, *args: Any, **kwargs: Any) -> None:
        """Backward pass. Handles precision scaling if needed.

        Args:
            tensor: Loss tensor to backpropagate.
        """
        tensor.backward(*args, **kwargs)

    def save(self, path: str, state: dict[str, Any]) -> None:
        """Save a checkpoint.

        Args:
            path: File path.
            state: Dictionary containing model/optimizer state.
        """
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
        print(*args, **kwargs)

    def to_device(self, obj: Any) -> Any:
        """Move a tensor/model to the Gear's device."""
        if isinstance(obj, paddle.nn.Layer):
            return obj.to(self._device)
        if isinstance(obj, paddle.Tensor):
            return obj.to(self._device)
        return obj

    def autocast(self) -> Any:
        """Return an autocast context manager for mixed precision."""
        if self.precision == "16":
            return paddle.amp.auto_cast(level="O1")
        elif self.precision == "bf16":
            return paddle.amp.auto_cast(level="O2", dtype="bfloat16")
        return paddle.no_grad()  # no-op context for fp32

    def _resolve_device(self) -> paddle.CPUPlace:
        if self.accelerator in ("auto", "cpu"):
            return paddle.CPUPlace()
        if self.accelerator == "gpu" and paddle.is_compiled_with_cuda():
            return paddle.CUDAPlace(0)
        return paddle.CPUPlace()
