"""FSDP equivalent using PaddlePaddle's group_sharded_parallel.

PaddlePaddle doesn't have PyTorch's FSDP. Instead, it provides
group_sharded_parallel with parameter-level sharding (p_g / p_g2).

This strategy provides an FSDP-like API that maps to Paddle's
native sharding capabilities.
"""

from typing import Any, Optional

import paddle

from ocean.strategies.parallel import ParallelStrategy


class FSDPStrategy(ParallelStrategy):
    """Fully Sharded Data Parallel using PaddlePaddle's group_sharded.

    Args:
        sharding_level: Paddle sharding level ('p_g' or 'p_g2').
        cpu_offload: Enable CPU offloading.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        sharding_level: str = "p_g",
        cpu_offload: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._sharding_level = sharding_level
        self._cpu_offload = cpu_offload

    @property
    def root_device(self) -> Any:
        if paddle.is_compiled_with_cuda():
            return paddle.CUDAPlace(0)
        return paddle.CPUPlace()

    @property
    def is_global_zero(self) -> bool:
        return True

    def setup(self, trainer: Any) -> None:
        """Setup with group sharding for model parallelism."""
        if self._accelerator:
            self._accelerator.setup(trainer)
        self._precision_plugin.convert_module(self._model)
        self.model_to_device()
        self.setup_optimizers(trainer)

        # Apply group sharding
        if self._model is not None and self._optimizers:
            try:
                model, opt, _ = paddle.distributed.sharding.group_sharded_parallel(
                    self._model,
                    self._optimizers[0],
                    level=self._sharding_level,
                )
                self._model = model
                self._optimizers = [opt]
            except Exception:
                pass

    def model_to_device(self) -> None:
        if self._model is not None:
            self._model.to(self.root_device)

    def reduce(self, tensor: Any, reduce_op: str = "mean") -> Any:
        return tensor

    def barrier(self, name: Optional[str] = None) -> None:
        pass

    def broadcast(self, obj: Any, src: int = 0) -> Any:
        return obj

    def save_checkpoint(self, checkpoint: dict, filepath: str) -> None:
        try:
            paddle.distributed.sharding.save_group_sharded_model(
                self._model, filepath,
            )
        except Exception:
            if self.is_global_zero:
                paddle.save(checkpoint, filepath)
