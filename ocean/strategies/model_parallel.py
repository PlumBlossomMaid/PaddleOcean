"""Model parallel strategy using PaddlePaddle's shard_tensor / ProcessMesh.

Supports tensor parallelism and pipeline parallelism via
paddle.distributed.shard_tensor and ProcessMesh.
"""

from typing import Any, Optional

import paddle

from ocean.strategies.parallel import ParallelStrategy
from ocean.utils.rank_zero import rank_zero_warn


class ModelParallelStrategy(ParallelStrategy):
    """Model parallel strategy using PaddlePaddle's ProcessMesh and shard_tensor.

    Args:
        tensor_parallel_size: Number of devices for tensor parallelism.
        data_parallel_size: Number of devices for data parallelism.
        pipeline_parallel_size: Number of devices for pipeline parallelism.
    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self._mesh = None

    @property
    def root_device(self) -> Any:
        return paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()

    @property
    def is_global_zero(self) -> bool:
        return True

    def setup(self, trainer: Any) -> None:
        """Setup with ProcessMesh for model parallelism."""
        super().setup(trainer)
        try:
            # pipeline_parallel_size was accepted and then left out of the mesh,
            # so a pipeline dimension was silently dropped.
            mesh_dims = [self.pipeline_parallel_size, self.tensor_parallel_size, self.data_parallel_size]
            mesh_dims = [d for d in mesh_dims if d > 1] or [1]
            self._mesh = paddle.distributed.ProcessMesh(mesh_dims)
            paddle.distributed.set_mesh(self._mesh)
        except Exception as exception:
            rank_zero_warn(f"Could not build the process mesh, model parallelism is NOT active: {exception!r}")

    def reduce(self, tensor: Any, reduce_op: str = "mean") -> Any:
        """Reduce a tensor across ranks using PaddlePaddle distributed."""
        if not paddle.distributed.is_initialized():
            return tensor
        try:
            import ocean.distributed as odist

            return odist.reduce(tensor, reduce_op=reduce_op)
        except Exception:
            return tensor

    def barrier(self, name: Optional[str] = None) -> None:
        """Synchronize all ranks."""
        if paddle.distributed.is_initialized():
            try:
                paddle.distributed.barrier()
            except Exception:
                pass

    def broadcast(self, obj: Any, src: int = 0) -> Any:
        """Broadcast an object from src rank to all ranks."""
        if not paddle.distributed.is_initialized():
            return obj
        try:
            import ocean.distributed as odist

            return odist.broadcast(obj, src=src)
        except Exception:
            return obj
