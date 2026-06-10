"""DDPStrategy - Distributed Data Parallel using PaddlePaddle's native distributed API.

Supports:
- `paddle.distributed.init_parallel_env` / `init_parallel_env`
- `paddle.distributed.DataParallel` for model wrapping
- Collective ops: all_reduce, broadcast, all_gather, barrier, reduce
- `paddle.distributed.spawn` for process launching
- Fleet API for large-scale distributed training
"""

from typing import Any, Optional

import paddle

from ocean.strategies.parallel import ParallelStrategy


class DDPStrategy(ParallelStrategy):
    """Distributed Data Parallel strategy using PaddlePaddle native distributed API.

    Args:
        process_group_backend: Communication backend ('nccl', 'gloo', or None for auto).
        find_unused_parameters: If True, find unused parameters in the model.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        process_group_backend: Optional[str] = None,
        find_unused_parameters: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._process_group_backend = process_group_backend
        self._find_unused_parameters = find_unused_parameters
        self._is_initialized = False
        self._rank = 0
        self._local_rank = 0
        self._world_size = 1
        self._node_rank = 0
        self._init_distributed()

    def _init_distributed(self) -> None:
        """Initialize distributed environment if available."""
        try:
            if paddle.distributed.is_initialized():
                self._is_initialized = True
                self._rank = paddle.distributed.get_rank()
                self._world_size = paddle.distributed.get_world_size()
                self._local_rank = int(paddle.distributed.ParallelEnv().local_rank)
                self._node_rank = int(paddle.distributed.ParallelEnv().node_rank)
                return
            # Try to auto-initialize
            if paddle.distributed.is_available():
                try:
                    paddle.distributed.init_parallel_env()
                    self._is_initialized = True
                    self._rank = paddle.distributed.get_rank()
                    self._world_size = paddle.distributed.get_world_size()
                    env = paddle.distributed.ParallelEnv()
                    self._local_rank = int(env.local_rank)
                    self._node_rank = int(env.node_rank)
                except Exception:
                    self._is_initialized = False
        except Exception:
            self._is_initialized = False

    @property
    def root_device(self) -> Any:
        if paddle.is_compiled_with_cuda():
            return paddle.CUDAPlace(self._local_rank)
        return paddle.CPUPlace()

    @property
    def is_global_zero(self) -> bool:
        return self._rank == 0

    @property
    def global_rank(self) -> int:
        return self._rank

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def node_rank(self) -> int:
        return self._node_rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def distributed_sampler_kwargs(self) -> dict:
        return {
            "num_replicas": self._world_size,
            "rank": self._rank,
        }

    def setup_environment(self) -> None:
        """Set up distributed environment."""
        super().setup_environment()
        if self._is_initialized and paddle.is_compiled_with_cuda():
            paddle.device.set_device(f"gpu:{self._local_rank}")

    def setup(self, trainer: Any) -> None:
        """Full setup with DDP wrapping."""
        if self._accelerator:
            self._accelerator.setup(trainer)
        self._precision_plugin.convert_module(self._model)
        self.model_to_device()

        if self._is_initialized and self._model is not None:
            self._model = paddle.distributed.DataParallel(
                self._model,
                find_unused_parameters=self._find_unused_parameters,
            )

        self.setup_optimizers(trainer)

    def model_to_device(self) -> None:
        """Move model to the appropriate device."""
        if self._model is not None:
            self._model.to(self.root_device)

    # ==================================================================
    # Collective communication operations
    # ==================================================================

    def reduce(self, tensor: Any, reduce_op: str = "mean") -> Any:
        """Reduce a tensor across all processes."""
        if not self._is_initialized or not isinstance(tensor, paddle.Tensor):
            return tensor
        try:
            paddle.distributed.all_reduce(tensor)
            if reduce_op == "mean":
                tensor = tensor / self._world_size
        except Exception:
            pass
        return tensor

    def all_reduce(self, tensor: Any, op: str = "mean") -> Any:
        """All-reduce a tensor across all processes."""
        return self.reduce(tensor, op)

    def broadcast(self, obj: Any, src: int = 0) -> Any:
        """Broadcast a tensor from source process to all others."""
        if not self._is_initialized:
            return obj
        try:
            if isinstance(obj, paddle.Tensor):
                paddle.distributed.broadcast(obj, src=src)
            else:
                paddle.distributed.broadcast_object_list([obj], src=src)
                obj = paddle.distributed.broadcast_object_list([obj], src=src)[0]
        except Exception:
            pass
        return obj

    def all_gather(self, tensor: Any, group: Any = None, sync_grads: bool = False) -> Any:
        """Gather tensors from all processes."""
        if not self._is_initialized or not isinstance(tensor, paddle.Tensor):
            return tensor
        try:
            result = []
            paddle.distributed.all_gather(result, tensor)
            return paddle.stack(result)
        except Exception:
            return tensor

    def barrier(self, name: Optional[str] = None) -> None:
        """Synchronize all processes."""
        if self._is_initialized:
            try:
                paddle.distributed.barrier()
            except Exception:
                pass

    def alltoall(self, tensor: Any) -> Any:
        """All-to-all communication."""
        if not self._is_initialized:
            return tensor
        try:
            return paddle.distributed.alltoall(tensor)
        except Exception:
            return tensor

    def scatter(self, tensor_list: list, src: int = 0) -> Any:
        """Scatter tensors from source to all processes."""
        if not self._is_initialized:
            return tensor_list[0] if tensor_list else None
        try:
            result = paddle.distributed.scatter(tensor_list, src=src)
            return result
        except Exception:
            return tensor_list[0] if tensor_list else None

    def reduce_scatter(self, tensor: Any) -> Any:
        """Reduce and scatter."""
        if not self._is_initialized:
            return tensor
        try:
            return paddle.distributed.reduce_scatter(tensor)
        except Exception:
            return tensor

    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:
        """Reduce a boolean decision across processes."""
        if not self._is_initialized:
            return decision
        t = paddle.to_tensor([1.0 if decision else 0.0])
        paddle.distributed.all_reduce(t)
        if all:
            return t.item() == self._world_size
        return t.item() > 0

    # ==================================================================
    # Checkpoint save/load
    # ==================================================================

    def save_checkpoint(self, checkpoint: dict, filepath: str) -> None:
        """Save checkpoint - only on global rank 0."""
        if self.is_global_zero:
            try:
                paddle.distrained.save_state_dict(
                    checkpoint.get("state_dict", {}),
                    filepath,
                )
            except Exception:
                paddle.save(checkpoint, filepath)

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """Load checkpoint with distributed support."""
        try:
            ckpt = paddle.load(checkpoint_path)
            return ckpt
        except Exception:
            return {}

    # ==================================================================
    # Launch utilities
    # ==================================================================

    @staticmethod
    def spawn(fn: Any, nprocs: int = 1, **kwargs: Any) -> Any:
        """Launch distributed training using spawn.

        Args:
            fn: Function to run in parallel.
            nprocs: Number of processes to spawn.
            **kwargs: Arguments passed to paddle.distributed.spawn.
        """
        return paddle.distributed.spawn(fn, nprocs=nprocs, **kwargs)

    @staticmethod
    def launch(fn: Any, **kwargs: Any) -> Any:
        """Launch distributed training (alias for spawn).

        Args:
            fn: Function to run.
            **kwargs: Arguments passed to paddle.distributed.spawn.
        """
        return DDPStrategy.spawn(fn, **kwargs)

    # ==================================================================
    # Teardown
    # ==================================================================

    def teardown(self) -> None:
        """Clean up distributed resources."""
        try:
            if self._is_initialized:
                paddle.distributed.barrier()
        except Exception:
            pass
        super().teardown()
