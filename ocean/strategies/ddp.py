"""DDPStrategy - Distributed Data Parallel using PaddlePaddle's native distributed API.

Supports:
- ``paddle.distributed.init_parallel_env`` / ``init_parallel_env``
- ``paddle.distributed.DataParallel`` for model wrapping
- Collective ops: all_reduce, broadcast, all_gather, barrier, reduce
- ``paddle.distributed.spawn`` for process launching
- Device-agnostic: works with CUDA, XPU, and CustomDevice accelerators
"""

import os
from typing import Any, Optional

import paddle

from ocean.strategies.parallel import ParallelStrategy


class DDPStrategy(ParallelStrategy):
    """Distributed Data Parallel strategy using PaddlePaddle native distributed API.

    Device-agnostic: works with CUDA, XPU, and CustomDevice accelerators.
    ``root_device`` is derived from ``parallel_devices[local_rank]``, so the
    device type is determined entirely by the accelerator.

    Args:
        process_group_backend: Communication backend (``'nccl'``, ``'gloo'``,
            or ``None`` for auto-detect based on accelerator type).
        find_unused_parameters: If True, find unused parameters in the model.
        parallel_devices: List of device ``Place`` objects for each process.
        accelerator: The accelerator instance (injected by ``_AcceleratorConnector``).
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        process_group_backend: Optional[str] = None,
        find_unused_parameters: bool = False,
        parallel_devices: Optional[list[Any]] = None,
        accelerator: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(accelerator=accelerator, *args, **kwargs)
        self._process_group_backend = process_group_backend
        self._find_unused_parameters = find_unused_parameters
        self.parallel_devices = parallel_devices or []
        self._is_initialized = False
        self._rank = 0
        self._local_rank = 0
        self._world_size = 1
        self._node_rank = 0
        self._detect_existing_distributed()

    def _detect_existing_distributed(self) -> None:
        """Detect if distributed env is already initialized and set rank from env vars.

        Lightning sets ``rank_zero_only.rank`` in ``set_world_ranks()`` **before**
        calling ``init_process_group``, reading ``RANK`` / ``LOCAL_RANK`` from the
        environment set by the launcher.

        Paddle's launcher sets ``PADDLE_TRAINER_ID`` (global rank) instead.
        We read it here so ``rank_zero_only`` works from the start, even before
        ``init_parallel_env()`` is called.
        """
        # Set rank from launcher env vars (Lightning pattern — before init)
        trainer_id = os.environ.get("PADDLE_TRAINER_ID")
        if trainer_id is not None:
            self._rank = int(trainer_id)
            self._local_rank = self._rank  # single-node: local == global
            self._world_size = int(os.environ.get("PADDLE_TRAINERS_NUM", "1"))
            from ocean.utils.rank_zero import rank_zero_only

            rank_zero_only.rank = self._rank

        try:
            if paddle.distributed.is_initialized():
                self._is_initialized = True
                self._rank = paddle.distributed.get_rank()
                self._world_size = paddle.distributed.get_world_size()
                self._local_rank = int(paddle.distributed.ParallelEnv().local_rank)
                self._node_rank = int(paddle.distributed.ParallelEnv().node_rank)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Device-agnostic root_device — delegates entirely to parallel_devices
    # ------------------------------------------------------------------

    @property
    def root_device(self) -> Any:
        """Device for this process.

        In both launch and spawn modes, each process uses
        ``parallel_devices[local_rank]`` to get its assigned GPU.
        The ``CUDAAccelerator.setup_device()`` then calls
        ``paddle.device.set_device()`` to make it the active device.
        """
        if self.parallel_devices and self._local_rank < len(self.parallel_devices):
            return self.parallel_devices[self._local_rank]
        # Fallback when parallel_devices is not set
        if paddle.is_compiled_with_cuda():
            return paddle.CUDAPlace(self._local_rank)
        return paddle.CPUPlace()

    @property
    def is_global_zero(self) -> bool:
        """Whether this process is global rank 0.

        Uses ``LOCAL_RANK`` env var as fallback before ``_rank`` is set
        by ``setup_environment()`` (e.g. during Trainer init).
        """
        if not self._is_initialized:
            return int(os.environ.get("LOCAL_RANK", 0)) == 0
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

    # ------------------------------------------------------------------
    # Setup — matches Lightning's pattern:
    #   1. accelerator.setup_device(root_device)     → set device
    #   2. paddle.distributed.init_parallel_env()    → init distributed
    #   3. rank_zero_only.rank = global_rank         → rank filter
    # ------------------------------------------------------------------

    def setup_environment(self) -> None:
        """Set up distributed environment: device + process group.

        Call chain::
            accelerator.setup_device(root_device)
            paddle.distributed.init_parallel_env()
        """
        # Two modes:
        # - launch (paddle.distributed.launch): FLAGS_selected_gpus already
        #   restricts each process to one GPU.  root_device = CUDAPlace(0)
        #   which is the only visible GPU (different physical GPU per rank).
        # - spawn (paddle.distributed.spawn): same restriction via
        #   FLAGS_selected_gpus.  We detect this and use index 0.

        # Step 1: Initialize distributed if not already done.
        # IMPORTANT: init BEFORE set_device so NCCL uses the current
        # FLAGS_selected_gpus (which restricts each process to one GPU).
        if not self._is_initialized:
            try:
                if paddle.distributed.is_available():
                    paddle.distributed.init_parallel_env()
                    self._is_initialized = True
                    self._rank = paddle.distributed.get_rank()
                    self._world_size = paddle.distributed.get_world_size()
                    try:
                        env = paddle.distributed.ParallelEnv()
                        self._local_rank = int(env.local_rank)
                        self._node_rank = int(env.node_rank)
                    except Exception:
                        self._local_rank = self._rank
            except Exception:
                pass

        # Step 2: Set device for this process via accelerator
        if self._accelerator:
            self._accelerator.setup_device(self.root_device)
        else:
            self._default_device_setup()

        # Step 3: Sync rank to rank_zero_only (Lightning pattern)
        from ocean.utils.rank_zero import rank_zero_only

        rank_zero_only.rank = self._rank

    def _default_device_setup(self) -> None:
        """Fallback device setup when no accelerator is configured."""
        device = self.root_device
        if isinstance(device, paddle.CUDAPlace):
            paddle.device.set_device(f"gpu:{self._local_rank}")
        elif isinstance(device, paddle.XPUPlace):
            paddle.device.set_device(f"xpu:{self._local_rank}")

    def setup(self, trainer: Any) -> None:
        """Full setup: precision, model to device, DDP wrapping, optimizers."""
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

    def _determine_ddp_device_ids(self) -> Optional[list[int]]:
        """Return device_ids for DDP, or ``None`` for CPU.

        Matches Lightning's ``determine_ddp_device_ids()``:
        - CUDA/XPU: return ``[local_rank]``
        - CPU: return ``None``
        """
        device = self.root_device
        if isinstance(device, (paddle.CUDAPlace, paddle.XPUPlace)):
            return [self._local_rank]
        # CustomDevicePlace check via string representation
        device_str = str(device)
        if device_str.startswith("Place(custom"):
            return [self._local_rank]
        return None

    def model_to_device(self) -> None:
        """Move model to the appropriate device (device-agnostic)."""
        if self._model is not None:
            self._model.to(self.root_device)

    # ==================================================================
    # Collective communication operations (unchanged, already device-agnostic)
    # ==================================================================

    def reduce(self, tensor: Any, reduce_op: str = "mean", group: Any = None) -> Any:
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
        """Save checkpoint - only on global rank 0.

        Uses ``paddle.save`` directly instead of ``paddle.distributed.save_state_dict``
        to avoid:
        1. Collective ops (``all_gather``) on a potentially-destroyed process group
        2. ``save_state_dict`` creating a directory that conflicts with subsequent saves
        """
        if self.is_global_zero:
            # Safety: if a directory was left behind by a previous failed
            # ``save_state_dict`` call, remove it so ``paddle.save`` works.
            import os

            if os.path.isdir(filepath):
                import shutil

                shutil.rmtree(filepath)
            paddle.save(checkpoint, filepath)

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """Load checkpoint with distributed support."""
        try:
            ckpt = paddle.load(checkpoint_path)
            return ckpt
        except Exception:
            return {}

    # ==================================================================
    # Launch utilities — matches Lightning's launcher pattern
    # ==================================================================

    @staticmethod
    def spawn(fn: Any, nprocs: int = 1, **kwargs: Any) -> Any:
        """Launch distributed training using ``paddle.distributed.spawn``.

        Args:
            fn: Function to run in parallel. Receives ``fn(*args)`` in each process.
            nprocs: Number of processes to spawn.
            **kwargs: Arguments passed to ``paddle.distributed.spawn``.
        """
        return paddle.distributed.spawn(fn, nprocs=nprocs, **kwargs)

    @staticmethod
    def launch(fn: Any, **kwargs: Any) -> Any:
        """Launch distributed training (alias for ``spawn``).

        Args:
            fn: Function to run.
            **kwargs: Arguments passed to ``paddle.distributed.spawn``.
        """
        return DDPStrategy.spawn(fn, **kwargs)

    def num_processes(self) -> int:
        """Return the total number of processes for this strategy."""
        return len(self.parallel_devices) if self.parallel_devices else self._world_size

    # ==================================================================
    # Teardown
    # ==================================================================

    def teardown(self) -> None:
        """Clean up distributed resources.

        Does NOT call ``destroy_process_group`` here because the caller
        (``Trainer._teardown``) is invoked after ``fit_loop`` but before
        ``test`` / ``save_checkpoint`` — destroying the group would break
        subsequent collective ops.  The process group is automatically
        cleaned up when the subprocess exits.
        """
        if self._is_initialized:
            try:
                paddle.distributed.barrier()
            except Exception:
                pass
