"""ocean.distributed - Complete PaddlePaddle distributed API wrapper.

Provides `import ocean; ocean.distributed.all_reduce(tensor)` style access
to all PaddlePaddle distributed operations.

Covers:
- Collective communication (all_reduce, broadcast, all_gather, reduce, etc.)
- Stream collective communication
- Initialization (init_parallel_env, get_rank, get_world_size, etc.)
- Process group management (new_group, destroy_process_group, etc.)
- DataParallel / DistributedDataParallel
- Fleet API integration
- Sharding (group_sharded, shard_tensor, shard_layer, etc.)
- RPC (init_rpc, rpc_sync, rpc_async, shutdown)
- Launch utilities (spawn, launch)
- State dict save/load
- Environment (ParallelEnv, get_backend, etc.)
"""

import functools
import os
from typing import Any, Callable, Optional, Union

import paddle


# ====================================================================
# Initialization and Environment
# ====================================================================

def is_available() -> bool:
    """Check if distributed environment is available."""
    return paddle.distributed.is_available()


def is_initialized() -> bool:
    """Check if distributed environment has been initialized."""
    return paddle.distributed.is_initialized()


def init_parallel_env() -> None:
    """Initialize parallel training environment for dynamic mode."""
    paddle.distributed.init_parallel_env()


def get_rank() -> int:
    """Get rank of current process."""
    return paddle.distributed.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    return paddle.distributed.get_world_size()


def get_backend(group: Any = None) -> str:
    """Get communication backend name for the given group."""
    return paddle.distributed.get_backend(group)


def ParallelEnv() -> Any:
    """Get parallel execution environment variables."""
    return paddle.distributed.ParallelEnv()


# ====================================================================
# Process Group Management
# ====================================================================

def new_group(ranks: list[int]) -> Any:
    """Create a new distributed communication group."""
    return paddle.distributed.new_group(ranks)


def get_group(group_id: int) -> Any:
    """Get communication group instance by id."""
    return paddle.distributed.get_group(group_id)


def destroy_process_group(group: Optional[Any] = None) -> None:
    """Destroy a distributed communication group."""
    paddle.distributed.destroy_process_group(group)


# ====================================================================
# Collective Communication Operations
# ====================================================================

def all_reduce(tensor: Any, op: str = "mean") -> Any:
    """All-reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce.
        op: Reduction operation ('mean', 'sum', 'min', 'max', 'prod').

    Returns:
        Reduced tensor (in-place).
    """
    reduce_op_map = {
        "sum": paddle.distributed.ReduceOp.SUM,
        "mean": paddle.distributed.ReduceOp.SUM,  # Manually divide by world_size
        "min": paddle.distributed.ReduceOp.MIN,
        "max": paddle.distributed.ReduceOp.MAX,
        "prod": paddle.distributed.ReduceOp.PROD,
    }
    dist_op = reduce_op_map.get(op, paddle.distributed.ReduceOp.SUM)
    paddle.distributed.all_reduce(tensor, op=dist_op)
    if op == "mean":
        tensor = tensor / get_world_size()
    return tensor


def broadcast(tensor: Any, src: int = 0) -> Any:
    """Broadcast tensor from source process to all others."""
    paddle.distributed.broadcast(tensor, src=src)
    return tensor


def broadcast_object_list(obj_list: list, src: int = 0) -> list:
    """Broadcast a list of Python objects from source to all processes."""
    paddle.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list


def all_gather(tensor: Any) -> list:
    """Gather tensors from all processes into a list."""
    result = []
    paddle.distributed.all_gather(result, tensor)
    return result


def all_gather_object(obj: Any) -> list:
    """Gather Python objects from all processes."""
    result = []
    paddle.distributed.all_gather_object(result, obj)
    return result


def reduce(tensor: Any, dst: int = 0, op: str = "sum") -> Any:
    """Reduce tensor across processes and send to destination process."""
    reduce_op_map = {
        "sum": paddle.distributed.ReduceOp.SUM,
        "mean": paddle.distributed.ReduceOp.SUM,
        "min": paddle.distributed.ReduceOp.MIN,
        "max": paddle.distributed.ReduceOp.MAX,
    }
    dist_op = reduce_op_map.get(op, paddle.distributed.ReduceOp.SUM)
    paddle.distributed.reduce(tensor, dst=dst, op=dist_op)
    if op == "mean":
        tensor = tensor / get_world_size()
    return tensor


def alltoall(tensor_list: list) -> list:
    """All-to-all communication: each process sends a tensor to every other process."""
    return paddle.distributed.alltoall(tensor_list)


def alltoall_single(tensor: Any) -> Any:
    """All-to-all with a single tensor: distribute and aggregate."""
    return paddle.distributed.alltoall_single(tensor)


def scatter(tensor_list: list, src: int = 0) -> Any:
    """Scatter tensors from source process to all processes."""
    return paddle.distributed.scatter(tensor_list, src=src)


def scatter_object_list(obj_list: list, src: int = 0) -> list:
    """Scatter Python objects from source to all processes."""
    return paddle.distributed.scatter_object_list(obj_list, src=src)


def reduce_scatter(tensor: Any) -> Any:
    """Reduce a list of tensors then scatter the result."""
    return paddle.distributed.reduce_scatter(tensor)


def barrier() -> None:
    """Synchronize all processes (barrier)."""
    paddle.distributed.barrier()


def gloo_barrier() -> None:
    """Barrier using gloo backend."""
    paddle.distributed.gloo_barrier()


def send(tensor: Any, dst: int) -> None:
    """Send a tensor to destination process synchronously."""
    paddle.distributed.send(tensor, dst=dst)


def recv(tensor: Any, src: int) -> None:
    """Receive a tensor from source process synchronously."""
    paddle.distributed.recv(tensor, src=src)


def isend(tensor: Any, dst: int) -> Any:
    """Send a tensor to destination process asynchronously."""
    return paddle.distributed.isend(tensor, dst=dst)


def irecv(tensor: Any, src: int) -> Any:
    """Receive a tensor from source process asynchronously."""
    return paddle.distributed.irecv(tensor, src=src)


def wait(tensor: Any, group: Any = None) -> None:
    """Wait for a communication operation to complete."""
    paddle.distributed.wait(tensor, group)


# ====================================================================
# Stream Collective Communication (Advanced)
# ====================================================================

class stream:
    """Stream collective communication operations.

    Usage::
        ocean.distributed.stream.all_reduce(tensor)
    """

    @staticmethod
    def all_reduce(tensor: Any, op: str = "sum") -> Any:
        reduce_op_map = {
            "sum": paddle.distributed.ReduceOp.SUM,
            "mean": paddle.distributed.ReduceOp.SUM,
            "min": paddle.distributed.ReduceOp.MIN,
            "max": paddle.distributed.ReduceOp.MAX,
        }
        dist_op = reduce_op_map.get(op, paddle.distributed.ReduceOp.SUM)
        paddle.distributed.stream.all_reduce(tensor, op=dist_op)
        if op == "mean":
            tensor = tensor / get_world_size()
        return tensor

    @staticmethod
    def broadcast(tensor: Any, src: int = 0) -> Any:
        return paddle.distributed.stream.broadcast(tensor, src=src)

    @staticmethod
    def all_gather(tensor: Any) -> list:
        result = []
        paddle.distributed.stream.all_gather(result, tensor)
        return result

    @staticmethod
    def alltoall(tensor_list: list) -> list:
        return paddle.distributed.stream.alltoall(tensor_list)

    @staticmethod
    def alltoall_single(tensor: Any) -> Any:
        return paddle.distributed.stream.alltoall_single(tensor)

    @staticmethod
    def scatter(tensor_list: list, src: int = 0) -> Any:
        return paddle.distributed.stream.scatter(tensor_list, src=src)

    @staticmethod
    def reduce_scatter(tensor: Any) -> Any:
        return paddle.distributed.stream.reduce_scatter(tensor)

    @staticmethod
    def reduce(tensor: Any, dst: int = 0, op: str = "sum") -> Any:
        reduce_op_map = {
            "sum": paddle.distributed.ReduceOp.SUM,
            "mean": paddle.distributed.ReduceOp.SUM,
        }
        dist_op = reduce_op_map.get(op, paddle.distributed.ReduceOp.SUM)
        paddle.distributed.stream.reduce(tensor, dst=dst, op=dist_op)
        if op == "mean":
            tensor = tensor / get_world_size()
        return tensor

    @staticmethod
    def send(tensor: Any, dst: int) -> None:
        paddle.distributed.stream.send(tensor, dst=dst)

    @staticmethod
    def recv(tensor: Any, src: int) -> None:
        paddle.distributed.stream.recv(tensor, src=src)


# ====================================================================
# Distributed DataParallel
# ====================================================================

def DataParallel(
    model: paddle.nn.Layer,
    find_unused_parameters: bool = False,
) -> paddle.nn.Layer:
    """Wrap model with Distributed Data Parallel.

    Args:
        model: The model to wrap.
        find_unused_parameters: If True, find unused parameters.

    Returns:
        DDP-wrapped model.
    """
    if is_initialized():
        return paddle.distributed.DataParallel(
            model,
            find_unused_parameters=find_unused_parameters,
        )
    return model


def parallelize(model: Any, optimizer: Any, strategy: Any = None) -> tuple:
    """Parallelize model and optimizer for distributed training.

    Args:
        model: Model to parallelize.
        optimizer: Optimizer to parallelize.
        strategy: Distributed strategy configuration.

    Returns:
        Tuple of (parallelized_model, parallelized_optimizer).
    """
    return paddle.distributed.parallelize(model, optimizer, strategy)


# ====================================================================
# Fleet API
# ====================================================================

class fleet:
    """Fleet API for large-scale distributed training.

    Usage::
        ocean.distributed.fleet.init(is_collective=True)
        model = ocean.distributed.fleet.distributedmodel(model)
        optimizer = ocean.distributed.fleet.distributed_optimizer(optimizer)
    """

    @staticmethod
    def init(is_collective: bool = True) -> None:
        """Initialize fleet distributed training environment."""
        paddle.distributed.fleet.init(is_collective=is_collective)

    @staticmethod
    def distributedmodel(model: paddle.nn.Layer) -> Any:
        """Wrap model with fleet distributed model."""
        return paddle.distributed.fleet.distributedmodel(model)

    @staticmethod
    def distributed_optimizer(optimizer: Any, strategy: Any = None) -> Any:
        """Wrap optimizer with fleet distributed optimizer."""
        return paddle.distributed.fleet.distributed_optimizer(optimizer, strategy)

    @staticmethod
    def DistributedStrategy() -> Any:
        """Create a distributed strategy configuration."""
        return paddle.distributed.fleet.DistributedStrategy()


# ====================================================================
# Launch Utilities
# ====================================================================

def spawn(fn: Callable, nprocs: int = 1, **kwargs: Any) -> Any:
    """Spawn distributed processes.

    Args:
        fn: Function to execute in each process.
        nprocs: Number of processes to spawn.
        **kwargs: Additional arguments to paddle.distributed.spawn.

    Returns:
        Results from spawned processes.
    """
    return paddle.distributed.spawn(fn, nprocs=nprocs, **kwargs)


def launch(fn: Callable, **kwargs: Any) -> Any:
    """Launch distributed processes (alias for spawn).

    Args:
        fn: Function to execute.
        **kwargs: Arguments passed to paddle.distributed.spawn.
    """
    return spawn(fn, **kwargs)


# ====================================================================
# Sharding
# ====================================================================

def shard_tensor(
    tensor: Any,
    dims: list[int],
    mesh: Optional[Any] = None,
) -> Any:
    """Create a distributed tensor with sharding information."""
    return paddle.distributed.shard_tensor(tensor, dims=dims, mesh=mesh)


def shard_layer(
    layer: paddle.nn.Layer,
    strategy: Any = None,
) -> paddle.nn.Layer:
    """Shard a layer's parameters into distributed tensors."""
    return paddle.distributed.shard_layer(layer, strategy=strategy)


def shard_dataloader(
    dataloader: Any,
    mesh: Optional[Any] = None,
    shuffle: bool = True,
) -> Any:
    """Convert a single-process dataloader into a distributed dataloader."""
    return paddle.distributed.shard_dataloader(dataloader, mesh=mesh, shuffle=shuffle)


def shard_optimizer(
    optimizer: Any,
    strategy: Any = None,
) -> Any:
    """Convert a single-process optimizer into a distributed one."""
    return paddle.distributed.shard_optimizer(optimizer, strategy=strategy)


def reshard(tensor: Any, target_dist_attr: Any) -> Any:
    """Reshard a distributed tensor to a different distribution."""
    return paddle.distributed.reshard(tensor, target_dist_attr)


def to_distributed(model: Any) -> Any:
    """Convert a model to distributed training mode."""
    return paddle.distributed.to_distributed(model)


def to_static(model: Any) -> Any:
    """Convert a distributed dynamic graph model to static graph."""
    return paddle.distributed.to_static(model)


# ====================================================================
# Group Sharding (ZeRO-style)
# ====================================================================

def group_sharded_parallel(
    model: paddle.nn.Layer,
    optimizer: Any,
    level: str = "os_g",
    scaler: Optional[Any] = None,
) -> tuple:
    """Group sharded parallel training (ZeRO-style).

    Args:
        model: Model to shard.
        optimizer: Optimizer to shard.
        level: Sharding level ('os_g', 'os_g2', 'p_g', 'p_g2').
        scaler: Optional GradScaler for AMP.

    Returns:
        Tuple of (sharded_model, sharded_optimizer, sharded_scaler).
    """
    return paddle.distributed.sharding.group_sharded_parallel(
        model, optimizer, level=level, scaler=scaler,
    )


def save_group_sharded_model(
    model: paddle.nn.Layer,
    output_path: str,
    optimizer: Optional[Any] = None,
) -> None:
    """Save model state from group sharded parallel training."""
    paddle.distributed.sharding.save_group_sharded_model(
        model, output_path, optimizer=optimizer,
    )


# ====================================================================
# State Dict Save/Load (Distributed Checkpoint)
# ====================================================================

def save_state_dict(
    state_dict: dict,
    path: str,
    **kwargs: Any,
) -> None:
    """Save distributed state dict."""
    paddle.distributed.save_state_dict(state_dict, path, **kwargs)


def load_state_dict(
    state_dict: dict,
    path: str,
    **kwargs: Any,
) -> dict:
    """Load distributed state dict from path."""
    paddle.distributed.load_state_dict(state_dict, path, **kwargs)
    return state_dict


# ====================================================================
# RPC (Remote Procedure Call)
# ====================================================================

class rpc:
    """RPC for distributed communication.

    Usage::
        ocean.distributed.rpc.init_rpc()
        result = ocean.distributed.rpc.rpc_sync("worker0", func, args=(x,))
        ocean.distributed.rpc.shutdown()
    """

    @staticmethod
    def init_rpc(name: str, rank: int = 0, world_size: int = 1, **kwargs: Any) -> None:
        """Initialize RPC."""
        paddle.distributed.rpc.init_rpc(name, rank=rank, world_size=world_size, **kwargs)

    @staticmethod
    def rpc_sync(to: str, func: Callable, args: tuple = (), kwargs: dict = None) -> Any:
        """Make a synchronous RPC call."""
        return paddle.distributed.rpc.rpc_sync(to, func, args=args, kwargs=kwargs or {})

    @staticmethod
    def rpc_async(to: str, func: Callable, args: tuple = (), kwargs: dict = None) -> Any:
        """Make an asynchronous RPC call."""
        return paddle.distributed.rpc.rpc_async(to, func, args=args, kwargs=kwargs or {})

    @staticmethod
    def shutdown() -> None:
        """Shutdown RPC."""
        paddle.distributed.rpc.shutdown()

    @staticmethod
    def get_worker_info(worker_name: Optional[str] = None) -> Any:
        """Get worker info."""
        return paddle.distributed.rpc.get_worker_info(worker_name)

    @staticmethod
    def get_all_worker_infos() -> list:
        """Get info of all workers."""
        return paddle.distributed.rpc.get_all_worker_infos()

    @staticmethod
    def get_current_worker_info() -> Any:
        """Get current worker info."""
        return paddle.distributed.rpc.get_current_worker_info()


# ====================================================================
# Process Mesh
# ====================================================================

def set_mesh(mesh: Any) -> None:
    """Set global ProcessMesh."""
    paddle.distributed.set_mesh(mesh)


def get_mesh() -> Any:
    """Get global ProcessMesh."""
    return paddle.distributed.get_mesh()


# ====================================================================
# DTensor utilities
# ====================================================================

def dtensor_from_fn(fn: Callable, dist_attr: Any, *args: Any, **kwargs: Any) -> Any:
    """Create a DTensor from a paddle API function with distributed attributes."""
    return paddle.distributed.dtensor_from_fn(fn, dist_attr, *args, **kwargs)
