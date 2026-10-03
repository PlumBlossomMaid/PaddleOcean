"""Seed utilities - set global random seeds."""

import os
import random
from typing import Optional, Union

import numpy as np
import paddle


def seed_everything(
    seed: Optional[int] = None,
    workers: bool = True,
    verbose: bool = True,
    deterministic: Optional[Union[bool, str]] = None,
    benchmark: Optional[bool] = None,
) -> int:
    """Set global random seed for reproducibility.

    Args:
        seed: Random seed. If None, a random seed is generated.
        workers: If True, also seed DataLoader workers.
        verbose: If True, print the seed used.
        deterministic: If True, use deterministic algorithms. Can be "warn" to
            warn instead of error on non-deterministic ops.
        benchmark: If True, enable cudnn benchmark (may hurt reproducibility).

    Returns:
        The seed used.
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    # Handle deterministic and benchmark flags
    if deterministic is True or deterministic == "warn":
        if benchmark is None:
            benchmark = False
        elif benchmark:
            print("Warning: deterministic=True and benchmark=True are incompatible; forcing benchmark=False")
            benchmark = False

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

    if deterministic is not None and verbose:
        print(f"deterministic={deterministic}, benchmark={benchmark}")

    if verbose:
        print(f"Global seed set to {seed}")

    return seed
