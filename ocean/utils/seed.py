"""Seed utilities - set global random seeds."""

import random
from typing import Optional

import numpy as np
import paddle


def seed_everything(seed: Optional[int] = None, workers: bool = True, verbose: bool = True) -> int:
    """Set global random seed for reproducibility.

    Args:
        seed: Random seed. If None, a random seed is generated.
        workers: If True, also seed DataLoader workers.
        verbose: If True, print the seed used.

    Returns:
        The seed used.
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    if verbose:
        print(f"Global seed set to {seed}")

    return seed
