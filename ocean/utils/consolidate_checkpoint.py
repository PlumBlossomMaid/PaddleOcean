"""Consolidate sharded checkpoints into a single file."""

from typing import Any, Optional

import paddle


def consolidate_checkpoint(
    shard_dir: str,
    output_path: str,
) -> dict:
    """Consolidate sharded checkpoint files into a single state dict.

    Args:
        shard_dir: Directory containing sharded checkpoint files.
        output_path: Path to save the consolidated checkpoint.

    Returns:
        Consolidated state dict.
    """
    import os
    import glob

    consolidated = {}
    for shard_file in sorted(glob.glob(os.path.join(shard_dir, "*.pdparams"))):
        shard = paddle.load(shard_file)
        consolidated.update(shard)

    paddle.save(consolidated, output_path)
    return consolidated
