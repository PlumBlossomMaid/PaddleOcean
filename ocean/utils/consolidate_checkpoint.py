"""Consolidate sharded checkpoints into a single file."""


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
    import glob
    import os

    consolidated = {}
    for shard_file in sorted(glob.glob(os.path.join(shard_dir, "*.pdparams"))):
        shard = paddle.load(shard_file)
        consolidated.update(shard)

    paddle.save(consolidated, output_path)
    return consolidated
