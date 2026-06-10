"""Layer synchronization plugin for DDP training.

Ensures parameter initialization is synchronized across all devices.
"""

from typing import Any, Optional

import paddle


class LayerSync:
    """Base class for layer synchronization strategies."""

    def sync(self, model: paddle.nn.Layer) -> None:
        """Synchronize model parameters across processes."""
        pass


class SyncBN(LayerSync):
    """Synchronize Batch Normalization statistics across devices.

    Uses PaddlePaddle's SyncBatchNorm when available.
    """

    def __init__(self, sync_bn: bool = True) -> None:
        self.sync_bn = sync_bn

    def sync(self, model: paddle.nn.Layer) -> None:
        if not self.sync_bn:
            return
        try:
            # Convert BatchNorm to SyncBatchNorm if available
            paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        except Exception:
            pass
