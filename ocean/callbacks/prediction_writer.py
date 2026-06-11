"""PredictionWriter callback - writes predictions during predict stage."""

from typing import Any

import paddle

from ocean.callbacks.callback import Callback


class PredictionWriter(Callback):
    """Write predictions to disk.

    Args:
        output_dir: Directory to write predictions to.
        write_interval: When to write ('batch' or 'epoch').
    """

    def __init__(self, output_dir: str, write_interval: str = "batch") -> None:
        self.output_dir = output_dir
        self.write_interval = write_interval
        import os
        os.makedirs(output_dir, exist_ok=True)

    def on_predict_batch_end(
        self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self.write_interval == "batch" and outputs is not None:
            self._write_batch(outputs, batch_idx, dataloader_idx)

    def _write_batch(self, outputs: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        import os
        if isinstance(outputs, paddle.Tensor):
            paddle.save(outputs, os.path.join(self.output_dir, f"pred_{dataloader_idx}_{batch_idx}.pdtensor"))
        elif isinstance(outputs, dict):
            paddle.save(outputs, os.path.join(self.output_dir, f"pred_{dataloader_idx}_{batch_idx}.pd"))
