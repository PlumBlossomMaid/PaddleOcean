"""ThroughputMonitor callback - measures training throughput."""

import time
from typing import Any

from ocean.callbacks.callback import Callback


class ThroughputMonitor(Callback):
    """Monitor training throughput (samples/sec).

    Args:
        window_size: Number of batches to average over.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._batch_times: list[float] = []
        self._batch_sizes: list[int] = []
        self._epoch_start: float = 0.0

    def on_train_epoch_start(self, trainer: Any, model: Any) -> None:
        self._epoch_start = time.monotonic()

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if hasattr(self, "_batch_end_time"):
            elapsed = time.monotonic() - self._batch_end_time
            self._batch_times.append(elapsed)
            batch_size = 1
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                b = batch[0]
                if hasattr(b, "shape"):
                    batch_size = b.shape[0]
            self._batch_sizes.append(batch_size)

            if len(self._batch_times) > self.window_size:
                self._batch_times.pop(0)
                self._batch_sizes.pop(0)

        self._batch_end_time = time.monotonic()

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if self._batch_times:
            total_time = sum(self._batch_times)
            total_samples = sum(self._batch_sizes)
            throughput = total_samples / total_time if total_time > 0 else 0
            model.log("throughput_samples_per_sec", throughput, logger=True)
