"""DeviceStatsMonitor callback - logs device stats (CPU/GPU memory, utilization)."""

from typing import Any, Optional

from ocean.callbacks.callback import Callback


class DeviceStatsMonitor(Callback):
    """Log device statistics during training.

    Args:
        cpu_stats: If True, log CPU stats as well.
        filter_keys: Optional set of metric keys to filter.
    """

    def __init__(
        self,
        cpu_stats: Optional[bool] = None,
        filter_keys: Optional[set[str]] = None,
    ) -> None:
        self.cpu_stats = cpu_stats
        self.filter_keys = filter_keys

    def on_train_batch_start(self, trainer: Any, model: Any, batch: Any, batch_idx: int) -> None:
        self._log_stats(trainer, "train_batch_start")

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        self._log_stats(trainer, "train_batch_end")

    def _log_stats(self, trainer: Any, hook_name: str) -> None:
        if not getattr(trainer, "loggers", None):
            return
        # Only rank 0 logs device stats (others would create duplicate VDL/CSV files)
        if not getattr(trainer, "is_global_zero", True):
            return

        import paddle

        stats = {}
        if paddle.is_compiled_with_cuda():
            stats["gpu_memory_allocated_mb"] = paddle.device.cuda.memory_allocated() / (1024 * 1024)
            stats["gpu_memory_reserved_mb"] = paddle.device.cuda.memory_reserved() / (1024 * 1024)
        if self.cpu_stats:
            import psutil

            stats["cpu_percent"] = psutil.cpu_percent()
            stats["ram_used_mb"] = psutil.virtual_memory().used / (1024 * 1024)

        if self.filter_keys:
            stats = {k: v for k, v in stats.items() if k in self.filter_keys}

        key_prefix = f"DeviceStatsMonitor/{hook_name}"
        for lg in getattr(trainer, "loggers", None) or []:
            if hasattr(lg, "log_metrics") and stats:
                lg.log_metrics({f"{key_prefix}/{k}": v for k, v in stats.items()})
