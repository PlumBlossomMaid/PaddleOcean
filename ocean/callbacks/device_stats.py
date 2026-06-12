import paddle

from ocean.callbacks.callback import Callback


class DeviceStatsMonitor(Callback):
    def __init__(self, cpu_stats: bool = True):
        self.cpu_stats = cpu_stats

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx % 100 == 0:
            self._log_stats(pl_module)

    def _log_stats(self, pl_module):
        if paddle.device.is_compiled_with_cuda():
            free, total = paddle.device.cuda.memory_allocated(0), paddle.device.cuda.max_memory_allocated(0)
            pl_module.log("device/gpu_allocated_gb", free / 1e9)
            pl_module.log("device/gpu_max_allocated_gb", total / 1e9)
