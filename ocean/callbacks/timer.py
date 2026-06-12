import time

from ocean.callbacks.callback import Callback


class Timer(Callback):
    def __init__(self):
        self.start_time = None
        self.epoch_start = None
        self.total_time = 0.0

    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_start:
            epoch_time = time.time() - self.epoch_start
            pl_module.log("time/epoch", epoch_time)

    def on_fit_end(self, trainer, pl_module):
        if self.start_time:
            self.total_time = time.time() - self.start_time
            pl_module.log("time/total", self.total_time)

    def time_elapsed(self):
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
