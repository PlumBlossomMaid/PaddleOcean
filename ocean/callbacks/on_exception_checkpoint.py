"""OnExceptionCheckpoint callback - saves checkpoint on exception."""

from typing import Any

from ocean.callbacks.callback import Callback


class OnExceptionCheckpoint(Callback):
    """Save a checkpoint when an exception occurs during training.

    Args:
        dirpath: Directory to save the checkpoint.
        filename: Checkpoint filename.
    """

    def __init__(self, dirpath: str = ".", filename: str = "exception.ckpt") -> None:
        self.dirpath = dirpath
        self.filename = filename
        import os
        os.makedirs(dirpath, exist_ok=True)

    def on_exception(self, trainer: Any, model: Any, exception: BaseException) -> None:
        import os

        import paddle
        path = os.path.join(self.dirpath, self.filename)
        checkpoint = {"state_dict": model.state_dict(), "exception": str(exception)}
        paddle.save(checkpoint, path)
        print(f"Saved exception checkpoint to '{path}'")
