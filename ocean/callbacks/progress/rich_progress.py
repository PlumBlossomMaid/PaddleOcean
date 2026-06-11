"""Rich progress bar callback."""

from typing import Any

from ocean.callbacks.progress.progress_bar import ProgressBar


class RichProgressBar(ProgressBar):
    """Progress bar using the rich library.

    Falls back to basic console output if rich is not installed.
    """

    def __init__(self) -> None:
        super().__init__()
        self._progress = None
        self._task_id = None

    def on_train_epoch_start(self, trainer: Any, model: Any) -> None:
        try:
            from rich.progress import Progress
            self._progress = Progress()
            self._progress.start()
            total = self._get_total(trainer, "train")
            self._task_id = self._progress.add_task(
                f"Epoch {trainer.current_epoch + 1}",
                total=total,
            )
        except ImportError:
            self._progress = None

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=1)

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None
