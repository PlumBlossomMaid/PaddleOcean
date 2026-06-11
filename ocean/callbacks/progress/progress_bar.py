"""ProgressBar and TQDMProgressBar callbacks."""

from typing import Any, Optional

from ocean.callbacks.callback import Callback


class ProgressBar(Callback):
    """Base progress bar callback.

    Subclass and override to implement custom progress bars.
    """

    def __init__(self) -> None:
        self._trainer: Optional[Any] = None

    @property
    def trainer(self) -> Any:
        if self._trainer is None:
            raise TypeError("ProgressBar not attached to a Trainer")
        return self._trainer

    def setup(self, trainer: Any, model: Any, stage: str) -> None:
        self._trainer = trainer

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        self._update_bar(trainer, "train", batch_idx)

    def on_validation_batch_end(
        self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._update_bar(trainer, "val", batch_idx)

    def _update_bar(self, trainer: Any, stage: str, batch_idx: int) -> None:
        pass  # Override in subclasses


class TQDMProgressBar(ProgressBar):
    """Progress bar using tqdm."""

    def __init__(self) -> None:
        super().__init__()
        self._tqdm = None

    def on_train_epoch_start(self, trainer: Any, model: Any) -> None:
        try:
            from tqdm import tqdm
            total = self._get_total(trainer, "train")
            self._tqdm = tqdm(total=total, desc=f"Epoch {trainer.current_epoch+1}", leave=True, unit="batch")
        except ImportError:
            self._tqdm = None

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._tqdm is not None:
            self._tqdm.update(1)

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None

    def _get_total(self, trainer: Any, stage: str) -> int:
        loader = getattr(trainer, f"{stage}_dataloader", None) or getattr(trainer, f"{stage}_dataloaders", None)
        if isinstance(loader, (list, tuple)):
            loader = loader[0] if loader else None
        if loader is not None:
            try:
                return len(loader)
            except (TypeError, AttributeError):
                pass
        return 100
