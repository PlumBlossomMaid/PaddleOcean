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
    """Progress bar using ColoredTqdm (rainbow)."""

    def __init__(self) -> None:
        super().__init__()
        self._train_tqdm = None
        self._val_tqdm = None

    def on_train_epoch_start(self, trainer: Any, model: Any) -> None:
        try:
            from ocean.utils.colored_tqdm import ColoredTqdm as tqdm  # noqa: N813

            total = self._get_total(trainer, "train")
            self._train_tqdm = tqdm(
                total=total,
                desc=f"Epoch {trainer.current_epoch}",
                leave=True,
                unit="batch",
                ncols=120,
            )
        except ImportError:
            self._train_tqdm = None

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._train_tqdm is not None:
            self._train_tqdm.update(1)
            metrics = trainer.callback_metrics
            if metrics:
                self._train_tqdm.set_postfix(**{k: f"{v:.4f}" for k, v in metrics.items()}, refresh=False)

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if self._train_tqdm is not None:
            self._train_tqdm.close()
            self._train_tqdm = None

    # ── Validation / sanity check progress ──

    def on_validation_start(self, trainer: Any, model: Any) -> None:
        """Create progress bar for validation steps (including sanity check)."""
        try:
            from ocean.utils.colored_tqdm import ColoredTqdm as tqdm  # noqa: N813

            total = (
                trainer.num_sanity_val_steps if trainer.num_sanity_val_steps > 0 else self._get_total(trainer, "val")
            )
            self._val_tqdm = tqdm(
                total=total,
                desc="Validation" if trainer.dataloader_step > 0 else "Sanity Check",
                leave=False,
                unit="step",
                ncols=80,
            )
        except ImportError:
            self._val_tqdm = None

    def on_validation_batch_end(
        self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self._val_tqdm is not None:
            self._val_tqdm.update(1)

    def on_validation_end(self, trainer: Any, model: Any) -> None:
        if self._val_tqdm is not None:
            self._val_tqdm.close()
            self._val_tqdm = None
