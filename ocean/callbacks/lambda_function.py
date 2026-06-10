"""LambdaCallback - creates a callback from lambda functions."""

from typing import Any, Callable, Optional

from ocean.callbacks.callback import Callback


class LambdaCallback(Callback):
    """Create a simple callback from lambda/functions for specific hooks.

    Args:
        on_fit_start: Function called on fit start.
        on_fit_end: Function called on fit end.
        on_train_start: Function called on train start.
        on_train_end: Function called on train end.
        on_train_epoch_start: Function called on train epoch start.
        on_train_epoch_end: Function called on train epoch end.
        on_train_batch_start: Function called on train batch start.
        on_train_batch_end: Function called on train batch end.
        on_validation_start: Function called on validation start.
        on_validation_end: Function called on validation end.
        on_validation_batch_start: Function called on validation batch start.
        on_validation_batch_end: Function called on validation batch end.
        on_test_start: Function called on test start.
        on_test_end: Function called on test end.
        on_exception: Function called on exception.
    """

    def __init__(
        self,
        on_fit_start: Optional[Callable] = None,
        on_fit_end: Optional[Callable] = None,
        on_train_start: Optional[Callable] = None,
        on_train_end: Optional[Callable] = None,
        on_train_epoch_start: Optional[Callable] = None,
        on_train_epoch_end: Optional[Callable] = None,
        on_train_batch_start: Optional[Callable] = None,
        on_train_batch_end: Optional[Callable] = None,
        on_validation_start: Optional[Callable] = None,
        on_validation_end: Optional[Callable] = None,
        on_validation_batch_start: Optional[Callable] = None,
        on_validation_batch_end: Optional[Callable] = None,
        on_test_start: Optional[Callable] = None,
        on_test_end: Optional[Callable] = None,
        on_exception: Optional[Callable] = None,
    ) -> None:
        self._on_fit_start = on_fit_start
        self._on_fit_end = on_fit_end
        self._on_train_start = on_train_start
        self._on_train_end = on_train_end
        self._on_train_epoch_start = on_train_epoch_start
        self._on_train_epoch_end = on_train_epoch_end
        self._on_train_batch_start = on_train_batch_start
        self._on_train_batch_end = on_train_batch_end
        self._on_validation_start = on_validation_start
        self._on_validation_end = on_validation_end
        self._on_validation_batch_start = on_validation_batch_start
        self._on_validation_batch_end = on_validation_batch_end
        self._on_test_start = on_test_start
        self._on_test_end = on_test_end
        self._on_exception = on_exception

    def on_fit_start(self, trainer: Any, model: Any) -> None:
        if self._on_fit_start:
            self._on_fit_start(trainer, model)

    def on_fit_end(self, trainer: Any, model: Any) -> None:
        if self._on_fit_end:
            self._on_fit_end(trainer, model)

    def on_train_start(self, trainer: Any, model: Any) -> None:
        if self._on_train_start:
            self._on_train_start(trainer, model)

    def on_train_end(self, trainer: Any, model: Any) -> None:
        if self._on_train_end:
            self._on_train_end(trainer, model)

    def on_train_epoch_start(self, trainer: Any, model: Any) -> None:
        if self._on_train_epoch_start:
            self._on_train_epoch_start(trainer, model)

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if self._on_train_epoch_end:
            self._on_train_epoch_end(trainer, model)

    def on_train_batch_start(self, trainer: Any, model: Any, batch: Any, batch_idx: int) -> None:
        if self._on_train_batch_start:
            self._on_train_batch_start(trainer, model, batch, batch_idx)

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._on_train_batch_end:
            self._on_train_batch_end(trainer, model, outputs, batch, batch_idx)

    def on_validation_start(self, trainer: Any, model: Any) -> None:
        if self._on_validation_start:
            self._on_validation_start(trainer, model)

    def on_validation_end(self, trainer: Any, model: Any) -> None:
        if self._on_validation_end:
            self._on_validation_end(trainer, model)

    def on_validation_batch_start(self, trainer: Any, model: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self._on_validation_batch_start:
            self._on_validation_batch_start(trainer, model, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self._on_validation_batch_end:
            self._on_validation_batch_end(trainer, model, outputs, batch, batch_idx, dataloader_idx)

    def on_test_start(self, trainer: Any, model: Any) -> None:
        if self._on_test_start:
            self._on_test_start(trainer, model)

    def on_test_end(self, trainer: Any, model: Any) -> None:
        if self._on_test_end:
            self._on_test_end(trainer, model)

    def on_exception(self, trainer: Any, model: Any, exception: BaseException) -> None:
        if self._on_exception:
            self._on_exception(trainer, model, exception)
