"""DataModule - data lifecycle management.

Inspired by LightningDataModule. Manages prepare_data/setup/teardown
and provides dataloader factory methods.
"""

from typing import Any, Optional

import paddle


class DataModule:
    """A DataModule standardizes data splits, preparation and transforms.

    Example::

        import ocean

        class MyData(ocean.DataModule):
            def prepare_data(self):
                download_dataset()

            def setup(self, stage):
                dataset = load_data()
                self.train_data, self.val_data = split(dataset)

            def train_dataloader(self):
                return paddle.io.DataLoader(self.train_data, batch_size=32)
    """

    prepare_data_per_node: bool = True
    allow_zero_length_dataloader_with_multiple_devices: bool = False

    def __init__(self) -> None:
        self.trainer: Optional["Trainer"] = None  # noqa: F821

    def prepare_data(self) -> None:
        """Download and prepare data. Called only once (on rank 0)."""

    def setup(self, stage: str) -> None:
        """Set up datasets. Called on every process.

        Args:
            stage: ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``.
        """

    def teardown(self, stage: str) -> None:
        """Clean up after the run.

        Args:
            stage: ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``.
        """

    def train_dataloader(self) -> paddle.io.DataLoader:
        """Return the training DataLoader."""
        raise NotImplementedError("train_dataloader must be implemented")

    def val_dataloader(self) -> paddle.io.DataLoader:
        """Return the validation DataLoader."""
        raise NotImplementedError("val_dataloader must be implemented")

    def test_dataloader(self) -> paddle.io.DataLoader:
        """Return the test DataLoader."""
        raise NotImplementedError("test_dataloader must be implemented")

    def predict_dataloader(self) -> paddle.io.DataLoader:
        """Return the predict DataLoader."""
        raise NotImplementedError("predict_dataloader must be implemented")

    def on_exception(self, exception: BaseException) -> None:
        """Called when the trainer faces an exception during fit."""
