"""Tests for dataloader_step / optimizer_step partitioning.

Verifies the step counter architecture is correct.
"""

from __future__ import annotations

import paddle
import paddle.nn as nn

import ocean


class SimpleModel(ocean.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2)
        self.losses = []

    def configure_optimizers(self):
        return paddle.optimizer.SGD(0.01, parameters=self.net.parameters())

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, list) else batch
        loss = self.net(x).mean()
        self.losses.append(loss.item())
        return loss


class TestStepCounters:
    """Test step counter partitioning."""

    def test_initial_state(self, tmp_path):
        """Counters start at zero."""
        trainer = ocean.Trainer(
            default_root_dir=str(tmp_path),
            max_steps=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        assert trainer.dataloader_step == 0
        assert trainer.optimizer_step == 0

    def test_dataloader_step_counts_batches(self, tmp_path):
        """dataloader_step = logical batches completed."""
        model = SimpleModel()
        loader = paddle.io.DataLoader(
            paddle.io.TensorDataset(paddle.randn([30, 4])),
            batch_size=10,
        )
        trainer = ocean.Trainer(
            default_root_dir=str(tmp_path),
            max_steps=5,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)
        assert trainer.dataloader_step == 5

    def test_max_stops_at_limit(self, tmp_path):
        """max_steps limits dataloader_step."""
        model = SimpleModel()
        loader = paddle.io.DataLoader(
            paddle.io.TensorDataset(paddle.randn([100, 4])),
            batch_size=10,
        )
        trainer = ocean.Trainer(
            default_root_dir=str(tmp_path),
            max_steps=7,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)
        assert trainer.dataloader_step == 7

    def test_optimizer_step_increments_with_dataloader(self, tmp_path):
        """optimizer_step == dataloader_step in automatic mode."""
        model = SimpleModel()
        loader = paddle.io.DataLoader(
            paddle.io.TensorDataset(paddle.randn([20, 4])),
            batch_size=10,
        )
        trainer = ocean.Trainer(
            default_root_dir=str(tmp_path),
            max_steps=3,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)
        # Each logical batch = 1 optimizer step in automatic mode
        assert trainer.optimizer_step == 3

    def test_manual_optimizer_can_be_set(self):
        """Direct _optimizer_step access works for manual GAN."""
        trainer = ocean.Trainer(
            default_root_dir="/tmp/_t",
            max_steps=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer._optimizer_step += 2  # G + D two steps per logical batch
        assert trainer.optimizer_step == 2
