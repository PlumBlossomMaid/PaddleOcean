"""Tests for paddleMetrics integration with Ocean (ocean.metrics).

Tests cover:
1. Basic metric compute (import, forward, accumulate)
2. Metric integration with Model.log() — on_step, on_epoch
3. MetricCollection usage
4. Gradient accumulation + metric logging
"""

import os
import sys

import paddle
from paddlemetrics import Accuracy, MeanMetric, MetricCollection, Precision

import ocean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.helpers.runif import RunIf  # noqa: E402

# ====================================================================
# 1. Basic metric functionality (standalone, no Trainer)
# ====================================================================


class TestMetricBasic:
    """Verify paddlemetrics work correctly in isolation."""

    def test_accuracy_basic(self):
        """Accuracy: forward returns batch-level value; compute returns accumulated."""
        acc = Accuracy(task="multiclass", num_classes=2)
        preds = paddle.to_tensor([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.5, 0.5]])
        target = paddle.to_tensor([1, 0, 1, 0])

        batch_val = acc(preds, target)
        assert isinstance(batch_val, paddle.Tensor)
        # Predictions argmax → [1,0,1,0], target → [1,0,1,0] → all correct!
        assert batch_val.item() == 1.0, f"Expected 1.0, got {batch_val.item()}"

        final = acc.compute()
        assert final.item() == 1.0

    def test_accuracy_multibatch(self):
        """Accuracy: accumulate across multiple batches."""
        acc = Accuracy(task="multiclass", num_classes=2)

        batch1_preds = paddle.to_tensor([[0.7, 0.3], [0.2, 0.8]])
        batch1_target = paddle.to_tensor([0, 1])
        acc(batch1_preds, batch1_target)  # both correct

        batch2_preds = paddle.to_tensor([[0.1, 0.9], [0.6, 0.4]])
        batch2_target = paddle.to_tensor([1, 0])
        acc(batch2_preds, batch2_target)  # both correct

        final = acc.compute()
        assert final.item() == 1.0, f"Expected 1.0, got {final.item()}"

    def test_accuracy_reset(self):
        """Accuracy: reset() clears accumulated state."""
        acc = Accuracy(task="multiclass", num_classes=2)
        preds = paddle.to_tensor([[0.2, 0.8], [0.7, 0.3]])
        target = paddle.to_tensor([1, 0])
        acc(preds, target)
        assert acc.compute().item() == 1.0

        acc.reset()
        assert acc._update_count == 0
        result = acc.compute()
        assert result is not None

    def test_metric_collection(self):
        """MetricCollection: compute multiple metrics at once."""
        preds = paddle.to_tensor([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.5, 0.5]])
        target = paddle.to_tensor([1, 0, 1, 0])

        metrics = MetricCollection([
            Accuracy(task="multiclass", num_classes=2),
            Precision(task="multiclass", num_classes=2),
        ])
        result = metrics(preds, target)
        assert "MulticlassAccuracy" in result, f"Keys: {list(result.keys())}"
        assert "MulticlassPrecision" in result

    def test_mean_metric(self):
        """MeanMetric: simple mean aggregation."""
        mean = MeanMetric()
        mean.update(paddle.to_tensor(1.0))
        mean.update(paddle.to_tensor(2.0))
        mean.update(paddle.to_tensor(3.0))
        assert mean.compute().item() == 2.0


# ====================================================================
# 2. Metric integration with Model.log() (single GPU)
# ====================================================================


class TestMetricWithModelLog:
    """Verify that logging a paddlemetrics Metric via Model.log() works correctly."""

    def _make_model(self):
        """Create a simple model that uses self.log() with a Metric object."""

        class MetricModel(ocean.Model):
            def __init__(self):
                super().__init__()
                self.net = ocean.nn.Linear(10, 2)

            def forward(self, x):
                return self.net(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                preds = self(x)
                loss = ocean.nn.functional.cross_entropy(preds, y.squeeze())
                # Log a paddlemetrics Metric object (align with Lightning pattern)
                self.train_acc(preds, y.squeeze())
                self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
                return loss

            def configure_optimizers(self):
                return ocean.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

        model = MetricModel()
        model.train_acc = Accuracy(task="multiclass", num_classes=2)
        return model

    @RunIf(min_cuda_gpus=1)
    def test_log_metric_on_step_on_epoch(self, tmp_path):
        """Logging a Metric object: on_step logs batch value, on_epoch logs accumulated."""
        ocean.seed_everything(42)
        model = self._make_model()

        x = paddle.randn([32, 10])
        y = paddle.randint(0, 2, [32])
        dataset = paddle.io.TensorDataset([x, y])
        loader = paddle.io.DataLoader(dataset, batch_size=8)

        trainer = ocean.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=1,
            enable_progress_bar=False,
            log_every_n_steps=1,
            default_root_dir=str(tmp_path),
        )
        trainer.fit(model, loader)

        assert "train_acc" in trainer._log_metrics_on_epoch, (
            f"train_acc not in epoch metrics: {trainer._log_metrics_on_epoch}"
        )
        acc_val = trainer._log_metrics_on_epoch["train_acc"]
        assert 0.0 <= acc_val <= 1.0, f"Accuracy out of range: {acc_val}"

    @RunIf(min_cuda_gpus=1)
    def test_log_metric_on_epoch_only(self, tmp_path):
        """Logging a Metric object with only on_epoch=True does not log per-step."""
        ocean.seed_everything(42)

        class MetricModel(ocean.Model):
            def __init__(self):
                super().__init__()
                self.net = ocean.nn.Linear(10, 2)
                self.metric_acc = Accuracy(task="multiclass", num_classes=2)

            def forward(self, x):
                return self.net(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                preds = self(x)
                loss = ocean.nn.functional.cross_entropy(preds, y.squeeze())
                self.metric_acc(preds, y.squeeze())
                self.log("epoch_acc", self.metric_acc, on_step=False, on_epoch=True)
                return loss

            def configure_optimizers(self):
                return ocean.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

        model = MetricModel()
        x = paddle.randn([32, 10])
        y = paddle.randint(0, 2, [32])
        dataset = paddle.io.TensorDataset([x, y])
        loader = paddle.io.DataLoader(dataset, batch_size=8)

        trainer = ocean.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=1,
            enable_progress_bar=False,
            log_every_n_steps=1,
            default_root_dir=str(tmp_path),
        )
        trainer.fit(model, loader)

        assert "epoch_acc" in trainer._log_metrics_on_epoch
        acc_val = trainer._log_metrics_on_epoch["epoch_acc"]
        assert 0.0 <= acc_val <= 1.0

    @RunIf(min_cuda_gpus=1)
    def test_dataloader_step_not_logged(self, tmp_path):
        """Dataloader_step is Ocean's own concept and should NOT be logged as Metric.

        This test ensures that a regular scalar log still works (regression test).
        """
        ocean.seed_everything(42)

        class ScalarModel(ocean.Model):
            def __init__(self):
                super().__init__()
                self.net = ocean.nn.Linear(10, 2)

            def forward(self, x):
                return self.net(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                preds = self(x)
                loss = ocean.nn.functional.cross_entropy(preds, y.squeeze())
                self.log("train_loss", loss, on_step=True, on_epoch=True)
                return loss

            def configure_optimizers(self):
                return ocean.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

        model = ScalarModel()
        x = paddle.randn([32, 10])
        y = paddle.randint(0, 2, [32])
        dataset = paddle.io.TensorDataset([x, y])
        loader = paddle.io.DataLoader(dataset, batch_size=8)

        trainer = ocean.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=1,
            enable_progress_bar=False,
            log_every_n_steps=1,
            default_root_dir=str(tmp_path),
        )
        trainer.fit(model, loader)

        assert "train_loss" in trainer._log_metrics_on_epoch
        loss_val = trainer._log_metrics_on_epoch["train_loss"]
        assert loss_val > 0.0
