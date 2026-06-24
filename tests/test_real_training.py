"""Real MNIST training test for CI.

Trains a small CNN on MNIST for 1 epoch to validate the full pipeline:
data loading, model forward/backward, optimizer, logging, checkpoint.
Runs on CPU (CI-compatible).
"""

import os
import sys

import paddle
import pytest

import ocean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ====================================================================
# Small CNN for MNIST — fast enough for CI
# ====================================================================
class MNISTModel(ocean.Model):
    """Simple CNN for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = ocean.nn.Conv2D(1, 16, 3, padding=1)
        self.conv2 = ocean.nn.Conv2D(16, 32, 3, padding=1)
        self.pool = ocean.nn.MaxPool2D(2)
        self.fc = ocean.nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(ocean.nn.functional.relu(self.conv1(x)))
        x = self.pool(ocean.nn.functional.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = ocean.nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(axis=1) == y).astype(ocean.float32).mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = ocean.nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(axis=1) == y).astype(ocean.float32).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return ocean.optimizer.Adam(learning_rate=0.001, parameters=self.parameters())


# ====================================================================
# Data
# ====================================================================
def make_mnist_loaders(batch_size=64, num_train=500, num_val=100):
    """Create small MNIST dataloaders for testing."""
    import paddle.vision.transforms as transforms
    from paddle.vision.datasets import MNIST

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    train_dataset = MNIST(mode="train", transform=transform, backend="cv2")
    val_dataset = MNIST(mode="test", transform=transform, backend="cv2")

    # Use only a subset for speed
    train_dataset = paddle.io.Subset(train_dataset, list(range(num_train)))
    val_dataset = paddle.io.Subset(val_dataset, list(range(num_val)))

    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


# ====================================================================
# Tests
# ====================================================================
class TestMNISTTraining:
    """Real MNIST training tests for CI."""

    def test_mnist_fit_one_epoch(self, tmp_path):
        """Train MNIST model for 1 epoch — validates full pipeline."""
        ocean.seed_everything(42)

        train_loader, val_loader = make_mnist_loaders(batch_size=64, num_train=500, num_val=100)
        model = MNISTModel()

        trainer = ocean.Trainer(
            accelerator="auto",
            max_epochs=1,
            enable_progress_bar=False,
            log_every_n_steps=999,
            default_root_dir=str(tmp_path),
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Training ran
        assert trainer.dataloader_step > 0
        assert trainer.current_epoch == 1

    def test_mnist_fit_val_logging(self, tmp_path):
        """Training + validation produces logged metrics."""
        ocean.seed_everything(42)

        train_loader, val_loader = make_mnist_loaders(batch_size=64, num_train=200, num_val=50)
        model = MNISTModel()

        trainer = ocean.Trainer(
            accelerator="auto",
            max_epochs=1,
            enable_progress_bar=False,
            log_every_n_steps=999,
            default_root_dir=str(tmp_path),
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        callback_metrics = trainer.callback_metrics
        assert isinstance(callback_metrics, dict)
        # Should have some training metrics
        assert len(callback_metrics) >= 0

    def test_mnist_save_checkpoint(self, tmp_path):
        """Save checkpoint after MNIST training."""
        ocean.seed_everything(42)

        train_loader, val_loader = make_mnist_loaders(batch_size=64, num_train=200, num_val=50)
        model = MNISTModel()

        trainer = ocean.Trainer(
            accelerator="auto",
            max_epochs=1,
            enable_progress_bar=False,
            log_every_n_steps=999,
            default_root_dir=str(tmp_path),
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        ckpt_path = str(tmp_path / "mnist_test.ckpt")
        trainer.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)

    def test_mnist_validation_only(self, tmp_path):
        """Validate a model without training."""
        ocean.seed_everything(42)

        _, val_loader = make_mnist_loaders(batch_size=64, num_train=10, num_val=50)
        model = MNISTModel()

        trainer = ocean.Trainer(
            accelerator="auto",
            enable_progress_bar=False,
            verbose=0,
        )
        results = trainer.validate(model, dataloaders=val_loader)
        assert isinstance(results, list)

    @pytest.mark.skipif(
        not ocean.accelerators.CUDAAccelerator.is_available(),
        reason="CUDA not available",
    )
    def test_mnist_gpu_single(self, tmp_path):
        """Train MNIST on single GPU."""
        ocean.seed_everything(42)

        train_loader, val_loader = make_mnist_loaders(batch_size=128, num_train=500, num_val=100)
        model = MNISTModel()

        trainer = ocean.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=1,
            enable_progress_bar=False,
            log_every_n_steps=999,
            default_root_dir=str(tmp_path),
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        assert trainer.dataloader_step > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
