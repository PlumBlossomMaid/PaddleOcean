#!/usr/bin/env python3
"""MNIST image classification with Ocean framework.

Usage:
    python ocean/examples/image_classifier.py              # CPU
    python ocean/examples/image_classifier.py --gpu        # single GPU

This script demonstrates a complete Ocean training pipeline:
  - Real data (MNIST) with transforms
  - Small CNN model
  - Training, validation, and testing
  - VisualDL logging
  - Checkpoint saving
"""

import argparse
import os
import time

import paddle
import paddle.vision.transforms as transforms
from paddle.vision.datasets import MNIST

import ocean


# ════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════
class MNISTClassifier(ocean.Model):
    """Simple CNN for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(1, 16, 3, padding=1)
        self.conv2 = paddle.nn.Conv2D(16, 32, 3, padding=1)
        self.pool = paddle.nn.MaxPool2D(2)
        self.fc = paddle.nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(paddle.nn.functional.relu(self.conv1(x)))
        x = self.pool(paddle.nn.functional.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = paddle.nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(axis=1) == y).astype(paddle.float32).mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = paddle.nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(axis=1) == y).astype(paddle.float32).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(axis=1) == y).astype(paddle.float32).mean()
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return paddle.optimizer.Adam(learning_rate=0.001, parameters=self.parameters())


# ════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════
def make_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = MNIST(mode="train", transform=transform, backend="cv2")
    test_dataset = MNIST(mode="test", transform=transform, backend="cv2")

    # Split test into val and test
    val_size = 1000
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = paddle.io.random_split(test_dataset, [val_size, test_size])

    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="MNIST image classifier with Ocean")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--log_dir", type=str, default="ocean_demo_logs", help="Log directory")
    args = parser.parse_args()

    ocean.seed_everything(42)

    accelerator = "gpu" if args.gpu and paddle.is_compiled_with_cuda() else "cpu"
    print(f"Using accelerator: {accelerator}")

    # ── Logger ──
    logger = ocean.VisualDLLogger(save_dir=args.log_dir, name="mnist")

    # ── Trainer ──
    trainer = ocean.Trainer(
        accelerator=accelerator,
        devices=1 if accelerator == "gpu" else "auto",
        max_epochs=args.epochs,
        log_every_n_steps=50,
        logger=logger,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir=args.log_dir,
    )

    # ── Model ──
    model = MNISTClassifier()

    # ── Data ──
    train_loader, val_loader, test_loader = make_dataloaders(batch_size=args.batch_size)

    # ── Fit ──
    start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    # ── Test ──
    results = trainer.test(model, dataloaders=test_loader)
    if results:
        for r in results:
            for k, v in r.items():
                print(f"  {k}: {v:.4f}")

    # ── Checkpoint ──
    ckpt_path = os.path.join(args.log_dir, "mnist_final.pdparams")
    trainer.save_checkpoint(ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
