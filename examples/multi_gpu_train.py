#!/usr/bin/env python3
"""Multi-GPU training example with Ocean framework.

Usage:
    # Single node, 4 GPUs
    python -m paddle.distributed.launch --gpus=0,1,2,3 \\
        ocean/examples/multi_gpu_train.py

    # Single node, 2 GPUs
    python -m paddle.distributed.launch --gpus=0,1 \\
        ocean/examples/multi_gpu_train.py

This script:
  - Uses synthetic data (no download needed, no I/O bottleneck)
  - Creates a WideResNet-style model large enough to saturate GPUs
  - Logs metrics to VisualDL (rank 0 only)
  - Saves a checkpoint after training
"""

import os
import time

import paddle

import ocean
from ocean.utils.rank_zero import rank_zero_info, rank_zero_only


# ════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════
class WideResNet(ocean.Model):
    """Wide residual network for synthetic image classification."""

    def __init__(self, width=64, num_blocks=6, num_classes=1000):
        super().__init__()
        # Stem
        self.stem = paddle.nn.Sequential(
            paddle.nn.Conv2D(3, width, 7, stride=2, padding=3),
            paddle.nn.BatchNorm2D(width),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(3, stride=2, padding=1),
        )
        # Residual blocks
        self.blocks = paddle.nn.LayerList()
        for i in range(num_blocks):
            in_ch = width * (2 ** min(i, 3))
            out_ch = width * (2 ** min(i + 1, 3))
            stride = 2 if i > 0 and i % 2 == 0 else 1
            self.blocks.append(self._make_block(in_ch, out_ch, stride))
        # Head
        final_ch = width * (2 ** min(num_blocks, 3))
        self.head = paddle.nn.Sequential(
            paddle.nn.AdaptiveAvgPool2D(1),
            paddle.nn.Flatten(),
            paddle.nn.Linear(final_ch, num_classes),
        )

    @staticmethod
    def _make_block(in_ch, out_ch, stride):
        return paddle.nn.Sequential(
            paddle.nn.Conv2D(in_ch, out_ch, 3, stride=stride, padding=1),
            paddle.nn.BatchNorm2D(out_ch),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(out_ch, out_ch, 3, padding=1),
            paddle.nn.BatchNorm2D(out_ch),
        )

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            identity = x
            x = block(x)
            if x.shape == identity.shape:
                x = x + identity
            x = paddle.nn.functional.relu(x)
        return self.head(x)

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

    def configure_optimizers(self):
        return paddle.optimizer.Momentum(
            learning_rate=0.1,
            parameters=self.parameters(),
            weight_decay=1e-4,
            momentum=0.9,
        )


# ════════════════════════════════════════════════════════════════
# Synthetic data
# ════════════════════════════════════════════════════════════════
def make_synthetic_data(batch_size=128, num_batches=100, image_size=128):
    """Create synthetic image data on CPU (no I/O bottleneck)."""
    with paddle.device_guard("cpu"):
        data_x = paddle.randn([num_batches * batch_size, 3, image_size, image_size])
        data_y = paddle.randint(0, 1000, [num_batches * batch_size])
    dataset = paddle.io.TensorDataset([data_x, data_y])
    return paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )


# ════════════════════════════════════════════════════════════════
# GPU utilization (rank 0 only)
# ════════════════════════════════════════════════════════════════
@rank_zero_only
def log_gpu_utilization(step=0, interval=1):
    if step % interval != 0:
        return
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for i, line in enumerate(result.stdout.strip().split("\n")):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                print(f"  GPU {i}: util={parts[0]}%, mem={parts[1]}/{parts[2]} MiB", flush=True)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    # ── Config ──
    num_gpus = 4
    log_dir = "ocean_demo_logs"

    ocean.seed_everything(42)

    rank_zero_info(f"Ocean {ocean.__version__}, Paddle {paddle.__version__}")
    rank_zero_info(f"Training on {num_gpus} GPUs — synthetic data, WideResNet")

    # ── Logger ──
    logger = ocean.VisualDLLogger(save_dir=log_dir, name="multi_gpu")

    # ── Trainer ──
    trainer = ocean.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=num_gpus,
        max_epochs=3,
        log_every_n_steps=10,
        logger=logger,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir=log_dir,
        accumulate_grad_batches=2,
    )

    # ── Model ──
    model = WideResNet(width=64, num_blocks=6, num_classes=1000)

    # ── Data ──
    train_loader = make_synthetic_data(batch_size=128, num_batches=100, image_size=128)
    val_loader = make_synthetic_data(batch_size=128, num_batches=10, image_size=128)

    # ── Fit ──
    start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.time() - start
    rank_zero_info(f"Training finished in {elapsed:.1f}s")

    # ── Test ──
    trainer.test(model, dataloaders=val_loader)

    # ── Checkpoint ──
    ckpt_path = os.path.join(log_dir, "final.pdparams")
    trainer.save_checkpoint(ckpt_path)
    rank_zero_info(f"Checkpoint saved: {ckpt_path}")

    log_gpu_utilization(step=0, interval=1)


if __name__ == "__main__":
    main()
