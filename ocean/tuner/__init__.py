"""Tuner — batch size finder and learning rate finder.

Analogous to Lightning's Tuner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import paddle


class Tuner:
    """Tuner for finding optimal batch size and learning rate."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def tune_batch_size(
        self,
        model: Any,
        train_dataloader: Any,
        min_batch_size: int = 2,
        max_batch_size: int = 512,
        steps_per_trial: int = 3,
    ) -> int:
        """Binary-search max batch size that fits in GPU memory."""
        device = self.trainer._resolve_device()
        model.to(device)
        model.train()

        # Get a single sample to determine input shape
        for batch in train_dataloader:
            sample = batch
            break
        ref = sample[0] if isinstance(sample, (list, tuple)) else sample
        if not hasattr(ref, "shape"):
            return min_batch_size
        feat_dim = ref.shape[1:]

        low, high = min_batch_size, max_batch_size
        best = min_batch_size

        while low <= high:
            mid = (low + high) // 2
            ok = self._try_batch_size(model, feat_dim, mid, steps_per_trial, device, sample)
            if ok:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _try_batch_size(self, model: Any, feat_dim, bs: int, steps: int, device, sample) -> bool:
        try:
            test_data = paddle.randn([bs, *feat_dim])
            test_batch = [test_data] if isinstance(sample, list) else test_data
            test_batch = self.trainer._move_to_device(test_batch, device)
            for _ in range(steps):
                loss = model.training_step(test_batch, 0)
                if isinstance(loss, dict):
                    loss = loss.get("loss", paddle.to_tensor(0.0))
                loss.backward()
                if self.trainer._optimizers:
                    self.trainer._optimizers[0]._optimizer.step()
                    self.trainer._optimizers[0]._optimizer.clear_grad()
            return True
        except Exception as e:
            s = str(e).lower()
            if "memory" in s or "cuda" in s or "oom" in s or "allocate" in s:
                paddle.device.cuda.empty_cache()
                return False
            return True

    def lr_find(
        self,
        model: Any,
        train_dataloader: Any,
        min_lr: float = 1e-8,
        max_lr: float = 1.0,
        num_steps: int = 100,
    ) -> float:
        """Exponentially increase LR, find steepest descent point."""
        if not self.trainer._optimizers:
            raise ValueError("No optimizer configured.")

        opt = self.trainer._optimizers[0]._optimizer
        orig_lr = float(opt.get_lr())
        device = self.trainer._resolve_device()
        model.train()

        losses, lrs = [], []
        batch_iter = iter(train_dataloader)

        for step in range(num_steps):
            progress = step / max(num_steps - 1, 1)
            lr = min_lr * (max_lr / min_lr) ** progress
            opt.set_lr(lr)

            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_dataloader)
                batch = next(batch_iter)

            batch = self.trainer._move_to_device(batch, device)
            loss = model.training_step(batch, step)
            if isinstance(loss, dict):
                loss = loss.get("loss", paddle.to_tensor(0.0))
            loss.backward()
            opt.step()
            opt.clear_grad()

            loss_val = float(loss.numpy()) if hasattr(loss, "numpy") else float(loss)
            losses.append(loss_val)
            lrs.append(lr)

        opt.set_lr(orig_lr)

        if len(losses) < 3:
            return orig_lr

        smoothed = np.convolve(losses, np.ones(3) / 3, mode="valid")
        min_idx = int(np.argmin(smoothed)) + 1
        return lrs[min(min_idx, len(lrs) - 1)]
