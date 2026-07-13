"""Tuner — batch size finder and learning rate finder.

Utilities for learning-rate and batch-size tuning. Both finders run real
forward/backward/step passes, so each finder snapshots the model and optimizer
state before it starts and restores it afterwards: tuning must never leave the
model with corrupted weights or the optimizer with polluted momentum.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import paddle


def _clone_state(obj: Any) -> Any:
    """Recursively clone a state dict, copying tensors so later in-place training
    steps cannot mutate the snapshot."""
    if isinstance(obj, paddle.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _clone_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone_state(v) for v in obj)
    return obj


def _is_oom_error(exc: Exception) -> bool:
    """Whether an exception looks like an out-of-memory failure."""
    s = str(exc).lower()
    return any(tok in s for tok in ("out of memory", "memory", "oom", "allocate", "resourceexhausted"))


def _release_memory() -> None:
    """Best-effort device cache release; a no-op if the device has no such API."""
    try:
        paddle.device.cuda.empty_cache()
    except Exception:
        pass


class Tuner:
    """Tuner for finding optimal batch size and learning rate."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    # ------------------------------------------------------------------
    # State save / restore
    # ------------------------------------------------------------------
    def _snapshot_state(self, model: Any) -> dict:
        """Deep-copy model + optimizer state so a tuning run can be rolled back."""
        snapshot: dict = {"model": _clone_state(model.state_dict()), "optimizers": [], "lrs": []}
        for opt in self.trainer.optimizers:
            raw = opt._optimizer
            snapshot["optimizers"].append(_clone_state(raw.state_dict()))
            try:
                snapshot["lrs"].append(float(raw.get_lr()))
            except Exception:
                snapshot["lrs"].append(None)
        return snapshot

    def _restore_state(self, model: Any, snapshot: dict) -> None:
        """Restore the model + optimizer state captured by :meth:`_snapshot_state`."""
        model.set_state_dict(snapshot["model"])
        for opt, opt_state, lr in zip(self.trainer.optimizers, snapshot["optimizers"], snapshot["lrs"]):
            raw = opt._optimizer
            try:
                raw.set_state_dict(opt_state)
            except Exception:
                pass
            if lr is not None:
                try:
                    raw.set_lr(lr)
                except Exception:
                    pass
            # Drop any gradients accumulated during the trials.
            try:
                raw.clear_grad()
            except Exception:
                pass
        model.clear_gradients()

    # ------------------------------------------------------------------
    # Batch size finder
    # ------------------------------------------------------------------
    def tune_batch_size(
        self,
        model: Any,
        train_dataloader: Any,
        min_batch_size: int = 2,
        max_batch_size: int = 512,
        steps_per_trial: int = 3,
    ) -> int:
        """Binary-search the largest batch size that fits in device memory.

        The model and optimizer state are restored on exit so the search leaves
        the model exactly as it was found.
        """
        device = self.trainer._resolve_device()
        model.to(device)
        model.train()

        # Get a single sample to determine input shape
        sample = None
        for batch in train_dataloader:
            sample = batch
            break
        if sample is None:
            return min_batch_size
        ref = sample[0] if isinstance(sample, (list, tuple)) else sample
        if not hasattr(ref, "shape"):
            return min_batch_size
        feat_dim = ref.shape[1:]

        snapshot = self._snapshot_state(model)
        try:
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
        finally:
            self._restore_state(model, snapshot)

    def _try_batch_size(self, model: Any, feat_dim, bs: int, steps: int, device, sample) -> bool:
        """Run ``steps`` train steps at batch size ``bs``; True if it fits in memory.

        Only out-of-memory failures count as "does not fit" (return False). Any
        other exception is a real bug in the model and is re-raised — treating it
        as a successful trial would mask the error and drive the search wrong.
        """
        try:
            test_data = paddle.randn([bs, *feat_dim])
            test_batch = [test_data] if isinstance(sample, list) else test_data
            test_batch = self.trainer._move_to_device(test_batch, device)
            for _ in range(steps):
                loss = model.training_step(test_batch, 0)
                if isinstance(loss, dict):
                    loss = loss.get("loss", paddle.to_tensor(0.0))
                loss.backward()
                if self.trainer.optimizers:
                    self.trainer.optimizers[0]._optimizer.step()
                    self.trainer.optimizers[0]._optimizer.clear_grad()
            return True
        except Exception as exc:
            if _is_oom_error(exc):
                _release_memory()
                return False
            raise

    # ------------------------------------------------------------------
    # Learning-rate finder
    # ------------------------------------------------------------------
    def lr_find(
        self,
        model: Any,
        train_dataloader: Any,
        min_lr: float = 1e-8,
        max_lr: float = 1.0,
        num_steps: int = 100,
    ) -> float:
        """Exponentially increase LR over ``num_steps`` and pick the steepest-descent LR.

        Model and optimizer state are snapshotted and restored, so the diagnostic
        training does not pollute the real weights.
        """
        if not self.trainer.optimizers:
            raise ValueError("No optimizer configured.")

        opt = self.trainer.optimizers[0]._optimizer
        device = self.trainer._resolve_device()
        model.train()

        snapshot = self._snapshot_state(model)
        losses, lrs = [], []
        batch_iter = iter(train_dataloader)
        try:
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
        finally:
            self._restore_state(model, snapshot)

        if len(losses) < 3:
            return float(lrs[0]) if lrs else min_lr

        smoothed = np.convolve(losses, np.ones(3) / 3, mode="valid")
        min_idx = int(np.argmin(smoothed)) + 1
        return lrs[min(min_idx, len(lrs) - 1)]
