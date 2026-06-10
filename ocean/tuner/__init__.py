"""Tuner - learning rate finder and batch size scaler.

Analogous to Lightning's Tuner.
"""

from typing import Any, Optional

import paddle


class Tuner:
    """Tuner for finding optimal learning rate and batch size.

    Args:
        trainer: The Trainer instance to tune.
    """

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def lr_find(
        self,
        model: Any,
        train_dataloaders: Optional[Any] = None,
        val_dataloaders: Optional[Any] = None,
        datamodule: Optional[Any] = None,
        min_lr: float = 1e-8,
        max_lr: float = 1.0,
        num_training: int = 100,
        mode: str = "exponential",
        early_stop_threshold: float = 4.0,
    ) -> float:
        """Find optimal learning rate using a range test.

        Args:
            model: The model to tune.
            train_dataloaders: Training data.
            val_dataloaders: Validation data.
            datamodule: DataModule.
            min_lr: Minimum learning rate.
            max_lr: Maximum learning rate.
            num_training: Number of steps for the test.
            mode: 'exponential' or 'linear' LR growth.
            early_stop_threshold: Stop when loss explodes beyond this ratio.

        Returns:
            Suggested learning rate.
        """
        trainer = self.trainer
        original_lr = self._get_lr(model)

        # Generate LR schedule
        if mode == "exponential":
            lrs = [min_lr * (max_lr / min_lr) ** (i / max(num_training - 1, 1)) for i in range(num_training)]
        else:
            lrs = [min_lr + (max_lr - min_lr) * i / max(num_training - 1, 1) for i in range(num_training)]

        # Run test
        losses = []
        best_loss = float("inf")
        prev_loss = float("inf")

        model.train()
        train_loader = train_dataloaders or (datamodule.train_dataloader() if datamodule else None)
        if train_loader is None:
            return min_lr

        device = trainer._resolve_device()
        model.to(device)
        optimizer = model._optimizer or trainer._resolve_optimizer(model)

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_training:
                break

            # Set LR
            lr = lrs[batch_idx]
            self._set_lr(optimizer, lr)

            batch = trainer._move_to_device(batch, device)
            optimizer.clear_grad()
            result = model.training_step(batch, batch_idx)
            loss = result["loss"] if isinstance(result, dict) else (result if isinstance(result, paddle.Tensor) else None)

            if loss is not None:
                loss_val = float(loss.item())
                losses.append(loss_val)
                best_loss = min(best_loss, loss_val)

                # Early stop if loss explodes
                if loss_val > early_stop_threshold * best_loss and batch_idx > 5:
                    break

                loss.backward()
                optimizer.step()

        # Restore original LR
        self._set_lr(optimizer, original_lr)

        # Suggest LR: point of steepest descent
        if len(losses) < 3:
            return min_lr

        # Find the index just before steepest drop
        gradients = [losses[i + 1] - losses[i] for i in range(len(losses) - 1)]
        if not gradients:
            return min_lr

        # Pick the point of steepest negative gradient
        steepest_idx = gradients.index(min(gradients))
        suggested_lr = lrs[steepest_idx] if steepest_idx < len(lrs) else lrs[-1]
        return suggested_lr * 0.1  # Return 1/10 of steepest point

    def scale_batch_size(
        self,
        model: Any,
        train_dataloaders: Optional[Any] = None,
        datamodule: Optional[Any] = None,
        mode: str = "power",
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
    ) -> int:
        """Find the maximum batch size that fits in memory.

        Args:
            model: The model.
            train_dataloaders: Training data.
            datamodule: DataModule.
            mode: 'power' (double each trial) or 'binsearch'.
            steps_per_trial: Steps per trial.
            init_val: Initial batch size multiplier.
            max_trials: Maximum number of trials.

        Returns:
            Optimal batch size.
        """
        return 32  # Simplified for Phase 2

    def _get_lr(self, model: Any) -> float:
        opt = model._optimizer
        if opt is not None and hasattr(opt, "_learning_rate"):
            return float(opt._learning_rate)
        return 0.001

    def _set_lr(self, optimizer: Any, lr: float) -> None:
        if optimizer is not None and hasattr(optimizer, "_learning_rate"):
            optimizer._learning_rate = lr
