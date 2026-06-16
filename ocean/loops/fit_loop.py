"""_FitLoop - orchestrates training across epochs.

Uses direct DataLoader iteration to avoid PaddlePaddle shared memory issues.
"""

from typing import Any, Optional

import paddle

from ocean.loops.loop import _Loop
from ocean.trainer.call import _call_callback_hooks, _call_lightning_module_hook


class _FitLoop(_Loop):
    """Top-level training loop that iterates over epochs."""

    def __init__(self, trainer: Any, min_epochs: int = 0, max_epochs: Optional[int] = None) -> None:
        super().__init__(trainer)
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs or 1000

    @property
    def done(self) -> bool:
        trainer = self.trainer
        if trainer.current_epoch >= self.max_epochs:
            return True
        if trainer.should_stop and trainer.current_epoch >= self.min_epochs:
            return True
        return False

    def run(self) -> None:
        trainer = self.trainer
        model = trainer._model
        train_loader = getattr(trainer, "train_dataloader", None)
        if train_loader is None:
            return

        # On train start
        _call_lightning_module_hook(trainer, "on_train_start")
        _call_callback_hooks(trainer, "on_train_start")

        device = trainer._resolve_device()

        while not self.done:
            # On epoch start
            _call_lightning_module_hook(trainer, "on_train_epoch_start")
            _call_callback_hooks(trainer, "on_train_epoch_start")

            # Reset optimizer accumulation
            opt_acc = 0

            for batch_idx, batch in enumerate(train_loader):
                if trainer._should_limit_batches(batch_idx, "train"):
                    break

                batch = trainer._move_to_device(batch, device)

                skip = model.on_train_batch_start(batch, batch_idx)
                if skip == -1:
                    continue

                # Training step
                result = model.training_step(batch, batch_idx)

                # Skip automatic backward/optimizer when manual optimization is used
                # (model handles backward and step inside training_step)
                if model.automatic_optimization:
                    loss = (
                        result["loss"]
                        if isinstance(result, dict)
                        else (result if isinstance(result, paddle.Tensor) else None)
                    )

                    if loss is not None:
                        model.on_before_backward(loss)
                        loss = loss / max(1, trainer.accumulate_grad_batches)
                        loss.backward()
                        model.on_after_backward()
                        opt_acc += 1

                        if opt_acc >= trainer.accumulate_grad_batches:
                            if trainer.gradient_clip_val is not None and trainer.gradient_clip_val > 0:
                                if trainer.gradient_clip_algorithm == "norm":
                                    paddle.nn.utils.clip_grad_norm_(model.parameters(), trainer.gradient_clip_val)
                                elif trainer.gradient_clip_algorithm == "value":
                                    paddle.nn.utils.clip_grad_value_(model.parameters(), trainer.gradient_clip_val)
                            model.on_before_optimizer_step(trainer._optimizer)
                            trainer._optimizers[0].step()
                            trainer._optimizers[0]._optimizer.clear_grad()
                            opt_acc = 0
                            trainer._dataloader_step += 1
                else:
                    # Manual optimization: model handles backward/step, just count steps
                    trainer._dataloader_step += 1

                model.on_train_batch_end(result, batch, batch_idx)

                # TQDMProgressBar handles progress display

                if 0 < trainer.max_steps <= trainer.dataloader_step:
                    trainer.should_stop = True
                    break

            # Flush remaining gradients
            if opt_acc > 0 and trainer._optimizer is not None:
                trainer._optimizer.step()
                trainer._optimizer.clear_grad()
                trainer._dataloader_step += 1

            # On epoch end
            trainer._compute_epoch_metrics()
            _call_lightning_module_hook(trainer, "on_train_epoch_end")
            _call_callback_hooks(trainer, "on_train_epoch_end")

            trainer.current_epoch += 1

            # Validation
            if trainer._should_check_val():
                self._run_validation()

            if trainer._should_stop():
                break

        # On train end
        _call_lightning_module_hook(trainer, "on_train_end")
        _call_callback_hooks(trainer, "on_train_end")

    def teardown(self) -> None:
        pass

    def _run_validation(self) -> None:
        trainer = self.trainer
        model = trainer._model
        val_loader = getattr(trainer, "val_dataloaders", None)
        if not val_loader:
            return

        model.on_validation_model_eval()
        _call_lightning_module_hook(trainer, "on_validation_start")
        _call_callback_hooks(trainer, "on_validation_start")
        _call_lightning_module_hook(trainer, "on_validation_epoch_start")

        device = trainer._resolve_device()
        for dataloader in val_loader if isinstance(val_loader, (list, tuple)) else [val_loader]:
            with paddle.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if trainer._should_limit_batches(batch_idx, "val"):
                        break
                    batch = trainer._move_to_device(batch, device)
                    model.on_validation_batch_start(batch, batch_idx)
                    result = model.validation_step(batch, batch_idx)
                    model.on_validation_batch_end(result, batch, batch_idx)

        trainer._compute_epoch_metrics()
        _call_lightning_module_hook(trainer, "on_validation_epoch_end")
        _call_callback_hooks(trainer, "on_validation_end")
        _call_lightning_module_hook(trainer, "on_validation_end")
        model.on_validation_model_train()
