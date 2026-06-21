"""ModelCheckpoint callback - saves model checkpoints during training."""

import os
from typing import Any, Optional

import paddle

from ocean.callbacks.callback import Callback


class ModelCheckpoint(Callback):
    """Save model checkpoints during training.

    Args:
        dirpath: Directory to save checkpoints. Default: current working dir.
        filename: Checkpoint filename template.
            Default: ``'{epoch}-{step}'``.
        monitor: Metric to monitor for saving.
        verbose: If True, prints save messages.
        save_last: If True, always saves a 'last.pdparams'.
        save_top_k: Number of best models to keep (k) or -1 for all.
        mode: ``'min'`` or ``'max'`` (direction for monitor comparison).
        save_weights_only: If True, saves only model weights (not optimizer state).
        every_n_epochs: Save checkpoint every N epochs.
        every_n_train_steps: Save checkpoint every N training steps.
    """

    FILE_EXTENSION = ".pdparams"

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: bool = True,
        save_top_k: int = 1,
        mode: str = "min",
        save_weights_only: bool = False,
        every_n_epochs: Optional[int] = None,
        every_n_train_steps: Optional[int] = None,
    ) -> None:
        self.dirpath = dirpath or os.getcwd()
        self.filename = filename or "{epoch}-{step}"
        self.monitor = monitor
        self.verbose = verbose
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.mode = mode
        self.save_weights_only = save_weights_only
        self.every_n_epochs = every_n_epochs
        self.every_n_train_steps = every_n_train_steps

        self.best_k_models: dict[str, float] = {}
        self.best_model_path: str = ""
        self.best_model_score: Optional[float] = None
        self.last_model_path: str = ""
        self.current_score: Optional[float] = None
        self._last_dataloader_step_saved: int = 0
        self._monitor_op = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

        os.makedirs(self.dirpath, exist_ok=True)

    def on_validation_end(self, trainer: Any, model: Any) -> None:
        self._save_if_needed(trainer, model)

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if self.every_n_epochs is not None and trainer.current_epoch % self.every_n_epochs != 0:
            return
        self._save_if_needed(trainer, model)

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.every_n_train_steps is not None:
            if trainer.dataloader_step - self._last_dataloader_step_saved >= self.every_n_train_steps:
                self._save(trainer, model, monitor_candidates={})
                self._last_dataloader_step_saved = trainer.dataloader_step

    def _save_if_needed(self, trainer: Any, model: Any) -> None:
        monitor_candidates = {}
        if self.monitor is not None:
            if self.monitor in trainer._log_metrics_on_epoch:
                self.current_score = trainer._log_metrics_on_epoch[self.monitor]
                monitor_candidates[self.monitor] = self.current_score
        self._save(trainer, model, monitor_candidates)

    def _save(self, trainer: Any, model: Any, monitor_candidates: dict) -> None:
        epoch = trainer.current_epoch + 1
        step = trainer.dataloader_step

        # Save "last" checkpoint
        if self.save_last:
            self._write_checkpoint(model, os.path.join(self.dirpath, "last" + self.FILE_EXTENSION))

        # Save top-k
        if self.monitor is not None and self.current_score is not None:
            score = self.current_score
            ckpt_name = self.filename.format(epoch=epoch, step=step, **monitor_candidates)
            ckpt_path = os.path.join(self.dirpath, ckpt_name + self.FILE_EXTENSION)

            is_better = self._monitor_op(score, self.best_model_score) if self.best_model_score is not None else True
            if is_better:
                self.best_model_score = score
                self.best_model_path = ckpt_path
                self._write_checkpoint(model, ckpt_path)
                if self.verbose:
                    print(f"ModelCheckpoint: saved '{ckpt_path}' (score={score:.4f})")

                # Manage top-k
                self.best_k_models[ckpt_path] = score
                if self.save_top_k > 0:
                    sorted_paths = sorted(self.best_k_models.items(), key=lambda x: x[1], reverse=(self.mode == "max"))
                    for path, _ in sorted_paths[self.save_top_k :]:
                        if os.path.exists(path):
                            os.remove(path)
                        self.best_k_models.pop(path, None)

    def _write_checkpoint(self, model: Any, path: str) -> None:
        if self.save_weights_only:
            state = model.state_dict()
            if hasattr(state, "items"):
                state = {k: v for k, v in state.items()}
            paddle.save(state, path)
        else:
            checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": model._trainer.current_epoch if model._trainer else 0,
                "dataloader_step": model._trainer.dataloader_step if model._trainer else 0,
                "optimizer_step": model._trainer.optimizer_step if model._trainer else 0,
            }
            if hasattr(model, "_optimizer") and model._optimizer is not None:
                checkpoint["optimizer"] = model._optimizer.state_dict()
            paddle.save(checkpoint, path)

    def state_dict(self) -> dict:
        return {
            "best_k_models": self.best_k_models,
            "best_model_path": self.best_model_path,
            "best_model_score": self.best_model_score,
            "last_model_path": self.last_model_path,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.best_k_models = state_dict.get("best_k_models", {})
        self.best_model_path = state_dict.get("best_model_path", "")
        self.best_model_score = state_dict.get("best_model_score", None)
        self.last_model_path = state_dict.get("last_model_path", "")
