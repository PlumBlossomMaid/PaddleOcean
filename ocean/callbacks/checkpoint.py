"""ModelCheckpoint callback - saves model checkpoints during training."""

import os
from typing import Any, Optional

import paddle

import ocean
from ocean.callbacks.callback import Callback


class ModelCheckpoint(Callback):
    """Save model checkpoints during training.

    Top-k semantics keep the best ``save_top_k`` models by the monitored
    metric (or all of them when ``save_top_k == -1``): a candidate is saved
    whenever it would belong to the current top-k set, not only when it beats
    the running best, so a "worse-than-best but better-than-kth" candidate is
    stored and the displaced kth checkpoint removed.

    Args:
        dirpath: Directory to save checkpoints. Default: current working dir.
        filename: Checkpoint filename template. Default ``'{epoch}-{step}'``.
        monitor: Metric to monitor for saving.
        verbose: If True, prints save messages.
        save_last: When ``True`` always saves a ``last.pdparams`` copy whenever a
            checkpoint is saved; ``None`` (default) does not write a last file.
        save_top_k: Number of best models to keep (``k``), ``-1`` for all, ``0``
            for none. Default ``1``.
        mode: ``'min'`` or ``'max'`` (direction for monitor comparison).
        save_weights_only: If True, saves only model weights (not optimizer state).
        every_n_epochs: Save a checkpoint every N epochs (mutually exclusive with
            ``every_n_train_steps``).
        every_n_train_steps: Save a checkpoint every N training steps (mutually
            exclusive with ``every_n_epochs``).
    """

    FILE_EXTENSION = ".pdparams"
    LAST_NAME = "last"
    _mode_dict = {"min": float("inf"), "max": float("-inf")}

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
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
        self.kth_best_model_path: str = ""
        self.best_model_path: str = ""
        self.best_model_score: Optional[float] = None
        self.kth_value: Optional[float] = None
        self.last_model_path: str = ""
        self.current_score: Optional[float] = None
        self._last_step_saved: int = -1
        self._last_checkpoint_saved: str = ""
        self._monitor_op = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

        self._validate_init_configuration()
        os.makedirs(self.dirpath, exist_ok=True)

    # ------------------------------------------------------------------
    # configuration validation
    # ------------------------------------------------------------------

    def _validate_init_configuration(self) -> None:
        if self.mode not in self._mode_dict:
            raise ValueError(f"`mode` can be 'min' or 'max' but got {self.mode!r}.")
        if self.save_top_k < -1:
            raise ValueError(f"Invalid value for save_top_k={self.save_top_k}. Must be >= -1.")
        every_n_epochs = self.every_n_epochs or 0
        every_n_train_steps = self.every_n_train_steps or 0
        if every_n_epochs < 0:
            raise ValueError(f"Invalid value for every_n_epochs={self.every_n_epochs}. Must be >= 0.")
        if every_n_train_steps < 0:
            raise ValueError(f"Invalid value for every_n_train_steps={self.every_n_train_steps}. Must be >= 0.")
        if (every_n_epochs >= 1) + (every_n_train_steps >= 1) > 1:
            raise ValueError(
                f"Combination of parameters every_n_train_steps={self.every_n_train_steps} and "
                f"every_n_epochs={self.every_n_epochs} should be mutually exclusive."
            )
        if self.monitor is None and self.save_top_k not in (-1, 0, 1):
            raise ValueError(
                f"ModelCheckpoint(save_top_k={self.save_top_k}, monitor=None) is not a valid"
                " configuration. No quantity for top_k to track."
            )

    # ------------------------------------------------------------------
    # hook entry points
    # ------------------------------------------------------------------

    def on_validation_end(self, trainer: Any, model: Any) -> None:
        self._save_if_needed(trainer, model)

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if self.every_n_epochs is not None and (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        self._save_if_needed(trainer, model)

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.every_n_train_steps is None or self.every_n_train_steps < 1:
            return
        if trainer.optimizer_step - self._last_step_saved < self.every_n_train_steps:
            return
        self._save_if_needed(trainer, model)

    def on_train_end(self, trainer: Any, model: Any) -> None:
        # Ensure save_last is applied when training ends without a final save.
        if self.save_last and not self._last_checkpoint_saved:
            self._save_last_checkpoint(trainer, model, self._monitor_candidates(trainer))

    # ------------------------------------------------------------------
    # monitor candidates + top-k decision
    # ------------------------------------------------------------------

    def _monitor_candidates(self, trainer: Any) -> dict[str, Any]:
        candidates = dict(trainer.callback_metrics)
        candidates["epoch"] = trainer.current_epoch
        candidates["step"] = trainer.optimizer_step
        return candidates

    def _should_skip_saving(self, trainer: Any) -> bool:
        return getattr(trainer, "sanity_checking", False) or trainer.optimizer_step == self._last_step_saved

    def _save_if_needed(self, trainer: Any, model: Any) -> None:
        if self._should_skip_saving(trainer):
            return
        monitor_candidates = self._monitor_candidates(trainer)
        self._save_topk_checkpoint(trainer, model, monitor_candidates)
        self._save_last_checkpoint(trainer, model, monitor_candidates)

    def check_monitor_top_k(self, current: Optional[float]) -> bool:
        if current is None:
            return False
        if self.save_top_k == -1:
            return True
        if len(self.best_k_models) < self.save_top_k:
            return True
        return bool(self._monitor_op(current, self.best_k_models[self.kth_best_model_path]))

    # ------------------------------------------------------------------
    # save paths
    # ------------------------------------------------------------------

    def _format_filename(self, monitor_candidates: dict[str, Any]) -> str:
        try:
            return self.filename.format(**monitor_candidates)
        except KeyError:
            return self.filename.format(epoch=monitor_candidates["epoch"], step=monitor_candidates["step"])

    def _save_topk_checkpoint(self, trainer: Any, model: Any, monitor_candidates: dict[str, Any]) -> None:
        if self.save_top_k == 0:
            return
        if self.monitor is not None:
            current = monitor_candidates.get(self.monitor)
            if current is None:
                return
            if self.check_monitor_top_k(current):
                self._update_best_and_save(current, model, monitor_candidates)
            elif self.verbose:
                print(f"ModelCheckpoint: {self.monitor!r} was not in top {self.save_top_k} (score={current:.4f})")
        else:
            self._save_none_monitor_checkpoint(trainer, model, monitor_candidates)

    def _save_none_monitor_checkpoint(self, trainer: Any, model: Any, monitor_candidates: dict[str, Any]) -> None:
        ckpt_path = os.path.join(self.dirpath, self._format_filename(monitor_candidates) + self.FILE_EXTENSION)
        previous, self.best_model_path = self.best_model_path, ckpt_path
        self._write_checkpoint(model, ckpt_path)
        self.best_k_models[ckpt_path] = float(monitor_candidates["step"])
        if self.save_top_k == 1 and previous and os.path.exists(previous):
            os.remove(previous)
            self.best_k_models.pop(previous, None)

    def _update_best_and_save(self, current: float, model: Any, monitor_candidates: dict[str, Any]) -> None:
        k = self.save_top_k

        del_filepath: Optional[str] = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath, None)

        # don't save nan — treat as worst possible
        if isinstance(current, float) and current != current:  # NaN check
            current = float("inf") if self.mode == "min" else float("-inf")

        ckpt_path = os.path.join(self.dirpath, self._format_filename(monitor_candidates) + self.FILE_EXTENSION)
        self.current_score = current
        self.best_k_models[ckpt_path] = current

        if len(self.best_k_models) == k and k > 0:
            # the kth-best is the worst of the k kept checkpoints
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        if del_filepath and os.path.exists(del_filepath):
            os.remove(del_filepath)

        self._write_checkpoint(model, ckpt_path)
        if self.verbose:
            print(f"ModelCheckpoint: saved '{ckpt_path}' (score={current:.4f})")

    def _save_last_checkpoint(self, trainer: Any, model: Any, monitor_candidates: dict[str, Any]) -> None:
        if not self.save_last:
            return
        filepath = os.path.join(self.dirpath, self.LAST_NAME + self.FILE_EXTENSION)
        self._write_checkpoint(model, filepath)
        self.last_model_path = filepath

    def _write_checkpoint(self, model: Any, path: str) -> None:
        trainer = model._trainer
        if self.save_weights_only:
            state = model.state_dict()
            if hasattr(state, "items"):
                state = {k: v for k, v in state.items()}
            paddle.save(state, path)
        else:
            checkpoint = {
                "ocean_version": ocean.__version__,
                "state_dict": model.state_dict(),
                "epoch": trainer.current_epoch if trainer else 0,
                "dataloader_step": trainer.dataloader_step if trainer else 0,
                "optimizer_step": trainer.optimizer_step if trainer else 0,
            }
            if trainer and trainer.optimizers:
                raw_opt = trainer.optimizers[0]._optimizer
                if raw_opt is not None:
                    checkpoint["optimizer_states"] = [raw_opt.state_dict()]

            if trainer and hasattr(trainer, "fit_loop"):
                loop_state = trainer.fit_loop.state_dict()
                if loop_state:
                    checkpoint["loops"] = loop_state

            if trainer:
                checkpoint["lr_schedulers"] = [cfg["scheduler"].state_dict() for cfg in trainer._lr_schedulers]

            if hasattr(model, "on_save_checkpoint"):
                checkpoint.update(model.on_save_checkpoint())

            if trainer and trainer.strategy is not None:
                pp = trainer.strategy.precision_plugin
                if pp is not None and hasattr(pp, "state_dict"):
                    ps = pp.state_dict()
                    if ps:
                        checkpoint[type(pp).__qualname__] = ps

            if trainer and trainer.datamodule is not None:
                if hasattr(trainer.datamodule, "state_dict"):
                    ds = trainer.datamodule.state_dict()
                    if ds:
                        checkpoint[type(trainer.datamodule).__qualname__] = ds

            if hasattr(model, "hparams") and model.hparams:
                checkpoint["hyper_parameters"] = model.hparams

            callback_states = {}
            for cb in trainer.callbacks if trainer else []:
                if hasattr(cb, "state_dict"):
                    state = cb.state_dict()
                    if state:
                        callback_states[type(cb).__qualname__] = state
            if callback_states:
                checkpoint["callbacks"] = callback_states
            paddle.save(checkpoint, path)

        self._last_step_saved = trainer.optimizer_step if trainer else self._last_step_saved
        self._last_checkpoint_saved = path

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_k_models": self.best_k_models,
            "kth_best_model_path": self.kth_best_model_path,
            "best_model_path": self.best_model_path,
            "best_model_score": self.best_model_score,
            "kth_value": self.kth_value,
            "last_model_path": self.last_model_path,
            "current_score": self.current_score,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.best_k_models = state_dict.get("best_k_models", {})
        self.kth_best_model_path = state_dict.get("kth_best_model_path", "")
        self.best_model_path = state_dict.get("best_model_path", "")
        self.best_model_score = state_dict.get("best_model_score", None)
        self.kth_value = state_dict.get("kth_value", None)
        self.last_model_path = state_dict.get("last_model_path", "")
        self.current_score = state_dict.get("current_score", None)
