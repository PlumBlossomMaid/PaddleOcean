"""_AutomaticOptimization - handles automatic backward + optimizer step.

Note: PaddlePaddle's Optimizer.step() does NOT accept a closure argument
(unlike the reference framework, where the closure is handed to the optimizer).
We run the forward/backward inline and then call step() directly.

The accumulation decision (whether this batch closes an accumulation window)
is owned by :class:`_TrainingEpochLoop` and passed in as ``should_step`` — this
loop is a pure executor that either accumulates gradients or performs the step.
"""

from typing import Any

import paddle

from ocean.trainer.call import _call_callback_hooks


class _AutomaticOptimization:
    """Automatic optimization - runs training_step, backward, and optimizer.step."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def run(self, optimizer: Any, batch_idx: int, kwargs: dict, should_step: bool) -> Any:
        """Run one automatic-optimization batch.

        Args:
            optimizer: the wrapped ``OceanOptimizer`` (stepping it advances
                ``trainer._optimizer_step`` via the ``_on_after_step`` hook).
            batch_idx: epoch-local batch index.
            kwargs: ``{"batch": ..., "batch_idx": ...}`` for ``training_step``.
            should_step: whether an optimizer step should happen this batch
                (accumulation window closed, or final batch of the epoch).

        Returns:
            The ``training_step`` output as a dict (``{"loss": ...}`` when the
            step returned a bare tensor).
        """
        trainer = self.trainer
        model = trainer._model
        raw_opt = optimizer._optimizer

        # Forward + loss (through strategy for AMP auto_cast)
        result = trainer.strategy.training_step(**kwargs)
        loss = result["loss"] if isinstance(result, dict) else (result if isinstance(result, paddle.Tensor) else None)

        if loss is not None:
            # Scale loss so accumulated gradients average across the window
            loss = loss / max(1, trainer.accumulate_grad_batches)

            # Backward pass (through strategy for GradScaler)
            model.on_before_backward(loss)
            _call_callback_hooks(trainer, "on_before_backward", loss)
            trainer.strategy.backward(loss)
            model.on_after_backward()
            _call_callback_hooks(trainer, "on_after_backward")

            if should_step:
                self._clip_gradients(model, raw_opt)
                model.on_before_optimizer_step(raw_opt)
                _call_callback_hooks(trainer, "on_before_optimizer_step", raw_opt)
                # Step through strategy for scaler.step()/update()
                trainer.strategy.optimizer_step(raw_opt)
                trainer._advance_optimizer_step()
                model.on_before_zero_grad(raw_opt)
                _call_callback_hooks(trainer, "on_before_zero_grad", raw_opt)
                optimizer.clear_grad()

        return result if isinstance(result, dict) else {"loss": loss}

    def _clip_gradients(self, model: Any, raw_opt: Any) -> None:
        """Apply gradient clipping per the trainer's configured algorithm.

        Args:
            raw_opt: The raw PaddlePaddle optimizer, used for AMP unscaling
                before clipping.
        """
        trainer = self.trainer
        clip_val = trainer.gradient_clip_val
        if clip_val is None or clip_val <= 0:
            return
        # Unscale gradients before clipping (AMP GradScaler)
        trainer.strategy.precision_plugin.unscale_gradients(raw_opt)
        if trainer.gradient_clip_algorithm == "value":
            paddle.nn.utils.clip_grad_value_(model.parameters(), clip_val)
        else:
            paddle.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
