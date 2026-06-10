"""_AutomaticOptimization - handles automatic backward + optimizer step.

Note: PaddlePaddle's Optimizer.step() does NOT accept a closure argument
(unlike PyTorch). We run the closure inline and then call step().
"""

from typing import Any

import paddle

from ocean.loops.optimization.closure import AbstractClosure, OutputResult


class Closure(AbstractClosure[OutputResult]):
    """Closure that performs training_step, zero_grad, and backward."""

    def __init__(self, step_fn: Any, backward_fn: Any = None, zero_grad_fn: Any = None) -> None:
        super().__init__()
        self.step_fn = step_fn
        self.backward_fn = backward_fn
        self.zero_grad_fn = zero_grad_fn

    def closure(self, *args: Any, **kwargs: Any) -> OutputResult:
        if self.zero_grad_fn is not None:
            self.zero_grad_fn()
        result = self.step_fn(*args, **kwargs)
        loss = result.loss if isinstance(result, OutputResult) else (result if isinstance(result, paddle.Tensor) else None)
        if loss is not None and self.backward_fn is not None:
            self.backward_fn(loss)
        return result if isinstance(result, OutputResult) else OutputResult(loss=loss)


class _AutomaticOptimization:
    """Automatic optimization - runs training_step, backward, and optimizer.step."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def run(self, optimizer: Any, batch_idx: int, kwargs: dict) -> Any:
        model = self.trainer._model
        step_kwargs = kwargs

        # Run the training step (forward + loss)
        result = model.training_step(**step_kwargs)
        loss = result["loss"] if isinstance(result, dict) else (result if isinstance(result, paddle.Tensor) else None)

        if loss is not None and optimizer is not None:
            # Gradient accumulation: scale loss
            loss = loss / max(1, self.trainer.accumulate_grad_batches)

            # Backward pass
            model.on_before_backward(loss)
            loss.backward()
            model.on_after_backward()

            # Gradient clipping
            if self.trainer.gradient_clip_val is not None:
                paddle.nn.utils.clip_grad_norm_(model.parameters(), self.trainer.gradient_clip_val)

            # Optimizer step (Paddle doesn't support closure in step())
            if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
                model.on_before_optimizer_step(optimizer)
                optimizer.step()
                optimizer.clear_grad()
                self.trainer.global_step += 1

        # Call model's on_train_batch_end (should be in the loop, not here)
        return result if isinstance(result, dict) else {"loss": loss}
