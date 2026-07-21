"""_ManualOptimization - handles manual training steps (user controls backward/step).

In manual mode the user drives ``backward()`` / ``optimizer.step()`` from inside
``training_step``. This loop does not scale the loss, clip gradients, or step
optimizers/schedulers — but it does run ``training_step`` through the strategy so
the precision plugin's forward context (AMP auto_cast) is active, and
``model.manual_backward`` routes the backward through the same plugin's GradScaler.
The optimizer-step counter advances only when the user steps a wrapped
``OceanOptimizer`` (via its ``_on_after_step`` hook), so this loop must NOT fake
``_optimizer_step`` increments.
"""

from typing import Any


class _ManualOptimization:
    """Manual optimization sub-loop - wraps training_step for manual mode."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def run(self, kwargs: dict) -> Any:
        # Route through the strategy so the precision plugin's forward_context
        # (AMP auto_cast) wraps the user's training_step, matching the
        # reference behavior of calling the strategy "training_step" hook.
        return self.trainer.strategy.training_step(**kwargs)
