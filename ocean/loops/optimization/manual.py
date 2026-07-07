"""_ManualOptimization - handles manual training steps (user controls backward/step).

In manual mode the user drives ``backward()`` / ``optimizer.step()`` from inside
``training_step``. This loop is a pass-through: it does not scale the loss, clip
gradients, or step optimizers/schedulers. The optimizer-step counter advances only
when the user steps a wrapped ``OceanOptimizer`` (via its ``_on_after_step`` hook),
so this loop must NOT fake ``_optimizer_step`` increments.
"""

from typing import Any


class _ManualOptimization:
    """Manual optimization sub-loop - wraps training_step for manual mode."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def run(self, kwargs: dict) -> Any:
        model = self.trainer._model
        return model.training_step(**kwargs)
