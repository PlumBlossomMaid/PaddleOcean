"""_ManualOptimization - handles manual training steps (user controls backward/step)."""

from typing import Any


class _ManualOptimization:
    """Manual optimization sub-loop - wraps training_step for manual mode."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def run(self, kwargs: dict) -> Any:
        model = self.trainer._model
        return model.training_step(**kwargs)
