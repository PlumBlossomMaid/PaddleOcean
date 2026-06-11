"""Progress tracking for loops - tracks ready/started/processed/completed counters."""

from typing import Any


class _ReadyCompletedTracker:
    """Tracks ready and completed counts."""

    def __init__(self) -> None:
        self.ready: int = 0
        self.completed: int = 0

    def state_dict(self) -> dict[str, int]:
        return {"ready": self.ready, "completed": self.completed}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.ready = state.get("ready", 0)
        self.completed = state.get("completed", 0)


class _StartedTracker(_ReadyCompletedTracker):
    """Adds started count."""

    def __init__(self) -> None:
        super().__init__()
        self.started: int = 0

    def state_dict(self) -> dict[str, int]:
        d = super().state_dict()
        d["started"] = self.started
        return d

    def load_state_dict(self, state: dict[str, int]) -> None:
        super().load_state_dict(state)
        self.started = state.get("started", 0)


class _ProcessedTracker(_StartedTracker):
    """Adds processed count."""

    def __init__(self) -> None:
        super().__init__()
        self.processed: int = 0

    def state_dict(self) -> dict[str, int]:
        d = super().state_dict()
        d["processed"] = self.processed
        return d

    def load_state_dict(self, state: dict[str, int]) -> None:
        super().load_state_dict(state)
        self.processed = state.get("processed", 0)


class _Progress:
    """Tracks total and current progress with ready/started/processed/completed counters."""

    def __init__(self) -> None:
        self.total = _ProcessedTracker()
        self.current = _ProcessedTracker()

    @property
    def is_last_batch(self) -> bool:
        return self.current.ready == self.total.ready

    def reset(self) -> None:
        self.current = _ProcessedTracker()

    def reset_on_restart(self) -> None:
        self.current.ready = self.total.completed
        self.current.started = self.total.completed
        self.current.processed = self.total.completed
        self.current.completed = self.total.completed

    def increment_ready(self) -> None:
        self.total.ready += 1
        self.current.ready += 1

    def increment_started(self) -> None:
        self.total.started += 1
        self.current.started += 1

    def increment_processed(self) -> None:
        self.total.processed += 1
        self.current.processed += 1

    def increment_completed(self) -> None:
        self.total.completed += 1
        self.current.completed += 1

    def state_dict(self) -> dict[str, Any]:
        return {"total": self.total.state_dict(), "current": self.current.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.total.load_state_dict(state.get("total", {}))
        self.current.load_state_dict(state.get("current", {}))


class _BatchProgress(_Progress):
    """Batch-level progress with is_last_batch."""

    def __init__(self) -> None:
        super().__init__()
        self._is_last_batch: bool = False

    @property
    def is_last_batch(self) -> bool:
        return self._is_last_batch

    def update_last_batch(self, is_last: bool) -> None:
        self._is_last_batch = is_last


class _SchedulerProgress:
    """Scheduler progress with ready/completed."""

    def __init__(self) -> None:
        self.total = _ReadyCompletedTracker()
        self.current = _ReadyCompletedTracker()

    def reset(self) -> None:
        self.current = _ReadyCompletedTracker()

    def increment_ready(self) -> None:
        self.total.ready += 1
        self.current.ready += 1

    def increment_completed(self) -> None:
        self.total.completed += 1
        self.current.completed += 1

    def state_dict(self) -> dict:
        return {"total": self.total.state_dict(), "current": self.current.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.total.load_state_dict(state.get("total", {}))
        self.current.load_state_dict(state.get("current", {}))


class _OptimizerProgress:
    """Optimizer progress."""

    def __init__(self) -> None:
        self.step = _Progress()
        self.zero_grad = _Progress()


class _OptimizationProgress:
    """Optimization progress containing optimizer step and zero_grad progress."""

    def __init__(self) -> None:
        self.optimizer = _OptimizerProgress()

    def state_dict(self) -> dict:
        return {"optimizer": {"step": self.optimizer.step.state_dict(), "zero_grad": self.optimizer.zero_grad.state_dict()}}

    def load_state_dict(self, state: dict) -> None:
        opt = state.get("optimizer", {})
        self.optimizer.step.load_state_dict(opt.get("step", {}))
        self.optimizer.zero_grad.load_state_dict(opt.get("zero_grad", {}))
