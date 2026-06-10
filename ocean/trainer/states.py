"""Trainer state management enums."""

from __future__ import annotations

from enum import Enum, auto


class TrainerStatus(Enum):
    INITIALIZING = auto()
    RUNNING = auto()
    FINISHED = auto()
    INTERRUPTED = auto()

    @property
    def stopped(self) -> bool:
        return self in (TrainerStatus.FINISHED, TrainerStatus.INTERRUPTED)


class TrainerFn(Enum):
    FITTING = auto()
    VALIDATING = auto()
    TESTING = auto()
    PREDICTING = auto()


class RunningStage(Enum):
    TRAINING = auto()
    SANITY_CHECKING = auto()
    VALIDATING = auto()
    TESTING = auto()
    PREDICTING = auto()

    @property
    def evaluating(self) -> bool:
        return self in (RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.SANITY_CHECKING)


class TrainerState:
    """Holds the current state of the Trainer."""

    def __init__(self) -> None:
        self.status: TrainerStatus = TrainerStatus.INITIALIZING
        self.fn: Optional[TrainerFn] = None
        self.stage: Optional[RunningStage] = None

    @property
    def finished(self) -> bool:
        return self.status.stopped

    @property
    def stopped(self) -> bool:
        return self.status.stopped
