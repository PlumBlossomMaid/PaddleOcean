"""Simple and PassThrough profilers."""

import logging
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger(__name__)


class Profiler:
    """Base profiler class.

    Subclasses define :meth:`start` / :meth:`stop`; :meth:`profile` wraps a block
    of code so the enclosed action is timed with a ``with`` statement.
    """

    def __init__(self) -> None:
        self._start_times: dict[str, float] = {}
        self._records: dict[str, list[float]] = defaultdict(list)
        self._stage: Optional[str] = None
        self._local_rank: Optional[int] = None

    def start(self, action_name: str) -> None:
        self._start_times[action_name] = time.perf_counter()

    def stop(self, action_name: str) -> None:
        if action_name in self._start_times:
            elapsed = time.perf_counter() - self._start_times.pop(action_name)
            self._records[action_name].append(elapsed)

    @contextmanager
    def profile(self, action_name: str) -> Generator:
        """Time the wrapped block under ``action_name``.

        The action starts on entry and stops on exit. A failure while stopping is
        swallowed so a profiler can never crash the run it is only measuring.

        Example::

            with trainer.profiler.profile("run_training_batch"):
                ...  # code to profile
        """
        try:
            self.start(action_name)
            yield action_name
        finally:
            try:
                self.stop(action_name)
            except Exception:  # noqa: BLE001 - profiling must never crash the run
                pass

    def summary(self) -> str:
        lines = ["Profiler Summary:"]
        for name, times in sorted(self._records.items()):
            if times:
                avg = sum(times) / len(times)
                total = sum(times)
                lines.append(f"  {name}: {len(times)} calls, avg {avg * 1000:.2f}ms, total {total * 1000:.2f}ms")
        return "\n".join(lines)

    def setup(self, stage: Optional[str] = None, local_rank: Optional[int] = None) -> None:
        """Record the running stage and rank for the final report."""
        self._stage = stage
        self._local_rank = local_rank

    def describe(self) -> None:
        """Log the profile report at the end of a run (rank zero only)."""
        if self._local_rank not in (None, 0):
            return
        summary = self.summary()
        if summary:
            log.info(summary)

    def teardown(self) -> None:
        self._start_times.clear()
        self._records.clear()


class SimpleProfiler(Profiler):
    """Simple profiler that records execution times of named actions."""

    def __init__(self) -> None:
        super().__init__()


class PassThroughProfiler(Profiler):
    """Profiler that does nothing (no-op)."""

    def start(self, action_name: str) -> None:
        pass

    def stop(self, action_name: str) -> None:
        pass

    def summary(self) -> str:
        return ""
