"""Simple and PassThrough profilers."""

import time
from collections import defaultdict


class Profiler:
    """Base profiler class."""

    def __init__(self) -> None:
        self._start_times: dict[str, float] = {}
        self._records: dict[str, list[float]] = defaultdict(list)

    def start(self, action_name: str) -> None:
        self._start_times[action_name] = time.perf_counter()

    def stop(self, action_name: str) -> None:
        if action_name in self._start_times:
            elapsed = time.perf_counter() - self._start_times.pop(action_name)
            self._records[action_name].append(elapsed)

    def summary(self) -> str:
        lines = ["Profiler Summary:"]
        for name, times in sorted(self._records.items()):
            if times:
                avg = sum(times) / len(times)
                total = sum(times)
                lines.append(f"  {name}: {len(times)} calls, avg {avg*1000:.2f}ms, total {total*1000:.2f}ms")
        return "\n".join(lines)

    def teardown(self) -> None:
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
