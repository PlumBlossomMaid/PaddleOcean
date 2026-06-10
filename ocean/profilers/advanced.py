"""Advanced profiler using PaddlePaddle's profiler API.

Analogous to Lightning's advanced profiler.
"""

import time
from collections import defaultdict
from typing import Any, Optional

import paddle


class AdvancedProfiler:
    """Advanced profiler using paddle.profiler.

    Records execution time and FLOPs for training steps.

    Args:
        dirpath: Directory to save profiling traces.
        filename: Profiling output filename.
    """

    def __init__(self, dirpath: str = ".", filename: str = "profile.json") -> None:
        self.dirpath = dirpath
        self.filename = filename
        self._records: dict[str, list[float]] = defaultdict(list)
        self._start_times: dict[str, float] = {}
        self._profiler: Optional[Any] = None
        self._enabled = False

    def start(self, action_name: str) -> None:
        self._start_times[action_name] = time.perf_counter()

    def stop(self, action_name: str) -> None:
        if action_name in self._start_times:
            elapsed = time.perf_counter() - self._start_times.pop(action_name)
            self._records[action_name].append(elapsed)

    def start_profile(self) -> None:
        """Start the Paddle profiler for detailed tracing."""
        try:
            self._profiler = paddle.profiler.Profiler(
                targets=[paddle.profiler.ProfilerTarget.CUSTOM],
                custom_device_types=[],
            )
            self._profiler.start()
            self._enabled = True
        except Exception:
            self._enabled = False

    def stop_profile(self) -> None:
        """Stop the Paddle profiler and export traces."""
        if self._profiler is not None and self._enabled:
            try:
                self._profiler.stop()
                import os
                path = os.path.join(self.dirpath, self.filename)
                self._profiler.export(path=path, format="json")
            except Exception:
                pass
            self._profiler = None
            self._enabled = False

    def summary(self) -> str:
        """Generate a profiling summary."""
        lines = ["Advanced Profiler Summary:"]
        for name, times in sorted(self._records.items()):
            if times:
                avg = sum(times) / len(times)
                total = sum(times)
                lines.append(f"  {name}: {len(times)} calls, avg {avg*1000:.3f}ms, total {total*1000:.3f}ms")
        return "\n".join(lines)

    def teardown(self) -> None:
        self.stop_profile()
        self._records.clear()
