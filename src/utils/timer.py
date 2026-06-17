from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Timer:
    start_time: float | None = None
    end_time: float | None = None

    def start(self) -> "Timer":
        self.start_time = time.perf_counter()
        self.end_time = None
        return self

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


def gpu_hours_from_seconds(seconds: float, device_count: int = 1) -> float:
    return seconds * max(1, int(device_count)) / 3600.0
