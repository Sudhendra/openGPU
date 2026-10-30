"""Stream `nvidia-smi` over SSH into a ring buffer for the live TUI."""
from __future__ import annotations

import collections
import subprocess
import threading
from dataclasses import dataclass

QUERY = "utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,name"
SMI = f"nvidia-smi --query-gpu={QUERY} --format=csv,noheader,nounits -lms 1000"
SPARK = "▁▂▃▄▅▆▇█"


@dataclass
class Sample:
    util: float
    mem_used_gb: float
    mem_total_gb: float
    power_w: float
    power_limit_w: float
    temp_c: float
    sm_mhz: float
    name: str


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return 0.0  # "[N/A]" etc.


def parse_line(line: str) -> Sample | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 8:
        return None
    u, mu, mt, pw, pl, t, sm = (_num(p) for p in parts[:7])
    return Sample(u, mu / 1024, mt / 1024, pw, pl, t, sm, ",".join(parts[7:]))


def sparkline(values, width: int) -> str:
    vals = list(values)[-width:]
    return "".join(SPARK[min(len(SPARK) - 1, int(v / 100 * len(SPARK)))] for v in vals).rjust(width, " ")


class Telemetry:
    def __init__(self, argv: list[str], history: int = 240):
        self.argv = argv
        self.latest: Sample | None = None
        self.util = collections.deque(maxlen=history)
        self.error = ""
        self._proc = None
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            try:
                self._proc = subprocess.Popen(self.argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for line in self._proc.stdout:
                    s = parse_line(line)
                    if s:
                        self.latest, self.error = s, ""
                        self.util.append(s.util)
                self.error = (self._proc.stderr.read() or "telemetry stream ended").strip()[:120]
            except OSError as e:
                self.error = str(e)
            if not self._stop:
                threading.Event().wait(3)

    def close(self):
        self._stop = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
