"""SQLite ledger of every run + content-hash cache + metric parsing for the OKC bench scripts."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

from .config import LEDGER_PATH

SOURCE_EXTS = {".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".c", ".json", ".txt", ".toml", ".yaml", ".yml",
               ".sh", ".ptx"}
SKIP_DIRS = {".git", "output", "out", "profiles", "runs", "__pycache__", ".venv", "node_modules", ".pytest_cache"}

SCHEMA = """
create table if not exists runs (
  id text primary key, key text, agent text, track text, cmd text, mode text, backend text,
  status text, exit_code int, started real, ended real, wall_s real, cost_usd real,
  metrics text, log_path text, cached_from text
);
create index if not exists runs_key on runs(key);
create table if not exists sessions (
  pod_id text, start real, uptime_s real, cost_per_hr real, primary key (pod_id, start)
);
"""


# columns added after v0; connect() adds them to existing ledgers
EXTRA_COLUMNS = {"track_path": "text", "note": "text", "failure": "text", "error": "text", "timeout_s": "real",
                 "submitted": "real", "extra_paths": "text",
                 "pool": "text"}


def connect(path: Path = LEDGER_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    have = {r["name"] for r in db.execute("pragma table_info(runs)")}
    for col, typ in EXTRA_COLUMNS.items():
        if col not in have:
            db.execute(f"alter table runs add column {col} {typ}")
    db.commit()
    return db


def source_files(root: Path, excludes: list[str] = ()) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts) or not p.is_file() or p.suffix not in SOURCE_EXTS:
            continue
        if any(fnmatch.fnmatch(str(rel), x) for x in excludes):
            continue
        out.append(p)
    return out


def content_key(roots: list[Path], cmd: str, mode: str, image: str) -> str:
    """Hash of every source file under the synced dirs + command + mode + image: identical inputs → identical key."""
    h = hashlib.sha256()
    for k in (cmd, mode, image):
        h.update(k.encode() + b"\0")
    for root in roots:
        for p in source_files(root):
            h.update(str(p.relative_to(root)).encode() + b"\0")
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


SCORE = re.compile(r"^SCORE (\S+): ([\d.]+)")
CHECK = re.compile(r"^CHECK (\S+) (\S+) (\S+): (PASS|FAIL)")
CHECKS = re.compile(r"^CHECKS: (ALL PASS|(\d+) FAIL)")
SIDE = re.compile(r"^SIDE (\S+): (\{.*\})\s*$")
METRIC = re.compile(r"^OGPU_METRIC (\w+)=(\S+)")


def parse_metrics(log: str) -> dict:
    """Pull the headline numbers out of pod/bench.py, pair_bench.py, and any `OGPU_METRIC k=v` lines."""
    m: dict = {}
    fails = passes = 0
    for line in log.splitlines():
        line = line.strip()
        if r := SCORE.match(line):
            m.setdefault("score", {})[r[1]] = float(r[2])
        elif r := CHECK.match(line):
            passes += r[4] == "PASS"
            fails += r[4] == "FAIL"
        elif r := CHECKS.match(line):
            m["checks"] = "PASS" if r[1] == "ALL PASS" else "FAIL"
        elif r := SIDE.match(line):
            try:
                m.setdefault("sides", {})[r[1]] = json.loads(r[2])
            except ValueError:
                pass
        elif r := METRIC.match(line):
            try:
                m[r[1]] = float(r[2])
            except ValueError:
                m[r[1]] = r[2]
    if passes or fails:
        m["check_pass"], m["check_fail"] = passes, fails
        m.setdefault("checks", "FAIL" if fails else "PASS")
    return m


def headline(metrics: dict) -> str:
    """One short human string, e.g. 'best 1.0831× · checks PASS'."""
    parts = []
    if metrics.get("score"):
        name, v = max(metrics["score"].items(), key=lambda kv: kv[1])
        parts.append(f"{name} {v:.4f}×")
    for k, s in (metrics.get("sides") or {}).items():
        if "ratio_vs_s0" in s:
            lo, hi = s.get("ci95", [None, None])
            ci = f" [{lo:.3f},{hi:.3f}]" if lo is not None else ""
            parts.append(f"{k} {s['ratio_vs_s0']:.4f}× vs s0{ci}")
    if "checks" in metrics:
        parts.append(f"checks {metrics['checks']}")
    return " · ".join(parts)


TERMINAL = ("done", "failed", "lost", "cancelled", "missing")
IN_FLIGHT_S = 3 * 3600


def cached(db: sqlite3.Connection, key: str):
    """A finished run of the same inputs (lost/cancelled runs don't count: they measured nothing)."""
    return db.execute("select * from runs where key=? and status in ('done','failed') and cached_from is null "
                      "order by ended desc limit 1", (key,)).fetchone()


def in_flight(db: sqlite3.Connection, key: str):
    """The same inputs already queued/running (e.g. two agents submitting one candidate): wait on it instead."""
    return db.execute("select * from runs where key=? and status in ('queued','running') and submitted > ? "
                      "order by submitted desc limit 1", (key, time.time() - IN_FLIGHT_S)).fetchone()


def get(db: sqlite3.Connection, run_id: str):
    return db.execute("select * from runs where id=?", (run_id,)).fetchone()


def update(db: sqlite3.Connection, run_id: str, **fields) -> None:
    if "metrics" in fields:
        fields["metrics"] = json.dumps(fields["metrics"])
    db.execute(f"update runs set {', '.join(f'{k}=?' for k in fields)} where id=?", (*fields.values(), run_id))
    db.commit()


def best_score(metrics: dict) -> float | None:
    vals = list((metrics.get("score") or {}).values())
    vals += [s["ratio_vs_s0"] for s in (metrics.get("sides") or {}).values() if s.get("ratio_vs_s0")]
    return max(vals) if vals else None


def track_best(db: sqlite3.Connection, track: str, exclude_id: str = "") -> float | None:
    best = None
    for r in db.execute("select id, metrics from runs where track=? and status='done' and cached_from is null "
                        "and id != ?", (track, exclude_id)):
        m = json.loads(r["metrics"] or "{}")
        if m.get("checks", "PASS") != "PASS":
            continue
        b = best_score(m)
        if b is not None and (best is None or b > best):
            best = b
    return best


def record(db: sqlite3.Connection, **row) -> None:
    row.setdefault("metrics", {})
    row["metrics"] = json.dumps(row["metrics"])
    cols = ",".join(row)
    db.execute(f"insert or replace into runs ({cols}) values ({','.join('?' * len(row))})", tuple(row.values()))
    db.commit()


def record_session(db: sqlite3.Connection, pod_id: str, uptime_s: float, cost_per_hr: float) -> None:
    """Upsert the current pod session. start ≈ now − uptime jitters by a few seconds between polls, so match any
    session of this pod that started within 2 minutes instead of keying on an exact start."""
    start = time.time() - uptime_s
    row = db.execute("select start from sessions where pod_id=? and abs(start - ?) < 120 order by abs(start - ?) "
                     "limit 1", (pod_id, start, start)).fetchone()
    if row:
        db.execute("update sessions set uptime_s=max(uptime_s, ?), cost_per_hr=? where pod_id=? and start=?",
                   (uptime_s, cost_per_hr, pod_id, row["start"]))
    else:
        db.execute("insert into sessions values (?,?,?,?)", (pod_id, start, uptime_s, cost_per_hr))
    db.commit()


def spend(db: sqlite3.Connection, since: float = 0.0) -> float:
    """$ spent in sessions that ended after `since` (clipped to `since`)."""
    total = 0.0
    for s in db.execute("select * from sessions"):
        start, end = s["start"], s["start"] + s["uptime_s"]
        if end > since:
            total += (end - max(start, since)) / 3600 * s["cost_per_hr"]
    return total


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(i for i in intervals if i[1] > i[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _overlap(xs: list[tuple[float, float]], ys: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(b, d) - max(a, c)) for a, b in xs for c, d in ys)


def utilization(db: sqlite3.Connection, since: float = 0.0) -> dict:
    """Where the paid pod time went: GPU-locked jobs, CPU-only jobs, and idle (boot, agents thinking, reaper wait)."""
    now = time.time()
    sessions, paid_cost = [], 0.0
    for s in db.execute("select * from sessions"):
        a, b = max(s["start"], since), min(s["start"] + s["uptime_s"], now)
        if b > a:
            sessions.append((a, b, s["cost_per_hr"]))
            paid_cost += (b - a) / 3600 * s["cost_per_hr"]
    spans = _union([(a, b) for a, b, _ in sessions])
    rows = db.execute("select mode, started, ended from runs where backend='runpod' and started is not null "
                      "and ended is not null and ended > ?", (since,)).fetchall()
    gpu = _union([(r["started"], r["ended"]) for r in rows if r["mode"] == "gpu"])
    busy = _union([(r["started"], r["ended"]) for r in rows])
    paid_s = sum(b - a for a, b in spans)
    gpu_s, busy_s = _overlap(gpu, spans), _overlap(busy, spans)
    rate = paid_cost / (paid_s / 3600) if paid_s else 0.0
    return {"paid_s": paid_s, "paid_usd": paid_cost, "gpu_busy_s": gpu_s, "any_busy_s": busy_s,
            "gpu_busy_frac": gpu_s / paid_s if paid_s else 0.0, "idle_usd": (paid_s - busy_s) / 3600 * rate}


def agent_spend(db: sqlite3.Connection, since: float = 0.0) -> dict[str, float]:
    return {r["agent"]: r["c"] or 0.0 for r in db.execute(
        "select agent, sum(cost_usd) c from runs where ended > ? group by agent order by c desc", (since,))}


def today_start() -> float:
    t = time.localtime()
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
