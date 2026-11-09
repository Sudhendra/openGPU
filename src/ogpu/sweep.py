"""`ogpu sweep`: a multi-fidelity tournament over many candidates, one command for the agent.

  1. compile    each candidate on CPU, in parallel            (optional --compile template; drops build failures)
  2. correct    all survivors in ONE job on the correctness pool (cheap same-ISA GPU; drops wrong ones)
  3. quick      best + survivors, few reps, ONE job on the timing pool → keep top-k and anything within noise
  4. full       best + finalists, full reps, ONE job on the timing pool → ranked, with a verdict vs best

Most candidates never reach the expensive GPU, and prepare is paid once per stage instead of once per candidate.

Bench contract: the bench takes candidate paths as arguments and prints, per candidate, `CHECK <case> <name> <tag>: PASS|FAIL ...` and `SCORE <name>: <speedup>` (higher = better),
where <name> is the argument's basename (`cand/a.py@X=1` → `a.py@X=1`, optionally prefixed `<i>:`).
"""
from __future__ import annotations

import concurrent.futures as cf
import re
import shlex
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import runner
from .config import Config

CHECK_LINE = re.compile(r"^CHECK \S+ (\S+) \S+: (PASS|FAIL)")
ERR = re.compile(r"relative L2 error ([\d.e+-]+)[^\[]*\[bound ([\d.e+-]+)")


@dataclass
class Cand:
    token: str                       # as passed to the bench, e.g. "cand/a.py@X=1"
    stage: str = "pending"           # how far it got: compile / correct / quick / full
    dropped: str | None = None       # why it stopped
    margin: float | None = None      # worst error / bound on the correctness run
    quick: float | None = None
    full: float | None = None
    vs_best: float | None = None     # full-score ratio vs the best (1.021 = +2.1%)
    jobs: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.token.split("/")[-1]


def match(printed: str, cands: list[Cand]) -> Cand | None:
    """Map a name printed by the bench back to its candidate (basename, optionally `<i>:`-prefixed)."""
    base = printed.split(":", 1)[1] if re.match(r"^\d+:", printed) else printed
    return next((c for c in cands if c.name == base or c.token == base), None)


def per_candidate_checks(log: str, cands: list[Cand]) -> dict[str, tuple[bool, float]]:
    """token → (all checks passed, worst error/bound) from CHECK lines."""
    out: dict[str, list] = {}
    for line in log.splitlines():
        m = CHECK_LINE.match(line.strip())
        if not m or not (c := match(m[1], cands)):
            continue
        ok, worst = out.get(c.token, [True, 0.0])
        ok = ok and m[2] == "PASS"
        for err, bound in ERR.findall(line):
            try:
                worst = max(worst, float(err) / float(bound))
            except (ValueError, ZeroDivisionError):
                pass
        out[c.token] = [ok, worst]
    return {k: (v[0], v[1]) for k, v in out.items()}


def per_candidate_scores(log: str, cands: list[Cand]) -> dict[str, float]:
    out = {}
    for line in log.splitlines():
        m = re.match(r"^SCORE (\S+): ([\d.]+)", line.strip())
        if m and (c := match(m[1], cands)):
            out[c.token] = float(m[2])
    return out


def build_cmd(template: str, tokens: list[str], extra: str) -> str:
    files = " ".join(shlex.quote(t) for t in tokens)
    cmd = template.replace("{files}", files) if "{files}" in template else f"{template} {files}"
    return f"{cmd} {extra}".strip()


def select_finalists(scores: dict[str, float], best: str | None, top_k: int, noise: float) -> list[str]:
    """Top-k candidates by quick score, plus anything within `noise` of the leader (quick runs are noisy)."""
    ranked = sorted(((s, t) for t, s in scores.items() if t != best), reverse=True)
    if not ranked:
        return []
    lead = ranked[0][0]
    return [t for i, (s, t) in enumerate(ranked) if i < top_k or s >= lead * (1 - noise)]


class Sweep:
    def __init__(self, cfg: Config, console: Console, *, track: Path, extra: list[Path], cands: list[str],
                 best: str | None, bench: str, compile_tpl: str | None, check_flag: str, correct_args: str,
                 quick_args: str, full_args: str, top_k: int, noise: float, note: str, timeout_s: float | None,
                 cache: bool | None = None):
        self.cfg, self.con = cfg, console
        self.track, self.extra = track, extra
        self.cands = [Cand(t) for t in dict.fromkeys(cands) if t != best]
        self.best = Cand(best) if best else None
        self.bench, self.compile_tpl = bench, compile_tpl
        self.check_flag, self.correct_args = check_flag, correct_args
        self.quick_args, self.full_args = quick_args, full_args
        self.top_k, self.noise, self.timeout_s, self.cache = top_k, noise, timeout_s, cache
        self.id = "sw-" + time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.note = f"[{self.id}] {note}".strip()
        self.cost = 0.0

    # ---- helpers
    def alive(self) -> list[Cand]:
        return [c for c in self.cands if not c.dropped]

    def drop(self, c: Cand, why: str) -> None:
        c.dropped = why

    def job(self, cmd: str, *, mode: str = "gpu", fidelity: str = "timing", stage: str) -> dict:
        sub = runner.submit(self.cfg, self.con, track=self.track, extra=self.extra, cmd=cmd, mode=mode,
                            cache=self.cache, note=f"{self.note} · {stage}", timeout_s=self.timeout_s,
                            fidelity=fidelity)
        r = sub if sub["status"] != "queued" else runner.wait(self.cfg, Console(stderr=True, quiet=True), sub["id"],
                                                              stream=False)
        self.cost += r.get("cost_usd") or 0.0
        return r

    @staticmethod
    def log_of(r: dict) -> str:
        p = Path(r.get("log_path") or "")
        return p.read_text(errors="replace") if p.is_file() else ""

    # ---- stages
    def stage_compile(self) -> None:
        if not self.compile_tpl:
            return
        self.con.print(f"[cyan]① compile[/] {len(self.alive())} candidates on CPU, in parallel")
        quiet = Console(stderr=True, quiet=True)
        subs = {}
        for c in self.alive():
            c.stage = "compile"
            cmd = self.compile_tpl.replace("{file}", shlex.quote(c.token.split("@")[0]))
            subs[c.token] = runner.submit(self.cfg, self.con, track=self.track, extra=self.extra, cmd=cmd, mode="cpu",
                                          cache=self.cache, note=f"{self.note} · compile", timeout_s=self.timeout_s)
        with cf.ThreadPoolExecutor(max(1, len(subs))) as ex:
            futs = {ex.submit(lambda s: s if s["status"] != "queued" else runner.wait(self.cfg, quiet, s["id"],
                                                                                      stream=False), s): t
                    for t, s in subs.items()}
            for f in cf.as_completed(futs):
                c = next(c for c in self.cands if c.token == futs[f])
                r = f.result()
                c.jobs["compile"] = r["id"]
                if r["status"] != "done":
                    self.drop(c, f"compile: {r.get('failure') or r['status']}")

    def stage_correct(self) -> None:
        cands = self.alive()
        if not cands:
            return
        self.con.print(f"[cyan]② correctness[/] {len(cands)} candidates in one job (correctness pool)")
        for c in cands:
            c.stage = "correct"
        r = self.job(build_cmd(self.bench, [c.token for c in cands], f"{self.check_flag} {self.correct_args}"),
                     fidelity="correctness",
                     stage="correctness")
        checks = per_candidate_checks(self.log_of(r), cands)
        for c in cands:
            c.jobs["correct"] = r["id"]
            if c.token not in checks:
                self.drop(c, f"correctness: no CHECK output ({r.get('failure') or 'bench printed nothing for it'})")
                continue
            ok, c.margin = checks[c.token]
            if not ok:
                self.drop(c, "correctness: FAIL")
        self.correct_pool = r.get("pool") or "timing"

    def stage_time(self, which: str, cands: list[Cand], args: str) -> dict[str, float]:
        tokens = ([self.best.token] if self.best else []) + [c.token for c in cands]
        # borderline correctness on the cheap pool: recheck on the timing pool while we're timing anyway
        recheck = [c for c in cands if (c.margin or 0) > runner.NEAR_BOUND and getattr(self, "correct_pool", "") != "timing"]
        extra = f"{args} {self.check_flag}" if which == "full" and recheck else args
        r = self.job(build_cmd(self.bench, tokens, extra), stage=which)
        everyone = cands + ([self.best] if self.best else [])
        scores = per_candidate_scores(self.log_of(r), everyone)
        if recheck:
            checks = per_candidate_checks(self.log_of(r), recheck)
            for c in recheck:
                if c.token in checks and not checks[c.token][0]:
                    self.drop(c, "correctness: FAIL on recheck (timing pool)")
        for c in cands:
            c.jobs[which] = r["id"]
            if c.token not in scores and not c.dropped:
                self.drop(c, f"{which}: no SCORE output ({r.get('failure') or 'missing'})")
        if self.best:
            self.best.jobs[which] = r["id"]
        return scores

    def stage_quick(self) -> None:
        cands = self.alive()
        if len(cands) <= self.top_k:  # nothing to prune: skip straight to full reps
            return
        self.con.print(f"[cyan]③ quick timing[/] best + {len(cands)} candidates, few reps (timing pool)")
        scores = self.stage_time("quick", cands, self.quick_args)
        for c in cands:
            c.stage, c.quick = "quick", scores.get(c.token)
        if self.best:
            self.best.quick = scores.get(self.best.token)
        finalists = set(select_finalists({c.token: c.quick for c in self.alive() if c.quick is not None},
                                         None, self.top_k, self.noise))
        for c in self.alive():
            if c.token not in finalists:
                self.drop(c, f"quick: not in top {self.top_k} (or within {self.noise:.1%} of the leader)")

    def stage_full(self) -> None:
        cands = self.alive()
        if not cands:
            return
        self.con.print(f"[cyan]④ full timing[/] best + {len(cands)} finalist(s), full reps (timing pool)")
        scores = self.stage_time("full", cands, self.full_args)
        if self.best:
            self.best.full = scores.get(self.best.token)
        for c in cands:
            c.stage, c.full = "full", scores.get(c.token)
            if c.full and self.best and self.best.full:
                c.vs_best = c.full / self.best.full

    # ---- run + report
    def run(self) -> dict:
        t0 = time.time()
        for stage in (self.stage_compile, self.stage_correct, self.stage_quick, self.stage_full):
            stage()
        ranked = sorted(self.cands, key=lambda c: (c.full is not None, c.full or 0, c.quick or 0), reverse=True)
        winner = next((c for c in ranked if c.vs_best and c.vs_best - 1 > self.noise), None) if self.best else \
            next((c for c in ranked if c.full), None)
        return {
            "sweep": self.id, "track": self.track.name, "wall_s": round(time.time() - t0, 1),
            "cost_usd": round(self.cost, 4), "noise": self.noise,
            "best": {"token": self.best.token, "quick": self.best.quick, "full": self.best.full} if self.best else None,
            "recommend": winner.token if winner else None,
            "verdict": (f"{winner.token} beats {self.best.token} by {winner.vs_best - 1:+.2%} (> {self.noise:.1%} noise)"
                        if winner and self.best else (f"best candidate: {winner.token}" if winner else
                                                       "no candidate beat the current best by more than noise")),
            "candidates": [{"token": c.token, "stage": c.stage, "dropped": c.dropped, "check_margin": c.margin,
                            "quick": c.quick, "full": c.full,
                            "vs_best": round(c.vs_best, 4) if c.vs_best else None, "jobs": c.jobs} for c in ranked],
        }


def print_report(con: Console, rep: dict) -> None:
    t = Table(box=None, header_style="bold magenta", pad_edge=False)
    for col in ("#", "candidate", "reached", "check margin", "quick", "full", "vs best", "dropped because"):
        t.add_column(col, overflow="fold")
    for i, c in enumerate(rep["candidates"], 1):
        vs = c["vs_best"]
        vs_txt = (f"[bold green]{vs - 1:+.2%}[/]" if vs and vs - 1 > rep["noise"] else
                  f"{vs - 1:+.2%}" if vs else "")
        t.add_row(str(i), c["token"], c["stage"], f"{c['check_margin']:.0%}" if c["check_margin"] is not None else "",
                  f"{c['quick']:.4f}" if c["quick"] else "", f"{c['full']:.4f}" if c["full"] else "", vs_txt,
                  f"[grey50]{c['dropped']}[/]" if c["dropped"] else "")
    con.print(t)
    if rep["best"]:
        b = rep["best"]
        con.print(f"[grey50]best {b['token']}: quick {b['quick'] or '—'} · full {b['full'] or '—'}[/]")
    mark = "[bold green]★[/]" if rep["recommend"] else "[yellow]·[/]"
    con.print(f"{mark} {rep['verdict']}   [grey50]{rep['wall_s']:.0f}s · ${rep['cost_usd']:.2f} · {rep['sweep']}[/]")
