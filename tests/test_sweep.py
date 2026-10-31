from pathlib import Path

from rich.console import Console

from ogpu import config, sweep


def test_match_handles_paths_variants_and_index_prefix():
    cands = [sweep.Cand("cand/a.py"), sweep.Cand("sub.py@PF_ON=False"), sweep.Cand("cand/best.py")]
    assert sweep.match("a.py", cands).token == "cand/a.py"
    assert sweep.match("sub.py@PF_ON=False", cands).token == "sub.py@PF_ON=False"
    assert sweep.match("2:best.py", cands).token == "cand/best.py"  # attention bench prefixes duplicates
    assert sweep.match("zzz.py", cands) is None


def test_per_candidate_checks_and_scores():
    cands = [sweep.Cand("cand/a.py"), sweep.Cand("cand/b.py")]
    log = ("CHECK d1s a.py judge: PASS relative L2 error 1.0e-02 (worst row 1e-2) [bound 0.100]\n"
           "CHECK d1f a.py seed12: PASS relative L2 error 4.0e-02 (worst row 4e-2) [bound 0.050]\n"
           "CHECK d1s b.py judge: FAIL relative L2 error 2.0e-01 (worst row 2e-1) [bound 0.100]\n"
           "SCORE a.py: 1.0712  d1=1.0\nSCORE b.py: 1.2000\n")
    checks = sweep.per_candidate_checks(log, cands)
    assert checks["cand/a.py"][0] is True and abs(checks["cand/a.py"][1] - 0.8) < 1e-9
    assert checks["cand/b.py"][0] is False
    assert sweep.per_candidate_scores(log, cands) == {"cand/a.py": 1.0712, "cand/b.py": 1.2}


def test_select_finalists_keeps_top_k_and_near_leader():
    scores = {"a": 1.10, "b": 1.095, "c": 1.05, "d": 1.00, "best": 1.2}
    assert sweep.select_finalists(scores, "best", 1, 0.01) == ["a", "b"]  # b is within 1% of the leader
    assert sweep.select_finalists(scores, "best", 3, 0.0) == ["a", "b", "c"]


def test_build_cmd():
    assert sweep.build_cmd("python3 pod/bench.py", ["a.py", "b.py@X=(1,2)"], "--reps 3") == \
        "python3 pod/bench.py a.py 'b.py@X=(1,2)' --reps 3"
    assert sweep.build_cmd("python3 run.py {files} --fast", ["a.py"], "") == "python3 run.py a.py --fast"


def test_full_tournament_flow(monkeypatch, tmp_path):
    """5 candidates: one wrong, quick timing prunes to top-2, full timing names a winner vs best."""
    logs = {
        "correctness": "".join(f"CHECK d1s {n}.py judge: {'FAIL' if n == 'bad' else 'PASS'} relative L2 error "
                               f"1e-2 (w) [bound 0.1]\n" for n in ("a", "b", "c", "d", "bad")),
        "quick": "SCORE best.py: 1.000\nSCORE a.py: 1.060\nSCORE b.py: 1.020\nSCORE c.py: 1.055\nSCORE d.py: 0.990\n",
        "full": "SCORE best.py: 1.000\nSCORE a.py: 1.031\nSCORE c.py: 1.008\n",
    }
    calls = []

    def fake_job(self, cmd, *, mode="gpu", fidelity="timing", stage):
        calls.append((stage, fidelity, cmd))
        self.cost += 0.1  # the real job() accumulates cost; the fake must too
        p = tmp_path / f"{stage}.log"
        p.write_text(logs[stage])
        return {"id": stage, "status": "done", "log_path": str(p), "cost_usd": 0.1,
                "pool": "correctness" if fidelity == "correctness" else "timing"}

    monkeypatch.setattr(sweep.Sweep, "job", fake_job)
    run = sweep.Sweep(config.Config(), Console(quiet=True), track=Path("."), extra=[],
                      cands=["a.py", "b.py", "c.py", "d.py", "bad.py"], best="best.py", bench="python3 pod/bench.py",
                      compile_tpl=None, check_flag="--check", correct_args="--reps 3", quick_args="--reps 5",
                      full_args="--reps 20", top_k=2, noise=0.015, note="test", timeout_s=None, force=False)
    rep = run.run()
    assert [c[0] for c in calls] == ["correctness", "quick", "full"]
    assert calls[0][1] == "correctness" and "best.py" not in calls[0][2]   # best isn't re-checked
    assert "best.py" in calls[1][2] and "bad.py" not in calls[1][2]          # wrong candidates never get timed
    assert "a.py" in calls[2][2] and "c.py" in calls[2][2] and "b.py" not in calls[2][2]
    by = {c["token"]: c for c in rep["candidates"]}
    assert by["bad.py"]["dropped"] == "correctness: FAIL"
    assert by["b.py"]["dropped"].startswith("quick:") and by["d.py"]["dropped"].startswith("quick:")
    assert rep["recommend"] == "a.py" and by["a.py"]["vs_best"] == 1.031
    assert by["c.py"]["dropped"] is None and rep["candidates"][0]["token"] == "a.py"
    assert "beats best.py by +3.10%" in rep["verdict"]
    assert abs(rep["cost_usd"] - 0.3) < 1e-9


def test_no_winner_when_inside_noise(monkeypatch, tmp_path):
    def fake_job(self, cmd, *, mode="gpu", fidelity="timing", stage):
        p = tmp_path / f"{stage}.log"
        p.write_text("CHECK d1s a.py judge: PASS x\n" if stage == "correctness" else "SCORE best.py: 1.0\nSCORE a.py: 1.01\n")
        return {"id": stage, "status": "done", "log_path": str(p), "cost_usd": 0.0, "pool": "timing"}
    monkeypatch.setattr(sweep.Sweep, "job", fake_job)
    rep = sweep.Sweep(config.Config(), Console(quiet=True), track=Path("."), extra=[], cands=["a.py"], best="best.py",
                      bench="b", compile_tpl=None, check_flag="--check", correct_args="", quick_args="", full_args="",
                      top_k=2, noise=0.015, note="", timeout_s=None, force=False).run()
    assert rep["recommend"] is None and "no candidate beat" in rep["verdict"]
