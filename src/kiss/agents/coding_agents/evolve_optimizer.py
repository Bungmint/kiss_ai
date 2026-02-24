"""Coevolving coder/monitor optimizer for alpha-math-evolve tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

import kiss.agents.coding_agents.config as _coding_config  # noqa: F401
from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent
from kiss.core import config as config_module

INITIAL_FILE = "initial_program.py"
EVAL_FILE = "evaluator.py"
MAX_CONSECUTIVE_FAILURES = 6
REQUIRED_TASK_FILES = (INITIAL_FILE, EVAL_FILE, "config.yaml", "requirements.txt")
DEFAULT_EVAL_TIMEOUT_SECONDS = 360.0
RUN_METADATA_FILE = "run_config.yaml"
BEST_REPO_DIRNAME = "best"
FINGERPRINT_IGNORE_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
CODER_TASK = (
    "You are the Coding Agent in a coevolving loop.\n"
    "Work dir: {repo_dir}\nLog file: {log_path}\n"
    "Target score: {target_score}\nTarget policy: {target_policy}\n"
    f"Improve {INITIAL_FILE} (and related files) to maximize `python {EVAL_FILE}` score. "
    "Run eval frequently, keep coder_notes.md with tried ideas/scores, and call finish "
    "with concise progress when done for this turn."
)
MONITOR_TASK = (
    "You are the Monitor/Optimizer in a coevolving loop.\n"
    "Work dir: {repo_dir}\nLog file: {log_path}\n"
    "Target score: {target_score}\nTarget policy: {target_policy}\n"
    f"Run `python {EVAL_FILE}` and monitor output in real time. Review log/code trend and decide "
    "autonomously whether optimization is worthwhile. If worthwhile, edit code and rerun eval; "
    "if not, explicitly skip intervention and explain why. Track tried ideas in monitor_notes.md "
    "and call finish with concise decision and latest score."
)


def append_log(
    path: Path,
    agent: str,
    action: str,
    eval_score: float | None,
    budget_used: float,
    summary: str,
    budget_delta: float | None = None,
) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "agent": agent,
        "action": action,
        "eval_score": eval_score,
        "budget_used": round(budget_used, 6),
        "summary": summary,
    }
    if budget_delta is not None:
        entry["budget_delta"] = round(budget_delta, 6)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _truncate_output(value: str | bytes | None, max_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value.strip()[:max_chars]


def run_eval(
    repo_dir: Path,
    eval_timeout: float = DEFAULT_EVAL_TIMEOUT_SECONDS,
) -> tuple[float | None, str]:
    proc: subprocess.CompletedProcess[str] | None
    proc_err = ""
    try:
        proc = subprocess.run(
            ["python", EVAL_FILE],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=eval_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        proc = None
        main_stdout = _truncate_output(exc.stdout, 120)
        main_stderr = _truncate_output(exc.stderr, 120)
        proc_err = (
            f"timeout after {eval_timeout:.1f}s, "
            f"stdout={main_stdout}, stderr={main_stderr}"
        )
    stdout = proc.stdout.strip() if proc is not None else ""
    lines = [x.strip() for x in stdout.splitlines() if x.strip()]
    score: float | None = None
    for line in reversed(lines):
        try:
            score = float(line)
            break
        except ValueError:
            continue
    if score is not None:
        msg = (
            f"eval_main rc={proc.returncode}, stdout={stdout[:200]}, "
            f"stderr={proc.stderr.strip()[:200]}; fallback_used=False"
        )
        return score, msg
    cmd = (
        "from evaluator import evaluate; "
        f"result = evaluate('{INITIAL_FILE}'); "
        "print(result.get('combined_score', 0.0))"
    )
    proc2: subprocess.CompletedProcess[str] | None
    proc2_err = ""
    try:
        proc2 = subprocess.run(
            ["python", "-c", cmd],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=eval_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        proc2 = None
        fallback_stdout = _truncate_output(exc.stdout, 120)
        fallback_stderr = _truncate_output(exc.stderr, 120)
        proc2_err = (
            f"timeout after {eval_timeout:.1f}s, "
            f"stdout={fallback_stdout}, stderr={fallback_stderr}"
        )
    out2 = proc2.stdout.strip() if proc2 is not None else ""
    try:
        score = (
            float(out2.splitlines()[-1])
            if proc2 is not None and proc2.returncode == 0 and out2
            else None
        )
    except ValueError:
        score = None
    main_status = (
        f"rc={proc.returncode}, stdout={stdout[:120]}, stderr={proc.stderr.strip()[:120]}"
        if proc is not None
        else proc_err
    )
    fallback_status = (
        f"rc={proc2.returncode}, stdout={out2[:120]}, stderr={proc2.stderr.strip()[:120]}"
        if proc2 is not None
        else proc2_err
    )
    msg = (
        f"eval_main {main_status}; "
        f"eval_func {fallback_status}; fallback_used=True"
    )
    return score, msg


def repo_fingerprint(repo_dir: Path) -> str:
    h = hashlib.sha256()
    files = []
    for p in repo_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(repo_dir)
        if any(part in FINGERPRINT_IGNORE_PARTS for part in rel.parts):
            continue
        files.append((p, rel))
    for p, rel in sorted(files, key=lambda item: item[1].as_posix()):
        try:
            stat = p.stat()
        except OSError:
            continue
        h.update(rel.as_posix().encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(stat.st_mtime_ns).encode("utf-8"))
    return h.hexdigest()


def prepare_experiment(task_dir: Path) -> tuple[Path, Path, Path]:
    missing = [n for n in REQUIRED_TASK_FILES if not (task_dir / n).is_file()]
    if missing:
        raise ValueError(f"Task directory missing required files: {missing}")
    base_exp_dir = Path("exp") / f"{task_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = base_exp_dir
    suffix = 1
    while exp_dir.exists():
        exp_dir = Path(f"{base_exp_dir}_{suffix:02d}")
        suffix += 1
    repo_dir, log_path = exp_dir / "repo", exp_dir / "log.jsonl"
    exp_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(task_dir, repo_dir)
    log_path.touch()
    return exp_dir, repo_dir, log_path


def write_run_metadata(exp_dir: Path, metadata: dict[str, Any]) -> Path:
    metadata_path = exp_dir / RUN_METADATA_FILE
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return metadata_path


def snapshot_best_repo(repo_dir: Path, best_repo_dir: Path) -> None:
    tmp_dir = best_repo_dir.with_name(f"{best_repo_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.copytree(repo_dir, tmp_dir)
    if best_repo_dir.exists():
        shutil.rmtree(best_repo_dir, ignore_errors=True)
    tmp_dir.rename(best_repo_dir)


def get_stop_reason(
    start_time: float,
    score: float | None,
    target_score: float | None,
    stop_on_target_score: bool,
    budget_used: float,
    max_budget: float,
    max_time: float,
) -> str | None:
    elapsed = time.time() - start_time
    if budget_used >= max_budget:
        return f"max_budget reached ({budget_used:.4f}/{max_budget:.4f})"
    if elapsed >= max_time:
        return f"max_time reached ({elapsed:.1f}/{max_time:.1f}s)"
    if (
        stop_on_target_score
        and target_score is not None
        and score is not None
        and score >= target_score
    ):
        return f"target_score reached ({score:.6f} >= {target_score:.6f})"
    return None


def run_agent_turn(
    agent: RelentlessCodingAgent,
    prompt: str,
    model_name: str,
    repo_dir: Path,
    log_path: Path,
    max_steps_per_session: int,
    max_sub_sessions: int,
    remaining_budget: float,
) -> tuple[bool, str, float]:
    if remaining_budget <= 0:
        return False, "No budget remaining", 0.0
    try:
        result = agent.run(
            prompt_template=prompt,
            model_name=model_name,
            max_steps=max_steps_per_session,
            max_budget=remaining_budget,
            max_sub_sessions=max_sub_sessions,
            work_dir=str(repo_dir),
            readable_paths=[str(log_path)],
            writable_paths=[str(repo_dir)],
        )
    except Exception as exc:
        return False, str(exc), agent.budget_used
    try:
        payload = yaml.safe_load(result)
    except yaml.YAMLError:
        return False, result, agent.budget_used
    if isinstance(payload, dict):
        return (
            bool(payload.get("success", False)),
            str(payload.get("summary", "")),
            agent.budget_used,
        )
    return False, result, agent.budget_used


def evolve(
    task_dir: Path,
    model_name: str,
    max_budget: float,
    max_time: float,
    max_steps_per_session: int,
    max_sub_sessions: int,
    target_score: float | None,
    stop_on_target_score: bool = False,
    eval_timeout: float = DEFAULT_EVAL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    exp_dir, repo_dir, log_path = prepare_experiment(task_dir)
    run_metadata_path = write_run_metadata(
        exp_dir,
        {
            "task_dir": str(task_dir),
            "model_name": model_name,
            "max_budget": max_budget,
            "max_time": max_time,
            "max_steps_per_session": max_steps_per_session,
            "max_sub_sessions": max_sub_sessions,
            "target_score": target_score,
            "stop_on_target_score": stop_on_target_score,
            "eval_timeout": eval_timeout,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    coder, monitor = RelentlessCodingAgent("Coder"), RelentlessCodingAgent("Monitor")
    target_text = "none" if target_score is None else f"{target_score:.6f}"
    target_policy = (
        "hard-stop run when target_score is reached"
        if stop_on_target_score
        else "milestone only; continue optimizing until budget/time/failure stop conditions"
    )
    coder_prompt = CODER_TASK.format(
        repo_dir=repo_dir,
        log_path=log_path,
        target_score=target_text,
        target_policy=target_policy,
    )
    monitor_prompt = MONITOR_TASK.format(
        repo_dir=repo_dir,
        log_path=log_path,
        target_score=target_text,
        target_policy=target_policy,
    )
    start_time, total_budget = time.time(), 0.0
    score, eval_msg = run_eval(repo_dir, eval_timeout=eval_timeout)
    append_log(
        log_path,
        "monitor",
        "eval",
        score,
        total_budget,
        f"baseline eval; {eval_msg}",
        budget_delta=0.0,
    )
    best_score = score
    best_repo_dir = exp_dir / BEST_REPO_DIRNAME
    snapshot_best_repo(repo_dir, best_repo_dir)

    reason = None
    coder_fail_streak = 0
    monitor_fail_streak = 0
    while True:
        reason = get_stop_reason(
            start_time,
            score,
            target_score,
            stop_on_target_score,
            total_budget,
            max_budget,
            max_time,
        )
        if reason is not None:
            break
        append_log(
            log_path,
            "coder",
            "run",
            score,
            total_budget,
            "starting coder turn",
            budget_delta=0.0,
        )
        ok, summary, cost = run_agent_turn(
            coder,
            coder_prompt,
            model_name,
            repo_dir,
            log_path,
            max_steps_per_session,
            max_sub_sessions,
            max_budget - total_budget,
        )
        coder_fail_streak = 0 if ok else coder_fail_streak + 1
        total_budget += cost
        score, eval_msg = run_eval(repo_dir, eval_timeout=eval_timeout)
        append_log(
            log_path,
            "coder",
            "eval",
            score,
            total_budget,
            f"success={ok}; {summary}; {eval_msg}",
            budget_delta=cost,
        )
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            snapshot_best_repo(repo_dir, best_repo_dir)
        if coder_fail_streak >= MAX_CONSECUTIVE_FAILURES:
            reason = f"max_consecutive_failures(coder) reached ({coder_fail_streak})"
            break
        reason = get_stop_reason(
            start_time,
            score,
            target_score,
            stop_on_target_score,
            total_budget,
            max_budget,
            max_time,
        )
        if reason is not None:
            break
        before = repo_fingerprint(repo_dir)
        append_log(
            log_path,
            "monitor",
            "run",
            score,
            total_budget,
            "starting monitor turn",
            budget_delta=0.0,
        )
        ok, summary, cost = run_agent_turn(
            monitor,
            monitor_prompt,
            model_name,
            repo_dir,
            log_path,
            max_steps_per_session,
            max_sub_sessions,
            max_budget - total_budget,
        )
        monitor_fail_streak = 0 if ok else monitor_fail_streak + 1
        total_budget += cost
        action = "optimize" if repo_fingerprint(repo_dir) != before else "eval"
        score, eval_msg = run_eval(repo_dir, eval_timeout=eval_timeout)
        append_log(
            log_path,
            "monitor",
            action,
            score,
            total_budget,
            f"success={ok}; {summary}; {eval_msg}",
            budget_delta=cost,
        )
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            snapshot_best_repo(repo_dir, best_repo_dir)
        if monitor_fail_streak >= MAX_CONSECUTIVE_FAILURES:
            reason = f"max_consecutive_failures(monitor) reached ({monitor_fail_streak})"
            break

    stop_reason = reason or get_stop_reason(
        start_time,
        score,
        target_score,
        stop_on_target_score,
        total_budget,
        max_budget,
        max_time,
    )
    append_log(
        log_path,
        "system",
        "stop",
        score,
        total_budget,
        f"stop_reason={stop_reason}",
        budget_delta=0.0,
    )
    return {
        "exp_dir": str(exp_dir),
        "repo_dir": str(repo_dir),
        "log_path": str(log_path),
        "run_metadata_path": str(run_metadata_path),
        "best_repo_dir": str(best_repo_dir),
        "best_score": best_score,
        "final_score": score,
        "budget_used": round(total_budget, 6),
        "elapsed_time": round(time.time() - start_time, 2),
        "stop_reason": stop_reason,
    }


def main() -> None:
    cfg = config_module.DEFAULT_CONFIG.coding_agent.evolve_optimizer
    parser = argparse.ArgumentParser(description="Coevolving optimizer for alpha-math-evolve tasks")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument("--max-budget", type=float, default=cfg.max_budget)
    parser.add_argument("--max-time", type=float, default=cfg.max_time)
    parser.add_argument("--target-score", type=float, default=None)
    parser.add_argument(
        "--stop-on-target-score",
        action=argparse.BooleanOptionalAction,
        default=cfg.stop_on_target_score,
        help=(
            "Stop immediately once target score is reached. "
            "Use --no-stop-on-target-score to keep running until budget/time/failure limits."
        ),
    )
    parser.add_argument("--max-steps-per-session", type=int, default=cfg.max_steps_per_session)
    parser.add_argument("--max-sub-sessions", type=int, default=cfg.max_sub_sessions)
    parser.add_argument("--eval-timeout", type=float, default=DEFAULT_EVAL_TIMEOUT_SECONDS)
    args = parser.parse_args()
    print(
        yaml.dump(
            evolve(
                task_dir=Path(args.task_dir).resolve(),
                model_name=args.model_name,
                max_budget=args.max_budget,
                max_time=args.max_time,
                max_steps_per_session=args.max_steps_per_session,
                max_sub_sessions=args.max_sub_sessions,
                target_score=args.target_score,
                stop_on_target_score=args.stop_on_target_score,
                eval_timeout=args.eval_timeout,
            ),
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
