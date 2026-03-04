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
REQUIRED_TASK_FILES = (INITIAL_FILE, EVAL_FILE, "config.yaml", "requirements.txt")
DEFAULT_EVAL_TIMEOUT_SECONDS = 360.0
RUN_METADATA_FILE = "run_config.yaml"
BEST_REPO_DIRNAME = "best"
MAX_CONSECUTIVE_FAILURES = 6
STOP_MODE_GPT_CALLS = "max_gpt_calls"
STOP_MODE_EVALS = "max_evals"
FINGERPRINT_IGNORE_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
COPYTREE_IGNORE_GLOBS = (
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "*.pyc",
    "*.pyo",
)
CODER_ALLOWED_BASH_COMMANDS = [
    "python",
    "python3",
    "ls",
    "rg",
    "cat",
    "sed",
    "head",
    "tail",
    "wc",
]
MONITOR_ALLOWED_BASH_COMMANDS = [
    "python",
    "python3",
    "ls",
    "rg",
    "cat",
    "sed",
    "head",
    "tail",
    "wc",
]
CODER_TASK = (
    "You are the Coding Agent in a coevolving loop.\n"
    "Work dir: {repo_dir}\nLog file: {log_path}\n"
    f"Improve {INITIAL_FILE} (and related files) to maximize Evaluate() score. "
    "Use Evaluate() for scoring, ReadLogTail() for log inspection, "
    "and RepoSearch() for code search. "
    "Use Bash only for simple commands, no shell operators or inline scripts. "
    f"Edit only {INITIAL_FILE} and coder_notes.md unless evaluator import/runtime "
    "errors require a minimal fix. "
    "Do not create helper runner files. "
    "Keep iterating on plausible hypotheses instead of stopping on tiny deltas alone. "
    "Keep coder_notes.md with tried ideas/scores, and call finish with concise progress when done."
)
MONITOR_TASK = (
    "You are the Monitor/Optimizer in a coevolving loop.\n"
    "Work dir: {repo_dir}\nLog file: {log_path}\n"
    "Use Evaluate() for scoring, ReadLogTail() for log inspection, "
    "and RepoSearch() for code search. "
    "Bash is allowed only for simple non-chained commands, "
    "with no shell operators or inline scripts. "
    f"Default to read-only monitoring. Edit {INITIAL_FILE} and monitor_notes.md when "
    "you have a plausible improvement path and justify intervention. "
    "If intervention is not worthwhile, explicitly skip with rationale. "
    "Do not create helper runner files. Track tried ideas in monitor_notes.md and call finish "
    "with concise decision and latest score."
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
    exp_root = (Path.cwd() / "exp").resolve()
    base_exp_dir = exp_root / f"{task_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = base_exp_dir
    suffix = 1
    while exp_dir.exists():
        exp_dir = exp_root / f"{base_exp_dir.name}_{suffix:02d}"
        suffix += 1
    repo_dir, log_path = exp_dir / "repo", exp_dir / "log.jsonl"
    exp_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(task_dir, repo_dir, ignore=shutil.ignore_patterns(*COPYTREE_IGNORE_GLOBS))
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
    shutil.copytree(repo_dir, tmp_dir, ignore=shutil.ignore_patterns(*COPYTREE_IGNORE_GLOBS))
    if best_repo_dir.exists():
        shutil.rmtree(best_repo_dir, ignore_errors=True)
    tmp_dir.rename(best_repo_dir)


def resolve_stop_condition(
    max_gpt_calls: int | None,
    max_evals: int | None,
) -> tuple[str, int]:
    has_gpt_limit = max_gpt_calls is not None
    has_eval_limit = max_evals is not None
    if has_gpt_limit == has_eval_limit:
        raise ValueError("Exactly one of max_gpt_calls or max_evals must be provided.")
    if has_gpt_limit:
        if max_gpt_calls <= 0:
            raise ValueError("max_gpt_calls must be a positive integer.")
        return STOP_MODE_GPT_CALLS, max_gpt_calls
    if max_evals is None or max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")
    return STOP_MODE_EVALS, max_evals


def get_stop_reason(
    stop_mode: str,
    stop_limit: int,
    total_gpt_calls: int,
    total_eval_calls: int,
) -> str | None:
    if stop_mode == STOP_MODE_GPT_CALLS and total_gpt_calls >= stop_limit:
        return f"{STOP_MODE_GPT_CALLS} reached ({total_gpt_calls}/{stop_limit})"
    if stop_mode == STOP_MODE_EVALS and total_eval_calls >= stop_limit:
        return f"{STOP_MODE_EVALS} reached ({total_eval_calls}/{stop_limit})"
    return None


def run_agent_turn(
    agent: RelentlessCodingAgent,
    prompt: str,
    model_name: str,
    repo_dir: Path,
    log_path: Path,
    max_steps_per_session: int,
    max_sub_sessions: int,
    max_total_model_calls: int | None = None,
    allowed_bash_commands: list[str] | None = None,
) -> tuple[bool, str, float, int]:
    if max_total_model_calls is not None and max_total_model_calls <= 0:
        return False, "No GPT calls remaining", 0.0, 0
    try:
        result = agent.run(
            prompt_template=prompt,
            model_name=model_name,
            max_steps=max_steps_per_session,
            max_budget=float("inf"),
            max_total_model_calls=max_total_model_calls,
            max_sub_sessions=max_sub_sessions,
            work_dir=str(repo_dir.resolve()),
            readable_paths=[str(log_path.resolve())],
            writable_paths=[str(repo_dir.resolve())],
            allowed_bash_commands=allowed_bash_commands,
            strict_bash=True,
            enable_repo_helper_tools=True,
            enable_bash_tool=True,
        )
    except Exception as exc:
        return False, str(exc), agent.budget_used, agent.total_model_calls
    try:
        payload = yaml.safe_load(result)
    except yaml.YAMLError:
        return False, result, agent.budget_used, agent.total_model_calls
    if isinstance(payload, dict):
        return (
            bool(payload.get("success", False)),
            str(payload.get("summary", "")),
            agent.budget_used,
            agent.total_model_calls,
        )
    return False, result, agent.budget_used, agent.total_model_calls


def evolve(
    task_dir: Path,
    model_name: str,
    max_gpt_calls: int | None,
    max_evals: int | None,
    max_steps_per_session: int,
    max_sub_sessions: int,
    eval_timeout: float = DEFAULT_EVAL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    stop_mode, stop_limit = resolve_stop_condition(max_gpt_calls, max_evals)
    exp_dir, repo_dir, log_path = prepare_experiment(task_dir)
    run_metadata_path = write_run_metadata(
        exp_dir,
        {
            "task_dir": str(task_dir),
            "model_name": model_name,
            "max_gpt_calls": max_gpt_calls,
            "max_evals": max_evals,
            "stop_mode": stop_mode,
            "stop_limit": stop_limit,
            "max_steps_per_session": max_steps_per_session,
            "max_sub_sessions": max_sub_sessions,
            "eval_timeout": eval_timeout,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    coder, monitor = RelentlessCodingAgent("Coder"), RelentlessCodingAgent("Monitor")
    coder_prompt = CODER_TASK.format(
        repo_dir=repo_dir,
        log_path=log_path,
    )
    monitor_prompt = MONITOR_TASK.format(
        repo_dir=repo_dir,
        log_path=log_path,
    )
    start_time, total_budget = time.time(), 0.0
    total_gpt_calls = 0
    total_eval_calls = 0
    old_global_max_budget = config_module.DEFAULT_CONFIG.agent.global_max_budget
    config_module.DEFAULT_CONFIG.agent.global_max_budget = float("inf")
    score: float | None = None
    best_score: float | None = None
    reason: str | None = None
    coder_fail_streak = 0
    monitor_fail_streak = 0
    best_repo_dir = exp_dir / BEST_REPO_DIRNAME
    try:
        score, eval_msg = run_eval(repo_dir, eval_timeout=eval_timeout)
        total_eval_calls += 1
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
        snapshot_best_repo(repo_dir, best_repo_dir)
        reason = get_stop_reason(
            stop_mode,
            stop_limit,
            total_gpt_calls,
            total_eval_calls,
        )
        while reason is None:
            append_log(
                log_path,
                "coder",
                "run",
                score,
                total_budget,
                "starting coder turn",
                budget_delta=0.0,
            )
            remaining_model_calls = (
                stop_limit - total_gpt_calls
                if stop_mode == STOP_MODE_GPT_CALLS
                else None
            )
            ok, summary, cost, gpt_calls = run_agent_turn(
                coder,
                coder_prompt,
                model_name,
                repo_dir,
                log_path,
                max_steps_per_session,
                max_sub_sessions,
                max_total_model_calls=remaining_model_calls,
                allowed_bash_commands=CODER_ALLOWED_BASH_COMMANDS,
            )
            total_budget += cost
            total_gpt_calls += gpt_calls
            coder_fail_streak = 0 if ok else coder_fail_streak + 1
            score, eval_msg = run_eval(repo_dir, eval_timeout=eval_timeout)
            total_eval_calls += 1
            append_log(
                log_path,
                "coder",
                "eval",
                score,
                total_budget,
                f"success={ok}; gpt_calls={gpt_calls}; {summary}; {eval_msg}",
                budget_delta=cost,
            )
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                snapshot_best_repo(repo_dir, best_repo_dir)
            if coder_fail_streak >= MAX_CONSECUTIVE_FAILURES:
                reason = f"max_consecutive_failures(coder) reached ({coder_fail_streak})"
                break
            reason = get_stop_reason(
                stop_mode,
                stop_limit,
                total_gpt_calls,
                total_eval_calls,
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
            remaining_model_calls = (
                stop_limit - total_gpt_calls
                if stop_mode == STOP_MODE_GPT_CALLS
                else None
            )
            ok, summary, cost, gpt_calls = run_agent_turn(
                monitor,
                monitor_prompt,
                model_name,
                repo_dir,
                log_path,
                max_steps_per_session,
                max_sub_sessions,
                max_total_model_calls=remaining_model_calls,
                allowed_bash_commands=MONITOR_ALLOWED_BASH_COMMANDS,
            )
            total_budget += cost
            total_gpt_calls += gpt_calls
            monitor_fail_streak = 0 if ok else monitor_fail_streak + 1
            action = "optimize" if repo_fingerprint(repo_dir) != before else "eval"
            score, eval_msg = run_eval(repo_dir, eval_timeout=eval_timeout)
            total_eval_calls += 1
            append_log(
                log_path,
                "monitor",
                action,
                score,
                total_budget,
                f"success={ok}; gpt_calls={gpt_calls}; {summary}; {eval_msg}",
                budget_delta=cost,
            )
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                snapshot_best_repo(repo_dir, best_repo_dir)
            if monitor_fail_streak >= MAX_CONSECUTIVE_FAILURES:
                reason = f"max_consecutive_failures(monitor) reached ({monitor_fail_streak})"
                break
            reason = get_stop_reason(
                stop_mode,
                stop_limit,
                total_gpt_calls,
                total_eval_calls,
            )
    finally:
        config_module.DEFAULT_CONFIG.agent.global_max_budget = old_global_max_budget

    stop_reason = reason or get_stop_reason(
        stop_mode,
        stop_limit,
        total_gpt_calls,
        total_eval_calls,
    )
    append_log(
        log_path,
        "system",
        "stop",
        score,
        total_budget,
        (
            f"stop_reason={stop_reason}; "
            f"total_gpt_calls={total_gpt_calls}; total_eval_calls={total_eval_calls}"
        ),
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
        "total_gpt_calls": total_gpt_calls,
        "total_eval_calls": total_eval_calls,
        "stop_mode": stop_mode,
        "stop_limit": stop_limit,
        "elapsed_time": round(time.time() - start_time, 2),
        "stop_reason": stop_reason,
    }


def main() -> None:
    cfg = config_module.DEFAULT_CONFIG.coding_agent.evolve_optimizer
    parser = argparse.ArgumentParser(description="Coevolving optimizer for alpha-math-evolve tasks")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--model-name", default=cfg.model_name)
    limit_group = parser.add_mutually_exclusive_group(required=True)
    limit_group.add_argument("--max-gpt-calls", type=int)
    limit_group.add_argument("--max-evals", type=int)
    parser.add_argument("--max-steps-per-session", type=int, default=cfg.max_steps_per_session)
    parser.add_argument("--max-sub-sessions", type=int, default=cfg.max_sub_sessions)
    parser.add_argument("--eval-timeout", type=float, default=DEFAULT_EVAL_TIMEOUT_SECONDS)
    args = parser.parse_args()
    print(
        yaml.dump(
            evolve(
                task_dir=Path(args.task_dir).resolve(),
                model_name=args.model_name,
                max_gpt_calls=args.max_gpt_calls,
                max_evals=args.max_evals,
                max_steps_per_session=args.max_steps_per_session,
                max_sub_sessions=args.max_sub_sessions,
                eval_timeout=args.eval_timeout,
            ),
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
