"""Iteratively optimize coding agents based on task descriptions.

Simple orchestrator that delegates to RelentlessCodingAgent, benchmarks,
and iteratively improves for success, speed, cost, and token efficiency.
"""

import ast
import importlib.util
import os
import py_compile
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from kiss.core.base import Base
from kiss.core.compact_formatter import CompactFormatter
from kiss.core.kiss_agent import KISSAgent
from kiss.core.useful_tools import SAFARI_HEADERS, UsefulTools, fetch_url

THEMES = [
    "Optimize TASK_PROMPT: concise, directive, step-by-step plan. "
    "Improve batching. Remove redundant text.",
    "Improve continuation: structured progress, smart step thresholds, "
    "progressive summarization for 10K+ steps.",
    "Reduce tokens: compress prompts, shorter progress, minimize overhead.",
    "Alternative architectures: hierarchical planner+executor, sub-task "
    "decomposition, Write tool for files, better error handling.",
    "Final tuning: finish conditions, trials/steps ratio, edge-case "
    "robustness for very long tasks, cleanup dead code.",
]

FALLBACK_STRATEGIES = """Key strategies for long-running AI coding agents:
1. Concise prompts save tokens and reduce latency
2. Structured progress prevents work repetition across continuations
3. Batch related operations in one call
4. Truncate long outputs to save context window
5. Finish immediately when done
6. Progressive summarization for growing conversations
7. Clear stopping conditions prevent loops
8. Graceful error handling avoids wasted steps
9. Plan before executing
10. Use Write tool for new files instead of multiple Edits
"""


def _project_root() -> Path:
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
        p = p.parent
    return Path.cwd()


def _parse_task(task: str) -> dict[str, str | None]:
    m = re.search(r"((?:src/|/)\S+\.py)", task)
    return {
        "agent_path": m.group(1) if m else None,
        "keep_model": (
            "true"
            if re.search(r"do\s+not\s+change\s+the\s+model", task, re.I)
            else None
        ),
        "model_constraint": "claude" if re.search(r"claude", task, re.I) else None,
    }


def _analyze_agent(path: str) -> dict[str, Any]:
    tree = ast.parse(Path(path).read_text())

    class_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                if isinstance(b, ast.Name) and b.id == "Base":
                    class_name = node.name
                    break
            if class_name:
                break

    bench_task = None
    run_kwargs: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "main"):
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if (
                        isinstance(t, ast.Name)
                        and t.id == "task_description"
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)
                    ):
                        bench_task = stmt.value.value
            if (
                isinstance(stmt, ast.Call)
                and isinstance(stmt.func, ast.Attribute)
                and stmt.func.attr == "run"
            ):
                for kw in stmt.keywords:
                    if isinstance(kw.value, ast.Constant) and kw.arg:
                        run_kwargs[kw.arg] = kw.value.value
        break

    return {"class_name": class_name, "benchmark_task": bench_task, "run_kwargs": run_kwargs}


def _validate(path: str, class_name: str) -> bool:
    try:
        py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError:
        return False
    name = f"_val_{random.randint(0, 99999)}"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if not spec or not spec.loader:
            return False
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return hasattr(mod, class_name)
    except Exception:
        return False
    finally:
        sys.modules.pop(name, None)


def _fail_metrics(error: str = "") -> dict[str, object]:
    return {"success": False, "time": 0.0, "cost": 0.0, "tokens": 0, "error": error}


def _benchmark(
    path: str, class_name: str, task: str, run_kw: dict[str, Any]
) -> dict[str, object]:
    work_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    name = f"_bench_{int(time.time())}_{random.randint(0, 99999)}"
    saved = Base.global_budget_used
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if not spec or not spec.loader:
            return _fail_metrics("Module load failed")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)

        cls = getattr(mod, class_name, None)
        if not cls:
            return _fail_metrics(f"Class {class_name} not found")

        Base.global_budget_used = 0.0
        agent = cls("Benchmark")
        os.chdir(work_dir)
        start = time.time()

        kw: dict[str, Any] = {
            "prompt_template": task,
            "work_dir": work_dir,
            "formatter": CompactFormatter(),
        }
        for k, v in run_kw.items():
            if k not in ("prompt_template", "work_dir", "formatter"):
                kw[k] = v
        kw.setdefault("trials", 3)

        try:
            result = agent.run(**kw)
            elapsed = time.time() - start
            data = yaml.safe_load(result)
            success = isinstance(data, dict) and data.get("success", False)
        except Exception:
            elapsed = time.time() - start
            success = False

        return {
            "success": success,
            "time": round(elapsed, 1),
            "cost": round(getattr(agent, "budget_used", 0.0), 4),
            "tokens": getattr(agent, "total_tokens_used", 0),
        }
    except Exception as e:
        return _fail_metrics(str(e))
    finally:
        os.chdir(old_cwd)
        sys.modules.pop(name, None)
        Base.global_budget_used = saved
        shutil.rmtree(work_dir, ignore_errors=True)


def _score(m: dict[str, Any]) -> float:
    s = 0.0 if m.get("success") else 10_000_000.0
    s += float(m.get("time", 0) or 0) * 1000
    s += float(m.get("cost", 0) or 0) * 100_000
    s += int(m.get("tokens", 0) or 0)
    return s


def _fmt(m: dict[str, Any]) -> str:
    return (
        f"success={m.get('success')}, "
        f"time={float(m.get('time', 0) or 0):.0f}s, "
        f"cost=${float(m.get('cost', 0) or 0):.4f}, "
        f"tokens={m.get('tokens', 0)}"
    )


def _fetch_strategies() -> str:
    urls = [
        "https://docs.anthropic.com/en/docs/build-with-claude/agentic-systems",
        "https://www.anthropic.com/engineering/building-effective-agents",
    ]
    parts = []
    for url in urls:
        try:
            content = fetch_url(url, SAFARI_HEADERS, max_characters=3000)
            if content and not content.startswith("Failed"):
                parts.append(content)
        except Exception:
            pass
    return "\n---\n".join(parts) if parts else FALLBACK_STRATEGIES


def _optimize_once(
    path: str,
    class_name: str,
    history: list[dict[str, Any]],
    strategies: str,
    iteration: int,
    constraints: str,
) -> str | None:
    opt_dir = tempfile.mkdtemp()
    dest = os.path.join(opt_dir, os.path.basename(path))
    shutil.copy2(path, dest)
    original = Path(path).read_text()

    hist = "\n".join(
        f"  #{i}: {'OK' if m.get('success') else 'FAIL'} | "
        f"{float(m.get('time', 0) or 0):.0f}s | "
        f"${float(m.get('cost', 0) or 0):.4f} | {m.get('tokens', 0)}tok"
        for i, m in enumerate(history)
    )
    theme = THEMES[min(iteration - 1, len(THEMES) - 1)]

    prompt = f"""Optimize the coding agent at {dest}.

GOAL: Improve these metrics IN ORDER: 1) Must succeed 2) Faster 3) Cheaper 4) Fewer tokens.
The agent must handle very long tasks (10K+ steps). Keep the same
class name ({class_name}) and the same run() API signature. Keep the main() function.
{constraints}

METRICS HISTORY (success|time|cost|tokens):
{hist}

THIS ITERATION FOCUS: {theme}

RESEARCH STRATEGIES:
{strategies[:1500]}

ARCHITECTURES TO CONSIDER:
- Single agent with smart continuation (current)
- Hierarchical: planner agent + executor agent
- Checkpoint-based with structured progress files
- Progressive summarization to manage context window
- Sub-task decomposition for complex tasks

HOW TO EDIT THE FILE (CRITICAL - follow exactly):
1. Read the file: Read(file_path="{dest}")
2. Make 3-5 TARGETED Edit calls. Each Edit must:
   - Include 5+ UNIQUE lines in old_string for matching
   - Change ONE specific section (e.g. TASK_PROMPT, a method, etc.)
   - Example: Edit(file_path="{dest}",
     old_string="def _calc(self, trial: int):\\n...",
     new_string="def _calc(self, trial: int):\\n...")
3. Do NOT try to rewrite the entire file at once - it WILL fail.
4. Verify: Bash(command='python3 -c "import py_compile;'
   ' py_compile.compile(\\"{dest}\\")"',
   description='check syntax')
5. Call finish when done."""

    tools = UsefulTools(
        base_dir=opt_dir, readable_paths=[opt_dir], writable_paths=[opt_dir]
    )
    saved = Base.global_budget_used
    try:
        Base.global_budget_used = 0.0
        KISSAgent(f"Opt-{iteration}").run(
            model_name="claude-sonnet-4-5",
            prompt_template=prompt,
            tools=[tools.Bash, tools.Read, tools.Edit],
            max_steps=20,
            max_budget=5.0,
            formatter=CompactFormatter(),
        )
    except Exception:
        pass
    finally:
        Base.global_budget_used = saved

    if not os.path.exists(dest) or not _validate(dest, class_name):
        shutil.rmtree(opt_dir, ignore_errors=True)
        return None
    if Path(dest).read_text() == original:
        shutil.rmtree(opt_dir, ignore_errors=True)
        return None
    return dest


def _run_general(task: str, fmt: CompactFormatter) -> None:
    from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

    work_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    start = time.time()
    agent = RelentlessCodingAgent("Optimizer")
    try:
        os.chdir(work_dir)
        result = agent.run(
            prompt_template=task,
            subtasker_model_name="claude-sonnet-4-5",
            max_steps=25,
            work_dir=work_dir,
            formatter=fmt,
        )
    finally:
        os.chdir(old_cwd)
    elapsed = time.time() - start
    data = yaml.safe_load(result)
    success = isinstance(data, dict) and data.get("success", False)
    fmt.print_status("\n=== RESULTS ===")
    fmt.print_status(f"Success: {success}")
    fmt.print_status(
        f"Time: {elapsed:.1f}s, Cost: ${agent.budget_used:.4f}, "
        f"Tokens: {agent.total_tokens_used}"
    )


def main(task_description: str | None = None) -> None:
    if task_description is None:
        task_description = (
            "can you optimize the agent at src/kiss/agents/coding_agents/"
            "relentless_coding_agent.py so that it finishes successfully "
            "(highest priority), runs faster (second highest priority), "
            "with lower cost (medium priority), and lower total_token_used "
            "(lower priority) while being robust for very very long tasks "
            "comprising of 10s of thousands of steps.  Make sure you run "
            "the agent with the given database task to compute the metrics.  "
            "Keep doing optimizations iteratively and stop when you think "
            "you cannot optimize it further.  You should also consider "
            "various architectures for the agent.  Use web to search for "
            "best strategies to build long running agents.  Do not change "
            "the model.  Use only claude series models."
        )

    fmt = CompactFormatter()
    fmt.print_status("=== Optimize Agent ===")
    fmt.print_status(f"Task: {task_description[:120]}...")

    parsed = _parse_task(task_description)
    if not parsed.get("agent_path"):
        _run_general(task_description, fmt)
        return

    root = _project_root()
    agent_path = (root / str(parsed["agent_path"])).resolve()
    if not agent_path.exists():
        raise FileNotFoundError(f"Agent not found: {agent_path}")

    info = _analyze_agent(str(agent_path))
    class_name: str = info["class_name"]
    bench_task: str = info["benchmark_task"]
    run_kwargs: dict[str, Any] = info.get("run_kwargs", {})

    if not class_name or not bench_task:
        raise ValueError("Could not extract class or benchmark from agent source.")

    fmt.print_status(f"Agent: {agent_path} ({class_name})")
    fmt.print_status(f"Benchmark: {bench_task.strip()[:100]}...")
    fmt.print_status(f"Run kwargs: {run_kwargs}")

    constraints = []
    if parsed.get("keep_model"):
        constraints.append("Do not change the model used by the agent.")
    if parsed.get("model_constraint"):
        constraints.append(f"Only use {parsed['model_constraint']} series models.")
    cstr = " ".join(constraints)

    fmt.print_status("\nFetching optimization strategies...")
    strategies = _fetch_strategies()
    fmt.print_status(f"Strategies loaded ({len(strategies)} chars)")

    fmt.print_status("\nRunning baseline benchmark...")
    baseline = _benchmark(str(agent_path), class_name, bench_task, run_kwargs)
    fmt.print_status(f"Baseline: {_fmt(baseline)}")

    best_path = str(agent_path)
    best_metrics = baseline
    best_sc = _score(baseline)
    history: list[dict[str, object]] = [baseline]
    stale = 0

    for i in range(1, len(THEMES) + 1):
        fmt.print_status(f"\n--- Iteration {i}/{len(THEMES)} ---")
        fmt.print_status(f"Theme: {THEMES[i - 1][:80]}...")

        new_path = _optimize_once(
            best_path, class_name, history, strategies, i, cstr
        )
        if not new_path:
            fmt.print_warning("  Optimization produced invalid code, skipping.")
            stale += 1
            if stale >= 2:
                fmt.print_status("  Stopping: 2 consecutive failures.")
                break
            continue

        fmt.print_status("  Benchmarking optimized version...")
        metrics = _benchmark(new_path, class_name, bench_task, run_kwargs)
        history.append(metrics)
        new_sc = _score(metrics)

        fmt.print_status(
            f"  {_fmt(metrics)} (score: {new_sc:.0f} vs {best_sc:.0f})"
        )

        if new_sc < best_sc:
            fmt.print_status("  >>> IMPROVEMENT! Keeping new version.")
            if best_path != str(agent_path):
                shutil.rmtree(str(Path(best_path).parent), ignore_errors=True)
            best_path, best_metrics, best_sc = new_path, metrics, new_sc
            stale = 0
        else:
            fmt.print_warning("  No improvement.")
            shutil.rmtree(str(Path(new_path).parent), ignore_errors=True)
            stale += 1
            if stale >= 2:
                fmt.print_status("  Converged after 2 consecutive non-improvements.")
                break

    fmt.print_status("\n=== FINAL RESULTS ===")
    fmt.print_status(f"Best: {_fmt(best_metrics)} (score: {best_sc:.0f})")
    fmt.print_status(
        f"Baseline: {_fmt(baseline)} (score: {_score(baseline):.0f})"
    )

    if best_path != str(agent_path):
        out = str(agent_path.parent / (agent_path.stem + "_optimized.py"))
        shutil.copy2(best_path, out)
        fmt.print_status(f"Optimized agent saved to {out}")
        shutil.rmtree(str(Path(best_path).parent), ignore_errors=True)
    else:
        fmt.print_status("Original agent is already optimal for this benchmark.")


if __name__ == "__main__":
    task_arg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    main(task_arg)
