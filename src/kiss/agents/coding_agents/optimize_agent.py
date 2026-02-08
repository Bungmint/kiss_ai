"""Iteratively optimize any coding agent based on a natural language task description.

Parses the task to identify the target agent, extracts benchmark details from
the agent's main() function, then iteratively improves it with LLM-guided optimization.
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

ITERATION_THEMES = [
    "Optimize TASK_PROMPT: make it concise and directive. Add an explicit step-by-step "
    "plan for the LLM. Improve batching instructions. Remove redundant text.",
    "Improve continuation architecture: structured progress format between trials, "
    "smart step threshold, progressive summarization for 10K+ step robustness.",
    "Reduce token usage: compress prompts, shorter previous_progress, "
    "eliminate usage-info bloat, minimize system message overhead.",
    "Consider alternative architectures: hierarchical planner+executor, "
    "sub-task decomposition, adding Write tool for faster file creation, "
    "better error handling to avoid wasted steps.",
    "Final tuning: optimize finish conditions, tune trials/steps ratio, "
    "edge-case robustness for very long tasks, cleanup dead code.",
]

FALLBACK_STRATEGIES = """
Key strategies for optimizing long-running AI coding agents:
1. Concise prompts save tokens and reduce latency
2. Structured progress tracking prevents work repetition across continuations
3. Batch related operations (test set+get+delete in one Bash call)
4. Truncate long tool outputs to save context window space
5. Finish IMMEDIATELY when tests pass — no extras like docs or demos
6. Progressive summarization: compress old context as conversation grows
7. Clear stopping conditions prevent infinite loops
8. Graceful error handling avoids wasting steps on retries
9. Checkpointing: write progress to file for recovery across continuations
10. Plan before executing: outline steps first, then execute methodically
11. Use appropriate step limits per trial for the task complexity
12. Pass structured continuation context (Done/Next format)
13. Avoid repeating work by maintaining a clear list of completed items
14. Use Write tool instead of multiple Edit calls for new files
"""


def _find_project_root() -> Path:
    path = Path(__file__).resolve().parent
    while path != path.parent:
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
        path = path.parent
    return Path.cwd()


def parse_optimization_task(task_description: str) -> dict[str, str | None]:
    """Extract agent path and constraints from the optimization task description."""
    path_match = re.search(r'((?:src/|/)\S+\.py)', task_description)
    agent_path = path_match.group(1) if path_match else None

    model_constraint = "claude" if re.search(r'claude', task_description, re.I) else None
    keep_model = (
        "true" if re.search(r'do\s+not\s+change\s+the\s+model', task_description, re.I) else None
    )

    return {
        "agent_path": agent_path,
        "model_constraint": model_constraint,
        "keep_model": keep_model,
    }


def analyze_agent_source(agent_path: str) -> dict[str, Any]:
    """Extract class name, benchmark task, and run() kwargs from agent source via AST."""
    source = Path(agent_path).read_text()
    tree = ast.parse(source)

    class_name: str | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base_cls in node.bases:
                if isinstance(base_cls, ast.Name) and base_cls.id == "Base":
                    class_name = node.name
                    break
            if class_name:
                break

    benchmark_task: str | None = None
    run_kwargs: dict[str, Any] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "task_description"
                            and isinstance(stmt.value, ast.Constant)
                            and isinstance(stmt.value.value, str)
                        ):
                            benchmark_task = stmt.value.value
                if (
                    isinstance(stmt, ast.Call)
                    and isinstance(stmt.func, ast.Attribute)
                    and stmt.func.attr == "run"
                ):
                    for kw in stmt.keywords:
                        if isinstance(kw.value, ast.Constant) and kw.arg is not None:
                            run_kwargs[kw.arg] = kw.value.value
            break

    return {
        "class_name": class_name,
        "benchmark_task": benchmark_task,
        "run_kwargs": run_kwargs,
    }


def fetch_strategies() -> str:
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


def validate_agent(path: str, class_name: str) -> bool:
    try:
        py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError:
        return False
    module_name = f"_val_{random.randint(0, 99999)}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return False
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return hasattr(mod, class_name)
    except Exception:
        return False
    finally:
        sys.modules.pop(module_name, None)


def _fail_metrics(error: str = "") -> dict[str, object]:
    return {"success": False, "time": 0.0, "cost": 0.0, "tokens": 0, "error": error}


def run_benchmark(
    agent_path: str,
    class_name: str,
    benchmark_task: str,
    run_kwargs: dict[str, Any],
) -> dict[str, object]:
    work_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    module_name = f"_bench_{int(time.time())}_{random.randint(0, 99999)}"
    saved_budget = Base.global_budget_used
    try:
        spec = importlib.util.spec_from_file_location(module_name, agent_path)
        if spec is None or spec.loader is None:
            return _fail_metrics("Module load failed")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

        agent_cls = getattr(mod, class_name, None)
        if agent_cls is None:
            return _fail_metrics(f"Class {class_name} not found")

        Base.global_budget_used = 0.0
        agent = agent_cls("Benchmark")
        os.chdir(work_dir)
        start = time.time()

        kwargs: dict[str, Any] = {
            "prompt_template": benchmark_task,
            "work_dir": work_dir,
            "formatter": CompactFormatter(),
        }
        for k, v in run_kwargs.items():
            if k not in ("prompt_template", "work_dir", "formatter"):
                kwargs[k] = v
        if "trials" not in kwargs:
            kwargs["trials"] = 3

        try:
            result = agent.run(**kwargs)
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
        sys.modules.pop(module_name, None)
        Base.global_budget_used = saved_budget
        shutil.rmtree(work_dir, ignore_errors=True)


def score(m: dict[str, object]) -> float:
    """Combined score (lower = better). Priority: success >> time > cost > tokens."""
    s = 0.0 if m.get("success") else 10_000_000.0
    time_val = m.get("time", 0)
    cost_val = m.get("cost", 0)
    tokens_val = m.get("tokens", 0)
    s += (time_val if isinstance(time_val, (int, float)) else 0) * 1000
    s += (cost_val if isinstance(cost_val, (int, float)) else 0) * 100_000
    s += tokens_val if isinstance(tokens_val, (int, float)) else 0
    return s


def format_metrics(m: dict[str, object]) -> str:
    return (
        f"success={m.get('success')}, "
        f"time={float(m.get('time', 0) or 0):.0f}s, "  # type: ignore[arg-type]
        f"cost=${float(m.get('cost', 0) or 0):.4f}, "  # type: ignore[arg-type]
        f"tokens={m.get('tokens', 0)}"
    )


def _finish_optimization(result: str) -> str:
    return result


def optimize_iteration(
    current_path: str,
    class_name: str,
    metrics_history: list[dict[str, object]],
    strategies: str,
    iteration: int,
    task_constraints: str,
) -> str | None:
    opt_dir = tempfile.mkdtemp()
    dest = os.path.join(opt_dir, os.path.basename(current_path))
    shutil.copy2(current_path, dest)
    original_content = Path(current_path).read_text()

    history_lines = "\n".join(
        f"  #{i}: {'OK' if m.get('success') else 'FAIL'} | "
        f"{float(m.get('time', 0) or 0):.0f}s | "  # type: ignore[arg-type]
        f"${float(m.get('cost', 0) or 0):.4f} | "  # type: ignore[arg-type]
        f"{m.get('tokens', 0)}tok"
        for i, m in enumerate(metrics_history)
    )
    theme = ITERATION_THEMES[min(iteration - 1, len(ITERATION_THEMES) - 1)]

    task = f"""Optimize the coding agent at {dest}.

GOAL: Improve these metrics IN ORDER: 1) Must succeed 2) Faster 3) Cheaper 4) Fewer tokens.
The agent must handle very long tasks (10K+ steps). Keep the same
class name ({class_name}) and the same run() API signature. Keep the main() function.
{task_constraints}

METRICS HISTORY (success|time|cost|tokens):
{history_lines}

THIS ITERATION FOCUS: {theme}

RESEARCH STRATEGIES:
{strategies[:1500]}

ARCHITECTURES TO CONSIDER:
- Single agent with smart continuation (current)
- Hierarchical: planner agent + executor agent
- Checkpoint-based with structured progress files
- Progressive summarization to manage context window
- Sub-task decomposition for complex tasks

HOW TO EDIT THE FILE:
To rewrite the file, use Bash with a Python script:
  Bash(command='python3 -c "
import textwrap
content = textwrap.dedent(chr(39)*3 + chr(10) + ... YOUR CODE HERE ... + chr(10) + chr(39)*3)
open(\\"{dest}\\", \\"w\\").write(content)
"', description='Rewrite agent file')

Or use multiple Edit() calls. When using Edit, include at least 5 unique lines of context.
After editing, verify with Bash: python3 -c "import py_compile; py_compile.compile('{dest}')"
Call finish when done.
"""

    tools = UsefulTools(
        base_dir=opt_dir,
        readable_paths=[opt_dir],
        writable_paths=[opt_dir],
    )

    saved_budget = Base.global_budget_used
    try:
        Base.global_budget_used = 0.0
        optimizer = KISSAgent(f"Optimizer-{iteration}")
        optimizer.run(
            model_name="claude-sonnet-4-5",
            prompt_template=task,
            tools=[
                _finish_optimization,
                tools.Bash,
                tools.Read,
                tools.Edit,
                tools.Write,
            ],
            max_steps=20,
            max_budget=5.0,
            formatter=CompactFormatter(),
        )
    except Exception:
        pass
    finally:
        Base.global_budget_used = saved_budget

    if not os.path.exists(dest) or not validate_agent(dest, class_name):
        shutil.rmtree(opt_dir, ignore_errors=True)
        return None

    new_content = Path(dest).read_text()
    if new_content == original_content:
        shutil.rmtree(opt_dir, ignore_errors=True)
        return None

    return dest


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

    formatter = CompactFormatter()
    formatter.print_status("=== Agent Optimizer ===")
    formatter.print_status(f"Task: {task_description[:120]}...")

    parsed = parse_optimization_task(task_description)
    agent_rel_path = parsed["agent_path"]
    if not agent_rel_path:
        raise ValueError("Could not extract agent path from task description.")

    project_root = _find_project_root()
    agent_path = (project_root / agent_rel_path).resolve()
    if not agent_path.exists():
        raise FileNotFoundError(f"Agent file not found: {agent_path}")

    formatter.print_status(f"Target agent: {agent_path}")

    agent_info = analyze_agent_source(str(agent_path))
    class_name: str = agent_info["class_name"]
    benchmark_task: str = agent_info["benchmark_task"]
    run_kwargs: dict[str, Any] = agent_info.get("run_kwargs", {})

    if not class_name:
        raise ValueError("Could not find agent class (subclass of Base) in source.")
    if not benchmark_task:
        raise ValueError("Could not find benchmark task in agent's main() function.")

    formatter.print_status(f"Agent class: {class_name}")
    formatter.print_status(f"Benchmark task: {benchmark_task.strip()[:100]}...")
    formatter.print_status(f"Run kwargs: {run_kwargs}")

    constraints_parts = []
    if parsed.get("keep_model"):
        constraints_parts.append("Do not change the model used by the agent.")
    if parsed.get("model_constraint"):
        constraints_parts.append(f"Only use {parsed['model_constraint']} series models.")
    task_constraints = " ".join(constraints_parts)

    formatter.print_status("\nFetching optimization strategies from web...")
    strategies = fetch_strategies()
    formatter.print_status(f"Strategies loaded ({len(strategies)} chars)")

    formatter.print_status("\nRunning baseline benchmark...")
    baseline = run_benchmark(str(agent_path), class_name, benchmark_task, run_kwargs)
    formatter.print_status(f"Baseline: {format_metrics(baseline)}")

    best_path = str(agent_path)
    best_metrics = baseline
    best_sc = score(baseline)
    history: list[dict[str, object]] = [baseline]
    stale = 0

    for i in range(1, len(ITERATION_THEMES) + 1):
        formatter.print_status(f"\n--- Iteration {i}/{len(ITERATION_THEMES)} ---")
        formatter.print_status(f"Theme: {ITERATION_THEMES[i - 1][:80]}...")

        new_path = optimize_iteration(
            best_path, class_name, history, strategies, i, task_constraints
        )
        if new_path is None:
            formatter.print_warning("  Optimization produced invalid code, skipping.")
            stale += 1
            if stale >= 2:
                formatter.print_status("  Stopping: 2 consecutive failures.")
                break
            continue

        formatter.print_status("  Benchmarking optimized version...")
        metrics = run_benchmark(new_path, class_name, benchmark_task, run_kwargs)
        history.append(metrics)
        new_sc = score(metrics)

        formatter.print_status(
            f"  {format_metrics(metrics)} (score: {new_sc:.0f} vs {best_sc:.0f})"
        )

        if new_sc < best_sc:
            formatter.print_status("  >>> IMPROVEMENT! Keeping new version.")
            if best_path != str(agent_path):
                shutil.rmtree(str(Path(best_path).parent), ignore_errors=True)
            best_path = new_path
            best_metrics = metrics
            best_sc = new_sc
            stale = 0
        else:
            formatter.print_warning("  No improvement.")
            shutil.rmtree(str(Path(new_path).parent), ignore_errors=True)
            stale += 1
            if stale >= 2:
                formatter.print_status("  Converged after 2 consecutive non-improvements.")
                break

    formatter.print_status("\n=== FINAL RESULTS ===")
    formatter.print_status(f"Best: {format_metrics(best_metrics)} (score: {best_sc:.0f})")
    formatter.print_status(f"Baseline: {format_metrics(baseline)} (score: {score(baseline):.0f})")

    if best_path != str(agent_path):
        out_name = agent_path.stem + "_optimized.py"
        out = str(agent_path.parent / out_name)
        shutil.copy2(best_path, out)
        formatter.print_status(f"Optimized agent saved to {out}")
        shutil.rmtree(str(Path(best_path).parent), ignore_errors=True)
    else:
        formatter.print_status("Original agent is already optimal for this benchmark.")


if __name__ == "__main__":
    task_arg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    main(task_arg)
