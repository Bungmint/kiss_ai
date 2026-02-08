# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Single-agent coding system with smart continuation for long tasks."""

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from kiss.core import config as config_module
from kiss.core.base import CODING_INSTRUCTIONS, Base
from kiss.core.compact_formatter import CompactFormatter
from kiss.core.formatter import Formatter
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.model import TokenCallback
from kiss.core.models.model_info import get_max_context_length
from kiss.core.useful_tools import UsefulTools
from kiss.core.utils import resolve_path
from kiss.docker.docker_manager import DockerManager

MAX_OUTPUT_CHARS = 5000
PER_TRIAL_STEPS = 15

TASK_PROMPT = """## Task

{task_description}

{coding_instructions}

## Efficiency Rules (follow strictly)
1. Write: create files. Edit: small fixes only. Bash: run commands.
2. Batch: chmod+test in one Bash call. Minimize total steps.
3. IMMEDIATELY finish(success=True) once test suite passes. No extra manual tests.
4. After step {step_threshold}: write ./progress.md then finish(success=False).
5. Never redo work. Read ./progress.md if it exists.
6. Update ./progress.md before calling finish.
{previous_progress}"""


def _truncate(output: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(output) <= max_chars:
        return output
    head = max_chars * 2 // 5
    tail = max_chars * 3 // 5
    mid = f"\n\n[...truncated {len(output) - head - tail} chars...]\n\n"
    return output[:head] + mid + output[-tail:]


def finish(success: bool, summary: str) -> str:
    """Finish execution with status and summary.

    Args:
        success: True if successful, False otherwise.
        summary: Summary of work done and remaining work.
    """
    return yaml.dump(
        {"success": success, "summary": summary}, indent=2, sort_keys=False
    )


class RelentlessCodingAgent(Base):
    """Single-agent coding system with auto-continuation for infinite tasks."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _reset(
        self,
        subtasker_model_name: str | None,
        trials: int | None,
        max_steps: int | None,
        max_budget: float | None,
        work_dir: str | None,
        base_dir: str | None,
        readable_paths: list[str] | None,
        writable_paths: list[str] | None,
        docker_image: str | None,
    ) -> None:
        global_cfg = config_module.DEFAULT_CONFIG
        cfg = global_cfg.agent.relentless_coding_agent
        default_work_dir = str(
            Path(global_cfg.agent.artifact_dir).resolve() / "kiss_workdir"
        )

        actual_base_dir = base_dir if base_dir is not None else default_work_dir
        actual_work_dir = work_dir if work_dir is not None else default_work_dir

        Path(actual_base_dir).mkdir(parents=True, exist_ok=True)
        Path(actual_work_dir).mkdir(parents=True, exist_ok=True)
        self.base_dir = str(Path(actual_base_dir).resolve())
        self.work_dir = str(Path(actual_work_dir).resolve())
        self.readable_paths = [
            resolve_path(p, self.base_dir) for p in readable_paths or []
        ]
        self.writable_paths = [
            resolve_path(p, self.base_dir) for p in writable_paths or []
        ]
        self.readable_paths.append(Path(self.work_dir))
        self.writable_paths.append(Path(self.work_dir))
        self.is_agentic = True

        self.trials = trials if trials is not None else cfg.trials
        self.max_steps = max_steps if max_steps is not None else cfg.max_steps
        self.max_budget = max_budget if max_budget is not None else cfg.max_budget
        self.subtasker_model_name = (
            subtasker_model_name
            if subtasker_model_name is not None
            else cfg.subtasker_model_name
        )
        self.max_tokens = get_max_context_length(self.subtasker_model_name)

        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0

        self.docker_image = docker_image
        self.docker_manager: DockerManager | None = None

        self.useful_tools = UsefulTools(
            base_dir=self.base_dir,
            readable_paths=[str(p) for p in self.readable_paths],
            writable_paths=[str(p) for p in self.writable_paths],
        )

    def _docker_bash(self, command: str, description: str) -> str:
        if self.docker_manager is None:
            raise KISSError("Docker manager not initialized")
        return self.docker_manager.run_bash_command(command, description)

    def perform_task(self) -> str:
        self.formatter.print_status(f"Executing task: {self.task_description}")
        previous_progress = ""
        per_trial_steps = min(self.max_steps, PER_TRIAL_STEPS)
        step_threshold = max(8, per_trial_steps - 3)
        ut = self.useful_tools
        docker_mgr = self.docker_manager

        def Bash(command: str, description: str, timeout_seconds: float = 120) -> str:  # noqa: N802
            """Run a bash command and return its output.

            Args:
                command: The bash command to run.
                description: A brief description of the command.
                timeout_seconds: Timeout in seconds.
            """
            if docker_mgr:
                return _truncate(
                    docker_mgr.run_bash_command(command, description)
                )
            return _truncate(ut.Bash(command, description, timeout_seconds))

        def Read(file_path: str, max_lines: int = 2000) -> str:  # noqa: N802
            """Read file contents.

            Args:
                file_path: Absolute path to file.
                max_lines: Maximum number of lines to return.
            """
            return _truncate(ut.Read(file_path, max_lines))

        def Edit(  # noqa: N802
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
        ) -> str:
            """Replace exact text in a file. Use Write instead for large changes.

            Args:
                file_path: Absolute path to the file to modify.
                old_string: Exact text to find and replace.
                new_string: Replacement text.
                replace_all: If True, replace all occurrences.
            """
            result = ut.Edit(file_path, old_string, new_string, replace_all)
            if "not unique" in result or "not found" in result:
                result += "\nHint: Use Write to rewrite the file."
            return _truncate(result)

        tools: list[Callable[..., Any]] = [finish, Bash, Read, Edit, ut.Write]
        progress_file = Path(self.work_dir) / "progress.md"

        for trial in range(self.trials):
            executor = KISSAgent(f"{self.name} Trial-{trial}")

            if progress_file.exists() and not previous_progress:
                previous_progress = progress_file.read_text()[:3000]
            if previous_progress:
                progress_section = (
                    f"\n## Previous Progress\n{previous_progress}"
                )
            elif trial == 0:
                progress_section = "\n## Status\nFresh start. No existing files."
            else:
                progress_section = ""

            try:
                result = executor.run(
                    model_name=self.subtasker_model_name,
                    prompt_template=TASK_PROMPT,
                    arguments={
                        "task_description": self.task_description,
                        "coding_instructions": CODING_INSTRUCTIONS,
                        "previous_progress": progress_section,
                        "step_threshold": str(step_threshold),
                    },
                    tools=tools,
                    max_steps=per_trial_steps,
                    max_budget=self.max_budget,
                    formatter=self.formatter,
                    token_callback=self.token_callback,
                )
            except KISSError:
                file_progress = (
                    progress_file.read_text()[:3000]
                    if progress_file.exists()
                    else ""
                )
                last_msgs = (
                    executor.messages[-3:]
                    if hasattr(executor, "messages")
                    else []
                )
                context = "\n".join(
                    m.get("content", "")[:200] for m in last_msgs
                )
                result = yaml.dump(
                    {
                        "success": False,
                        "summary": (
                            f"Step limit.\n{file_progress}"
                            f"\nRecent:\n{context}"
                        ),
                    },
                    sort_keys=False,
                )
            self.budget_used += executor.budget_used  # type: ignore
            self.total_tokens_used += executor.total_tokens_used  # type: ignore

            ret = yaml.safe_load(result)
            payload = ret if isinstance(ret, dict) else {}
            if payload.get("success", False):
                return result
            previous_progress = payload.get("summary", "")
            continue
        raise KISSError(f"Task failed after {self.trials} trials")

    def run(
        self,
        prompt_template: str,
        arguments: dict[str, str] | None = None,
        subtasker_model_name: str | None = None,
        trials: int | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        work_dir: str | None = None,
        base_dir: str | None = None,
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
        docker_image: str | None = None,
        formatter: Formatter | None = None,
        token_callback: TokenCallback | None = None,
    ) -> str:
        self._reset(
            subtasker_model_name,
            trials,
            max_steps,
            max_budget,
            work_dir,
            base_dir,
            readable_paths,
            writable_paths,
            docker_image,
        )
        self.prompt_template = prompt_template
        self.arguments = arguments or {}
        self.task_description = prompt_template.format(**self.arguments)
        self.formatter = formatter or CompactFormatter()
        self.token_callback = token_callback

        if self.docker_image:
            with DockerManager(self.docker_image) as docker_mgr:
                self.docker_manager = docker_mgr
                try:
                    return self.perform_task()
                finally:
                    self.docker_manager = None
        else:
            return self.perform_task()


def main() -> None:
    import time as time_mod

    agent = RelentlessCodingAgent("Example Multi-Agent")
    task_description = """
 **Task:** Create a robust database engine using only Bash scripts.

 **Requirements:**
 1.  Create a script named `db.sh` that interacts with a local data folder.
 2.  **Basic Operations:** Implement `db.sh set <key> <value>`,
     `db.sh get <key>`, and `db.sh delete <key>`.
 3.  **Atomicity:** Implement transaction support.
     *   `db.sh begin` starts a session where writes are cached but not visible to others.
     *   `db.sh commit` atomically applies all cached changes.
     *   `db.sh rollback` discards pending changes.
 4.  **Concurrency:** Ensure that if two different terminal windows run `db.sh`
     simultaneously, the data is never corrupted (use `mkdir`-based mutex locking).
 5.  **Validation:** Write a test script `test_stress.sh` that launches 10
     concurrent processes to spam the database, verifying no data is lost.

 **Constraints:**
 *   No external database tools (no sqlite3, no python).
 *   Standard Linux utilities only (sed, awk, grep, flock/mkdir).
 *   Safe: Operate entirely within a `./my_db` directory.
    """

    work_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    start_time = time_mod.time()
    try:
        os.chdir(work_dir)
        result = agent.run(
            prompt_template=task_description,
            subtasker_model_name="claude-sonnet-4-5",
            max_steps=25,
            work_dir=work_dir,
            formatter=CompactFormatter(),
        )
    finally:
        os.chdir(old_cwd)
    elapsed = time_mod.time() - start_time

    agent.formatter.print_status("FINAL RESULT:")
    result_data = yaml.safe_load(result)
    agent.formatter.print_status(
        "Completed successfully: " + str(result_data["success"])
    )
    agent.formatter.print_status(result_data["summary"])
    agent.formatter.print_status("Work directory was: " + work_dir)
    agent.formatter.print_status(f"Time: {elapsed:.1f}s")
    agent.formatter.print_status(f"Cost: ${agent.budget_used:.4f}")
    agent.formatter.print_status(f"Total tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    main()
