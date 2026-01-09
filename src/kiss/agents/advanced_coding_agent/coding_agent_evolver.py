# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Coding Agent Evolver - Evolves the coding agent for better performance."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kiss.agents.advanced_coding_agent.config  # noqa: F401
from kiss.agents.kiss_evolve.kiss_evolve import CodeVariant, KISSEvolve
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.docker.docker_manager import DockerManager


@dataclass
class EvaluationTask:
    """A task for evaluating coding agent performance."""

    name: str
    description: str
    test_script: str  # Python script that returns True if task succeeded
    expected_files: list[str]  # Files that should exist after completion
    timeout: int = 300  # Timeout in seconds


# Predefined evaluation tasks of increasing difficulty
EVALUATION_TASKS = [
    EvaluationTask(
        name="fibonacci",
        description="""
        Create a Python script that:
        1. Generates the first 15 Fibonacci numbers
        2. Saves them to 'fibonacci.txt', one number per line
        3. The script should be named 'fib.py'
        """,
        test_script="""
import os
if not os.path.exists('fibonacci.txt'):
    print("FAIL: fibonacci.txt not found")
    exit(1)
with open('fibonacci.txt') as f:
    nums = [int(x.strip()) for x in f.readlines() if x.strip()]
expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
if nums == expected:
    print("PASS")
    exit(0)
else:
    print(f"FAIL: Got {nums}")
    exit(1)
""",
        expected_files=["fibonacci.txt", "fib.py"],
        timeout=120,
    ),
    EvaluationTask(
        name="prime_sieve",
        description="""
        Create a Python script called 'primes.py' that:
        1. Implements the Sieve of Eratosthenes
        2. Finds all prime numbers up to 100
        3. Saves them to 'primes.txt', one per line
        """,
        test_script="""
import os
if not os.path.exists('primes.txt'):
    print("FAIL: primes.txt not found")
    exit(1)
with open('primes.txt') as f:
    primes = [int(x.strip()) for x in f.readlines() if x.strip()]
expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97]
if primes == expected:
    print("PASS")
    exit(0)
else:
    print(f"FAIL: Got {primes}")
    exit(1)
""",
        expected_files=["primes.txt", "primes.py"],
        timeout=120,
    ),
    EvaluationTask(
        name="json_processor",
        description="""
        Create a Python script called 'process_json.py' that:
        1. Creates a JSON file 'data.json' with a list of 5 dictionaries,
           each having 'name' and 'score' keys
        2. Reads the JSON file
        3. Calculates the average score
        4. Writes the result to 'result.txt' as "Average: X.XX"
        """,
        test_script="""
import os
import json
if not os.path.exists('data.json'):
    print("FAIL: data.json not found")
    exit(1)
if not os.path.exists('result.txt'):
    print("FAIL: result.txt not found")
    exit(1)
with open('data.json') as f:
    data = json.load(f)
if not isinstance(data, list) or len(data) != 5:
    cnt = len(data) if isinstance(data, list) else 'not a list'
    print(f"FAIL: data.json should have 5 items, got {cnt}")
    exit(1)
avg = sum(d['score'] for d in data) / len(data)
with open('result.txt') as f:
    result = f.read().strip()
if f"Average: {avg:.2f}" in result or f"Average: {avg}" in result:
    print("PASS")
    exit(0)
else:
    print(f"FAIL: Expected 'Average: {avg:.2f}', got '{result}'")
    exit(1)
""",
        expected_files=["data.json", "result.txt", "process_json.py"],
        timeout=180,
    ),
]


# The base coding agent code that will be evolved
BASE_AGENT_CODE = """
'''Evolvable coding agent implementation.'''

from kiss.core.kiss_agent import KISSAgent
from kiss.docker.docker_manager import DockerManager


AGENT_PROMPT = '''
## Role ##
You are a coding agent that completes programming tasks.

## Task ##
{task}

## Instructions ##
1. Understand the task requirements
2. Write the necessary code
3. Execute and verify your solution
4. Call finish when done

## Important ##
All files should be created in the current working directory.
'''


def run_task(task: str, model_name: str, docker: DockerManager) -> str:
    '''Run a coding task.

    Args:
        task: The task description
        model_name: The LLM model to use
        docker: The Docker manager for execution

    Returns:
        The result from the agent
    '''
    def run_bash(command: str, description: str = "") -> str:
        '''Execute a bash command.'''
        return docker.run_bash_command(command, description or "Running command")

    def read_file(path: str) -> str:
        '''Read a file.'''
        return docker.run_bash_command(f"cat {path}", f"Reading {path}")

    def write_file(path: str, content: str) -> str:
        '''Write to a file.'''
        cmd = f"cat > {path} << 'EOF'\\n{content}\\nEOF"
        return docker.run_bash_command(cmd, f"Writing {path}")

    agent = KISSAgent(name="CodingAgent")
    result = agent.run(
        model_name=model_name,
        prompt_template=AGENT_PROMPT,
        arguments={"task": task},
        tools=[run_bash, read_file, write_file],
        max_steps=20,
        max_budget=0.5,
    )
    return result
"""


def evaluate_agent_code(
    agent_code: str,
    tasks: list[EvaluationTask],
    model_name: str,
) -> dict[str, Any]:
    """Evaluate agent code on a set of tasks.

    Args:
        agent_code: The Python code for the agent
        tasks: List of evaluation tasks
        model_name: LLM model to use

    Returns:
        Dictionary with fitness and metrics
    """
    results: dict[str, Any] = {
        "fitness": 0.0,
        "metrics": {
            "tasks_passed": 0,
            "tasks_total": len(tasks),
            "total_time": 0.0,
            "avg_time": 0.0,
        },
        "artifacts": {},
        "error": None,
    }

    # Create a namespace to exec the agent code
    namespace: dict[str, Any] = {}

    try:
        # Execute the agent code to get the run_task function
        exec(agent_code, namespace)
        run_task_fn = namespace.get("run_task")
        if run_task_fn is None:
            results["error"] = "Agent code does not define run_task function"
            return results
    except Exception as e:
        results["error"] = f"Failed to compile agent code: {e}"
        return results

    passed = 0
    total_time = 0.0

    for task in tasks:
        task_start = time.time()
        task_passed = False

        try:
            with DockerManager("python:3.12-slim", workdir="/") as docker:
                # Setup workspace - create directory first, then change workdir
                docker.run_bash_command("mkdir -p /workspace", "Creating workspace")
                docker.workdir = "/workspace"

                # Run the agent
                _ = run_task_fn(task.description, model_name, docker)

                # Run the test script
                docker.run_bash_command(
                    f"cat > /tmp/test.py << 'EOF'\n{task.test_script}\nEOF",
                    "Creating test script",
                )
                test_result = docker.run_bash_command(
                    "python /tmp/test.py",
                    "Running test",
                )

                if "PASS" in test_result:
                    task_passed = True
                    passed += 1

        except Exception as e:
            results["artifacts"][task.name] = f"Error: {e}"

        task_time = time.time() - task_start
        total_time += task_time
        results["artifacts"][task.name] = {
            "passed": task_passed,
            "time": task_time,
        }

    results["metrics"]["tasks_passed"] = passed
    results["metrics"]["total_time"] = total_time
    results["metrics"]["avg_time"] = total_time / len(tasks) if tasks else 0

    # Fitness: primarily based on pass rate, with time as secondary factor
    pass_rate = passed / len(tasks) if tasks else 0
    time_bonus = max(0, 1 - (total_time / (len(tasks) * 60)))  # Bonus for speed
    results["fitness"] = pass_rate * 0.9 + time_bonus * 0.1

    return results


def create_code_agent_wrapper(model_name: str) -> Callable[..., str]:
    """Create a code agent wrapper for KISSEvolve.

    Args:
        model_name: The LLM model to use for code generation

    Returns:
        A function that generates code variations
    """

    def code_agent_wrapper(
        prompt_template: str,
        arguments: dict[str, str],
        model_name: str = model_name,
    ) -> str:
        """Generate code using an LLM agent."""
        agent = KISSAgent(name="CodeEvolver")
        result = agent.run(
            model_name=model_name,
            prompt_template=prompt_template,
            arguments=arguments,
            is_agentic=True,
            max_steps=10,
            max_budget=0.3,
        )
        return result

    return code_agent_wrapper


class CodingAgentEvolver:
    """Evolves coding agent code for better performance."""

    def __init__(
        self,
        model_name: str | None = None,
        population_size: int | None = None,
        max_generations: int | None = None,
        mutation_rate: float | None = None,
        elite_size: int | None = None,
        tasks: list[EvaluationTask] | None = None,
    ):
        """Initialize the evolver.

        Args:
            model_name: LLM model to use
            population_size: Number of variants per generation
            max_generations: Maximum generations
            mutation_rate: Probability of mutation
            elite_size: Number of elite variants to keep
            tasks: Evaluation tasks (defaults to EVALUATION_TASKS)
        """
        cfg = DEFAULT_CONFIG.advanced_coding_agent  # type: ignore[attr-defined]

        self.model_name = model_name or cfg.evolver_model
        self.population_size = population_size or cfg.evolver_population_size
        self.max_generations = max_generations or cfg.evolver_max_generations
        self.mutation_rate = mutation_rate or cfg.evolver_mutation_rate
        self.elite_size = elite_size or cfg.evolver_elite_size
        self.tasks = tasks or EVALUATION_TASKS

    def evolve(self) -> CodeVariant:
        """Run the evolutionary optimization.

        Returns:
            The best code variant found
        """
        print("=" * 60)
        print("Coding Agent Evolver")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Population: {self.population_size}")
        print(f"Generations: {self.max_generations}")
        print(f"Tasks: {len(self.tasks)}")
        print()

        def evaluation_fn(code: str) -> dict[str, Any]:
            return evaluate_agent_code(code, self.tasks, self.model_name)

        evolver = KISSEvolve(
            code_agent_wrapper=create_code_agent_wrapper(self.model_name),
            initial_code=BASE_AGENT_CODE,
            evaluation_fn=evaluation_fn,
            model_names=[(self.model_name, 1.0)],
            population_size=self.population_size,
            max_generations=self.max_generations,
            mutation_rate=self.mutation_rate,
            elite_size=self.elite_size,
            extra_coding_instructions="""
Focus on improving the agent's:
1. Task understanding and planning
2. Error handling and recovery
3. Efficient use of bash commands
4. Code verification before finishing
""",
        )

        best = evolver.evolve()

        print("\n" + "=" * 60)
        print("EVOLUTION COMPLETE")
        print("=" * 60)
        print(f"Best fitness: {best.fitness:.4f}")
        print(f"Metrics: {best.metrics}")

        return best

    def save_best(self, variant: CodeVariant, path: str = "evolved_agent.py") -> None:
        """Save the best variant to a file.

        Args:
            variant: The code variant to save
            path: Output file path
        """
        Path(path).write_text(variant.code)
        print(f"Saved best variant to {path}")


def generate_random_task() -> EvaluationTask:
    """Generate a random evaluation task for testing."""
    import random

    task_templates: list[dict[str, Any]] = [
        {
            "name": "list_operations",
            "description": """
            Create a Python script 'list_ops.py' that:
            1. Creates a list of {n} random integers between 1 and 100
            2. Sorts the list
            3. Removes duplicates
            4. Saves the result to 'sorted_unique.txt'
            """,
            "test_script": """
import os
if not os.path.exists('sorted_unique.txt'):
    print("FAIL: sorted_unique.txt not found")
    exit(1)
with open('sorted_unique.txt') as f:
    nums = [int(x.strip()) for x in f.readlines() if x.strip()]
if nums == sorted(set(nums)):
    print("PASS")
    exit(0)
else:
    print("FAIL: Not sorted or has duplicates")
    exit(1)
""",
            "expected_files": ["sorted_unique.txt", "list_ops.py"],
        },
        {
            "name": "word_count",
            "description": """
            Create a Python script 'wordcount.py' that:
            1. Creates a text file 'sample.txt' with at least {n} words
            2. Counts the number of words
            3. Writes the count to 'count.txt' as "Word count: N"
            """,
            "test_script": """
import os
if not os.path.exists('sample.txt') or not os.path.exists('count.txt'):
    print("FAIL: Required files not found")
    exit(1)
with open('sample.txt') as f:
    words = len(f.read().split())
with open('count.txt') as f:
    result = f.read().strip()
if f"Word count: {words}" in result:
    print("PASS")
    exit(0)
else:
    print(f"FAIL: Expected 'Word count: {words}', got '{result}'")
    exit(1)
""",
            "expected_files": ["sample.txt", "count.txt", "wordcount.py"],
        },
    ]

    template = random.choice(task_templates)
    n = random.randint(10, 50)

    description: str = template["description"].format(n=n)
    test_script: str = template["test_script"]
    expected_files: list[str] = template["expected_files"]

    return EvaluationTask(
        name=f"{template['name']}_{n}",
        description=description,
        test_script=test_script,
        expected_files=expected_files,
        timeout=180,
    )


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evolve the coding agent")
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="LLM model to use",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of generations",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=4,
        help="Population size",
    )
    parser.add_argument(
        "--output",
        default="evolved_agent.py",
        help="Output file for best agent",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test the base agent without evolution",
    )

    args = parser.parse_args()

    if args.test_only:
        # Just test the base agent
        print("Testing base agent code...")
        result = evaluate_agent_code(BASE_AGENT_CODE, EVALUATION_TASKS[:1], args.model)
        print(f"Result: {json.dumps(result, indent=2)}")
        return

    evolver = CodingAgentEvolver(
        model_name=args.model,
        population_size=args.population,
        max_generations=args.generations,
    )

    best = evolver.evolve()
    evolver.save_best(best, args.output)


if __name__ == "__main__":
    main()
