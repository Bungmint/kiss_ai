# Agent Creator

A module for evolving and improving AI agents through multi-objective optimization. It provides tools to automatically optimize existing agent code for **token efficiency** and **execution speed** using evolutionary algorithms with Pareto frontier maintenance.

## Overview

The Agent Creator module consists of two main components:

1. **ImproverAgent**: Takes existing agent source code and creates optimized versions through iterative improvement
1. **AgentEvolver**: Maintains a population of agent variants and evolves them using mutation and crossover operations

Both components use a **Pareto frontier** approach to track non-dominated solutions, optimizing for multiple objectives simultaneously without requiring a single combined metric.

## Key Features

- **Multi-Objective Optimization**: Optimizes for flexible metrics (e.g., success, token usage, execution time)
- **Pareto Frontier Maintenance**: Keeps track of all non-dominated solutions
- **Evolutionary Operations**: Supports mutation (improving one variant) and crossover (combining ideas from two variants)
- **Automatic Pruning**: Removes dominated variants to manage memory and storage
- **Lineage Tracking**: Records parent relationships and improvement history
- **Configurable Coding Agents**: Supports Claude Code, Gemini CLI, and OpenAI Codex
- **Configurable Parameters**: Extensive configuration options for generations, frontier size, thresholds, etc.

## Installation

The module is part of the `kiss` package. No additional installation required.

## Quick Start

### Improving an Existing Agent

```python
import anyio
from kiss.agents.agent_creator import ImproverAgent

async def improve_agent():
    improver = ImproverAgent(
        model_name="claude-sonnet-4-5",
        max_steps=150,
        max_budget=15.0,
        coding_agent_type="claude code",
    )

    success, report = await improver.improve(
        source_folder="/path/to/agent",
        target_folder="/path/to/improved_agent",
    )

    if success and report:
        print(f"Improvement completed in {report.metrics.get('execution_time', 0):.2f}s")
        print(f"Tokens used: {report.metrics.get('tokens_used', 0)}")

anyio.run(improve_agent)
```

### Evolving a New Agent from Scratch

```python
import anyio
from kiss.agents.agent_creator import AgentEvolver

async def evolve_agent():
    evolver = AgentEvolver(
        task_description="Build a code analysis assistant that can parse and analyze large codebases",
        max_generations=10,
        max_frontier_size=6,
        mutation_probability=0.8,
        coding_agent_type="claude code",
    )

    best_variant = await evolver.evolve()

    print(f"Best agent: {best_variant.folder_path}")
    print(f"Metrics: {best_variant.metrics}")

anyio.run(evolve_agent)
```

## Components

### ImproverAgent

The `ImproverAgent` optimizes existing agent code by analyzing and improving it for token efficiency and execution speed.

**Parameters:**

- `model_name`: LLM model to use (default: `"claude-sonnet-4-5"`)
- `max_steps`: Maximum steps for the improvement agent (default: `150`)
- `max_budget`: Maximum USD budget for improvement (default: `15.0`)
- `coding_agent_type`: Which coding agent to use: `"claude code"`, `"gemini cli"`, or `"openai codex"`

**Methods:**

- `improve(source_folder, target_folder, report_path, feedback, base_dir)`: Improve an agent's code
- `crossover_improve(primary_folder, primary_report_path, secondary_report_path, primary_feedback, secondary_feedback, target_folder, base_dir)`: Combine ideas from two agents

### AgentEvolver

The `AgentEvolver` creates and evolves agent populations from a task description.

**Parameters:**

- `task_description`: Description of the task the agent should solve
- `model_name`: LLM model for orchestration (default: `"claude-sonnet-4-5"`)
- `max_generations`: Maximum evolutionary generations (default: `10`)
- `max_frontier_size`: Maximum Pareto frontier size (default: `6`)
- `mutation_probability`: Probability of mutation vs crossover (default: `0.8`)
- `coding_agent_type`: Which coding agent to use: `"claude code"`, `"gemini cli"`, or `"openai codex"`

**Methods:**

- `evolve()`: Run the evolutionary optimization, returns the best variant
- `get_best_variant()`: Get the current best variant by combined score
- `get_pareto_frontier()`: Get all variants in the Pareto frontier
- `save_state(path)`: Save evolver state to JSON

### Data Classes

**ImprovementReport**: Tracks improvements made to an agent

- `implemented_ideas`: List of successful optimizations with idea and source
- `failed_ideas`: List of failed optimizations with idea and reason
- `generation`: The generation number of this improvement
- `metrics`: Dictionary of flexible metrics (e.g., `tokens_used`, `execution_time`)
- `summary`: Summary of the improvement

**AgentVariant**: Represents an agent variant in the Pareto frontier

- `folder_path`: Path to the variant's source code
- `report_path`: Path to the variant's improvement report
- `report`: The ImprovementReport instance
- `metrics`: Dictionary of flexible metrics (e.g., `success`, `tokens_used`, `execution_time`)
- `id`: Unique variant identifier
- `generation`: Generation when created
- `parent_ids`: List of parent variant IDs
- `feedback`: Feedback from the agent evaluation
- `dominates(other, minimize)`: Check if this variant Pareto-dominates another
- `score(weights)`: Compute combined ranking score (lower is better)

## Configuration

Configuration can be provided via the global config system:

```python
from kiss.core.config import DEFAULT_CONFIG

# Access agent_creator config
cfg = DEFAULT_CONFIG.agent_creator

# Improver settings
cfg.improver.model_name = "claude-sonnet-4-5"
cfg.improver.max_steps = 150
cfg.improver.max_budget = 15.0

# Evolver settings
cfg.evolver.model_name = "claude-sonnet-4-5"
cfg.evolver.max_generations = 10
cfg.evolver.max_frontier_size = 6
cfg.evolver.mutation_probability = 0.8
cfg.evolver.initial_agent_max_steps = 50
cfg.evolver.initial_agent_max_budget = 5.0
cfg.evolver.coding_agent_type = "claude code"
```

## How It Works

### Pareto Frontier

The module uses **Pareto dominance** to compare solutions. A solution A dominates solution B if:

- A is at least as good as B in all objectives
- A is strictly better than B in at least one objective

The Pareto frontier contains all non-dominated solutions, representing the best trade-offs between objectives.

By default, `tokens_used` and `execution_time` are minimized, while `success` is maximized. The `dominates()` method accepts a `minimize` parameter to customize which metrics to minimize.

### Scoring

Variants are ranked using a combined score (lower is better). Default weights:

- `success`: -1,000,000 (maximize success - most important)
- `tokens_used`: 1 (minimize token usage)
- `execution_time`: 1,000 (minimize execution time)

### Evolutionary Operations

1. **Mutation**: Select one variant from the frontier and apply improvements
1. **Crossover**: Select two variants, use the better one (by score) as the base, and incorporate ideas from the other's improvement report

### Improvement Process

1. Copy source agent to target folder
1. Analyze code structure and existing optimizations
1. Apply optimizations (prompt reduction, caching, batching, etc.)
1. Generate improvement report with metrics
1. Update Pareto frontier and prune dominated variants

### Agent Creation

The `AgentEvolver` creates agents with these patterns:

- **Orchestrator Pattern**: Central coordinator managing workflow
- **Dynamic To-Do List**: Task tracking with dependencies and priorities
- **Dynamic Tool Creation**: On-the-fly tool generation for subtasks
- **Checkpointing**: State persistence for recovery
- **Sub-Agent Delegation**: Specialized agents for complex subtasks

## Output

### Improvement Report JSON

```json
{
    "implemented_ideas": [
        {"idea": "Reduced prompt verbosity", "source": "improver"}
    ],
    "failed_ideas": [
        {"idea": "Aggressive caching", "reason": "Caused correctness issues"}
    ],
    "generation": 5,
    "metrics": {
        "tokens_used": 8000,
        "execution_time": 25.0
    },
    "summary": "Optimized prompts and added caching for repeated operations"
}
```

### Evolver State JSON

```json
{
    "task_description": "Build a code analysis assistant...",
    "generation": 10,
    "variant_counter": 15,
    "pareto_frontier": [
        {
            "folder_path": "/path/to/variant_3",
            "report_path": "/path/to/variant_3/improvement_report.json",
            "report": {
                "implemented_ideas": [...],
                "failed_ideas": [...],
                "generation": 4,
                "metrics": {"tokens_used": 5000, "execution_time": 12.5},
                "summary": "..."
            },
            "metrics": {
                "success": 1,
                "tokens_used": 5000,
                "execution_time": 12.5
            },
            "id": 3,
            "generation": 4,
            "parent_ids": [1],
            "feedback": "Agent completed task successfully with minor issues..."
        }
    ]
}
```

## Optimization Strategies

The improver applies various optimization strategies:

- **Prompt Optimization**: Reduce verbosity while maintaining clarity
- **Caching**: Cache repeated operations and intermediate results
- **Batching**: Batch API calls and operations where possible
- **Algorithm Efficiency**: Use more efficient algorithms
- **Context Reduction**: Minimize unnecessary context in conversations
- **Early Termination**: Stop when goals are achieved
- **Incremental Processing**: Use streaming or incremental processing
- **Step Minimization**: Reduce agent steps while maintaining correctness

## API Reference

### ImproverAgent

```python
class ImproverAgent:
    def __init__(
        self,
        model_name: str | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        coding_agent_type: Literal["claude code", "gemini cli", "openai codex"] | None = None,
    ): ...

    async def improve(
        self,
        source_folder: str,
        target_folder: str,
        report_path: str | None = None,
        feedback: str = "",
        base_dir: str | None = None,
    ) -> tuple[bool, ImprovementReport | None]: ...

    async def crossover_improve(
        self,
        primary_folder: str,
        primary_report_path: str,
        secondary_report_path: str,
        primary_feedback: str,
        secondary_feedback: str,
        target_folder: str,
        base_dir: str | None = None,
    ) -> tuple[bool, ImprovementReport | None]: ...
```

### AgentEvolver

```python
class AgentEvolver:
    def __init__(
        self,
        task_description: str,
        model_name: str | None = None,
        max_generations: int | None = None,
        max_frontier_size: int | None = None,
        mutation_probability: float | None = None,
        coding_agent_type: Literal["claude code", "gemini cli", "openai codex"] | None = None,
    ): ...

    async def evolve(self) -> AgentVariant: ...
    def get_best_variant(self) -> AgentVariant: ...
    def get_pareto_frontier(self) -> list[AgentVariant]: ...
    def save_state(self, path: str) -> None: ...
```

### ImprovementReport

```python
@dataclass
class ImprovementReport:
    implemented_ideas: list[dict[str, str]] = field(default_factory=list)
    failed_ideas: list[dict[str, str]] = field(default_factory=list)
    generation: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "ImprovementReport": ...
```

### AgentVariant

```python
@dataclass
class AgentVariant:
    folder_path: str
    report_path: str
    report: ImprovementReport
    metrics: dict[str, float] = field(default_factory=dict)
    id: int = 0
    generation: int = 0
    parent_ids: list[int] = field(default_factory=list)
    feedback: str = ""

    def dominates(self, other: "AgentVariant", minimize: set[str] | None = None) -> bool: ...
    def score(self, weights: dict[str, float] | None = None) -> float: ...
    def to_dict(self) -> dict[str, Any]: ...
```

## License

See the main project LICENSE file.
