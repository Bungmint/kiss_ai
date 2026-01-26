"""
Database Engine Agent Package

A hierarchical orchestration agent for building robust bash-based database systems
with support for transactions, concurrency control, and stress testing.

Key Features:
- Orchestrator pattern for long-horizon tasks
- Dynamic to-do list with dependency tracking
- Checkpointing for fault tolerance
- Sub-agent delegation for specialized tasks
- Dynamic tool creation
- Efficient token usage through minimal prompts

Usage:
    from agent import OrchestratorAgent
    from config import AgentConfig

    config = AgentConfig(work_dir="./output")
    orchestrator = OrchestratorAgent(work_dir=config.work_dir)
    orchestrator.initialize_workflow()
    await orchestrator.execute_workflow()
"""

from .agent import (
    OrchestratorAgent,
    SubAgent,
    DynamicTool,
    DynamicTodoList,
    Task,
    TaskStatus,
    TaskPriority,
    Checkpoint
)

from .config import (
    AgentConfig,
    DEFAULT_CONFIG,
    PERFORMANCE_PRESETS,
    get_config
)

__version__ = "1.0.0"

__all__ = [
    # Agent classes
    "OrchestratorAgent",
    "SubAgent",
    "DynamicTool",
    "DynamicTodoList",

    # Data classes
    "Task",
    "TaskStatus",
    "TaskPriority",
    "Checkpoint",

    # Configuration
    "AgentConfig",
    "DEFAULT_CONFIG",
    "PERFORMANCE_PRESETS",
    "get_config",
]
