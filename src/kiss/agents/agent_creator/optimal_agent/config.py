"""
Configuration module for the Database Engine Agent

Centralizes all configuration parameters for easy tuning
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass
class AgentConfig:
    """Main agent configuration"""

    # Working directory
    work_dir: Path = Path("./db_engine_output")

    # Checkpoint settings
    checkpoint_interval: int = 5  # Tasks between checkpoints
    checkpoint_dir_name: str = "checkpoints"
    auto_checkpoint: bool = True

    # Retry settings
    max_task_retries: int = 3
    retry_delay: float = 1.0  # seconds

    # Timeout settings
    shell_timeout: int = 30  # seconds
    task_timeout: int = 300  # seconds (5 minutes)

    # Concurrency settings
    max_concurrent_subagents: int = 3
    enable_parallel_execution: bool = False  # Future enhancement

    # Logging settings
    log_level: str = "INFO"
    log_file: str = "agent.log"
    verbose: bool = True

    # Performance tuning
    cache_tool_results: bool = True
    minimize_prompts: bool = True  # Use minimal token prompts
    batch_file_operations: bool = True

    # Database engine specific settings
    db_dir_name: str = "my_db"
    db_script_name: str = "db.sh"
    test_script_name: str = "test_stress.sh"

    # Stress test parameters
    stress_test_processes: int = 10
    stress_test_operations: int = 100
    stress_test_timeout: int = 60

    # Sub-agent configuration
    subagent_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "implementation_agent": {
            "max_complexity": "high",
            "optimization_level": "performance",
            "code_style": "minimal"
        },
        "testing_agent": {
            "test_thoroughness": "comprehensive",
            "parallel_tests": False,
            "coverage_target": 90.0
        },
        "validation_agent": {
            "strict_mode": True,
            "validate_bash_syntax": True,
            "validate_permissions": True
        },
        "planning_agent": {
            "detail_level": "high",
            "consider_alternatives": True,
            "document_decisions": True
        }
    })

    def __post_init__(self):
        """Validate and normalize configuration"""
        self.work_dir = Path(self.work_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'work_dir': str(self.work_dir),
            'checkpoint_interval': self.checkpoint_interval,
            'checkpoint_dir_name': self.checkpoint_dir_name,
            'auto_checkpoint': self.auto_checkpoint,
            'max_task_retries': self.max_task_retries,
            'retry_delay': self.retry_delay,
            'shell_timeout': self.shell_timeout,
            'task_timeout': self.task_timeout,
            'max_concurrent_subagents': self.max_concurrent_subagents,
            'enable_parallel_execution': self.enable_parallel_execution,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'verbose': self.verbose,
            'cache_tool_results': self.cache_tool_results,
            'minimize_prompts': self.minimize_prompts,
            'batch_file_operations': self.batch_file_operations,
            'db_dir_name': self.db_dir_name,
            'db_script_name': self.db_script_name,
            'test_script_name': self.test_script_name,
            'stress_test_processes': self.stress_test_processes,
            'stress_test_operations': self.stress_test_operations,
            'stress_test_timeout': self.stress_test_timeout,
            'subagent_configs': self.subagent_configs
        }


# Default configuration instance
DEFAULT_CONFIG = AgentConfig()


# Performance optimization presets
PERFORMANCE_PRESETS = {
    "fast": AgentConfig(
        checkpoint_interval=10,
        max_task_retries=2,
        shell_timeout=20,
        minimize_prompts=True,
        cache_tool_results=True
    ),
    "balanced": AgentConfig(
        checkpoint_interval=5,
        max_task_retries=3,
        shell_timeout=30,
        minimize_prompts=True,
        cache_tool_results=True
    ),
    "thorough": AgentConfig(
        checkpoint_interval=3,
        max_task_retries=5,
        shell_timeout=60,
        minimize_prompts=False,
        cache_tool_results=True
    )
}


def get_config(preset: str = "balanced") -> AgentConfig:
    """Get configuration by preset name"""
    if preset in PERFORMANCE_PRESETS:
        return PERFORMANCE_PRESETS[preset]
    return DEFAULT_CONFIG
