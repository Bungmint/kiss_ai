# Database Engine Agent

A sophisticated long-running agent system for building robust bash-based database engines with transaction support, concurrency control, and comprehensive testing.

## Architecture

This agent implements a **hierarchical orchestration pattern** based on state-of-the-art research in long-horizon task planning, specifically the Plan-and-Act framework and Deep Agents architecture.

### Core Components

#### 1. **Orchestrator Agent**
- Central coordinator managing the entire workflow
- Maintains high-level state and progress tracking
- Delegates subtasks to specialized sub-agents
- Handles error recovery and automatic retries
- Implements checkpointing for fault tolerance

#### 2. **Dynamic To-Do List**
- Structured task list that evolves during execution
- Support for task dependencies and priorities
- Automatic task scheduling based on dependency resolution
- Real-time progress tracking

#### 3. **Sub-Agent Delegation**
- Specialized lightweight agents for focused tasks:
  - **Implementation Agent**: Builds database components
  - **Testing Agent**: Executes and validates tests
  - **Validation Agent**: Ensures requirements are met
  - **Planning Agent**: Designs architecture
- Each sub-agent has access to relevant tools
- Execution history tracking per agent

#### 4. **Dynamic Tool Creation**
- Tools created on-the-fly for specific needs
- Reusable across execution
- Built-in tools:
  - `write_file`: Write content to files
  - `read_file`: Read file content
  - `run_shell`: Execute bash commands safely

#### 5. **Checkpointing System**
- Periodic state saves for recovery
- Resume from any checkpoint
- Track completed vs pending tasks
- Metadata storage for context

### Design Patterns

#### Efficiency Patterns
Based on research into Python and Bash optimization:

- **Minimal Token Usage**: Concise context prompts (truncated descriptions, compact JSON)
- **Async/Await**: Non-blocking I/O operations using asyncio
- **Bash Built-ins**: Prefer built-in commands over external utilities
- **Batched Operations**: Group file operations where possible
- **Early Termination**: Exit when goals achieved
- **Result Caching**: Cache intermediate results to avoid recomputation

#### Robustness Patterns
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Isolation**: Sub-agent failures don't crash orchestrator
- **State Persistence**: Checkpoints enable crash recovery
- **Timeout Protection**: All operations have timeouts

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import asyncio
from agent import OrchestratorAgent
from pathlib import Path

async def main():
    # Create orchestrator
    orchestrator = OrchestratorAgent(
        work_dir=Path("./output"),
        checkpoint_interval=5
    )

    # Initialize workflow
    orchestrator.initialize_workflow()

    # Execute
    await orchestrator.execute_workflow()

asyncio.run(main())
```

### Command Line

```bash
# Run with defaults
python agent.py

# Specify work directory
python agent.py --work-dir ./my_output

# Resume from checkpoint
python agent.py --checkpoint checkpoints/checkpoint_20260126_120000.json

# Adjust checkpoint frequency
python agent.py --checkpoint-interval 10
```

### Using Configuration Presets

```python
from config import get_config

# Fast preset (fewer checkpoints, shorter timeouts)
config = get_config("fast")

# Balanced preset (recommended)
config = get_config("balanced")

# Thorough preset (more validation, more retries)
config = get_config("thorough")

orchestrator = OrchestratorAgent(
    work_dir=config.work_dir,
    checkpoint_interval=config.checkpoint_interval
)
```

## Workflow Overview

The agent implements the following workflow for building the database engine:

### Phase 1: Planning
1. **Task 001**: Design architecture with locking and transactions

### Phase 2: Core Implementation
2. **Task 002**: Implement basic operations (set, get, delete)
3. **Task 003**: Implement mkdir-based mutex locking
4. **Task 004**: Implement transaction support (begin, commit, rollback)

### Phase 3: Integration
5. **Task 005**: Create main `db.sh` script

### Phase 4: Testing
6. **Task 006**: Create `test_stress.sh` with concurrent processes
7. **Task 007**: Test basic operations
8. **Task 008**: Test transaction atomicity
9. **Task 009**: Run stress test (10 concurrent processes)

### Phase 5: Validation
10. **Task 010**: Final validation of all requirements

## Task Dependencies

```
task_001 (planning)
├── task_002 (basic ops)
│   └── task_004 (transactions)
│       └── task_005 (db.sh)
│           ├── task_006 (stress test)
│           │   └── task_009 (run stress)
│           ├── task_007 (test basic)
│           └── task_008 (test transactions)
└── task_003 (locking)
    └── task_004 (transactions)

task_009, task_007, task_008 → task_010 (validation)
```

## Output Structure

```
output/
├── checkpoints/
│   └── checkpoint_20260126_120000.json
├── my_db/              # Database directory
│   └── (data files)
├── db.sh               # Main database script
├── test_stress.sh      # Stress test script
└── logs/
    └── agent.log
```

## API Reference

### OrchestratorAgent

Main orchestrator class.

```python
class OrchestratorAgent:
    def __init__(self, work_dir: Path, checkpoint_interval: int = 5)

    async def execute_workflow()
    """Execute the complete workflow"""

    async def save_checkpoint()
    """Save current state"""

    async def load_checkpoint(checkpoint_file: Path) -> bool
    """Load from checkpoint"""

    def initialize_workflow()
    """Initialize task workflow"""

    def get_status() -> Dict[str, Any]
    """Get current status"""
```

### SubAgent

Specialized agent for focused tasks.

```python
class SubAgent:
    def __init__(self, name: str, purpose: str, tools: List[DynamicTool])

    async def execute_task(task: Task, context: Dict) -> Any
    """Execute a task with context"""
```

### DynamicTodoList

Task management with dependencies.

```python
class DynamicTodoList:
    def add_task(task: Task)
    """Add new task"""

    def get_next_task() -> Optional[Task]
    """Get next available task"""

    def complete_task(task_id: str, result: Any)
    """Mark task completed"""

    def get_progress() -> Dict[str, Any]
    """Get progress statistics"""
```

### Task

Task representation.

```python
@dataclass
class Task:
    id: str
    name: str
    description: str
    status: TaskStatus
    priority: TaskPriority
    dependencies: List[str]
    assigned_agent: Optional[str]
    result: Optional[Any]
    max_retries: int = 3
```

## Configuration

All configuration is centralized in `config.py`. Key parameters:

```python
@dataclass
class AgentConfig:
    work_dir: Path = Path("./db_engine_output")
    checkpoint_interval: int = 5
    max_task_retries: int = 3
    shell_timeout: int = 30
    task_timeout: int = 300
    max_concurrent_subagents: int = 3
    log_level: str = "INFO"

    # Database specific
    db_dir_name: str = "my_db"
    stress_test_processes: int = 10
    stress_test_operations: int = 100
```

## Testing

```bash
# Run agent tests
python -m pytest test_agent.py -v

# Run with coverage
python -m pytest test_agent.py --cov=agent --cov-report=html
```

## Performance Characteristics

### Token Efficiency
- **Minimal Prompts**: Context truncated to essentials (~100 chars max)
- **Compact JSON**: No whitespace in serialization
- **Result Caching**: Avoid redundant computations

### Execution Speed
- **Async I/O**: Non-blocking operations
- **Parallel Ready**: Infrastructure for parallel sub-agents
- **Bash Optimization**: Uses built-in commands, avoids pipes

### Resource Usage
- **Low Memory**: Lightweight sub-agents
- **Disk Efficient**: Incremental checkpoints
- **CPU Optimized**: Minimal subprocess spawning

## Research Foundation

This agent is based on cutting-edge research:

### Long-Horizon Task Planning
- **Plan-and-Act Framework**: Separates planning from execution ([arxiv.org](https://arxiv.org/html/2503.09572v3))
- **Deep Agents Architecture**: 4 pillars - planning, delegation, memory, context ([Medium](https://medium.com/@amirkiarafiei/the-agent-2-0-era-mastering-long-horizon-tasks-with-deep-agents-part-1-c566efaa951b))
- **Hierarchical Orchestration**: Manager-subordinate pattern ([AIMultiple](https://research.aimultiple.com/agentic-orchestration/))

### Performance Optimization
- **Python Asyncio**: Modern concurrency patterns ([FastAPI](https://fastapi.tiangolo.com/async/))
- **Bash Optimization**: Built-ins over external commands ([DEV](https://dev.to/heinanca/stop-writing-slow-bash-scripts-performance-optimization-techniques-that-actually-work-181b))

## Limitations

- Currently single-threaded (parallel execution planned)
- Sub-agents are simulated (can be extended with LLM calls)
- Bash script implementation is templated (can be made dynamic)

## Future Enhancements

1. **True Parallel Execution**: Run independent tasks concurrently
2. **LLM Integration**: Connect sub-agents to actual LLMs
3. **Advanced Caching**: More sophisticated result caching
4. **Distributed Execution**: Run sub-agents on different machines
5. **Web Dashboard**: Real-time progress visualization

## Contributing

Contributions welcome! Please ensure:
- Code follows style guidelines (simple, readable, minimal)
- All functions have tests
- Tests use real inputs, no mocks
- Documentation is updated

## License

MIT License - See LICENSE file for details

## References

### Long-Horizon Agent Research
- [Deep Agents: Long-Horizon Task Completion Framework](https://ssahuupgrad-93226.medium.com/deep-agents-long-horizon-task-completion-framework-8a702ce9da18)
- [Plan-and-Act: Improving Planning of Agents](https://arxiv.org/html/2503.09572v3)
- [Agent 2.0 Era: Mastering Long-Horizon Tasks](https://medium.com/@amirkiarafiei/the-agent-2-0-era-mastering-long-horizon-tasks-with-deep-agents-part-1-c566efaa951b)
- [Agentic Orchestration Frameworks 2026](https://research.aimultiple.com/agentic-orchestration/)

### Performance Optimization
- [FastAPI Async/Await](https://fastapi.tiangolo.com/async/)
- [Python Concurrency Guide](https://realpython.com/python-concurrency/)
- [Stop Writing Slow Bash Scripts](https://dev.to/heinanca/stop-writing-slow-bash-scripts-performance-optimization-techniques-that-actually-work-181b)
- [Bash Performance Optimization](https://moldstud.com/articles/p-top-bash-script-optimization-best-practices-for-enhanced-speed-and-efficiency)
