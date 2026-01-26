# Quick Start Guide

Get up and running with the Database Engine Agent in 5 minutes!

## Installation

```bash
# Clone or navigate to the agent directory
cd variant_1

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python -m pytest test_agent.py -v
```

## Basic Usage

### 1. Run the Agent

```bash
python agent.py
```

This will:
- Initialize a 10-task workflow for building the database engine
- Execute tasks in dependency order
- Save checkpoints every 5 tasks
- Output to `./db_engine_output/`

### 2. Watch Progress

The agent will output progress information:

```
============================================================
Executing: implement_basic_ops
Description: Implement set, get, delete operations
============================================================

Progress: 20.0% (2/10 tasks)
```

### 3. Check Results

After completion, inspect the output:

```bash
ls db_engine_output/
# my_db/          - Database directory
# db.sh           - Main database script
# test_stress.sh  - Stress test script
# checkpoints/    - Saved checkpoints
```

## Using Make Commands

```bash
# Quick demo
make demo

# Run with tests
make setup

# Run examples
make examples

# View all commands
make help
```

## Configuration Presets

### Fast Execution

```python
from config import get_config
from agent import OrchestratorAgent
import asyncio

async def main():
    config = get_config("fast")
    orchestrator = OrchestratorAgent(
        work_dir=config.work_dir,
        checkpoint_interval=config.checkpoint_interval
    )
    orchestrator.initialize_workflow()
    await orchestrator.execute_workflow()

asyncio.run(main())
```

### Custom Workflow

```python
from agent import OrchestratorAgent, Task, TaskPriority
from pathlib import Path
import asyncio

async def main():
    orchestrator = OrchestratorAgent(work_dir=Path("./custom_output"))

    # Add custom tasks
    orchestrator.todo_list.add_task(Task(
        id="t1",
        name="implement_feature",
        description="Build custom feature",
        priority=TaskPriority.HIGH
    ))

    await orchestrator.execute_workflow()

asyncio.run(main())
```

## Recovery from Checkpoint

If the agent is interrupted, resume from the last checkpoint:

```bash
python agent.py --checkpoint checkpoints/checkpoint_20260126_120000.json
```

## Examples

Run interactive examples:

```bash
# All examples
python examples.py

# Specific example
python examples.py basic
python examples.py checkpoint
python examples.py monitor
```

Available examples:
- `basic` - Basic usage
- `config` - Configuration presets
- `custom` - Custom workflows
- `checkpoint` - Checkpoint and recovery
- `monitor` - Progress monitoring
- `tools` - Custom tools
- `errors` - Error handling
- `status` - Agent status

## Testing the Database Engine

Once the agent completes, test the generated database:

```bash
cd db_engine_output

# Make scripts executable
chmod +x db.sh test_stress.sh

# Basic operations
./db.sh set mykey myvalue
./db.sh get mykey
./db.sh delete mykey

# Transactions
./db.sh begin
./db.sh set key1 value1
./db.sh set key2 value2
./db.sh commit

# Run stress test
./test_stress.sh
```

## Understanding the Output

### Console Output

```
=== Starting Orchestrator Workflow ===

============================================================
Executing: plan_architecture
Description: Design database architecture with locking...
============================================================

SubAgent 'planning_agent' executing task: plan_architecture

Progress: 10.0% (1/10 tasks)

Checkpoint saved: checkpoints/checkpoint_20260126_120000.json
```

### File Structure

```
db_engine_output/
├── checkpoints/
│   └── checkpoint_20260126_120000.json    # State snapshots
├── my_db/                                 # Database storage
│   ├── .transactions/                     # Transaction logs
│   └── (key files)                        # Data files
├── db.sh                                  # Main DB script
└── test_stress.sh                         # Stress test
```

### Checkpoint File

```json
{
  "timestamp": 1706259600.0,
  "tasks": [...],
  "completed_ids": ["task_001", "task_002"],
  "tools_created": ["write_file", "read_file", "run_shell"],
  "metadata": {
    "session_id": "20260126_120000",
    "progress": {
      "total": 10,
      "completed": 2,
      "completion_percentage": 20.0
    }
  }
}
```

## Common Patterns

### Monitoring Progress

```python
import asyncio
from agent import OrchestratorAgent
from pathlib import Path

async def monitor_progress(orchestrator):
    """Print progress every 2 seconds"""
    while True:
        progress = orchestrator.todo_list.get_progress()
        print(f"Progress: {progress['completion_percentage']:.1f}%")

        if progress['completion_percentage'] >= 100:
            break

        await asyncio.sleep(2)

async def main():
    orchestrator = OrchestratorAgent(work_dir=Path("./output"))
    orchestrator.initialize_workflow()

    # Run monitoring and execution concurrently
    await asyncio.gather(
        orchestrator.execute_workflow(),
        monitor_progress(orchestrator)
    )

asyncio.run(main())
```

### Custom Tools

```python
from agent import OrchestratorAgent
from pathlib import Path

orchestrator = OrchestratorAgent(work_dir=Path("./output"))

# Register custom tool
async def validate_syntax(script_path: str) -> bool:
    """Validate bash script syntax"""
    import subprocess
    result = subprocess.run(
        ["bash", "-n", script_path],
        capture_output=True
    )
    return result.returncode == 0

orchestrator.register_tool(
    "validate_bash",
    "Validate bash syntax",
    validate_syntax
)

# Use in workflow...
```

### Error Handling

```python
from agent import OrchestratorAgent, Task
from pathlib import Path
import asyncio

async def main():
    orchestrator = OrchestratorAgent(work_dir=Path("./output"))

    # Task with custom retry settings
    task = Task(
        id="t1",
        name="implement_complex",
        description="Complex implementation",
        max_retries=5  # Will retry up to 5 times
    )

    orchestrator.todo_list.add_task(task)

    try:
        await orchestrator.execute_workflow()
    except Exception as e:
        print(f"Workflow failed: {e}")
        # Checkpoint is automatically saved
        print("Progress saved to checkpoint")

asyncio.run(main())
```

## Troubleshooting

### Issue: Tests Fail

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run with verbose output
python -m pytest test_agent.py -v
```

### Issue: Permission Denied on Scripts

```bash
chmod +x db.sh test_stress.sh
```

### Issue: Import Errors

```bash
# Ensure you're in the correct directory
cd variant_1

# Add to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Checkpoint Not Loading

```bash
# Check checkpoint file exists
ls checkpoints/

# Verify JSON is valid
python -c "import json; print(json.load(open('checkpoints/checkpoint_*.json')))"
```

## Next Steps

1. **Read the Architecture**: See `ARCHITECTURE.md` for design details
2. **Explore Examples**: Run `python examples.py` for interactive demos
3. **Customize**: Modify `config.py` for your needs
4. **Extend**: Add custom agents and tools for new capabilities
5. **Test**: Run `make test-cov` for coverage analysis

## Performance Tips

1. **Adjust Checkpoint Interval**
   ```bash
   python agent.py --checkpoint-interval 10  # Less frequent checkpoints
   ```

2. **Use Fast Preset**
   ```python
   config = get_config("fast")
   ```

3. **Disable Verbose Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.WARNING)
   ```

## Getting Help

- **Documentation**: See `README.md` for full documentation
- **Architecture**: See `ARCHITECTURE.md` for design details
- **Examples**: Run `python examples.py` for code samples
- **Tests**: Check `test_agent.py` for usage patterns

## Minimal Example

Here's the absolute minimal code to run the agent:

```python
import asyncio
from agent import OrchestratorAgent
from pathlib import Path

async def main():
    agent = OrchestratorAgent(work_dir=Path("./output"))
    agent.initialize_workflow()
    await agent.execute_workflow()

asyncio.run(main())
```

That's it! The agent handles everything else automatically.
