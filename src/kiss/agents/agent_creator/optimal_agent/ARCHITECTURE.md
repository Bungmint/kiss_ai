# Agent Architecture Document

## Overview

This document describes the architecture of the Database Engine Agent, a sophisticated long-running agent system implementing hierarchical orchestration patterns for complex task execution.

## Design Philosophy

### Core Principles

1. **Separation of Concerns**: Planning, execution, and validation are handled by specialized sub-agents
2. **Fault Tolerance**: Checkpointing enables recovery from failures
3. **Efficiency**: Minimal token usage through concise prompts and result caching
4. **Scalability**: Hierarchical delegation allows handling complex multi-step tasks
5. **Simplicity**: Clean, readable code with minimal indirection

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                   Orchestrator Agent                     │
│  (Workflow Management, State Tracking, Coordination)     │
└────────────┬────────────────────────────────┬───────────┘
             │                                │
    ┌────────▼────────┐              ┌───────▼────────┐
    │  Dynamic Tools  │              │   Sub-Agents   │
    │  - write_file   │              │ - Implementation│
    │  - read_file    │              │ - Testing       │
    │  - run_shell    │              │ - Validation    │
    └─────────────────┘              │ - Planning      │
                                     └─────────────────┘
             │
    ┌────────▼────────┐
    │ Checkpoint      │
    │ System          │
    └─────────────────┘
```

## Component Breakdown

### 1. Orchestrator Agent (`OrchestratorAgent`)

**Purpose**: Central coordinator managing the entire workflow.

**Responsibilities**:
- Maintain high-level state and progress
- Schedule tasks based on dependencies and priorities
- Delegate subtasks to specialized sub-agents
- Handle error recovery and retries
- Implement checkpointing for fault tolerance
- Manage tool registry

**Key Methods**:
- `initialize_workflow()`: Set up task dependency graph
- `execute_workflow()`: Main execution loop
- `save_checkpoint()`: Persist current state
- `load_checkpoint()`: Restore from checkpoint
- `create_sub_agent()`: Instantiate specialized agents

**State Management**:
```python
{
    'todo_list': DynamicTodoList,
    'sub_agents': Dict[str, SubAgent],
    'tools': Dict[str, DynamicTool],
    'context': Dict[str, Any],  # Shared context
    'completed_ids': List[str]
}
```

### 2. Dynamic To-Do List (`DynamicTodoList`)

**Purpose**: Intelligent task queue with dependency resolution.

**Features**:
- Dependency-aware scheduling
- Priority-based ordering
- Real-time progress tracking
- Task state management

**Scheduling Algorithm**:
```python
def get_next_task():
    1. Filter pending tasks
    2. Check dependency satisfaction
    3. Sort by priority (CRITICAL > HIGH > MEDIUM > LOW)
    4. Return highest priority available task
```

**State Transitions**:
```
PENDING → IN_PROGRESS → COMPLETED
                      → FAILED (with retry) → PENDING
                                           → FAILED (max retries)
```

### 3. Sub-Agent Delegation (`SubAgent`)

**Purpose**: Lightweight specialized agents for focused tasks.

**Agent Types**:

1. **Implementation Agent**
   - Generates bash scripts
   - Optimizes for performance
   - Follows best practices

2. **Testing Agent**
   - Creates test scripts
   - Validates functionality
   - Measures coverage

3. **Validation Agent**
   - Checks requirements
   - Validates syntax
   - Ensures correctness

4. **Planning Agent**
   - Designs architecture
   - Makes technical decisions
   - Documents approach

**Context Minimization**:
Sub-agents receive minimal context to reduce token usage:
```python
{
    'task': task.name,
    'desc': task.description[:100],  # Truncated
    'deps': [results of dependencies]  # Only relevant data
}
```

### 4. Dynamic Tools (`DynamicTool`)

**Purpose**: Reusable operations for file I/O, shell execution, etc.

**Built-in Tools**:
- `write_file`: Write content to filesystem
- `read_file`: Read file content
- `run_shell`: Execute bash commands safely

**Tool Registration**:
Tools are registered dynamically and can be created during execution:
```python
orchestrator.register_tool(
    name="custom_tool",
    description="Tool description",
    func=tool_function
)
```

**Tool Execution**:
- Tracks usage count
- Supports both sync and async functions
- Automatic error handling

### 5. Checkpointing System (`Checkpoint`)

**Purpose**: Enable fault tolerance and resumption.

**Checkpoint Contents**:
```python
{
    'timestamp': float,
    'tasks': List[Dict],  # All task states
    'completed_ids': List[str],
    'tools_created': List[str],
    'metadata': {
        'session_id': str,
        'progress': Dict
    }
}
```

**Checkpoint Frequency**:
- Configurable interval (default: every 5 tasks)
- Automatic on workflow completion
- Manual on interruption (Ctrl+C)

**Recovery Process**:
1. Load checkpoint JSON
2. Restore todo list state
3. Resume from last completed task
4. Continue execution

## Workflow Execution

### Task Execution Flow

```
1. Get next available task (dependency-aware)
2. Mark task as IN_PROGRESS
3. Determine appropriate sub-agent
4. Create sub-agent if doesn't exist
5. Build minimal context prompt
6. Delegate to sub-agent
7. Collect and store result
8. Mark task COMPLETED
9. Checkpoint if interval reached
10. Report progress
11. Repeat until all tasks done
```

### Error Handling Flow

```
Task Execution
    │
    ├─ Success → Mark COMPLETED → Continue
    │
    └─ Failure
        │
        ├─ retries < max_retries
        │   └─ Increment retry counter
        │       └─ Mark PENDING
        │           └─ Retry later
        │
        └─ retries >= max_retries
            └─ Mark FAILED
                └─ Log error
                    └─ Continue with other tasks
```

## Performance Optimizations

### Token Efficiency

1. **Minimal Prompts**
   - Truncate descriptions to 100 chars
   - Use compact JSON (no whitespace)
   - Only include essential dependencies

2. **Context Caching**
   - Store task results in shared context
   - Avoid redundant computations
   - Reuse tool instances

3. **Batching**
   - Group file operations
   - Minimize subprocess spawning

### Python Performance

1. **Async/Await**
   - Non-blocking I/O operations
   - Concurrent tool execution (future)
   - Efficient resource utilization

2. **Data Structures**
   - Dict for O(1) lookup
   - List for ordered iteration
   - Dataclasses for efficiency

### Bash Performance

1. **Built-in Commands**
   - Prefer bash built-ins over external utilities
   - Minimize process creation
   - Use parameter expansion

2. **Locking Strategy**
   - mkdir-based atomic locking
   - Short lock hold times
   - Exponential backoff on contention

## Extensibility Points

### Adding New Agent Types

```python
def _get_agent_for_task(self, task: Task) -> str:
    if task.name.startswith("new_type_"):
        return "new_agent_type"
    # ... existing logic
```

### Adding New Tools

```python
async def custom_tool(param: str) -> Any:
    # Tool implementation
    pass

orchestrator.register_tool("tool_name", "description", custom_tool)
```

### Customizing Workflow

```python
# Override initialize_workflow
orchestrator.todo_list.add_task(Task(...))
```

### Configuring Behavior

```python
# Use config.py
config = AgentConfig(
    checkpoint_interval=10,
    max_task_retries=5,
    # ... other settings
)
```

## Research Foundation

### Long-Horizon Task Planning

Based on **Plan-and-Act** framework:
- Separate planning from execution
- Hierarchical task decomposition
- Adaptive planning with feedback

Reference: [arxiv.org/html/2503.09572v3](https://arxiv.org/html/2503.09572v3)

### Deep Agents Architecture

Implements four pillars:
1. **Explicit Planning**: Task workflow with dependencies
2. **Hierarchical Delegation**: Specialized sub-agents
3. **Persistent Memory**: Checkpointing system
4. **Context Engineering**: Minimal token prompts

Reference: [Medium - Agent 2.0 Era](https://medium.com/@amirkiarafiei/the-agent-2-0-era-mastering-long-horizon-tasks-with-deep-agents-part-1-c566efaa951b)

### Hierarchical Orchestration

Manager-subordinate pattern:
- Orchestrator as central coordinator
- Sub-agents as specialized workers
- Clear delegation boundaries

Reference: [AIMultiple - Agentic Orchestration](https://research.aimultiple.com/agentic-orchestration/)

## Scalability Considerations

### Current Limitations

1. **Single-threaded**: Tasks execute sequentially
2. **Local only**: No distributed execution
3. **In-memory state**: Limited by available RAM

### Future Enhancements

1. **True Parallelization**
   - Run independent tasks concurrently
   - Use asyncio.gather() for parallel sub-agents
   - Implement semaphore-based rate limiting

2. **Distributed Execution**
   - Remote sub-agent execution
   - Message queue for task distribution
   - Shared checkpoint storage

3. **Persistent Storage**
   - Database-backed task queue
   - Stream large results to disk
   - Incremental checkpointing

## Testing Strategy

### Unit Tests
- Test each class independently
- Use real inputs, no mocks
- Verify state transitions
- Test edge cases

### Integration Tests
- Test workflow execution end-to-end
- Verify checkpoint recovery
- Test concurrent tool execution
- Validate error handling

### Stress Tests
- Large task graphs (100+ tasks)
- Multiple checkpoint cycles
- Simulated failures
- Resource monitoring

## Monitoring and Observability

### Logging

Structured logging at multiple levels:
- INFO: Task execution, progress updates
- WARNING: Retries, blocked tasks
- ERROR: Failures, exceptions
- DEBUG: Detailed execution flow (optional)

### Metrics

Track key metrics:
- Task completion rate
- Average task duration
- Retry rate
- Checkpoint frequency
- Tool usage counts

### Progress Reporting

Real-time progress information:
```python
{
    'total': int,
    'completed': int,
    'failed': int,
    'in_progress': int,
    'pending': int,
    'completion_percentage': float
}
```

## Security Considerations

1. **Input Validation**
   - Sanitize shell commands
   - Validate file paths
   - Check key names (alphanumeric only)

2. **Sandboxing**
   - Operate within work_dir only
   - No access to parent directories
   - Configurable timeouts

3. **Lock Safety**
   - Prevent deadlocks
   - Timeout on lock acquisition
   - Clean up on process exit

## Conclusion

This architecture provides a robust foundation for long-running agent tasks through:
- Clear separation of concerns
- Fault-tolerant execution
- Efficient resource usage
- Easy extensibility
- Research-backed design patterns

The system balances simplicity with sophistication, making it suitable for both the current database engine task and future complex workflows.
