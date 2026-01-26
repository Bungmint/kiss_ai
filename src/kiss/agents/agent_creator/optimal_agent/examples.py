"""
Usage Examples for the Database Engine Agent

This module demonstrates how to use the agent system
for various scenarios.
"""

import asyncio
from pathlib import Path

from agent import OrchestratorAgent, Task, TaskPriority, TaskStatus
from config import get_config


async def example_basic_usage():
    """Basic usage example"""
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create orchestrator with default settings
    orchestrator = OrchestratorAgent(
        work_dir=Path("./output_basic"),
        checkpoint_interval=5
    )

    # Initialize the complete workflow
    orchestrator.initialize_workflow()

    # Show initial status
    status = orchestrator.get_status()
    print(f"Session ID: {status['session_id']}")
    print(f"Total tasks: {status['progress']['total']}")
    print(f"Available tools: {', '.join(status['tools'])}")

    # Execute workflow
    print("\nExecuting workflow...\n")
    await orchestrator.execute_workflow()

    # Show final results
    final_status = orchestrator.get_status()
    print(f"\nFinal completion: {final_status['progress']['completion_percentage']:.1f}%")


async def example_with_config_preset():
    """Using configuration presets"""
    print("\nExample 2: Using Configuration Presets")
    print("=" * 60)

    # Use 'fast' preset for quicker execution
    config = get_config("fast")

    orchestrator = OrchestratorAgent(
        work_dir=config.work_dir,
        checkpoint_interval=config.checkpoint_interval
    )

    print(f"Using preset: fast")
    print(f"Checkpoint interval: {config.checkpoint_interval}")
    print(f"Max retries: {config.max_task_retries}")

    orchestrator.initialize_workflow()
    await orchestrator.execute_workflow()


async def example_custom_workflow():
    """Creating a custom workflow"""
    print("\nExample 3: Custom Workflow")
    print("=" * 60)

    orchestrator = OrchestratorAgent(
        work_dir=Path("./output_custom")
    )

    # Don't use initialize_workflow - create custom tasks instead
    orchestrator.todo_list.add_task(Task(
        id="custom_001",
        name="implement_feature_x",
        description="Implement feature X with special requirements",
        priority=TaskPriority.CRITICAL
    ))

    orchestrator.todo_list.add_task(Task(
        id="custom_002",
        name="test_feature_x",
        description="Test feature X thoroughly",
        priority=TaskPriority.HIGH,
        dependencies=["custom_001"]
    ))

    orchestrator.todo_list.add_task(Task(
        id="custom_003",
        name="validate_feature_x",
        description="Validate feature X meets requirements",
        priority=TaskPriority.HIGH,
        dependencies=["custom_002"]
    ))

    print("Created custom workflow with 3 tasks")

    # Execute
    await orchestrator.execute_workflow()


async def example_checkpoint_recovery():
    """Checkpoint and recovery example"""
    print("\nExample 4: Checkpoint and Recovery")
    print("=" * 60)

    work_dir = Path("./output_checkpoint")

    # Phase 1: Start work and checkpoint
    print("Phase 1: Starting initial work...")
    orch1 = OrchestratorAgent(work_dir=work_dir, checkpoint_interval=2)
    orch1.initialize_workflow()

    # Complete a few tasks manually
    for _ in range(3):
        task = orch1.todo_list.get_next_task()
        if task:
            orch1.todo_list.complete_task(task.id, "completed manually")

    await orch1.save_checkpoint()
    print(f"Completed 3 tasks and saved checkpoint")

    # Phase 2: Simulate crash and recovery
    print("\nPhase 2: Recovering from checkpoint...")
    orch2 = OrchestratorAgent(work_dir=work_dir)

    # Find latest checkpoint
    checkpoint_files = sorted(orch1.checkpoint_dir.glob("*.json"))
    if checkpoint_files:
        latest = checkpoint_files[-1]
        success = await orch2.load_checkpoint(latest)

        if success:
            print(f"Successfully loaded checkpoint: {latest.name}")
            progress = orch2.todo_list.get_progress()
            print(f"Recovered state: {progress['completed']} tasks completed")

            # Continue execution
            print("\nContinuing workflow from checkpoint...")
            await orch2.execute_workflow()
        else:
            print("Failed to load checkpoint")


async def example_monitoring_progress():
    """Monitoring progress during execution"""
    print("\nExample 5: Monitoring Progress")
    print("=" * 60)

    orchestrator = OrchestratorAgent(work_dir=Path("./output_monitor"))
    orchestrator.initialize_workflow()

    # Create a monitoring task
    async def monitor():
        """Monitor progress every 2 seconds"""
        while True:
            progress = orchestrator.todo_list.get_progress()
            print(f"Progress: {progress['completion_percentage']:.1f}% "
                  f"({progress['completed']}/{progress['total']} tasks)")

            if progress['completion_percentage'] >= 100:
                break

            await asyncio.sleep(2)

    # Run orchestrator and monitor concurrently
    await asyncio.gather(
        orchestrator.execute_workflow(),
        monitor()
    )


async def example_custom_tools():
    """Creating and using custom tools"""
    print("\nExample 6: Custom Tools")
    print("=" * 60)

    orchestrator = OrchestratorAgent(work_dir=Path("./output_tools"))

    # Register custom tool
    async def validate_bash_syntax(script_path: str) -> bool:
        """Validate bash script syntax"""
        import subprocess
        result = subprocess.run(
            ["bash", "-n", script_path],
            capture_output=True,
            text=True
        )
        return result.returncode == 0

    orchestrator.register_tool(
        "validate_bash",
        "Validate bash script syntax",
        validate_bash_syntax
    )

    print(f"Registered custom tool: validate_bash")
    print(f"Total tools: {len(orchestrator.tools)}")

    # Create agent with custom tool
    agent = orchestrator.create_sub_agent(
        "validation_expert",
        "Expert in bash validation",
        ["validate_bash", "read_file"]
    )

    print(f"Created sub-agent with {len(agent.tools)} tools")


async def example_error_handling():
    """Error handling and retry example"""
    print("\nExample 7: Error Handling and Retry")
    print("=" * 60)

    orchestrator = OrchestratorAgent(work_dir=Path("./output_errors"))

    # Create task that might fail
    task = Task(
        id="error_001",
        name="implement_complex",
        description="Complex task that might fail",
        priority=TaskPriority.HIGH,
        max_retries=5
    )

    orchestrator.todo_list.add_task(task)

    print(f"Task max retries: {task.max_retries}")

    # Note: In real execution, failed tasks will be automatically retried
    print("Task will be retried up to 5 times on failure")


async def example_get_agent_status():
    """Getting detailed agent status"""
    print("\nExample 8: Agent Status Information")
    print("=" * 60)

    orchestrator = OrchestratorAgent(work_dir=Path("./output_status"))
    orchestrator.initialize_workflow()

    # Get comprehensive status
    status = orchestrator.get_status()

    print("\nAgent Status:")
    print(f"  Session ID: {status['session_id']}")
    print(f"  Work Directory: {status['work_dir']}")
    print(f"  Completed Tasks: {status['completed_tasks']}")

    print("\nProgress:")
    progress = status['progress']
    print(f"  Total: {progress['total']}")
    print(f"  Completed: {progress['completed']}")
    print(f"  Failed: {progress['failed']}")
    print(f"  In Progress: {progress['in_progress']}")
    print(f"  Pending: {progress['pending']}")
    print(f"  Completion: {progress['completion_percentage']:.1f}%")

    print("\nRegistered Tools:")
    for tool in status['tools']:
        print(f"  - {tool}")

    print("\nSub-Agents:")
    for agent in status['sub_agents']:
        print(f"  - {agent}")


async def run_all_examples():
    """Run all examples sequentially"""
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Config Presets", example_with_config_preset),
        ("Custom Workflow", example_custom_workflow),
        ("Checkpoint Recovery", example_checkpoint_recovery),
        ("Progress Monitoring", example_monitoring_progress),
        ("Custom Tools", example_custom_tools),
        ("Error Handling", example_error_handling),
        ("Agent Status", example_get_agent_status),
    ]

    for name, example_func in examples:
        print(f"\n{'='*60}")
        print(f"Running Example: {name}")
        print('='*60)

        try:
            await example_func()
            print(f"\n✓ {name} completed successfully")
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")

        print("\nPress Enter to continue...")
        input()


def main():
    """Main entry point for examples"""
    import sys

    if len(sys.argv) > 1:
        example_name = sys.argv[1]

        examples = {
            "basic": example_basic_usage,
            "config": example_with_config_preset,
            "custom": example_custom_workflow,
            "checkpoint": example_checkpoint_recovery,
            "monitor": example_monitoring_progress,
            "tools": example_custom_tools,
            "errors": example_error_handling,
            "status": example_get_agent_status,
        }

        if example_name in examples:
            print(f"Running example: {example_name}")
            asyncio.run(examples[example_name]())
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available examples: {', '.join(examples.keys())}")
    else:
        print("Database Engine Agent - Usage Examples")
        print("=" * 60)
        print("\nUsage:")
        print("  python examples.py [example_name]")
        print("\nAvailable examples:")
        print("  basic      - Basic usage")
        print("  config     - Using configuration presets")
        print("  custom     - Custom workflow creation")
        print("  checkpoint - Checkpoint and recovery")
        print("  monitor    - Progress monitoring")
        print("  tools      - Custom tools")
        print("  errors     - Error handling")
        print("  status     - Agent status")
        print("\nOr run without arguments to run all examples interactively")

        print("\nRun all examples? (y/n): ", end="")
        if input().lower() == 'y':
            asyncio.run(run_all_examples())


if __name__ == "__main__":
    main()
