"""
Comprehensive tests for the Database Engine Agent

Tests use real inputs and outputs - NO MOCKS
"""

import asyncio
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from agent import (
    OrchestratorAgent,
    SubAgent,
    DynamicTool,
    DynamicTodoList,
    Task,
    TaskStatus,
    TaskPriority,
    Checkpoint
)
from config import AgentConfig, get_config


class TestTask(unittest.TestCase):
    """Test Task class"""

    def test_task_creation(self):
        """Test creating a task"""
        task = Task(
            id="test_001",
            name="test_task",
            description="Test description",
            priority=TaskPriority.HIGH
        )

        self.assertEqual(task.id, "test_001")
        self.assertEqual(task.name, "test_task")
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.priority, TaskPriority.HIGH)
        self.assertEqual(task.dependencies, [])

    def test_task_serialization(self):
        """Test task to/from dict conversion"""
        task = Task(
            id="test_002",
            name="serialization_test",
            description="Test serialization",
            dependencies=["test_001"]
        )

        # To dict
        task_dict = task.to_dict()
        self.assertIsInstance(task_dict, dict)
        self.assertEqual(task_dict['id'], "test_002")
        self.assertEqual(task_dict['dependencies'], ["test_001"])

        # From dict
        restored_task = Task.from_dict(task_dict)
        self.assertEqual(restored_task.id, task.id)
        self.assertEqual(restored_task.name, task.name)
        self.assertEqual(restored_task.status, task.status)

    def test_task_with_result(self):
        """Test task with result data"""
        task = Task(
            id="test_003",
            name="result_task",
            description="Task with result"
        )

        result_data = {"key": "value", "count": 42}
        task.result = result_data
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()

        self.assertEqual(task.result, result_data)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(task.completed_at)

    def test_task_priorities(self):
        """Test task priority ordering"""
        tasks = [
            Task(id="t1", name="low", description="", priority=TaskPriority.LOW),
            Task(id="t2", name="critical", description="", priority=TaskPriority.CRITICAL),
            Task(id="t3", name="medium", description="", priority=TaskPriority.MEDIUM),
            Task(id="t4", name="high", description="", priority=TaskPriority.HIGH),
        ]

        sorted_tasks = sorted(tasks, key=lambda t: t.priority.value)

        self.assertEqual(sorted_tasks[0].priority, TaskPriority.CRITICAL)
        self.assertEqual(sorted_tasks[1].priority, TaskPriority.HIGH)
        self.assertEqual(sorted_tasks[2].priority, TaskPriority.MEDIUM)
        self.assertEqual(sorted_tasks[3].priority, TaskPriority.LOW)


class TestDynamicTodoList(unittest.TestCase):
    """Test DynamicTodoList class"""

    def setUp(self):
        """Set up test fixtures"""
        self.todo_list = DynamicTodoList()

    def test_add_task(self):
        """Test adding tasks"""
        task = Task(id="t1", name="task1", description="First task")
        self.todo_list.add_task(task)

        self.assertIn("t1", self.todo_list.tasks)
        self.assertEqual(len(self.todo_list.task_order), 1)

    def test_complete_task(self):
        """Test completing a task"""
        task = Task(id="t1", name="task1", description="Task to complete")
        self.todo_list.add_task(task)

        result_data = "Task completed successfully"
        self.todo_list.complete_task("t1", result_data)

        completed_task = self.todo_list.tasks["t1"]
        self.assertEqual(completed_task.status, TaskStatus.COMPLETED)
        self.assertEqual(completed_task.result, result_data)
        self.assertIsNotNone(completed_task.completed_at)

    def test_fail_task(self):
        """Test failing a task"""
        task = Task(id="t1", name="task1", description="Task to fail")
        self.todo_list.add_task(task)

        error_msg = "Task failed due to error"
        self.todo_list.fail_task("t1", error_msg)

        failed_task = self.todo_list.tasks["t1"]
        self.assertEqual(failed_task.status, TaskStatus.FAILED)
        self.assertEqual(failed_task.error, error_msg)

    def test_get_next_task_no_dependencies(self):
        """Test getting next task when no dependencies"""
        task1 = Task(id="t1", name="task1", description="First")
        task2 = Task(id="t2", name="task2", description="Second")

        self.todo_list.add_task(task1)
        self.todo_list.add_task(task2)

        next_task = self.todo_list.get_next_task()
        self.assertIsNotNone(next_task)
        self.assertEqual(next_task.id, "t1")

    def test_get_next_task_with_dependencies(self):
        """Test getting next task with dependencies"""
        task1 = Task(id="t1", name="task1", description="First")
        task2 = Task(id="t2", name="task2", description="Second",
                    dependencies=["t1"])

        self.todo_list.add_task(task1)
        self.todo_list.add_task(task2)

        # Should get task1 first
        next_task = self.todo_list.get_next_task()
        self.assertEqual(next_task.id, "t1")

        # Complete task1
        self.todo_list.complete_task("t1")

        # Now should get task2
        next_task = self.todo_list.get_next_task()
        self.assertEqual(next_task.id, "t2")

    def test_get_next_task_blocked_dependencies(self):
        """Test that tasks with unmet dependencies are blocked"""
        task1 = Task(id="t1", name="task1", description="First")
        task2 = Task(id="t2", name="task2", description="Second",
                    dependencies=["t1"])

        self.todo_list.add_task(task2)  # Add dependent first
        self.todo_list.add_task(task1)

        # Get first task
        next_task = self.todo_list.get_next_task()
        self.assertEqual(next_task.id, "t1")

        # Don't complete it, task2 should still be blocked
        self.todo_list.update_task("t1", status=TaskStatus.IN_PROGRESS)

        next_task = self.todo_list.get_next_task()
        self.assertIsNone(next_task)  # Nothing available

    def test_get_progress(self):
        """Test progress tracking"""
        # Add various tasks
        tasks = [
            Task(id="t1", name="task1", description="Task 1"),
            Task(id="t2", name="task2", description="Task 2"),
            Task(id="t3", name="task3", description="Task 3"),
            Task(id="t4", name="task4", description="Task 4"),
        ]

        for task in tasks:
            self.todo_list.add_task(task)

        # Complete some, fail some
        self.todo_list.complete_task("t1")
        self.todo_list.complete_task("t2")
        self.todo_list.fail_task("t3", "Error")

        progress = self.todo_list.get_progress()

        self.assertEqual(progress['total'], 4)
        self.assertEqual(progress['completed'], 2)
        self.assertEqual(progress['failed'], 1)
        self.assertEqual(progress['pending'], 1)
        self.assertEqual(progress['completion_percentage'], 50.0)

    def test_priority_ordering(self):
        """Test that higher priority tasks are selected first"""
        low = Task(id="t1", name="low", description="", priority=TaskPriority.LOW)
        high = Task(id="t2", name="high", description="", priority=TaskPriority.HIGH)
        critical = Task(id="t3", name="critical", description="",
                       priority=TaskPriority.CRITICAL)

        # Add in wrong order
        self.todo_list.add_task(low)
        self.todo_list.add_task(high)
        self.todo_list.add_task(critical)

        # Should get critical first
        next_task = self.todo_list.get_next_task()
        self.assertEqual(next_task.priority, TaskPriority.CRITICAL)

    def test_serialization(self):
        """Test serialization of todo list"""
        task = Task(id="t1", name="task1", description="Test")
        self.todo_list.add_task(task)
        self.todo_list.complete_task("t1", "done")

        # Serialize
        data = self.todo_list.to_dict()
        self.assertIn('tasks', data)
        self.assertIn('task_order', data)

        # Deserialize
        restored = DynamicTodoList.from_dict(data)
        self.assertEqual(len(restored.tasks), len(self.todo_list.tasks))
        self.assertEqual(restored.tasks["t1"].status, TaskStatus.COMPLETED)


class TestDynamicTool(IsolatedAsyncioTestCase):
    """Test DynamicTool class"""

    async def test_tool_creation(self):
        """Test creating a dynamic tool"""
        def simple_func(x: int) -> int:
            return x * 2

        tool = DynamicTool("doubler", "Double a number", simple_func)

        self.assertEqual(tool.name, "doubler")
        self.assertEqual(tool.usage_count, 0)

    async def test_tool_execution_sync(self):
        """Test executing a synchronous tool"""
        def add(a: int, b: int) -> int:
            return a + b

        tool = DynamicTool("add", "Add two numbers", add)
        result = await tool.execute(5, 3)

        self.assertEqual(result, 8)
        self.assertEqual(tool.usage_count, 1)

    async def test_tool_execution_async(self):
        """Test executing an async tool"""
        async def async_multiply(a: int, b: int) -> int:
            await asyncio.sleep(0.01)
            return a * b

        tool = DynamicTool("multiply", "Multiply two numbers", async_multiply)
        result = await tool.execute(4, 7)

        self.assertEqual(result, 28)
        self.assertEqual(tool.usage_count, 1)

    async def test_tool_usage_count(self):
        """Test that usage count increments"""
        def identity(x):
            return x

        tool = DynamicTool("identity", "Return input", identity)

        await tool.execute(1)
        await tool.execute(2)
        await tool.execute(3)

        self.assertEqual(tool.usage_count, 3)


class TestSubAgent(IsolatedAsyncioTestCase):
    """Test SubAgent class"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = SubAgent(
            name="test_agent",
            purpose="Testing purposes"
        )

    async def test_subagent_creation(self):
        """Test creating a sub-agent"""
        self.assertEqual(self.agent.name, "test_agent")
        self.assertEqual(self.agent.purpose, "Testing purposes")
        self.assertEqual(len(self.agent.execution_history), 0)

    async def test_execute_task(self):
        """Test task execution"""
        task = Task(
            id="t1",
            name="implement_test",
            description="Test implementation"
        )

        context = {}
        result = await self.agent.execute_task(task, context)

        self.assertIsNotNone(result)
        self.assertEqual(len(self.agent.execution_history), 1)
        self.assertTrue(self.agent.execution_history[0]['success'])

    async def test_execution_history(self):
        """Test execution history tracking"""
        tasks = [
            Task(id="t1", name="test_basic", description="Basic test"),
            Task(id="t2", name="validate_data", description="Validate"),
        ]

        context = {}
        for task in tasks:
            await self.agent.execute_task(task, context)

        self.assertEqual(len(self.agent.execution_history), 2)

        # Check history details
        history = self.agent.execution_history
        self.assertEqual(history[0]['task_id'], "t1")
        self.assertEqual(history[1]['task_id'], "t2")
        self.assertIn('duration', history[0])
        self.assertIn('timestamp', history[1])


class TestCheckpoint(unittest.TestCase):
    """Test Checkpoint class"""

    def test_checkpoint_creation(self):
        """Test creating a checkpoint"""
        tasks = [
            Task(id="t1", name="task1", description="First").to_dict()
        ]

        checkpoint = Checkpoint(
            timestamp=time.time(),
            tasks=tasks,
            completed_ids=["t1"],
            tools_created=["tool1", "tool2"]
        )

        self.assertEqual(len(checkpoint.tasks), 1)
        self.assertEqual(checkpoint.completed_ids, ["t1"])

    def test_checkpoint_serialization(self):
        """Test checkpoint serialization"""
        checkpoint = Checkpoint(
            timestamp=time.time(),
            tasks=[],
            completed_ids=["t1", "t2"],
            tools_created=["write_file"],
            metadata={"session": "test_session"}
        )

        # To dict
        data = checkpoint.to_dict()
        self.assertIn('timestamp', data)
        self.assertIn('metadata', data)
        self.assertEqual(data['metadata']['session'], "test_session")

        # From dict
        restored = Checkpoint.from_dict(data)
        self.assertEqual(restored.completed_ids, checkpoint.completed_ids)
        self.assertEqual(restored.metadata, checkpoint.metadata)


class TestOrchestratorAgent(IsolatedAsyncioTestCase):
    """Test OrchestratorAgent class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.orchestrator = OrchestratorAgent(
            work_dir=self.test_dir,
            checkpoint_interval=2
        )

    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_orchestrator_creation(self):
        """Test creating an orchestrator"""
        self.assertTrue(self.test_dir.exists())
        self.assertTrue(self.orchestrator.checkpoint_dir.exists())
        self.assertGreater(len(self.orchestrator.tools), 0)

    def test_initialize_workflow(self):
        """Test workflow initialization"""
        self.orchestrator.initialize_workflow()

        progress = self.orchestrator.todo_list.get_progress()
        self.assertGreater(progress['total'], 0)
        self.assertEqual(progress['completed'], 0)

    def test_register_tool(self):
        """Test registering a new tool"""
        def custom_tool(x):
            return x + 1

        self.orchestrator.register_tool("custom", "Custom tool", custom_tool)

        self.assertIn("custom", self.orchestrator.tools)
        self.assertEqual(self.orchestrator.tools["custom"].name, "custom")

    def test_create_sub_agent(self):
        """Test creating a sub-agent"""
        agent = self.orchestrator.create_sub_agent(
            "test_agent",
            "Testing",
            ["write_file", "read_file"]
        )

        self.assertIn("test_agent", self.orchestrator.sub_agents)
        self.assertEqual(len(agent.tools), 2)

    async def test_save_checkpoint(self):
        """Test saving a checkpoint"""
        self.orchestrator.initialize_workflow()
        await self.orchestrator.save_checkpoint()

        # Check checkpoint file exists
        checkpoint_files = list(self.orchestrator.checkpoint_dir.glob("*.json"))
        self.assertGreater(len(checkpoint_files), 0)

    async def test_load_checkpoint(self):
        """Test loading a checkpoint"""
        # Create and save checkpoint
        self.orchestrator.initialize_workflow()
        task_id = list(self.orchestrator.todo_list.tasks.keys())[0]
        self.orchestrator.todo_list.complete_task(task_id, "done")

        await self.orchestrator.save_checkpoint()

        # Create new orchestrator and load
        new_orchestrator = OrchestratorAgent(work_dir=self.test_dir)
        checkpoint_file = list(self.orchestrator.checkpoint_dir.glob("*.json"))[0]

        success = await new_orchestrator.load_checkpoint(checkpoint_file)

        self.assertTrue(success)
        self.assertGreater(len(new_orchestrator.todo_list.tasks), 0)

    def test_get_status(self):
        """Test getting orchestrator status"""
        self.orchestrator.initialize_workflow()
        status = self.orchestrator.get_status()

        self.assertIn('session_id', status)
        self.assertIn('progress', status)
        self.assertIn('tools', status)
        self.assertIn('sub_agents', status)

    async def test_tool_execution(self):
        """Test executing built-in tools"""
        # Test write_file
        write_tool = self.orchestrator.tools["write_file"]
        success = await write_tool.execute("test.txt", "Hello World")
        self.assertTrue(success)

        # Verify file exists
        test_file = self.test_dir / "test.txt"
        self.assertTrue(test_file.exists())

        # Test read_file
        read_tool = self.orchestrator.tools["read_file"]
        content = await read_tool.execute("test.txt")
        self.assertEqual(content, "Hello World")

    async def test_shell_tool(self):
        """Test shell execution tool"""
        shell_tool = self.orchestrator.tools["run_shell"]

        # Simple echo command
        result = await shell_tool.execute("echo 'test'")

        self.assertIn('returncode', result)
        self.assertIn('stdout', result)
        self.assertTrue(result['success'])
        self.assertIn('test', result['stdout'])

    async def test_shell_tool_failure(self):
        """Test shell tool with failing command"""
        shell_tool = self.orchestrator.tools["run_shell"]

        # Command that should fail
        result = await shell_tool.execute("false")

        self.assertFalse(result['success'])
        self.assertNotEqual(result['returncode'], 0)


class TestConfig(unittest.TestCase):
    """Test configuration module"""

    def test_default_config(self):
        """Test default configuration"""
        config = AgentConfig()

        self.assertEqual(config.checkpoint_interval, 5)
        self.assertEqual(config.max_task_retries, 3)
        self.assertEqual(config.db_script_name, "db.sh")

    def test_config_presets(self):
        """Test configuration presets"""
        fast = get_config("fast")
        balanced = get_config("balanced")
        thorough = get_config("thorough")

        # Fast should have longer checkpoint interval
        self.assertGreater(fast.checkpoint_interval, balanced.checkpoint_interval)

        # Thorough should have more retries
        self.assertGreater(thorough.max_task_retries, fast.max_task_retries)

    def test_config_serialization(self):
        """Test config to dict"""
        config = AgentConfig(work_dir=Path("/tmp/test"))
        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertIn('checkpoint_interval', config_dict)
        self.assertIn('db_script_name', config_dict)


class TestIntegration(IsolatedAsyncioTestCase):
    """Integration tests"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    async def test_simple_workflow(self):
        """Test a simple workflow execution"""
        orchestrator = OrchestratorAgent(
            work_dir=self.test_dir,
            checkpoint_interval=1
        )

        # Create simple workflow
        orchestrator.todo_list.add_task(Task(
            id="t1",
            name="implement_test",
            description="Simple test task",
            priority=TaskPriority.HIGH
        ))

        # Get and mark complete
        task = orchestrator.todo_list.get_next_task()
        self.assertIsNotNone(task)

        orchestrator.todo_list.complete_task("t1", "success")

        # Check progress
        progress = orchestrator.todo_list.get_progress()
        self.assertEqual(progress['completed'], 1)
        self.assertEqual(progress['completion_percentage'], 100.0)

    async def test_checkpoint_recovery(self):
        """Test checkpoint and recovery workflow"""
        # Create orchestrator and do some work
        orch1 = OrchestratorAgent(work_dir=self.test_dir)
        orch1.initialize_workflow()

        # Complete first task
        task = orch1.todo_list.get_next_task()
        orch1.todo_list.complete_task(task.id, "done")
        await orch1.save_checkpoint()

        # Simulate crash and recovery
        orch2 = OrchestratorAgent(work_dir=self.test_dir)
        checkpoint_file = list(orch1.checkpoint_dir.glob("*.json"))[0]
        success = await orch2.load_checkpoint(checkpoint_file)

        self.assertTrue(success)

        # Check that completed task is recorded
        progress = orch2.todo_list.get_progress()
        self.assertEqual(progress['completed'], 1)


# Run tests when module is executed
if __name__ == "__main__":
    unittest.main()
