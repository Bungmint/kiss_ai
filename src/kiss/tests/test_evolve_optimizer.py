"""Tests for the coevolving coding optimizer."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import yaml

from kiss.agents.coding_agents import evolve_optimizer as eo


class TestEvolveOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)

    def tearDown(self) -> None:
        os.chdir(self.old_cwd)
        self.temp_dir.cleanup()

    def _create_task_dir(self, name: str = "task") -> Path:
        task_dir = self.root / name
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "initial_program.py").write_text(
            "def solve():\n    return 0\n",
            encoding="utf-8",
        )
        (task_dir / "evaluator.py").write_text(
            "def evaluate(path):\n"
            "    return {'combined_score': 0.0}\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    print(0.0)\n",
            encoding="utf-8",
        )
        (task_dir / "config.yaml").write_text("task: test\n", encoding="utf-8")
        (task_dir / "requirements.txt").write_text("", encoding="utf-8")
        return task_dir

    def test_run_eval_timeout(self) -> None:
        repo_dir = self.root / "repo_timeout"
        repo_dir.mkdir()
        (repo_dir / "initial_program.py").write_text(
            "def solve():\n    return 0\n",
            encoding="utf-8",
        )
        (repo_dir / "evaluator.py").write_text(
            "import time\n"
            "\n"
            "def evaluate(path):\n"
            "    time.sleep(2)\n"
            "    return {'combined_score': 0.5}\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    time.sleep(2)\n",
            encoding="utf-8",
        )
        score, msg = eo.run_eval(repo_dir, eval_timeout=0.1)
        self.assertIsNone(score)
        self.assertIn("timeout after", msg)
        self.assertIn("fallback_used=True", msg)

    def test_run_eval_fallback_reports_usage(self) -> None:
        repo_dir = self.root / "repo_fallback"
        repo_dir.mkdir()
        (repo_dir / "initial_program.py").write_text(
            "def solve():\n    return 0\n",
            encoding="utf-8",
        )
        (repo_dir / "evaluator.py").write_text(
            "def evaluate(path):\n"
            "    return {'combined_score': 1.25}\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    print('not-a-float')\n",
            encoding="utf-8",
        )
        score, msg = eo.run_eval(repo_dir, eval_timeout=1.0)
        self.assertEqual(score, 1.25)
        self.assertIn("fallback_used=True", msg)

    def test_prepare_experiment_disambiguates_same_second(self) -> None:
        task_dir = self._create_task_dir()
        with patch.object(eo, "datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 1, 1, 1, 1, 1)
            exp1, _, _ = eo.prepare_experiment(task_dir)
            exp2, _, _ = eo.prepare_experiment(task_dir)

        self.assertNotEqual(exp1, exp2)
        self.assertTrue(exp1.exists())
        self.assertTrue(exp2.exists())
        self.assertTrue(exp2.name.endswith("_01"))

    def test_evolve_uses_independent_fail_streak_and_writes_stop_log(self) -> None:
        task_dir = self._create_task_dir("task_evolve")
        with (
            patch.object(
                eo,
                "run_eval",
                side_effect=[(0.0, "baseline"), (0.0, "coder"), (0.0, "monitor")],
            ),
            patch.object(
                eo,
                "run_agent_turn",
                side_effect=[(False, "coder failed", 0.1), (False, "monitor failed", 0.1)],
            ),
            patch.object(eo, "repo_fingerprint", side_effect=["same", "same"]),
            patch.object(eo, "MAX_CONSECUTIVE_FAILURES", 2),
        ):
            result = eo.evolve(
                task_dir=task_dir,
                model_name="dummy-model",
                max_budget=0.2,
                max_time=120.0,
                max_steps_per_session=1,
                max_sub_sessions=1,
                target_score=None,
                stop_on_target_score=False,
                eval_timeout=1.0,
            )

        self.assertIn("max_budget reached", result["stop_reason"])
        self.assertTrue(Path(result["run_metadata_path"]).is_file())
        self.assertTrue(Path(result["best_repo_dir"]).is_dir())
        self.assertEqual(result["best_score"], 0.0)

        run_metadata = yaml.safe_load(Path(result["run_metadata_path"]).read_text(encoding="utf-8"))
        self.assertEqual(run_metadata["model_name"], "dummy-model")
        self.assertEqual(run_metadata["max_budget"], 0.2)
        self.assertEqual(run_metadata["eval_timeout"], 1.0)

        entries = [
            json.loads(line)
            for line in Path(result["log_path"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(entries[-1]["action"], "stop")
        self.assertEqual(entries[-1]["agent"], "system")
        self.assertIn("budget_delta", entries[-1])

    def test_get_stop_reason_target_is_milestone_when_disabled(self) -> None:
        reason = eo.get_stop_reason(
            start_time=time.time(),
            score=1.0,
            target_score=0.9,
            stop_on_target_score=False,
            budget_used=0.1,
            max_budget=1.0,
            max_time=3600.0,
        )
        self.assertIsNone(reason)

    def test_get_stop_reason_target_triggers_when_enabled(self) -> None:
        reason = eo.get_stop_reason(
            start_time=time.time(),
            score=1.0,
            target_score=0.9,
            stop_on_target_score=True,
            budget_used=0.1,
            max_budget=1.0,
            max_time=3600.0,
        )
        self.assertIsNotNone(reason)
        self.assertIn("target_score reached", str(reason))

    def test_evolve_does_not_stop_on_target_when_disabled(self) -> None:
        task_dir = self._create_task_dir("task_target_soft")
        with (
            patch.object(eo, "run_eval", side_effect=[(1.0, "baseline"), (1.0, "coder")]),
            patch.object(eo, "run_agent_turn", return_value=(True, "coder ok", 0.1)),
            patch.object(eo, "repo_fingerprint", side_effect=["same", "same"]),
        ):
            result = eo.evolve(
                task_dir=task_dir,
                model_name="dummy-model",
                max_budget=0.1,
                max_time=120.0,
                max_steps_per_session=1,
                max_sub_sessions=1,
                target_score=0.9,
                stop_on_target_score=False,
                eval_timeout=1.0,
            )

        self.assertIn("max_budget reached", result["stop_reason"])

    def test_evolve_stops_on_target_when_enabled(self) -> None:
        task_dir = self._create_task_dir("task_target_hard")
        with (
            patch.object(eo, "run_eval", return_value=(1.0, "baseline")),
            patch.object(eo, "run_agent_turn") as mock_run_agent_turn,
        ):
            result = eo.evolve(
                task_dir=task_dir,
                model_name="dummy-model",
                max_budget=1.0,
                max_time=120.0,
                max_steps_per_session=1,
                max_sub_sessions=1,
                target_score=0.9,
                stop_on_target_score=True,
                eval_timeout=1.0,
            )

        self.assertIn("target_score reached", result["stop_reason"])
        mock_run_agent_turn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
