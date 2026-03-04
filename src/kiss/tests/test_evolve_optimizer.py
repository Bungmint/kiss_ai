"""Tests for the coevolving coding optimizer."""

from __future__ import annotations

import json
import os
import tempfile
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

    def test_resolve_stop_condition_requires_exactly_one_limit(self) -> None:
        with self.assertRaises(ValueError):
            eo.resolve_stop_condition(None, None)
        with self.assertRaises(ValueError):
            eo.resolve_stop_condition(10, 10)
        with self.assertRaises(ValueError):
            eo.resolve_stop_condition(0, None)
        with self.assertRaises(ValueError):
            eo.resolve_stop_condition(None, -1)

        self.assertEqual(
            eo.resolve_stop_condition(7, None),
            (eo.STOP_MODE_GPT_CALLS, 7),
        )
        self.assertEqual(
            eo.resolve_stop_condition(None, 5),
            (eo.STOP_MODE_EVALS, 5),
        )

    def test_get_stop_reason_by_mode(self) -> None:
        self.assertIsNone(
            eo.get_stop_reason(
                eo.STOP_MODE_GPT_CALLS,
                3,
                total_gpt_calls=2,
                total_eval_calls=100,
            )
        )
        self.assertIn(
            eo.STOP_MODE_GPT_CALLS,
            eo.get_stop_reason(
                eo.STOP_MODE_GPT_CALLS,
                3,
                total_gpt_calls=3,
                total_eval_calls=0,
            )
            or "",
        )
        self.assertIn(
            eo.STOP_MODE_EVALS,
            eo.get_stop_reason(
                eo.STOP_MODE_EVALS,
                4,
                total_gpt_calls=0,
                total_eval_calls=4,
            )
            or "",
        )

    def test_evolve_stops_on_max_evals(self) -> None:
        task_dir = self._create_task_dir("task_max_evals")
        with (
            patch.object(
                eo,
                "run_eval",
                side_effect=[(0.0, "baseline"), (0.1, "coder"), (0.2, "monitor")],
            ),
            patch.object(
                eo,
                "run_agent_turn",
                side_effect=[(True, "coder ok", 0.3, 2), (True, "monitor ok", 0.2, 3)],
            ),
            patch.object(eo, "repo_fingerprint", side_effect=["same", "same"]),
        ):
            result = eo.evolve(
                task_dir=task_dir,
                model_name="dummy-model",
                max_gpt_calls=None,
                max_evals=3,
                max_steps_per_session=5,
                max_sub_sessions=1,
                eval_timeout=1.0,
            )

        self.assertIn("max_evals reached", result["stop_reason"])
        self.assertEqual(result["stop_mode"], eo.STOP_MODE_EVALS)
        self.assertEqual(result["stop_limit"], 3)
        self.assertEqual(result["total_eval_calls"], 3)
        self.assertEqual(result["total_gpt_calls"], 5)

        run_metadata = yaml.safe_load(Path(result["run_metadata_path"]).read_text(encoding="utf-8"))
        self.assertEqual(run_metadata["max_evals"], 3)
        self.assertIsNone(run_metadata["max_gpt_calls"])

        entries = [
            json.loads(line)
            for line in Path(result["log_path"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(entries[-1]["agent"], "system")
        self.assertEqual(entries[-1]["action"], "stop")
        self.assertIn("total_eval_calls=3", entries[-1]["summary"])

    def test_evolve_stops_on_max_gpt_calls(self) -> None:
        task_dir = self._create_task_dir("task_max_gpt")
        with (
            patch.object(
                eo,
                "run_eval",
                side_effect=[(0.0, "baseline"), (0.1, "coder"), (0.2, "monitor")],
            ),
            patch.object(
                eo,
                "run_agent_turn",
                side_effect=[(True, "coder ok", 0.3, 3), (True, "monitor ok", 0.2, 1)],
            ),
            patch.object(eo, "repo_fingerprint", side_effect=["same", "same"]),
        ):
            result = eo.evolve(
                task_dir=task_dir,
                model_name="dummy-model",
                max_gpt_calls=4,
                max_evals=None,
                max_steps_per_session=5,
                max_sub_sessions=1,
                eval_timeout=1.0,
            )

        self.assertIn("max_gpt_calls reached", result["stop_reason"])
        self.assertEqual(result["stop_mode"], eo.STOP_MODE_GPT_CALLS)
        self.assertEqual(result["stop_limit"], 4)
        self.assertEqual(result["total_gpt_calls"], 4)
        self.assertEqual(result["total_eval_calls"], 3)

    def test_evolve_max_evals_one_skips_agent_turns(self) -> None:
        task_dir = self._create_task_dir("task_eval_one")
        with (
            patch.object(eo, "run_eval", return_value=(0.0, "baseline")),
            patch.object(eo, "run_agent_turn") as mock_run_agent_turn,
        ):
            result = eo.evolve(
                task_dir=task_dir,
                model_name="dummy-model",
                max_gpt_calls=None,
                max_evals=1,
                max_steps_per_session=5,
                max_sub_sessions=1,
                eval_timeout=1.0,
            )

        mock_run_agent_turn.assert_not_called()
        self.assertEqual(result["total_eval_calls"], 1)
        self.assertEqual(result["total_gpt_calls"], 0)
        self.assertIn("max_evals reached", result["stop_reason"])

    def test_evolve_stops_on_consecutive_zero_call_failures_in_max_gpt_mode(self) -> None:
        task_dir = self._create_task_dir("task_max_gpt_failures")

        def mock_run_agent_turn(agent, *args, **kwargs):  # type: ignore[no-untyped-def]
            if agent.name == "Coder":
                return False, "coder failed before any model call", 0.0, 0
            return True, "monitor noop", 0.0, 0

        eval_calls = 0
        max_allowed_eval_calls = 2 * eo.MAX_CONSECUTIVE_FAILURES

        def mock_run_eval(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal eval_calls
            eval_calls += 1
            if eval_calls > max_allowed_eval_calls:
                raise AssertionError("evolve exceeded expected eval call bound without stopping")
            return 0.0, "loop"

        with (
            patch.object(eo, "run_eval", side_effect=mock_run_eval),
            patch.object(eo, "run_agent_turn", side_effect=mock_run_agent_turn),
            patch.object(eo, "repo_fingerprint", return_value="same"),
        ):
            result = eo.evolve(
                task_dir=task_dir,
                model_name="dummy-model",
                max_gpt_calls=100,
                max_evals=None,
                max_steps_per_session=5,
                max_sub_sessions=1,
                eval_timeout=1.0,
            )

        self.assertEqual(result["total_gpt_calls"], 0)
        self.assertEqual(result["total_eval_calls"], max_allowed_eval_calls)
        self.assertIn("max_consecutive_failures(coder) reached", result["stop_reason"])


if __name__ == "__main__":
    unittest.main()
