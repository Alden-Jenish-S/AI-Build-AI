import os
import subprocess
import sys
import tempfile
import time
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from agents.aggregator_agent import AggregatorAgent
from agents.data_analyzer import run_dataset_analysis
from agents.implementation_agent import ImplementationAgent
from agents.llm_utils import call_llm
from agents.manager_agent import ManagerAgent
from agents.setup_agent import _validate_requirement
from eval.run_ablation import _prepare_run_input
from memory_pool.query_tool import query_l2
from memory_pool.builder.l2_builder import L2Builder
from runtime_utils import (
    sanitized_subprocess_env,
    validate_path_component,
    validate_storage_identifier,
)
from tree.node import NodeState
from tree.scheduler import UCB1Scheduler


class RuntimeSafetyTests(unittest.TestCase):
    def test_sensitive_environment_values_are_removed(self):
        clean = sanitized_subprocess_env(
            {
                "PATH": "/bin",
                "NVIDIA_API_KEY": "secret",
                "AWS_SESSION_TOKEN": "secret",
                "DATABASE_PASSWORD": "secret",
            }
        )
        self.assertEqual(clean, {"PATH": "/bin"})

    def test_unknown_llm_provider_is_rejected(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}, clear=True):
            with self.assertRaises(ValueError):
                call_llm("system", "user")

    def test_path_and_storage_identifiers_reject_traversal(self):
        for value in ("../outside", "a/b", ".."):
            with self.assertRaises(ValueError):
                validate_path_component(value, "task")
        with self.assertRaises(ValueError):
            validate_storage_identifier("../../outside", "artifact_id")
        with self.assertRaises(ValueError):
            query_l2("../outside", "artifact")

    def test_direct_dependency_urls_and_pip_flags_are_rejected(self):
        self.assertEqual(_validate_requirement("pandas>=2.0"), "pandas>=2.0")
        for value in ("https://example.com/pkg.whl", "pkg @ https://example.com/pkg.whl", "--index-url"):
            with self.assertRaises(ValueError):
                _validate_requirement(value)

    def test_builder_rejects_generated_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            memory_pool = project_root / "memory_pool"
            (memory_pool / "builder").mkdir(parents=True)
            (memory_pool / "l2_store").mkdir()
            (memory_pool / "l1_index.json").write_text("{}")
            response = json.dumps(
                {
                    "category": "../../outside",
                    "model_card": {"artifact_id": "safe_name"},
                    "code": "def run(): return 1",
                }
            )
            builder = L2Builder(project_root, venv_path=sys.executable)
            with patch("memory_pool.builder.l2_builder.call_llm", return_value=response):
                result = builder.build_from_source("source", "content")
            self.assertFalse(result[0])
            self.assertFalse((project_root.parent / "outside").exists())

    def test_failed_local_artifact_is_preserved_for_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            memory_pool = project_root / "memory_pool"
            builder_dir = memory_pool / "builder"
            builder_dir.mkdir(parents=True)
            (memory_pool / "l2_store").mkdir()
            (memory_pool / "l1_index.json").write_text(
                json.dumps(
                    {
                        "custom_models": {
                            "description": "Custom models",
                            "l2_pointers": [],
                        }
                    }
                )
            )
            (builder_dir / "sandbox_verifier.py").write_text(
                "raise SystemExit(1)\n"
            )
            response = json.dumps(
                {
                    "category": "custom_models",
                    "model_card": {
                        "artifact_id": "candidate_model",
                        "category": "custom_models",
                        "description": "candidate",
                        "interface": {"entrypoint": "run(X_train, X_test)"},
                        "dependencies": [],
                    },
                    "code": "def run(X_train, X_test): return X_test",
                }
            )
            node_dir = project_root / "node_1"
            builder = L2Builder(project_root, venv_path=sys.executable)
            with patch("memory_pool.builder.l2_builder.call_llm", return_value=response):
                success, category, artifact_id, _ = builder.build_from_source(
                    "source", "content", commit=False, target_dir=node_dir
                )
            self.assertFalse(success)
            self.assertEqual(category, "custom_models")
            self.assertEqual(artifact_id, "candidate_model")
            self.assertTrue((node_dir / "candidate_model.py").is_file())
            self.assertTrue((node_dir / "candidate_model.json").is_file())

    def test_verifier_can_run_directly_without_pythonpath(self):
        verifier = (
            Path(__file__).resolve().parents[1]
            / "memory_pool"
            / "builder"
            / "sandbox_verifier.py"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            (artifact_dir / "identity_model.py").write_text(
                "def run(X_train, X_test):\n    return X_test\n"
            )
            card_file = artifact_dir / "identity_model.json"
            card_file.write_text(
                json.dumps(
                    {
                        "artifact_id": "identity_model",
                        "category": "custom_models",
                        "description": "identity",
                        "interface": {"entrypoint": "run(X_train, X_test)"},
                        "dependencies": [],
                        "verified": False,
                        "code_path": "identity_model.py",
                    }
                )
            )
            result = subprocess.run(
                [sys.executable, str(verifier), str(card_file)],
                cwd=artifact_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_run_input_excludes_previous_submission(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            run_dir = root / "run"
            task_dir.mkdir()
            (task_dir / "train.csv").write_text("feature,target\n1,0\n")
            (task_dir / "submission.csv").write_text("id,prediction\n1,1\n")
            _prepare_run_input(task_dir, run_dir)
            self.assertTrue((run_dir / "input" / "train.csv").exists())
            self.assertFalse((run_dir / "input" / "submission.csv").exists())

    def test_continuous_target_is_not_reported_as_rare_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            pd.DataFrame(
                {
                    "id": range(100),
                    "feature": range(100, 200),
                    "sale_price": [value / 7 for value in range(100)],
                }
            ).to_csv(task_dir / "train.csv", index=False)
            pd.DataFrame(
                {"id": range(100, 120), "feature": range(200, 220)}
            ).to_csv(task_dir / "test.csv", index=False)
            pd.DataFrame(
                {"id": range(100, 120), "sale_price": [0.0] * 20}
            ).to_csv(task_dir / "sample_submission.csv", index=False)

            report = run_dataset_analysis(task_dir)
            self.assertIn("Inferred Target Column: 'sale_price'", report)
            self.assertIn("Inferred task type: regression", report)
            self.assertNotIn("CRITICAL INCONSISTENCY", report)

    def test_pending_nodes_are_persisted_with_explicit_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ManagerAgent.__new__(ManagerAgent)
            manager.run_root = Path(temp_dir)
            manager.task_name = "example_task"
            manager.metric_direction = "maximize"
            manager.baseline_score = 0.5
            manager.total_budget = 2
            root = NodeState(
                "root", None, "technique", plan="Root", executed=True
            )
            pending = NodeState(
                "node_8", "root", "technique", plan="Try another method"
            )
            root.children_ids = ["node_8"]
            manager.all_nodes = {"root": root, "node_8": pending}

            manager._persist_node("node_8")
            manager._persist_tree_state()

            node_state = json.loads(
                (Path(temp_dir) / "node_8" / "node_state.json").read_text()
            )
            tree_state = json.loads(
                (Path(temp_dir) / "tree_state.json").read_text()
            )
            self.assertEqual(node_state["status"], "pending")
            self.assertEqual(tree_state["nodes"]["node_8"]["status"], "pending")

    def test_task_validation_is_written_to_committed_model_card(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            category_dir = (
                project_root / "memory_pool" / "l2_store" / "custom_models"
            )
            category_dir.mkdir(parents=True)
            (project_root / "memory_pool" / "builder").mkdir()
            (project_root / "memory_pool" / "l1_index.json").write_text("{}")
            card_file = category_dir / "verified_model.json"
            card_file.write_text(
                json.dumps(
                    {
                        "artifact_id": "verified_model",
                        "category": "custom_models",
                        "verified": True,
                    }
                )
            )
            builder = L2Builder(project_root, venv_path=sys.executable)
            validation = {
                "task_name": "task_a",
                "node_id": "node_4",
                "score": 0.81,
                "status": "completed",
            }
            self.assertTrue(
                builder.record_task_validation(
                    "custom_models", "verified_model", validation
                )
            )
            saved_card = json.loads(card_file.read_text())
            self.assertEqual(saved_card["task_validations"], [validation])


class SchedulerTests(unittest.TestCase):
    def test_exhausted_tree_returns_none_instead_of_reexecuting_root(self):
        root = NodeState("root", None, "technique", executed=True)
        child = NodeState("child", "root", "implementation", executed=True)
        root.children_ids.append("child")
        nodes = {"root": root, "child": child}
        scheduler = UCB1Scheduler(total_budget=2)
        self.assertIsNone(scheduler.select_next_node("root", nodes))

    def test_shallow_pending_sibling_is_selected_before_deeper_branch(self):
        root = NodeState("root", None, "technique", executed=True)
        branch_one = NodeState("branch_one", "root", "technique", executed=True)
        branch_two = NodeState("branch_two", "root", "technique", executed=True)
        implementation_one = NodeState(
            "implementation_one", "branch_one", "implementation", executed=True
        )
        implementation_two = NodeState(
            "implementation_two", "branch_two", "implementation", executed=False
        )
        deeper = NodeState(
            "deeper", "implementation_one", "technique", executed=False
        )
        root.children_ids = ["branch_one", "branch_two"]
        branch_one.children_ids = ["implementation_one"]
        branch_two.children_ids = ["implementation_two"]
        implementation_one.children_ids = ["deeper"]
        nodes = {
            node.node_id: node
            for node in (
                root,
                branch_one,
                branch_two,
                implementation_one,
                implementation_two,
                deeper,
            )
        }
        scheduler = UCB1Scheduler(total_budget=6)
        self.assertEqual(
            scheduler.select_next_node("root", nodes), "implementation_two"
        )


class AggregatorTests(unittest.TestCase):
    def test_predictions_are_aligned_by_id_before_averaging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for node_id in ("one", "two"):
                (root / node_id / "submission").mkdir(parents=True)
            pd.DataFrame({"id": [1, 2], "a": [0.2, 0.4], "b": [2.0, 4.0]}).to_csv(
                root / "one" / "submission" / "submission.csv", index=False
            )
            pd.DataFrame({"id": [2, 1], "a": [0.6, 0.8], "b": [6.0, 8.0]}).to_csv(
                root / "two" / "submission" / "submission.csv", index=False
            )
            output = root / "ensemble.csv"
            self.assertTrue(
                AggregatorAgent().aggregate_submissions(root, ["one", "two"], output)
            )
            result = pd.read_csv(output)
            self.assertEqual(result["id"].tolist(), [1, 2])
            self.assertEqual(result["a"].tolist(), [0.5, 0.5])
            self.assertEqual(result["b"].tolist(), [5.0, 5.0])

    def test_mismatched_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for node_id, ids in (("one", [1, 2]), ("two", [1, 3])):
                path = root / node_id / "submission"
                path.mkdir(parents=True)
                pd.DataFrame({"id": ids, "prediction": [0.1, 0.2]}).to_csv(
                    path / "submission.csv", index=False
                )
            self.assertFalse(
                AggregatorAgent().aggregate_submissions(
                    root, ["one", "two"], root / "ensemble.csv"
                )
            )


class ImplementationExecutionTests(unittest.TestCase):
    def test_stale_result_is_not_accepted_after_process_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            node_dir = root / "node"
            task_dir.mkdir()
            node_dir.mkdir()
            (task_dir / "initial_algorithm.py").write_text("print('baseline')\n")
            (node_dir / "result.json").write_text('{"score": 0.99}')
            agent = ImplementationAgent(venv_python_path=sys.executable)
            failing_code = "raise SystemExit(1)"
            with patch("agents.implementation_agent.call_llm", return_value=failing_code):
                result = agent.run(node_dir, {}, task_dir, timeout=1)
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["score"])

    def test_configured_timeout_is_enforced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            node_dir = root / "node"
            task_dir.mkdir()
            (task_dir / "initial_algorithm.py").write_text("print('baseline')\n")
            agent = ImplementationAgent(venv_python_path=sys.executable)
            sleeping_code = "import time\ntime.sleep(10)"
            started = time.monotonic()
            with patch("agents.implementation_agent.call_llm", return_value=sleeping_code):
                result = agent.run(node_dir, {}, task_dir, timeout=0.15)
            elapsed = time.monotonic() - started
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["timeout_kill"])
            self.assertLess(elapsed, 3.0)


if __name__ == "__main__":
    unittest.main()
