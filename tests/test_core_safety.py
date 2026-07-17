import os
import subprocess
import sys
import tempfile
import time
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from agents.aggregator_agent import AggregatorAgent
from agents.data_analyzer import run_dataset_analysis
from agents.implementation_agent import ImplementationAgent
from agents.llm_utils import (
    _resolve_llm_config,
    call_llm,
    get_token_usage,
    reset_token_usage,
)
from agents.manager_agent import ManagerAgent
from agents.setup_agent import SetupAgent, _validate_requirement
from agents.technique_agent import TechniqueAgent
from agents.validation_guard import inspect_generated_code
from eval.run_ablation import (
    _prepare_run_input,
    _run_baseline,
    _write_token_usage_report,
)
from memory_pool.query_tool import infer_artifact_scope, query_l1, query_l2
from memory_pool.builder.l2_builder import L2Builder
from runtime_utils import (
    accelerator_subprocess_env,
    expose_task_data,
    sanitized_subprocess_env,
    select_preferred_accelerator,
    validate_path_component,
    validate_storage_identifier,
)
from tree.node import NodeState
from tree.scheduler import UCB1Scheduler
from evaluation_contract import (
    FIDELITY_PROFILES,
    prepare_evaluation_data,
    validate_evaluation_outputs,
)
from eval.metrics import calculate_ablation_metrics


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

    def test_accelerator_selection_prefers_gpu_and_falls_back_safely(self):
        self.assertEqual(
            select_preferred_accelerator({"cpu", "cuda", "mps"}, "auto"),
            "cuda",
        )
        self.assertEqual(
            select_preferred_accelerator({"cpu", "mps"}, "cuda"), "mps"
        )
        self.assertEqual(select_preferred_accelerator({"cpu"}, "auto"), "cpu")

    def test_accelerator_state_is_refreshed_after_backend_setup(self):
        manager = ManagerAgent.__new__(ManagerAgent)
        manager.venv_path = "/selected/venv/python"
        manager.accelerator_allowlist = {"cpu", "cuda"}
        manager.accelerator_preference = "auto"
        manager.available_accelerators = {"cpu"}
        manager.preferred_accelerator = "cpu"

        with patch(
            "agents.manager_agent.detect_available_accelerators",
            return_value={"cpu", "cuda"},
        ) as detect:
            manager._refresh_accelerator_state()

        detect.assert_called_once_with("/selected/venv/python")
        self.assertEqual(manager.available_accelerators, {"cpu", "cuda"})
        self.assertEqual(manager.preferred_accelerator, "cuda")

    def test_accelerator_contract_is_passed_to_sanitized_child(self):
        env = accelerator_subprocess_env(
            "cuda", {"PATH": "/bin", "API_KEY": "secret"}
        )
        self.assertNotIn("API_KEY", env)
        self.assertEqual(env["AIBUILDAI_ACCELERATOR"], "cuda")
        self.assertEqual(env["AIBUILDAI_PREFER_GPU"], "1")
        self.assertEqual(env["AIBUILDAI_CUDA_DEVICES"], "0")

    def test_custom_openai_compatible_llm_provider_is_resolved(self):
        config = _resolve_llm_config(
            environ={
                "LLM_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "test-key",
                "LLM_BASE_URL": "https://openrouter.example/v1",
                "LLM_MODEL": "vendor/model-name",
                "LLM_DEFAULT_HEADERS_JSON": '{"HTTP-Referer": "https://example.test"}',
            }
        )
        self.assertEqual(config["provider"], "openrouter")
        self.assertEqual(config["api_key"], "test-key")
        self.assertEqual(config["base_url"], "https://openrouter.example/v1")
        self.assertEqual(config["model"], "vendor/model-name")
        self.assertEqual(
            config["default_headers"],
            {"HTTP-Referer": "https://example.test"},
        )

    def test_custom_llm_provider_requires_base_url_and_model(self):
        with self.assertRaisesRegex(ValueError, "base URL"):
            _resolve_llm_config(
                environ={
                    "LLM_PROVIDER": "my-provider",
                    "MY_PROVIDER_API_KEY": "test-key",
                    "LLM_MODEL": "model",
                }
            )
        with self.assertRaisesRegex(ValueError, "model"):
            _resolve_llm_config(
                environ={
                    "LLM_PROVIDER": "my-provider",
                    "MY_PROVIDER_API_KEY": "test-key",
                    "LLM_BASE_URL": "http://localhost:8000/v1",
                }
            )

    def test_legacy_provider_defaults_and_local_no_key_mode(self):
        nvidia = _resolve_llm_config(environ={"NVIDIA_API_KEY": "test-key"})
        self.assertEqual(nvidia["provider"], "nvidia")
        self.assertEqual(nvidia["model"], "openai/gpt-oss-120b")

        local = _resolve_llm_config(
            environ={
                "LLM_PROVIDER": "local-vllm",
                "LLM_ALLOW_NO_API_KEY": "1",
                "LLM_BASE_URL": "http://localhost:8000/v1",
                "LLM_MODEL": "local-model",
                "LLM_SEND_TEMPERATURE": "0",
            }
        )
        self.assertEqual(local["api_key"], "not-required")
        self.assertFalse(local["send_temperature"])

    def test_call_llm_uses_custom_provider_and_usage_schema(self):
        captured = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured["client"] = kwargs

                def create(**request):
                    captured["request"] = request
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content=[{"text": "custom response"}]
                                )
                            )
                        ],
                        usage={"input_tokens": 7, "output_tokens": 3},
                    )

                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=create)
                )

        reset_token_usage()
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "custom-gateway",
                "LLM_API_KEY": "test-key",
                "LLM_BASE_URL": "http://localhost:9000/v1",
                "LLM_MODEL": "custom-model",
                "LLM_SEND_TEMPERATURE": "0",
            },
            clear=True,
        ), patch.dict(
            sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}
        ):
            response = call_llm("system", "user")

        self.assertEqual(response, "custom response")
        self.assertEqual(captured["client"]["base_url"], "http://localhost:9000/v1")
        self.assertEqual(captured["request"]["model"], "custom-model")
        self.assertNotIn("temperature", captured["request"])
        self.assertEqual(get_token_usage()["input_tokens"], 7)
        self.assertEqual(get_token_usage()["output_tokens"], 3)

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

    def test_generated_dependencies_use_exact_project_allowlist_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements = Path(temp_dir) / "requirements.txt"
            requirements.write_text("pytorch-tabnet==4.1.0\nnumpy==2.0.0\n")
            agent = SetupAgent(venv_python_path=sys.executable)
            card = {
                "artifact_id": "generated_tabnet",
                "verified": False,
                "dependencies": ["pytorch-tabnet>=4"],
            }
            with patch.object(agent, "install_dependencies") as install:
                agent.install_allowlisted_dependencies([card], requirements)
            install.assert_called_once_with(
                [{"verified": True, "dependencies": ["pytorch-tabnet==4.1.0"]}]
            )

            card["dependencies"] = ["not-in-project>=1"]
            with self.assertRaises(ValueError):
                agent.install_allowlisted_dependencies([card], requirements)

    def test_dependency_version_must_satisfy_resolved_requirement(self):
        agent = SetupAgent.__new__(SetupAgent)
        agent.venv_python = sys.executable
        agent.log_file = None
        stale = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Name: torch\nVersion: 2.7.0\n",
            stderr="",
        )
        installed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="installed", stderr=""
        )
        with patch(
            "agents.setup_agent.subprocess.run", side_effect=[stale, installed]
        ) as run, patch.object(agent, "_verify_dependency_imports") as verify:
            agent.install_dependencies(
                [{"verified": True, "dependencies": ["torch==2.8.0"]}]
            )

        self.assertEqual(
            run.call_args_list[1].args[0],
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "torch==2.8.0",
            ],
        )
        verify.assert_called_once_with({"torch==2.8.0"})

    def test_satisfied_dependency_version_skips_install(self):
        agent = SetupAgent.__new__(SetupAgent)
        agent.venv_python = sys.executable
        agent.log_file = None
        current = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Name: torch\nVersion: 2.8.0\n",
            stderr="",
        )
        with patch(
            "agents.setup_agent.subprocess.run", return_value=current
        ) as run, patch.object(agent, "_verify_dependency_imports") as verify:
            agent.install_dependencies(
                [{"verified": True, "dependencies": ["torch==2.8.0"]}]
            )

        run.assert_called_once()
        verify.assert_called_once_with({"torch==2.8.0"})

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
                        "scope": "model_family",
                        "resource_profile": {
                            "accelerator": "cpu",
                            "min_ram_gb": 0,
                            "estimated_runtime_seconds": 1,
                        },
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

    def test_verifier_rejects_predictions_with_wrong_test_length(self):
        verifier = (
            Path(__file__).resolve().parents[1]
            / "memory_pool"
            / "builder"
            / "sandbox_verifier.py"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            (artifact_dir / "short_model.py").write_text(
                "def run(X_train, X_test):\n    return [0.5] * (len(X_test) // 2)\n"
            )
            card_file = artifact_dir / "short_model.json"
            card_file.write_text(
                json.dumps(
                    {
                        "artifact_id": "short_model",
                        "category": "custom_models",
                        "interface": {"entrypoint": "run(X_train, X_test)"},
                        "code_path": "short_model.py",
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
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("aligns with X_test length", result.stdout + result.stderr)

    def test_neural_verifier_exercises_mixed_and_missing_inputs(self):
        verifier = (
            Path(__file__).resolve().parents[1]
            / "memory_pool"
            / "builder"
            / "sandbox_verifier.py"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            (artifact_dir / "mixed_model.py").write_text(
                "def run(X_train, X_test, y_train):\n"
                "    assert 'cat1' in X_train.columns\n"
                "    assert X_train['num3'].isna().any()\n"
                "    assert X_train['all_missing_num'].isna().all()\n"
                "    assert set(y_train) == {'negative', 'positive'}\n"
                "    return X_test['num1'].rank(pct=True).to_numpy()\n"
            )
            card_file = artifact_dir / "mixed_model.json"
            card_file.write_text(
                json.dumps(
                    {
                        "artifact_id": "mixed_model",
                        "category": "tabular_deep_learning",
                        "interface": {
                            "entrypoint": "run(X_train, X_test, y_train)",
                            "output_contract": {
                                "kind": "predictions",
                                "aligned_to": "X_test",
                                "value_type": "probability",
                            },
                        },
                        "capabilities": {
                            "target_types": ["binary_classification"]
                        },
                        "dependencies": ["torch"],
                        "verified": False,
                        "code_path": "mixed_model.py",
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
            verified_card = json.loads(card_file.read_text())
            self.assertEqual(
                verified_card["verification_level"],
                "mixed-missing-contract-mock-data",
            )

    def test_prediction_contract_rejects_constant_artifact(self):
        verifier = (
            Path(__file__).resolve().parents[1]
            / "memory_pool"
            / "builder"
            / "sandbox_verifier.py"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            (artifact_dir / "constant_model.py").write_text(
                "def run(X_train, X_test):\n"
                "    return [[0.4, 0.6]] * len(X_test)\n"
            )
            card_file = artifact_dir / "constant_model.json"
            card_file.write_text(
                json.dumps(
                    {
                        "artifact_id": "constant_model",
                        "category": "models",
                        "interface": {
                            "entrypoint": "run(X_train, X_test)",
                            "output_contract": {
                                "kind": "predictions",
                                "aligned_to": "X_test",
                                "value_type": "probability",
                            },
                        },
                        "dependencies": [],
                        "verified": False,
                        "code_path": "constant_model.py",
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
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("constant on varied test rows", result.stdout + result.stderr)

    def test_evaluation_contract_restores_full_rows_and_recomputes_uncertainty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            X = pd.DataFrame({"feature": range(80)})
            X_val = pd.DataFrame({"feature": range(80, 100)})
            train_data = {
                "X": X,
                "y": [0, 1] * 40,
                "X_val": X_val,
                "y_val": [0, 1] * 10,
            }
            X_eval, y_eval, row_ids, fold_ids, metadata = prepare_evaluation_data(
                train_data, "full", output_dir=temp_dir
            )
            self.assertEqual(len(X_eval), 100)
            self.assertEqual(metadata["source_row_count"], 100)
            predictions = [0.1 if target == 0 else 0.9 for target in y_eval]
            pd.DataFrame(
                {
                    "row_id": row_ids,
                    "target": y_eval,
                    "prediction": predictions,
                    "fold_id": fold_ids,
                }
            ).to_csv(Path(temp_dir) / "oof_predictions.csv", index=False)
            validation = validate_evaluation_outputs(temp_dir, "full", "roc_auc")
            self.assertEqual(validation["folds"], 5)
            self.assertEqual(validation["row_count"], 100)
            self.assertEqual(len(validation["fold_scores"]), 5)

    def test_skipped_implementations_do_not_distort_ablation_denominators(self):
        metrics = calculate_ablation_metrics(
            [
                {"type": "implementation", "status": "completed", "score": 0.8},
                {"type": "implementation", "status": "failed", "score": None},
                {"type": "implementation", "status": "skipped_infeasible", "score": None},
            ],
            baseline_score=0.7,
        )
        self.assertEqual(metrics["overcome_rate"], 0.5)

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
            self.assertTrue((run_dir / "input" / "train.csv").is_symlink())
            self.assertEqual(
                (run_dir / "input" / "train.csv").resolve(),
                (task_dir / "train.csv").resolve(),
            )
            self.assertFalse((run_dir / "input" / "submission.csv").exists())

    def test_task_input_directory_is_linked_without_copying(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            source_input = task_dir / "input"
            run_dir = root / "run"
            source_input.mkdir(parents=True)
            (source_input / "train.csv").write_text("feature,target\n1,0\n")

            expose_task_data(task_dir, run_dir)

            run_input = run_dir / "input"
            self.assertTrue(run_input.is_symlink())
            self.assertEqual(run_input.resolve(), source_input.resolve())

    def test_task_data_link_failure_never_falls_back_to_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            run_dir = root / "run"
            task_dir.mkdir()
            (task_dir / "train.csv").write_text("feature,target\n1,0\n")

            with patch("runtime_utils.os.symlink", side_effect=OSError("disabled")):
                with self.assertRaisesRegex(RuntimeError, "refusing to copy"):
                    expose_task_data(task_dir, run_dir)

            self.assertFalse((run_dir / "input" / "train.csv").exists())

    def test_token_usage_report_includes_end_to_end_input_and_output_totals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_file = _write_token_usage_report(
                "example_task",
                {"input_tokens": 120, "output_tokens": 30},
                {"input_tokens": 450, "output_tokens": 90},
                runs_root=Path(temp_dir),
            )
            report = json.loads(report_file.read_text())
            self.assertEqual(
                report["baseline"],
                {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
            )
            self.assertEqual(
                report["complete_system"],
                {"input_tokens": 450, "output_tokens": 90, "total_tokens": 540},
            )
            self.assertEqual(
                report["overall"],
                {"input_tokens": 570, "output_tokens": 120, "total_tokens": 690},
            )

    def test_failed_baseline_invokes_bounded_debugging_then_retries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            baseline_dir = root / "baseline"
            task_dir.mkdir()
            (task_dir / "initial_dataloader.py").write_text(
                "class MyDataLoader:\n    pass\n"
            )
            (task_dir / "initial_algorithm.py").write_text("raise RuntimeError('bad')\n")

            class ManagerStub:
                venv_path = sys.executable
                subprocess_timeout = 3
                metric_direction = "maximize"
                metric_name = "roc_auc"
                model_name = None

            manager = ManagerStub()
            manager.task_dir = task_dir
            executions = 0

            def fake_run(*args, **kwargs):
                nonlocal executions
                executions += 1
                if executions == 1:
                    return subprocess.CompletedProcess(
                        args[0], 1, stdout="", stderr="loader lifecycle error"
                    )
                (baseline_dir / "result.json").write_text(
                    '{"score": 0.75, "direction": "maximize"}'
                )
                (baseline_dir / "submission").mkdir(exist_ok=True)
                (baseline_dir / "submission" / "submission.csv").write_text(
                    "id,prediction\n1,0.5\n"
                )
                return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

            with (
                patch("eval.run_ablation.subprocess.run", side_effect=fake_run),
                patch(
                    "eval.run_ablation.InitialAgent.repair_initial_algorithm"
                ) as repair,
            ):
                score = _run_baseline(manager, baseline_dir, max_debug_attempts=1)

            self.assertEqual(score, 0.75)
            self.assertEqual(executions, 2)
            repair.assert_called_once()
            self.assertTrue((baseline_dir / "baseline_debug.log").is_file())

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

    def test_existing_run_is_archived_before_a_fresh_search(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ManagerAgent.__new__(ManagerAgent)
            manager.run_root = Path(temp_dir) / "complete_system"
            manager.run_root.mkdir()
            (manager.run_root / "tree_state.json").write_text("stale")

            manager._prepare_run_root()

            self.assertEqual(list(manager.run_root.iterdir()), [])
            archives = list((Path(temp_dir) / "archive").iterdir())
            self.assertEqual(len(archives), 1)
            self.assertEqual(
                (archives[0] / "tree_state.json").read_text(), "stale"
            )

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

    def test_pool_summary_exposes_empirical_validation_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pool = Path(temp_dir)
            category_dir = pool / "l2_store" / "models"
            category_dir.mkdir(parents=True)
            (pool / "l1_index.json").write_text(
                json.dumps(
                    {"models": {"description": "models", "l2_pointers": ["model_a"]}}
                )
            )
            (category_dir / "model_a.json").write_text(
                json.dumps(
                    {
                        "artifact_id": "model_a",
                        "category": "models",
                        "description": "test model",
                        "interface": {"entrypoint": "run(X)"},
                        "verified": True,
                        "task_validations": [
                            {
                                "status": "completed",
                                "score": 0.8,
                                "reward": 0.2,
                                "improved_over_baseline": True,
                            },
                            {"status": "failed", "score": None},
                        ],
                    }
                )
            )
            with patch("memory_pool.query_tool.BASE_DIR", pool):
                summary = query_l1("models")["artifacts"][0]["validation_summary"]
            self.assertEqual(summary["runs"], 2)
            self.assertEqual(summary["failures"], 1)
            self.assertEqual(summary["improvement_rate"], 1.0)

    def test_validation_guard_detects_test_fitted_preprocessing(self):
        code = """
import pandas as pd
combined = pd.concat([X_train, X_test])
encoder.fit(combined)
X_test = X_test.fillna(X_test.mean())
"""
        issues = inspect_generated_code(code)
        self.assertEqual(len(issues), 2)
        self.assertTrue(any("fit" in issue for issue in issues))
        self.assertTrue(any("test-set mean" in issue for issue in issues))


class SchedulerTests(unittest.TestCase):
    def test_ucb_decay_starts_after_forced_warmup(self):
        scheduler = UCB1Scheduler(total_budget=6)
        scheduler.set_warmup_budget(2)
        self.assertEqual(scheduler.get_exploration_constant(0), scheduler.c_0)
        self.assertEqual(scheduler.get_exploration_constant(2), scheduler.c_0)
        self.assertEqual(scheduler.get_exploration_constant(6), scheduler.c_min)

    def test_initial_fanout_scales_with_budget(self):
        self.assertEqual(ManagerAgent._initial_fanout_for_budget(1), 1)
        self.assertEqual(ManagerAgent._initial_fanout_for_budget(6), 2)
        self.assertEqual(ManagerAgent._initial_fanout_for_budget(60), 3)

    def test_cv_uncertainty_reduces_scheduling_reward(self):
        manager = ManagerAgent.__new__(ManagerAgent)
        manager.baseline_score = 0.95
        manager.metric_direction = "maximize"
        manager.uncertainty_weight = 1.0
        certain = manager._score_to_reward(0.954, cv_std=0.0)
        noisy = manager._score_to_reward(0.954, cv_std=0.003)
        self.assertLess(noisy, certain)

    def test_gpu_only_artifact_is_skipped_on_cpu_capacity(self):
        manager = ManagerAgent.__new__(ManagerAgent)
        manager.available_accelerators = {"cpu"}
        manager.available_ram_gb = 16.0
        manager.subprocess_timeout = 300
        reason = manager._feasibility_reason(
            {
                "resource_profile": {
                    "accelerator": "gpu",
                    "min_ram_gb": 8,
                    "estimated_runtime_seconds": 60,
                }
            }
        )
        self.assertIn("requires a GPU", reason)

    def test_runtime_estimate_does_not_skip_unbounded_node(self):
        manager = ManagerAgent.__new__(ManagerAgent)
        manager.available_accelerators = {"cpu"}
        manager.available_ram_gb = 16.0
        manager.subprocess_timeout = 1
        reason = manager._feasibility_reason(
            {
                "resource_profile": {
                    "accelerator": "cpu",
                    "min_ram_gb": 8,
                    "estimated_runtime_seconds": 86400,
                }
            }
        )
        self.assertIsNone(reason)

    def test_dependency_failure_preserves_branch_as_self_contained_fallback(self):
        record = {
            "status": "pool_hit",
            "artifact_id": "catboost_10fold_ensemble",
            "category": "gbdt_ensembling",
            "plan": "Use the selected CatBoost ensemble.",
            "model_card": {
                "artifact_id": "catboost_10fold_ensemble",
                "category": "gbdt_ensembling",
                "dependencies": ["catboost"],
            },
        }
        fallback = ManagerAgent._dependency_fallback_record(
            record, OSError(28, "No space left on device")
        )

        self.assertEqual(fallback["status"], "dependency_fallback")
        self.assertNotIn("model_card", fallback)
        self.assertNotIn("artifact_id", fallback)
        self.assertIn("catboost_10fold_ensemble", fallback["plan"])
        self.assertIn("No space left", fallback["unavailable_artifact"]["reason"])

    def test_web_bootstrap_failure_preserves_self_contained_branch(self):
        agent = TechniqueAgent()
        with patch(
            "agents.technique_agent.call_llm",
            side_effect=ValueError("Unexpected token from provider"),
        ), patch(
            "agents.technique_agent.search_web", return_value="synthetic results"
        ) as search:
            result = agent._bootstrap_from_web(
                "binary tabular classification",
                "train a robust neural tabular model",
            )

        self.assertEqual(result["status"], "self_contained_fallback")
        self.assertIn("train a robust neural", result["plan"])
        self.assertIn("Unexpected token", result["planning_error"])
        search.assert_called_once()

    def test_l1_prefilter_caps_categories_before_prompting(self):
        agent = TechniqueAgent(max_l1_categories=3)
        l1_index = {
            f"category_{index}": {"description": f"method {index}"}
            for index in range(10)
        }
        visible = agent._prefilter_l1(l1_index, "method 7")
        self.assertEqual(len(visible), 3)
        self.assertIn("category_7", visible)

    def test_legacy_pool_scope_is_inferred_conservatively(self):
        self.assertEqual(
            infer_artifact_scope({}, "feature_engineering_tabular"), "component"
        )
        self.assertEqual(
            infer_artifact_scope({}, "gbdt_ensembling"), "model_family"
        )
        self.assertEqual(
            infer_artifact_scope({}, "blending_stacking"), "full_pipeline"
        )

    def test_artifact_prior_uses_empirical_reward_not_only_llm_text(self):
        agent = TechniqueAgent()
        proven = {
            "artifact_id": "catboost_model",
            "category": "gbdt_ensembling",
            "description": "catboost model",
            "interface": {},
            "validation_summary": {
                "runs": 5,
                "mean_reward": 0.5,
                "improvement_rate": 1.0,
            },
        }
        untested = {
            **proven,
            "artifact_id": "untested_catboost_model",
            "validation_summary": {
                "runs": 0,
                "mean_reward": None,
                "improvement_rate": None,
            },
        }
        self.assertGreater(
            agent._artifact_prior(proven, "catboost model", total_runs=5),
            agent._artifact_prior(untested, "catboost model", total_runs=5),
        )

    def test_artifact_prior_rewards_compatible_gpu_artifact(self):
        agent = TechniqueAgent()
        candidate = {
            "artifact_id": "gpu_model",
            "category": "models",
            "description": "model",
            "interface": {},
            "validation_summary": {"runs": 0},
            "capabilities": {
                "gpu_accelerated": True,
                "supported_accelerators": ["cpu", "cuda"],
            },
        }
        cpu_only = {
            **candidate,
            "artifact_id": "cpu_model",
            "capabilities": {"gpu_accelerated": False},
        }
        self.assertGreater(
            agent._artifact_prior(
                candidate, "model", total_runs=0, preferred_accelerator="cuda"
            ),
            agent._artifact_prior(
                cpu_only, "model", total_runs=0, preferred_accelerator="cuda"
            ),
        )

    def test_follow_up_slots_are_created_without_eager_llm_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ManagerAgent.__new__(ManagerAgent)
            manager.run_root = Path(temp_dir)
            manager.task_name = "example"
            manager.metric_direction = "maximize"
            manager.baseline_score = 0.5
            manager.total_budget = 6
            manager.experiments_executed = 1
            manager.initial_fanout = 2
            manager.available_accelerators = {"cpu"}
            manager.available_ram_gb = 8.0
            manager.enable_multi_fidelity = True
            manager.node_counter = 1
            manager.technique_agent = TechniqueAgent()
            parent_code = Path(temp_dir) / "algorithm.py"
            parent_code.write_text("print('parent')\n")
            root = NodeState("root", None, "technique", executed=True)
            parent = NodeState(
                "node_1",
                "root",
                "implementation",
                code=str(parent_code),
                result={"score": 0.7, "reward": 0.2},
                executed=True,
                fidelity="screen",
                config={
                    "technique_record": {
                        "model_card": {
                            "artifact_id": "locked_model",
                            "capabilities": {
                                "tunable_parameters": ["epochs", "lr"]
                            },
                        }
                    }
                },
            )
            root.children_ids = ["node_1"]
            manager.all_nodes = {"root": root, "node_1": parent}

            with patch.object(
                manager.technique_agent, "generate_follow_up_approaches"
            ) as eager_generation:
                manager._spawn_follow_up_nodes(parent, "node_1")

            eager_generation.assert_not_called()
            children = [manager.all_nodes[node_id] for node_id in parent.children_ids]
            self.assertEqual(len(children), 3)
            self.assertTrue(all(child.config["lazy_proposal"] for child in children))
            self.assertTrue(all(not child.config["materialized"] for child in children))
            tune = next(child for child in children if child.operator == "tune")
            self.assertEqual(tune.fidelity, "screen")
            self.assertTrue(tune.config["priority_locked"])
            self.assertEqual(
                tune.config["tuning_context"]["tunable_parameters"],
                ["epochs", "lr"],
            )
            self.assertEqual(
                tune.config["locked_technique_record"]["model_card"]["artifact_id"],
                "locked_model",
            )

    def test_fine_tuning_is_metric_aware_uncertainty_gated_and_depth_bounded(self):
        cases = (
            ("maximize", 0.5, 0.49, 0.0, 0, None, False),
            ("maximize", 0.5, 0.70, 0.21, 0, None, False),
            ("minimize", 0.5, 0.40, 0.02, 0, None, True),
            ("maximize", 0.5, 0.70, 0.0, 2, None, False),
            ("maximize", 0.5, 0.70, 0.0, 0, [], False),
        )
        for direction, baseline, score, cv_std, depth, tunables, expected in cases:
            with self.subTest(
                direction=direction,
                score=score,
                cv_std=cv_std,
                depth=depth,
                tunables=tunables,
            ), tempfile.TemporaryDirectory() as temp_dir:
                manager = ManagerAgent.__new__(ManagerAgent)
                manager.run_root = Path(temp_dir)
                manager.task_name = "example"
                manager.metric_direction = direction
                manager.baseline_score = baseline
                manager.uncertainty_weight = 1.0
                manager.total_budget = 6
                manager.experiments_executed = 1
                manager.initial_fanout = 2
                manager.available_accelerators = {"cpu"}
                manager.available_ram_gb = 8.0
                manager.enable_multi_fidelity = True
                manager.max_fine_tune_rounds = 2
                manager.node_counter = 1
                parent_code = Path(temp_dir) / "algorithm.py"
                parent_code.write_text("print('parent')\n")
                root = NodeState("root", None, "technique", executed=True)
                parent_config = {"fine_tune_depth": depth}
                if tunables is not None:
                    parent_config["technique_record"] = {
                        "model_card": {
                            "capabilities": {"tunable_parameters": tunables}
                        }
                    }
                parent = NodeState(
                    "node_1",
                    "root",
                    "implementation",
                    code=str(parent_code),
                    result={
                        "score": score,
                        "reward": 0.1,
                        "validation": {"cv_std": cv_std},
                    },
                    executed=True,
                    fidelity="screen",
                    config=parent_config,
                )
                root.children_ids = ["node_1"]
                manager.all_nodes = {"root": root, "node_1": parent}
                manager._spawn_follow_up_nodes(parent, "node_1")
                has_tune = any(
                    manager.all_nodes[node_id].operator == "tune"
                    for node_id in parent.children_ids
                )
                self.assertEqual(has_tune, expected)

    def test_tuning_metadata_and_literal_resource_caps_are_enforced(self):
        profile = FIDELITY_PROFILES["screen"]
        params, trials = ImplementationAgent._validate_tuning_metadata(
            {
                "hyperparameters": {
                    "model_kwargs": {"epochs": 8, "patience": 2},
                    "lr": 0.001,
                },
                "tuning_trials": 4,
            },
            profile,
        )
        self.assertEqual(params["model_kwargs"]["epochs"], 8)
        self.assertEqual(trials, 4)
        for value in (9, 8.9):
            with self.assertRaises(ValueError):
                ImplementationAgent._validate_tuning_metadata(
                    {
                        "hyperparameters": {"model_kwargs": {"epochs": value}},
                        "tuning_trials": 1,
                    },
                    profile,
                )
        with self.assertRaises(ValueError):
            ImplementationAgent._validate_tuning_metadata(
                {"hyperparameters": {"epochs": 4}, "tuning_trials": 1.9},
                profile,
            )
        with self.assertRaises(ValueError):
            ImplementationAgent._validate_tuning_metadata(
                {
                    "hyperparameters": {"epochs": 4, "new_model": 1},
                    "tuning_trials": 1,
                },
                profile,
                ["epochs"],
            )
        with self.assertRaises(ValueError):
            ImplementationAgent._validate_tuning_metadata(
                {"hyperparameters": {"epochs": 4}, "tuning_trials": 1},
                profile,
                [],
            )
        self.assertTrue(
            ImplementationAgent._resource_limit_issues(
                "params = {'epochs': [4, 12]}", profile
            )
        )

    def test_locked_artifact_must_be_imported_and_called(self):
        card = {
            "code_path": "locked_model.py",
            "interface": {"entrypoint": "train_predict(X_train, X_test)"},
        }
        self.assertTrue(
            ImplementationAgent._uses_locked_artifact(
                "from locked_model import train_predict\n"
                "import pandas as pd\n"
                "p = train_predict(X, T)\n"
                "pd.DataFrame({'prediction': p}).to_csv('oof_predictions.csv')",
                card,
            )
        )
        self.assertFalse(
            ImplementationAgent._uses_locked_artifact(
                "from locked_model import train_predict\nprint('unused')", card
            )
        )
        self.assertFalse(
            ImplementationAgent._uses_locked_artifact(
                "from locked_model import train_predict\n"
                "p = train_predict(X, T)\n"
                "print(p)",
                card,
            )
        )
        self.assertFalse(
            ImplementationAgent._uses_locked_artifact(
                "from locked_model import train_predict\nif False:\n    p = train_predict(X, T)\n    print(p)",
                card,
            )
        )

    def test_local_artifact_variant_is_not_credited_to_global_card(self):
        record = {
            "status": "pool_hit",
            "artifact_id": "global_model",
            "category": "models",
        }
        variant = {
            "artifact_id": "global_model",
            "variant_id": "global_model@abc123",
            "verified": True,
        }
        self.assertEqual(
            ManagerAgent._artifact_validation_source(record, variant), {}
        )
        self.assertIs(
            ManagerAgent._artifact_validation_source(record, None), record
        )

    def test_tuning_lock_rejects_new_model_family(self):
        parent = "from catboost import CatBoostClassifier\nmodel = CatBoostClassifier()"
        candidate = (
            parent
            + "\nfrom xgboost import XGBClassifier\n"
            + "other = XGBClassifier()\n"
        )
        issues = ImplementationAgent._tuning_lock_issues(candidate, parent, {})
        self.assertTrue(any("different model family" in issue for issue in issues))

    def test_identical_parent_and_child_outputs_are_marked_no_effect(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ManagerAgent.__new__(ManagerAgent)
            manager.run_root = Path(temp_dir)
            parent_dir = manager.run_root / "parent"
            child_dir = manager.run_root / "child"
            for directory in (parent_dir, child_dir):
                (directory / "submission").mkdir(parents=True)
                (directory / "oof_predictions.csv").write_text(
                    "row_id,target,prediction\n1,0,0.2\n"
                )
                (directory / "submission" / "submission.csv").write_text(
                    "id,target\n1,0.2\n"
                )
            node = NodeState(
                "child",
                "technique",
                "implementation",
                config={"base_node_id": "parent"},
            )
            self.assertIn(
                "byte-identical", manager._no_effect_reason(node, child_dir)
            )

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

    def test_oof_history_can_weight_ranked_candidates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for node_id, test_predictions, oof_predictions in (
                ("good", [0.1, 0.9], [0.1, 0.9]),
                ("bad", [0.9, 0.1], [0.9, 0.1]),
            ):
                submission_dir = root / node_id / "submission"
                submission_dir.mkdir(parents=True)
                pd.DataFrame(
                    {"id": [1, 2], "prediction": test_predictions}
                ).to_csv(submission_dir / "submission.csv", index=False)
                pd.DataFrame(
                    {
                        "row_id": [10, 11],
                        "target": [0, 1],
                        "prediction": oof_predictions,
                    }
                ).to_csv(root / node_id / "oof_predictions.csv", index=False)
            output = root / "ensemble.csv"
            selected = AggregatorAgent().aggregate_ranked_candidates(
                root,
                [
                    {"node_id": "good", "score": 0.9},
                    {"node_id": "bad", "score": 0.6},
                ],
                output,
                top_k=2,
                strategy="average",
                metric_name="roc_auc",
                correlation_limit=1.1,
            )
            self.assertEqual(selected, ["good", "bad"])
            result = pd.read_csv(output)
            self.assertLess(result.loc[0, "prediction"], 0.5)
            self.assertGreater(result.loc[1, "prediction"], 0.5)


class ImplementationExecutionTests(unittest.TestCase):
    def test_descendant_uses_parent_code_and_inherits_support_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            parent_dir = root / "parent"
            node_dir = root / "child"
            task_dir.mkdir()
            parent_dir.mkdir()
            (task_dir / "initial_algorithm.py").write_text("BASELINE_MARKER = True\n")
            (task_dir / "initial_dataloader.py").write_text("LOADER_MARKER = True\n")
            (task_dir / "train.csv").write_text("feature,target\n1,0\n")
            parent_code = parent_dir / "algorithm.py"
            parent_code.write_text("PARENT_MARKER = True\n")
            (parent_dir / "support.json").write_text('{"value": 1}')
            (parent_dir / "trained_model.pt").write_text("fold-trained state")
            generated = """
import json
from pathlib import Path
Path('submission').mkdir(exist_ok=True)
Path('submission/submission.csv').write_text('id,prediction\\n1,0.5\\n')
json.dump({'score': 0.7, 'direction': 'maximize', 'fidelity': 'medium'}, open('result.json', 'w'))
"""
            prompts = []

            def fake_llm(system, user, **kwargs):
                prompts.append(user)
                return generated

            agent = ImplementationAgent(venv_python_path=sys.executable)
            with patch("agents.implementation_agent.call_llm", side_effect=fake_llm):
                result = agent.run(
                    node_dir,
                    {},
                    task_dir,
                    timeout=3,
                    base_algorithm_path=parent_code,
                    parent_node_dir=parent_dir,
                    fidelity="medium",
                    operator="refine",
                )
            self.assertEqual(result["status"], "completed")
            self.assertIn("PARENT_MARKER", prompts[0])
            self.assertNotIn("BASELINE_MARKER", prompts[0])
            self.assertTrue((node_dir / "support.json").is_file())
            self.assertTrue((node_dir / "initial_dataloader.py").is_file())
            self.assertTrue((node_dir / "input" / "train.csv").is_symlink())
            self.assertEqual(
                (node_dir / "input" / "train.csv").resolve(),
                (task_dir / "train.csv").resolve(),
            )
            self.assertFalse((node_dir / "trained_model.pt").exists())

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

    def test_optional_direct_call_timeout_is_enforced(self):
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

    def test_implementation_execution_has_no_timeout_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            node_dir = root / "node"
            task_dir.mkdir()
            (task_dir / "initial_algorithm.py").write_text("print('baseline')\n")
            completed_code = """
import json
import time
from pathlib import Path
time.sleep(0.25)
Path('submission').mkdir(exist_ok=True)
Path('submission/submission.csv').write_text('id,prediction\\n1,0.5\\n')
json.dump({'score': 0.71, 'direction': 'maximize', 'fidelity': 'full'}, open('result.json', 'w'))
"""
            agent = ImplementationAgent(venv_python_path=sys.executable)
            started = time.monotonic()
            with patch(
                "agents.implementation_agent.call_llm", return_value=completed_code
            ):
                result = agent.run(
                    node_dir,
                    {},
                    task_dir,
                    max_debug_attempts=0,
                )
            elapsed = time.monotonic() - started

            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["timeout_kill"])
            self.assertGreaterEqual(elapsed, 0.2)
            resource = json.loads((node_dir / "execution_resource.json").read_text())
            self.assertIsNone(resource["subprocess_timeout_seconds"])

    def test_debug_repair_gets_a_fresh_execution_timeout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            node_dir = root / "node"
            task_dir.mkdir()
            (task_dir / "initial_algorithm.py").write_text("print('baseline')\n")
            sleeping_code = "import time\ntime.sleep(10)"
            repaired_code = """
import json
from pathlib import Path
Path('submission').mkdir(exist_ok=True)
Path('submission/submission.csv').write_text('id,prediction\\n1,0.5\\n')
json.dump({'score': 0.71, 'direction': 'maximize', 'fidelity': 'full'}, open('result.json', 'w'))
"""
            agent = ImplementationAgent(venv_python_path=sys.executable)
            with patch(
                "agents.implementation_agent.call_llm",
                side_effect=[sleeping_code, repaired_code],
            ):
                result = agent.run(
                    node_dir,
                    {},
                    task_dir,
                    timeout=0.15,
                    max_debug_attempts=1,
                )
            self.assertEqual(result["status"], "completed")
            self.assertAlmostEqual(result["score"], 0.71)
            self.assertTrue((node_dir / "attempt_1.log").is_file())
            self.assertFalse(result["timeout_kill"])

    def test_invalid_tuning_metadata_cannot_fall_back_to_printed_score(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            node_dir = root / "node"
            task_dir.mkdir()
            (task_dir / "initial_algorithm.py").write_text("print('baseline')\n")
            invalid_tune = """
import json
from pathlib import Path
Path('submission').mkdir(exist_ok=True)
Path('submission/submission.csv').write_text('id,prediction\\n1,0.5\\n')
print('Score: 0.91')
json.dump({'score': 0.91, 'direction': 'maximize', 'fidelity': 'full'}, open('result.json', 'w'))
"""
            agent = ImplementationAgent(venv_python_path=sys.executable)
            with patch(
                "agents.implementation_agent.call_llm", return_value=invalid_tune
            ):
                result = agent.run(
                    node_dir,
                    {},
                    task_dir,
                    timeout=2,
                    operator="tune",
                    tuning_context={"parent_score": 0.8},
                    max_debug_attempts=0,
                )
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["score"])
            self.assertFalse((node_dir / "fine_tuning.json").exists())

    def test_successful_tuning_persists_selected_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            node_dir = root / "node"
            task_dir.mkdir()
            (task_dir / "initial_algorithm.py").write_text("print('baseline')\n")
            tuned_code = """
import json
from pathlib import Path
Path('submission').mkdir(exist_ok=True)
Path('submission/submission.csv').write_text('id,prediction\\n1,0.5\\n')
json.dump({
    'score': 0.92,
    'direction': 'maximize',
    'fidelity': 'screen',
    'hyperparameters': {'epochs': 8, 'lr': 0.001},
    'tuning_trials': 4,
}, open('result.json', 'w'))
"""
            agent = ImplementationAgent(venv_python_path=sys.executable)
            with patch(
                "agents.implementation_agent.call_llm", return_value=tuned_code
            ):
                result = agent.run(
                    node_dir,
                    {},
                    task_dir,
                    timeout=2,
                    fidelity="screen",
                    operator="tune",
                    tuning_context={"parent_score": 0.8, "fine_tune_round": 1},
                    max_debug_attempts=0,
                )
            self.assertEqual(result["status"], "completed")
            tuning = json.loads((node_dir / "fine_tuning.json").read_text())
            self.assertEqual(tuning["hyperparameters"]["epochs"], 8)
            self.assertEqual(tuning["tuning_trials"], 4)

    def test_node_local_artifact_repair_is_reverified_before_use(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            node_dir = Path(temp_dir)
            artifact_path = node_dir / "local_neural.py"
            original = (
                "import torch\n"
                "def train_predict(X_train, X_test):\n"
                "    raise RuntimeError('broken artifact')\n"
            )
            artifact_path.write_text(original)
            card = {
                "artifact_id": "local_neural",
                "category": "tabular_deep_learning",
                "description": "test artifact",
                "interface": {
                    "entrypoint": "train_predict(X_train, X_test)",
                    "output_contract": {
                        "kind": "predictions",
                        "aligned_to": "X_test",
                        "value_type": "probability",
                    },
                },
                "dependencies": ["torch"],
                "verified": True,
                "code_path": "local_neural.py",
            }
            repaired = (
                "import torch\n"
                "def train_predict(X_train, X_test):\n"
                "    assert 'cat1' in X_train.columns\n"
                "    values = torch.tensor(X_test['num1'].to_numpy(), dtype=torch.float32)\n"
                "    return torch.sigmoid(values).numpy()\n"
            )
            agent = ImplementationAgent(venv_python_path=sys.executable)
            with patch(
                "agents.implementation_agent.call_llm", return_value=repaired
            ):
                code, verified, diagnostics = agent._repair_node_local_artifact(
                    node_dir,
                    card,
                    original,
                    "RuntimeError: dtype mismatch in local_neural.py",
                    FIDELITY_PROFILES["screen"],
                )
            self.assertTrue(verified, diagnostics)
            self.assertEqual(code.strip(), repaired.strip())
            self.assertEqual(artifact_path.read_text().strip(), repaired.strip())
            audit = json.loads((node_dir / "artifact_repair.json").read_text())
            self.assertTrue(audit["verified"])
            self.assertEqual(len(audit["code_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
