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
from agents.initial_agent import InitialAgent
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
    SupervisedProcessResult,
    absolute_path_without_symlink_resolution,
    accelerator_subprocess_env,
    detect_available_accelerators,
    expose_task_data,
    run_supervised_process,
    sanitized_subprocess_env,
    select_preferred_accelerator,
    task_data_files,
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
    def test_progress_lease_allows_jobs_to_outlive_stall_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            script = (
                "import pathlib,time\n"
                "p=pathlib.Path('training_progress.txt')\n"
                "for i in range(6):\n"
                " p.write_text(str(i))\n"
                " time.sleep(0.07)\n"
            )
            result = run_supervised_process(
                [sys.executable, "-c", script],
                cwd=run_dir,
                stall_seconds=0.15,
                activity_root=run_dir,
                resource_sample_seconds=0.03,
                terminate_grace_seconds=0.2,
                label="ProgressLeaseTest",
            )

            self.assertEqual(result.returncode, 0)
            self.assertFalse(result.stalled)
            self.assertGreater(result.elapsed_seconds, 0.3)
            self.assertGreater(result.progress_events, 0)

    def test_progress_lease_automatically_recycles_stalled_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            started = time.monotonic()
            result = run_supervised_process(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cwd=Path(temp_dir),
                stall_seconds=0.15,
                activity_root=Path(temp_dir),
                resource_sample_seconds=0.03,
                terminate_grace_seconds=0.2,
                label="StallTest",
            )
            elapsed = time.monotonic() - started

            self.assertTrue(result.stalled)
            self.assertEqual(result.termination_reason, "progress_stalled")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("automatically recycling", result.stderr)
            self.assertLess(elapsed, 3.0)

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

    def test_incompatible_torch_architecture_is_not_reported_as_cuda(self):
        incompatible = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="incompatible:sm_61\n",
            stderr="",
        )
        with patch(
            "runtime_utils.subprocess.run", return_value=incompatible
        ) as run, patch("runtime_utils.shutil.which", return_value="/bin/nvidia-smi"):
            available = detect_available_accelerators(sys.executable)

        self.assertEqual(available, {"cpu"})
        run.assert_called_once()

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

    def test_virtualenv_executable_symlink_is_not_resolved_to_base_python(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            link = Path(temp_dir) / "venv-python"
            link.symlink_to(sys.executable)
            preserved = absolute_path_without_symlink_resolution(link)
            self.assertEqual(preserved, link)
            self.assertNotEqual(preserved, link.resolve())

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

    def test_titan_xp_torch_pin_uses_official_cuda_126_index(self):
        agent = SetupAgent.__new__(SetupAgent)
        agent.venv_python = sys.executable
        agent.log_file = None
        stale = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Name: torch\nVersion: 2.8.0+cu128\n",
            stderr="",
        )
        installed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="installed", stderr=""
        )
        # Use an unconditional equivalent to isolate command construction from
        # the host platform running this test.
        dependency = "torch==2.8.0+cu126"
        with patch(
            "agents.setup_agent.subprocess.run",
            side_effect=[stale, installed],
        ) as run, patch.object(agent, "_verify_dependency_imports") as verify:
            agent.install_dependencies(
                [{"verified": True, "dependencies": [dependency]}]
            )

        self.assertEqual(
            run.call_args_list[1].args[0],
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu126",
                dependency,
            ],
        )
        verify.assert_called_once_with({dependency})

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

    def test_dependency_import_timeout_is_reported_as_validation_failure(self):
        agent = SetupAgent.__new__(SetupAgent)
        agent.venv_python = sys.executable
        agent.log_file = None
        timeout = subprocess.TimeoutExpired(["python", "-c", "import sklearn"], 120)
        with patch("agents.setup_agent.subprocess.run", side_effect=timeout):
            with self.assertRaisesRegex(
                RuntimeError, "import validation timed out after 120 seconds"
            ):
                agent._verify_dependency_imports({"scikit-learn==1.7.2"})

    def test_titan_xp_torch_import_check_requires_sm_61(self):
        agent = SetupAgent.__new__(SetupAgent)
        agent.venv_python = sys.executable
        agent.log_file = None
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="2.8.0+cu126\n", stderr=""
        )
        with patch(
            "agents.setup_agent.subprocess.run", return_value=completed
        ) as run:
            agent._verify_dependency_imports({"torch==2.8.0+cu126"})

        verification_code = run.call_args.args[0][2]
        self.assertIn("torch._C._cuda_getArchFlags()", verification_code)
        self.assertIn("torch.version.cuda == '12.6'", verification_code)
        self.assertIn("'sm_61' in arches", verification_code)

    def test_docker_installs_dependencies_from_requirements(self):
        dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text()

        self.assertIn("COPY requirements.txt .", dockerfile)
        self.assertIn("pip install --no-cache-dir -r requirements.txt", dockerfile)
        self.assertNotIn("torch", dockerfile)
        self.assertNotIn("_cuda_getArchFlags", dockerfile)
        self.assertLess(
            dockerfile.index("COPY . ."),
            dockerfile.index("RUN chmod +x run_all_tasks.sh"),
        )

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

    def test_builder_rejects_artifact_filesystem_and_process_side_effects(self):
        code = """
import os
import subprocess
open('/tmp/null', 'w').close()
os.devnull = '/tmp/null'
def run(X_train, X_test):
    return X_test
"""
        issues = L2Builder._artifact_code_issues(code)
        self.assertTrue(any("'open'" in issue for issue in issues))
        self.assertTrue(any("subprocess" in issue for issue in issues))
        self.assertIn("must not monkey-patch os.devnull", issues)

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

    def test_verifier_allows_standard_devnull_at_module_import(self):
        verifier = (
            Path(__file__).resolve().parents[1]
            / "memory_pool"
            / "builder"
            / "sandbox_verifier.py"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            (artifact_dir / "devnull_model.py").write_text(
                "import os\n"
                "with open(os.devnull, 'w') as sink:\n"
                "    sink.write('')\n"
                "def run(X_train, X_test):\n"
                "    return X_test['num1'].rank(pct=True).to_numpy()\n"
            )
            card_file = artifact_dir / "devnull_model.json"
            card_file.write_text(
                json.dumps(
                    {
                        "artifact_id": "devnull_model",
                        "category": "custom_models",
                        "interface": {
                            "entrypoint": "run(X_train, X_test)",
                            "output_contract": {
                                "kind": "predictions",
                                "aligned_to": "X_test",
                                "value_type": "continuous",
                            },
                        },
                        "dependencies": [],
                        "verified": False,
                        "code_path": "devnull_model.py",
                    }
                )
            )
            result = subprocess.run(
                [sys.executable, str(verifier), str(card_file)],
                cwd=artifact_dir,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(
                result.returncode, 0, result.stdout + result.stderr
            )

    def test_verifier_uses_declared_modality_and_file_contracts(self):
        verifier = (
            Path(__file__).resolve().parents[1]
            / "memory_pool"
            / "builder"
            / "sandbox_verifier.py"
        )
        cases = [
            {
                "artifact_id": "text_model",
                "modality": "text",
                "input_contract": {
                    "modality": "text",
                    "container": "list",
                    "parameter_roles": {
                        "train_texts": "train",
                        "test_texts": "test",
                        "labels": "target",
                    },
                },
                "entrypoint": "run(train_texts, test_texts, labels)",
                "aligned_to": "test_texts",
                "dependencies": [],
                "code": (
                    "def run(train_texts, test_texts, labels):\n"
                    "    assert isinstance(train_texts[0], str)\n"
                    "    assert len(labels) == len(train_texts)\n"
                    "    return [len(text) / 100 for text in test_texts]\n"
                ),
            },
            {
                "artifact_id": "image_model",
                "modality": "image",
                "input_contract": {
                    "modality": "image",
                    "container": "numpy",
                    "sample_shape": [8, 8, 3],
                    "parameter_roles": {
                        "train_images": "train",
                        "test_images": "test",
                        "labels": "target",
                    },
                },
                "entrypoint": "run(train_images, test_images, labels)",
                "aligned_to": "test_images",
                "dependencies": ["numpy"],
                "code": (
                    "import numpy as np\n"
                    "def run(train_images, test_images, labels):\n"
                    "    assert train_images.ndim == 4\n"
                    "    assert train_images.shape[1:] == (8, 8, 3)\n"
                    "    return np.asarray(test_images).mean(axis=(1, 2, 3))\n"
                ),
            },
            {
                "artifact_id": "text_file_model",
                "modality": "text",
                "input_contract": {
                    "modality": "text",
                    "container": "paths",
                    "file_extension": ".txt",
                    "parameter_roles": {
                        "train_files": "train",
                        "test_files": "test",
                        "labels": "target",
                    },
                },
                "entrypoint": "run(train_files, test_files, labels)",
                "aligned_to": "test_files",
                "dependencies": [],
                "code": (
                    "from pathlib import Path\n"
                    "def run(train_files, test_files, labels):\n"
                    "    assert all(Path(path).suffix == '.txt' for path in train_files)\n"
                    "    return [len(Path(path).read_text()) / 100 for path in test_files]\n"
                ),
            },
        ]
        for case in cases:
            with self.subTest(case=case["artifact_id"]):
                with tempfile.TemporaryDirectory() as temp_dir:
                    artifact_dir = Path(temp_dir)
                    artifact_id = case["artifact_id"]
                    (artifact_dir / f"{artifact_id}.py").write_text(
                        case["code"]
                    )
                    card_file = artifact_dir / f"{artifact_id}.json"
                    card_file.write_text(
                        json.dumps(
                            {
                                "artifact_id": artifact_id,
                                "category": "custom_models",
                                "interface": {
                                    "entrypoint": case["entrypoint"],
                                    "input_contract": case["input_contract"],
                                    "output_contract": {
                                        "kind": "predictions",
                                        "aligned_to": case["aligned_to"],
                                        "value_type": "continuous",
                                    },
                                },
                                "capabilities": {
                                    "input_types": [case["modality"]],
                                    "target_types": [
                                        "binary_classification"
                                    ],
                                },
                                "dependencies": case["dependencies"],
                                "verified": False,
                                "code_path": f"{artifact_id}.py",
                            }
                        )
                    )
                    result = subprocess.run(
                        [sys.executable, str(verifier), str(card_file)],
                        cwd=artifact_dir,
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    self.assertEqual(
                        result.returncode,
                        0,
                        result.stdout + result.stderr,
                    )
                    verified = json.loads(card_file.read_text())
                    self.assertEqual(
                        verified["verification_level"],
                        f"{case['modality']}-contract-synthetic-data",
                    )
                    self.assertEqual(
                        verified["verification_contract_source"],
                        "declared",
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

    def test_task_input_files_are_linked_into_run_owned_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            source_input = task_dir / "input"
            run_dir = root / "run"
            source_input.mkdir(parents=True)
            (source_input / "train.csv").write_text("feature,target\n1,0\n")

            expose_task_data(task_dir, run_dir)

            run_input = run_dir / "input"
            self.assertTrue(run_input.is_dir())
            self.assertFalse(run_input.is_symlink())
            linked_train = run_input / "train.csv"
            self.assertTrue(linked_train.is_symlink())
            self.assertEqual(
                linked_train.resolve(), (source_input / "train.csv").resolve()
            )
            (run_input / "generated_cache.json").write_text("{}")
            self.assertFalse((source_input / "generated_cache.json").exists())

    def test_generated_task_artifacts_are_never_discovered_as_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            (task_dir / "train.csv").write_text("feature,target\n1,0\n")
            (task_dir / "sample_submission.csv").write_text("id,target\n1,0\n")
            (task_dir / "submission.csv").write_text("id,target\n1,1\n")
            (task_dir / "dataset_analysis_report.txt").write_text("generated")

            discovered = {path.name for path in task_data_files(task_dir)}

            self.assertEqual(
                discovered, {"train.csv", "sample_submission.csv"}
            )

    def test_initial_generation_writes_only_to_run_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "tasks" / "example"
            baseline_dir = root / "runs" / "example" / "baseline"
            task_dir.mkdir(parents=True)
            (task_dir / "task_description.md").write_text(
                "Binary classification scored by accuracy."
            )
            (task_dir / "train.csv").write_text(
                "feature,target\n1,0\n2,1\n"
            )
            before = {
                path.name: path.read_bytes()
                for path in task_dir.iterdir()
                if path.is_file()
            }
            generated_loader = "class MyDataLoader:\n    pass\n"
            generated_algorithm = "print('baseline')\n"

            with patch(
                "agents.initial_agent.call_llm",
                side_effect=[generated_loader, generated_algorithm],
            ):
                InitialAgent().generate_initial_code(
                    task_dir, baseline_dir, temperature=0.0
                )

            after = {
                path.name: path.read_bytes()
                for path in task_dir.iterdir()
                if path.is_file()
            }
            self.assertEqual(after, before)
            self.assertEqual(
                (baseline_dir / "initial_dataloader.py").read_text(),
                generated_loader.strip(),
            )
            self.assertEqual(
                (baseline_dir / "initial_algorithm.py").read_text(),
                generated_algorithm.strip(),
            )
            self.assertTrue(
                (baseline_dir / "dataset_analysis_report.txt").is_file()
            )

    def test_initial_generation_rejects_output_inside_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "tasks" / "example"
            task_dir.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "read-only task directory"):
                InitialAgent().generate_initial_code(
                    task_dir, task_dir / "generated"
                )

    def test_unsupervised_baseline_prompt_uses_discovered_data_role(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "tasks" / "clustering"
            baseline_dir = root / "runs" / "clustering" / "baseline"
            task_dir.mkdir(parents=True)
            (task_dir / "task_description.md").write_text(
                "Completely unsupervised clustering scored by Adjusted Rand Index."
            )
            pd.DataFrame(
                {"id": range(8), "feature": range(8)}
            ).to_csv(task_dir / "data.csv", index=False)
            pd.DataFrame(
                {"Id": range(8), "Predicted": [0] * 8}
            ).to_csv(task_dir / "sample_submission.csv", index=False)
            prompts = []

            def fake_llm(system_prompt, user_prompt, **kwargs):
                prompts.append(system_prompt)
                return (
                    "class MyDataLoader:\n    pass\n"
                    if len(prompts) == 1
                    else "print('baseline')\n"
                )

            with patch("agents.initial_agent.call_llm", side_effect=fake_llm):
                InitialAgent().generate_initial_code(
                    task_dir,
                    baseline_dir,
                    temperature=0.0,
                    fidelity="screen",
                )

            self.assertIn("./input/data.csv", prompts[0])
            self.assertIn("UNSUPERVISED CLUSTERING", prompts[0])
            self.assertIn("evaluate_clustering_predictions", prompts[1])
            self.assertIn(
                "prepare_evaluation_data(train_data, 'screen')", prompts[1]
            )
            self.assertNotIn(
                "prepare_evaluation_data(train_data, 'full')", prompts[1]
            )

    def test_active_interpreter_is_default_when_no_venv_is_requested(self):
        self.assertEqual(SetupAgent().venv_python, sys.executable)
        self.assertEqual(
            ImplementationAgent().venv_python, sys.executable
        )

    def test_final_submission_is_written_only_inside_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "tasks" / "example"
            run_root = root / "runs" / "example" / "complete_system"
            submission_dir = run_root / "node_1" / "submission"
            task_dir.mkdir(parents=True)
            submission_dir.mkdir(parents=True)
            pd.DataFrame(
                {"id": [2, 1], "prediction": [0.2, 0.8]}
            ).to_csv(task_dir / "sample_submission.csv", index=False)
            pd.DataFrame(
                {"id": [1, 2], "prediction": [0.7, 0.3]}
            ).to_csv(submission_dir / "submission.csv", index=False)

            node = NodeState(
                "node_1", "root", "implementation", fidelity="full"
            )
            node.result = {"score": 0.9, "status": "completed"}
            manager = ManagerAgent.__new__(ManagerAgent)
            manager.task_dir = task_dir
            manager.run_root = run_root
            manager.all_nodes = {"node_1": node}
            manager.metric_direction = "maximize"
            manager.metric_name = "accuracy"
            manager.ensemble_top_k = 1
            manager.ensemble_strategy = "average"
            manager.aggregator_agent = AggregatorAgent()

            self.assertTrue(manager.generate_final_submission("node_1"))
            self.assertFalse((task_dir / "submission.csv").exists())
            final = pd.read_csv(run_root / "submission.csv")
            self.assertEqual(final["id"].tolist(), [2, 1])
            self.assertEqual(final["prediction"].tolist(), [0.3, 0.7])

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

    def test_stalled_baseline_invokes_bounded_debugging_then_retries(self):
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
                progress_stall_seconds = 3
                metric_direction = "maximize"
                metric_name = "roc_auc"
                baseline_fidelity = "screen"
                model_name = None

            manager = ManagerStub()
            manager.task_dir = task_dir
            executions = 0

            def fake_run(*args, **kwargs):
                nonlocal executions
                executions += 1
                stalled = executions == 1
                returncode = -15 if stalled else 0
                stderr = "baseline made no observable progress" if stalled else ""
                if not stalled:
                    (baseline_dir / "result.json").write_text(
                        '{"score": 0.75, "direction": "maximize"}'
                    )
                    (baseline_dir / "submission").mkdir(exist_ok=True)
                    (baseline_dir / "submission" / "submission.csv").write_text(
                        "id,prediction\n1,0.5\n"
                    )
                    (baseline_dir / "evaluation_manifest.json").write_text("{}")
                    (baseline_dir / "oof_predictions.csv").write_text(
                        "row_id,target,prediction\n1,0,0.1\n"
                    )
                return SupervisedProcessResult(
                    args=tuple(str(item) for item in args[0]),
                    returncode=returncode,
                    stdout="",
                    stderr=stderr,
                    elapsed_seconds=0.1,
                    stalled=stalled,
                    hard_limit_reached=False,
                    termination_reason=(
                        "progress_stalled" if stalled else None
                    ),
                    progress_events=0 if stalled else 1,
                    last_progress_source=(
                        "process_started" if stalled else "process_output"
                    ),
                    last_progress_age_seconds=0.0,
                )

            with (
                patch(
                    "eval.run_ablation.run_supervised_process",
                    side_effect=fake_run,
                ),
                patch(
                    "eval.run_ablation.InitialAgent.repair_initial_algorithm"
                ) as repair,
                patch(
                    "eval.run_ablation.validate_evaluation_outputs",
                    return_value={"cv_mean": 0.75},
                ) as validate,
            ):
                score = _run_baseline(manager, baseline_dir, max_debug_attempts=1)

            self.assertEqual(score, 0.75)
            self.assertEqual(executions, 2)
            repair.assert_called_once()
            validate.assert_called_once_with(
                baseline_dir, "screen", "roc_auc"
            )
            self.assertTrue((baseline_dir / "baseline_debug.log").is_file())
            stalled_report = json.loads(
                (
                    baseline_dir / "execution_supervision_attempt_1.json"
                ).read_text()
            )
            self.assertTrue(stalled_report["stalled"])
            self.assertIsNone(
                stalled_report["total_runtime_limit_seconds"]
            )

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

    def test_explicit_regression_overrides_low_integer_cardinality(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            (task_dir / "task_description.md").write_text(
                "Predict integer loss. Evaluation is root mean squared error."
            )
            pd.DataFrame(
                {
                    "id": range(100),
                    "feature": range(100, 200),
                    "loss": [value % 5 for value in range(100)],
                }
            ).to_csv(task_dir / "train.csv", index=False)
            pd.DataFrame(
                {"id": range(100, 120), "feature": range(200, 220)}
            ).to_csv(task_dir / "test.csv", index=False)

            report = run_dataset_analysis(task_dir)
            self.assertIn("Inferred task type: regression", report)
            self.assertNotIn("Class Counts", report)

            train_data = {
                "X": pd.DataFrame({"feature": range(100)}),
                "y": [value % 5 for value in range(100)],
                "task_type": "regression",
            }
            _, _, _, _, metadata = prepare_evaluation_data(
                train_data, "screen", output_dir=Path(temp_dir) / "evaluation"
            )
            self.assertFalse(metadata["classification"])
            self.assertEqual(metadata["cv_folds"], 2)

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

    def test_validation_guard_rejects_task_input_writes(self):
        code = """
from pathlib import Path
import pandas as pd
Path('input/cache.json').write_text('{}')
pd.DataFrame({'x': [1]}).to_csv('./input/generated.csv', index=False)
with open('/workspace/tasks/example/result.txt', 'w') as stream:
    stream.write('bad')
"""
        issues = inspect_generated_code(code)
        self.assertEqual(len(issues), 3)
        self.assertTrue(all("must not write" in issue for issue in issues))


class SchedulerTests(unittest.TestCase):
    def test_standard_ucb_constant_is_not_hand_decayed(self):
        scheduler = UCB1Scheduler(total_budget=6)
        scheduler.set_warmup_budget(2)
        self.assertEqual(scheduler.get_exploration_constant(0), scheduler.c_0)
        self.assertEqual(scheduler.get_exploration_constant(2), scheduler.c_0)
        self.assertEqual(scheduler.get_exploration_constant(6), scheduler.c_0)

    def test_initial_fanout_scales_with_budget(self):
        self.assertEqual(ManagerAgent._initial_fanout_for_budget(1), 1)
        self.assertEqual(ManagerAgent._initial_fanout_for_budget(6), 2)
        self.assertEqual(ManagerAgent._initial_fanout_for_budget(60), 3)

    def test_failed_root_can_promote_a_pre_generated_backup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ManagerAgent.__new__(ManagerAgent)
            manager.run_root = Path(temp_dir)
            manager.node_counter = 0
            manager.enable_multi_fidelity = True
            manager.initial_fanout = 1
            manager.experiments_executed = 1
            manager.scheduler = UCB1Scheduler(total_budget=3)
            manager.all_nodes = {
                "root": NodeState(
                    "root", None, "technique", executed=True
                )
            }
            manager._backup_initial_approaches = [
                {"name": "backup_a", "plan": "Try robust trees."},
                {"name": "backup_b", "plan": "Try linear features."},
            ]

            promoted = manager._promote_backup_approach("root")

            self.assertEqual(promoted, "node_1")
            self.assertTrue(
                manager.all_nodes[promoted].config["replacement_branch"]
            )
            self.assertEqual(manager.all_nodes["root"].children_ids, ["node_1"])
            self.assertEqual(len(manager._backup_initial_approaches), 1)
            self.assertEqual(manager.scheduler.warmup_budget, 2)

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

    def test_diversify_can_exclude_the_parent_artifact_from_retrieval(self):
        agent = TechniqueAgent()
        l1_index = {
            "gbdt_ensembling": {
                "description": "Tree ensemble model families."
            }
        }
        catboost = {
            "artifact_id": "catboost_10fold_ensemble",
            "category": "gbdt_ensembling",
            "description": "CatBoost ensemble",
            "interface": {},
            "capabilities": {},
            "verified": True,
            "scope": "model_family",
            "validation_summary": {"runs": 10},
        }
        fallback = {"status": "pool_miss", "artifact_id": "new_model"}
        with patch(
            "agents.technique_agent.query",
            return_value={"artifacts": [catboost]},
        ), patch(
            "agents.technique_agent.call_llm",
            return_value="gbdt_ensembling",
        ), patch.object(
            agent, "_bootstrap_from_web", return_value=fallback
        ) as bootstrap:
            result = agent.run(
                "tabular regression",
                "diversify away from the parent",
                {},
                l1_index,
                allowed_scopes={"model_family"},
                excluded_artifact_ids={"catboost_10fold_ensemble"},
            )

        self.assertEqual(result, fallback)
        bootstrap.assert_called_once()

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
            self.assertEqual(len(children), 4)
            self.assertTrue(all(child.config["lazy_proposal"] for child in children))
            self.assertTrue(all(not child.config["materialized"] for child in children))
            diversify = next(
                child for child in children if child.operator == "diversify"
            )
            self.assertEqual(
                diversify.config["excluded_artifact_ids"], ["locked_model"]
            )
            self.assertEqual(
                diversify.config["priority"],
                next(
                    child.config["priority"]
                    for child in children
                    if child.operator == "refine"
                ),
            )
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
            promotion = next(
                child for child in children if child.operator == "promote"
            )
            self.assertEqual(promotion.fidelity, "medium")
            self.assertTrue(promotion.config["preserve_parent_technique"])
            self.assertEqual(promotion.parent_id, "node_1")

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

    def test_regression_rank_average_is_replaced_by_oof_safe_raw_blend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets = [10.0, 20.0, 30.0, 40.0]
            for node_id, predictions in (
                ("one", [11.0, 19.0, 31.0, 39.0]),
                ("two", [13.0, 18.0, 29.0, 42.0]),
            ):
                submission_dir = root / node_id / "submission"
                submission_dir.mkdir(parents=True)
                pd.DataFrame(
                    {"id": [1, 2], "prediction": predictions[:2]}
                ).to_csv(submission_dir / "submission.csv", index=False)
                pd.DataFrame(
                    {
                        "row_id": range(4),
                        "target": targets,
                        "prediction": predictions,
                    }
                ).to_csv(root / node_id / "oof_predictions.csv", index=False)
            agent = AggregatorAgent()
            selected = agent.aggregate_ranked_candidates(
                root,
                [
                    {"node_id": "one", "score": 1.0},
                    {"node_id": "two", "score": 2.0},
                ],
                root / "ensemble.csv",
                maximize=False,
                top_k=2,
                strategy="rank_average",
                metric_name="rmse",
                correlation_limit=1.1,
            )
            self.assertEqual(selected, ["one", "two"])
            self.assertEqual(agent.last_ensemble_manifest["strategy"], "average")
            self.assertLessEqual(
                agent.last_ensemble_manifest["ensemble_oof_score"],
                agent.last_ensemble_manifest["best_single_oof_score"] + 1e-12,
            )
            self.assertGreater(
                pd.read_csv(root / "ensemble.csv")["prediction"].min(), 1.0
            )

    def test_missing_oof_uses_best_single_instead_of_blind_average(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for node_id, predictions in (
                ("best", [2.0, 3.0]),
                ("other", [20.0, 30.0]),
            ):
                submission_dir = root / node_id / "submission"
                submission_dir.mkdir(parents=True)
                pd.DataFrame(
                    {"id": [1, 2], "prediction": predictions}
                ).to_csv(submission_dir / "submission.csv", index=False)
            agent = AggregatorAgent()
            selected = agent.aggregate_ranked_candidates(
                root,
                [
                    {"node_id": "best", "score": 1.0},
                    {"node_id": "other", "score": 2.0},
                ],
                root / "ensemble.csv",
                maximize=False,
                top_k=2,
                metric_name="rmse",
                correlation_limit=1.1,
            )
            self.assertEqual(selected, ["best"])
            self.assertTrue(agent.last_ensemble_manifest["guardrail_applied"])
            self.assertEqual(
                pd.read_csv(root / "ensemble.csv")["prediction"].tolist(),
                [2.0, 3.0],
            )


class ImplementationExecutionTests(unittest.TestCase):
    def test_dependency_fallback_cannot_silently_import_failed_model(self):
        record = {
            "status": "dependency_fallback",
            "unavailable_artifact": {
                "artifact_id": "tabnet_sparse",
                "dependencies": ["pytorch-tabnet"],
            },
        }
        code = """
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
except Exception:
    from sklearn.linear_model import LinearRegression
"""
        issues = ImplementationAgent._dependency_fallback_issues(code, record)
        self.assertTrue(issues)
        self.assertIn("pytorch_tabnet", issues[0])

    def test_unverified_parent_artifact_is_not_inherited(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir) / "parent"
            child = Path(temp_dir) / "child"
            parent.mkdir()
            child.mkdir()
            (parent / "candidate.py").write_text("def run(): return 1\n")
            (parent / "candidate.json").write_text(
                json.dumps(
                    {
                        "artifact_id": "candidate",
                        "code_path": "candidate.py",
                        "verified": False,
                    }
                )
            )
            (parent / "notes.txt").write_text("keep")
            agent = ImplementationAgent.__new__(ImplementationAgent)
            inherited = agent._inherit_parent_workspace(parent, child)
            self.assertEqual(inherited, ["notes.txt"])
            self.assertFalse((child / "candidate.py").exists())
            self.assertFalse((child / "candidate.json").exists())

    def test_missing_engineered_columns_get_feature_parity_repair_guidance(self):
        guidance = ImplementationAgent._debug_repair_guidance(
            "",
            'KeyError: "[\'num_sum\', \'num_mean\'] not in index"',
            "",
            False,
            "cpu",
            FIDELITY_PROFILES["screen"],
        )
        self.assertIn("Feature-parity repair", guidance)
        self.assertIn("Never index raw test data", guidance)

    def test_implementation_infers_rmse_when_task_config_is_absent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            node_dir = root / "node"
            task_dir.mkdir()
            (task_dir / "initial_algorithm.py").write_text("print('baseline')\n")
            (task_dir / "task_description.md").write_text(
                "Submissions are scored on the root mean squared error.\n"
            )
            generated = """
import json
from pathlib import Path
Path('submission').mkdir(exist_ok=True)
Path('submission/submission.csv').write_text('id,prediction\\n1,0.5\\n')
json.dump({'score': 0.7, 'metric': 'rmse', 'direction': 'minimize',
           'fidelity': 'full'}, open('result.json', 'w'))
"""
            prompts = []

            def fake_llm(system, user, **kwargs):
                prompts.append(system)
                return generated

            agent = ImplementationAgent(venv_python_path=sys.executable)
            with patch("agents.implementation_agent.call_llm", side_effect=fake_llm):
                result = agent.run(
                    node_dir,
                    {},
                    task_dir,
                    timeout=3,
                    max_debug_attempts=0,
                )

            self.assertEqual(result["status"], "completed")
            result_json = json.loads((node_dir / "result.json").read_text())
            self.assertEqual(result_json["metric"], "rmse")
            self.assertIn('"metric": "rmse"', prompts[0])

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
            self.assertEqual(
                resource["execution_mode"], "renewable_progress_lease"
            )
            self.assertIsNone(resource["total_runtime_limit_seconds"])
            self.assertEqual(resource["progress_stall_seconds"], 1800.0)

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

    def test_stalled_training_is_automatically_repaired_and_retried(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            node_dir = root / "node"
            task_dir.mkdir()
            (task_dir / "initial_algorithm.py").write_text("print('baseline')\n")
            stalled_code = "import time\ntime.sleep(10)"
            repaired_code = """
import json
from pathlib import Path
Path('submission').mkdir(exist_ok=True)
Path('submission/submission.csv').write_text('id,prediction\\n1,0.5\\n')
json.dump({'score': 0.73, 'direction': 'maximize', 'fidelity': 'full'}, open('result.json', 'w'))
"""
            agent = ImplementationAgent(venv_python_path=sys.executable)
            with patch(
                "agents.implementation_agent.call_llm",
                side_effect=[stalled_code, repaired_code],
            ):
                result = agent.run(
                    node_dir,
                    {},
                    task_dir,
                    stall_seconds=0.15,
                    max_debug_attempts=1,
                )

            self.assertEqual(result["status"], "completed")
            self.assertAlmostEqual(result["score"], 0.73)
            attempt_log = (node_dir / "attempt_1.log").read_text()
            self.assertIn("termination_reason=progress_stalled", attempt_log)

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
