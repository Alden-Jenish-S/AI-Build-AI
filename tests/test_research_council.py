from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.architecture_policy import classify_architecture
from agents.council.contracts import CouncilBrief, EvaluationProtocol
from agents.council.coordinator import CouncilCoordinator
from agents.council.diagnostics import (
    build_problem_fingerprint,
    classify_input_access,
    validate_diagnostic_script,
)
from agents.council.research import (
    ResearchRetriever,
    _openalex_search,
    validate_research_query,
)
from agents.implementation_agent import ImplementationAgent, _architecture_source_errors
from agents.manager_agent import ManagerAgent
from agents.modality_policy import (
    predictive_modality_inventory,
    validate_modality_ablation_report,
)
from agents.submission_validator import SubmissionValidator
from agents.task_analyzer import TaskAnalysis
from agents.technique_agent import TechniqueAgent
from runtime_utils import SupervisedProcessResult, expose_task_data
from tree.node import NodeState


class ResearchCouncilTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.task_dir = self.root / "SecretBenchmark2026"
        self.task_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def analysis(self) -> TaskAnalysis:
        return TaskAnalysis(
            task_name="SecretBenchmark2026",
            task_dir=self.task_dir,
            goal="predict a private binary outcome",
            target="diagnosis_label",
            expected_output="submission.csv",
            metric="ROC AUC",
            direction="maximize",
            files=[
                {"path": "train.csv", "kind": "table", "extension": ".csv", "bytes": 1},
                {"path": "test.csv", "kind": "table", "extension": ".csv", "bytes": 1},
                {
                    "path": "test_labels.csv",
                    "kind": "table",
                    "extension": ".csv",
                    "bytes": 1,
                },
                {
                    "path": "sample_submission.csv",
                    "kind": "table",
                    "extension": ".csv",
                    "bytes": 1,
                },
            ],
        )

    @staticmethod
    def protocol() -> EvaluationProtocol:
        return EvaluationProtocol.from_mapping(
            {
                "metric": "hallucinated metric",
                "direction": "minimize",
                "mode": "cross_validation",
                "split_strategy": "stratified group folds",
                "folds": 3,
                "seed": 17,
                "leakage_unit": "entity",
                "rationale": "Repeated entities require grouped validation.",
            },
            metric="ROC AUC",
            direction="maximize",
        )

    def brief(self) -> CouncilBrief:
        hypothesis = {
            "hypothesis_id": "H_grouped_gbdt",
            "title": "Grouped out-of-fold tree ensemble",
            "model_family": "gradient-boosted trees",
            "rationale": "Nonlinear mixed-type interactions are plausible.",
            "experiment": "Measure a grouped three-fold tree control.",
            "expected_signal": "Stable fold AUC above the linear control.",
            "estimated_cost": "moderate",
            "risks": ["small groups"],
            "stopping_rule": "Stop if two folds underperform the control.",
            "compatible_with": [],
        }
        return CouncilBrief(
            task_name="SecretBenchmark2026",
            status="completed",
            problem_fingerprint={"data_kinds": {"table": 4}},
            allowed_input_paths=("train.csv", "test.csv", "sample_submission.csv"),
            prohibited_inputs=(
                {"path": "test_labels.csv", "reason": "held-out labels"},
            ),
            evaluation_protocol=self.protocol(),
            selected_portfolio=[hypothesis],
            hypotheses=[hypothesis],
            recommended_root_count=1,
        )

    def test_answer_artifacts_are_denied_by_default(self) -> None:
        allowed, prohibited = classify_input_access(self.analysis())
        self.assertIn("train.csv", allowed)
        self.assertIn("test.csv", allowed)
        self.assertNotIn("test_labels.csv", allowed)
        self.assertEqual("test_labels.csv", prohibited[0]["path"])

    def test_allowlist_is_enforced_when_inputs_are_exposed(self) -> None:
        for name in ("train.csv", "test.csv", "test_labels.csv"):
            (self.task_dir / name).write_text("x\n1\n", encoding="utf-8")
        run_dir = self.root / "node"
        linked = expose_task_data(
            self.task_dir,
            run_dir,
            allowed_paths=("train.csv", "test.csv"),
        )
        self.assertEqual(2, len(linked))
        self.assertTrue((run_dir / "input" / "train.csv").exists())
        self.assertFalse((run_dir / "input" / "test_labels.csv").exists())

    def test_diagnostic_gate_allows_only_the_declared_output_write(self) -> None:
        safe = """
import json
from pathlib import Path
rows = Path('input/train.csv').read_text()
Path('analysis_result.json').write_text(json.dumps({'question': 'q', 'method': 'm', 'findings': [len(rows)], 'limitations': [], 'suggested_next_questions': []}))
"""
        self.assertEqual([], validate_diagnostic_script(safe))
        unsafe = """
from pathlib import Path
Path('input/train.csv').write_text('changed')
"""
        errors = validate_diagnostic_script(unsafe)
        self.assertTrue(any("analysis_result.json" in error for error in errors))

        unsafe_open = "open('input/train.csv', 'w').write('changed')"
        errors = validate_diagnostic_script(unsafe_open)
        self.assertTrue(any("write-mode open" in error for error in errors))

    def test_problem_fingerprint_removes_task_file_and_column_identity(self) -> None:
        diagnostics = {
            "tables": [
                {
                    "path": "train.csv",
                    "status": "profiled",
                    "sampled_rows": 100,
                    "column_count": 3,
                    "dtype_families": {"numeric": 2, "text_or_categorical": 1},
                    "top_missing_fraction": {"patient_id": 0.0},
                    "top_cardinalities": {"patient_id": 100},
                    "target_summary": {
                        "diagnosis_label": {
                            "observed": 100,
                            "unique": 2,
                            "missing_fraction": 0.0,
                            "top_distribution": {"0": 0.7, "1": 0.3},
                        }
                    },
                }
            ],
            "media": {},
            "resources": {"cpu_count": 8},
            "train_test_shift": {
                "available": True,
                "train_path": "train.csv",
                "test_path": "test.csv",
                "largest_standardized_mean_shifts": [
                    {"column": "patient_id", "absolute_standardized_shift": 2.0}
                ],
            },
        }
        rendered = json.dumps(build_problem_fingerprint(self.analysis(), diagnostics))
        for protected in (
            "SecretBenchmark2026",
            "train.csv",
            "test.csv",
            "patient_id",
            "diagnosis_label",
        ):
            self.assertNotIn(protected, rendered)

    def test_research_query_policy_requires_precision_and_prevents_solution_search(self) -> None:
        accepted, _ = validate_research_query(
            'site:arxiv.org "group-aware cross-validation" imbalanced binary classification calibration',
            task_name="SecretBenchmark2026",
            forbidden_terms=("patient_id", "diagnosis_label"),
        )
        self.assertTrue(accepted)

        for query in (
            "tabular binary classification ROC AUC feature interactions deep learning",
            "Kaggle winning solution for SecretBenchmark2026",
            'site:arxiv.org "patient_id" grouped validation leakage',
        ):
            accepted, _ = validate_research_query(
                query,
                task_name="SecretBenchmark2026",
                forbidden_terms=("patient_id", "diagnosis_label"),
            )
            self.assertFalse(accepted, query)

    def test_generic_task_domain_terms_are_not_mistaken_for_task_identity(self) -> None:
        accepted, reason = validate_research_query(
            'site:arxiv.org "tabular neural networks" feature interactions calibration',
            task_name="tabular-playground-series-may-2022",
        )
        self.assertTrue(accepted, reason)

        accepted, reason = validate_research_query(
            'site:arxiv.org "secretbenchmark2026" neural feature interactions calibration',
            task_name="SecretBenchmark2026",
        )
        self.assertFalse(accepted)
        self.assertIn("task-identity", reason)

    @staticmethod
    def research_requests(count: int = 6) -> list[dict[str, object]]:
        return [
            {
                "question": f"robust validation question {index}",
                "queries": [
                    f'site:arxiv.org "group-aware validation" imbalanced binary calibration {2020 + index}'
                ],
            }
            for index in range(count)
        ]

    def test_research_circuit_breaker_stops_retry_storms(self) -> None:
        retriever = ResearchRetriever(
            self.analysis().task_name,
            self.root / "circuit_council",
        )
        settings = {
            "OPENALEX_API_KEY": "",
            "AIBUILDAI_COUNCIL_MAX_QUERIES": "6",
            "AIBUILDAI_COUNCIL_SEARCH_WORKERS": "1",
            "AIBUILDAI_COUNCIL_SEARCH_FAILURE_LIMIT": "2",
            "AIBUILDAI_COUNCIL_SEARCH_DELAY_SECONDS": "0",
        }
        with (
            patch.dict(os.environ, settings),
            patch("agents.council.research.search_web", return_value="") as search,
        ):
            sources, audit = retriever.collect(self.research_requests())
        self.assertEqual([], sources)
        self.assertEqual(2, search.call_count)
        self.assertTrue(
            any("circuit breaker" in str(item.get("reason")) for item in audit)
        )
        self.assertTrue(
            all(call.kwargs.get("max_retries") == 1 for call in search.call_args_list)
        )

    def test_research_stops_after_primary_source_target(self) -> None:
        retriever = ResearchRetriever(
            self.analysis().task_name,
            self.root / "target_council",
        )
        formatted = (
            "Title: Group-aware validation for calibrated classifiers\n"
            "URL: https://arxiv.org/abs/2501.01234\n"
            "Snippet: A primary research abstract.\n"
        )
        settings = {
            "OPENALEX_API_KEY": "",
            "AIBUILDAI_COUNCIL_MAX_QUERIES": "6",
            "AIBUILDAI_COUNCIL_SEARCH_WORKERS": "1",
            "AIBUILDAI_COUNCIL_SOURCE_TARGET": "1",
            "AIBUILDAI_COUNCIL_SEARCH_DELAY_SECONDS": "0",
        }
        with (
            patch.dict(os.environ, settings),
            patch(
                "agents.council.research.search_web", return_value=formatted
            ) as search,
            patch(
                "agents.council.research._fetch_visible_text",
                return_value="paper text",
            ),
        ):
            sources, audit = retriever.collect(self.research_requests())
        self.assertEqual(1, len(sources))
        self.assertEqual(1, search.call_count)
        self.assertTrue(
            any("evidence target reached" in str(item.get("reason")) for item in audit)
        )

    def test_openalex_key_uses_structured_scholarly_search_without_leaking_key(self) -> None:
        payload = {
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "doi": "https://doi.org/10.1000/example",
                    "title": "Group-aware validation for calibrated classifiers",
                    "publication_year": 2025,
                    "publication_date": "2025-02-01",
                    "primary_location": {
                        "landing_page_url": "https://arxiv.org/abs/2501.01234"
                    },
                    "best_oa_location": None,
                    "open_access": {"is_oa": True},
                    "abstract_inverted_index": {
                        "Robust": [0],
                        "validation": [1],
                    },
                    "cited_by_count": 12,
                    "type": "article",
                    "authorships": [
                        {"author": {"display_name": "Researcher One"}}
                    ],
                }
            ]
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        captured_url = ""

        def open_url(request, **_kwargs):
            nonlocal captured_url
            captured_url = request.full_url
            return Response()

        settings = {
            "OPENALEX_API_KEY": "super-secret-openalex-key",
            "AIBUILDAI_OPENALEX_MIN_INTERVAL_SECONDS": "0",
        }
        with (
            patch.dict(os.environ, settings),
            patch("agents.council.research.urllib.request.urlopen", side_effect=open_url),
        ):
            sources = _openalex_search(
                'site:arxiv.org "group-aware validation" calibration 2025'
            )
        self.assertEqual(1, len(sources))
        self.assertEqual("openalex", sources[0]["provider"])
        self.assertEqual("Robust validation", sources[0]["retrieved_text"])
        self.assertIn("api_key=super-secret-openalex-key", captured_url)
        self.assertIn("from_publication_date%3A2025-01-01", captured_url)
        self.assertNotIn("super-secret-openalex-key", json.dumps(sources))

    def test_retriever_prefers_openalex_and_skips_web_search(self) -> None:
        retriever = ResearchRetriever(
            "SecretBenchmark2026",
            self.root / "council",
            forbidden_terms=("diagnosis_label",),
        )
        source = {
            "source_id": "oa_W123",
            "question": "",
            "query": "",
            "title": "Reliable validation under group dependence",
            "url": "https://doi.org/10.1000/example",
            "snippet": "A primary-literature abstract.",
            "retrieved_text": "A primary-literature abstract.",
            "provider": "openalex",
        }
        settings = {
            "OPENALEX_API_KEY": "super-secret-openalex-key",
            "AIBUILDAI_COUNCIL_MAX_QUERIES": "1",
            "AIBUILDAI_COUNCIL_SOURCE_TARGET": "1",
            "AIBUILDAI_COUNCIL_SEARCH_WORKERS": "1",
            "AIBUILDAI_COUNCIL_SEARCH_WAVE_DELAY_SECONDS": "0",
        }
        with (
            patch.dict(os.environ, settings),
            patch(
                "agents.council.research._openalex_search",
                return_value=[source],
            ) as openalex,
            patch("agents.council.research.search_web") as web_search,
        ):
            sources, audit = retriever.collect(self.research_requests(count=1))
        self.assertEqual(1, len(sources))
        self.assertEqual("openalex", sources[0]["provider"])
        self.assertEqual("robust validation question 0", sources[0]["question"])
        openalex.assert_called_once()
        web_search.assert_not_called()
        self.assertTrue(
            any(item.get("provider") == "openalex" for item in audit)
        )
        self.assertNotIn("super-secret-openalex-key", json.dumps(audit))

    def test_protocol_hash_is_stable_and_task_metric_is_authoritative(self) -> None:
        first = self.protocol()
        second = self.protocol()
        self.assertEqual("ROC AUC", first.metric)
        self.assertEqual("maximize", first.direction)
        self.assertEqual(first.protocol_hash, second.protocol_hash)

    def test_implementation_score_must_match_shared_protocol(self) -> None:
        protocol = self.protocol()
        valid = {
            "score": 0.81,
            "metric": "ROC AUC",
            "direction": "maximize",
            "evaluation_protocol_hash": protocol.protocol_hash,
            "fold_scores": [0.80, 0.82, 0.81],
            "validation_sample_count": 120,
        }
        self.assertEqual(
            [], ImplementationAgent._validate_reported_evaluation(valid, protocol)
        )
        invalid = {**valid, "evaluation_protocol_hash": "wrong", "fold_scores": [0.9]}
        errors = ImplementationAgent._validate_reported_evaluation(invalid, protocol)
        self.assertTrue(any("hash" in error for error in errors))
        self.assertTrue(any("exactly 3" in error for error in errors))

    def test_evaluation_contract_failure_enters_repair_loop(self) -> None:
        analysis = self.analysis()
        analysis.submission = {
            "path": "sample_submission.csv",
            "kind": "table",
            "extension": ".csv",
        }
        (self.task_dir / "train.csv").write_text(
            "feature,diagnosis_label\n1,0\n2,1\n", encoding="utf-8"
        )
        (self.task_dir / "test.csv").write_text(
            "feature\n3\n4\n", encoding="utf-8"
        )
        (self.task_dir / "test_labels.csv").write_text(
            "diagnosis_label\n0\n1\n", encoding="utf-8"
        )
        (self.task_dir / "sample_submission.csv").write_text(
            "id,prediction\na,0.5\nb,0.5\n", encoding="utf-8"
        )
        brief = self.brief()
        executions = 0

        def execute(_command, *, cwd: Path, **_kwargs):
            nonlocal executions
            executions += 1
            output_dir = Path(cwd) / "submission"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "submission.csv").write_text(
                "id,prediction\na,0.2\nb,0.8\n", encoding="utf-8"
            )
            payload = {
                "score": 0.8,
                "metric": "ROC AUC",
                "direction": "maximize",
                "evaluation_protocol_hash": (
                    "wrong" if executions == 1 else brief.evaluation_protocol.protocol_hash
                ),
                "fold_scores": [0.79, 0.80, 0.81],
                "validation_sample_count": 6,
                "output": "submission/submission.csv",
                "diagnostics": {},
            }
            (Path(cwd) / "result.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            return SupervisedProcessResult(
                args=("python", "algorithm.py"),
                returncode=0,
                stdout="score=0.8\n",
                stderr="",
                elapsed_seconds=0.01,
                stalled=False,
                hard_limit_reached=False,
                termination_reason=None,
                progress_events=1,
                last_progress_source="process_output",
                last_progress_age_seconds=0.0,
            )

        generated_code = "\n".join(
            ["from pathlib import Path", "# generated implementation"]
            + [f"VALUE_{index} = {index}" for index in range(20)]
        )
        node = self.root / "implementation"
        agent = ImplementationAgent(submission_validator=SubmissionValidator())
        with (
            patch("agents.implementation_agent.call_llm", return_value=generated_code),
            patch(
                "agents.implementation_agent.run_supervised_process",
                side_effect=execute,
            ),
        ):
            result = agent.run(
                node,
                "measure the selected hypothesis",
                self.task_dir,
                analysis,
                max_debug_attempts=2,
                stall_seconds=1,
                council_brief=brief,
            )
        self.assertEqual("completed", result["status"])
        self.assertEqual(2, result["attempts"])
        self.assertFalse((node / "input" / "test_labels.csv").exists())
        self.assertTrue((node / "evaluation_protocol.json").is_file())
        self.assertIn(
            "EVALUATION CONTRACT",
            (node / "attempt_1.log").read_text(encoding="utf-8"),
        )

    def test_initial_plans_are_the_council_portfolio_without_an_llm_call(self) -> None:
        brief = self.brief()
        plans = TechniqueAgent(council_brief=brief).generate_initial_approaches(
            self.analysis(), count=3
        )
        self.assertEqual(1, len(plans))
        self.assertIn("Hypothesis ID: H_grouped_gbdt", plans[0])
        self.assertIn(brief.evaluation_protocol.protocol_hash, plans[0])

    def test_architecture_policy_distinguishes_custom_networks_from_tree_libraries(self) -> None:
        self.assertEqual(
            "conventional",
            classify_architecture("Tune a LightGBM gradient-boosted tree ensemble."),
        )
        self.assertEqual(
            "custom_neural",
            classify_architecture(
                "Build a custom neural network as an nn.Module with a learned gating block."
            ),
        )

    def test_architecture_planner_has_a_non_template_fallback(self) -> None:
        with patch(
            "agents.technique_agent.call_llm",
            side_effect=RuntimeError("provider unavailable"),
        ):
            plan = TechniqueAgent(council_brief=self.brief()).propose_architecture_exploration(
                self.analysis(),
                "Use a measured LightGBM control.",
                0.8,
                measured_alternatives="LightGBM and HistGradientBoosting plateaued.",
                plateau_evidence="no material gain in two measurements",
            )
        self.assertIn("Architecture exploration track: custom_neural", plan)
        self.assertIn("PyTorch `nn.Module`", plan)
        self.assertIn("Do not instantiate TabNet", plan)
        self.assertIn("Do not claim global novelty", plan)

    def test_plateau_reserves_custom_architecture_before_more_merging(self) -> None:
        manager = object.__new__(ManagerAgent)
        manager.metric_direction = "maximize"
        manager.total_budget = 6
        manager.experiments_executed = 2
        manager.best_node_id = "node1"
        manager.architecture_exploration_enabled = True
        manager.architecture_min_budget = 3
        manager.plateau_patience = 1
        manager.plateau_relative_gain = 5e-4
        first = NodeState(
            node_id="node1",
            parent_id="root",
            node_type="implementation",
            plan="Train a LightGBM model.",
            operator="root",
            executed=True,
            code="from lightgbm import LGBMClassifier",
            result={"status": "completed", "score": 0.8},
        )
        second = NodeState(
            node_id="node2",
            parent_id="root",
            node_type="implementation",
            plan="Train a HistGradientBoosting model.",
            operator="root",
            executed=True,
            code="from sklearn.ensemble import HistGradientBoostingClassifier",
            result={"status": "completed", "score": 0.8001},
        )
        manager.all_nodes = {"node1": first, "node2": second}
        plateau = manager._plateau_state()
        self.assertTrue(plateau["plateaued"])
        reason = manager._architecture_intervention_reason(experiments_remaining=4)
        self.assertIsNotNone(reason)
        self.assertIn("conventional", str(reason))

        custom = NodeState(
            node_id="node3",
            parent_id="root",
            node_type="implementation",
            plan="Build a custom neural network with a learned gating block.",
            operator="architect",
            executed=True,
            code="class TaskNet(nn.Module):\n    pass\n",
            result={"status": "completed", "score": 0.79},
        )
        manager.all_nodes["node3"] = custom
        self.assertIsNone(
            manager._architecture_intervention_reason(experiments_remaining=3)
        )

    def test_architecture_implementation_contract_replaces_parent_predictor(self) -> None:
        prompt = ImplementationAgent()._prompt(
            self.analysis(),
            "Architecture exploration track: custom_neural",
            "from lightgbm import LGBMClassifier",
            None,
            None,
            "",
            "",
            self.brief(),
            "architect",
        )
        self.assertIn("ARCHITECTURE EXPERIMENT CONTRACT", prompt)
        self.assertIn("DO NOT retain its conventional estimator", prompt)
        self.assertIn("plain MLP may appear only as an ablation", prompt)

    def test_hardware_device_contract_is_always_included(self) -> None:
        prompt = ImplementationAgent()._prompt(
            self.analysis(),
            "Train a gradient boosted model.",
            None,
            None,
            None,
            "",
            "",
            self.brief(),
            "root",
        )
        self.assertIn("HARDWARE / DEVICE CONTRACT", prompt)
        self.assertIn("torch.cuda.is_available()", prompt)
        self.assertIn("not compatible with the current PyTorch installation", prompt)
        self.assertIn("Run on CPU", prompt)

    def test_cuda_incompatibility_error_detection(self) -> None:
        p100_warning = (
            "Found GPU0 Tesla P100-PCIE-16GB which is of cuda capability 6.0.\n"
            "Minimum and Maximum cuda capability supported by this version of "
            "PyTorch is (7.0) - (12.0)\n"
            "Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible "
            "with the current PyTorch installation."
        )
        self.assertTrue(ImplementationAgent._cuda_incompatibility_error(p100_warning))
        self.assertTrue(
            ImplementationAgent._cuda_incompatibility_error(
                "RuntimeError: CUDA error: out of memory"
            )
        )
        self.assertTrue(
            ImplementationAgent._cuda_incompatibility_error(
                "AssertionError: Torch not compiled with CUDA enabled"
            )
        )
        self.assertFalse(
            ImplementationAgent._cuda_incompatibility_error(
                "ValueError: bad input shape"
            )
        )

    def test_gpu_compute_capability_parsing(self) -> None:
        instance = ImplementationAgent("unused-node-dir", "unused-task-dir")
        with patch(
            "agents.implementation_agent.subprocess.run",
            return_value=unittest.mock.Mock(
                returncode=0, stdout="6.0\n", stderr=""
            ),
        ):
            self.assertEqual(instance._gpu_compute_capability(), 6.0)
        with patch(
            "agents.implementation_agent.subprocess.run",
            return_value=unittest.mock.Mock(
                returncode=0, stdout="8.6\n", stderr=""
            ),
        ):
            self.assertEqual(instance._gpu_compute_capability(), 8.6)
        with patch(
            "agents.implementation_agent.subprocess.run",
            return_value=unittest.mock.Mock(
                returncode=1, stdout="", stderr="No devices found"
            ),
        ):
            self.assertIsNone(instance._gpu_compute_capability())

    def test_cuda_probe_treats_capability_warnings_as_incompatible(self) -> None:
        instance = ImplementationAgent("unused-node-dir", "unused-task-dir")
        instance._cuda_probe_cache = {}
        warning = (
            "Found GPU0 Tesla P100-PCIE-16GB which is of cuda capability 6.0.\n"
            "Minimum and Maximum cuda capability supported by this version of "
            "PyTorch is (7.0) - (12.0)"
        )

        def fake_run(command, *args, **kwargs):
            stdout = "1\n"
            stderr = warning if "synchronize" in command[-1] else ""
            return unittest.mock.Mock(returncode=0, stdout=stdout, stderr=stderr)

        with patch(
            "agents.implementation_agent.subprocess.run", side_effect=fake_run
        ):
            self.assertTrue(instance._cuda_incompatible_environment())

        instance._cuda_probe_cache = {}
        with patch(
            "agents.implementation_agent.subprocess.run",
            return_value=unittest.mock.Mock(
                returncode=0, stdout="1\n", stderr=""
            ),
        ):
            self.assertFalse(instance._cuda_incompatible_environment())

    def test_architecture_source_gate_rejects_library_or_plain_mlp_fallbacks(self) -> None:
        tree_only = "from lightgbm import LGBMClassifier\nmodel = LGBMClassifier()\n"
        self.assertTrue(_architecture_source_errors(tree_only))

        plain_mlp = """
import torch
from torch import nn
class PlainMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    def forward(self, x):
        return self.layers(x)
"""
        self.assertTrue(
            any("custom interaction" in error for error in _architecture_source_errors(plain_mlp))
        )

        custom_network = """
import torch
from torch import nn
class GatedInteractionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Linear(4, 8)
        self.gate = nn.Linear(4, 8)
        self.head = nn.Linear(8, 1)
    def forward(self, x):
        mixed = self.value(x) * torch.sigmoid(self.gate(x))
        return self.head(mixed + self.value(x))
"""
        self.assertEqual([], _architecture_source_errors(custom_network))

    def test_predictive_modality_detection_ignores_instructions_and_output_templates(self) -> None:
        files = [
            {"path": "train.csv", "kind": "table"},
            {"path": "sample_submission.csv", "kind": "table"},
            {"path": "task_description.md", "kind": "text"},
        ]
        inventory = predictive_modality_inventory(files)
        self.assertEqual(["tabular"], inventory["modalities"])
        self.assertFalse(inventory["is_multimodal"])

        files.append({"path": "images/example.png", "kind": "image"})
        inventory = predictive_modality_inventory(files)
        self.assertEqual(["image", "tabular"], inventory["modalities"])
        self.assertTrue(inventory["is_multimodal"])

    def test_multimodal_report_requires_full_and_single_modality_controls(self) -> None:
        incomplete = {
            "modality_ablation_scores": [
                {
                    "modalities": ["image", "tabular"],
                    "score": 0.82,
                    "fold_scores": [0.81, 0.83],
                    "validation_indices_hash": "shared-folds",
                },
                {
                    "modalities": ["tabular"],
                    "score": 0.81,
                    "fold_scores": [0.80, 0.82],
                    "validation_indices_hash": "shared-folds",
                },
            ]
        }
        errors = validate_modality_ablation_report(
            incomplete, ["image", "tabular"]
        )
        self.assertTrue(any("excluding 'tabular'" in error for error in errors))

        complete = {
            "modality_ablation_scores": [
                {
                    "modalities": ["image", "tabular"],
                    "score": 0.82,
                    "fold_scores": [0.81, 0.83],
                    "validation_indices_hash": "shared-folds",
                },
                {
                    "modalities": ["tabular"],
                    "score": 0.81,
                    "fold_scores": [0.80, 0.82],
                    "validation_indices_hash": "shared-folds",
                },
                {
                    "modalities": ["image"],
                    "score": 0.83,
                    "fold_scores": [0.82, 0.84],
                    "validation_indices_hash": "shared-folds",
                },
            ]
        }
        self.assertEqual(
            [], validate_modality_ablation_report(complete, ["image", "tabular"])
        )

    def test_multimodal_plan_gets_an_enforced_ablation_contract(self) -> None:
        analysis = self.analysis()
        analysis.files.extend(
            [
                {
                    "path": "images/example.png",
                    "kind": "image",
                    "extension": ".png",
                    "bytes": 1,
                },
                {
                    "path": "task_description.md",
                    "kind": "text",
                    "extension": ".md",
                    "bytes": 1,
                },
            ]
        )
        prompt = ImplementationAgent()._prompt(
            analysis,
            "Modality scope: modality_ablation\nMeasure modality contribution.",
            None,
            None,
            None,
            "",
            "",
            self.brief(),
            "root",
        )
        self.assertIn("MULTIMODAL CONTRIBUTION CONTRACT", prompt)
        self.assertIn("full fusion", prompt)
        self.assertIn("single-modality controls", prompt)
        self.assertIn("modality_ablation_scores", prompt)

    def test_implementation_prompt_lays_static_prefix_first_for_caching(self) -> None:
        analysis = self.analysis()
        brief = self.brief()
        attempt_one = ImplementationAgent()._prompt(
            analysis,
            "Node-specific plan",
            "PARENT-CODE",
            None,
            "CANDIDATE-1",
            "attempt-1 feedback",
            "",
            brief,
            "root",
        )
        attempt_two = ImplementationAgent()._prompt(
            analysis,
            "Node-specific plan",
            "PARENT-CODE",
            None,
            "CANDIDATE-2",
            "attempt-2 feedback",
            "",
            brief,
            "root",
        )
        council_pos = attempt_one.find("ML RESEARCH COUNCIL CONTRACT")
        plan_pos = attempt_one.find("Implementation plan:")
        repair_pos = attempt_one.find("Previous attempted program:")
        self.assertGreater(plan_pos, council_pos)
        self.assertGreater(repair_pos, plan_pos)
        self.assertTrue(attempt_one.startswith(attempt_two[: council_pos + 1]))
        shared_prefix = attempt_two[: council_pos]
        self.assertEqual(attempt_one[: len(shared_prefix)], shared_prefix)

    def test_member_report_prompt_shares_evidence_prefix_across_members(self) -> None:
        coordinator = CouncilCoordinator(enable_web=False)
        fingerprint = {"data_kinds": {"table": 4}}
        diagnostics = {"diagnostics_hash": "abc", "tables": []}
        sources = [{"source_id": "S1", "title": "T", "url": "U", "snippet": "S"}]
        # Capture the member prompts by monkeypatching call_llm_json.
        captured = []

        def capture(*args, **kwargs):
            if len(args) >= 2 and isinstance(args[1], str):
                captured.append(args[1])
            return {"member_id": "x", "summary": "s", "findings": [],
                    "hypotheses": [{
                        "hypothesis_id": "H",
                        "title": "t",
                        "model_family": "m",
                        "rationale": "r",
                        "evidence_ids": ["local_preflight"],
                        "experiment": "e",
                        "expected_signal": "s",
                        "estimated_cost": "low",
                        "risks": [],
                        "stopping_rule": "s",
                        "compatible_with": [],
                        "architecture_track": "other",
                        "architecture_spec": "",
                        "novelty_test": "",
                        "modality_scope": "not_applicable",
                        "modality_ablation": "",
                    }],
                    "assumptions": [],
                    "unresolved_questions": [],
                    "member_id": "x"}

        with patch("agents.council.coordinator.call_llm_json", side_effect=capture):
            coordinator._member_report(
                {"member_id": "member_a", "mandate": "mandate A",
                 "research_focus": "focus A", "key_uncertainties": ["q"]},
                fingerprint,
                diagnostics,
                {"member_id": "member_a", "status": "ok"},
                sources,
            )
            coordinator._member_report(
                {"member_id": "member_b", "mandate": "mandate B",
                 "research_focus": "focus B", "key_uncertainties": ["q"]},
                fingerprint,
                diagnostics,
                {"member_id": "member_b", "status": "ok"},
                sources,
            )
        self.assertEqual(2, len(captured))
        marker_a = captured[0].find("Focused diagnostic")
        marker_b = captured[1].find("Focused diagnostic")
        self.assertGreater(marker_a, -1)
        self.assertGreater(marker_b, -1)
        self.assertEqual(captured[0][:marker_a], captured[1][:marker_b])
        self.assertIn("De-identified fingerprint:", captured[0])
        self.assertIn("Primary literature evidence:", captured[0])
        self.assertLess(
            captured[0].find("De-identified fingerprint:"),
            captured[0].find("Focused diagnostic"),
        )

    def test_brief_writes_machine_and_human_readable_artifacts(self) -> None:
        council_dir = self.root / "council"
        brief = self.brief()
        json_path, report_path = brief.write(council_dir)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["brief_hash"], brief.brief_hash)
        self.assertIn("Selected research portfolio", report_path.read_text(encoding="utf-8"))

    def test_council_degrades_to_auditable_control_when_llm_is_unavailable(self) -> None:
        (self.task_dir / "train.csv").write_text(
            "feature,diagnosis_label\n1,0\n2,1\n", encoding="utf-8"
        )
        (self.task_dir / "test.csv").write_text(
            "feature\n3\n4\n", encoding="utf-8"
        )
        (self.task_dir / "test_labels.csv").write_text(
            "diagnosis_label\n0\n1\n", encoding="utf-8"
        )
        (self.task_dir / "sample_submission.csv").write_text(
            "id,prediction\na,0.5\nb,0.5\n", encoding="utf-8"
        )
        with patch(
            "agents.council.coordinator.call_llm_json",
            side_effect=RuntimeError("provider unavailable"),
        ):
            brief = CouncilCoordinator(
                enable_web=False,
                enable_generated_diagnostics=False,
            ).run(self.analysis(), self.root / "run")
        self.assertEqual("degraded", brief.status)
        self.assertNotIn("test_labels.csv", brief.allowed_input_paths)
        self.assertEqual(1, len(brief.selected_portfolio))
        self.assertTrue((self.root / "run" / "council" / "council_brief.json").is_file())
        preflight = brief.evidence[0]["summary"]
        sample_profile = next(
            item for item in preflight["tables"] if item["path"] == "sample_submission.csv"
        )
        self.assertEqual({}, sample_profile["target_summary"])


if __name__ == "__main__":
    unittest.main()
