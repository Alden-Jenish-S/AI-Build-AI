import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from agents.manager_agent import ManagerAgent
from search.policies import DiversityAssessment
from search.evidence import EvidenceService
from search.policies import PromotionController, PruningPolicy
from search.provenance import ArtifactRecord, ProvenanceGraph
from search.tuning import (
    TrialRecord,
    TuningCoordinator,
    TuningKnowledgeBase,
)
from tree.node import NodeState


class StatisticalPolicyTests(unittest.TestCase):
    def test_paired_fold_evidence_drives_pruning_and_promotion(self):
        service = EvidenceService()
        strong = service.compare(
            {
                "score": 0.75,
                "validation": {
                    "fold_scores": [0.74, 0.76, 0.75, 0.75],
                    "folds": 4,
                },
            },
            {
                "score": 0.60,
                "validation": {
                    "fold_scores": [0.59, 0.61, 0.60, 0.60],
                    "folds": 4,
                },
            },
            direction="maximize",
        )
        self.assertEqual(strong.method, "paired_folds")
        self.assertGreater(strong.probability_material_improvement, 0.99)
        self.assertFalse(PruningPolicy().decide(strong).prune)
        self.assertTrue(
            PromotionController().decide(
                strong, current_fidelity="screen"
            ).promote
        )

        weak = service.compare(
            {
                "score": 0.40,
                "validation": {
                    "fold_scores": [0.39, 0.41, 0.40, 0.40],
                    "folds": 4,
                },
            },
            {
                "score": 0.60,
                "validation": {
                    "fold_scores": [0.59, 0.61, 0.60, 0.60],
                    "folds": 4,
                },
            },
            direction="maximize",
        )
        self.assertTrue(PruningPolicy().decide(weak).prune)
        self.assertFalse(
            PromotionController().decide(
                weak, current_fidelity="screen"
            ).promote
        )


class ProvenanceTests(unittest.TestCase):
    def test_multi_source_dag_does_not_change_execution_parentage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "provenance.json"
            graph = ProvenanceGraph(path)
            graph.record(
                ArtifactRecord("a:predictions", "prediction_bundle", "a")
            )
            graph.record(
                ArtifactRecord("b:predictions", "prediction_bundle", "b")
            )
            graph.record(
                ArtifactRecord("stack:predictions", "prediction_bundle", "stack"),
                sources=[
                    ("a:predictions", "ensembled_from"),
                    ("b:predictions", "ensembled_from"),
                ],
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["artifacts"]), 3)
            self.assertEqual(len(payload["edges"]), 2)
            self.assertEqual(
                {
                    edge["source_artifact_id"] for edge in payload["edges"]
                },
                {"a:predictions", "b:predictions"},
            )
            self.assertNotIn("parent_ids", payload)

    def test_ensemble_action_has_one_tree_parent_and_multiple_artifact_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ManagerAgent.__new__(ManagerAgent)
            manager.run_root = Path(temp_dir)
            manager.task_name = "example"
            manager.task_type = "classification"
            manager.metric_name = "roc_auc"
            manager.metric_direction = "maximize"
            manager.total_budget = 6
            manager.experiments_executed = 2
            manager.node_counter = 2
            manager.all_nodes = {}
            manager._ensure_search_services()
            manager.diversity_controller = Mock()
            manager.diversity_controller.best_partner.return_value = (
                DiversityAssessment(
                    partner_node_id="node_1",
                    residual_diversity=0.5,
                    blend_gain=0.01,
                    information_value=0.1,
                    utility=0.11,
                )
            )
            parent = NodeState(
                "node_2",
                "root",
                "implementation",
                result={"score": 0.8, "status": "completed"},
                executed=True,
                fidelity="screen",
            )
            manager.all_nodes = {
                "root": NodeState(
                    "root", None, "technique", executed=True
                ),
                "node_1": NodeState(
                    "node_1",
                    "root",
                    "implementation",
                    result={"score": 0.79, "status": "completed"},
                    executed=True,
                    fidelity="screen",
                ),
                "node_2": parent,
            }
            evidence = EvidenceService().compare(
                {"score": 0.8}, {"score": 0.7}, direction="maximize"
            )
            manager._spawn_merge_ensemble_slot(parent, "node_2", evidence)
            merge = manager.all_nodes["node_3"]
            self.assertEqual(merge.parent_id, "node_2")
            self.assertEqual(
                merge.config["input_artifact_ids"],
                ["node_2:predictions", "node_1:predictions"],
            )
            self.assertNotIn("parent_ids", merge.config)
            self.assertEqual(merge.operator, "merge_ensemble")
            self.assertTrue(merge.config["manager_owned_merge"])
            self.assertFalse(merge.config["raw_code_fusion"])


class TuningKnowledgeTests(unittest.TestCase):
    def test_compatible_global_trials_are_reused_without_raw_score_pooling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge = TuningKnowledgeBase(
                Path(temp_dir) / "tuning_history.jsonl"
            )
            coordinator = TuningCoordinator(knowledge)
            search_space = coordinator.search_space_version(
                ["learning_rate", "max_depth"]
            )
            knowledge.append(
                TrialRecord(
                    trial_id="task-a-trial",
                    task_name="task-a",
                    model_family="gbdt",
                    search_space_version=search_space,
                    parameters={"learning_rate": 0.05, "max_depth": 6},
                    score=0.82,
                    normalized_improvement=0.12,
                    metric_name="roc_auc",
                    metric_direction="maximize",
                    fidelity="full",
                    status="completed",
                    dataset_fingerprint="dataset-a",
                )
            )
            context = coordinator.build_context(
                task_name="task-b",
                model_family="gbdt",
                tunable_parameters=["learning_rate", "max_depth"],
                metric_name="roc_auc",
                metric_direction="maximize",
                dataset_fingerprint="dataset-b",
            )
            self.assertTrue(context["global_trial_reuse"])
            self.assertEqual(
                context["suggested_initial_parameters"],
                [{"learning_rate": 0.05, "max_depth": 6}],
            )
            self.assertFalse(context["reused_trials"][0]["same_dataset"])
            self.assertIn(
                "normalized_improvement", context["reused_trials"][0]
            )
            self.assertNotIn("score", context["reused_trials"][0])


class ManagerOwnedEnsembleTests(unittest.TestCase):
    def test_manager_ensembles_two_node_models_with_oof_selected_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            row_ids = list(range(12))
            targets = [float(index) for index in row_ids]
            fold_ids = [index % 3 for index in row_ids]
            alternating_error = [
                1.0 if index % 2 == 0 else -1.0 for index in row_ids
            ]
            predictions = {
                "one": [
                    target + error
                    for target, error in zip(targets, alternating_error)
                ],
                "two": [
                    target - error
                    for target, error in zip(targets, alternating_error)
                ],
            }
            for node_id in ("one", "two"):
                node_dir = run_root / node_id
                (node_dir / "submission").mkdir(parents=True)
                pd.DataFrame(
                    {
                        "row_id": row_ids,
                        "target": targets,
                        "prediction": predictions[node_id],
                        "fold_id": fold_ids,
                    }
                ).to_csv(node_dir / "oof_predictions.csv", index=False)
                pd.DataFrame(
                    {
                        "id": [100, 101, 102, 103],
                        "target": predictions[node_id][:4],
                    }
                ).to_csv(
                    node_dir / "submission" / "submission.csv", index=False
                )
                (node_dir / "evaluation_manifest.json").write_text(
                    json.dumps(
                        {
                            "fidelity": "screen",
                            "cv_folds": 3,
                            "data_fraction": 1.0,
                        }
                    ),
                    encoding="utf-8",
                )

            manager = ManagerAgent.__new__(ManagerAgent)
            manager.run_root = run_root
            manager.task_name = "example"
            manager.metric_name = "rmse"
            manager.metric_direction = "minimize"
            manager.baseline_score = 2.0
            manager.total_budget = 4
            manager.initial_fanout = 2
            manager.experiments_executed = 2
            manager.all_nodes = {
                "root": NodeState(
                    "root", None, "technique", executed=True
                ),
                "one": NodeState(
                    "one",
                    "root",
                    "implementation",
                    result={
                        "score": 1.0,
                        "status": "completed",
                        "validation": {"fidelity": "screen"},
                    },
                    executed=True,
                    fidelity="screen",
                ),
                "two": NodeState(
                    "two",
                    "root",
                    "implementation",
                    result={
                        "score": 1.0,
                        "status": "completed",
                        "validation": {"fidelity": "screen"},
                    },
                    executed=True,
                    fidelity="screen",
                ),
            }
            manager._ensure_search_services()
            from agents.aggregator_agent import AggregatorAgent
            from tree.global_memory import GlobalMemory
            from tree.scheduler import UCB1Scheduler

            manager.aggregator_agent = AggregatorAgent()
            manager.global_memory = GlobalMemory()
            manager.scheduler = UCB1Scheduler(total_budget=4)
            merge = NodeState(
                "merge",
                "two",
                "implementation",
                operator="merge_ensemble",
                fidelity="screen",
                config={
                    "input_node_ids": ["one", "two"],
                    "input_artifact_ids": [
                        "one:predictions",
                        "two:predictions",
                    ],
                    "requested_strategy": "auto",
                    "manager_owned_merge": True,
                    "raw_code_fusion": False,
                },
            )
            manager.all_nodes["merge"] = merge
            manager.all_nodes["two"].children_ids.append("merge")

            self.assertTrue(manager._execute_ensemble_node(merge, "merge"))
            self.assertEqual(merge.result["status"], "completed")
            self.assertTrue(math.isfinite(merge.result["score"]))
            self.assertEqual(
                merge.result["merge"]["source_node_ids"], ["one", "two"]
            )
            self.assertTrue(merge.result["merge"]["manager_owned"])
            output = run_root / "merge"
            self.assertTrue((output / "oof_predictions.csv").is_file())
            self.assertTrue(
                (output / "submission" / "submission.csv").is_file()
            )
            manifest = json.loads(
                (output / "merge_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["manager_owned"])
            self.assertEqual(manifest["source_node_ids"], ["one", "two"])
            self.assertFalse(manifest["raw_code_fusion"])


if __name__ == "__main__":
    unittest.main()
