from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from agents.manager_agent import ManagerAgent
from agents.task_analyzer import TaskAnalysis
from prediction_evidence import (
    cross_fitted_blend,
    evidence_compatible,
    inspect_prediction_evidence,
)
from tree.node import NodeState


class PredictionEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.target = np.asarray([0, 1] * 20)
        self.index = np.arange(len(self.target))
        self.fold = self.index % 5
        self.test_index = np.asarray(["a", "b", "c"])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_bundle(self, name: str, confidence: float) -> Path:
        correct = np.where(self.target == 0, confidence, 1.0 - confidence)
        oof = np.column_stack((correct, 1.0 - correct))
        test = np.asarray(
            [[confidence, 1.0 - confidence], [0.5, 0.5], [1.0 - confidence, confidence]]
        )
        path = self.root / f"{name}.npz"
        np.savez_compressed(
            path,
            oof_pred=oof,
            oof_target=self.target,
            oof_index=self.index,
            oof_fold=self.fold,
            test_pred=test,
            test_index=self.test_index,
        )
        return path

    def test_supported_metric_is_recomputed_from_saved_predictions(self) -> None:
        evidence = inspect_prediction_evidence(self.write_bundle("first", 0.8), "log loss")
        self.assertTrue(evidence.valid, evidence.errors)
        self.assertTrue(evidence.centrally_scored)
        self.assertTrue(evidence.blendable)
        self.assertAlmostEqual(evidence.score, -np.log(0.8), places=10)

    def test_task_native_metric_stays_valid_but_is_not_blendable(self) -> None:
        evidence = inspect_prediction_evidence(
            self.write_bundle("native", 0.8), "simulation reward"
        )
        self.assertTrue(evidence.valid, evidence.errors)
        self.assertFalse(evidence.centrally_scored)
        self.assertFalse(evidence.blendable)

    def test_duplicate_indices_are_rejected(self) -> None:
        path = self.write_bundle("duplicate", 0.8)
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        arrays["oof_index"] = np.zeros_like(arrays["oof_index"])
        np.savez_compressed(path, **arrays)
        evidence = inspect_prediction_evidence(path, "log loss")
        self.assertFalse(evidence.valid)
        self.assertTrue(any("duplicate" in error for error in evidence.errors))

    def test_duplicate_test_identifiers_are_rejected(self) -> None:
        path = self.write_bundle("duplicate_test", 0.8)
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        arrays["test_index"] = np.asarray(["a", "a", "c"])
        np.savez_compressed(path, **arrays)
        evidence = inspect_prediction_evidence(path, "log loss")
        self.assertFalse(evidence.valid)
        self.assertTrue(any("test_index" in error for error in evidence.errors))

    def test_different_fold_contracts_cannot_be_blended(self) -> None:
        first = inspect_prediction_evidence(self.write_bundle("first", 0.8), "log loss")
        second_path = self.write_bundle("second", 0.78)
        with np.load(second_path, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        arrays["oof_fold"] = (arrays["oof_fold"] + 1) % 5
        np.savez_compressed(second_path, **arrays)
        second = inspect_prediction_evidence(second_path, "log loss")
        self.assertFalse(evidence_compatible([first, second]))

    def test_cross_fitted_blend_is_nontrivial_and_aligned(self) -> None:
        items = [
            inspect_prediction_evidence(self.write_bundle("first", 0.8), "log loss"),
            inspect_prediction_evidence(self.write_bundle("second", 0.7), "log loss"),
        ]
        self.assertTrue(evidence_compatible(items))
        blend = cross_fitted_blend(items, "log loss", "minimize", trials=20)
        self.assertEqual((2,), blend.weights.shape)
        self.assertGreaterEqual(np.count_nonzero(blend.weights), 2)
        self.assertAlmostEqual(float(blend.weights.sum()), 1.0)
        self.assertEqual(items[0].oof_pred.shape, blend.oof_pred.shape)
        self.assertEqual(items[0].test_pred.shape, blend.test_pred.shape)

    def test_pruned_candidates_remain_eligible_and_duplicates_are_removed(self) -> None:
        first_path = self.write_bundle("first", 0.8)
        duplicate_path = self.root / "duplicate.npz"
        duplicate_path.write_bytes(first_path.read_bytes())
        third_path = self.write_bundle("third", 0.7)
        manager = object.__new__(ManagerAgent)
        manager.metric_name = "log loss"
        manager.metric_direction = "minimize"

        def node(node_id: str, path: Path, *, pruned: bool = False) -> NodeState:
            return NodeState(
                node_id=node_id,
                parent_id="root",
                node_type="implementation",
                operator="root",
                executed=True,
                result={
                    "status": "completed",
                    "score": 9.0,
                    "pruned": pruned,
                    "oof_predictions": str(path),
                },
            )

        manager.all_nodes = {
            "node1": node("node1", first_path, pruned=True),
            "node2": node("node2", duplicate_path),
            "node3": node("node3", third_path),
        }
        candidates = manager._ensemble_candidates()
        self.assertEqual(["node1", "node3"], [item.node_id for item in candidates])
        self.assertAlmostEqual(candidates[0].result["score"], -np.log(0.8))

    def test_manager_builds_deterministic_final_ensemble_without_an_llm(self) -> None:
        task_dir = self.root / "task"
        run_root = self.root / "run"
        task_dir.mkdir()
        run_root.mkdir()
        (task_dir / "sample_submission.csv").write_text(
            "id,p0,p1\na,0,0\nb,0,0\nc,0,0\n", encoding="utf-8"
        )
        first_path = self.write_bundle("first", 0.8)
        with np.load(first_path, allow_pickle=False) as payload:
            first_arrays = {key: payload[key] for key in payload.files}
        # This candidate has the better mean but a small catastrophic slice.
        # The second candidate is >25% worse locally yet useful for hedging, so
        # this guards against reintroducing a narrow local-score cutoff.
        confidence = np.full(len(self.target), 0.99)
        confidence[[0, 1]] = 0.01
        first_arrays["oof_pred"] = np.column_stack(
            (
                np.where(self.target == 0, confidence, 1.0 - confidence),
                np.where(self.target == 1, confidence, 1.0 - confidence),
            )
        )
        np.savez_compressed(first_path, **first_arrays)
        second_path = self.write_bundle("second", 0.7)
        manager = object.__new__(ManagerAgent)
        manager.metric_name = "log loss"
        manager.metric_direction = "minimize"
        manager.task_dir = task_dir
        manager.run_root = run_root
        manager.task_analysis = TaskAnalysis(
            task_name="synthetic",
            task_dir=task_dir,
            goal="predict",
            target="target",
            expected_output="sample_submission.csv",
            metric="log loss",
            direction="minimize",
            submission={
                "kind": "table",
                "path": "sample_submission.csv",
                "extension": ".csv",
            },
        )
        manager.all_nodes = {
            "node1": NodeState(
                node_id="node1",
                parent_id="root",
                node_type="implementation",
                operator="root",
                executed=True,
                result={
                    "status": "completed",
                    "score": 999.0,
                    "oof_predictions": str(first_path),
                },
            ),
            "node2": NodeState(
                node_id="node2",
                parent_id="root",
                node_type="implementation",
                operator="tune",
                executed=True,
                result={
                    "status": "completed",
                    "score": -999.0,
                    "oof_predictions": str(second_path),
                },
            ),
        }
        manager._node_counter = 2
        manager._dirty_node_ids = set()
        manager.implementation_attempts = 2
        manager.attempt_limit = 10
        manager.ensemble_attempts = 0
        manager.experiments = []
        manager.council_brief = None
        manager._persist_tree_state = lambda: None
        manager._log = lambda _message: None

        ensemble = manager._build_final_ensemble()

        self.assertIsNotNone(ensemble)
        assert ensemble is not None
        self.assertEqual("ensemble", ensemble.operator)
        self.assertEqual(2, len(ensemble.result["ensemble_members"]))
        self.assertGreaterEqual(
            np.count_nonzero(ensemble.result["ensemble_weights"]), 2
        )
        self.assertTrue(Path(ensemble.result["output"]).is_file())
        self.assertTrue((run_root / "ensemble_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
