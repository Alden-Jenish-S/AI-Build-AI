"""Tests for the search-evidence helpers added with the supervised-search ideas."""

from __future__ import annotations

import unittest

from agents.manager_agent import ManagerAgent
from search_evidence import (
    estimator_families,
    family_fingerprint,
    pearson_correlation,
    relative_noise_floor,
    score_noise_estimate,
    signature_from_result,
    valid_signature,
)
from tree.node import NodeState
from tree.scheduler import UCB1Scheduler


class SignatureTests(unittest.TestCase):
    def test_valid_signature_accepts_finite_floats(self) -> None:
        self.assertTrue(valid_signature([0.5] * 8))

    def test_valid_signature_rejects_short_or_nonfinite(self) -> None:
        self.assertFalse(valid_signature([0.5] * 7))
        self.assertFalse(valid_signature([0.5, float("nan"), 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
        self.assertFalse(valid_signature([True] * 8))
        self.assertFalse(valid_signature("0.5" * 8))

    def test_signature_from_result_absent_when_missing_or_invalid(self) -> None:
        self.assertIsNone(signature_from_result({}))
        self.assertIsNone(signature_from_result({"prediction_signature": "garbage"}))

    def test_signature_from_result_extracts(self) -> None:
        result = {"prediction_signature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
        self.assertEqual(signature_from_result(result), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


class PearsonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vector = [float(value) for value in range(8)]

    def test_perfect_positive_correlation(self) -> None:
        self.assertAlmostEqual(pearson_correlation(self.vector, list(self.vector)), 1.0)

    def test_perfect_negative_correlation(self) -> None:
        self.assertAlmostEqual(
            pearson_correlation(self.vector, [-value for value in self.vector]), -1.0
        )

    def test_orthogonal_signals_are_decorrelated(self) -> None:
        orthogonal = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]
        self.assertAlmostEqual(pearson_correlation(self.vector, orthogonal), 0.0, places=6)

    def test_mismatched_lengths_or_zero_variance_degrade_to_zero(self) -> None:
        self.assertEqual(pearson_correlation(self.vector, [1.0, 2.0]), 0.0)
        self.assertEqual(pearson_correlation([1.0] * 8, [2.0] * 8), 0.0)


class NoiseEstimateTests(unittest.TestCase):
    def test_fold_scores_produce_stddev(self) -> None:
        result = {"fold_scores": [0.6, 0.7, 0.8]}
        self.assertAlmostEqual(score_noise_estimate(result), 0.1)

    def test_seed_scores_fallback(self) -> None:
        result = {"seed_scores": [0.9, 0.95]}
        self.assertAlmostEqual(score_noise_estimate(result), 0.05 / 2**0.5)

    def test_no_dispersion_evidence_yields_none(self) -> None:
        self.assertIsNone(score_noise_estimate({"score": 0.7}))
        self.assertIsNone(score_noise_estimate({"fold_scores": [0.7]}))

    def test_relative_floor_scales_with_score(self) -> None:
        self.assertEqual(relative_noise_floor(0.0), 1e-6)
        self.assertAlmostEqual(relative_noise_floor(1.0), 1.5e-3)


class FamilyFingerprintTests(unittest.TestCase):
    def test_fingerprint_empty_without_family_evidence(self) -> None:
        self.assertEqual(family_fingerprint("build a pipeline and score it locally"), "")

    def test_fingerprint_detects_xgboost_and_mlp(self) -> None:
        self.assertIn("xgboost", estimator_families("use XGBClassifier with early stopping"))
        self.assertIn("mlp", estimator_families("fit a multi-layer perceptron"))

    def test_fingerprint_nonempty_for_measured_family(self) -> None:
        self.assertTrue(family_fingerprint("train a random forest on the cleaned table"))


class SchedulerComplementarityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = UCB1Scheduler(total_budget=8)
        self.root = NodeState(
            node_id="root", parent_id=None, node_type="planning", plan="", operator="root"
        )
        self.root.executed = True

    def node(self, node_id: str, parent_id: str, *, operator: str = "refine") -> NodeState:
        node = NodeState(
            node_id=node_id,
            parent_id=parent_id,
            node_type="implementation",
            plan="",
            operator=operator,
        )
        return node

    def test_decorrelated_branch_is_favored(self) -> None:
        best = self.node("best", "root", operator="root")
        best.executed = True
        best.result = {"prediction_signature": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]}
        aligned = self.node("aligned", "best")
        orthogonal = self.node("orthogonal", "best")
        aligned.result = {"prediction_signature": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]}
        orthogonal.result = {
            "prediction_signature": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
        }
        all_nodes = {
            "root": self.root,
            "best": best,
            "aligned": aligned,
            "orthogonal": orthogonal,
        }
        self.root.children_ids = ["best"]
        best.children_ids = ["aligned", "orthogonal"]
        scores = self.scheduler.frontier_scores(
            "root",
            all_nodes,
            best_signature=signature_from_result(best.result),
            signature_provider=lambda node: signature_from_result(node.result),
        )
        self.assertIn("aligned", scores)
        self.assertIn("orthogonal", scores)
        self.assertGreater(scores["orthogonal"], scores["aligned"])

    def test_complementarity_inert_without_incumbent_signature(self) -> None:
        child = self.node("child", "root", operator="root")
        all_nodes = {"root": self.root, "child": child}
        self.root.children_ids = ["child"]
        scores = self.scheduler.frontier_scores("root", all_nodes, signature_provider=lambda n: [])
        selected = self.scheduler.select_next_node("root", all_nodes)
        self.assertEqual(selected, "child")
        self.assertGreaterEqual(scores["child"], 0.0)


class ManagerSearchRuleTests(unittest.TestCase):
    def bare_manager(self) -> ManagerAgent:
        manager = object.__new__(ManagerAgent)
        manager.metric_direction = "maximize"
        manager.metric_name = "auroc"
        manager.improvement_noise_k = 0.35
        manager.implementation_attempts = 0
        manager.tuning_attempts = 0
        manager.probe_attempts = 0
        manager.ensemble_attempts = 0
        manager.attempt_limit = 50
        manager.tuning_attempt_limit = 4
        manager.probe_attempt_limit = 3
        manager.experiments_executed = 0
        manager.total_budget = 10
        return manager

    def test_probe_and_ensemble_accounting(self) -> None:
        manager = self.bare_manager()
        self.assertTrue(manager._can_attempt("probe"))
        manager.probe_attempts = 3
        self.assertFalse(manager._can_attempt("probe"))
        manager.ensemble_attempts = 1
        self.assertFalse(manager._can_attempt("ensemble"))
        self.assertTrue(manager._can_attempt("root"))

    def test_improvement_requires_margin_over_noise(self) -> None:
        manager = self.bare_manager()
        self.assertTrue(manager._improved(0.51, 0.50, noise=0.01))
        self.assertFalse(manager._improved(0.501, 0.50, noise=0.01))

    def test_noise_from_folds_then_seed_scores_then_floor(self) -> None:
        manager = self.bare_manager()
        folded = NodeState(
            node_id="n1", parent_id=None, node_type="implementation", plan="", operator="root"
        )
        folded.executed = True
        folded.result = {"score": 0.7, "fold_scores": [0.6, 0.7, 0.8]}
        self.assertAlmostEqual(manager._noise_for(folded), 0.1)
        seeded = NodeState(
            node_id="n2", parent_id=None, node_type="implementation", plan="", operator="root"
        )
        seeded.executed = True
        seeded.result = {"score": 0.9, "seed_scores": [0.9, 0.95]}
        self.assertAlmostEqual(manager._noise_for(seeded), 0.05 / 2**0.5)
        bare = NodeState(
            node_id="n3", parent_id=None, node_type="implementation", plan="", operator="root"
        )
        bare.executed = True
        bare.result = {"score": 1.0}
        self.assertEqual(manager._noise_for(bare), relative_noise_floor(1.0))
        self.assertEqual(manager._noise_for(None), 0.0)

    def test_merge_pair_prefers_decorrelated_pair(self) -> None:
        manager = self.bare_manager()
        manager.complementarity_weight = 0.4
        base = [0.0, 1.0] * 4
        nodes = []
        for index, (score, signature) in enumerate(
            [
                (0.90, base),
                (0.89, list(base)),
                (0.85, [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]),
            ],
            start=1,
        ):
            node = NodeState(
                node_id=f"n{index}",
                parent_id=None,
                node_type="implementation",
                plan="",
                operator="root",
            )
            node.executed = True
            node.result = {"score": score, "prediction_signature": list(signature)}
            nodes.append(node)
        pair = manager._merge_pair(nodes)
        self.assertIsNotNone(pair)
        best_correlation = abs(pearson_correlation(nodes[0].result["prediction_signature"], nodes[1].result["prediction_signature"]))
        chosen_correlation = abs(pearson_correlation(pair[0].result["prediction_signature"], pair[1].result["prediction_signature"]))
        self.assertLess(chosen_correlation, best_correlation)


if __name__ == "__main__":
    unittest.main()