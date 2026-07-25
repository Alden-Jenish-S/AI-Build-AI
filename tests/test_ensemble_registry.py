"""Unit tests for the ensemble module strategy registry and stackers."""

import unittest
import numpy as np

from ensemble.registry import EnsembleStrategyRegistry, default_ensemble_registry
from ensemble.stacking import fit_cross_validated_stacker, optimize_constrained_blend
from ensemble.structured import (
    merge_embeddings,
    merge_segmentation_logits,
    merge_structured_outputs,
)


class EnsembleRegistryTests(unittest.TestCase):
    def test_default_registry_resolutions(self):
        reg = default_ensemble_registry()
        self.assertEqual(
            reg.resolve("class_probabilities")["preferred"],
            "cross_validated_stacking",
        )
        self.assertEqual(
            reg.resolve("embeddings")["preferred"],
            "normalized_learned_fusion",
        )
        self.assertEqual(
            reg.resolve("unknown_type")["fallback"],
            "average",
        )

    def test_custom_registration(self):
        reg = EnsembleStrategyRegistry()
        reg.register("custom_boxes", "wbf_v2", "nms")
        resolved = reg.resolve("custom_boxes")
        self.assertEqual(resolved["preferred"], "wbf_v2")
        self.assertEqual(resolved["fallback"], "nms")


class EnsembleStackingTests(unittest.TestCase):
    def test_optimize_constrained_blend_finds_convex_weights(self):
        targets = np.array([1, 0, 1, 0, 1, 0], dtype=float)
        m1 = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3], dtype=float)
        m2 = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9], dtype=float)
        matrix = np.stack([m1, m2])

        def neg_mse(y_true, pred):
            return -float(np.mean((y_true - pred) ** 2))

        weights, score = optimize_constrained_blend(targets, matrix, neg_mse)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        self.assertGreaterEqual(weights[0], 0.9)
        self.assertGreater(score, -0.1)

    def test_fit_cross_validated_stacker_guardrail(self):
        targets = np.array([1, 0, 1, 0], dtype=float)
        m1 = np.array([0.9, 0.1, 0.9, 0.1], dtype=float)  # perfect
        m2 = np.array([0.1, 0.9, 0.1, 0.9], dtype=float)  # terrible

        def mse_negative(y_true, pred):
            return -float(np.mean((y_true - pred) ** 2))

        result = fit_cross_validated_stacker([m1, m2], targets, mse_negative)
        self.assertEqual(result["best_single_index"], 0)
        self.assertAlmostEqual(result["weights"][0], 1.0, places=4)


class StructuredEnsembleTests(unittest.TestCase):
    def test_merge_embeddings_normalization(self):
        e1 = np.array([[3.0, 4.0]])
        e2 = np.array([[3.0, 4.0]])
        merged = merge_embeddings([e1, e2], weights=[0.5, 0.5], normalize=True)
        norm = np.linalg.norm(merged, axis=-1)
        self.assertAlmostEqual(float(norm[0]), 1.0, places=5)

    def test_merge_segmentation_logits(self):
        l1 = np.ones((2, 4, 4))
        l2 = np.full((2, 4, 4), 3.0)
        merged = merge_segmentation_logits([l1, l2], weights=[0.5, 0.5])
        self.assertTrue(np.allclose(merged, 2.0))

    def test_merge_structured_outputs_router(self):
        e1 = np.array([[1.0, 0.0]])
        e2 = np.array([[0.0, 1.0]])
        merged = merge_structured_outputs("embeddings", [e1, e2], weights=[0.5, 0.5])
        self.assertAlmostEqual(float(np.linalg.norm(merged)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
