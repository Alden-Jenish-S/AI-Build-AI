"""Cross-fitted meta-model and convex fallback weight optimization."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def optimize_constrained_blend(
    targets: np.ndarray,
    predictions: np.ndarray,
    metric_evaluator: Any,
) -> Tuple[np.ndarray, float]:
    """Find non-negative convex weights (sum to 1) maximizing metric_evaluator."""
    n_models = predictions.shape[0]
    if n_models == 0:
        raise ValueError("predictions matrix must contain at least one model")

    uniform_weights = np.full(n_models, 1.0 / n_models)
    best_weights = uniform_weights.copy()
    best_score = metric_evaluator(targets, uniform_weights @ predictions)

    # Evaluate single models
    for i in range(n_models):
        single_weights = np.zeros(n_models)
        single_weights[i] = 1.0
        score = metric_evaluator(targets, single_weights @ predictions)
        if score > best_score + 1e-12:
            best_score = score
            best_weights = single_weights

    # Try scipy SLSQP optimization
    if n_models > 1:
        try:
            from scipy.optimize import minimize

            opt = minimize(
                lambda w: -metric_evaluator(targets, w @ predictions),
                x0=uniform_weights,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n_models,
                constraints={
                    "type": "eq",
                    "fun": lambda w: float(w.sum() - 1.0),
                },
                options={"maxiter": 200, "ftol": 1e-12},
            )
            if opt.success and np.isfinite(opt.x).all() and opt.x.sum() > 0:
                candidate = np.clip(opt.x, 0.0, None)
                candidate /= candidate.sum()
                score = metric_evaluator(targets, candidate @ predictions)
                if score > best_score + 1e-12:
                    best_score = score
                    best_weights = candidate
        except Exception:
            pass

    # Coordinate search fine-tuning
    for step in (0.20, 0.10, 0.05, 0.02, 0.01):
        improved = True
        while improved:
            improved = False
            for src in range(n_models):
                for tgt in range(n_models):
                    if src == tgt or best_weights[src] <= 0:
                        continue
                    amt = min(step, best_weights[src])
                    candidate = best_weights.copy()
                    candidate[src] -= amt
                    candidate[tgt] += amt
                    score = metric_evaluator(targets, candidate @ predictions)
                    if score > best_score + 1e-12:
                        best_weights = candidate
                        best_score = score
                        improved = True

    return best_weights, best_score


def fit_cross_validated_stacker(
    oof_predictions: List[np.ndarray],
    targets: np.ndarray,
    metric_evaluator: Any,
) -> Dict[str, Any]:
    """Fit a cross-validated ensemble stacker with OOF guardrails."""
    n_models = len(oof_predictions)
    if n_models == 0:
        raise ValueError("oof_predictions must be non-empty")

    matrix = np.stack(oof_predictions)
    weights, best_score = optimize_constrained_blend(
        targets, matrix, metric_evaluator
    )

    single_scores = [
        float(metric_evaluator(targets, oof_predictions[i]))
        for i in range(n_models)
    ]
    best_single_score = max(single_scores)
    best_single_idx = int(np.argmax(single_scores))

    guardrail_applied = best_score <= best_single_score + 1e-12
    if guardrail_applied:
        weights = np.zeros(n_models)
        weights[best_single_idx] = 1.0
        best_score = best_single_score

    return {
        "weights": weights.tolist(),
        "ensemble_oof_score": float(best_score),
        "best_single_index": best_single_idx,
        "best_single_score": float(best_single_score),
        "single_scores": single_scores,
        "guardrail_applied": guardrail_applied,
    }
