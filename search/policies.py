"""First-class pruning, promotion, diversity, and information-gain policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .evidence import EvidenceEstimate


@dataclass(frozen=True)
class PruningDecision:
    prune: bool
    probability_competitive: float
    boundary: float
    information_value: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class PruningPolicy:
    """Bayes-risk pruning with an explicit false-prune/wasted-compute tradeoff."""

    def __init__(
        self,
        *,
        false_prune_cost: float = 9.0,
        wasted_compute_cost: float = 1.0,
        information_value: float = 1.0,
    ):
        if min(false_prune_cost, wasted_compute_cost, information_value) < 0:
            raise ValueError("policy costs must be non-negative")
        denominator = false_prune_cost + wasted_compute_cost
        self.boundary = (
            wasted_compute_cost / denominator if denominator > 0 else 0.0
        )
        self.information_value = float(information_value)

    def decide(
        self, evidence: EvidenceEstimate, *, expected_next_cost: float = 1.0
    ) -> PruningDecision:
        probability = evidence.probability_improvement
        information_value = (
            self.information_value * evidence.information_gain
            / max(float(expected_next_cost), 1e-12)
        )
        prune = probability < self.boundary and information_value < self.boundary
        reason = (
            "posterior competitiveness and value of more information are both "
            "below the cost-calibrated continuation boundary"
            if prune
            else "candidate remains competitive or decision-relevant uncertainty remains"
        )
        return PruningDecision(
            prune=prune,
            probability_competitive=probability,
            boundary=self.boundary,
            information_value=information_value,
            reason=reason,
        )


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    target_fidelity: str
    probability_material_improvement: float
    probability_boundary: float
    information_value: float
    utility: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class PromotionController:
    """Cost-aware fidelity promotion outside the scheduler."""

    FIDELITY_COST = {"screen": 1.0, "medium": 3.6, "full": 10.0}

    def __init__(
        self,
        *,
        wasted_promotion_cost: float = 4.0,
        missed_opportunity_cost: float = 1.0,
        information_value: float = 2.0,
    ):
        denominator = wasted_promotion_cost + missed_opportunity_cost
        self.boundary = (
            wasted_promotion_cost / denominator if denominator > 0 else 1.0
        )
        self.information_value = max(0.0, float(information_value))

    @staticmethod
    def next_fidelity(fidelity: str) -> str:
        return {"screen": "medium", "medium": "full", "full": "full"}.get(
            fidelity, "full"
        )

    def decide(
        self, evidence: EvidenceEstimate, *, current_fidelity: str
    ) -> PromotionDecision:
        target = self.next_fidelity(current_fidelity)
        if target == current_fidelity:
            return PromotionDecision(
                promote=False,
                target_fidelity=target,
                probability_material_improvement=evidence.probability_material_improvement,
                probability_boundary=self.boundary,
                information_value=0.0,
                utility=0.0,
                reason="candidate is already at full fidelity",
            )
        incremental_cost = max(
            1.0,
            self.FIDELITY_COST[target] - self.FIDELITY_COST.get(
                current_fidelity, 1.0
            ),
        )
        information_value = (
            self.information_value * evidence.information_gain / incremental_cost
        )
        utility = (
            evidence.expected_material_improvement + information_value
        ) / incremental_cost
        promote = (
            evidence.probability_material_improvement >= self.boundary
            or information_value >= self.boundary
        )
        reason = (
            "material-gain probability or decision-relevant information justifies "
            "the next-fidelity cost"
            if promote
            else "next-fidelity cost exceeds expected gain and information value"
        )
        return PromotionDecision(
            promote=promote,
            target_fidelity=target,
            probability_material_improvement=(
                evidence.probability_material_improvement
            ),
            probability_boundary=self.boundary,
            information_value=information_value,
            utility=utility,
            reason=reason,
        )


class InformationGainStrategy:
    """Score uncertainty only when it can affect a downstream choice."""

    def __init__(self, information_weight: float = 1.0):
        self.information_weight = max(0.0, float(information_weight))

    def utility(
        self,
        evidence: EvidenceEstimate,
        *,
        expected_cost: float = 1.0,
        decision_relevance: float = 1.0,
    ) -> float:
        return (
            evidence.expected_material_improvement
            + self.information_weight
            * evidence.information_gain
            * min(1.0, max(0.0, float(decision_relevance)))
        ) / max(float(expected_cost), 1e-12)


@dataclass(frozen=True)
class DiversityAssessment:
    partner_node_id: str
    residual_diversity: float
    blend_gain: float
    information_value: float
    utility: float
    strategy: str = "average"
    weights: tuple[float, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


class DiversityController:
    """Lazily evaluate only plausible OOF pairs; no global pairwise matrix."""

    @staticmethod
    def _aligned_oof(
        run_root: Path, first_node_id: str, second_node_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        frames = []
        for node_id in (first_node_id, second_node_id):
            path = run_root / node_id / "oof_predictions.csv"
            if not path.is_file():
                return None
            frame = pd.read_csv(path)
            required = {"row_id", "target", "prediction"}
            if (
                not required.issubset(frame.columns)
                or frame["row_id"].duplicated().any()
            ):
                return None
            frames.append(
                frame[["row_id", "target", "prediction"]]
                .set_index("row_id")
                .sort_index()
            )
        if not frames[0].index.equals(frames[1].index):
            return None
        targets = frames[0]["target"].to_numpy(dtype=float)
        if not np.array_equal(
            targets, frames[1]["target"].to_numpy(dtype=float)
        ):
            return None
        first = frames[0]["prediction"].to_numpy(dtype=float)
        second = frames[1]["prediction"].to_numpy(dtype=float)
        if not (
            np.isfinite(targets).all()
            and np.isfinite(first).all()
            and np.isfinite(second).all()
        ):
            return None
        return targets, first, second

    def best_partner(
        self,
        *,
        node_id: str,
        node_fidelity: str,
        all_nodes: Mapping[str, Any],
        run_root: Path,
        metric_name: str,
        evidence: EvidenceEstimate,
        excluded_pairs: set[tuple[str, str]] | None = None,
        strategy: str = "auto",
    ) -> DiversityAssessment | None:
        from agents.aggregator_agent import AggregatorAgent

        excluded_pairs = excluded_pairs or set()
        best: DiversityAssessment | None = None
        for partner_id, partner in all_nodes.items():
            pair = tuple(sorted((node_id, partner_id)))
            if (
                partner_id == node_id
                or pair in excluded_pairs
                or getattr(partner, "node_type", None) != "implementation"
                or getattr(partner, "fidelity", None) != node_fidelity
                or not getattr(partner, "result", None)
                or partner.result.get("status") != "completed"
                or partner.result.get("score") is None
            ):
                continue
            aligned = self._aligned_oof(run_root, node_id, partner_id)
            if aligned is None:
                continue
            targets, first, second = aligned
            if np.allclose(first, second, rtol=1e-12, atol=1e-12):
                continue
            first_residual = targets - first
            second_residual = targets - second
            if np.std(first_residual) == 0 or np.std(second_residual) == 0:
                residual_correlation = 1.0
            else:
                residual_correlation = float(
                    np.corrcoef(first_residual, second_residual)[0, 1]
                )
            residual_diversity = min(
                1.0, max(0.0, (1.0 - residual_correlation) / 2.0)
            )
            applied_strategy = AggregatorAgent._resolve_strategy(
                strategy, metric_name
            )
            plan = AggregatorAgent()._oof_plan(
                run_root,
                [node_id, partner_id],
                metric_name,
                strategy=applied_strategy,
            )
            if plan is None:
                continue
            prediction_matrix = np.stack([first, second])
            if applied_strategy == "rank_average":
                prediction_matrix = AggregatorAgent._rank_predictions(
                    prediction_matrix
                )
            weights = np.asarray(plan["weights"], dtype=float)
            ensemble_prediction = weights @ prediction_matrix
            try:
                single_objectives = [
                    AggregatorAgent._validation_metric(
                        targets, prediction, metric_name
                    )
                    for prediction in prediction_matrix
                ]
                ensemble_objective = AggregatorAgent._validation_metric(
                    targets, ensemble_prediction, metric_name
                )
            except ValueError:
                continue
            blend_gain = ensemble_objective - max(single_objectives)
            if (
                np.count_nonzero(weights > 1e-8) < 2
                or blend_gain <= 0.0
            ):
                continue
            information_value = evidence.information_gain * residual_diversity
            utility = blend_gain + information_value
            assessment = DiversityAssessment(
                partner_node_id=partner_id,
                residual_diversity=residual_diversity,
                blend_gain=blend_gain,
                information_value=information_value,
                utility=utility,
                strategy=applied_strategy,
                weights=tuple(float(value) for value in weights),
            )
            if best is None or assessment.utility > best.utility:
                best = assessment
        return best
