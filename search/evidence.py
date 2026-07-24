"""Statistical evidence used by pruning, promotion, and action policies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Iterable


_STANDARD_NORMAL = NormalDist()


def _finite_floats(values: Iterable[Any] | None) -> list[float]:
    clean: list[float] = []
    for value in values or []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    return clean


@dataclass(frozen=True)
class EvidenceEstimate:
    """Uncertainty-aware estimate of a direction-normalized score improvement."""

    candidate_score: float
    reference_score: float
    delta_mean: float
    standard_error: float
    minimum_worthwhile_effect: float
    probability_improvement: float
    probability_material_improvement: float
    expected_material_improvement: float
    information_gain: float
    paired_observations: int
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


class EvidenceService:
    """Build paired evidence without hard-coded metric-scale score thresholds."""

    @staticmethod
    def _probability_above(mean: float, standard_error: float, boundary: float) -> float:
        if standard_error <= 1e-15:
            if mean > boundary:
                return 1.0
            if mean < boundary:
                return 0.0
            return 0.5
        z = (mean - boundary) / standard_error
        return min(1.0, max(0.0, _STANDARD_NORMAL.cdf(z)))

    @staticmethod
    def _positive_part_expectation(
        mean: float, standard_error: float, boundary: float
    ) -> float:
        shifted = mean - boundary
        if standard_error <= 1e-15:
            return max(0.0, shifted)
        z = shifted / standard_error
        density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return max(
            0.0,
            shifted * _STANDARD_NORMAL.cdf(z) + standard_error * density,
        )

    @staticmethod
    def _binary_entropy(probability: float) -> float:
        p = min(1.0 - 1e-15, max(1e-15, float(probability)))
        return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

    def compare(
        self,
        candidate: dict,
        reference: dict,
        *,
        direction: str,
        minimum_worthwhile_effect: float | None = None,
    ) -> EvidenceEstimate:
        """Compare results on paired folds when available.

        The candidate and reference mappings may be NodeState results, result.json
        payloads, or compact ``{"score": ..., "validation": ...}`` records.
        """
        if direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be 'maximize' or 'minimize'")
        sign = 1.0 if direction == "maximize" else -1.0

        candidate_validation = candidate.get("validation") or candidate
        reference_validation = reference.get("validation") or reference
        candidate_score = float(
            candidate.get("score", candidate_validation.get("cv_mean"))
        )
        reference_score = float(
            reference.get("score", reference_validation.get("cv_mean"))
        )
        if not math.isfinite(candidate_score) or not math.isfinite(reference_score):
            raise ValueError("candidate and reference scores must be finite")

        candidate_folds = _finite_floats(candidate_validation.get("fold_scores"))
        reference_folds = _finite_floats(reference_validation.get("fold_scores"))
        deltas: list[float] = []
        method = "independent_normal"
        candidate_fidelity = candidate_validation.get("fidelity")
        reference_fidelity = reference_validation.get("fidelity")
        candidate_split = candidate_validation.get("fold_assignment_sha256")
        reference_split = reference_validation.get("fold_assignment_sha256")
        aligned_protocol = not (
            candidate_fidelity
            and reference_fidelity
            and candidate_fidelity != reference_fidelity
        ) and not (
            candidate_split
            and reference_split
            and candidate_split != reference_split
        )
        if (
            len(candidate_folds) >= 2
            and len(candidate_folds) == len(reference_folds)
            and aligned_protocol
        ):
            deltas = [
                sign * (candidate_value - reference_value)
                for candidate_value, reference_value in zip(
                    candidate_folds, reference_folds
                )
            ]
            method = "paired_folds"
        elif not aligned_protocol:
            method = "independent_cross_protocol"

        delta_mean = sign * (candidate_score - reference_score)
        if deltas:
            # Preserve the reported aggregate delta while estimating its sampling
            # uncertainty from matched folds.
            sample_mean = sum(deltas) / len(deltas)
            variance = sum((value - sample_mean) ** 2 for value in deltas) / (
                len(deltas) - 1
            )
            standard_error = math.sqrt(max(0.0, variance) / len(deltas))
            paired_observations = len(deltas)
        else:
            candidate_std = max(
                0.0, float(candidate_validation.get("cv_std", 0.0) or 0.0)
            )
            reference_std = max(
                0.0, float(reference_validation.get("cv_std", 0.0) or 0.0)
            )
            candidate_count = max(
                1, int(candidate_validation.get("folds", len(candidate_folds) or 1))
            )
            reference_count = max(
                1, int(reference_validation.get("folds", len(reference_folds) or 1))
            )
            standard_error = math.sqrt(
                candidate_std * candidate_std / candidate_count
                + reference_std * reference_std / reference_count
            )
            paired_observations = min(candidate_count, reference_count)

        # A worthwhile effect must clear the observed measurement noise. Callers
        # may supply a domain/cost-derived value, but there is no metric-wide fixed
        # score gap.
        material_effect = (
            max(0.0, float(minimum_worthwhile_effect))
            if minimum_worthwhile_effect is not None
            else standard_error
        )
        p_improve = self._probability_above(delta_mean, standard_error, 0.0)
        p_material = self._probability_above(
            delta_mean, standard_error, material_effect
        )
        return EvidenceEstimate(
            candidate_score=candidate_score,
            reference_score=reference_score,
            delta_mean=delta_mean,
            standard_error=standard_error,
            minimum_worthwhile_effect=material_effect,
            probability_improvement=p_improve,
            probability_material_improvement=p_material,
            expected_material_improvement=self._positive_part_expectation(
                delta_mean, standard_error, material_effect
            ),
            information_gain=self._binary_entropy(p_material),
            paired_observations=paired_observations,
            method=method,
        )
