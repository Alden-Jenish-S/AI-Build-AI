"""Deterministic model-aware evaluation strategy selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping


EVALUATION_MODES = {
    "cross_validation",
    "holdout",
    "task_native",
}

_MODE_ALIASES = {
    "auto": "",
    "cv": "cross_validation",
    "cross_validation": "cross_validation",
    "cross_validation_oof": "cross_validation",
    "oof": "cross_validation",
    "holdout": "holdout",
    "validation": "holdout",
    "single_holdout": "holdout",
    "native": "task_native",
    "task_native": "task_native",
}

_HOLDOUT_MARKERS = {
    "bert",
    "cnn",
    "deep learning",
    "foundation model",
    "generative",
    "keras",
    "language model",
    "llm",
    "neural",
    "pretrained",
    "resnet",
    "tensorflow",
    "torch",
    "transformer",
    "zero-shot",
}

_CROSS_VALIDATION_MARKERS = {
    "catboost",
    "elastic net",
    "elasticnet",
    "extra trees",
    "gradient boost",
    "histgradientboost",
    "knn",
    "lasso",
    "lightgbm",
    "linear model",
    "logistic regression",
    "naive bayes",
    "out-of-fold",
    "random forest",
    "ridge",
    "sklearn",
    "stacking",
    "support vector",
    "svm",
    "xgboost",
}


@dataclass(frozen=True)
class EvaluationPolicy:
    """The validation protocol a generated implementation must follow."""

    mode: str
    reason: str
    source: str

    @property
    def requires_oof(self) -> bool:
        return self.mode == "cross_validation"

    @property
    def supports_oof_ensemble(self) -> bool:
        return self.requires_oof

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "requires_oof": self.requires_oof,
            "supports_oof_ensemble": self.supports_oof_ensemble,
        }


def normalize_evaluation_mode(value: object, *, allow_auto: bool = True) -> str:
    """Normalize a declared strategy and reject silent misspellings."""
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized not in _MODE_ALIASES:
        raise ValueError(
            "evaluation mode must be one of auto, cross_validation, "
            f"holdout, or task_native; got {value!r}"
        )
    resolved = _MODE_ALIASES[normalized]
    if not resolved and not allow_auto:
        raise ValueError("evaluation mode cannot be auto in this context")
    return resolved


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _declared_mode(
    task_spec: Any,
    technique_record: Mapping[str, Any],
    model_card: Mapping[str, Any],
) -> tuple[str, str] | None:
    output = getattr(task_spec, "output", None)
    task_options = _mapping(getattr(output, "options", {}))
    capabilities = _mapping(model_card.get("capabilities"))
    evaluation = _mapping(model_card.get("evaluation"))
    declarations = (
        ("task output options", task_options.get("evaluation_mode")),
        ("technique record", technique_record.get("evaluation_mode")),
        ("model card evaluation", evaluation.get("mode")),
        ("model capabilities", capabilities.get("evaluation_mode")),
    )
    for source, raw_mode in declarations:
        if raw_mode is None:
            continue
        mode = normalize_evaluation_mode(raw_mode)
        if mode:
            return mode, source
    return None


def select_evaluation_policy(
    task_spec: Any,
    technique_record: Mapping[str, Any] | None = None,
    model_card: Mapping[str, Any] | None = None,
    *,
    operator: str | None = None,
) -> EvaluationPolicy:
    """Select OOF only when the task and model make it useful and feasible.

    Explicit task/model declarations win. Otherwise unsupervised methods use
    their native evaluator, costly/non-independent neural methods use one
    harness-owned holdout, and inexpensive classical supervised estimators use
    OOF cross-validation. Unknown models conservatively use holdout.
    """
    technique = _mapping(technique_record)
    card = _mapping(model_card)
    problem_type = str(
        getattr(task_spec, "problem_type", "") or ""
    ).strip().lower()

    declared = _declared_mode(task_spec, technique, card)
    if declared is not None:
        mode, source = declared
        if problem_type == "unsupervised_clustering" and mode != "task_native":
            return EvaluationPolicy(
                mode="task_native",
                reason=(
                    "unsupervised clustering has no supervised OOF target; "
                    "using its task-native internal validation"
                ),
                source="task objective",
            )
        return EvaluationPolicy(
            mode=mode,
            reason=f"evaluation mode explicitly declared by {source}",
            source=source,
        )

    if problem_type == "unsupervised_clustering":
        return EvaluationPolicy(
            mode="task_native",
            reason=(
                "unsupervised clustering is evaluated with a task-native "
                "internal metric rather than supervised OOF"
            ),
            source="task objective",
        )

    capabilities = _mapping(card.get("capabilities"))
    if capabilities.get("requires_oof") is True:
        return EvaluationPolicy(
            mode="cross_validation",
            reason="the selected model explicitly requires OOF predictions",
            source="model capabilities",
        )
    if capabilities.get("supports_cross_validation") is False:
        return EvaluationPolicy(
            mode="holdout",
            reason="the selected model explicitly disallows cross-validation",
            source="model capabilities",
        )
    if capabilities.get("accepts_harness_fold_ids") is True:
        return EvaluationPolicy(
            mode="cross_validation",
            reason=(
                "the verified artifact explicitly consumes harness fold IDs "
                "and can produce valid OOF predictions"
            ),
            source="model capabilities",
        )

    searchable = json.dumps(
        {
            "operator": operator,
            "plan": technique.get("plan"),
            "name": technique.get("name"),
            "category": card.get("category"),
            "artifact_id": card.get("artifact_id"),
            "description": card.get("description"),
            "interface": card.get("interface"),
        },
        default=str,
    ).lower()
    holdout_hits = sorted(
        marker for marker in _HOLDOUT_MARKERS if marker in searchable
    )
    if holdout_hits:
        return EvaluationPolicy(
            mode="holdout",
            reason=(
                "the selected model is expensive or not naturally "
                f"fold-independent ({', '.join(holdout_hits[:3])})"
            ),
            source="model-family inference",
        )
    cv_hits = sorted(
        marker
        for marker in _CROSS_VALIDATION_MARKERS
        if marker in searchable
    )
    if cv_hits and problem_type in {
        "classification",
        "multilabel_classification",
        "regression",
        "supervised",
    }:
        return EvaluationPolicy(
            mode="cross_validation",
            reason=(
                "the selected supervised estimator is inexpensive and "
                f"fold-independent ({', '.join(cv_hits[:3])})"
            ),
            source="model-family inference",
        )

    return EvaluationPolicy(
        mode="holdout",
        reason=(
            "the model does not declare safe fold-independent training; "
            "using one harness-owned holdout avoids forcing OOF"
        ),
        source="conservative default",
    )
