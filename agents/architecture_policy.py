"""Small, auditable policy helpers for model-family and architecture coverage."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


_CUSTOM_NEURAL_PATTERNS = (
    r"\bcustom_neural\b",
    r"\bcustom (?:neural )?(?:network|architecture|module|block)\b",
    r"\btask[- ](?:invented|tailored) (?:network|architecture)\b",
    r"\bfrom (?:primitive|first[- ]principles) (?:layers|operations|blocks)\b",
    r"\blearned (?:gating|interaction|routing) (?:block|module|layer)\b",
)
_NEURAL_PATTERNS = (
    *_CUSTOM_NEURAL_PATTERNS,
    r"\bestablished_neural\b",
    r"\bnn\.module\b",
    r"\bclass\s+\w+\s*\(\s*(?:torch\.)?nn\.module",
    r"\bneural (?:network|model|architecture)\b",
    r"\b(?:pytorch|torch|tensorflow|keras|jax)\b",
    r"\b(?:mlp|multilayer perceptron|tabnet|tabtransformer|ft[- ]transformer)\b",
    r"\b(?:transformer|attention|embedding|convolutional|recurrent) (?:network|model|layer|block)\b",
    r"\b(?:cnn|rnn|lstm|gru)\b",
)
_CONVENTIONAL_PATTERNS = (
    r"\b(?:lightgbm|lgbm|xgboost|catboost)\b",
    r"\bhist(?:ogram)?_?\s*gradient_?\s*boost",
    r"\bhistgradientboost",
    r"\bgradient[- ]boost(?:ed|ing)? (?:tree|trees|model)\b",
    r"\b(?:random forest|extra trees|logistic regression|naive bayes|support vector|svm)\b",
    r"\b(?:randomforest|extratrees|logisticregression|naivebayes)\w*\b",
)


def _matches(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def classify_architecture(text: object) -> str:
    """Return a coarse exploration track without pretending to infer full semantics."""
    normalized = " ".join(str(text or "").split())
    if _matches(_CUSTOM_NEURAL_PATTERNS, normalized):
        return "custom_neural"
    if _matches(_NEURAL_PATTERNS, normalized):
        return "established_neural"
    if _matches(_CONVENTIONAL_PATTERNS, normalized):
        return "conventional"
    return "other"


def hypothesis_text(hypothesis: Mapping[str, Any]) -> str:
    """Build the bounded text used to classify a council hypothesis."""
    fields = (
        "title",
        "model_family",
        "experiment",
        "architecture_spec",
    )
    return "\n".join(str(hypothesis.get(field) or "") for field in fields)


def annotate_hypothesis(hypothesis: dict[str, Any]) -> dict[str, Any]:
    """Attach a deterministic architecture track to an LLM-produced hypothesis."""
    predicted = classify_architecture(hypothesis_text(hypothesis))
    declared = str(hypothesis.get("architecture_track") or "").strip().casefold()
    allowed = {
        "conventional",
        "established_neural",
        "custom_neural",
        "representation",
        "hybrid",
        "other",
    }
    # Deterministic text evidence wins when it identifies a known track. This
    # prevents a vague or mistaken declared label from distorting coverage.
    if predicted != "other":
        hypothesis["architecture_track"] = predicted
    elif declared in allowed:
        hypothesis["architecture_track"] = declared
    else:
        hypothesis["architecture_track"] = predicted
    return hypothesis


def coverage_from_texts(texts: Iterable[object]) -> dict[str, Any]:
    return coverage_from_tracks(classify_architecture(text) for text in texts)


def coverage_from_tracks(tracks: Iterable[object]) -> dict[str, Any]:
    normalized = [str(track) for track in tracks]
    counts = {
        track: normalized.count(track)
        for track in ("conventional", "established_neural", "custom_neural", "other")
    }
    return {
        "tracks": normalized,
        "counts": counts,
        "neural_attempted": bool(
            counts["established_neural"] or counts["custom_neural"]
        ),
        "custom_neural_attempted": bool(counts["custom_neural"]),
    }
