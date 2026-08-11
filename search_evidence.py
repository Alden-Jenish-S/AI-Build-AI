"""Shared search evidence helpers: prediction signatures, noise estimates,
and model-family fingerprints.

Everything here is task-type agnostic: each helper degrades gracefully when
the underlying evidence (supervised validation set, per-fold scores, estimator
names) does not exist for a task (RL, generation, control, ...).
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

from agents.architecture_policy import classify_architecture

SIGNATURE_MIN_LEN = 8
SIGNATURE_MAX_LEN = 8192

_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("linear", r"\b(?:logistic\s*regression|linear\s*regression|ridgeclassifier|ridgeregressor|lassocv|lasso|elasticnet|sgd(?:classifier|regressor)|linearmodel)\b"),
    ("svm", r"\b(?:support\s*vector|svc\b|svr\b|libsvm|linearsvc)\b"),
    ("knn", r"\b(?:k[- ]?nearest(?:[- ]neighbors?)?|nearest[- ]neighbors?|kneighbors)\b"),
    ("naive_bayes", r"\bnaive\s*bayes|gaussiannb|multinomialnb|bernoullinb|complementnb\b"),
    ("decision_tree", r"\bdecision\s*tree|decisiontree|cart\b"),
    ("random_forest", r"\brandom\s*forest|randomforest\b"),
    ("extra_trees", r"\bextra(?:m)?\s*trees|extratrees\b"),
    ("gbdt", r"\b(?:gradient\s*boost(?:ed|ing)?\s*trees?|gbdt|histgradientboosting|hist\s*gradient\s*boosting)\b"),
    ("lightgbm", r"\blightgbm|lgbm\b"),
    ("xgboost", r"\bxgboost|xgb(?:classifier|regressor)?\b"),
    ("catboost", r"\bcatboost\b"),
    ("adaboost", r"\badaboost\b"),
    ("bagging", r"\bbagging|gradient\s*boosting\s*regressor\b"),
    ("mlp", r"\b(?:mlp|multilayer\s*perceptron|multi-?layer\s*perceptron)\b"),
    ("cnn", r"\b(?:cnn|convolutional(?:l)?\s*neural|convnet)\b"),
    ("transformer", r"\b(?:transformer|self[- ]attention|attention\s*mechanism|multi[- ]head\s*attention)\b"),
    ("recurrent", r"\b(?:rnn|lstm|gru|recurrent(?:l)?\s*neural)\b"),
    ("tabular_arch", r"\btabnet|ft[- ]transformer|tabtransformer\b"),
    ("autoencoder", r"\bautoencoder|vae|variational\s*autoencoder\b"),
    ("gan", r"\bgan\b|generative\s*adversarial"),
    ("diffusion", r"\bdiffusion\b"),
    ("clustering", r"\bkmeans|k[- ]means|gaussian\s*mixture|dbscan|hierarchical\s*clustering\b"),
    ("kde", r"\bkernel\s*density|kde\b"),
    ("forecast", r"\barima|sarima|prophet|holt[- ]winters|ets\b"),
    ("graph", r"\bgraph\s*neural|gnn|message\s*passing|gcn\b"),
    ("rl", r"\breinforcement\s*learning|ppo\b|dqn\b|sac\b|td3\b|a2c\b|policy\s*gradient\b"),
    ("resnet", r"\bresnet\b"),
    ("efficientnet", r"\befficientnet\b"),
    ("densenet", r"\bdensenet\b"),
    ("vit", r"\bvit\b"),
    ("bert", r"\b(?:bert|roberta)\b"),
    ("wav2vec", r"\bwav2vec\b"),
    ("whisper", r"\bwhisper\b"),
    ("foundation", r"\b(?:word2vec|fasttext|glove\b|sentence[- ]transformer)\b"),
    ("polynomial", r"\bpolynomial\s*(?:features?|regression)|kernel\s*ridge\b"),
    ("metric_learning", r"\bsiamese|triplet\s*loss|contrastive\s*learning\b"),
)


def valid_signature(value: object) -> bool:
    """Return whether a value is a usable fixed-size prediction signature."""
    if (
        not isinstance(value, list)
        or len(value) < SIGNATURE_MIN_LEN
        or len(value) > SIGNATURE_MAX_LEN
    ):
        return False
    return all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        for item in value
    )


def signature_from_result(result: dict[str, Any] | None) -> list[float] | None:
    """Extract a prediction signature from a node result, or None when absent."""
    signature = (result or {}).get("prediction_signature")
    if not valid_signature(signature):
        return None
    return [float(item) for item in signature]


def pearson_correlation(left: Iterable[float], right: Iterable[float]) -> float:
    """Pearson correlation with safe degradation to 0.0."""
    left_list = list(left)
    right_list = list(right)
    if len(left_list) != len(right_list) or len(left_list) < SIGNATURE_MIN_LEN:
        return 0.0
    count = len(left_list)
    mean_left = sum(left_list) / count
    mean_right = sum(right_list) / count
    covariance = sum(
        (left_value - mean_left) * (right_value - mean_right)
        for left_value, right_value in zip(left_list, right_list)
    )
    var_left = sum((value - mean_left) ** 2 for value in left_list)
    var_right = sum((value - mean_right) ** 2 for value in right_list)
    if var_left <= 0.0 or var_right <= 0.0:
        return 0.0
    value = covariance / math.sqrt(var_left * var_right)
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def _finite_floats(values: object) -> list[float]:
    if not isinstance(values, (list, tuple)):
        return []
    collected: list[float] = []
    for item in values:
        if isinstance(item, bool):
            continue
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            collected.append(value)
    return collected


def score_noise_estimate(result: dict[str, Any] | None) -> float | None:
    """Estimate evaluation noise from fold or repeated-seed scores.

    Returns None when the result carries no usable dispersion evidence
    (e.g. holdout/task-native runs without repeated seeds).
    """
    for key in ("fold_scores", "seed_scores"):
        values = _finite_floats((result or {}).get(key))
        if len(values) >= 2:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            if variance > 0.0:
                return math.sqrt(variance)
    return None


def relative_noise_floor(score: float) -> float:
    """Conservative floor for tasks without fold/seed dispersion evidence."""
    return max(1e-6, abs(score) * 1.5e-3)


def estimator_families(text: object) -> list[str]:
    """Return the sorted set of model-family names mentioned in a text."""
    normalized = " ".join(str(text or "").split())
    found: list[str] = []
    for name, pattern in _FAMILY_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            found.append(name)
    return sorted(set(found))


def family_fingerprint(text: object) -> str:
    """Compact fingerprint of a model family described by a plan or code.

    An empty string means the text carried no usable family evidence; such
    fingerprints never collide and never guard anything.
    """
    families = estimator_families(text)
    if not families:
        match = re.search(r"(?im)^\s*(?:Model family|Model):\s*(.+?)\s*$", str(text or ""))
        if match:
            families = [match.group(1).strip().casefold()]
    track = classify_architecture(text)
    if not families and track == "other":
        return ""
    return f"{track}|{','.join(families)}"