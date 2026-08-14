"""Generate, run, and repair one self-contained task implementation."""

from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime_utils import (
    expose_task_data,
    run_supervised_process,
    sanitized_subprocess_env,
    task_dir_snapshot,
    verify_task_dir_unchanged,
)
from search_evidence import valid_signature
from prediction_evidence import inspect_prediction_evidence

from .council.contracts import CouncilBrief, EvaluationProtocol
from .llm_utils import call_llm
from .modality_policy import (
    predictive_modality_inventory,
    validate_modality_ablation_report,
)
from .submission_validator import SubmissionValidator
from .task_analyzer import TaskAnalysis
from .web_search import search_web


def _parses_as_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _trim_python_tail(code: str, max_attempts: int = 250) -> str:
    """Drop trailing prose lines until the code parses (or give up unchanged)."""
    if not code.strip() or _parses_as_python(code):
        return code
    lines = code.splitlines()
    for cut in range(1, min(max_attempts, len(lines)) + 1):
        candidate = "\n".join(lines[:-cut])
        if not candidate.strip():
            break
        if _parses_as_python(candidate):
            return candidate
    return code


def _trim_python_head(code: str, max_attempts: int = 150) -> str:
    """Drop leading prose lines until the code parses (or give up unchanged)."""
    if not code.strip() or _parses_as_python(code):
        return code
    lines = code.splitlines()
    for skip in range(1, min(max_attempts, len(lines)) + 1):
        candidate = "\n".join(lines[skip:])
        if not candidate.strip():
            break
        if _parses_as_python(candidate):
            return candidate
    return code


def _python_source(response: str) -> str:
    text = re.sub(r"(?s)<thinking>.*?</thinking>", "", response).strip()
    candidates: list[str] = []
    for match in re.finditer(
        r"```(?:python|py)?[^\n]*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    ):
        candidates.append(match.group(1).strip())
    opener = re.search(r"```(?:python|py)?[^\n]*\n", text)
    if opener is not None and not candidates:
        candidates.append(text[opener.end():].strip())
    if candidates:
        for candidate in sorted((item for item in candidates if item), key=len, reverse=True):
            fixed = _trim_python_tail(candidate)
            fixed = _trim_python_head(fixed)
            if fixed.strip():
                return fixed + "\n"
        return candidates[0] + "\n"
    text = _trim_python_tail(text)
    text = _trim_python_head(text)
    return text.strip() + "\n"


def _tail(text: str, limit: int = 7000) -> str:
    return text if len(text) <= limit else text[-limit:]


def _compact_code(code: str, limit: int) -> str:
    """Retain imports/setup and the final pipeline/output logic."""
    if len(code) <= limit:
        return code
    head = max(2000, limit // 3)
    return code[:head] + "\n# ... middle omitted from prompt ...\n" + code[-(limit - head):]


def _architecture_source_errors(code: str, operator: str = "architect") -> list[str]:
    """Reject an `architect` response that silently falls back to a library model."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"architecture source is not valid Python: {exc}"]
    imported_roots: set[str] = set()
    identifiers: set[str] = set()
    module_classes = 0
    has_forward = False
    has_custom_composition = False
    forward_functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    composition_calls = {
        "bmm",
        "cat",
        "einsum",
        "matmul",
        "sigmoid",
        "softmax",
        "stack",
    }
    composition_names = (
        "attention",
        "cross",
        "gate",
        "gating",
        "interaction",
        "mixer",
        "residual",
        "router",
        "skip",
    )

    def dotted(node: ast.AST) -> str:
        parts: list[str] = []
        current: ast.AST | None = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".", 1)[0].casefold())
                identifiers.add(alias.name.casefold())
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_roots.add(node.module.split(".", 1)[0].casefold())
                identifiers.add(node.module.casefold())
            identifiers.update(alias.name.casefold() for alias in node.names)
        elif isinstance(node, (ast.Name, ast.Attribute)):
            name = dotted(node) if isinstance(node, ast.Attribute) else node.id
            identifiers.add(name.casefold())
        elif isinstance(node, ast.ClassDef):
            identifiers.add(node.name.casefold())
            bases = {dotted(base).casefold() for base in node.bases}
            if bases.intersection({"nn.module", "torch.nn.module"}):
                module_classes += 1
                forwards = [
                    item
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "forward"
                ]
                if forwards:
                    has_forward = True
                    forward_functions.extend(forwards)

    for forward in forward_functions:
        for node in ast.walk(forward):
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = dotted(node) if isinstance(node, ast.Attribute) else node.id
                if any(marker in name.casefold() for marker in composition_names):
                    has_custom_composition = True
            elif isinstance(node, ast.Call):
                called = dotted(node.func).rsplit(".", 1)[-1].casefold()
                if called in composition_calls:
                    has_custom_composition = True
            elif isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Mult, ast.MatMult)
            ):
                has_custom_composition = True

    errors: list[str] = []
    if "torch" not in imported_roots:
        errors.append("the custom candidate does not import PyTorch")
    if module_classes < 1 or not has_forward:
        errors.append("the custom candidate has no explicit nn.Module subclass with forward()")

    if operator == "transfer":
        return errors

    if not has_custom_composition:
        errors.append(
            "the network exposes no custom interaction, gating, routing, residual, "
            "branch-composition, or tensor-composition mechanism beyond a plain layer stack"
        )
    prohibited = ("tabnet", "fttransformer", "ft_transformer", "tabtransformer")
    used_prohibited = sorted(
        marker for marker in prohibited if any(marker in name for name in identifiers)
    )
    if used_prohibited:
        errors.append(
            "the primary architecture uses prohibited named model identifier(s): "
            + ", ".join(used_prohibited)
        )
    return errors


_STDOUT_SCORE_PATTERNS: dict[str, str] = {
    "accuracy": r"accuracy",
    "roc_auc": r"\bauc\b",
    "auc": r"\bauc\b",
    "f1": r"\bf1\b",
    "dice": r"\bdice\b",
    "iou": r"\biou\b",
    "silhouette": r"silhouette",
    "log_loss": r"log[\s_-]?loss",
    "cross_entropy": r"cross[\s_-]?entropy",
    "rmse": r"\brmse\b",
    "mae": r"\bmae\b",
    "mean_average_precision": r"\bmap\b",
    "adjusted_rand_index": r"adjusted[\s_-]?rand",
}


def _metric_stdout_pattern(metric: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(metric or "").strip().lower()).strip("_")
    for key, pattern in _STDOUT_SCORE_PATTERNS.items():
        if key in normalized or normalized in key:
            return pattern
    return None


def _metric_name_token(value: object) -> str:
    """Canonical token for comparing metric names across write aliases.

    Generated programs legitimately write ``log_loss`` while the council
    protocol spells it ``log loss``; both must compare equal.
    """
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _parse_stdout_score(stdout: str, pattern: str) -> float | None:
    matches = re.findall(
        rf"(?i){pattern}\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        stdout,
    )
    if not matches:
        return None
    try:
        score = float(matches[-1])
        return score if math.isfinite(score) else None
    except ValueError:
        return None


def _score_from_stdout(stdout: str, metric: str | None, direction: str) -> float | None:
    """Extract the last plausible score from program output."""
    if metric:
        metric_pattern = _metric_stdout_pattern(metric)
        if metric_pattern is not None:
            candidate = _parse_stdout_score(stdout, metric_pattern)
            if candidate is not None:
                return candidate
        
        # Fallback to literal metric name or generic "score"
        normalized_metric = re.escape(str(metric).strip().lower())
        pattern = rf"(?:score|{normalized_metric})"
        candidate = _parse_stdout_score(stdout, pattern)
        if candidate is not None:
            return candidate
        return None

    # Generic fallback only if no metric was provided
    pattern = r"(?:score|auc|accuracy|f1|dice|iou|silhouette)"
    if direction != "maximize":
        pattern = r"(?:score|auc|accuracy|f1|dice|iou|silhouette|rmse|mae|loss)"
    return _parse_stdout_score(stdout, pattern)


def _fold_mean_consistency(score: float, fold_scores: list[Any]) -> bool:
    """Return whether ``score`` agrees with the mean of the reported fold scores."""
    try:
        numeric = [float(value) for value in fold_scores]
    except (TypeError, ValueError):
        return False
    if not numeric or not all(math.isfinite(value) for value in numeric):
        return False
    mean = sum(numeric) / len(numeric)
    tolerance = max(1e-3, abs(score) * 0.01)
    return abs(score - mean) <= tolerance


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def _plan_requests_modality_ablation(plan: str | None) -> bool:
    """Return whether this specific plan requests cross-modality controls."""
    text = str(plan or "")
    scope = re.search(
        r"(?im)^\s*modality\s+scope\s*:\s*([a-z_ -]+)", text
    )
    if scope is not None:
        normalized_scope = re.sub(
            r"[^a-z]+", "_", scope.group(1).casefold()
        ).strip("_")
        return normalized_scope in {"fused_multimodal", "modality_ablation"}
    normalized = " ".join(re.sub(r"[^a-z]+", " ", text.casefold()).split())
    return any(
        marker in normalized
        for marker in ("leave one modality out", "full fusion", "modality contribution")
    )




class ImplementationAgent:
    """One compact code-writing call followed by bounded execution repairs."""

    def __init__(
        self,
        venv_python_path: str | None = None,
        model_name: str | None = None,
        submission_validator: SubmissionValidator | None = None,
    ) -> None:
        self.python = str(venv_python_path or sys.executable)
        self.model_name = model_name
        self.submission_validator = submission_validator or SubmissionValidator()

    @staticmethod
    def _log(node_name: str, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{timestamp}] ImplementationAgent[{node_name}]: {message}", flush=True)

    @staticmethod
    def _output_path(node_dir: Path) -> Path | None:
        preferred = node_dir / "submission" / "submission.csv"
        root_csv = node_dir / "submission.csv"
        if preferred.is_file():
            return preferred
        if root_csv.is_file():
            preferred.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root_csv, preferred)
            return preferred
        submission_dir = node_dir / "submission"
        if submission_dir.is_dir():
            files = [path for path in sorted(submission_dir.rglob("*")) if path.is_file()]
            if files:
                return submission_dir
        return None

    @staticmethod
    def _read_result(node_dir: Path) -> dict[str, Any]:
        path = node_dir / "result.json"
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _validate_reported_evaluation(
        payload: dict[str, Any], protocol: EvaluationProtocol
    ) -> list[str]:
        """Reject node scores that are not comparable under the council protocol."""
        errors: list[str] = []
        for field in protocol.required_result_fields:
            if field not in payload:
                errors.append(f"result.json is missing required field {field!r}")
        if str(payload.get("evaluation_protocol_hash", "")) != protocol.protocol_hash:
            errors.append("evaluation_protocol_hash does not match evaluation_protocol.json")
        if _metric_name_token(payload.get("metric")) != _metric_name_token(protocol.metric):
            errors.append(
                f"reported metric {payload.get('metric')!r} does not match {protocol.metric!r}"
            )
        if str(payload.get("direction", "")).strip().casefold() != protocol.direction:
            errors.append(
                f"reported direction {payload.get('direction')!r} does not match {protocol.direction!r}"
            )
        reported_score = payload.get("score")
        try:
            if isinstance(reported_score, bool):
                raise ValueError("boolean score")
            score = float(str(reported_score))
            if not math.isfinite(score):
                errors.append("reported score is not finite")
        except (TypeError, ValueError):
            errors.append("reported score is not numeric")
        validation_count = payload.get("validation_sample_count")
        if (
            not isinstance(validation_count, int)
            or isinstance(validation_count, bool)
            or validation_count < 1
        ):
            errors.append("validation_sample_count must be a positive integer")
        fold_scores = payload.get("fold_scores")
        if not isinstance(fold_scores, list) or len(fold_scores) != protocol.folds:
            errors.append(
                f"fold_scores must contain exactly {protocol.folds} numeric value(s)"
            )
        else:
            try:
                if not all(
                    not isinstance(value, bool) and math.isfinite(float(value))
                    for value in fold_scores
                ):
                    errors.append("fold_scores contains a non-finite value")
                elif isinstance(reported_score, (int, float, str)) and not isinstance(
                    reported_score, bool
                ):
                    try:
                        numeric_score = float(str(reported_score))
                    except (TypeError, ValueError):
                        numeric_score = math.nan
                    if math.isfinite(numeric_score) and not _fold_mean_consistency(
                        numeric_score, fold_scores
                    ):
                        errors.append(
                            "reported score does not match the mean of the reported fold_scores"
                        )
            except (TypeError, ValueError):
                errors.append("fold_scores must contain only numeric values")
        signature = payload.get("prediction_signature")
        if signature is not None and not valid_signature(signature):
            errors.append(
                "prediction_signature must be a list of finite floats "
                "between 8 and 8192 entries"
            )
        seed_scores = payload.get("seed_scores")
        if seed_scores is not None:
            seed_values: list[float] = []
            if isinstance(seed_scores, list):
                for value in seed_scores:
                    if isinstance(value, bool):
                        continue
                    try:
                        seed_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(seed_value):
                        seed_values.append(seed_value)
            if len(seed_values) < 2:
                errors.append("seed_scores must contain at least 2 finite values")
        return errors

    def _prompt(
        self,
        analysis: TaskAnalysis,
        plan: str,
        parent_code: str | None,
        companion_code: str | None,
        candidate_code: str | None,
        feedback: str,
        web_notes: str,
        council_brief: CouncilBrief | None,
        operator: str,
        *,
        probe: bool = False,
        abort_context: dict[str, Any] | None = None,
        tune_search: bool = False,
    ) -> str:
        parent = ""
        if parent_code:
            if operator in ("architect", "transfer"):
                parent = (
                    "\nA working measured control is included below. Reuse its proven input "
                    "discovery, preprocessing contract, validation indices, metric, and output "
                    "behavior, but DO NOT retain its conventional estimator as the primary "
                    "candidate. The architecture/transfer experiment must train and evaluate the custom "
                    "neural predictor requested by the plan. Keep the parent only as the reported "
                    "control/fallback.\n"
                    f"```python\n{_compact_code(parent_code, 20000)}\n```\n"
                )
            else:
                parent = (
                    "\nA working parent implementation is included below. Preserve its working "
                    "paths and output behavior; edit it narrowly for this plan.\n"
                    f"```python\n{_compact_code(parent_code, 20000)}\n```\n"
                )
        companion = ""
        if companion_code:
            companion = (
                "\nA second measured implementation is included for the selected merge. "
                "Use only complementary logic from it and compare the merged candidate against "
                "both measured parents.\n"
                f"```python\n{_compact_code(companion_code, 14000)}\n```\n"
            )
        repair = ""
        if feedback:
            repair = (
                "\nThe previous execution did not finish correctly. Repair the actual error; "
                "do not invent imports from evaluation, contracts, loaders, or this agent system.\n"
                + (
                    "Previous attempted program:\n"
                    f"```python\n{_compact_code(candidate_code, 16000)}\n```\n"
                    if candidate_code
                    else ""
                )
                + f"Execution output:\n{feedback}\n"
            )
        research = f"\nRelevant web-search notes & literature insights:\n{web_notes[:5000]}\n" if web_notes else ""
        runtime_inventory = (
            list(council_brief.allowed_input_paths)
            if council_brief is not None
            else [str(item["path"]) for item in analysis.files]
        )
        runtime_paths = [f"input/{path}" for path in runtime_inventory]
        modality_inventory = predictive_modality_inventory(analysis.files)
        modalities = [str(item) for item in modality_inventory["modalities"]]
        requires_modality_ablation = bool(
            modality_inventory["is_multimodal"]
            and _plan_requests_modality_ablation(plan)
        )
        displayed_paths: list[str] = []
        displayed_chars = 0
        for path in runtime_paths:
            rendered = f"- `{path}`"
            if displayed_paths and displayed_chars + len(rendered) + 1 > 6000:
                break
            displayed_paths.append(rendered)
            displayed_chars += len(rendered) + 1
        exact_paths = "\n".join(displayed_paths)
        if len(runtime_paths) > len(displayed_paths):
            exact_paths += (
                f"\n- … {len(runtime_paths) - len(displayed_paths)} more paths are described "
                "by the inventory above"
            )
        council = ""
        result_contract = (
            "At the end, write a small `result.json` object containing `score`, "
            "`metric`, `direction`, and `diagnostics`. Include `output` when a "
            "task-native deliverable was materialized."
        )
        if council_brief is not None:
            protocol = council_brief.evaluation_protocol
            hypothesis_match = re.search(
                r"(?im)^\s*Hypothesis ID:\s*([A-Za-z0-9_.-]+)", plan
            )
            active_hypothesis_id = (
                hypothesis_match.group(1) if hypothesis_match is not None else None
            )
            council = f"""
ML RESEARCH COUNCIL CONTRACT (authoritative):
{council_brief.implementation_context(active_hypothesis_id, 6000)}

Only the exact input paths listed below are exposed. Prohibited inputs are unavailable
by design. Follow `evaluation_protocol.json` exactly; do not choose a different split,
seed, fold count, leakage unit, metric, or direction. A score measured under a different
protocol is invalid even if it is numerically better.
"""
            result_contract = (
                "At the end, write `result.json` containing `score`, `metric`, `direction`, "
                "`diagnostics`, `evaluation_protocol_hash`, `fold_scores`, and "
                "`validation_sample_count`. Set `evaluation_protocol_hash` to "
                f"`{protocol.protocol_hash}`. `fold_scores` must have exactly "
                f"{protocol.folds} numeric value(s), and `validation_sample_count` must be positive."
            )
            result_contract += (
                " Optionally report `status` ('completed' or 'truncated'), "
                "`prediction_signature` (up to 2048 finite floats), and `seed_scores` "
                "(at least 2 finite floats when using repeated-seed evaluations)."
            )
        if requires_modality_ablation:
            result_contract += (
                " Also include `modality_ablation_scores`, a list of objects containing "
                "`modalities`, finite `score`, `fold_scores`, and a shared "
                "`validation_indices_hash` for the all-modality model, "
                "single-modality controls, and leave-one-modality-out comparisons."
            )
        architecture_contract = ""
        if operator == "architect":
            architecture_contract = """
ARCHITECTURE EXPERIMENT CONTRACT (mandatory):
- Implement the primary candidate as an explicit custom PyTorch `nn.Module` composed
  from primitive layers/tensor operations and derived from the observed data geometry.
- Do not substitute LightGBM, XGBoost, CatBoost, HistGradientBoosting, TabNet,
  FT-Transformer, TabTransformer, or a pretrained model for the requested candidate.
- A plain MLP may appear only as an ablation. The custom mechanism and its tensor flow
  must be visible in code rather than represented by a renamed wrapper class.
- Reuse identical validation indices for the measured parent/control, plain ablation,
  and custom network. Report their scores in diagnostics and choose the final predictor
  by that comparison. Never blend merely to avoid reporting a failed architecture.
- Bound epochs and parameter count, use deterministic seeds and early stopping, select
  CUDA/MPS only when available, and always provide a CPU path.
"""
        elif operator == "transfer":
            architecture_contract = """
ARCHITECTURE EXPERIMENT CONTRACT (mandatory):
- Implement the primary candidate as an explicit custom PyTorch `nn.Module` that wraps a pretrained backbone (e.g. EfficientNet) and adds a custom head/attention layer.
- Network egress is UNAVAILABLE in this runtime (see RUNTIME FACTS): pretrained-weight downloads always fail. Never attempt to download weights, never add retry loops for downloads, and never treat a failed download as a repairable error.
- Reuse identical validation indices for the measured parent/control and custom network. Report scores in diagnostics and choose the final predictor by that comparison.
- Bound epochs and parameter count, use deterministic seeds and early stopping, select CUDA/MPS only when available, and always provide a CPU path.
"""
        modality_contract = ""
        if requires_modality_ablation:
            modality_contract = f"""
MULTIMODAL CONTRIBUTION CONTRACT (mandatory):
- Predictive modalities detected: {', '.join(modalities)}.
- Do not assume all provided modalities help. Using identical validation indices,
  compare full fusion, a credible model for every modality alone, and each
  leave-one-modality-out variant.
- Reuse cached preprocessing/embeddings and match model capacity where practical so
  fusion is not rewarded merely for having more parameters or training compute.
- Record every comparison in `result.json` under `modality_ablation_scores` as
  `{{"modalities": ["modality", ...], "score": finite_number,
  "fold_scores": [...], "validation_indices_hash": "same-hash-for-all-variants"}}`.
- Select the smallest modality subset within validation uncertainty of the best score.
   It is valid—and preferred when supported—for one modality to beat full fusion.
"""
        plan_scope_contract = """
PLAN-SCOPE CONTRACT (mandatory):
- The implementation plan is the experiment boundary. Do not add model families,
  modalities, fusion branches, or ablations that the plan did not request.
- A `single_modality` plan must remain single-modality even when other predictive
  inputs exist. Use only the modality named by its experiment/ablation text.
- A fused or modality-ablation plan must run the contribution controls specified
  by the multimodal contract.
"""
        detected_modalities = ", ".join(modalities) if modalities else "none detected"
        modality_fidelity = f"""
MODALITY FIDELITY CONTRACT (mandatory):
- Predictive modalities detected for THIS task: {detected_modalities}. Work ONLY with
  these modalities, exactly as the task provides them.
- NEVER convert one modality into another. For example, do not turn images into handcrafted
  tabular feature rows (pixel statistics, color histograms, hashes) and classify those as a
  "tabular" model; do not embed text into static vectors and call that a separate modality.
  Feature engineering inside the native model is fine; a fabricated second modality is not.
- Identifier/join columns (image_id, id, filename, ...) are KEYS, not features. A table that
  contains only keys plus the target/output columns is a label file, NOT a tabular modality.
- If exactly one modality (or none) is detected: do NOT build fusion models, do NOT run
  modality ablations, do NOT report `modality_ablation_scores`, and do NOT waste time on
  other-modality pipelines. Do not extract features from files into a separate table and
  model that table as if it were provided data.
- Train and predict using only the files and modalities observed under `input/`.
"""
        robustness_contract = """
ROBUST CODE CONTRACT (mandatory):
- DataLoader batches are lists of item tensors. If a dataset yields (features, label)
  tuples, iterate `for features, labels in loader:` (or `features, labels = batch`)
  and call `.to(device)` on the TENSORS only, never on the batch container.
- If an estimator performs an internal stratified validation split (for example
  sklearn's HistGradientBoostingClassifier with early_stopping=True), the split size
  must be >= the number of classes, and its folds must be class-balanced; otherwise
  disable early stopping or lower validation_fraction so every class still appears.
- When datasets have many classes relative to samples, prefer CV splits and estimators
  that always expose one probability column per class; pass `labels` explicitly to
  log_loss and align all probability arrays to the full class order.
- In isolated or spawn-based runtimes, construct PyTorch DataLoaders with
  `num_workers=0` unless an in-process smoke test proved worker startup safe.
  Repair any worker crash by falling back to zero workers.
- Create the deliverable directory before writing to it (e.g., `os.makedirs(Path(<out_dir>).parent, exist_ok=True)`
  or `os.makedirs(<out_dir>, exist_ok=True)` if writing a directory). Writing into a non-existent directory aborts at the final step.
- Always save trained models/checkpoints to disk immediately after training. If the script is re-executed (e.g. due to an error at the end), load the saved checkpoint from disk instead of retraining to save compute.
"""
        probe_contract = ""
        if probe:
            probe_contract = """
PROBE MODE (cheap screening pass — mandatory):
- This is a cheap screening run, not the final submission run. Keep the SAME split,
  metric, evaluation_protocol_hash, and deliverable schema as the plan, but bound the work:
  * If subsampling ROWS: ALWAYS subsample per class (stratified), keeping at least
    5 samples of EVERY class. With many classes (roughly >25) or fewer than a few
    thousand rows, do NOT subsample rows at all; bound cost with model size and
    iterations instead. A random subsample that silently drops whole classes is invalid.
  * Iterative learners (neural, boosting, RL): use roughly 30-40% of the iterations/epochs/episodes.
  * Cheap one-shot fits may simply run as-is.
- CLASS ALIGNMENT (mandatory for classification): every evaluation must cover ALL classes.
  Score with log_loss(..., labels=<full class range in label order>), size OOF and
  test-probability arrays to the FULL class count, and never broadcast or average a
  prediction matrix with fewer columns than classes. If a training fold lacks a class,
  fix the sampling above; do not silently drop that class's column.
- Do NOT tune. Record the honest screening score in result.json. Still write the requested deliverable.
"""
        abort_section = ""
        if abort_context and not probe:
            abort_section = f"""
COMPUTE-SAVING EARLY-ABORT (recommended for iterative learners):
- The current measured incumbent is {abort_context.get('best_score', 'n/a')} ({abort_context.get('direction', '')}).
- If training is iterative (epochs, boosting rounds, RL episodes), monitor a small fixed
  checkpoint set roughly every 15-25% of progress after a warmup of the first 20%.
- If the running score trails the incumbent by more than {abort_context.get('margin', 0.0):.6g}
  for {abort_context.get('patience', 2)} consecutive checks, stop training early. Still write result.json
  with the honest `score`, `fold_scores`, `evaluation_protocol_hash`, and `status: "truncated"`,
  and write the deliverable when practical.
- One-shot predictors (.fit() style) or tasks where early stopping is meaningless: ignore this section.
"""
        tune_contract = ""
        if tune_search:
            tune_contract = """
TUNING SEARCH (mandatory):
- The plan defines a bounded hyperparameter search space. Actually SEARCH it: sample the
  specified configurations (bounded count per the plan), train each on the identical validation
  protocol with per-config early stopping, keep the best configuration honestly, report the final
  score in result.json, and record the winning configuration in diagnostics.
- Do not run a single hand-picked configuration and call it a search.
"""
        prediction_contract = """
PREDICTION EVIDENCE (capability-gated):
- Supervised tasks with machine-readable validation predictions: save `oof_predictions.npz`
  with arrays `oof_pred`, `oof_target`, `oof_index`, `oof_fold`, `test_pred`, and `test_index`.
  `oof_target` must be the labels/targets in exactly the same row order as `oof_pred`; `oof_fold`
  identifies the validation fold for every row. For probability matrices, encode `oof_target`
  as integer column indices 0..K-1. Store indices as numeric or fixed-width Unicode arrays,
  never pickle-dependent object arrays. Test predictions should be the average of the
  same fold/seed models that produced OOF evidence, not an unrelated final holdout model.
- Recompute the reported score from the exact `oof_pred` saved to disk. Hyperparameter-search
  trial scores are not a substitute for the stored winning candidate's OOF score.
- When the requested final output is a numeric CSV/TSV table and this complete evidence bundle
  is available, this search node MUST omit `submission/` and set `output` to null; the manager
  will materialize the one final deliverable after selection. Do not run a separate final-model
  training phase merely to create a per-node submission.
- Include `prediction_signature` in result.json: up to 2048 finite floats sampled from OOF
  predictions and aligned to the fixed validation rows.
- Tasks without labeled validation data, with task-native metrics, or with structured/non-numeric
  outputs should omit prediction evidence and write their native deliverable normally. Never
  invent targets, folds, or numeric predictions merely to use this path.
- For holdout-style or task-native protocols, optionally report `seed_scores`: at least 2 finite
  scores from independent repeated-seed evaluations.
"""
        hardware_contract = """
HARDWARE / DEVICE CONTRACT (mandatory):
- Accelerators (CUDA/MPS) are OPTIONAL, never a requirement. Select the device by probing:
  `device = "cuda" if torch.cuda.is_available() else "cpu"` (Apple hardware:
  `torch.backends.mps.is_available()`). Never hard-code a device.
- Never call `torch.cuda.*` APIs (set_device, empty_cache, device_count, synchronize, ...) unless
  `torch.cuda.is_available()` returned True. Gate every CUDA-specific call behind that check.
- Warnings like "GPU0 ... is of cuda capability X", "Minimum and Maximum cuda capability
  supported", "not compatible with the current PyTorch installation", or "no kernel image is
  available" mean the installed PyTorch cannot use this GPU: `torch.cuda.is_available()` returns
  False. Run on CPU. Such warnings are NON-FATAL: continue normally, do not treat them as a
  failure, and do not spend repair attempts "fixing" them.
- If CPU training is too slow for the time budget, reduce epochs/batches or subsample instead of
  assuming a GPU exists.
"""
        runtime_facts = """
RUNTIME FACTS (mandatory; these are true for every execution):
- The isolated runtime has NO network egress. Every external HTTP(S) request fails with a
  connection error: torch.hub weight downloads, timm/huggingface model downloads,
  download.pytorch.org, and pip installs from inside the program are impossible. Do NOT attempt
  them, do NOT write retry/fallback download code, and do NOT spend attempts "fixing" refused
  connections. Build every model from random initialization or weights that are already present
  locally in the runtime. If a library function would download weights, pass the argument that
  skips downloading (e.g. `weights=None`, plain constructors) instead.
- The installed PyTorch is a modern 2.x release unless proven otherwise at runtime. Never pass
  removed or deprecated keyword arguments: no `verbose=` on `ReduceLROnPlateau`, no
  `pretrained=True` on torchvision model constructors (use the `weights=` enum or no argument),
  no `model_name_or_path` download triggers.
"""
# Prompt layout is prefix-cache friendly: per-run static content
        # (analysis, council contract, standing contracts, input paths, rules)
        # is contiguous and first so provider-side caching covers every node
        # and attempt; then the per-node block (result contract, plan, parent
        # code, node-type contracts); attempt-specific repair notes last.
        return f"""
Write one complete, self-contained Python program for this task.


{analysis.prompt_context(6000)}

{council}

{prediction_contract}
{hardware_contract}
{runtime_facts}
Exact runtime input paths available under input/:
{exact_paths}

CRITICAL RULES & AGENT REASONING WORKFLOW:

1. THINK BEFORE YOU DECIDE & WRITE CODE:
   Begin your response with a <thinking> ... </thinking> block where you step-by-step:
   a. Analyze the task goal, score metric ({analysis.metric} {analysis.direction}), and target deliverable.
   b. Resource & File Inspection: Dynamically inspect the actual files and schemas listed under `input/`. Do NOT assume hardcoded file names, columns, or data structures. Check available resources at runtime.
   c. Model & Architecture Suitability: Verify if the proposed approach actually suits the observed data modalities (tabular, vision, text, audio, time-series, multimodal).
   d. Custom Task-Tailored Engineering: Do not limit yourself to standard library `.fit()` / `.predict()` wrapper calls. Implement custom PyTorch `nn.Module` blocks, custom data augmentation functions (text noise/masking/synonym swapping, tabular noise injection/mixup, custom transforms), metric-aligned loss functions/objectives (custom GBDT objective, focal/ranking/asymmetric loss), or custom feature engineering transformers whenever appropriate for the task.
   e. Lightweight vs Heavy Models: Keep in mind that well-tailored compact models with custom modules, custom augmentations, or domain features often outperform heavy generic off-the-shelf models. Choose the optimal tradeoff for validation score and execution efficiency.
   f. If repairing a failed run or tuning, analyze the execution traceback, diagnose why it failed, and verify if fixing/tuning the current model will improve performance vs refactoring the approach.
   g. Outline your end-to-end data pipeline, validation split strategy, model training, and exact output generation.
   h. Follow the plan's declared modality scope exactly. Test modalities individually
      before combination only for a fused or modality-ablation experiment.

2. RUNTIME & SUBMISSION CONSTRAINTS:
   - For your first debug attempt, verify end-to-end execution on a small data slice: subsample PER CLASS (at least 5 samples of every class, and never fewer rows than the class count) so no class disappears. Scale to full data once the pipeline logic is proven.
   - The program runs from an isolated node directory. Every approved task file is available under `input/<inventory path>`.
   - Do NOT hardcode file names or column names. Inspect actual files and dataframes dynamically at runtime from `input/`.
   - Treat `input/` as immutable: never write, rename, move, or delete files in `input/`.
   - Do not import repository-internal modules (`evaluation`, `contracts`, `agents`, `core`, `modalities`). Use standard installed Python packages only.
   - Choose an honest local validation strategy matching the metric ({analysis.metric} {analysis.direction}); when a council contract is present, its protocol is mandatory.
   - Bound memory and runtime. Include graceful fallbacks if optional hardware/accelerators are unavailable.
   - Produce the exact requested deliverable under `submission/` unless the capability-gated
     prediction-evidence contract explicitly permits this search node to defer it. For sample
     CSV deliverables, preserve row order and exact column headers.
   - A task-provided sample output is a structural contract, regardless of its file type. Without a sample, native non-empty files or directory bundles are allowed.
   - The agent validates the produced artifact independently. A successful process or self-reported score cannot make an invalid output eligible.
   - Follow the RESULT/OUTPUT CONTRACT below exactly.

RESULT/OUTPUT CONTRACT (this execution):
{result_contract}

Implementation plan:
{plan[:3000]}
{parent}{companion}
{architecture_contract}{plan_scope_contract}{modality_contract}{modality_fidelity}{robustness_contract}{probe_contract}{abort_section}{tune_contract}
{repair}{research}

3. OUTPUT FORMAT:
   First return your step-by-step reasoning in a <thinking> ... </thinking> block.
   Then return your complete Python code in ONE fenced block ```python ... ```.
""".strip()
    @staticmethod
    def _cuda_incompatibility_error(stderr: str) -> bool:
        """Return whether captured stderr shows an unsupported-GPU/accelerator failure.

        Matches the Tesla P100 sm_60-style capability warnings, missing-kernel
        crashes, and builds whose torch was compiled without CUDA, so the caller
        can retry with `CUDA_VISIBLE_DEVICES=""` before spending a repair attempt.
        """
        text = str(stderr or "")
        patterns = (
            r"is not compatible with the current PyTorch installation",
            r"no kernel image is available",
            r"not compiled with CUDA enabled",
            r"cuda capability",
            r"cuda runtime error",
            r"cuda error:",
        )
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _cuda_incompatible_environment(self) -> bool:
        """True when the child interpreter's torch cannot use any local GPU.

        Probes the interpreter once per process and caches the result. The
        probe performs a real CUDA allocation and sync, and treats capability
        warnings (for example a capability-6.0 Tesla P100 under a sm_70+ torch
        build) as incompatibility even when ``is_available`` reports True.
        When True, child runs start with ``CUDA_VISIBLE_DEVICES=""``, which
        prevents both the crash and the repeated capability warnings before
        they ever appear.
        """
        cache = getattr(self, "_cuda_probe_cache", None)
        if cache is None:
            cache = self._cuda_probe_cache = {}
        if self.python not in cache:
            incompatible = False
            try:
                probe_script = (
                    "import sys\n"
                    "import torch\n"
                    "usable = torch.cuda.is_available()\n"
                    "if usable:\n"
                    "    x = torch.zeros(8, device='cuda')\n"
                    "    torch.cuda.synchronize()\n"
                    "    sys.stdout.write('1' if usable else '0')\n"
                )
                completed = subprocess.run(
                    [self.python, "-c", probe_script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                usable = bool(
                    completed.returncode == 0 and completed.stdout.strip() == "1"
                )
                incompatible = (
                    completed.returncode == 0
                    and (
                        not usable
                        or self._cuda_incompatibility_error(completed.stderr)
                    )
                )
            except (OSError, subprocess.SubprocessError, ValueError):
                incompatible = False
            cache[self.python] = incompatible
        return cache[self.python]

    @staticmethod
    def _nvidia_gpu_present() -> bool:
        """Return whether an NVIDIA GPU is physically visible on this host."""
        try:
            completed = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            return bool(completed.returncode == 0 and "GPU" in completed.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            return False

    def _torch_cuda_usable(self) -> bool:
        """Return whether the child interpreter's torch can currently use CUDA."""
        cache = getattr(self, "_cuda_probe_cache", None)
        if cache is None or self.python not in cache:
            self._cuda_incompatible_environment()
            cache = getattr(self, "_cuda_probe_cache", None)
        return bool(cache and cache.get(self.python) is False)

    @staticmethod
    def _gpu_compute_capability() -> float | None:
        """Return the primary GPU's CUDA compute capability (e.g. 6.0) or None."""
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if completed.returncode != 0:
                return None
            for line in completed.stdout.splitlines():
                line = line.strip()
                if line:
                    try:
                        return float(line)
                    except ValueError:
                        continue
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        return None

    def _attempt_gpu_upgrade(self, display_name: str) -> bool:
        """Reinstall a GPU-compatible torch to unlock an unusable local GPU.

        Modern torch CUDA 12.8+ wheels omit Pascal (sm_60/sm_61); the official
        CUDA 12.6 build still includes them. When an NVIDIA GPU is physically
        present but the current torch reports CUDA unusable, reinstall torch
        (and any installed torchvision/torchaudio) so the run can use the GPU
        instead of silently falling back to CPU. A pinned version is used so
        pip downgrades a newer incompatible build (``pip install --upgrade``
        alone never downgrades). Runs at most once per process; ``AIBUILDAI_GPU_UPGRADE=0``
        disables it while ``AIBUILDAI_GPU_UPGRADE_INDEX`` overrides the index.
        """
        if not getattr(self, "_gpu_upgrade_attempted", False):
            self._gpu_upgrade_attempted = True
            if (
                _env_enabled("AIBUILDAI_GPU_UPGRADE", default=True)
                and self._nvidia_gpu_present()
                and not self._torch_cuda_usable()
            ):
                capability = self._gpu_compute_capability()
                use_pascal_build = capability is not None and capability < 7.0
                index = os.getenv(
                    "AIBUILDAI_GPU_UPGRADE_INDEX",
                    "https://download.pytorch.org/whl/cu126",
                ).strip()
                extras: list[str] = []
                try:
                    probe = subprocess.run(
                        [
                            self.python,
                            "-c",
                            "import importlib.util, sys, torch;"
                            "names=[n for n in ('torchvision','torchaudio') "
                            "if importlib.util.find_spec(n) is not None];"
                            "sys.stdout.write(','.join(names))",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    extras = [
                        name for name in probe.stdout.split(",") if name.strip()
                    ]
                except (OSError, subprocess.SubprocessError, ValueError):
                    extras = []
                if use_pascal_build:
                    pins = {
                        "torch": "torch==2.6.0+cu126",
                        "torchvision": "torchvision==0.21.0+cu126",
                        "torchaudio": "torchaudio==2.6.0+cu126",
                    }
                    packages = [pins["torch"]]
                    for name in extras:
                        packages.append(pins.get(name, name))
                    self._log(
                        display_name,
                        f"GPU is a pre-7.0 capability device (Pascal-class, {capability}); "
                        f"reinstalling torch {packages} from {index} to enable CUDA. "
                        "This downloads a large package and may take several minutes.",
                    )
                else:
                    packages = ["torch", *extras]
                    self._log(
                        display_name,
                        "GPU is present but the current torch cannot use it; "
                        f"reinstalling torch (+{', '.join(extras) or 'none'}) from "
                        f"{index} to enable CUDA. This downloads a large package and "
                        "may take several minutes.",
                    )
                try:
                    subprocess.run(
                        [
                            self.python,
                            "-m",
                            "pip",
                            "install",
                            "--upgrade",
                            *packages,
                            "--index-url",
                            index,
                            "--disable-pip-version-check",
                            "--no-input",
                        ],
                        env=sanitized_subprocess_env(),
                        check=False,
                        timeout=1800,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    self._log(
                        display_name,
                        f"GPU upgrade failed ({exc}); will run on CPU instead.",
                    )
                cache = getattr(self, "_cuda_probe_cache", None)
                if cache is not None:
                    cache.pop(self.python, None)
                if self._torch_cuda_usable():
                    self._log(
                        display_name,
                        "GPU enabled via a compatible torch build; proceeding on CUDA.",
                    )
                    return True
                self._log(
                    display_name,
                    "GPU upgrade did not enable CUDA (driver/index mismatch?); "
                    "running on CPU.",
                )
        return self._torch_cuda_usable()

    @staticmethod
    def _web_query(stderr: str) -> str:
        meaningful = [line.strip() for line in stderr.splitlines() if line.strip()]
        last = meaningful[-1] if meaningful else stderr.strip()
        return f"Python debugging {last}"[:99]

    def run(
        self,
        node_dir: Path,
        plan: str,
        task_dir: Path,
        task_analysis: TaskAnalysis,
        *,
        parent_code: str | None = None,
        companion_code: str | None = None,
        operator: str = "root",
        node_label: str | None = None,
        max_debug_attempts: int = 4,
        stall_seconds: float = 1200.0,
        hard_limit_seconds: float | None = None,
        council_brief: CouncilBrief | None = None,
        probe: bool = False,
        abort_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        node_dir = Path(node_dir)
        display_name = node_label or node_dir.name
        node_dir.mkdir(parents=True, exist_ok=True)
        allowed_paths = council_brief.allowed_input_paths if council_brief is not None else None
        linked_inputs = expose_task_data(
            Path(task_dir), node_dir, allowed_paths=allowed_paths
        )
        modality_inventory = predictive_modality_inventory(task_analysis.files)
        requires_modality_ablation = bool(
            modality_inventory["is_multimodal"]
            and _plan_requests_modality_ablation(plan)
        )
        self._log(display_name, f"Exposed {len(linked_inputs)} task files under input/.")
        task_input_snapshot = task_dir_snapshot(Path(task_dir))
        if hard_limit_seconds is None:
            configured_hard_limit = os.getenv("AIBUILDAI_HARD_LIMIT_SECONDS", "").strip()
            try:
                hard_limit_seconds = float(configured_hard_limit) if configured_hard_limit else 3600.0
            except ValueError:
                hard_limit_seconds = 3600.0
        if council_brief is not None:
            (node_dir / "evaluation_protocol.json").write_text(
                json.dumps(
                    council_brief.evaluation_protocol.to_dict(), indent=2, ensure_ascii=False
                )
                + "\n",
                encoding="utf-8",
            )
            (node_dir / "council_brief_reference.json").write_text(
                json.dumps(
                    {
                        "brief_hash": council_brief.brief_hash,
                        "brief_path": "../council/council_brief.json",
                        "evaluation_protocol_hash": (
                            council_brief.evaluation_protocol.protocol_hash
                        ),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        source_file = node_dir / "algorithm.py"
        feedback = ""
        web_notes = ""
        last_code = parent_code or ""
        last_attempt_code = ""
        diagnostics: list[str] = []

        for attempt in range(1, max(1, int(max_debug_attempts)) + 1):
            self._log(
                display_name,
                f"Generating implementation attempt {attempt}/{max_debug_attempts} ({operator}).",
            )
            prompt = self._prompt(
                task_analysis,
                plan,
                parent_code,
                companion_code,
                last_attempt_code if feedback else None,
                feedback,
                web_notes,
                council_brief,
                operator,
                probe=probe,
                abort_context=abort_context,
                tune_search=(
                    operator == "tune"
                    and _env_enabled("AIBUILDAI_TUNE_SEARCH", default=True)
                ),
            )
            try:
                response = call_llm(
                    "You write reliable, direct, high-performing SOTA implementations from observed task files.",
                    prompt,
                    model=self.model_name,
                    temperature=0.1 if attempt > 1 else 0.2,
                )
                code = _python_source(response)
                if len(code) < 80:
                    raise ValueError("code response was empty or too short")
                last_code = code
                last_attempt_code = code
                self._log(display_name, f"Received {len(code)} characters of executable code.")
            except Exception as exc:
                diagnostics.append(f"attempt {attempt}: LLM response error: {exc}")
                self._log(display_name, f"LLM attempt {attempt} failed: {exc}")
                (node_dir / f"attempt_{attempt}.log").write_text(
                    f"LLM response error: {type(exc).__name__}: {exc}\n",
                    encoding="utf-8",
                )
                feedback = _tail(str(exc))
                continue

            if operator in ("architect", "transfer"):
                architecture_errors = _architecture_source_errors(code, operator)
                if architecture_errors:
                    feedback = (
                        "ARCHITECTURE CONTRACT FAILURE:\n- "
                        + "\n- ".join(architecture_errors)
                        + "\nRevise the program into a genuine custom PyTorch architecture; "
                        "do not replace this experiment with another library estimator."
                    )
                    diagnostics.append(f"attempt {attempt}: {feedback}")
                    self._log(
                        display_name,
                        "Generated code did not implement the requested custom architecture; "
                        "returning it to the repair loop.",
                    )
                    (node_dir / f"attempt_{attempt}.log").write_text(
                        feedback + "\n", encoding="utf-8"
                    )
                    continue

            source_file.write_text(code, encoding="utf-8")
            # A repaired attempt must prove its own deliverable instead of
            # inheriting stale output from a crashed prior attempt.
            if attempt > 1:
                for stale_file in (node_dir / "result.json", node_dir / "submission.csv"):
                    try:
                        stale_file.unlink()
                    except FileNotFoundError:
                        pass
                stale_submission = node_dir / "submission"
                if stale_submission.is_dir():
                    shutil.rmtree(stale_submission)
            child_env = sanitized_subprocess_env()
            child_env.update(
                {
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": os.getenv("AIBUILDAI_MODEL_THREADS", "4"),
                    "MKL_NUM_THREADS": os.getenv("AIBUILDAI_MODEL_THREADS", "4"),
                }
            )
            if _env_enabled("AIBUILDAI_GPU_UPGRADE", default=True):
                self._attempt_gpu_upgrade(display_name)
            if self._cuda_incompatible_environment():
                child_env["CUDA_VISIBLE_DEVICES"] = ""
                self._log(
                    display_name,
                    "Child torch cannot use the local GPU (incompatible capability "
                    "or no GPU); preemptively disabling CUDA for this node.",
                )
            if not _env_enabled("AIBUILDAI_ALLOW_NETWORK", default=False):
                # Generated programs run without network egress; only the parent
                # performs pip installs and web-assisted repair.
                for proxy in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                    child_env[proxy] = "http://127.0.0.1:9"
                child_env.pop("NO_PROXY", None)
                child_env.pop("no_proxy", None)
            try:
                self._log(display_name, f"Executing {source_file.name}; child logs follow.")
                
                # 3. Create node-local venv for true isolation (with fallback)
                venv_dir = node_dir / ".venv"
                node_python = self.python
                if not venv_dir.exists():
                    self._log(display_name, "Creating isolated virtual environment...")
                    try:
                        subprocess.run([self.python, "-m", "venv", "--system-site-packages", "--without-pip", str(venv_dir)], check=True, capture_output=True)
                        if os.name == "nt":
                            node_python = str(venv_dir / "Scripts" / "python.exe")
                        else:
                            node_python = str(venv_dir / "bin" / "python")
                    except (subprocess.CalledProcessError, OSError) as exc:
                        self._log(display_name, f"Failed to create venv: {exc}. Falling back to system Python.")
                        node_python = self.python
                        if venv_dir.exists():
                            shutil.rmtree(venv_dir, ignore_errors=True)
                else:
                    if os.name == "nt":
                        node_python = str(venv_dir / "Scripts" / "python.exe")
                    else:
                        node_python = str(venv_dir / "bin" / "python")

                install_attempts = 0
                cuda_fallback_used = False
                while True:
                    completed = run_supervised_process(
                        [node_python, source_file.name],
                        cwd=node_dir,
                        env=child_env,
                        stall_seconds=stall_seconds,
                        hard_limit_seconds=hard_limit_seconds,
                        activity_root=node_dir,
                        stdout_stream=sys.stdout,
                        stderr_stream=sys.stderr,
                        label=f"Implementation {display_name} attempt {attempt}",
                    )
                    
                    if completed.returncode != 0 and install_attempts < 3:
                        # 1. Handle missing modules
                        err_match = re.search(r"(?:ModuleNotFoundError|ImportError): No module named '([^']+)'", completed.stderr)
                        if err_match:
                            missing_module = err_match.group(1).split('.')[0]
                            pkg_map = {"cv2": "opencv-python", "sklearn": "scikit-learn", "PIL": "pillow", "yaml": "pyyaml"}
                            pkg_to_install = pkg_map.get(missing_module, missing_module)
                            
                            self._log(display_name, f"Missing module '{missing_module}' detected. Auto-installing {pkg_to_install}...")
                            install_attempts += 1
                            pip_env = sanitized_subprocess_env()
                            try:
                                pip_args = [
                                    self.python, "-m", "pip", "install", pkg_to_install,
                                    "--index-url", "https://pypi.org/simple",
                                    "--disable-pip-version-check", "--no-input",
                                ]
                                if node_python != self.python:
                                    pip_args.extend(["--prefix", str(venv_dir)])
                                
                                subprocess.run(
                                    pip_args,
                                    check=False,
                                    env=pip_env,
                                    timeout=240,
                                )
                                self._log(display_name, f"Retrying execution after installing '{pkg_to_install}'...")
                            except (OSError, subprocess.SubprocessError) as exc:
                                self._log(display_name, f"Package install failed ({exc}); continuing without it.")
                            continue
                            
                        # 2. Handle CUDA compatibility/GPU errors
                        if self._cuda_incompatibility_error(completed.stderr):
                            if child_env.get("CUDA_VISIBLE_DEVICES") != "":
                                self._log(display_name, "CUDA GPU compatibility error detected. Forcing CPU fallback and retrying...")
                                child_env["CUDA_VISIBLE_DEVICES"] = ""
                                cuda_fallback_used = True
                                continue
                    break

            except Exception as exc:
                feedback = _tail(f"Process launch failed: {exc}")
                diagnostics.append(f"attempt {attempt}: {feedback}")
                (node_dir / f"attempt_{attempt}.log").write_text(
                    feedback + "\n", encoding="utf-8"
                )
                self._log(display_name, feedback)
                continue

            changed_inputs = verify_task_dir_unchanged(Path(task_dir), task_input_snapshot)
            if changed_inputs:
                feedback = (
                    "INPUT INTEGRITY VIOLATION: the generated program modified task-owned "
                    f"files ({', '.join(changed_inputs[:5])}). Inputs under input/ must be "
                    "opened read-only; the task data is shared with every other search node."
                )
                diagnostics.append(f"attempt {attempt}: {feedback}")
                (node_dir / f"attempt_{attempt}.log").write_text(
                    feedback + "\n", encoding="utf-8"
                )
                self._log(display_name, feedback)
                continue

            output = self._output_path(node_dir)
            payload = self._read_result(node_dir)
            oof_path = node_dir / "oof_predictions.npz"
            prediction_evidence = (
                inspect_prediction_evidence(oof_path, task_analysis.metric)
                if oof_path.is_file()
                else None
            )
            submission = task_analysis.submission or {}
            deferred_table_output = bool(
                prediction_evidence is not None
                and prediction_evidence.blendable
                and str(submission.get("kind") or "").casefold() == "table"
                and str(submission.get("extension") or "").casefold()
                in {".csv", ".tsv"}
            )
            payload_status = payload.get("status")
            status = str(payload_status or "completed")
            if status not in {"completed", "truncated"}:
                status = "completed"
            combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
            (node_dir / f"attempt_{attempt}.log").write_text(
                "\n".join(
                    (
                        f"returncode: {completed.returncode}",
                        f"elapsed_seconds: {completed.elapsed_seconds:.3f}",
                        "",
                        "STDOUT:",
                        _tail(completed.stdout, 2500),
                        "",
                        "STDERR:",
                        _tail(completed.stderr, 2500),
                        "",
                    )
                ),
                encoding="utf-8",
            )
            diagnostics.append(
                f"attempt {attempt}: exit={completed.returncode}; elapsed={completed.elapsed_seconds:.1f}s\n{_tail(combined, 2000)}"
            )
            self._log(
                display_name,
                f"Attempt {attempt} exited {completed.returncode} after {completed.elapsed_seconds:.1f}s.",
            )

            if prediction_evidence is not None and not prediction_evidence.valid:
                feedback = _tail(
                    "Prediction evidence validation failed:\n- "
                    + "\n- ".join(prediction_evidence.errors)
                )
                diagnostics.append(f"attempt {attempt}: {feedback}")
                with (node_dir / f"attempt_{attempt}.log").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(f"\nPREDICTION EVIDENCE:\n{feedback}\n")
                self._log(display_name, "Generated prediction evidence was invalid.")
                continue

            if completed.returncode == 0 and (
                output is not None or deferred_table_output or status == "truncated"
            ):
                validation = None
                if output is not None:
                    validation = self.submission_validator.validate(
                        output,
                        task_analysis,
                        allowed_root=node_dir,
                    )
                    if not validation.valid:
                        feedback = _tail(validation.feedback())
                        diagnostics.append(
                            f"attempt {attempt}: output validation failed\n{feedback}"
                        )
                        with (node_dir / f"attempt_{attempt}.log").open(
                            "a", encoding="utf-8"
                        ) as stream:
                            stream.write(f"\nSUBMISSION VALIDATION:\n{feedback}\n")
                        self._log(display_name, "Generated output failed submission validation.")
                        for message in validation.errors:
                            self._log(display_name, f"Validation error: {message}")
                        continue
                if council_brief is not None:
                    evaluation_errors = self._validate_reported_evaluation(
                        payload, council_brief.evaluation_protocol
                    )
                    if evaluation_errors:
                        feedback = _tail(
                            "Evaluation contract validation failed:\n- "
                            + "\n- ".join(evaluation_errors)
                        )
                        diagnostics.append(f"attempt {attempt}: {feedback}")
                        with (node_dir / f"attempt_{attempt}.log").open(
                            "a", encoding="utf-8"
                        ) as stream:
                            stream.write(f"\nEVALUATION CONTRACT:\n{feedback}\n")
                        self._log(
                            display_name,
                            "Generated score failed the shared evaluation contract.",
                        )
                        continue
                if requires_modality_ablation and status == "completed":
                    modality_errors = validate_modality_ablation_report(
                        payload,
                        modality_inventory["modalities"],
                        expected_folds=(
                            council_brief.evaluation_protocol.folds
                            if council_brief is not None
                            else None
                        ),
                    )
                    if modality_errors:
                        feedback = _tail(
                            "Modality contribution validation failed:\n- "
                            + "\n- ".join(modality_errors)
                        )
                        diagnostics.append(f"attempt {attempt}: {feedback}")
                        with (node_dir / f"attempt_{attempt}.log").open(
                            "a", encoding="utf-8"
                        ) as stream:
                            stream.write(f"\nMODALITY ABLATION CONTRACT:\n{feedback}\n")
                        self._log(
                            display_name,
                            "Generated result did not prove modality contribution on comparable variants.",
                        )
                        continue
                output = validation.output_path if output is not None else None
                if deferred_table_output and output is not None:
                    # A generated implementation may ignore the deferral prompt. Once its
                    # reusable evidence and optional output have both been checked, retain
                    # only the evidence so candidate nodes cannot accumulate submissions.
                    submission_dir = node_dir / "submission"
                    if submission_dir.is_dir():
                        shutil.rmtree(submission_dir)
                    root_submission = node_dir / "submission.csv"
                    if root_submission.is_file():
                        root_submission.unlink()
                    output = None
                reported_metric = str(payload.get("metric") or task_analysis.metric)
                reported_direction = str(payload.get("direction") or task_analysis.direction)
                payload_score = payload.get("score")
                score: float | None = None
                if not isinstance(payload_score, bool):
                    try:
                        candidate = float(str(payload_score))
                        if math.isfinite(candidate):
                            score = candidate
                    except (TypeError, ValueError):
                        pass
                if score is None:
                    score = _score_from_stdout(
                        completed.stdout, reported_metric, reported_direction
                    )
                if score is None:
                    score = _score_from_stdout(completed.stdout, None, reported_direction)
                if score is None:
                    if status == "truncated":
                        feedback = (
                            "The truncated run reported no usable score in result.json or stdout."
                        )
                        diagnostics.append(f"attempt {attempt}: {feedback}")
                        self._log(display_name, feedback)
                        continue
                    score = 1e30 if reported_direction == "minimize" else -1e30
                reported_score = score
                if prediction_evidence is not None and prediction_evidence.centrally_scored:
                    score = float(prediction_evidence.score)
                    tolerance = max(1e-8, abs(score) * 1e-5)
                    if abs(reported_score - score) > tolerance:
                        message = (
                            f"Central evidence audit replaced reported score {reported_score:.8g} "
                            f"with {score:.8g} recomputed from saved OOF predictions."
                        )
                        diagnostics.append(message)
                        self._log(display_name, message)
                accepted_artifact = (
                    str(output) if output is not None else str(oof_path)
                )
                self._log(
                    display_name,
                    f"Accepted runnable candidate artifact {accepted_artifact}; score={score:.8g}.",
                )
                test_pred_path = node_dir / "test_predictions.npz"
                return {
                    "status": status,
                    "score": score,
                    "metric": reported_metric,
                    "direction": reported_direction,
                    "output": str(output) if output is not None else None,
                    "reported_score": reported_score,
                    "diagnostics": "\n\n".join(diagnostics)[-10000:],
                    "attempts": attempt,
                    "operator": operator,
                    "submission_validation": (
                        validation.to_dict() if validation is not None else {}
                    ),
                    "evaluation_protocol_hash": (
                        council_brief.evaluation_protocol.protocol_hash
                        if council_brief is not None
                        else None
                    ),
                    "fold_scores": (
                        prediction_evidence.fold_scores
                        if prediction_evidence is not None
                        and prediction_evidence.centrally_scored
                        and prediction_evidence.fold_scores
                        else payload.get("fold_scores")
                    ),
                    "validation_sample_count": payload.get("validation_sample_count"),
                    "modality_ablation_scores": payload.get(
                        "modality_ablation_scores"
                    ),
                    "prediction_signature": payload.get("prediction_signature"),
                    "seed_scores": payload.get("seed_scores"),
                    "oof_predictions": str(oof_path) if oof_path.is_file() else None,
                    "prediction_evidence": (
                        prediction_evidence.summary()
                        if prediction_evidence is not None
                        else None
                    ),
                    "test_predictions": (
                        str(test_pred_path) if test_pred_path.is_file() else None
                    ),
                    "code": last_code,
                }

            reason = (
                f"exit code {completed.returncode}; "
                + ("no deliverable was written under submission/." if output is None else "deliverable exists but execution failed.")
            )
            feedback = _tail(f"{reason}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
            if attempt < max_debug_attempts and os.getenv("AIBUILDAI_WEB_SEARCH", "1").lower() not in {"0", "false", "no"}:
                try:
                    self._log(display_name, "Searching the web for the concrete execution error.")
                    web_notes = search_web(self._web_query(combined or reason))
                except Exception as exc:
                    web_notes = f"Web search was unavailable: {exc}"
                    self._log(display_name, web_notes)

        self._log(display_name, f"All {max(1, int(max_debug_attempts))} repair attempts were exhausted.")
        return {
            "status": "failed",
            "score": None,
            "metric": task_analysis.metric,
            "direction": task_analysis.direction,
            "output": None,
            "diagnostics": "\n\n".join(diagnostics)[-12000:] or feedback,
            "attempts": max(1, int(max_debug_attempts)),
            "operator": operator,
            "code": last_code,
        }
