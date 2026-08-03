import ast
import hashlib
import re
import os
import sys
import json
import math
import subprocess
import shutil
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from packaging.requirements import Requirement
from core.runtime_contracts import ModelBundle, SplitPlan
from evaluation.metrics import resolve_metric_name
from evaluation.error_analysis import build_error_analysis
from evaluation.policy import (
    EvaluationPolicy,
    normalize_evaluation_mode,
    select_evaluation_policy,
)
from evaluation.prediction_io import (
    legacy_prediction_payload,
    load_assignment_table,
    load_prediction_bundle,
    load_prediction_table,
    write_prediction_bundle,
)
from evaluation.submission import (
    task_requires_submission,
    validate_node_submission,
)
from .llm_utils import call_llm
from .modality_scaffold import (
    runtime_data_prompt,
    task_loader_source,
    write_runtime_data_contract,
)
from .prompt_context import modality_prompt_context
from .task_analyzer import TaskAnalyzer
from .strategy_patterns import render_strategy_patterns
from .validation_guard import inspect_generated_code
from evaluation_contract import FIDELITY_PROFILES, validate_evaluation_outputs
from runtime_utils import (
    absolute_path_without_symlink_resolution,
    accelerator_subprocess_env,
    expose_task_data,
    resolve_within,
    run_supervised_process,
    validate_storage_identifier,
)

class ImplementationAgent:
    _candidate_registry_lock = threading.Lock()

    def __init__(
        self,
        venv_python_path: str | None = None,
        model_name: str = None,
    ):
        import sys
        if venv_python_path is None:
            self.venv_python = sys.executable
            self.model_name = model_name
            self.project_root = Path(__file__).resolve().parent.parent
            return
        resolved_path = str(
            absolute_path_without_symlink_resolution(venv_python_path)
        )
        # Check if the resolved venv python is fully functional
        use_fallback = True
        if Path(resolved_path).exists():
            try:
                res = subprocess.run([resolved_path, "-c", "import sys; print('ok')"], capture_output=True, text=True, timeout=5)
                if res.returncode == 0 and "ok" in res.stdout:
                    use_fallback = False
            except Exception:
                pass
                
        if use_fallback:
            print(f"ImplementationAgent WARNING: Specified python path '{resolved_path}' is invalid or non-functional. Falling back to active running interpreter: {sys.executable}")
            self.venv_python = sys.executable
        else:
            self.venv_python = resolved_path
            
        self.model_name = model_name
        self.project_root = Path(__file__).resolve().parent.parent

    @staticmethod
    def _candidate_code_fingerprint(
        code: str,
        *,
        task_spec: object,
        evaluation_mode: str,
    ) -> str:
        """Hash executable structure and literal hyperparameters pre-training."""
        tree = ast.parse(code)
        canonical = ast.dump(
            tree,
            annotate_fields=True,
            include_attributes=False,
        )
        task_payload = (
            task_spec.to_dict()
            if hasattr(task_spec, "to_dict")
            else task_spec
        )
        payload = {
            "ast": canonical,
            "evaluation_mode": evaluation_mode,
            "task": task_payload,
        }
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, default=str
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _register_candidate_fingerprint(
        cls,
        registry_root: Path,
        *,
        fingerprint: str,
        node_id: str,
    ) -> str | None:
        """Return the prior node for a duplicate, otherwise register it."""
        registry_path = Path(registry_root) / "candidate_fingerprints.json"
        with cls._candidate_registry_lock:
            try:
                registry = json.loads(
                    registry_path.read_text(encoding="utf-8")
                )
            except FileNotFoundError:
                registry = {}
            except (OSError, ValueError, TypeError):
                registry = {}
            previous = registry.get(fingerprint)
            if previous and previous != node_id:
                return str(previous)
            registry[fingerprint] = node_id
            registry_path.write_text(
                json.dumps(registry, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return None

    @staticmethod
    def _looks_like_deep_learning(code: str) -> bool:
        lowered = str(code).lower()
        return any(
            marker in lowered
            for marker in (
                "import torch",
                "tensorflow",
                "keras",
                "pytorch_tabnet",
                "dataloader(",
                "nn.module",
                "epochs",
            )
        )

    @staticmethod
    def _materialize_output_contracts(
        node_dir: Path,
        task_spec,
        result_data: dict,
        algorithm_path: Path,
    ) -> dict[str, object]:
        """Create typed prediction/model bundles from validated outputs."""
        artifacts: dict[str, object] = {}
        evaluation_path = node_dir / "evaluation_manifest.json"
        task_fingerprint = hashlib.sha256(
            json.dumps(
                task_spec.to_dict(), sort_keys=True, default=str
            ).encode("utf-8")
        ).hexdigest()
        if evaluation_path.is_file():
            evaluation = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
            task_fingerprint = str(
                evaluation.get("task_fingerprint")
                or evaluation.get("dataset_fingerprint")
                or task_fingerprint
            )
        else:
            evaluation = {}
        try:
            frame = load_prediction_table(node_dir / "oof_predictions")
        except FileNotFoundError:
            frame = None
        if frame is not None:
            required = {"row_id"}
            if required.issubset(frame.columns):
                predictions, inferred_class_names = (
                    legacy_prediction_payload(frame)
                )
                if "fold_id" in frame.columns:
                    fold_ids = frame["fold_id"].to_numpy(dtype=int)
                else:
                    assignments = load_assignment_table(
                        node_dir / "fold_assignments"
                    ).set_index("row_id")
                    fold_ids = (
                        assignments.reindex(frame["row_id"])["fold_id"]
                        .to_numpy(dtype=int)
                    )
                sample_ids = tuple(str(item) for item in frame["row_id"])
                split_plan = SplitPlan(
                    assignments={
                        sample_id: int(fold)
                        for sample_id, fold in zip(sample_ids, fold_ids)
                    },
                    strategy=str(
                        evaluation.get("split_strategy")
                        or "legacy_harness"
                    ),
                    seed=int(evaluation.get("seed", 42)),
                    leakage_unit=str(
                        evaluation.get("leakage_unit") or "row_id"
                    ),
                    split_fingerprint=str(
                        evaluation.get("split_fingerprint")
                        or evaluation.get("fold_assignment_sha256")
                        or ""
                    ),
                )
                prediction_bundle = write_prediction_bundle(
                    node_dir,
                    task_fingerprint=task_fingerprint,
                    split_plan=split_plan,
                    output_type=task_spec.output.type,
                    sample_ids=sample_ids,
                    predictions=predictions,
                    targets=(
                        frame["target"].to_numpy()
                        if "target" in frame.columns
                        else None
                    ),
                    fold_ids=fold_ids,
                    class_names=(
                        inferred_class_names
                        or task_spec.output.class_names
                    ),
                    metadata={
                        "modality": task_spec.modality,
                        "problem_type": task_spec.problem_type,
                        "metric": task_spec.primary_metric,
                    },
                )
                artifacts["prediction_bundle"] = str(
                    node_dir / "predictions" / "manifest.json"
                )
                artifacts["compatibility_key"] = (
                    prediction_bundle.compatibility_key
                )
        typed_manifest = node_dir / "predictions" / "manifest.json"
        if typed_manifest.is_file() and "prediction_bundle" not in artifacts:
            typed_bundle, _, _, _ = load_prediction_bundle(typed_manifest)
            artifacts["prediction_bundle"] = str(typed_manifest)
            artifacts["compatibility_key"] = typed_bundle.compatibility_key

        checkpoint_suffixes = {
            ".bin",
            ".cbm",
            ".joblib",
            ".onnx",
            ".pkl",
            ".pt",
            ".pth",
            ".safetensors",
        }
        checkpoints = tuple(
            path.relative_to(node_dir).as_posix()
            for path in sorted(node_dir.rglob("*"))
            if path.is_file()
            and path.suffix.lower() in checkpoint_suffixes
            and "input" not in path.relative_to(node_dir).parts
        )
        model_dir = node_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_bundle = ModelBundle(
            model_family=(
                next(
                    iter(
                        ImplementationAgent._model_family_imports(
                            algorithm_path.read_text(encoding="utf-8")
                        )
                    ),
                    "generated_pipeline",
                )
            ),
            task_fingerprint=task_fingerprint,
            output_type=task_spec.output.type,
            checkpoint_paths=checkpoints,
            entrypoint=f"{algorithm_path.name}:__main__",
            dependencies=(),
            metadata={
                "modality": task_spec.modality,
                "problem_type": task_spec.problem_type,
                "generated_script": True,
            },
        )
        model_manifest = model_dir / "manifest.json"
        model_manifest.write_text(
            json.dumps(model_bundle.to_dict(), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        artifacts["model_bundle"] = str(model_manifest)
        result_data.update(artifacts)
        (node_dir / "result.json").write_text(
            json.dumps(result_data, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return artifacts

    @classmethod
    def _debug_repair_guidance(
        cls,
        code: str,
        stderr: str,
        stdout: str,
        execution_stopped: bool,
        accelerator: str,
        fidelity_profile: dict,
        evaluation_mode: str = "cross_validation",
    ) -> str:
        """Return deterministic, failure-specific constraints for the LLM debugger."""
        combined = (str(stderr) + "\n" + str(stdout)).lower()
        guidance = []
        if (
            execution_stopped
            or "stalled" in combined
            or "timeout" in combined
            or "deadline" in combined
        ):
            work_unit = (
                "inside each fold"
                if evaluation_mode == "cross_validation"
                else "inside the single training run"
            )
            guidance.append(
                f"Execution recovery: keep the required rows, but reduce work {work_unit}; "
                "cap epochs/iterations, add early stopping, avoid repeated final refits, and emit "
                "progress at least once per epoch or trial. Do not increase training duration while debugging."
            )
        if any(
            marker in combined
            for marker in ("out of memory", "cuda error", "cublas", "cudnn", "mps backend")
        ):
            guidance.append(
                "Accelerator repair: use mini-batches, clear references between folds, call "
                "torch.cuda.empty_cache() only when CUDA exists, retry once with a smaller batch, "
                "and fall back to CPU only if the selected backend is unusable."
            )
        if any(marker in combined for marker in ("dtype", "expected scalar type", "object", "can't convert")):
            guidance.append(
                "Dtype repair: encode categorical/object columns using training-fold state; use float32 "
                "features, float labels for BCE losses, and long labels for cross-entropy/embedding indices."
            )
        if any(marker in combined for marker in ("shape", "size mismatch", "mat1 and mat2", "dimension")):
            guidance.append(
                "Shape repair: derive the network input width after fold-local transformation, keep "
                "binary logits one-dimensional, and assert prediction length before writing outputs."
            )
        if any(
            marker in combined
            for marker in (
                "not in index",
                "columns are missing",
                "feature names",
                "feature_names",
            )
        ):
            guidance.append(
                "Feature-parity repair: keep raw feature columns separate from engineered columns, "
                "apply the same fitted transformation function to fold-train, fold-validation, full "
                "evaluation, and test frames, then assert and align transformed test columns to the "
                "transformed training columns. Never index raw test data with post-engineering columns."
            )
        if any(marker in combined for marker in ("no module named", "modulenotfounderror", "importerror")):
            guidance.append(
                "Dependency repair: use only installed/project-allowlisted packages and the copied local "
                "artifact module; remove every import of the missing package throughout the script and "
                "implement the intended mechanism with an available lower-level library. Do not merely "
                "wrap the missing import in try/except or invent package paths."
            )
        if "numpy.ndarray" in combined and any(
            marker in combined for marker in ("iloc", "loc")
        ):
            guidance.append(
                "Array-indexing repair: y_eval, row_ids, fold_ids, test_ids, and aliases passed "
                "from them are NumPy arrays. Replace every .iloc/.loc access on those values across "
                "the complete script with array[index] or np.asarray(array)[index]. DataFrame feature "
                "rows may continue to use .iloc."
            )
        if "keyerror" in combined:
            guidance.append(
                "Component-layout repair: obey runtime_data_contract.json exactly. Nested input "
                "objects are flattened by TaskDataLoader; select declared single columns directly "
                "and select structured components with their '<component>__' column prefix. Repair "
                "every train, validation, full-data, and test access consistently."
            )
        if any(
            marker in combined
            for marker in ("result contract", "oof", "fold_id", "evaluation_manifest", "submission")
        ):
            if evaluation_mode == "cross_validation":
                guidance.append(
                    "Evaluation repair: preserve evaluation_contract rows and "
                    "fold_ids exactly, write one OOF prediction per scheduled "
                    "row, and regenerate result.json and submission.csv."
                )
            elif evaluation_mode == "holdout":
                guidance.append(
                    "Evaluation repair: preserve the harness train/validation "
                    "split, train once, write validation_predictions.npz for "
                    "validation_row_ids, and do not create OOF output."
                )
            else:
                guidance.append(
                    "Evaluation repair: preserve task-native validation rows, "
                    "write validation_predictions.npz, and do not create "
                    "supervised OOF output."
                )
        if cls._looks_like_deep_learning(code):
            release_clause = (
                "and release the model between folds"
                if evaluation_mode == "cross_validation"
                else "and train only one validation model"
            )
            guidance.append(
                "Deep-learning invariant: never move the complete dataset to GPU; use DataLoader "
                "mini-batches, place model and each batch on the same device, detach predictions to CPU, "
                f"use at most {fidelity_profile['max_epochs']} epochs with patience "
                f"{fidelity_profile['early_stopping_patience']}, {release_clause}."
            )
        if not guidance:
            guidance.append(
                "Trace the first concrete exception to its source, make the smallest causal repair, "
                "and preserve the measured parent method and evaluation contract."
            )
        return "\n".join(f"- {item}" for item in guidance)

    @staticmethod
    def _fine_tuning_instruction(
        operator: Optional[str], tuning_context: Optional[dict], fidelity_profile: dict
    ) -> str:
        if operator != "tune" or not tuning_context:
            return ""
        suggested = tuning_context.get("suggested_initial_parameters", [])
        reused = tuning_context.get("reused_trials", [])
        duplicate_hashes = tuning_context.get(
            "avoid_duplicate_parameter_hashes", []
        )
        return (
            "This is an evidence-triggered fine-tuning run, not an architecture rewrite. Preserve the "
            "parent preprocessing, feature set, model family, folds, and output schema. Search only "
            "meaningful existing hyperparameters, reuse the parent settings as a control trial, and use "
            f"at most {fidelity_profile['max_tuning_trials']} deterministic/pruned trials. For neural "
            f"models, tune learning rate, batch size, weight decay, dropout/width, and epochs up to "
            f"{fidelity_profile['max_epochs']} with early-stopping patience "
            f"{fidelity_profile['early_stopping_patience']}; increasing epochs is allowed only within "
            "that cap. For boosted trees, tune depth/leaves, learning rate, regularization, sampling, "
            f"and iterations up to {fidelity_profile['max_estimator_iterations']}. Optimize only the "
            "harness-provided validation folds—never the test set. result.json must include a non-empty "
            "`hyperparameters` object and integer `tuning_trials`. Treat compatible historical trials "
            "as prior evidence: evaluate the suggested configurations first when they fit the current "
            "bounds, do not repeat an identical parameter configuration, and retain broad exploration "
            "when transferred configurations underperform. Never compare raw scores across different "
            "tasks; the supplied history is ranked by normalized improvement and compatibility.\n"
            f"Suggested initial configurations: {json.dumps(suggested, default=str)}\n"
            f"Compatible reused trials: {json.dumps(reused, default=str)}\n"
            f"Known parameter hashes to avoid duplicating: {json.dumps(duplicate_hashes)}\n"
            f"Fine-tuning trigger context: {json.dumps(tuning_context, default=str)}\n"
        )

    @staticmethod
    def _validate_tuning_metadata(
        result_data: Dict[str, Any],
        fidelity_profile: dict,
        allowed_parameters: Optional[list[str]] = None,
    ) -> tuple[dict, int]:
        """Validate that a tuning run stayed inside harness-owned search limits."""
        hyperparameters = result_data.get("hyperparameters")
        if not isinstance(hyperparameters, dict) or not hyperparameters:
            raise ValueError(
                "fine-tuning result must include non-empty hyperparameters"
            )
        raw_tuning_trials = result_data.get("tuning_trials", 0)
        if (
            isinstance(raw_tuning_trials, bool)
            or not isinstance(raw_tuning_trials, (int, float))
            or not math.isfinite(float(raw_tuning_trials))
            or not float(raw_tuning_trials).is_integer()
        ):
            raise ValueError("fine-tuning tuning_trials must be an integer")
        tuning_trials = int(raw_tuning_trials)
        if tuning_trials < 1:
            raise ValueError("fine-tuning result must complete at least one trial")
        if tuning_trials > int(fidelity_profile["max_tuning_trials"]):
            raise ValueError("fine-tuning result exceeds the fidelity trial cap")

        epoch_names = {"epochs", "n_epochs", "num_epochs", "max_epochs"}
        patience_names = {"patience", "early_stopping_patience"}
        iteration_names = {
            "iterations",
            "n_estimators",
            "num_iterations",
            "num_boost_round",
            "max_iter",
        }
        caps = (
            (epoch_names, int(fidelity_profile["max_epochs"]), "epoch"),
            (
                patience_names,
                int(fidelity_profile["early_stopping_patience"]),
                "early-stopping patience",
            ),
            (
                iteration_names,
                int(fidelity_profile["max_estimator_iterations"]),
                "estimator iteration",
            ),
        )
        def iter_parameters(mapping):
            for raw_name, raw_value in mapping.items():
                if isinstance(raw_value, dict):
                    yield from iter_parameters(raw_value)
                else:
                    yield raw_name, raw_value

        allowed = (
            None
            if allowed_parameters is None
            else {
                str(name).strip().lower() for name in allowed_parameters
            }
        )
        for raw_name, raw_value in iter_parameters(hyperparameters):
            name = str(raw_name).strip().lower()
            if allowed is not None and name not in allowed:
                raise ValueError(
                    f"fine-tuning parameter {raw_name!r} is not declared tunable"
                )
            for aliases, cap, label in caps:
                if name not in aliases:
                    continue
                if (
                    isinstance(raw_value, bool)
                    or not isinstance(raw_value, (int, float))
                    or not math.isfinite(float(raw_value))
                    or not float(raw_value).is_integer()
                ):
                    raise ValueError(
                        f"fine-tuning {raw_name!r} must be an integer"
                    )
                value = int(raw_value)
                if value < 1 or value > cap:
                    raise ValueError(
                        f"fine-tuning {label} {raw_name!r}={value} exceeds "
                        f"the allowed range 1..{cap}"
                    )
        return hyperparameters, tuning_trials

    @staticmethod
    def _uses_locked_artifact(code: str, model_card: dict) -> bool:
        """Require a fine-tuning script to consume the selected artifact output."""
        if not model_card:
            return True
        module_name = Path(str(model_card.get("code_path", ""))).stem
        entrypoint = str(
            (model_card.get("interface") or {}).get("entrypoint", "")
        ).split("(", 1)[0].strip()
        if not module_name or not entrypoint:
            return False
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        direct_names = set()
        module_aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == module_name:
                for imported in node.names:
                    if imported.name == entrypoint:
                        direct_names.add(imported.asname or imported.name)
            elif isinstance(node, ast.Import):
                for imported in node.names:
                    if imported.name == module_name:
                        module_aliases.add(imported.asname or imported.name)
        parents = {
            id(child): parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }

        def artifact_call(node):
            return (
                isinstance(node.func, ast.Name)
                and node.func.id in direct_names
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == entrypoint
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in module_aliases
            )

        def assigned_names(target):
            return {
                child.id
                for child in ast.walk(target)
                if isinstance(child, ast.Name)
            }

        def enclosing_function(node):
            current = parents.get(id(node))
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return current.name
                current = parents.get(id(current))
            return None

        def in_constant_dead_branch(node):
            current = parents.get(id(node))
            while current is not None:
                if (
                    isinstance(current, ast.If)
                    and isinstance(current.test, ast.Constant)
                    and not bool(current.test.value)
                ):
                    return True
                current = parents.get(id(current))
            return False

        # Mark functions reachable from module-level calls. A mere name reference
        # (or a recursive self-reference) is not enough to make dead code valid.
        reachable_functions = set()
        changed = True
        while changed:
            changed = False
            for candidate in ast.walk(tree):
                if (
                    not isinstance(candidate, ast.Call)
                    or not isinstance(candidate.func, ast.Name)
                    or in_constant_dead_branch(candidate)
                ):
                    continue
                scope = enclosing_function(candidate)
                if scope is None or scope in reachable_functions:
                    if candidate.func.id not in reachable_functions:
                        reachable_functions.add(candidate.func.id)
                        changed = True

        def is_live(node):
            return (
                not in_constant_dead_branch(node)
                and (
                    enclosing_function(node) is None
                    or enclosing_function(node) in reachable_functions
                )
            )

        artifact_call_ids = {
            id(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and artifact_call(node) and is_live(node)
        }
        if not artifact_call_ids:
            return False

        def contains_taint(node, names):
            return any(
                id(child) in artifact_call_ids
                or (
                    isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Load)
                    and child.id in names
                )
                for child in ast.walk(node)
            )

        tainted_names = set()
        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                if not is_live(node):
                    continue
                targets = []
                value = None
                if isinstance(node, ast.Assign):
                    targets, value = node.targets, node.value
                elif isinstance(node, ast.AnnAssign):
                    targets, value = [node.target], node.value
                elif isinstance(node, ast.AugAssign):
                    targets, value = [node.target], node.value
                if targets and value is not None and (
                    contains_taint(value, tainted_names)
                    or any(
                        isinstance(target, ast.Name)
                        and target.id in tainted_names
                        for target in targets
                    )
                ):
                    discovered = set().union(
                        *(assigned_names(target) for target in targets)
                    )
                    if not discovered.issubset(tainted_names):
                        tainted_names.update(discovered)
                        changed = True
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr
                    in {"append", "extend", "update", "fit", "partial_fit"}
                    and contains_taint(node, tainted_names)
                ):
                    receiver_names = assigned_names(node.func.value)
                    if not receiver_names.issubset(tainted_names):
                        tainted_names.update(receiver_names)
                        changed = True

        output_methods = {"to_csv", "savetxt", "save", "write_text"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not is_live(node):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr in output_methods:
                output_expression = ast.Tuple(
                    elts=[
                        node.func.value,
                        *node.args,
                        *(keyword.value for keyword in node.keywords),
                    ],
                    ctx=ast.Load(),
                )
                if contains_taint(output_expression, tainted_names):
                    return True
        return False

    @staticmethod
    def _model_family_imports(code: str) -> set[str]:
        """Extract model-library imports while ignoring preprocessing utilities."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        model_roots = {
            "catboost",
            "lightgbm",
            "xgboost",
            "torch",
            "tensorflow",
            "keras",
            "pytorch_tabnet",
        }
        sklearn_model_modules = {
            "ensemble",
            "linear_model",
            "naive_bayes",
            "neighbors",
            "neural_network",
            "svm",
            "tree",
            "discriminant_analysis",
            "gaussian_process",
        }
        families = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for imported in node.names:
                    parts = imported.name.split(".")
                    if parts[0] in model_roots:
                        families.add(parts[0])
                    elif (
                        len(parts) > 1
                        and parts[0] == "sklearn"
                        and parts[1] in sklearn_model_modules
                    ):
                        families.add(".".join(parts[:2]))
            elif isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if parts[0] in model_roots:
                    families.add(parts[0])
                elif (
                    len(parts) > 1
                    and parts[0] == "sklearn"
                    and parts[1] in sklearn_model_modules
                ):
                    families.add(".".join(parts[:2]))
        return families

    @staticmethod
    def _dependency_fallback_issues(
        code: str, technique_record: dict
    ) -> list[str]:
        """Prevent an unavailable method from masquerading behind try/except."""
        if technique_record.get("status") != "dependency_fallback":
            return []
        unavailable = technique_record.get("unavailable_artifact") or {}
        artifact_id = str(unavailable.get("artifact_id") or "").strip()
        import_roots = {
            "scikit-learn": "sklearn",
            "pytorch-tabnet": "pytorch_tabnet",
            "pytorch-lightning": "pytorch_lightning",
            "opencv-python": "cv2",
            "imbalanced-learn": "imblearn",
        }
        blocked = {artifact_id} if artifact_id else set()
        for dependency in unavailable.get("dependencies", []):
            try:
                package = Requirement(str(dependency)).name.lower()
            except ValueError:
                package = str(dependency).strip().lower()
            blocked.add(
                import_roots.get(package, package.replace("-", "_"))
            )
        blocked.discard("")
        if not blocked:
            return []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        violations = sorted(imported & blocked)
        if not violations:
            return []
        return [
            "dependency-fallback code must not import unavailable modules "
            f"{violations}; implement the feasible branch directly instead of "
            "silently catching the import and running another model"
        ]

    @staticmethod
    def _artifact_evaluation_contract_issues(
        code: str,
        model_card: dict | None,
        evaluation_mode: str = "cross_validation",
    ) -> list[str]:
        """Require CV artifacts to consume harness-owned fold assignments."""
        if evaluation_mode != "cross_validation":
            return []
        capabilities = (
            model_card.get("capabilities", {})
            if isinstance(model_card, dict)
            else {}
        )
        if not isinstance(capabilities, dict) or (
            capabilities.get("accepts_harness_fold_ids") is not True
        ):
            return []
        entrypoint = str(
            (model_card.get("interface", {}) or {}).get("entrypoint", "")
        ).split("(", 1)[0].strip()
        if not entrypoint:
            return []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Name)
                    and node.func.id == entrypoint
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == entrypoint
                )
            )
        ]
        if calls and not any(
            any(
                keyword.arg == "fold_ids"
                for keyword in call.keywords
            )
            for call in calls
        ):
            return [
                f"artifact entrypoint {entrypoint} must receive "
                "fold_ids=fold_ids for its evaluation call"
            ]
        return []

    def _unavailable_import_issues(
        self, code: str, node_dir: Path
    ) -> list[str]:
        """Reject imports that the selected interpreter cannot resolve."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
            ):
                roots.add(node.module.split(".", 1)[0])
        roots -= set(getattr(sys, "stdlib_module_names", ()))
        roots = {
            root
            for root in roots
            if not (node_dir / f"{root}.py").is_file()
            and not (node_dir / root).is_dir()
        }
        if not roots:
            return []
        probe = (
            "import importlib.util,json,sys;"
            "print(json.dumps([name for name in sys.argv[1:] "
            "if importlib.util.find_spec(name) is None]))"
        )
        try:
            probe_env = os.environ.copy()
            project_root = Path(
                getattr(
                    self,
                    "project_root",
                    Path(__file__).resolve().parents[1],
                )
            )
            probe_env["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    (
                        str(project_root),
                        probe_env.get("PYTHONPATH", ""),
                    ),
                )
            )
            result = subprocess.run(
                [self.venv_python, "-c", probe, *sorted(roots)],
                cwd=node_dir,
                capture_output=True,
                text=True,
                timeout=15,
                env=probe_env,
            )
            if result.returncode != 0:
                return []
            unavailable = json.loads(result.stdout.strip())
        except (
            OSError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            TypeError,
        ):
            return []
        if not unavailable:
            return []
        return [
            "selected runtime cannot import modules "
            f"{sorted(unavailable)}; use installed/project-allowlisted packages "
            "or implement the same method with an available lower-level library"
        ]

    @classmethod
    def _tuning_lock_issues(
        cls, code: str, parent_code: str, model_card: dict
    ) -> list[str]:
        issues = []
        if model_card and not cls._uses_locked_artifact(code, model_card):
            issues.append(
                "fine-tuning code must consume the locked artifact entrypoint output"
            )
        parent_families = cls._model_family_imports(parent_code)
        candidate_families = cls._model_family_imports(code)
        introduced = candidate_families - parent_families
        if introduced:
            issues.append(
                "fine-tuning code introduces a different model family: "
                + ", ".join(sorted(introduced))
            )
        if not model_card and parent_families - candidate_families:
            issues.append(
                "fine-tuning code removes the measured parent's model family: "
                + ", ".join(sorted(parent_families - candidate_families))
            )
        return issues

    @staticmethod
    def _resource_limit_issues(code: str, fidelity_profile: dict) -> list[str]:
        """Reject common literal hyperparameters above harness-owned caps."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        caps = {
            "epochs": (int(fidelity_profile["max_epochs"]), "epochs"),
            "n_epochs": (int(fidelity_profile["max_epochs"]), "epochs"),
            "num_epochs": (int(fidelity_profile["max_epochs"]), "epochs"),
            "max_epochs": (int(fidelity_profile["max_epochs"]), "epochs"),
            "patience": (
                int(fidelity_profile["early_stopping_patience"]),
                "early-stopping patience",
            ),
            "early_stopping_patience": (
                int(fidelity_profile["early_stopping_patience"]),
                "early-stopping patience",
            ),
            "iterations": (
                int(fidelity_profile["max_estimator_iterations"]),
                "estimator iterations",
            ),
            "n_estimators": (
                int(fidelity_profile["max_estimator_iterations"]),
                "estimator iterations",
            ),
            "num_iterations": (
                int(fidelity_profile["max_estimator_iterations"]),
                "estimator iterations",
            ),
            "num_boost_round": (
                int(fidelity_profile["max_estimator_iterations"]),
                "estimator iterations",
            ),
            "max_iter": (
                int(fidelity_profile["max_estimator_iterations"]),
                "estimator iterations",
            ),
            "n_trials": (
                int(fidelity_profile["max_tuning_trials"]),
                "tuning trials",
            ),
            "max_trials": (
                int(fidelity_profile["max_tuning_trials"]),
                "tuning trials",
            ),
        }

        def literal_values(value):
            if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
                return [float(value.value)]
            if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                values = []
                for item in value.elts:
                    values.extend(literal_values(item))
                return values
            return []

        candidates = []
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg in caps:
                candidates.append((node.arg, node.value, getattr(node, "lineno", 0)))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in caps:
                        candidates.append((target.id, node.value, getattr(node, "lineno", 0)))
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value in caps:
                        candidates.append((key.value, value, getattr(node, "lineno", 0)))
        issues = []
        for name, value_node, line in candidates:
            cap, label = caps[name]
            over = [value for value in literal_values(value_node) if value > cap]
            if over:
                issues.append(
                    f"line {line}: {label} literal {max(over):g} exceeds fidelity cap {cap}"
                )
        return list(dict.fromkeys(issues))

    def _repair_node_local_artifact(
        self,
        node_dir: Path,
        model_card: dict,
        technique_code: str,
        failure_output: str,
        fidelity_profile: dict,
    ) -> tuple[str, bool, str]:
        """Repair a copied artifact once, then re-verify it without mutating L2."""
        if not model_card or not technique_code:
            return technique_code, False, "no copied artifact is available"
        try:
            artifact_id = validate_storage_identifier(
                model_card.get("artifact_id"), "artifact_id"
            )
            code_name = str(model_card.get("code_path", ""))
            if code_name != f"{artifact_id}.py":
                raise ValueError("code_path must match '<artifact_id>.py'")
            artifact_path = resolve_within(node_dir, code_name)
        except ValueError as exc:
            return technique_code, False, f"invalid local artifact metadata: {exc}"
        if not artifact_path.is_file():
            return technique_code, False, f"copied artifact is missing: {code_name}"

        try:
            response = call_llm(
                "You repair a node-local machine learning artifact after a concrete runtime failure. "
                "Return the complete corrected artifact in one ```python block. Preserve its public "
                "entrypoint and model family, add no dependencies, and make only causal reliability "
                "changes. For neural code, handle mixed/missing inputs with training-fit preprocessing, "
                "use float32 mini-batches on one device, restore the best early-stopped state, predict "
                "in batches on CPU output, and retain accelerator-to-CPU fallback. Never fit on test data.",
                f"""
Artifact model card:
{json.dumps({key: value for key, value in model_card.items() if key != 'verification_log'}, indent=2, default=str)}

Fidelity resource limits:
{json.dumps(fidelity_profile)}

Runtime failure:
```
{str(failure_output)[-8000:]}
```

Current copied artifact:
```python
{technique_code[-20000:]}
```
""",
                model=self.model_name,
                temperature=0.0,
            )
        except Exception as exc:
            return technique_code, False, f"artifact repair LLM call failed: {exc}"
        repaired_code = response
        if "```python" in response:
            repaired_code = response.split("```python", 1)[1].split("```", 1)[0]
        elif "```" in response:
            repaired_code = response.split("```", 1)[1].split("```", 1)[0]
        repaired_code = repaired_code.strip()
        try:
            ast.parse(repaired_code)
        except SyntaxError as exc:
            return technique_code, False, f"artifact repair produced invalid Python: {exc}"
        repair_issues = inspect_generated_code(repaired_code)
        repair_issues.extend(
            self._resource_limit_issues(repaired_code, fidelity_profile)
        )
        original_families = self._model_family_imports(technique_code)
        repaired_families = self._model_family_imports(repaired_code)
        if original_families and repaired_families != original_families:
            repair_issues.append(
                "artifact repair must preserve model-library families exactly; "
                f"expected={sorted(original_families)}, got={sorted(repaired_families)}"
            )
        if repair_issues:
            return (
                technique_code,
                False,
                "artifact repair failed static validation: " + "; ".join(repair_issues),
            )

        original_code = artifact_path.read_text(encoding="utf-8")
        local_card_path = resolve_within(node_dir, f"{artifact_id}.json")
        local_card = {
            key: value
            for key, value in model_card.items()
            if key not in {"verification_log", "task_validations"}
        }
        local_card["verified"] = False
        local_card["verification_log"] = "Pending node-local repair verification."
        artifact_path.write_text(repaired_code + "\n", encoding="utf-8")
        local_card_path.write_text(
            json.dumps(local_card, indent=2, default=str) + "\n", encoding="utf-8"
        )

        verifier = self.project_root / "memory_pool" / "builder" / "sandbox_verifier.py"
        verify_env = accelerator_subprocess_env("cpu")
        verify_env.update(
            {
                "AIBUILDAI_MAX_EPOCHS": "2",
                "AIBUILDAI_EARLY_STOPPING_PATIENCE": "1",
            }
        )
        try:
            verification = subprocess.run(
                [self.venv_python, str(verifier), str(local_card_path)],
                cwd=node_dir,
                capture_output=True,
                text=True,
                timeout=60,
                env=verify_env,
            )
            verification_output = (
                verification.stdout + "\n" + verification.stderr
            ).strip()
            verified = verification.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            verification_output = f"node-local verification failed to run: {exc}"
            verified = False

        audit = {
            "artifact_id": artifact_id,
            "verified": verified,
            "code_sha256": hashlib.sha256(
                repaired_code.encode("utf-8")
            ).hexdigest(),
            "fidelity_limits": fidelity_profile,
            "verification_output_tail": verification_output[-4000:],
        }
        (node_dir / "artifact_repair.json").write_text(
            json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8"
        )
        if not verified:
            artifact_path.write_text(original_code, encoding="utf-8")
            local_card_path.unlink(missing_ok=True)
            return technique_code, False, verification_output[-4000:]
        return repaired_code, True, verification_output[-4000:]

    def _inherit_parent_workspace(self, parent_node_dir: Path, node_dir: Path) -> list[str]:
        """Seed a child with reusable parent artifacts without copying stale outputs/data."""
        inherited = []
        if not parent_node_dir or not parent_node_dir.is_dir():
            return inherited
        excluded = {
            "algorithm.py",
            "result.json",
            "node_state.json",
            "technique_record.json",
            "error.log",
            "oof_predictions.npz",
            "oof_predictions.csv",
            "validation_predictions.npz",
            "validation_predictions.csv",
            "evaluation_manifest.json",
            "fold_assignments.npz",
            "fold_assignments.csv",
            "validation_assignments.npz",
            "validation_assignments.csv",
            "evaluation_policy.json",
            "execution_resource.json",
            "fine_tuning.json",
            "artifact_repair.json",
            "error_analysis.json",
        }
        allowed_suffixes = {
            ".py", ".json", ".yaml", ".yml", ".txt",
        }
        unverified_artifact_files = set()
        for card_path in parent_node_dir.glob("*.json"):
            if card_path.name in excluded or card_path.is_symlink():
                continue
            try:
                card = json.loads(card_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if (
                isinstance(card, dict)
                and card.get("artifact_id")
                and card.get("code_path")
                and card.get("verified") is not True
            ):
                unverified_artifact_files.update(
                    {card_path.name, str(card["code_path"])}
                )
        for source in parent_node_dir.iterdir():
            if (
                not source.is_file()
                or source.is_symlink()
                or source.name in excluded
                or source.name in unverified_artifact_files
                or source.suffix.lower() not in allowed_suffixes
            ):
                continue
            destination = node_dir / source.name
            if destination.exists():
                continue
            shutil.copy2(source, destination)
            inherited.append(source.name)
        return inherited

    def run(
        self,
        node_dir: Path,
        technique_record: dict,
        task_dir: Path,
        timeout: Optional[float] = None,
        stall_seconds: float = 1800.0,
        metric_direction: str = "maximize",
        base_algorithm_path: Optional[Path] = None,
        parent_node_dir: Optional[Path] = None,
        fidelity: str = "full",
        operator: Optional[str] = None,
        enforce_evaluation_contract: bool = False,
        accelerator: str = "cpu",
        available_accelerators: Optional[set[str]] = None,
        tuning_context: Optional[dict] = None,
        max_debug_attempts: int = 3,
        metric_name: Optional[str] = None,
        task_assets_dir: Optional[Path] = None,
        evaluation_mode: Optional[str] = None,
        parallel_processes: int = 1,
    ) -> Dict[str, Any]:
        """
        1. Reads immutable task inputs and harness-generated task metadata.
        2. Calls the LLM to build a root method or evolve a measured parent.
        3. Writes it to node_dir / "algorithm.py".
        4. Runs it and parses result.json for the evaluation metric score.
        
        Args:
            node_dir: Directory for this node's run outputs
            technique_record: Dict from TechniqueAgent with plan/model_card
            task_dir: Read-only task directory with description/config/data
            task_assets_dir: Run directory containing the canonical task
                contract, profile, sample index, and deterministic data loader.
                Direct callers may omit it; the assets are then created locally.
            timeout: Optional hard runtime limit reserved for focused direct
                callers. Normal workflow nodes always pass ``None``.
            stall_seconds: Renewable progress-lease duration. Total runtime is
                unlimited while output, artifacts, or process activity continue.
            metric_direction: "maximize" or "minimize" — used in the prompt
            metric_name: Harness metric resolved by the manager. When omitted,
                use the canonical task contract.
            evaluation_mode: Optional manager-selected validation protocol.
                When omitted, the model-aware evaluation policy selects it.
            parallel_processes: Number of independent training subprocesses in
                the current root batch. CPU thread pools are divided across
                them to avoid oversubscription.
        """
        if timeout is not None and (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError(
                f"timeout must be None or a positive finite number, got {timeout!r}"
            )
        if (
            not isinstance(stall_seconds, (int, float))
            or isinstance(stall_seconds, bool)
            or not math.isfinite(float(stall_seconds))
            or stall_seconds <= 0
        ):
            raise ValueError(
                "stall_seconds must be a positive finite number, "
                f"got {stall_seconds!r}"
            )
        if metric_direction not in {"maximize", "minimize"}:
            raise ValueError(
                f"metric_direction must be 'maximize' or 'minimize', got {metric_direction!r}"
            )
        if not isinstance(max_debug_attempts, int) or max_debug_attempts < 0:
            raise ValueError("max_debug_attempts must be a non-negative integer")
        if tuning_context is not None and not isinstance(tuning_context, dict):
            raise ValueError("tuning_context must be a dictionary when provided")
        if (
            not isinstance(parallel_processes, int)
            or isinstance(parallel_processes, bool)
            or parallel_processes < 1
        ):
            raise ValueError("parallel_processes must be a positive integer")
        accelerator = str(accelerator).lower()
        if accelerator not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported accelerator: {accelerator!r}")
        exposed_accelerators = {
            str(item).lower() for item in (available_accelerators or {"cpu"})
        }
        exposed_accelerators.add("cpu")
        if accelerator not in exposed_accelerators:
            raise ValueError(
                f"Selected accelerator {accelerator!r} is not exposed by this run"
            )
        task_dir = Path(task_dir)
        task_spec = TaskAnalyzer().resolve(task_dir)
        require_submission = task_requires_submission(task_dir, task_spec)
        modality_context = modality_prompt_context(task_spec, fidelity)
        robust_strategy_context = render_strategy_patterns(
            task_spec.to_dict()
        )
        node_dir.mkdir(parents=True, exist_ok=True)
        run_started = time.monotonic()
        execution_resource = {
            "selected_accelerator": accelerator,
            "available_accelerators": sorted(exposed_accelerators),
            "environment_variable": "AIBUILDAI_ACCELERATOR",
            "fallback": "cpu",
            "execution_mode": "renewable_progress_lease",
            "total_runtime_limit_seconds": timeout,
            "progress_stall_seconds": float(stall_seconds),
        }
        with open(node_dir / "execution_resource.json", "w", encoding="utf-8") as f:
            json.dump(execution_resource, f, indent=2)
            f.write("\n")
        inherited_files = self._inherit_parent_workspace(
            Path(parent_node_dir) if parent_node_dir else None, node_dir
        )

        if task_assets_dir is None:
            TaskAnalyzer().analyze(
                task_dir,
                output_dir=node_dir,
                include_index=True,
            )
            (node_dir / "task_dataloader.py").write_text(
                task_loader_source(), encoding="utf-8"
            )
            write_runtime_data_contract(
                node_dir / "dataset_index.jsonl",
                node_dir / "runtime_data_contract.json",
                task_spec_path=node_dir / "resolved_task_spec.json",
                task_dir=task_dir,
            )
        else:
            task_assets_source = Path(task_assets_dir)
            required_task_assets = [
                "task_dataloader.py",
                "resolved_task_spec.json",
                "dataset_profile.json",
                "runtime_data_contract.json",
            ]
            analysis_asset = (
                "dataset_analysis.md"
                if (task_assets_source / "dataset_analysis.md").is_file()
                else "dataset_analysis_report.txt"
            )
            required_task_assets.append(analysis_asset)
            task_profile = json.loads(
                (task_assets_source / "dataset_profile.json").read_text(
                    encoding="utf-8"
                )
            )
            index_manifest = task_profile.get("dataset_index")
            if not isinstance(index_manifest, dict):
                legacy_manifest = (
                    task_assets_source / "dataset_index_manifest.json"
                )
                if legacy_manifest.is_file():
                    index_manifest = json.loads(
                        legacy_manifest.read_text(encoding="utf-8")
                    )
                    required_task_assets.append(
                        "dataset_index_manifest.json"
                    )
                else:
                    raise ValueError(
                        "dataset_profile.json is missing its dataset_index contract"
                    )
            if index_manifest.get("storage") != "direct_tabular":
                required_task_assets.append("dataset_index.jsonl")
            missing = [
                name
                for name in required_task_assets
                if not (task_assets_source / name).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "Method-tree task assets are incomplete: "
                    + ", ".join(missing)
                )
            for name in required_task_assets:
                source = task_assets_source / name
                destination = node_dir / name
                if name == "dataset_index.jsonl":
                    # The generated loader streams this immutable, potentially
                    # very large asset from AIBUILDAI_TASK_ASSETS_DIR.
                    continue
                if source.resolve() != destination.resolve():
                    shutil.copy2(source, destination)
        runtime_contract = json.loads(
            (node_dir / "runtime_data_contract.json").read_text(
                encoding="utf-8"
            )
        )
        concrete_data_context = runtime_data_prompt(runtime_contract)
        # Framework modules are imported from the shared project root through
        # the child PYTHONPATH. Copying core/, evaluation/, and
        # evaluation_contract.py into every node bloats runs and lets stale
        # snapshots drift from the harness that validates them.
        # Copy the memory pool technique python file to node_dir so it can be imported (Fixes ModuleNotFoundError)
        model_card = technique_record.get("model_card", {})
        if model_card and "category" in model_card and "code_path" in model_card:
            try:
                category = validate_storage_identifier(model_card["category"], "category")
                artifact_id = validate_storage_identifier(
                    model_card.get("artifact_id"), "artifact_id"
                )
                code_name = model_card["code_path"]
                if code_name != f"{artifact_id}.py":
                    raise ValueError("code_path must match '<artifact_id>.py'")
                store = self.project_root / "memory_pool" / "l2_store"
                src_tech = resolve_within(store, category, code_name)
                dest_tech = resolve_within(node_dir, code_name)
                if dest_tech.is_file():
                    print(
                        "ImplementationAgent: Reusing inherited node-local "
                        f"artifact {dest_tech}"
                    )
                elif src_tech.is_file():
                    shutil.copy(src_tech, dest_tech)
                    print(f"ImplementationAgent: Copied technique from {src_tech} to {dest_tech}")
                else:
                    print(f"ImplementationAgent WARNING: Technique code file not found at {src_tech}")
            except ValueError as exc:
                raise ValueError(f"Unsafe or invalid model card: {exc}") from exc

        # Dataloaders use ./input while the dataset remains owned by tasks/<task>.
        linked_inputs = expose_task_data(task_dir, node_dir)
        print(
            "ImplementationAgent: Linked task-owned input data into "
            f"{node_dir / 'input'} ({len(linked_inputs)} link(s); no dataset copy)."
        )

        # Descendants evolve a measured parent. Root candidates are generated
        # directly from their method plans, with no starter model.
        if base_algorithm_path is not None:
            src_algo = Path(base_algorithm_path)
            if not src_algo.is_file():
                raise FileNotFoundError(
                    f"Measured parent algorithm does not exist: {src_algo}"
                )
            original_code = src_algo.read_text(encoding="utf-8")
        else:
            original_code = ""
            
        # Direct callers may omit the metric; tree-search callers pass the
        # manager's already-resolved value so generation and validation agree.
        if metric_name is None:
            metric_name = task_spec.primary_metric
            metric_direction = task_spec.metric_direction
            
        # Never execute an artifact that failed verification.
        if model_card and model_card.get("verified") is not True:
            raise ValueError(
                f"Model card {model_card.get('artifact_id', '<unknown>')!r} is not verified"
            )
        if evaluation_mode is None:
            evaluation_policy = select_evaluation_policy(
                task_spec,
                technique_record,
                model_card,
                operator=operator,
            )
        else:
            evaluation_mode = normalize_evaluation_mode(
                evaluation_mode, allow_auto=False
            )
            evaluation_policy = EvaluationPolicy(
                mode=evaluation_mode,
                reason="evaluation mode preserved by the manager",
                source="manager",
            )
        evaluation_mode = evaluation_policy.mode
        execution_resource["evaluation_mode"] = evaluation_mode
        execution_resource["evaluation_policy_reason"] = (
            evaluation_policy.reason
        )
        with open(
            node_dir / "execution_resource.json", "w", encoding="utf-8"
        ) as stream:
            json.dump(execution_resource, stream, indent=2)
            stream.write("\n")
        (node_dir / "evaluation_policy.json").write_text(
            json.dumps(evaluation_policy.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
            
        # Get dataset analysis report to pass to LLM
        dataset_snapshot = ""
        diagnostic_directive_context = ""
        report_file = node_dir / "dataset_analysis.md"
        if not report_file.is_file():
            legacy_report = node_dir / "dataset_analysis_report.txt"
            if legacy_report.is_file():
                report_file = legacy_report
        try:
            task_profile = json.loads(
                (node_dir / "dataset_profile.json").read_text(
                    encoding="utf-8"
                )
            )
            diagnostics = task_profile.get("diagnostics", {})
            directives = (
                diagnostics.get("synthesized_directives", [])
                if isinstance(diagnostics, dict)
                else []
            )
            if isinstance(directives, list):
                diagnostic_directive_context = "\n".join(
                    f"- {str(item)}" for item in directives if str(item).strip()
                )
            if report_file.exists():
                with open(report_file, 'r', encoding='utf-8') as f:
                    analysis_report = f.read()
            else:
                from .data_analyzer import run_dataset_analysis
                print(f"ImplementationAgent: Checking/running dataset analysis fallback for {task_dir.name}...")
                analysis_report = run_dataset_analysis(task_dir)
                (node_dir / "dataset_analysis.md").write_text(
                    analysis_report, encoding="utf-8"
                )
                    
            dataset_snapshot = (
                "=== Dataset Analysis & Schema Report ===\n"
                f"{analysis_report}\n"
                "=== Canonical Task Contract ===\n"
                f"{json.dumps(task_spec.to_dict(), indent=2)}\n"
                "=== Concrete Runtime Data Contract ===\n"
                f"{concrete_data_context}\n"
                "========================================\n"
            )
        except Exception as e:
            print(f"ImplementationAgent WARNING: Could not get dataset analysis report: {e}")
            dataset_snapshot = ""

        # Read the technique source code if it exists (local or global)
        technique_code = ""
        if model_card and "code_path" in model_card:
            category = model_card.get("category", "")
            code_name = model_card["code_path"]
            # Look in node_dir (local) or global store
            tech_file = node_dir / code_name
            if not tech_file.exists():
                tech_file = Path(self.project_root) / "memory_pool" / "l2_store" / category / code_name
            if tech_file.exists():
                try:
                    with open(tech_file, 'r', encoding='utf-8') as f:
                        technique_code = f.read()
                except Exception as e:
                    print(f"ImplementationAgent WARNING: Could not read technique code: {e}")

        tech_code_str = ""
        if technique_code:
            tech_code_str = f"Chosen Technique Source Code:\n```python\n{technique_code}\n```"

        # Call LLM to generate glue code. The artifact file is copied beside
        # algorithm.py, so importing an invented memory_pool package is incorrect.
        if model_card and technique_code:
            module_name = Path(model_card["code_path"]).stem
            entrypoint_signature = model_card.get("interface", {}).get("entrypoint", "")
            entrypoint_name = entrypoint_signature.split("(", 1)[0].strip()
            integration_instruction = (
                f"Import the verified local artifact exactly from module {module_name!r}; for example, "
                f"`from {module_name} import {entrypoint_name}`. Do not import from a `memory_pool` package "
                "and do not reimplement the artifact's internal logic."
            )
            if (
                evaluation_mode == "cross_validation"
                and
                model_card.get("capabilities", {}).get(
                    "accepts_harness_fold_ids"
                )
                is True
            ):
                integration_instruction += (
                    " For the OOF evaluation call, pass the harness array "
                    "explicitly as `fold_ids=fold_ids`; do not let the artifact "
                    "create an independent split."
                )

        else:
            integration_instruction = (
                "No verified artifact is available for this node. Implement the "
                "chosen technique as a complete self-contained method. If a measured "
                "parent is supplied, change it according to the operator; otherwise "
                "design the root pipeline directly from the task contract. Do not "
                "import any `memory_pool` module or imaginary artifact."
            )

        if fidelity not in FIDELITY_PROFILES:
            raise ValueError("fidelity must be 'screen', 'medium', or 'full'")
        fidelity_profile = FIDELITY_PROFILES[fidelity]
        operator_instruction = {
            "root": (
                "Build the complete root pipeline directly from this method "
                "plan; there is no starter or parent implementation."
            ),
            "refine": "Modify only the highest-impact relevant component; preserve working parent behavior elsewhere.",
            "tune": "Act as a fine-tuner: preserve the measured architecture and run a compact pruned hyperparameter search with a fixed seed.",
            "diversify": "Favor a sound model or representation whose errors are likely less correlated with the parent.",
            "promote": "Preserve the parent method and evaluate it more rigorously at the requested fidelity.",
        }.get(operator or "", "Apply the requested technique as a focused change to the parent pipeline.")
        fine_tuning_instruction = self._fine_tuning_instruction(
            operator, tuning_context, fidelity_profile
        )
        deep_preprocessing_scope = (
            "fold-local"
            if evaluation_mode == "cross_validation"
            else "training-split-only"
        )
        deep_lifecycle = (
            "release models/tensors between folds"
            if evaluation_mode == "cross_validation"
            else "train only one validation model"
        )
        deep_learning_instruction = (
            "Deep-learning execution contract (when applicable): use "
            f"{deep_preprocessing_scope} numeric/categorical "
            "preprocessing; float32 features; the loss-appropriate label dtype; mini-batch DataLoaders; "
            "model and batches on the same selected device; validation-based early stopping with the "
            "best state restored; and CPU-detached predictions. Never place the full dataset on GPU or "
            f"retain unnecessary models/tensors; {deep_lifecycle}. "
            f"Use no more than {fidelity_profile['max_epochs']} epochs and patience "
            f"{fidelity_profile['early_stopping_patience']} at this fidelity. Print concise progress "
            "at least once per epoch so long-running training remains observable.\n"
        )
        tuning_result_fields = (
            ', "hyperparameters": {"parameter_name": <selected value>}, '
            '"tuning_trials": <int>'
            if operator == "tune" and tuning_context
            else ""
        )
        structured_prediction_output = task_spec.output.type in {
            "boxes",
            "embeddings",
            "masks",
            "ranked_items",
            "text",
        }
        if evaluation_mode == "cross_validation":
            prediction_artifact_instruction = (
                "Import `write_structured_predictions` from the local "
                "`evaluation_contract` module and call it with output_dir='.', "
                "sample_ids=row_ids, predictions=<aligned N-D or structured "
                "predictions>, targets=y_eval, fold_ids=fold_ids, "
                "evaluation_meta=evaluation_meta, and "
                f"output_type={task_spec.output.type!r}. This must create "
                "`predictions/manifest.json`; do not flatten masks, boxes, text, "
                "or ranked outputs into a scalar table."
                if structured_prediction_output
                else
                "Import `write_prediction_table` from "
                "`evaluation.prediction_io` and save every scheduled row to "
                "`oof_predictions.npz`, passing sample_ids=row_ids, "
                "targets=y_eval, predictions, fold_ids, and stable class_names "
                "for class-probability output. Do not serialize evaluation "
                "arrays through pandas CSV."
            )
            evaluation_contract_prompt = (
                "SELECTED EVALUATION MODE: cross_validation. OOF is required "
                "because this model was classified as fold-independent and "
                "appropriate for CV/model comparison.\n"
                "Import `prepare_evaluation_data` from the local "
                "`evaluation_contract` module and call:\n"
                f"  X_eval, y_eval, row_ids, fold_ids, evaluation_meta = "
                f"prepare_evaluation_data(train_data, '{fidelity}', "
                "evaluation_mode='cross_validation')\n"
                "Use X_eval/y_eval for every cross-validation operation and "
                "use the supplied fold_ids exactly. The harness has already "
                "applied fidelity sampling: never subsample these rows or "
                "create an independent split. Fit learned preprocessing on "
                "each fold's training rows only. "
                f"{prediction_artifact_instruction}\n"
            )
            evaluation_output_prompt = (
                "Persist the complete aligned evaluation output described above so the "
                "harness can independently recompute every fold metric."
            )
            result_statistics_prompt = (
                f'"evaluation_mode": "cross_validation", '
                '"cv_mean": <float>, "cv_std": <float>, "folds": <int>, '
            )
            progress_unit = "fold"
        elif evaluation_mode == "holdout":
            prediction_artifact_instruction = (
                "Import `write_structured_predictions` from the local "
                "`evaluation_contract` module and call it with output_dir='.', "
                "sample_ids=validation_row_ids, predictions=<aligned N-D or "
                "structured predictions>, targets=y_valid, "
                "evaluation_meta=evaluation_meta, and "
                f"output_type={task_spec.output.type!r}. This must create "
                "`predictions/manifest.json`; do not flatten structured outputs "
                "into a scalar table."
                if structured_prediction_output
                else
                "Import `write_prediction_table` from "
                "`evaluation.prediction_io` and write "
                "`validation_predictions.npz` for exactly the validation rows, "
                "passing sample_ids=validation_row_ids, targets=y_valid, "
                "predictions, and stable class_names for probability output. "
                "Do not serialize evaluation arrays through pandas CSV."
            )
            evaluation_contract_prompt = (
                "SELECTED EVALUATION MODE: holdout. OOF IS NOT REQUIRED for "
                "this model. Do not create a CV loop and do not write "
                "`oof_predictions.npz`.\n"
                "Import `prepare_holdout_evaluation_data` from the local "
                "`evaluation_contract` module and call:\n"
                f"  X_train, y_train, X_valid, y_valid, "
                f"validation_row_ids, evaluation_meta = "
                f"prepare_holdout_evaluation_data(train_data, '{fidelity}')\n"
                "Train the selected model exactly once on X_train/y_train. "
                "Fit all preprocessing only on X_train and use X_valid only "
                "for evaluation, early stopping, and model selection. The "
                "split is harness-owned; never call train_test_split or make "
                f"another split. {prediction_artifact_instruction}\n"
            )
            evaluation_output_prompt = (
                (
                    "Persist `predictions/manifest.json`; OOF files are not part "
                    "of this node's contract."
                    if structured_prediction_output
                    else
                    "Persist `validation_predictions.npz`; OOF files are not part "
                    "of this node's contract."
                )
            )
            result_statistics_prompt = (
                f'"evaluation_mode": "holdout", '
                '"validation_score": <float>, "folds": 1, '
            )
            progress_unit = "training stage"
        else:
            evaluation_contract_prompt = (
                "SELECTED EVALUATION MODE: task_native. OOF IS NOT REQUIRED "
                "because this task has no supervised fold target. Do not write "
                "`oof_predictions.npz`.\n"
                "Import `prepare_evaluation_data` and "
                "`evaluate_clustering_predictions` from the local "
                "`evaluation_contract` module. Call:\n"
                f"  X_eval, y_eval, row_ids, fold_ids, evaluation_meta = "
                f"prepare_evaluation_data(train_data, '{fidelity}', "
                "evaluation_mode='task_native')\n"
                "Fit the unsupervised method once on X_eval, then call "
                "`evaluate_clustering_predictions(X_eval, labels, row_ids, "
                "fold_ids, fidelity=evaluation_meta['fidelity'])`. The helper "
                "writes `validation_predictions.npz` and computes the bounded "
                "task-native validation proxy. The fold IDs are scoring "
                "partitions only; do not train one model per fold.\n"
            )
            evaluation_output_prompt = (
                "Persist the task-native validation output; do not create OOF."
            )
            result_statistics_prompt = (
                f'"evaluation_mode": "task_native", '
                '"validation_score": <float>, "folds": <int>, '
            )
            progress_unit = "training stage"

        system_prompt = (
            "You are the Implementation Agent. Produce a complete executable "
            "method. When a measured parent is supplied, evolve it according "
            "to the selected operator. "
            f"{integration_instruction} Ensure the output is valid Python code wrapped in a ```python block.\n"
            f"Search operator: {operator or 'root'}. {operator_instruction}\n"
            f"Evaluation fidelity: {fidelity} ({json.dumps(fidelity_profile)}). These limits are mandatory.\n"
            "CRITICAL MODEL ARCHITECTURE CONTRACT:\n"
            "- You MUST implement the complete representation, preprocessing, feature, robustness, and model hypothesis specified in the Chosen Technique Plan.\n"
            "- The model family is one pipeline component; do not collapse a feature/representation branch into generic model.fit code.\n"
            "- Do NOT fall back to or re-execute an unchanged parent pipeline from another node.\n"
            "- If an optional package is unavailable, implement the intended mechanism self-contained with importable project libraries.\n"
            "ROBUST PIPELINE DESIGN PRINCIPLES (apply only when supported by dataset diagnostics):\n"
            f"{robust_strategy_context}\n"
            "AUTHORITATIVE DATASET-DIAGNOSTIC DIRECTIVES:\n"
            f"{diagnostic_directive_context or '- No synthesized directive was available.'}\n"
            "Use these directives as evidence, not as permission to leak test "
            "statistics or fit preprocessing outside training folds.\n"
            f"{fine_tuning_instruction}"
            f"{deep_learning_instruction}"
            f"Modality correctness contract:\n{modality_context}\n"
            f"Execution accelerator: {accelerator}; available={sorted(exposed_accelerators)}. "
            "The subprocess also exposes this value as AIBUILDAI_ACCELERATOR. When the selected accelerator is "
            "CUDA or MPS, use the framework-native GPU/device option for every compatible training component "
            "and move neural-network models and tensors to that device. Check that the framework backend is "
            "usable and fall back to CPU only when that model/library has no working backend for the selected "
            "accelerator. When the technique permits equivalent model families, prefer a GPU-capable learner "
            "over a CPU-only one. Do not send small preprocessing or unsupported scikit-learn estimators to a GPU. "
            "AIBUILDAI_ACTUAL_ACCELERATOR starts as 'cpu'. Change it to 'cuda' or "
            "'mps' only after this script successfully trains or infers with that "
            "backend, then report the environment value in result.json.\n"
            "CRITICAL DATA LOADING CONTRACT:\n"
            "1. You must ALWAYS load the dataset first using the custom loader:\n"
            "   from task_dataloader import TaskDataLoader\n"
            "   loader = TaskDataLoader()\n"
            "   train_data, test_data = loader()\n"
            "2. Pass the loaded `train_data` dictionary directly to the "
            "selected evaluation helper. Never read raw files directly or "
            "pass placeholder dictionaries to an evaluation helper.\n"
            f"3. {concrete_data_context}\n"
            f"{evaluation_contract_prompt}"
            "Import `metric_value` from `evaluation.metrics` and call it with the exact signature "
            f"`metric_value('{metric_name}', y_true, y_pred)` for validation reporting; the metric "
            "name is always the FIRST argument and must not also be passed as a keyword. Never "
            "replace the task metric with a model-default accuracy/loss.\n"
            "Fit all learned preprocessing (imputers, encoders, scalers, tokenizers, feature extractors, target-dependent transforms) "
            "only on the selected protocol's training data. Never use test-set statistics or target values in feature engineering or model fitting. "
            "Define data preprocessing once and apply it symmetrically to training, validation, full-training, and test data.\n"
            + (
                "FINAL-PREDICTION CONTRACT (MANDATORY FOR THIS TASK):\n"
                "After the selected validation protocol, import and call "
                "`prepare_final_training_data(train_data, test_data)` from "
                "`evaluation_contract`. Fit a fresh final model and fresh preprocessing "
                "on ALL returned full-training rows, never on only the evaluation subset. Predict every "
                "returned test row and write `submission/submission.csv`. Use test_ids as "
                "the first column and the exact output column names/order in "
                "`resolved_task_spec.json`; multiclass probabilities must be finite, "
                "within [0,1], and sum to one per row. The implementation is incomplete "
                "without both `final_training_manifest.json` and the submission file. "
                "`prepare_final_training_data` owns the manifest; NEVER create, overwrite, "
                "or append to that file yourself. The exact submission ID and prediction "
                "column names are listed in resolved_task_spec.json under output.options. "
                "For structured outputs, obey submission_encoding, rle_flatten_order, "
                "rle_index_base, and rle_pair_format exactly; encode each prediction "
                "rather than writing an array or Python representation into a CSV cell.\n"
                if require_submission
                else ""
            )
            +
            f"Emit and flush a concise progress line before and after every {progress_unit}, "
            "training stage, and tuning trial so the workflow can distinguish "
            "healthy long-running work from a stalled process.\n"
            f"{evaluation_output_prompt}\n"
            f"IMPORTANT: At the END of your script, write a JSON file 'result.json' in the current directory:\n"
            f'  import json; json.dump({{"score": <float>, "metric": "{metric_name}", "direction": "{metric_direction}", '
            f'{result_statistics_prompt}"fidelity": "{fidelity}", '
            f'"accelerator": <actual "cpu"|"cuda"|"mps">{tuning_result_fields}}}, open("result.json", "w"))\n'
            "The score must be the metric from the selected evaluation protocol."
        )
        
        technique_plan = technique_record.get("plan", "")
        
        prompt_model_card = dict(model_card or {})
        prompt_model_card.pop("code_content", None)
        if prompt_model_card.get("verification_log"):
            prompt_model_card["verification_log"] = str(
                prompt_model_card["verification_log"]
            )[-1200:]
        user_prompt = f"""
            Parent implementation:
            {f"```python{chr(10)}{original_code}{chr(10)}```" if original_code else "None. This is a root method; build the complete pipeline directly from the chosen technique plan."}

            Chosen Technique Plan:
            {technique_plan}

            Model Card details:
            {json.dumps(prompt_model_card, indent=2) if prompt_model_card else "None"}

            {tech_code_str}

            {dataset_snapshot}

            {integration_instruction}
            Inherited reusable files: {inherited_files}
            Return the complete updated file content.
            """
        response = call_llm(system_prompt, user_prompt, model=self.model_name)
        
        # Clean markdown code block formatting
        clean_code = response
        if "```python" in response:
            clean_code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            clean_code = response.split("```")[1].split("```")[0]

        def execution_contract_issues(candidate_code: str) -> list[str]:
            issues = inspect_generated_code(
                candidate_code,
                task_spec=task_spec.to_dict(),
                runtime_contract=runtime_contract,
            )
            issues.extend(
                self._resource_limit_issues(candidate_code, fidelity_profile)
            )
            issues.extend(
                self._dependency_fallback_issues(
                    candidate_code, technique_record
                )
            )
            issues.extend(
                self._artifact_evaluation_contract_issues(
                    candidate_code, model_card, evaluation_mode
                )
            )
            issues.extend(
                self._unavailable_import_issues(candidate_code, node_dir)
            )
            if operator == "tune" and tuning_context:
                issues.extend(
                    self._tuning_lock_issues(
                        candidate_code, original_code, model_card
                    )
                )
            return list(dict.fromkeys(issues))

        guard_issues = execution_contract_issues(clean_code)
        if guard_issues:
            print(
                "ImplementationAgent: Validation guard found contract risks; "
                "requesting a pre-execution repair."
            )
            repair_response = call_llm(
                "You are an ML execution-contract reviewer. Repair every listed defect while "
                "preserving the intended and locked model family. Fit preprocessing only on training "
                "folds, obey fidelity resource caps, and call the selected local artifact entrypoint "
                "when one is locked. Return the complete corrected Python file in a ```python block.",
                f"""
Leakage defects:
{json.dumps(guard_issues, indent=2)}

Authoritative runtime data contract:
{concrete_data_context}

Code:
```python
{clean_code}
```
""",
                model=self.model_name,
            )
            if "```python" in repair_response:
                clean_code = repair_response.split("```python", 1)[1].split("```", 1)[0]
            elif "```" in repair_response:
                clean_code = repair_response.split("```", 1)[1].split("```", 1)[0]
            else:
                clean_code = repair_response
            remaining_issues = execution_contract_issues(clean_code)
            if remaining_issues:
                raise ValueError(
                    "Generated implementation failed execution-contract guard after repair: "
                    + "; ".join(remaining_issues)
                )
            
        dest_code_file = node_dir / "algorithm.py"
        with open(dest_code_file, 'w', encoding='utf-8') as f:
            f.write(clean_code.strip())

        code_fingerprint = self._candidate_code_fingerprint(
            clean_code,
            task_spec=task_spec,
            evaluation_mode=evaluation_mode,
        )
        duplicate_of = self._register_candidate_fingerprint(
            (
                Path(task_assets_dir)
                if task_assets_dir is not None
                else node_dir.parent
            ),
            fingerprint=code_fingerprint,
            node_id=node_dir.name,
        )
        (node_dir / "candidate_fingerprint.json").write_text(
            json.dumps(
                {
                    "fingerprint": code_fingerprint,
                    "duplicate_of": duplicate_of,
                    "evaluation_mode": evaluation_mode,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if duplicate_of is not None:
            duplicate_result = {
                "score": None,
                "status": "skipped_duplicate_pre_execution",
                "duplicate_of": duplicate_of,
                "code_fingerprint": code_fingerprint,
                "diagnostics": (
                    "Candidate code structure and literal hyperparameters "
                    f"duplicate {duplicate_of}; training was not started."
                ),
            }
            (node_dir / "result.json").write_text(
                json.dumps(duplicate_result, indent=2) + "\n",
                encoding="utf-8",
            )
            return {
                **duplicate_result,
                "code_path": str(dest_code_file),
                "base_code_path": (
                    str(base_algorithm_path)
                    if base_algorithm_path is not None
                    else None
                ),
                "parent_node_dir": (
                    str(parent_node_dir) if parent_node_dir else None
                ),
                "operator": operator,
                "fidelity": fidelity,
                "evaluation_mode": evaluation_mode,
                "evaluation_policy": evaluation_policy.to_dict(),
                "elapsed_seconds": time.monotonic() - run_started,
                "validation": {},
                "oof_path": None,
                "validation_path": None,
            }
            
        print(f"ImplementationAgent: Wrote glue code to {dest_code_file}")
                    
        cmd = [self.venv_python, "algorithm.py"]
        
        # Debug/Coder retry loop
        max_attempts = max_debug_attempts + 1
        attempt = 0
        score = None
        score_source = "none"
        status = "completed"
        stdout = ""
        stderr = ""
        exit_code = 0
        timeout_kill = False
        execution_stalled = False
        hard_limit_reached = False
        termination_reason = None
        progress_events = 0
        last_progress_source = "not_started"
        last_progress_age_seconds = 0.0
        diagnostics = ""
        result_data: Dict[str, Any] = {}
        artifact_repair_attempted = False
        artifact_repair_summary = None
        
        while attempt < max_attempts:
            if attempt > 0:
                repair_guidance = self._debug_repair_guidance(
                    clean_code,
                    stderr,
                    stdout,
                    timeout_kill,
                    accelerator,
                    fidelity_profile,
                    evaluation_mode,
                )
                artifact_debug_context = ""
                failure_output = (stderr + "\n" + stdout).strip()
                should_repair_artifact = (
                    not artifact_repair_attempted
                    and bool(model_card)
                    and bool(technique_code)
                    and self._looks_like_deep_learning(technique_code)
                    and (
                        timeout_kill
                        or Path(str(model_card.get("code_path", ""))).stem.lower()
                        in failure_output.lower()
                        or any(
                            marker in failure_output.lower()
                            for marker in (
                                "dtype",
                                "expected scalar type",
                                "out of memory",
                                "size mismatch",
                                "mat1 and mat2",
                                "nan",
                            )
                        )
                    )
                )
                if should_repair_artifact:
                    print(
                        "ImplementationAgent: Failure points to copied neural "
                        "artifact; attempting one node-local repair and re-verification..."
                    )
                    repaired_artifact, verified, verification_note = (
                        self._repair_node_local_artifact(
                            node_dir,
                            model_card,
                            technique_code,
                            failure_output,
                            fidelity_profile,
                        )
                    )
                    artifact_repair_attempted = True
                    artifact_repair_summary = {
                        "attempted": True,
                        "verified": verified,
                        "artifact_id": model_card.get("artifact_id"),
                        "code_sha256": (
                            hashlib.sha256(
                                repaired_artifact.encode("utf-8")
                            ).hexdigest()
                            if verified
                            else None
                        ),
                        "diagnostics_tail": verification_note[-2000:],
                    }
                    if verified:
                        artifact_repair_summary["variant_id"] = (
                            f"{model_card.get('artifact_id')}@"
                            f"{artifact_repair_summary['code_sha256'][:12]}"
                        )
                    if verified:
                        technique_code = repaired_artifact
                if technique_code and (
                    self._looks_like_deep_learning(technique_code)
                    or Path(model_card.get("code_path", "")).stem.lower()
                    in (stderr or "").lower()
                ):
                    artifact_debug_context = (
                        "Copied artifact source (a node-local version may already have been "
                        "repaired and re-verified; integrate its public API exactly):\n```python\n"
                        + technique_code[-12000:]
                        + "\n```"
                    )
                print(
                    f"ImplementationAgent: Debug Attempt {attempt}/{max_debug_attempts} "
                    "— invoking failure-focused repair..."
                )
                debug_system_prompt = (
                    "You are the Implementation Agent in failure-repair mode. The previous generated ML script failed.\n"
                    "Use the supplied code, concrete output, and deterministic repair focus to fix the causal defect.\n"
                    "Do not opportunistically fine-tune a failing model: first make it complete within the fidelity caps.\n"
                    "CRITICAL ARCHITECTURE REPAIR CONTRACT:\n"
                    "- Debug and fix the INTENDED model architecture specified in the plan.\n"
                    "- Do NOT abandon the intended architecture or fall back to re-executing a parent model architecture from another node.\n"
                    f"{integration_instruction}\n"
                    "CRITICAL: Start your Python code response with a brief comment block explaining the cause of the error and how you are fixing it. "
                    "This helps trace execution logical errors correctly.\n"
                    "Ensure the output is valid Python code wrapped in a ```python block.\n"
                    "Fit preprocessing only on the selected protocol's training data; never derive preprocessing statistics from test data.\n"
                    "Apply the same fitted transformations symmetrically to training, validation, and test data. "
                    "Ensure exact post-transform feature parity and structure before running prediction.\n"
                    "CRITICAL DATA LOADING CONTRACT:\n"
                    "1. You must ALWAYS load the dataset first using the custom loader:\n"
                    "   from task_dataloader import TaskDataLoader\n"
                    "   loader = TaskDataLoader()\n"
                    "   train_data, test_data = loader()\n"
                    "2. Pass `train_data` directly to the evaluation helper selected below. Never read raw files or invent a split.\n"
                    f"3. {concrete_data_context}\n"
                    f"{evaluation_contract_prompt}"
                    "Recompute validation values with "
                    f"`evaluation.metrics.metric_value('{metric_name}', y_true, y_pred)`; the "
                    "metric name is the first argument and must not also be passed as a keyword.\n"
                    + (
                        "The task requires final predictions. Call "
                        "`evaluation_contract.prepare_final_training_data(train_data, test_data)`, "
                        "fit a fresh model on every returned full-training row, predict every returned "
                        "test row, and write the exact template schema to "
                        "`submission/submission.csv`. This must also create "
                        "`final_training_manifest.json`. The helper owns that manifest; "
                        "never create or overwrite it yourself. Read the exact submission "
                        "columns from output.options in resolved_task_spec.json.\n"
                        if require_submission
                        else ""
                    )
                    +
                    f"The selected accelerator is {accelerator}, exposed as AIBUILDAI_ACCELERATOR. Preserve "
                    "framework-native GPU configuration when supported and retain a safe CPU fallback. "
                    "AIBUILDAI_ACTUAL_ACCELERATOR starts as cpu; promote it only after actual GPU-backed "
                    "training or inference, then report it.\n"
                    f"{fine_tuning_instruction}"
                    f"{deep_learning_instruction}"
                    f"Modality correctness contract:\n{modality_context}\n"
                    f"IMPORTANT: At the END of your script, write a JSON file 'result.json' in the current directory:\n"
                    f'  import json; json.dump({{"score": <float>, "metric": "{metric_name}", "direction": "{metric_direction}", '
                    f'{result_statistics_prompt}"fidelity": "{fidelity}", '
                    f'"accelerator": <actual "cpu"|"cuda"|"mps">{tuning_result_fields}}}, open("result.json", "w"))\n'
                    "The score should be the final validation metric value."
                )
                
                debug_user_prompt = f"""
                Your Previous Generated Code (which failed):
                ```python
                {clean_code}
                ```

                Subprocess Exit Code: {exit_code}
                Automatic execution stop: {termination_reason or "none"}

                Traceback / Error Output (stderr):
                ```
                {stderr}
                ```

                Stdout output (if any):
                ```

                Deterministic repair focus:
                {repair_guidance}
                {stdout}
                ```

                Artifact interface metadata:
                {json.dumps(prompt_model_card.get('interface', {}), indent=2)}

                Required evaluation profile:
                {json.dumps(fidelity_profile)}

                Dataset/schema context:
                {dataset_snapshot[-6000:]}

                {artifact_debug_context}

                Please debug and fix the code. Follow the integration instruction exactly and ensure all variables/datasets are loaded properly. Return the complete corrected code file.
                """
                response = call_llm(
                    debug_system_prompt,
                    debug_user_prompt,
                    model=self.model_name,
                    temperature=0.0,
                )
                
                # Clean markdown code block formatting
                clean_code = response
                if "```python" in response:
                    clean_code = response.split("```python")[1].split("```")[0]
                elif "```" in response:
                    clean_code = response.split("```")[1].split("```")[0]

                debug_guard_issues = execution_contract_issues(clean_code)
                if debug_guard_issues:
                    stderr = (
                        "Static leakage guard rejected the debug revision:\n"
                        + "\n".join(debug_guard_issues)
                    )
                    stdout = ""
                    exit_code = -2
                    timeout_kill = False
                    execution_stalled = False
                    hard_limit_reached = False
                    termination_reason = "static_guard_rejected_revision"
                    attempt += 1
                    continue

                with open(dest_code_file, 'w', encoding='utf-8') as f:
                    f.write(clean_code.strip())
                print(f"ImplementationAgent: Wrote revised glue code to {dest_code_file}")

            # Execute under a renewable progress lease. There is no total
            # workflow runtime ceiling; focused direct callers may still supply
            # ``timeout`` as an explicit test/integration limit.
            timeout_kill = False
            execution_stalled = False
            hard_limit_reached = False
            termination_reason = None

            # Do not accept stale outputs from an earlier run in the same node folder.
            result_json_path = node_dir / "result.json"
            submission_path = node_dir / "submission" / "submission.csv"
            for stale_output in (
                result_json_path,
                submission_path,
                node_dir / "oof_predictions.npz",
                node_dir / "oof_predictions.csv",
                node_dir / "validation_predictions.npz",
                node_dir / "validation_predictions.csv",
                node_dir / "predictions" / "manifest.json",
                node_dir / "predictions" / "payload.npz",
                node_dir / "predictions" / "payload.json",
                node_dir / "evaluation_manifest.json",
                node_dir / "fold_assignments.npz",
                node_dir / "fold_assignments.csv",
                node_dir / "validation_assignments.npz",
                node_dir / "validation_assignments.csv",
                node_dir / "error_analysis.json",
                node_dir / "final_training_manifest.json",
                node_dir
                / ".evaluation_contract"
                / "final_training_manifest.json",
                node_dir
                / ".evaluation_contract"
                / "validation_targets.npz",
                node_dir
                / ".evaluation_contract"
                / "validation_targets.json",
                node_dir
                / ".evaluation_contract"
                / "evaluation_targets.npz",
            ):
                if stale_output.exists() or stale_output.is_symlink():
                    stale_output.unlink()
            
            child_env = accelerator_subprocess_env(accelerator)
            if accelerator == "cpu" and parallel_processes > 1:
                thread_limit = max(
                    1, int(os.cpu_count() or 1) // parallel_processes
                )
                for variable in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    child_env[variable] = str(thread_limit)
                child_env["AIBUILDAI_PARALLEL_ROOT_PROCESSES"] = str(
                    parallel_processes
                )
            child_env.update(
                {
                    "AIBUILDAI_TASK_ASSETS_DIR": str(
                        Path(task_assets_dir).resolve()
                        if task_assets_dir is not None
                        else node_dir.resolve()
                    ),
                    "AIBUILDAI_MAX_EPOCHS": str(fidelity_profile["max_epochs"]),
                    "AIBUILDAI_EARLY_STOPPING_PATIENCE": str(
                        fidelity_profile["early_stopping_patience"]
                    ),
                    "AIBUILDAI_MAX_ESTIMATOR_ITERATIONS": str(
                        fidelity_profile["max_estimator_iterations"]
                    ),
                    "AIBUILDAI_MAX_TUNING_TRIALS": str(
                        fidelity_profile["max_tuning_trials"]
                    ),
                    "AIBUILDAI_EVALUATION_MODE": evaluation_mode,
                    "PYTHONPATH": os.pathsep.join(
                        filter(
                            None,
                            (
                                str(self.project_root),
                                child_env.get("PYTHONPATH", ""),
                            ),
                        )
                    ),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            execution = run_supervised_process(
                cmd,
                cwd=node_dir,
                env=child_env,
                stall_seconds=float(stall_seconds),
                hard_limit_seconds=(
                    float(timeout) if timeout is not None else None
                ),
                activity_root=node_dir,
                stdout_stream=sys.stdout,
                stderr_stream=sys.stderr,
                label="ImplementationAgent",
            )
            exit_code = execution.returncode
            stdout = execution.stdout
            stderr = execution.stderr
            execution_stalled = execution.stalled
            hard_limit_reached = execution.hard_limit_reached
            termination_reason = execution.termination_reason
            progress_events = execution.progress_events
            last_progress_source = execution.last_progress_source
            last_progress_age_seconds = execution.last_progress_age_seconds
            timeout_kill = execution_stalled or hard_limit_reached
            if hard_limit_reached:
                stderr = (
                    stderr
                    + "\nExecution stopped at the explicit direct-call runtime "
                    f"limit of {float(timeout):.1f}s."
                ).strip()
                
            # Preserve individual failed-attempt diagnostics. error.log is reserved
            # for the final failed state so recovered nodes are not mislabeled.
            diagnostics = stdout + "\n" + stderr
            if stderr.strip() or exit_code != 0 or timeout_kill:
                attempt_log_path = node_dir / f"attempt_{attempt + 1}.log"
                with open(attempt_log_path, 'w', encoding='utf-8') as f:
                    f.write(
                        f"Attempt={attempt + 1}\nexit_code={exit_code}\n"
                        f"termination_reason={termination_reason}\n\n"
                    )
                    f.write("=== STDERR ===\n")
                    f.write(stderr)
                    f.write("\n\n=== STDOUT ===\n")
                    f.write(stdout)
            
            # Parse score — prefer structured result.json, fall back to stdout regex
            score = None
            score_source = "none"
            status = "completed"
            
            # Strategy 1: Parse result.json (structured contract)
            result_data = {}
            if exit_code == 0 and result_json_path.exists():
                try:
                    with open(result_json_path, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    parsed_score = result_data.get("score")
                    if parsed_score is not None:
                        score = float(parsed_score)
                        if not math.isfinite(score):
                            raise ValueError("score must be finite")
                        declared_direction = result_data.get("direction")
                        if declared_direction and declared_direction != metric_direction:
                            raise ValueError(
                                f"result direction {declared_direction!r} does not match {metric_direction!r}"
                            )
                        declared_metric = result_data.get("metric")
                        if declared_metric:
                            resolved_declared_metric = resolve_metric_name(
                                declared_metric,
                                problem_type=task_spec.problem_type,
                                output_type=task_spec.output.type,
                            )
                            if resolved_declared_metric != metric_name:
                                raise ValueError(
                                    f"result metric {declared_metric!r} resolves "
                                    f"to {resolved_declared_metric!r}, which does "
                                    f"not match {metric_name!r}"
                                )
                            result_data["metric"] = metric_name
                        declared_fidelity = result_data.get("fidelity")
                        if declared_fidelity and declared_fidelity != fidelity:
                            raise ValueError(
                                f"result fidelity {declared_fidelity!r} does not match {fidelity!r}"
                            )
                        declared_accelerator = result_data.get("accelerator")
                        if enforce_evaluation_contract and declared_accelerator is None:
                            raise ValueError(
                                "result must declare the accelerator actually used"
                            )
                        if declared_accelerator is not None:
                            declared_accelerator = str(declared_accelerator).lower()
                            if declared_accelerator not in {"cpu", "cuda", "mps"}:
                                raise ValueError(
                                    "result accelerator must be cpu, cuda, or mps"
                                )
                            if declared_accelerator not in {"cpu", accelerator}:
                                raise ValueError(
                                    f"result claims {declared_accelerator!r}, but this node "
                                    f"selected {accelerator!r}"
                                )
                            result_data["accelerator"] = declared_accelerator
                        if result_data.get("cv_std") is not None:
                            cv_std = float(result_data["cv_std"])
                            if not math.isfinite(cv_std) or cv_std < 0:
                                raise ValueError("cv_std must be finite and non-negative")
                        if result_data.get("folds") is not None:
                            folds = int(result_data["folds"])
                            if folds < 1:
                                raise ValueError("folds must be positive")
                        if operator == "tune" and tuning_context:
                            _, tuning_trials = self._validate_tuning_metadata(
                                result_data,
                                fidelity_profile,
                                (
                                    tuning_context.get("tunable_parameters", [])
                                    if tuning_context.get(
                                        "tunable_parameters_declared", False
                                    )
                                    else None
                                ),
                            )
                            result_data["tuning_trials"] = tuning_trials
                        if enforce_evaluation_contract:
                            contract_validation = validate_evaluation_outputs(
                                node_dir,
                                fidelity,
                                metric_name,
                                expected_class_names=(
                                    task_spec.output.class_names
                                ),
                                expected_evaluation_mode=evaluation_mode,
                            )
                            if require_submission:
                                result_data["submission_validation"] = (
                                    validate_node_submission(
                                        node_dir,
                                        task_dir=task_dir,
                                        task_spec=task_spec,
                                    )
                                )
                            result_data.update(contract_validation)
                            score = float(contract_validation["score"])
                            error_analysis = build_error_analysis(
                                node_dir,
                                evaluation_mode=evaluation_mode,
                                problem_type=task_spec.problem_type,
                                metric_name=metric_name,
                            )
                            result_data["error_analysis"] = error_analysis
                            (node_dir / "error_analysis.json").write_text(
                                json.dumps(
                                    error_analysis,
                                    indent=2,
                                    default=str,
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                            result_data["score"] = score
                            result_data["metric"] = metric_name
                            result_data["direction"] = metric_direction
                            with open(result_json_path, "w", encoding="utf-8") as f:
                                json.dump(result_data, f, indent=2)
                        score_source = "result.json"
                        print(f"ImplementationAgent: Score from result.json: {score} "
                              f"(metric={result_data.get('metric', '?')}, direction={result_data.get('direction', '?')})")
                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                    OverflowError,
                    OSError,
                ) as e:
                    score = None
                    score_source = "none"
                    stderr = (stderr + "\nResult contract error: " + str(e)).strip()
                    diagnostics = stdout + "\n" + stderr
                    print(f"ImplementationAgent: WARNING: result.json exists but couldn't parse: {e}")
            
            # Strategy 2: Regex fallback on stdout (handles negatives and scientific notation)
            if (
                score is None
                and exit_code == 0
                and not timeout_kill
                and not enforce_evaluation_contract
                and operator != "tune"
            ):
                # Match patterns like: "Score: 0.93245", "AUC: -0.123", "accuracy = 9.5e-3"
                score_matches = re.findall(
                    r'(?:score|auc|accuracy|metric|rmse|mae|loss|f1)[:\s=]+(-?[0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)',
                    diagnostics, re.IGNORECASE
                )
                if score_matches:
                    try:
                        score = float(score_matches[-1])
                        if not math.isfinite(score):
                            raise ValueError("score must be finite")
                        score_source = "stdout_regex"
                        print(f"ImplementationAgent: Score from stdout regex (fallback): {score}")
                    except ValueError:
                        score = None
                        pass
            
            # If successful execution and a score was found, stop debugging
            if exit_code == 0 and score is not None:
                break

            # Rewrite the attempt log after result-contract parsing so clean exits
            # with invalid OOF/result files are as diagnosable as process crashes.
            attempt_log_path = node_dir / f"attempt_{attempt + 1}.log"
            with open(attempt_log_path, "w", encoding="utf-8") as f:
                f.write(
                    f"Attempt={attempt + 1}\nexit_code={exit_code}\n"
                    f"termination_reason={termination_reason}\n\n"
                )
                f.write("=== STDERR ===\n")
                f.write(stderr)
                f.write("\n\n=== STDOUT ===\n")
                f.write(stdout)
                
            attempt += 1
        
        # A crashing process is always a failed run, even if it wrote a partial score.
        if exit_code != 0:
            status = "failed"
            score = None
            score_source = "none"
            if execution_stalled:
                print(
                    "ImplementationAgent: FAILED — execution stopped after the "
                    "progress lease detected a stalled process "
                    f"(exit_code={exit_code})."
                )
            elif hard_limit_reached:
                print(
                    "ImplementationAgent: FAILED — explicit direct-call runtime "
                    f"limit reached (exit_code={exit_code})."
                )
            else:
                print(f"ImplementationAgent: FAILED — subprocess crashed (exit_code={exit_code}). "
                      f"See {node_dir / 'error.log'} for details.")
        elif score is None:
            # exit_code == 0 but no score produced — unusual but not a crash
            status = "failed"
            print(f"ImplementationAgent: FAILED — subprocess exited cleanly but produced no score.")
        else:
            (node_dir / "error.log").unlink(missing_ok=True)
            print(f"ImplementationAgent: Execution completed — exit_code={exit_code}, score={score}, "
                  f"source={score_source}, progress_events={progress_events}")

        if status == "failed":
            with open(node_dir / "error.log", "w", encoding="utf-8") as f:
                f.write(stderr or diagnostics or "experiment failed without diagnostics")

        actual_accelerator = (
            result_data.get("accelerator", accelerator)
            if status == "completed"
            else None
        )
        execution_resource["reported_accelerator"] = actual_accelerator
        execution_resource["last_execution"] = {
            "termination_reason": termination_reason,
            "progress_events": progress_events,
            "last_progress_source": last_progress_source,
            "last_progress_age_seconds": last_progress_age_seconds,
        }
        with open(node_dir / "execution_resource.json", "w", encoding="utf-8") as f:
            json.dump(execution_resource, f, indent=2)
            f.write("\n")

        tuning_summary = None
        if status == "completed" and operator == "tune" and tuning_context:
            tuning_summary = {
                **tuning_context,
                "hyperparameters": result_data.get("hyperparameters"),
                "tuning_trials": result_data.get("tuning_trials"),
                "score": score,
                "fidelity": fidelity,
            }
            with open(node_dir / "fine_tuning.json", "w", encoding="utf-8") as f:
                json.dump(tuning_summary, f, indent=2, default=str)
                f.write("\n")

        contract_artifacts: dict[str, object] = {}
        if status == "completed":
            try:
                contract_artifacts = self._materialize_output_contracts(
                    node_dir,
                    task_spec,
                    result_data,
                    dest_code_file,
                )
            except Exception as exc:
                status = "failed"
                score = None
                diagnostics = (
                    diagnostics
                    + "\nTyped artifact contract error: "
                    + str(exc)
                ).strip()
                (node_dir / "error.log").write_text(
                    diagnostics, encoding="utf-8"
                )
        
        return {
            "status": status,
            "score": score,
            "score_source": score_source,
            "exit_code": exit_code,
            "execution_stalled": execution_stalled,
            "hard_limit_reached": hard_limit_reached,
            "termination_reason": termination_reason,
            "progress_events": progress_events,
            "last_progress_source": last_progress_source,
            "timeout_kill": timeout_kill,
            "stdout": stdout,
            "stderr": stderr,
            "diagnostics": diagnostics,
            "code_path": str(dest_code_file),
            "base_code_path": (
                str(base_algorithm_path)
                if base_algorithm_path is not None
                else None
            ),
            "parent_node_dir": str(parent_node_dir) if parent_node_dir else None,
            "inherited_files": inherited_files,
            "operator": operator,
            "fidelity": fidelity,
            "evaluation_mode": evaluation_mode,
            "evaluation_policy": evaluation_policy.to_dict(),
            "error_analysis": result_data.get("error_analysis"),
            "error_analysis_path": (
                str(node_dir / "error_analysis.json")
                if (node_dir / "error_analysis.json").is_file()
                else None
            ),
            "accelerator": actual_accelerator,
            "selected_accelerator": accelerator,
            "tuning": tuning_summary,
            "artifact_repair": artifact_repair_summary,
            "implementation_families": sorted(
                self._model_family_imports(clean_code)
            ),
            "code_fingerprint": code_fingerprint,
            "elapsed_seconds": time.monotonic() - run_started,
            "validation": {
                key: result_data.get(key)
                for key in (
                    "score", "score_std", "evaluation_mode",
                    "validation_score", "validation_row_count",
                    "cv_mean", "cv_std", "folds", "fold_scores", "seed",
                    "fidelity", "row_count", "source_row_count",
                    "fold_assignment_sha256",
                )
                if result_data.get(key) is not None
            },
            "oof_path": (
                str(node_dir / "oof_predictions.npz")
                if (node_dir / "oof_predictions.npz").is_file()
                else str(node_dir / "oof_predictions.csv")
                if (node_dir / "oof_predictions.csv").is_file()
                else str(node_dir / "predictions" / "manifest.json")
                if evaluation_mode == "cross_validation"
                and (node_dir / "predictions" / "manifest.json").is_file()
                else None
            ),
            "validation_path": (
                str(node_dir / "validation_predictions.npz")
                if (node_dir / "validation_predictions.npz").is_file()
                else str(node_dir / "validation_predictions.csv")
                if (node_dir / "validation_predictions.csv").is_file()
                else str(node_dir / "predictions" / "manifest.json")
                if evaluation_mode == "holdout"
                and (node_dir / "predictions" / "manifest.json").is_file()
                else None
            ),
            **contract_artifacts,
        }
