import ast
import hashlib
import re
import os
import sys
import json
import math
import signal
import subprocess
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional
from .llm_utils import call_llm
from .validation_guard import inspect_generated_code
from evaluation_contract import FIDELITY_PROFILES, validate_evaluation_outputs
from runtime_utils import (
    accelerator_subprocess_env,
    expose_task_data,
    resolve_within,
    validate_storage_identifier,
)

class ImplementationAgent:
    def __init__(self, venv_python_path: str = "./.venv/bin/python", model_name: str = None):
        import sys
        resolved_path = str(Path(venv_python_path).resolve())
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

    @classmethod
    def _debug_repair_guidance(
        cls,
        code: str,
        stderr: str,
        stdout: str,
        timeout_kill: bool,
        accelerator: str,
        fidelity_profile: dict,
    ) -> str:
        """Return deterministic, failure-specific constraints for the LLM debugger."""
        combined = (str(stderr) + "\n" + str(stdout)).lower()
        guidance = []
        if timeout_kill or "timeout" in combined or "deadline" in combined:
            guidance.append(
                "Runtime repair: keep the required rows/folds, but reduce work inside each fold; "
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
        if any(marker in combined for marker in ("no module named", "modulenotfounderror", "importerror")):
            guidance.append(
                "Dependency repair: use only installed/project-allowlisted packages and the copied local "
                "artifact module; do not invent package paths."
            )
        if any(
            marker in combined
            for marker in ("result contract", "oof", "fold_id", "evaluation_manifest", "submission")
        ):
            guidance.append(
                "Evaluation repair: preserve evaluation_contract rows and fold_ids exactly, write one "
                "OOF prediction per scheduled row, and regenerate result.json and submission.csv."
            )
        if cls._looks_like_deep_learning(code):
            guidance.append(
                "Deep-learning invariant: never move the complete dataset to GPU; use DataLoader "
                "mini-batches, place model and each batch on the same device, detach predictions to CPU, "
                f"use at most {fidelity_profile['max_epochs']} epochs with patience "
                f"{fidelity_profile['early_stopping_patience']}, and release the model between folds."
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
        return (
            "This is a baseline-triggered fine-tuning run, not an architecture rewrite. Preserve the "
            "parent preprocessing, feature set, model family, folds, and output schema. Search only "
            "meaningful existing hyperparameters, reuse the parent settings as a control trial, and use "
            f"at most {fidelity_profile['max_tuning_trials']} deterministic/pruned trials. For neural "
            f"models, tune learning rate, batch size, weight decay, dropout/width, and epochs up to "
            f"{fidelity_profile['max_epochs']} with early-stopping patience "
            f"{fidelity_profile['early_stopping_patience']}; increasing epochs is allowed only within "
            "that cap. For boosted trees, tune depth/leaves, learning rate, regularization, sampling, "
            f"and iterations up to {fidelity_profile['max_estimator_iterations']}. Optimize only the "
            "harness-provided validation folds—never the test set. result.json must include a non-empty "
            "`hyperparameters` object and integer `tuning_trials`.\n"
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
                "You repair a node-local tabular-ML artifact after a concrete runtime failure. "
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
            "oof_predictions.csv",
            "evaluation_manifest.json",
            "fold_assignments.csv",
            "execution_resource.json",
            "fine_tuning.json",
            "artifact_repair.json",
        }
        allowed_suffixes = {
            ".py", ".json", ".yaml", ".yml", ".txt",
        }
        for source in parent_node_dir.iterdir():
            if (
                not source.is_file()
                or source.is_symlink()
                or source.name in excluded
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
        metric_direction: str = "maximize",
        base_algorithm_path: Optional[Path] = None,
        parent_node_dir: Optional[Path] = None,
        fidelity: str = "full",
        operator: Optional[str] = None,
        enforce_evaluation_contract: bool = False,
        accelerator: str = "cpu",
        available_accelerators: Optional[set[str]] = None,
        tuning_context: Optional[dict] = None,
        max_debug_attempts: int = 2,
    ) -> Dict[str, Any]:
        """
        1. Reads task skeletons from task_dir.
        2. Calls LLM to generate updated code wiring the artifact in.
        3. Writes it to node_dir / "algorithm.py".
        4. Runs it and parses result.json for the evaluation metric score.
        
        Args:
            node_dir: Directory for this node's run outputs
            technique_record: Dict from TechniqueAgent with plan/model_card
            task_dir: Task directory with initial code and data
            timeout: Optional subprocess timeout in seconds. Normal search nodes
                pass ``None`` and have no inactivity or wall-clock execution limit.
            metric_direction: "maximize" or "minimize" — used in the prompt
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
        if metric_direction not in {"maximize", "minimize"}:
            raise ValueError(
                f"metric_direction must be 'maximize' or 'minimize', got {metric_direction!r}"
            )
        if not isinstance(max_debug_attempts, int) or max_debug_attempts < 0:
            raise ValueError("max_debug_attempts must be a non-negative integer")
        if tuning_context is not None and not isinstance(tuning_context, dict):
            raise ValueError("tuning_context must be a dictionary when provided")
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
        node_dir.mkdir(parents=True, exist_ok=True)
        run_started = time.monotonic()
        execution_resource = {
            "selected_accelerator": accelerator,
            "available_accelerators": sorted(exposed_accelerators),
            "environment_variable": "AIBUILDAI_ACCELERATOR",
            "fallback": "cpu",
            "subprocess_timeout_seconds": timeout,
        }
        with open(node_dir / "execution_resource.json", "w", encoding="utf-8") as f:
            json.dump(execution_resource, f, indent=2)
            f.write("\n")
        inherited_files = self._inherit_parent_workspace(
            Path(parent_node_dir) if parent_node_dir else None, node_dir
        )
        
        # Copy dataloader to node_dir so imports work locally
        src_loader = task_dir / "initial_dataloader.py"
        dest_loader = node_dir / "initial_dataloader.py"
        if src_loader.exists():
            shutil.copy(src_loader, dest_loader)
        contract_source = self.project_root / "evaluation_contract.py"
        if contract_source.is_file():
            shutil.copy2(contract_source, node_dir / "evaluation_contract.py")
            
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

        # Descendants evolve the measured parent implementation. Only root-level
        # candidates start from the generated baseline.
        src_algo = Path(base_algorithm_path) if base_algorithm_path else (
            task_dir / "initial_algorithm.py"
        )
        if not src_algo.is_file():
            raise FileNotFoundError(f"Base algorithm does not exist: {src_algo}")
        with open(src_algo, 'r', encoding='utf-8') as f:
            original_code = f.read()
            
        # Load task config for metric info
        config_file = task_dir / "task_config.json"
        metric_name = "score"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                task_config = json.load(f)
            metric_name = task_config.get("metric_name", "score")
            metric_direction = task_config.get("metric_direction", metric_direction)
            
        # Never execute an artifact that failed verification.
        if model_card and model_card.get("verified") is not True:
            raise ValueError(
                f"Model card {model_card.get('artifact_id', '<unknown>')!r} is not verified"
            )
            
        # Get dataset analysis report to pass to LLM
        dataset_snapshot = ""
        report_file = task_dir / "dataset_analysis_report.txt"
        try:
            if report_file.exists():
                with open(report_file, 'r', encoding='utf-8') as f:
                    analysis_report = f.read()
            else:
                from .data_analyzer import run_dataset_analysis
                print(f"ImplementationAgent: Checking/running dataset analysis fallback for {task_dir.name}...")
                analysis_report = run_dataset_analysis(task_dir)
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(analysis_report)
                    
            dataset_snapshot = (
                "=== Dataset Analysis & Schema Report ===\n"
                f"{analysis_report}\n"
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

        else:
            integration_instruction = (
                "No verified artifact is available for this node. Improve the supplied parent baseline directly "
                "according to the technique plan, keep the script self-contained, and do not import any "
                "`memory_pool` module or imaginary artifact."
            )

        if fidelity not in FIDELITY_PROFILES:
            raise ValueError("fidelity must be 'screen', 'medium', or 'full'")
        fidelity_profile = FIDELITY_PROFILES[fidelity]
        operator_instruction = {
            "refine": "Modify only the highest-impact relevant component; preserve working parent behavior elsewhere.",
            "tune": "Act as a fine-tuner: preserve the measured architecture and run a compact pruned hyperparameter search with a fixed seed.",
            "diversify": "Favor a sound model or representation whose errors are likely less correlated with the parent.",
            "promote": "Preserve the parent method and evaluate it more rigorously at the requested fidelity.",
        }.get(operator or "", "Apply the requested technique as a focused change to the parent pipeline.")
        fine_tuning_instruction = self._fine_tuning_instruction(
            operator, tuning_context, fidelity_profile
        )
        deep_learning_instruction = (
            "Deep-learning execution contract (when applicable): use fold-local numeric/categorical "
            "preprocessing; float32 features; the loss-appropriate label dtype; mini-batch DataLoaders; "
            "model and batches on the same selected device; validation-based early stopping with the "
            "best state restored; and CPU-detached predictions. Never place the full dataset on GPU or "
            "retain models/tensors across folds. "
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

        system_prompt = (
            "You are the Implementation Agent. Produce a complete executable revision of the supplied parent algorithm. "
            f"{integration_instruction} Ensure the output is valid Python code wrapped in a ```python block.\n"
            f"Search operator: {operator or 'initial'}. {operator_instruction}\n"
            f"Evaluation fidelity: {fidelity} ({json.dumps(fidelity_profile)}). These limits are mandatory.\n"
            f"{fine_tuning_instruction}"
            f"{deep_learning_instruction}"
            f"Execution accelerator: {accelerator}; available={sorted(exposed_accelerators)}. "
            "The subprocess also exposes this value as AIBUILDAI_ACCELERATOR. When the selected accelerator is "
            "CUDA or MPS, use the framework-native GPU/device option for every compatible training component "
            "and move neural-network models and tensors to that device. Check that the framework backend is "
            "usable and fall back to CPU only when that model/library has no working backend for the selected "
            "accelerator. When the technique permits equivalent model families, prefer a GPU-capable learner "
            "over a CPU-only one. Do not send small preprocessing or unsupported scikit-learn estimators to a GPU. "
            "After training, report os.environ.get('AIBUILDAI_ACTUAL_ACCELERATOR', selected_device) in "
            "result.json; GPU-aware pool artifacts update this variable when they fall back.\n"
            "Import `prepare_evaluation_data` from the local `evaluation_contract` module and call "
            f"`X_eval, y_eval, row_ids, fold_ids, evaluation_meta = prepare_evaluation_data(train_data, '{fidelity}')`. "
            "Use X_eval/y_eval for every training and validation operation, use the supplied fold_ids exactly, "
            "and train final test-prediction models on all X_eval rows. Save one OOF prediction per scheduled "
            "row to oof_predictions.csv with columns row_id,target,prediction. Do not create a separate split.\n"
            "Fit every imputer, encoder, scaler, feature selector, and target-dependent transform on training folds only. "
            "Never use test-set statistics or target values in feature generation. Use stratification for classification.\n"
            "When practical, save validation predictions to 'oof_predictions.csv' with columns row_id,target,prediction.\n"
            f"IMPORTANT: At the END of your script, write a JSON file 'result.json' in the current directory:\n"
            f'  import json; json.dump({{"score": <float>, "metric": "{metric_name}", "direction": "{metric_direction}", '
            f'"cv_mean": <float>, "cv_std": <float>, "folds": <int>, "fidelity": "{fidelity}", '
            f'"accelerator": <actual "cpu"|"cuda"|"mps">{tuning_result_fields}}}, open("result.json", "w"))\n'
            "The score must be the cross-validation mean when CV is used, otherwise the held-out validation score."
        )
        
        technique_plan = technique_record.get("plan", "")
        
        prompt_model_card = dict(model_card or {})
        prompt_model_card.pop("code_content", None)
        if prompt_model_card.get("verification_log"):
            prompt_model_card["verification_log"] = str(
                prompt_model_card["verification_log"]
            )[-1200:]
        user_prompt = f"""
            Parent Code ({'measured ancestor' if base_algorithm_path else 'generated baseline'}):
            ```python
            {original_code}
            ```

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

        guard_issues = inspect_generated_code(clean_code)
        guard_issues.extend(self._resource_limit_issues(clean_code, fidelity_profile))
        if operator == "tune" and tuning_context:
            guard_issues.extend(
                self._tuning_lock_issues(clean_code, original_code, model_card)
            )
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
            remaining_issues = inspect_generated_code(clean_code)
            remaining_issues.extend(
                self._resource_limit_issues(clean_code, fidelity_profile)
            )
            if operator == "tune" and tuning_context:
                remaining_issues.extend(
                    self._tuning_lock_issues(
                        clean_code, original_code, model_card
                    )
                )
            if remaining_issues:
                raise ValueError(
                    "Generated implementation failed execution-contract guard after repair: "
                    + "; ".join(remaining_issues)
                )
            
        dest_code_file = node_dir / "algorithm.py"
        with open(dest_code_file, 'w', encoding='utf-8') as f:
            f.write(clean_code.strip())
            
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
                    f"{integration_instruction}\n"
                    "CRITICAL: Start your Python code response with a brief comment block explaining the cause of the error and how you are fixing it. "
                    "This helps trace execution logical errors correctly.\n"
                    "Ensure the output is valid Python code wrapped in a ```python block.\n"
                    "Fit preprocessing on training folds only; never derive preprocessing statistics from test data.\n"
                    "Keep using evaluation_contract.prepare_evaluation_data and its supplied fold_ids exactly; "
                    "write complete OOF predictions for the scheduled rows.\n"
                    f"The selected accelerator is {accelerator}, exposed as AIBUILDAI_ACCELERATOR. Preserve "
                    "framework-native GPU configuration when supported and retain a safe CPU fallback. Report "
                    "AIBUILDAI_ACTUAL_ACCELERATOR after training.\n"
                    f"{fine_tuning_instruction}"
                    f"{deep_learning_instruction}"
                    f"IMPORTANT: At the END of your script, write a JSON file 'result.json' in the current directory:\n"
                    f'  import json; json.dump({{"score": <float>, "metric": "{metric_name}", "direction": "{metric_direction}", '
                    f'"cv_mean": <float>, "cv_std": <float>, "folds": <int>, "fidelity": "{fidelity}", '
                    f'"accelerator": <actual "cpu"|"cuda"|"mps">{tuning_result_fields}}}, open("result.json", "w"))\n'
                    "The score should be the final validation metric value."
                )
                
                debug_user_prompt = f"""
                Your Previous Generated Code (which failed):
                ```python
                {clean_code}
                ```

                Subprocess Exit Code: {exit_code}
                Subprocess Timeout: {timeout_kill}

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

                debug_guard_issues = inspect_generated_code(clean_code)
                debug_guard_issues.extend(
                    self._resource_limit_issues(clean_code, fidelity_profile)
                )
                if operator == "tune" and tuning_context:
                    debug_guard_issues.extend(
                        self._tuning_lock_issues(
                            clean_code, original_code, model_card
                        )
                    )
                if debug_guard_issues:
                    stderr = (
                        "Static leakage guard rejected the debug revision:\n"
                        + "\n".join(debug_guard_issues)
                    )
                    stdout = ""
                    exit_code = -2
                    timeout_kill = False
                    attempt += 1
                    continue

                with open(dest_code_file, 'w', encoding='utf-8') as f:
                    f.write(clean_code.strip())
                print(f"ImplementationAgent: Wrote revised glue code to {dest_code_file}")

            # Execute with activity-based watchdog timeout
            timeout_kill = False
            stdout_lines = []
            stderr_lines = []
            
            import fcntl

            # Do not accept stale outputs from an earlier run in the same node folder.
            result_json_path = node_dir / "result.json"
            submission_path = node_dir / "submission" / "submission.csv"
            for stale_output in (
                result_json_path,
                submission_path,
                node_dir / "oof_predictions.csv",
                node_dir / "evaluation_manifest.json",
                node_dir / "fold_assignments.csv",
            ):
                if stale_output.exists() or stale_output.is_symlink():
                    stale_output.unlink()
            
            child_env = accelerator_subprocess_env(accelerator)
            child_env.update(
                {
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
                }
            )
            proc = subprocess.Popen(
                cmd,
                cwd=node_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=child_env,
                start_new_session=True,
            )

            def terminate_process_group() -> None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    return
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            
            # Make stdout/stderr non-blocking
            for fd in (proc.stdout, proc.stderr):
                fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                
            start_time = time.time()
            last_output_time = time.time()
            # Search nodes run without a wall-clock or inactivity limit. The
            # optional timeout remains available only for explicit direct callers
            # (for example, focused safety tests or external integrations).
            hard_timeout = float(timeout) if timeout is not None else None
            inactivity_timeout = (
                min(1200.0, hard_timeout) if hard_timeout is not None else None
            )
            
            while True:
                ret = proc.poll()
                
                # Read non-blocking stdout bytes
                try:
                    out = proc.stdout.read()
                    if out:
                        decoded_out = out.decode('utf-8', errors='replace')
                        stdout_lines.append(decoded_out)
                        last_output_time = time.time()
                        sys.stdout.write(decoded_out)
                        sys.stdout.flush()
                except IOError:
                    pass
                    
                # Read non-blocking stderr bytes
                try:
                    err = proc.stderr.read()
                    if err:
                        decoded_err = err.decode('utf-8', errors='replace')
                        stderr_lines.append(decoded_err)
                        last_output_time = time.time()
                        sys.stderr.write(decoded_err)
                        sys.stderr.flush()
                except IOError:
                    pass
                    
                if ret is not None:
                    # Flush remaining buffers
                    try:
                        out = proc.stdout.read()
                        if out:
                            decoded_out = out.decode('utf-8', errors='replace')
                            stdout_lines.append(decoded_out)
                    except IOError:
                        pass
                    try:
                        err = proc.stderr.read()
                        if err:
                            decoded_err = err.decode('utf-8', errors='replace')
                            stderr_lines.append(decoded_err)
                    except IOError:
                        pass
                    exit_code = ret
                    break
                    
                now = time.time()
                elapsed = now - start_time
                inactive_elapsed = now - last_output_time
                
                if (
                    inactivity_timeout is not None
                    and inactive_elapsed > inactivity_timeout
                ):
                    print(f"\nImplementationAgent: INACTIVITY TIMEOUT after {inactivity_timeout}s with no output. Killing process.")
                    terminate_process_group()
                    exit_code = -1
                    timeout_kill = True
                    break
                    
                if hard_timeout is not None and elapsed > hard_timeout:
                    print(f"\nImplementationAgent: HARD TIMEOUT ceiling of {hard_timeout}s reached. Killing process.")
                    terminate_process_group()
                    exit_code = -1
                    timeout_kill = True
                    break
                    
                time.sleep(0.1)

            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)
            if timeout_kill:
                stderr = (
                    stderr
                    + f"\nExecution attempt exceeded the {hard_timeout:.1f}s runtime limit."
                ).strip()
            proc.stdout.close()
            proc.stderr.close()
                
            # Preserve individual failed-attempt diagnostics. error.log is reserved
            # for the final failed state so recovered nodes are not mislabeled.
            diagnostics = stdout + "\n" + stderr
            if stderr.strip() or exit_code != 0 or timeout_kill:
                attempt_log_path = node_dir / f"attempt_{attempt + 1}.log"
                with open(attempt_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"Attempt={attempt + 1}\nexit_code={exit_code}\ntimeout_kill={timeout_kill}\n\n")
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
                        if declared_metric and declared_metric != metric_name:
                            raise ValueError(
                                f"result metric {declared_metric!r} does not match {metric_name!r}"
                            )
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
                                node_dir, fidelity, metric_name
                            )
                            result_data.update(contract_validation)
                            score = float(contract_validation["cv_mean"])
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
                    f"timeout_kill={timeout_kill}\n\n"
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
            if timeout_kill:
                print(
                    f"ImplementationAgent: FAILED — timeout after {timeout}s "
                    f"(exit_code={exit_code})"
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
                  f"source={score_source}, timeout_kill={timeout_kill}")

        if status == "failed":
            with open(node_dir / "error.log", "w", encoding="utf-8") as f:
                f.write(stderr or diagnostics or "experiment failed without diagnostics")

        actual_accelerator = (
            result_data.get("accelerator", accelerator)
            if status == "completed"
            else None
        )
        execution_resource["reported_accelerator"] = actual_accelerator
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
        
        return {
            "status": status,
            "score": score,
            "score_source": score_source,
            "exit_code": exit_code,
            "timeout_kill": timeout_kill,
            "stdout": stdout,
            "stderr": stderr,
            "diagnostics": diagnostics,
            "code_path": str(dest_code_file),
            "base_code_path": str(src_algo),
            "parent_node_dir": str(parent_node_dir) if parent_node_dir else None,
            "inherited_files": inherited_files,
            "operator": operator,
            "fidelity": fidelity,
            "accelerator": actual_accelerator,
            "selected_accelerator": accelerator,
            "tuning": tuning_summary,
            "artifact_repair": artifact_repair_summary,
            "elapsed_seconds": time.monotonic() - run_started,
            "validation": {
                key: result_data.get(key)
                for key in (
                    "cv_mean", "cv_std", "folds", "fold_scores", "seed",
                    "fidelity", "row_count", "source_row_count",
                    "fold_assignment_sha256",
                )
                if result_data.get(key) is not None
            },
            "oof_path": (
                str(node_dir / "oof_predictions.csv")
                if (node_dir / "oof_predictions.csv").is_file()
                else None
            ),
        }
