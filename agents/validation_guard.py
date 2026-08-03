"""Static guardrails for common leakage patterns in generated ML pipelines."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Mapping


TEST_NAMES = {"x_test", "test_df", "test_data", "test_features"}
FIT_METHODS = {"fit", "fit_transform", "partial_fit"}
STAT_METHODS = {"mean", "median", "std", "quantile", "mode", "value_counts"}


def _root_name(node: ast.AST) -> str:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id.lower() if isinstance(node, ast.Name) else ""


def _contains_test_reference(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id.lower() in TEST_NAMES
        for child in ast.walk(node)
    )


def _literal_path(node: ast.AST) -> str:
    """Best-effort extraction for literal paths used by generated code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"Path", "PurePath"}
        and node.args
    ):
        return _literal_path(node.args[0])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _literal_path(node.left).rstrip("/")
        right = _literal_path(node.right).lstrip("/")
        if left and right:
            return f"{left}/{right}"
    return ""


def _targets_task_input(path: str) -> bool:
    normalized = str(path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return (
        normalized == "input"
        or normalized.startswith("input/")
        or "/tasks/" in normalized
    )


def inspect_generated_code(
    code: str,
    task_spec: Mapping[str, object] | None = None,
    runtime_contract: Mapping[str, object] | None = None,
) -> List[str]:
    """Return high-confidence leakage defects that should be repaired before execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # The normal debugging loop supplies better syntax diagnostics.

    issues: List[str] = []
    task_spec = dict(task_spec or {})
    runtime_contract = dict(runtime_contract or {})
    group_sensitive = bool(
        task_spec.get("group_id_field")
        or task_spec.get("entity_id_field")
    )
    components = runtime_contract.get("components", {})
    flattened_components = {
        str(name)
        for name, details in (
            components.items() if isinstance(components, Mapping) else ()
        )
        if isinstance(details, Mapping)
        and details.get("storage") == "flattened_columns"
    }
    array_names = {
        str(name).lower()
        for name in (
            runtime_contract.get("array_variables", {}).keys()
            if isinstance(runtime_contract.get("array_variables"), Mapping)
            else ()
        )
    }
    # Common aliases used when the authoritative arrays are passed into helpers.
    array_names.update({"y", "y_train", "y_valid", "y_val", "row_ids", "fold_ids"})
    harness_folds_active = any(
        isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id == "prepare_evaluation_data"
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "prepare_evaluation_data"
            )
        )
        for node in ast.walk(tree)
    )
    harness_holdout_active = any(
        isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id == "prepare_holdout_evaluation_data"
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "prepare_holdout_evaluation_data"
            )
        )
        for node in ast.walk(tree)
    )
    harness_split_active = harness_folds_active or harness_holdout_active
    literal_path_variables = {
        target.id: _literal_path(node.value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name) and _literal_path(node.value)
    }

    def resolved_literal_path(node: ast.AST) -> str:
        direct = _literal_path(node)
        if direct:
            return direct
        if isinstance(node, ast.Name):
            return literal_path_variables.get(node.id, "")
        return ""

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr.lower() in {"iloc", "loc"}
            and _root_name(node.value) in array_names
        ):
            issues.append(
                f"line {node.lineno}: {_root_name(node.value)} is a NumPy "
                f"array under the runtime data contract; use array[index] rather "
                f"than .{node.attr}"
            )
        if not isinstance(node, ast.Subscript):
            continue
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            component = key.value
        else:
            continue
        owner = _root_name(node.value)
        if (
            component in flattened_components
            and (owner.startswith("x") or "feature" in owner or "data" in owner)
        ):
            prefix = components[component].get(
                "column_prefix", f"{component}__"
            )
            issues.append(
                f"line {node.lineno}: component {component!r} is flattened "
                f"in the feature DataFrame and is not a column; select columns "
                f"whose names start with {prefix!r}"
            )
    test_derived_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        function = node.value.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr.lower() == "concat"
            and _contains_test_reference(node.value)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    test_derived_names.add(target.id)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr.lower()
        fit_uses_test = any(_contains_test_reference(arg) for arg in node.args) or any(
            isinstance(child, ast.Name) and child.id in test_derived_names
            for arg in node.args
            for child in ast.walk(arg)
        )
        if method in FIT_METHODS and fit_uses_test:
            issues.append(
                f"line {node.lineno}: {method} receives test data; fit transformations on training folds only"
            )
        if method in STAT_METHODS and _root_name(node.func.value) in TEST_NAMES:
            issues.append(
                f"line {node.lineno}: test-set {method} statistic is used; derive it from training data"
            )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            function_name = node.func.id.lower()
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr.lower()
        else:
            function_name = ""
        if function_name == "metric_value":
            keyword_names = {
                keyword.arg for keyword in node.keywords if keyword.arg
            }
            invalid_metric_call = (
                ("metric_name" in keyword_names and len(node.args) >= 2)
                or (
                    len(node.args) >= 3
                    and isinstance(node.args[2], ast.Constant)
                    and isinstance(node.args[2].value, str)
                    and not (
                        isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    )
                )
            )
            if invalid_metric_call:
                issues.append(
                    f"line {node.lineno}: metric_value uses the canonical "
                    "signature metric_value(metric_name, target, prediction); "
                    "put the metric first and do not also pass metric_name="
                )
        if group_sensitive and function_name == "train_test_split":
            issues.append(
                f"line {node.lineno}: group/entity-sensitive tasks must use "
                "the harness fold_ids; an independent train_test_split can "
                "place one identity in multiple folds"
            )
        elif harness_split_active and function_name == "train_test_split":
            issues.append(
                f"line {node.lineno}: use the harness-provided evaluation "
                "split; an independent "
                "train_test_split violates the scheduled evaluation protocol"
            )
        if harness_split_active and function_name in {
            "shufflesplit",
            "stratifiedshufflesplit",
            "random_split",
        }:
            issues.append(
                f"line {node.lineno}: {function_name} creates an independent "
                "split after the harness has supplied evaluation rows and fold_ids"
            )
        if harness_folds_active and function_name in {"choice", "sample"}:
            referenced_names = {
                child.id.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
            }
            if any(
                name in referenced_names
                for name in {
                    "x_eval",
                    "y_eval",
                    "fold_ids",
                    "train_indices",
                    "train_indices_full",
                    "train_mask",
                    "train_mask_full",
                }
            ):
                issues.append(
                    f"line {node.lineno}: do not resample harness-owned "
                    "evaluation rows or fold-training indices"
                )
        if (
            "augment" in function_name or function_name.startswith("random_")
        ) and any(
            isinstance(child, ast.Name)
            and any(
                marker in child.id.lower()
                for marker in ("valid", "val_", "test")
            )
            for child in ast.walk(node)
        ):
            issues.append(
                f"line {node.lineno}: stochastic augmentation receives "
                "validation/test data; augmentation must be training-only"
            )
    if harness_folds_active:
        suspicious_masks = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            if isinstance(target, ast.Name)
            and any(
                marker in target.id.lower()
                for marker in ("submask", "subsample", "sample_mask")
            )
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            referenced_names = {
                child.id
                for child in ast.walk(node.slice)
                if isinstance(child, ast.Name)
            }
            if suspicious_masks & referenced_names:
                issues.append(
                    f"line {node.lineno}: do not apply a secondary random sample "
                    "mask to harness-owned fold rows"
                )
    output_methods = {
        "dump",
        "save",
        "savetxt",
        "to_csv",
        "to_feather",
        "to_json",
        "to_parquet",
        "touch",
        "write_bytes",
        "write_text",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        output_path = ""
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            mode = (
                resolved_literal_path(node.args[1])
                if len(node.args) > 1
                else next(
                    (
                        resolved_literal_path(keyword.value)
                        for keyword in node.keywords
                        if keyword.arg == "mode"
                    ),
                    "r",
                )
            )
            if any(flag in mode for flag in ("w", "a", "x", "+")) and node.args:
                output_path = resolved_literal_path(node.args[0])
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr.lower() in output_methods
        ):
            receiver_path = resolved_literal_path(node.func.value)
            output_path = (
                receiver_path
                if receiver_path
                else (
                    resolved_literal_path(node.args[0])
                    if node.args
                    else ""
                )
            )
        if output_path and _targets_task_input(output_path):
            issues.append(
                f"line {node.lineno}: generated code must not write to read-only "
                f"task input path {output_path!r}; write run artifacts in the "
                "current working directory"
            )
        if output_path and (
            Path(output_path).name
            in {
                "evaluation_manifest.json",
                "final_training_manifest.json",
                "fold_assignments.npz",
                "fold_assignments.csv",
                "validation_assignments.npz",
                "validation_assignments.csv",
            }
            or output_path.replace("\\", "/").startswith(
                ".evaluation_contract/"
            )
        ):
            issues.append(
                f"line {node.lineno}: {output_path!r} is a harness-owned "
                "contract artifact; call the evaluation_contract helper and "
                "do not create or overwrite it directly"
            )
    # Preserve order while removing duplicate messages.
    return list(dict.fromkeys(issues))
