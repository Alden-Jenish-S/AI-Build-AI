"""Static guardrails for common leakage patterns in generated ML pipelines."""

from __future__ import annotations

import ast
from typing import List


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


def inspect_generated_code(code: str) -> List[str]:
    """Return high-confidence leakage defects that should be repaired before execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # The normal debugging loop supplies better syntax diagnostics.

    issues: List[str] = []
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
                _literal_path(node.args[1])
                if len(node.args) > 1
                else next(
                    (
                        _literal_path(keyword.value)
                        for keyword in node.keywords
                        if keyword.arg == "mode"
                    ),
                    "r",
                )
            )
            if any(flag in mode for flag in ("w", "a", "x", "+")) and node.args:
                output_path = _literal_path(node.args[0])
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr.lower() in output_methods
        ):
            receiver_path = _literal_path(node.func.value)
            output_path = (
                receiver_path
                if receiver_path
                else (_literal_path(node.args[0]) if node.args else "")
            )
        if output_path and _targets_task_input(output_path):
            issues.append(
                f"line {node.lineno}: generated code must not write to read-only "
                f"task input path {output_path!r}; write run artifacts in the "
                "current working directory"
            )
    # Preserve order while removing duplicate messages.
    return list(dict.fromkeys(issues))
