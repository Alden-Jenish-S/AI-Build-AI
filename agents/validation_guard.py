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
    # Preserve order while removing duplicate messages.
    return list(dict.fromkeys(issues))
