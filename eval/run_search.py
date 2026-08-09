"""Run the lean implementation search for one task."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.llm_utils import get_token_usage, reset_token_usage
from agents.manager_agent import ManagerAgent


def _token_counts(usage: dict[str, Any]) -> dict[str, int]:
    input_tokens = max(0, int(usage.get("input_tokens", 0)))
    output_tokens = max(0, int(usage.get("output_tokens", 0)))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _write_token_usage_report(manager: ManagerAgent, usage: dict[str, Any]) -> Path:
    """Write compact per-call and aggregate LLM metrics for this run."""
    counts = _token_counts(usage)
    path = manager.run_root / "token_usage.json"
    payload = {
        "task_name": manager.task_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": (
            "Provider-reported token usage when available; otherwise the "
            "shared client's word-based fallback estimate."
        ),
        "method_tree": counts,
        "search_accounting": {
            "new_idea_budget": manager.total_budget,
            "new_idea_budget_used": manager.experiments_executed,
            "completed_implementations": manager.completed_implementations,
            "free_tuning_attempts": manager.tuning_attempts,
            "implementation_attempts": manager.implementation_attempts,
        },
        "calls": list(usage.get("calls", [])),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _write_results(
    manager: ManagerAgent,
    *,
    elapsed_seconds: float,
    usage: dict[str, int],
    failure: str | None,
) -> Path:
    best = manager.all_nodes.get(manager.best_node_id or "")
    best_score = best.result.get("score") if best and best.result else None
    completed = sum(
        1 for node in manager.all_nodes.values()
        if node.node_type == "implementation"
        and node.result
        and node.result.get("status") == "completed"
    )
    pruned = sum(
        1 for node in manager.all_nodes.values()
        if node.result and node.result.get("pruned")
    )
    node_lines: list[str] = []
    for node in manager.all_nodes.values():
        result = node.result or {}
        node_lines.append(
            f"- `{manager.node_label(node.node_id)}` — {node.node_type}/{node.operator or 'run'}; "
            f"status={result.get('status', 'unknown')}; "
            f"score={result.get('score', 'none')}; "
            f"pruned={bool(result.get('pruned'))}"
        )
        if result.get("status") != "completed" and result.get("diagnostics"):
            diagnostic = " ".join(str(result["diagnostics"])[-1200:].split())
            node_lines.append(f"  - last error: {diagnostic}")
    path = manager.run_root / "results.md"
    path.write_text(
        "\n".join(
            [
                "# Implementation search",
                "",
                f"Task: {manager.task_name}",
                f"Goal: {manager.task_analysis.goal}",
                f"Metric: {manager.metric_name} ({manager.metric_direction})",
                f"Status: {'completed' if manager.final_output_path else 'failed'}",
                f"Failure: {failure or 'none'}",
                f"Best node: {manager.node_label(manager.best_node_id)}",
                f"Best score: {best_score if best_score is not None else 'none'}",
                f"Final output: {manager.final_output_path or 'none'}",
                f"Runnable implementations: {completed}",
                f"New-idea budget used: {manager.experiments_executed}/{manager.total_budget}",
                f"Free tuning nodes attempted: {manager.tuning_attempts}",
                f"Implementation nodes attempted: {manager.implementation_attempts}",
                f"Non-improving branches pruned: {pruned}",
                f"Elapsed seconds: {elapsed_seconds:.2f}",
                f"LLM input tokens: {usage['input_tokens']}",
                f"LLM output tokens: {usage['output_tokens']}",
                f"Tree state: {manager.run_root / 'tree_state.json'}",
                f"Method tree image: {manager.run_root / 'method_tree.png'}",
                f"Token usage report: {manager.run_root / 'token_usage.json'}",
                "",
                "## Nodes",
                "",
                *node_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def run_method_tree(task_name: str, budget: int) -> dict[str, Any]:
    reset_token_usage()
    manager = ManagerAgent(task_name=task_name, total_budget=budget)
    started = time.monotonic()
    failure: Exception | None = None
    best_node_id: str | None = None
    try:
        best_node_id = manager.run_tree_search()
        if best_node_id is None:
            raise RuntimeError("No implementation produced a deliverable; see node diagnostics.")
        if not manager.generate_final_submission(best_node_id):
            raise RuntimeError("The strongest runnable implementation did not expose its deliverable.")
    except Exception as exc:
        failure = exc
    elapsed = time.monotonic() - started
    try:
        tree_artifacts = manager.finalize_run_artifacts()
    except Exception as exc:
        tree_artifacts = {}
        print(f"ManagerAgent WARNING: Could not refresh final tree reports: {exc}", flush=True)
    usage_snapshot = get_token_usage()
    usage = _token_counts(usage_snapshot)
    token_report = _write_token_usage_report(manager, usage_snapshot)
    results_file = _write_results(
        manager,
        elapsed_seconds=elapsed,
        usage=usage,
        failure=str(failure) if failure else None,
    )
    if failure is not None:
        raise failure
    best = manager.all_nodes[manager.best_node_id]
    return {
        "best_node_id": manager.best_node_id,
        "best_node": manager.node_label(manager.best_node_id),
        "best_score": best.result["score"],
        "metric": manager.metric_name,
        "direction": manager.metric_direction,
        "experiments": manager.experiments_executed,
        "new_idea_budget_used": manager.experiments_executed,
        "completed_implementations": manager.completed_implementations,
        "tuning_attempts": manager.tuning_attempts,
        "implementation_attempts": manager.implementation_attempts,
        "elapsed_seconds": elapsed,
        "token_usage": usage,
        "token_report": str(token_report),
        "tree_state": str(tree_artifacts.get("tree_state", manager.run_root / "tree_state.json")),
        "method_tree": str(tree_artifacts.get("method_tree", manager.run_root / "method_tree.png")),
        "results_file": str(results_file),
        "final_output": str(manager.final_output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct task analysis and implementation search.")
    parser.add_argument(
        "task_name",
        nargs="?",
        default=os.getenv("AIBUILDAI_TASK"),
        help="Task directory name under tasks/ (or set AIBUILDAI_TASK).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=6,
        help=(
            "Completed new root/branch ideas to fund; tuning is free and failures "
            "preserve this budget (default: 6)."
        ),
    )
    args = parser.parse_args()
    if not args.task_name:
        parser.error("task_name is required when AIBUILDAI_TASK is not set")
    if args.budget < 1:
        parser.error("--budget must be positive")

    result = run_method_tree(args.task_name, args.budget)
    usage = result["token_usage"]
    print(
        f"Search completed: best={result['best_node']}, "
        f"{result['metric']}={float(result['best_score']):.8f}."
    )
    print(f"Final output: {result['final_output']}")
    print(
        f"Search accounting: new ideas={result['new_idea_budget_used']}/{args.budget}, "
        f"free tuning attempts={result['tuning_attempts']}, "
        f"completed implementations={result['completed_implementations']}."
    )
    print(f"LLM tokens: input={usage['input_tokens']}, output={usage['output_tokens']}")
    print(f"Results: {result['results_file']}")
    print(f"Tree state: {result['tree_state']}")
    print(f"Token usage: {result['token_report']}")
    print(f"Method tree: {result['method_tree']}")


if __name__ == "__main__":
    main()
