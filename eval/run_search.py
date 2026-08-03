"""Launch the method tree directly for one task."""

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
    """Return a serializable, non-negative LLM usage snapshot."""
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token usage cannot be negative")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _write_token_usage_report(
    task_name: str,
    usage: dict[str, Any],
    *,
    runs_root: Path | None = None,
    calls: list[dict] | None = None,
) -> Path:
    """Persist token usage for the direct method-tree search."""
    search = _token_counts(usage)
    root = Path(runs_root) if runs_root is not None else PROJECT_ROOT / "runs"
    report_file = root / task_name / "token_usage.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "task_name": task_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": (
            "Provider-reported LLM token usage when available; otherwise the "
            "llm_utils word-based fallback estimate."
        ),
        "method_tree": search,
        "calls": list(calls or []),
    }
    report_file.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report_file


def _write_results(
    manager: ManagerAgent,
    best_node_id: str | None,
    *,
    elapsed_seconds: float,
    usage: dict[str, int],
    submission_written: bool = False,
    failure: str | None = None,
) -> Path:
    """Write a method-tree run summary without a synthetic comparison model."""
    best_node = (
        manager.all_nodes.get(best_node_id) if best_node_id else None
    )
    best_score = (
        float(best_node.result["score"])
        if best_node is not None and best_node.result
        else None
    )
    technique_nodes = [
        node
        for node in manager.all_nodes.values()
        if node.node_type == "technique"
        and node.parent_id is not None
        and node.executed
    ]
    pool_hits = sum(
        1
        for node in technique_nodes
        if ((node.config or {}).get("technique_record") or {}).get("status")
        == "pool_hit"
    )
    result_file = manager.run_root / "results.md"
    result_file.write_text(
        "\n".join(
            (
                "# Method Tree Search",
                "",
                f"Task: {manager.task_name}",
                f"Metric: {manager.metric_name} ({manager.metric_direction})",
                f"Status: {'completed' if submission_written else 'failed'}",
                f"Failure: {failure or 'none'}",
                f"Best node: {best_node_id or 'none'}",
                (
                    f"Best score: {best_score:.8f}"
                    if best_score is not None
                    else "Best score: none"
                ),
                (
                    f"Best fidelity: {best_node.fidelity}"
                    if best_node is not None
                    else "Best fidelity: none"
                ),
                f"Final submission written: {submission_written}",
                (
                    "Completed evaluation experiments: "
                    f"{manager.experiments_executed}"
                ),
                (
                    "Implementation attempts: "
                    f"{getattr(manager, 'implementation_attempts', 0)}"
                ),
                f"Technique nodes resolved: {len(technique_nodes)}",
                (
                    "Memory-pool hit rate: "
                    f"{pool_hits / len(technique_nodes):.1%}"
                    if technique_nodes
                    else "Memory-pool hit rate: 0.0%"
                ),
                f"Elapsed seconds: {elapsed_seconds:.2f}",
                f"LLM tokens: {usage['total_tokens']}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return result_file


def run_method_tree(task_name: str, budget: int) -> dict[str, Any]:
    """Run search immediately, select the best method, and emit a submission."""
    reset_token_usage()
    manager = ManagerAgent(task_name=task_name, total_budget=budget)
    started = time.time()
    best_node_id = None
    submission_written = False
    failure: Exception | None = None
    try:
        best_node_id = manager.run_tree_search()
        if not best_node_id:
            raise RuntimeError(
                "method tree produced no successful implementation node"
            )
        submission_written = manager.generate_final_submission(best_node_id)
        if not submission_written:
            raise RuntimeError(
                "method tree produced an invalid final submission"
            )
    except Exception as exc:
        failure = exc
    elapsed_seconds = time.time() - started
    usage_snapshot = get_token_usage()
    usage = _token_counts(usage_snapshot)
    token_report = _write_token_usage_report(
        task_name,
        usage,
        calls=list(usage_snapshot.get("calls", [])),
    )
    results_file = _write_results(
        manager,
        best_node_id,
        elapsed_seconds=elapsed_seconds,
        usage=usage,
        submission_written=submission_written,
        failure=str(failure) if failure else None,
    )
    if failure is not None:
        raise failure
    return {
        "best_node_id": best_node_id,
        "best_score": manager.all_nodes[best_node_id].result["score"],
        "metric": manager.metric_name,
        "direction": manager.metric_direction,
        "experiments": manager.experiments_executed,
        "implementation_attempts": getattr(
            manager, "implementation_attempts", 0
        ),
        "elapsed_seconds": elapsed_seconds,
        "token_usage": usage,
        "results_file": str(results_file),
        "token_report": str(token_report),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the autonomous method tree directly for one task."
    )
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
            "Number of completed evaluation experiments; planning and bounded "
            "technical recovery actions are free (default: 6)."
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
        "\nMethod-tree search completed. "
        f"Best node={result['best_node_id']}, "
        f"{result['metric']}={float(result['best_score']):.8f}."
    )
    print(
        "LLM token usage: "
        f"input={usage['input_tokens']}, "
        f"output={usage['output_tokens']}, "
        f"total={usage['total_tokens']}"
    )
    print(f"Results written to {result['results_file']}")
    print(f"Token usage report written to {result['token_report']}")


if __name__ == "__main__":
    main()
