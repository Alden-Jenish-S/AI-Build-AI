"""Evaluate the generated baseline against the complete memory-pool tree system."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.llm_utils import get_token_usage, reset_token_usage
from agents.manager_agent import ManagerAgent
from eval.metrics import calculate_ablation_metrics
from runtime_utils import sanitized_subprocess_env


def _prepare_run_input(task_dir: Path, run_dir: Path) -> None:
    """Expose task data below a run directory without copying large datasets."""
    destination = run_dir / "input"
    source = task_dir / "input"
    if source.is_dir():
        if not destination.exists() and not destination.is_symlink():
            try:
                os.symlink(source, destination)
            except OSError:
                shutil.copytree(source, destination)
        return

    destination.mkdir(parents=True, exist_ok=True)
    for source_file in task_dir.iterdir():
        if (
            not source_file.is_file()
            or source_file.suffix not in {".csv", ".tsv", ".txt", ".json"}
            or source_file.name in {"task_config.json", "submission.csv"}
        ):
            continue
        target = destination / source_file.name
        if target.exists() or target.is_symlink():
            continue
        try:
            os.symlink(source_file, target)
        except OSError:
            shutil.copy(source_file, target)


def _run_baseline(manager: ManagerAgent, baseline_dir: Path) -> float:
    """Run the exact generated baseline and return a trustworthy finite score."""
    baseline_dir.mkdir(parents=True, exist_ok=True)
    loader = baseline_dir / "initial_dataloader.py"
    algorithm = baseline_dir / "initial_algorithm.py"
    shutil.copy(manager.task_dir / "initial_dataloader.py", loader)
    shutil.copy(manager.task_dir / "initial_algorithm.py", algorithm)
    _prepare_run_input(manager.task_dir, baseline_dir)

    result_file = baseline_dir / "result.json"
    submission_file = baseline_dir / "submission" / "submission.csv"
    for stale_file in (result_file, submission_file):
        if stale_file.exists() or stale_file.is_symlink():
            stale_file.unlink()

    result = subprocess.run(
        [manager.venv_path, str(algorithm)],
        cwd=baseline_dir,
        capture_output=True,
        text=True,
        timeout=manager.subprocess_timeout,
        env=sanitized_subprocess_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"baseline process exited with {result.returncode}: {result.stderr[-2000:]}"
        )

    if result_file.is_file():
        with open(result_file, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        score = float(result_data["score"])
        declared_direction = result_data.get("direction")
        if declared_direction and declared_direction != manager.metric_direction:
            raise ValueError(
                f"baseline direction {declared_direction!r} does not match "
                f"{manager.metric_direction!r}"
            )
    else:
        matches = re.findall(
            r"(?:score|auc|accuracy|metric|rmse|mae|loss|f1)[:\s=]+"
            r"(-?[0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)",
            result.stdout + "\n" + result.stderr,
            re.IGNORECASE,
        )
        if not matches:
            raise RuntimeError("baseline completed without producing a score")
        score = float(matches[-1])

    if not math.isfinite(score):
        raise ValueError("baseline score must be finite")
    print(f"Derived baseline score: {score}")
    return score


def run_complete_system(
    task_name: str,
    baseline_score: float,
    baseline_dir: Path,
    budget: int,
) -> Dict[str, Any]:
    """Run only the complete pool-enabled tree-search system."""
    print(f"\n>>> Running Complete System on task: {task_name} <<<")
    reset_token_usage()
    os.environ["ABLATION_USE_POOL"] = "1"
    os.environ["ABLATION_USE_TREE"] = "1"

    manager = ManagerAgent(
        task_name=task_name,
        total_budget=budget,
        baseline_score=baseline_score,
        run_suffix="complete_system",
    )
    shutil.copy(baseline_dir / "initial_dataloader.py", manager.task_dir / "initial_dataloader.py")
    shutil.copy(baseline_dir / "initial_algorithm.py", manager.task_dir / "initial_algorithm.py")

    started = time.time()
    best_node_id = manager.run_tree_search()
    if not best_node_id:
        raise RuntimeError("complete system produced no successful implementation node")
    if not manager.generate_final_submission(best_node_id):
        raise RuntimeError("complete system produced an invalid final submission")

    best_result = manager.all_nodes[best_node_id].result or {}
    best_score = best_result.get("score")
    history = []
    pool_additions = 0
    for node_id, node in manager.all_nodes.items():
        if node_id == "root" or not node.executed:
            continue
        technique_record = (node.config or {}).get("technique_record", {})
        status = technique_record.get("status", "pool_miss")
        if node.node_type == "technique" and status == "pool_added":
            pool_additions += 1
        history.append(
            {
                "type": node.node_type,
                "status": status,
                "score": node.result.get("score") if node.result else None,
            }
        )

    metrics = calculate_ablation_metrics(
        history,
        baseline_score,
        maximize=manager.metric_direction == "maximize",
    )
    tokens = get_token_usage()
    return {
        "condition": "Complete system",
        "best_score": best_score,
        "medal_rate": metrics["medal_rate"],
        "gold_rate": metrics["gold_rate"],
        "avg_tokens": (tokens["input_tokens"] + tokens["output_tokens"])
        / max(len(history), 1),
        "pool_hit_rate": metrics["pool_hit_rate"],
        "pool_additions": pool_additions,
        "overcome_rate": metrics["overcome_rate"],
        "time_elapsed": time.time() - started,
    }


def _write_results(
    task_name: str,
    direction: str,
    baseline_score: float,
    baseline_tokens: int,
    complete_result: Dict[str, Any],
) -> Path:
    result_file = PROJECT_ROOT / "eval" / "results.md"
    improvement = (
        complete_result["best_score"] - baseline_score
        if direction == "maximize"
        else baseline_score - complete_result["best_score"]
    )
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("# Baseline vs Complete System\n\n")
        f.write(f"Task evaluated: {task_name}\n\n")
        f.write(
            "| Condition | Best Score | Improvement | Avg tokens/executed node "
            "| Pool-hit rate | New pool techniques | Overcome rate |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        f.write(
            f"| Baseline | {baseline_score:.6f} | — | {baseline_tokens} "
            "| n/a | 0 | n/a |\n"
        )
        f.write(
            f"| Complete system | {complete_result['best_score']:.6f} "
            f"| {improvement:+.6f} | {complete_result['avg_tokens']:.1f} "
            f"| {complete_result['pool_hit_rate']:.1%} "
            f"| {complete_result['pool_additions']} "
            f"| {complete_result['overcome_rate']:.1%} |\n"
        )
    return result_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the generated baseline with the complete agent system."
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
        help="Number of complete-system tree node executions (default: 6).",
    )
    args = parser.parse_args()
    if not args.task_name:
        parser.error("task_name is required when AIBUILDAI_TASK is not set")
    if args.budget < 1:
        parser.error("--budget must be positive")

    reset_token_usage()
    baseline_manager = ManagerAgent(task_name=args.task_name)
    baseline_manager.initialize_task(temperature=0.0)
    baseline_dir = PROJECT_ROOT / "runs" / args.task_name / "baseline"
    baseline_score = _run_baseline(baseline_manager, baseline_dir)
    baseline_usage = get_token_usage()
    baseline_tokens = baseline_usage["input_tokens"] + baseline_usage["output_tokens"]

    complete_result = run_complete_system(
        args.task_name,
        baseline_score,
        baseline_dir,
        args.budget,
    )
    result_file = _write_results(
        args.task_name,
        baseline_manager.metric_direction,
        baseline_score,
        baseline_tokens,
        complete_result,
    )
    print(f"\nEvaluation completed. Results written to {result_file}")


if __name__ == "__main__":
    main()
