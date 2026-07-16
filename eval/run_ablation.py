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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.llm_utils import get_token_usage, reset_token_usage
from agents.initial_agent import InitialAgent
from agents.manager_agent import ManagerAgent
from eval.metrics import calculate_ablation_metrics
from runtime_utils import accelerator_subprocess_env
from evaluation_contract import validate_evaluation_outputs


def _token_counts(usage: Dict[str, Any]) -> Dict[str, int]:
    """Return a serializable, non-negative snapshot of LLM token usage."""
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
    baseline_usage: Dict[str, Any],
    complete_usage: Dict[str, Any],
    runs_root: Path | None = None,
    baseline_calls: list[dict] | None = None,
    complete_calls: list[dict] | None = None,
) -> Path:
    """Persist exact baseline, complete-system, and end-to-end token totals."""
    baseline = _token_counts(baseline_usage)
    complete_system = _token_counts(complete_usage)
    overall = {
        "input_tokens": baseline["input_tokens"] + complete_system["input_tokens"],
        "output_tokens": baseline["output_tokens"] + complete_system["output_tokens"],
    }
    overall["total_tokens"] = overall["input_tokens"] + overall["output_tokens"]
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
        "baseline": baseline,
        "complete_system": complete_system,
        "overall": overall,
        "calls": {
            "baseline": list(baseline_calls or []),
            "complete_system": list(complete_calls or []),
        },
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report_file


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


def _run_baseline(
    manager: ManagerAgent,
    baseline_dir: Path,
    max_debug_attempts: int = 2,
) -> float:
    """Run and, when necessary, repair the generated baseline."""
    if max_debug_attempts < 0:
        raise ValueError("max_debug_attempts cannot be negative")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    loader = baseline_dir / "initial_dataloader.py"
    algorithm = baseline_dir / "initial_algorithm.py"
    shutil.copy(manager.task_dir / "initial_dataloader.py", loader)
    shutil.copy(manager.task_dir / "initial_algorithm.py", algorithm)
    contract_source = PROJECT_ROOT / "evaluation_contract.py"
    if contract_source.is_file():
        shutil.copy2(contract_source, baseline_dir / "evaluation_contract.py")
    _prepare_run_input(manager.task_dir, baseline_dir)

    result_file = baseline_dir / "result.json"
    submission_file = baseline_dir / "submission" / "submission.csv"
    debug_log = baseline_dir / "baseline_debug.log"
    if debug_log.exists() or debug_log.is_symlink():
        debug_log.unlink()
    debugger = InitialAgent(model_name=getattr(manager, "model_name", None))

    for attempt in range(max_debug_attempts + 1):
        for stale_file in (
            result_file,
            submission_file,
            baseline_dir / "oof_predictions.csv",
            baseline_dir / "evaluation_manifest.json",
            baseline_dir / "fold_assignments.csv",
        ):
            if stale_file.exists() or stale_file.is_symlink():
                stale_file.unlink()

        try:
            result = subprocess.run(
                [manager.venv_path, str(algorithm)],
                cwd=baseline_dir,
                capture_output=True,
                text=True,
                timeout=manager.subprocess_timeout,
                env=accelerator_subprocess_env(
                    getattr(manager, "preferred_accelerator", "cpu")
                ),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"baseline process exited with {result.returncode}\n"
                    f"stdout:\n{result.stdout[-2000:]}\n"
                    f"stderr:\n{result.stderr[-4000:]}"
                )

            if result_file.is_file():
                with open(result_file, "r", encoding="utf-8") as f:
                    result_data = json.load(f)
                score = float(result_data["score"])
                declared_metric = result_data.get("metric")
                if declared_metric:
                    manager.metric_name = str(declared_metric)
                declared_direction = result_data.get("direction")
                if declared_direction and declared_direction != manager.metric_direction:
                    raise ValueError(
                        f"baseline direction {declared_direction!r} does not match "
                        f"{manager.metric_direction!r}"
                    )
                contract_files_exist = (
                    (baseline_dir / "evaluation_manifest.json").is_file()
                    and (baseline_dir / "oof_predictions.csv").is_file()
                )
                if getattr(manager, "enforce_evaluation_contract", False) and not contract_files_exist:
                    raise ValueError(
                        "baseline omitted the harness-owned evaluation manifest or OOF predictions"
                    )
                if contract_files_exist:
                    validated = validate_evaluation_outputs(
                        baseline_dir, "full", manager.metric_name
                    )
                    score = float(validated["cv_mean"])
                    result_data.update(validated)
                    result_data["score"] = score
                    with open(result_file, "w", encoding="utf-8") as f:
                        json.dump(result_data, f, indent=2)
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
            if not submission_file.is_file() or submission_file.stat().st_size == 0:
                raise RuntimeError(
                    "baseline completed without producing submission/submission.csv"
                )
            print(f"Derived baseline score: {score}")
            return score
        except (
            subprocess.TimeoutExpired,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            failure_output = f"{type(exc).__name__}: {exc}"
            with open(debug_log, "a", encoding="utf-8") as f:
                f.write(f"\n=== Baseline attempt {attempt + 1} failed ===\n")
                f.write(failure_output + "\n")
            if attempt >= max_debug_attempts:
                raise RuntimeError(
                    f"baseline failed after {attempt + 1} attempts; see {debug_log}: "
                    f"{failure_output[-2000:]}"
                ) from exc

            print(
                f"Baseline Debug Attempt {attempt + 1}/{max_debug_attempts}: "
                "execution failed; asking InitialAgent to repair initial_algorithm.py..."
            )
            debugger.repair_initial_algorithm(
                loader,
                algorithm,
                failure_output,
                getattr(manager, "metric_name", "score"),
                manager.metric_direction,
            )

    raise AssertionError("unreachable baseline execution state")


def run_complete_system(
    task_name: str,
    baseline_score: float,
    baseline_dir: Path,
    budget: int,
    metric_name: str | None = None,
    metric_direction: str | None = None,
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
    if metric_name:
        manager.metric_name = metric_name
    if metric_direction:
        manager.metric_direction = metric_direction
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
        status = (
            (node.result or {}).get("status", "failed")
            if node.node_type == "implementation"
            else technique_record.get("status", "pool_miss")
        )
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
    tokens = _token_counts(get_token_usage())
    token_calls = list(get_token_usage().get("calls", []))
    implementation_count = manager.experiments_executed
    return {
        "condition": "Complete system",
        "best_score": best_score,
        "medal_rate": metrics["medal_rate"],
        "gold_rate": metrics["gold_rate"],
        "avg_tokens": tokens["total_tokens"] / max(implementation_count, 1),
        "token_usage": tokens,
        "token_calls": token_calls,
        "pool_hit_rate": metrics["pool_hit_rate"],
        "pool_additions": pool_additions,
        "overcome_rate": metrics["overcome_rate"],
        "normalized_improvement": metrics["best_normalized_improvement"],
        "experiments": implementation_count,
        "best_fidelity": manager.all_nodes[best_node_id].fidelity,
        "time_elapsed": time.time() - started,
    }


def _write_results(
    task_name: str,
    direction: str,
    baseline_score: float,
    baseline_usage: Dict[str, Any],
    complete_result: Dict[str, Any],
    overall_usage: Dict[str, Any],
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
            "| Pool-hit rate | New pool techniques | Overcome rate "
            "| Experiments | Best fidelity | Normalized improvement |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|\n")
        f.write(
            f"| Baseline | {baseline_score:.6f} | — | {baseline_usage['total_tokens']} "
            "| n/a | 0 | n/a | 1 | full | 0.0% |\n"
        )
        f.write(
            f"| Complete system | {complete_result['best_score']:.6f} "
            f"| {improvement:+.6f} | {complete_result['avg_tokens']:.1f} "
            f"| {complete_result['pool_hit_rate']:.1%} "
            f"| {complete_result['pool_additions']} "
            f"| {complete_result['overcome_rate']:.1%} "
            f"| {complete_result['experiments']} "
            f"| {complete_result['best_fidelity']} "
            f"| {complete_result['normalized_improvement']:.1%} |\n"
        )
        f.write("\n## LLM Token Usage\n\n")
        f.write("| Phase | Input tokens | Output tokens | Total tokens |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(
            f"| Baseline | {baseline_usage['input_tokens']} | "
            f"{baseline_usage['output_tokens']} | {baseline_usage['total_tokens']} |\n"
        )
        complete_usage = complete_result["token_usage"]
        f.write(
            f"| Complete system | {complete_usage['input_tokens']} | "
            f"{complete_usage['output_tokens']} | {complete_usage['total_tokens']} |\n"
        )
        f.write(
            f"| Overall | {overall_usage['input_tokens']} | "
            f"{overall_usage['output_tokens']} | {overall_usage['total_tokens']} |\n"
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
        help="Number of implementation experiments; planning actions are free (default: 6).",
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
    baseline_snapshot = get_token_usage()
    baseline_usage = _token_counts(baseline_snapshot)
    baseline_calls = list(baseline_snapshot.get("calls", []))

    complete_result = run_complete_system(
        args.task_name,
        baseline_score,
        baseline_dir,
        args.budget,
        metric_name=baseline_manager.metric_name,
        metric_direction=baseline_manager.metric_direction,
    )
    token_report_file = _write_token_usage_report(
        args.task_name,
        baseline_usage,
        complete_result["token_usage"],
        baseline_calls=baseline_calls,
        complete_calls=complete_result.get("token_calls", []),
    )
    overall_usage = _token_counts(
        {
            "input_tokens": (
                baseline_usage["input_tokens"]
                + complete_result["token_usage"]["input_tokens"]
            ),
            "output_tokens": (
                baseline_usage["output_tokens"]
                + complete_result["token_usage"]["output_tokens"]
            ),
        }
    )
    result_file = _write_results(
        args.task_name,
        baseline_manager.metric_direction,
        baseline_score,
        baseline_usage,
        complete_result,
        overall_usage,
    )
    print(
        "\nOverall LLM token usage: "
        f"input={overall_usage['input_tokens']}, "
        f"output={overall_usage['output_tokens']}, "
        f"total={overall_usage['total_tokens']}"
    )
    print(f"Evaluation completed. Results written to {result_file}")
    print(f"Token usage report written to {token_report_file}")


if __name__ == "__main__":
    main()
