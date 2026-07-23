"""Offline, review-gated harness evolution from completed experiment traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.llm_utils import call_llm


def collect_trace_summary(run_dirs: List[Path]) -> Dict[str, Any]:
    experiments = []
    for run_dir in run_dirs:
        state_file = run_dir / "tree_state.json"
        if not state_file.is_file():
            continue
        state = json.loads(state_file.read_text(encoding="utf-8"))
        for node_id, node in state.get("nodes", {}).items():
            if node.get("node_type") != "implementation" or not node.get("executed"):
                continue
            result = node.get("result") or {}
            experiments.append(
                {
                    "task": state.get("task_name"),
                    "node_id": node_id,
                    "operator": node.get("operator"),
                    "fidelity": node.get("fidelity"),
                    "score": result.get("score"),
                    "reward": result.get("reward"),
                    "status": result.get("status"),
                    "diagnostics_tail": result.get("diagnostics_tail", "")[-1500:],
                }
            )
    return {"experiment_count": len(experiments), "experiments": experiments}


def propose_harness_candidates(summary: Dict[str, Any], model: str | None = None) -> list:
    if not summary["experiments"]:
        raise ValueError("No executed implementation traces were found")
    response = call_llm(
        "You optimize an autonomous ML engineering harness offline. Analyze cross-task traces and "
        "propose conservative prompt/control-flow amendments. Return ONLY a JSON list of up to five "
        "items with fields target_component, hypothesis, proposed_change, expected_metric, risks. "
        "Do not propose task-specific shortcuts or any use of held-out/test labels.",
        json.dumps(summary, indent=2, default=str),
        model=model,
        temperature=0.7,
    )
    if "```json" in response:
        response = response.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in response:
        response = response.split("```", 1)[1].split("```", 1)[0]
    candidates = json.loads(response.strip())
    if not isinstance(candidates, list):
        raise ValueError("Harness optimizer did not return a JSON list")
    return candidates[:5]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate review-gated harness-evolution candidates from completed runs."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "harness_candidates.json",
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    summary = collect_trace_summary([path.resolve() for path in args.run_dirs])
    candidates = propose_harness_candidates(summary, model=args.model)
    output = args.output.resolve()
    runs_root = (PROJECT_ROOT / "runs").resolve()
    if output != runs_root and runs_root not in output.parents:
        parser.error("--output must be inside the project's runs/ directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    print(f"Wrote {len(candidates)} review-gated candidates to {output}")


if __name__ == "__main__":
    main()
