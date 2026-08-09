"""Lean score-driven manager with tuning, pruning, and mid-search merging."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime_utils import absolute_path_without_symlink_resolution, validate_path_component
from tree.node import NodeState
from tree.scheduler import UCB1Scheduler

from .aggregator_agent import AggregatorAgent
from .architecture_policy import classify_architecture, coverage_from_tracks
from .council import CouncilBrief, CouncilCoordinator
from .implementation_agent import ImplementationAgent
from .submission_validator import SubmissionValidator
from .task_analyzer import TaskAnalysis, TaskAnalyzer
from .technique_agent import TechniqueAgent


class ManagerAgent:
    """Build only on runnable implementations and stop weak branches early."""

    def __init__(
        self,
        task_name: str,
        total_budget: int = 10,
        venv_path: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.task_name = validate_path_component(task_name, "task_name")
        if not isinstance(total_budget, int) or isinstance(total_budget, bool) or total_budget < 1:
            raise ValueError("total_budget must be a positive integer")
        self.total_budget = total_budget
        self.model_name = model_name
        self.project_root = Path(__file__).resolve().parent.parent
        self.task_dir = self.project_root / "tasks" / self.task_name
        if not self.task_dir.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {self.task_dir}")

        # A task has one flat active run. Historical runs are placed beside it
        # with readable `-previous` names; no session/timestamp layer is used.
        self.run_root = self.project_root / "runs" / self.task_name
        selected_python = absolute_path_without_symlink_resolution(venv_path) if venv_path else Path(sys.executable)
        self.python = str(selected_python if selected_python.is_file() else Path(sys.executable))
        self.task_analyzer = TaskAnalyzer(model_name=model_name)
        self.task_analysis: TaskAnalysis = self.task_analyzer.analyze(self.task_dir)

        # Do not displace a useful prior run until the new task can at least be
        # inventoried successfully.
        self._archive_existing_run()
        self.run_root.mkdir(parents=True, exist_ok=True)
        (self.run_root / "task_analysis.md").write_text(self.task_analysis.report, encoding="utf-8")

        self.metric_name = self.task_analysis.metric
        self.metric_direction = self.task_analysis.direction
        self.technique_agent = TechniqueAgent(model_name=model_name)
        self.council_brief: CouncilBrief | None = None
        self.council_coordinator = CouncilCoordinator(
            python=self.python,
            model_name=model_name,
            enable_web=self._env_enabled("AIBUILDAI_COUNCIL_WEB", default=True),
            enable_generated_diagnostics=self._env_enabled(
                "AIBUILDAI_COUNCIL_DIAGNOSTICS", default=True
            ),
        )
        self.submission_validator = SubmissionValidator()
        self.implementation_agent = ImplementationAgent(
            self.python,
            model_name=model_name,
            submission_validator=self.submission_validator,
        )
        self.aggregator_agent = AggregatorAgent()
        self.scheduler = UCB1Scheduler(total_budget)

        search_root = NodeState(
            node_id="root",
            parent_id=None,
            node_type="planning",
            plan="Search root",
            operator="root",
            executed=True,
            result={"status": "completed", "planning_only": True},
        )
        self.all_nodes: dict[str, NodeState] = {"root": search_root}
        # `experiments_executed` is intentionally the charged idea budget:
        # root/refine/diversify/merge/recovery completions consume it; tuning
        # completions do not.
        self.experiments_executed = 0
        self.completed_implementations = 0
        self.implementation_attempts = 0
        self.tuning_attempts = 0
        self.best_node_id: str | None = None
        self.final_output_path: Path | None = None
        self._node_counter = 0
        self._merged_pairs: set[frozenset[str]] = set()
        self._merge_attempts = 0
        self._expanded_nodes: set[str] = set()
        self._backup_plans: list[str] = []
        # Two independent roots are enough to establish diversity for larger
        # searches; additional budget is more valuable on measured refinement.
        self.initial_fanout = min(2, max(1, self.total_budget // 3))
        self.max_tune_depth = 1
        self.architecture_exploration_enabled = self._env_enabled(
            "AIBUILDAI_ARCHITECTURE_EXPLORATION", default=True
        )
        self.architecture_min_budget = max(
            2, self._env_int("AIBUILDAI_ARCHITECTURE_MIN_BUDGET", 3)
        )
        self.plateau_patience = max(
            1, self._env_int("AIBUILDAI_PLATEAU_PATIENCE", 1)
        )
        self.plateau_relative_gain = max(
            0.0,
            min(
                0.05,
                self._env_float("AIBUILDAI_PLATEAU_RELATIVE_GAIN", 5e-4),
            ),
        )
        # Tuning is free with respect to the idea budget, but independent caps
        # guarantee termination under persistent external/generated failures.
        self.tuning_attempt_limit = max(1, self.total_budget)
        self.attempt_limit = max(self.total_budget * 5, self.total_budget + 12)
        self._log(
            f"Prepared task '{self.task_name}' with {len(self.task_analysis.files)} files; "
            f"metric={self.metric_name} ({self.metric_direction}); budget={self.total_budget}."
        )
        self._persist_tree_state()

    @staticmethod
    def _env_enabled(name: str, *, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().casefold() not in {"0", "false", "no", "off"}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return int(default)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return float(default)

    @staticmethod
    def _log(message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{timestamp}] ManagerAgent: {message}", flush=True)

    def _archive_existing_run(self) -> None:
        """Keep the prior run without timestamp-named directories."""
        if not self.run_root.is_dir() or not any(self.run_root.iterdir()):
            return
        base = self.run_root.parent / f"{self.run_root.name}-previous"
        destination = base
        index = 2
        while destination.exists():
            destination = self.run_root.parent / f"{base.name}-{index}"
            index += 1
        self.run_root.rename(destination)
        self._normalize_archived_node_folders(destination)
        self._log(f"Archived the previous run as {destination.name}.")

    @staticmethod
    def _normalize_archived_node_folders(archive: Path) -> None:
        """Remove legacy timestamp and underscore node directories from an archive."""
        def legacy_number(name: str) -> int | None:
            simple = re.fullmatch(r"node_?(\d+)", name)
            if simple:
                return int(simple.group(1))
            timestamped = re.fullmatch(r"node_\d{8}T\d+_(\d+)(?:_.*)?", name)
            return int(timestamped.group(1)) if timestamped else None

        candidates: list[Path] = [
            path for path in sorted(archive.iterdir())
            if path.is_dir() and legacy_number(path.name) is not None
        ]
        legacy_root = archive / "nodes"
        if legacy_root.is_dir():
            for session in sorted(legacy_root.iterdir()):
                if session.is_dir():
                    candidates.extend(
                        path for path in sorted(session.iterdir())
                        if path.is_dir() and legacy_number(path.name) is not None
                    )

        used_numbers = {
            number
            for path in archive.iterdir()
            if path.is_dir()
            for number in [legacy_number(path.name)]
            if number is not None and re.fullmatch(r"node\d+", path.name)
        }
        for source in candidates:
            number = legacy_number(source.name)
            if number is None:
                continue
            target = archive / f"node{number}"
            if target.exists() and target != source:
                number = max(used_numbers, default=0) + 1
                while (archive / f"node{number}").exists():
                    number += 1
                target = archive / f"node{number}"
            if source != target:
                source.rename(target)
            used_numbers.add(number)
        if legacy_root.is_dir():
            shutil.rmtree(legacy_root)

    @staticmethod
    def node_label(node_id: str | None) -> str:
        if not node_id:
            return "none"
        if node_id == "root":
            return "Search Root"
        match = re.fullmatch(r"node_?(\d+)", node_id)
        if match:
            return f"Node {int(match.group(1))}"
        return str(node_id)

    def _node_payload(self, node: NodeState) -> dict[str, Any]:
        result = dict(node.result or {})
        if result.get("diagnostics"):
            result["diagnostics"] = str(result["diagnostics"])[-4000:]
        return {
            "node_id": node.node_id,
            "display_name": self.node_label(node.node_id),
            "parent_id": node.parent_id,
            "node_type": node.node_type,
            "operator": node.operator,
            "plan": (node.plan or "")[:5000],
            "executed": node.executed,
            "children_ids": list(node.children_ids),
            "config": {
                key: value
                for key, value in (node.config or {}).items()
                if key in {
                    "priority", "base_node_id", "companion_node_id",
                    "tune_depth", "refine_depth", "diversify_depth",
                    "materialized", "pruned_reason", "replacement",
                    "hypothesis_id", "council_brief_hash",
                    "evaluation_protocol_hash", "architecture_track",
                    "architecture_trigger",
                }
            } or None,
            "visits": node.visits,
            "total_reward": node.total_reward,
            "result": result or None,
        }

    def _persist_node(self, node: NodeState) -> Path:
        if node.node_id == "root":
            return self.run_root / "tree_state.json"
        node_dir = self.run_root / node.node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        path = node_dir / "node_state.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._node_payload(node), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def _persist_tree_state(self) -> Path:
        """Persist a compact progress snapshot; it never gates execution."""
        for node in self.all_nodes.values():
            self._persist_node(node)
        path = self.run_root / "tree_state.json"
        payload = {
            "task_name": self.task_name,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "metric": self.metric_name,
            "direction": self.metric_direction,
            "budget": self.total_budget,
            "budget_kind": "new_branch_ideas",
            "budget_used": self.experiments_executed,
            "budget_remaining": max(0, self.total_budget - self.experiments_executed),
            "implementation_attempts": self.implementation_attempts,
            "attempt_limit": self.attempt_limit,
            "completed_implementations": self.completed_implementations,
            "tuning_attempts": self.tuning_attempts,
            "tuning_attempt_limit": self.tuning_attempt_limit,
            "architecture_policy": {
                "enabled": self.architecture_exploration_enabled,
                "minimum_idea_budget": self.architecture_min_budget,
                "plateau_patience": self.plateau_patience,
                "relative_gain_threshold": self.plateau_relative_gain,
                "coverage": self._architecture_coverage(),
                "plateau": self._plateau_state(),
            },
            "best_node_id": self.best_node_id,
            "best_node": self.node_label(self.best_node_id),
            "final_output": str(self.final_output_path) if self.final_output_path else None,
            "council": (
                {
                    "status": self.council_brief.status,
                    "brief_hash": self.council_brief.brief_hash,
                    "evaluation_protocol_hash": (
                        self.council_brief.evaluation_protocol.protocol_hash
                    ),
                    "artifact": str(self.run_root / "council" / "council_brief.json"),
                }
                if self.council_brief is not None
                else None
            ),
            "nodes": {
                node_id: self._node_payload(node)
                for node_id, node in self.all_nodes.items()
            },
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path

    def _new_node_id(self, operator: str) -> str:
        self._node_counter += 1
        return f"node{self._node_counter}"

    @staticmethod
    def _hypothesis_id(plan: str) -> str | None:
        match = re.search(r"(?im)^\s*Hypothesis ID:\s*([A-Za-z0-9_.-]+)", plan)
        return match.group(1) if match else None

    def _council_node_config(self, plan: str) -> dict[str, str]:
        if self.council_brief is None:
            return {}
        values = {
            "council_brief_hash": self.council_brief.brief_hash,
            "evaluation_protocol_hash": (
                self.council_brief.evaluation_protocol.protocol_hash
            ),
        }
        hypothesis_id = self._hypothesis_id(plan)
        if hypothesis_id:
            values["hypothesis_id"] = hypothesis_id
        return values

    def _prepare_research_council(self) -> None:
        if self.council_brief is not None:
            return
        if not self._env_enabled("AIBUILDAI_COUNCIL", default=True):
            self._log("ML research council disabled by AIBUILDAI_COUNCIL.")
            return
        self._log("Starting the pre-search ML research council.")
        try:
            brief = self.council_coordinator.run(self.task_analysis, self.run_root)
        except Exception as exc:
            self._log(
                "Council preflight could not complete; preserving legacy planning fallback: "
                f"{type(exc).__name__}: {exc}"
            )
            return
        self.council_brief = brief
        self.technique_agent.set_council_brief(brief)
        self.initial_fanout = min(
            self.total_budget,
            max(1, int(brief.recommended_root_count)),
        )
        self._log(
            f"Council {brief.status}: {len(brief.selected_portfolio)} selected hypotheses; "
            f"initial roots={self.initial_fanout}; protocol="
            f"{brief.evaluation_protocol.protocol_hash[:12]}."
        )
        self._persist_tree_state()

    def _better(self, left: float, right: float) -> bool:
        return left < right if self.metric_direction == "minimize" else left > right

    def _improved(self, candidate: float, parent: float) -> bool:
        tolerance = max(1e-9, abs(parent) * 1e-6)
        return (
            candidate < parent - tolerance
            if self.metric_direction == "minimize"
            else candidate > parent + tolerance
        )

    def _score_to_reward(self, score: float) -> float:
        oriented = -float(score) if self.metric_direction == "minimize" else float(score)
        return oriented / (1.0 + abs(oriented))

    def _can_attempt(self, operator: str | None = None) -> bool:
        """Return whether an action can run under its own accounting rule."""
        if self.implementation_attempts >= self.attempt_limit:
            return False
        if operator == "tune":
            return self.tuning_attempts < self.tuning_attempt_limit
        return self.experiments_executed < self.total_budget

    def _can_continue_search(self) -> bool:
        """Continue for remaining ideas or already-queued free tuning work."""
        if self.implementation_attempts >= self.attempt_limit:
            return False
        if self.experiments_executed < self.total_budget:
            return True
        if not self._can_attempt("tune"):
            return False
        return any(
            node.node_type == "planning"
            and node.operator == "tune"
            and not node.executed
            for node in self.all_nodes.values()
        )

    def _high_performer(self, score: float) -> bool:
        if self.best_node_id is None:
            return True
        best = float(self.all_nodes[self.best_node_id].result["score"])
        band = max(0.01, abs(best) * 0.05)
        return (
            score <= best + band
            if self.metric_direction == "minimize"
            else score >= best - band
        )

    def _successful_nodes(self) -> list[NodeState]:
        nodes = []
        for node in self.all_nodes.values():
            result = node.result or {}
            score_value = result.get("score")
            if score_value is None:
                continue
            try:
                score = float(score_value)
            except (TypeError, ValueError):
                continue
            if (
                node.node_type == "implementation"
                and result.get("status") == "completed"
                and not result.get("pruned")
                and math.isfinite(score)
            ):
                nodes.append(node)
        return sorted(
            nodes,
            key=lambda node: float(node.result["score"]),
            reverse=self.metric_direction != "minimize",
        )

    def _architecture_coverage(self) -> dict[str, Any]:
        """Summarize architectures actually measured by completed implementations."""
        measured: list[tuple[NodeState, str]] = []
        dedicated_attempted = False
        for node in self.all_nodes.values():
            if node.node_type != "implementation" or not node.executed:
                continue
            dedicated_attempted = dedicated_attempted or node.operator == "architect"
            result = node.result or {}
            if result.get("status") != "completed" or result.get("score") is None:
                continue
            code_track = classify_architecture(node.code or "")
            track = (
                code_track
                if code_track != "other"
                else classify_architecture(node.plan or "")
            )
            measured.append((node, track))
        coverage = coverage_from_tracks(track for _, track in measured)
        coverage["dedicated_architect_attempted"] = dedicated_attempted
        coverage["measured_nodes"] = [
            {
                "node_id": node.node_id,
                "operator": node.operator,
                "track": track,
                "score": (node.result or {}).get("score"),
            }
            for (node, _), track in zip(measured, coverage["tracks"])
        ]
        return coverage

    def _plateau_state(self) -> dict[str, Any]:
        """Detect saturation using material score gains, not exact float equality."""
        measured: list[tuple[str, float]] = []
        for node in self.all_nodes.values():
            result = node.result or {}
            score_value = result.get("score")
            if score_value is None:
                continue
            try:
                score = float(score_value)
            except (TypeError, ValueError):
                continue
            if (
                node.node_type == "implementation"
                and result.get("status") == "completed"
                and math.isfinite(score)
            ):
                measured.append((node.node_id, score))
        if not measured:
            return {
                "plateaued": False,
                "completed_measurements": 0,
                "stagnant_measurements": 0,
                "reason": "no completed measurements",
            }
        best = measured[0][1]
        last_material_index = 0
        material_improvements: list[dict[str, Any]] = []
        for index, (node_id, candidate) in enumerate(measured[1:], start=1):
            gain = best - candidate if self.metric_direction == "minimize" else candidate - best
            threshold = max(1e-9, abs(best) * self.plateau_relative_gain)
            if gain > threshold:
                material_improvements.append(
                    {
                        "node_id": node_id,
                        "gain": gain,
                        "threshold": threshold,
                    }
                )
                best = candidate
                last_material_index = index
        stagnant = len(measured) - 1 - last_material_index
        plateaued = len(measured) >= 2 and stagnant >= self.plateau_patience
        return {
            "plateaued": plateaued,
            "completed_measurements": len(measured),
            "stagnant_measurements": stagnant,
            "patience": self.plateau_patience,
            "relative_gain_threshold": self.plateau_relative_gain,
            "best_material_score": best,
            "last_material_improvement_node": measured[last_material_index][0],
            "material_improvements": material_improvements[-6:],
            "reason": (
                "no material score gain within the configured patience"
                if plateaued
                else "material progress remains or more measurements are needed"
            ),
        }

    def _architecture_intervention_reason(
        self, experiments_remaining: int
    ) -> str | None:
        """Reserve one bounded custom-network experiment when coverage is missing."""
        if (
            not self.architecture_exploration_enabled
            or self.total_budget < self.architecture_min_budget
            or experiments_remaining <= 0
            or self.best_node_id is None
        ):
            return None
        coverage = self._architecture_coverage()
        if coverage.get("custom_neural_attempted") or coverage.get(
            "dedicated_architect_attempted"
        ):
            return None
        plateau = self._plateau_state()
        reserve_due = experiments_remaining <= min(2, max(1, self.total_budget // 3))
        if plateau.get("plateaued"):
            if coverage.get("neural_attempted"):
                return (
                    "score progress plateaued after an established neural model, but no "
                    "task-invented neural computation graph has been measured"
                )
            return (
                "score progress plateaued while measured coverage remained limited to "
                "conventional or non-neural model families"
            )
        if reserve_due and not coverage.get("neural_attempted"):
            return (
                "the remaining idea budget reached its architecture-reserve boundary "
                "without any measured neural representation"
            )
        return None

    def _branch_root_id(self, node: NodeState) -> str:
        current = node
        seen: set[str] = set()
        while (
            current.parent_id
            and current.parent_id != "root"
            and current.parent_id in self.all_nodes
            and current.node_id not in seen
        ):
            seen.add(current.node_id)
            current = self.all_nodes[current.parent_id]
        return current.node_id

    def _create_planning_node(
        self,
        plan: str,
        *,
        operator: str,
        parent: NodeState | None = None,
        executed: bool = True,
        priority: float = 0.0,
        base: NodeState | None = None,
        companion: NodeState | None = None,
        merge_sources: list[str] | None = None,
    ) -> NodeState:
        """Create the visible planning step that precedes an implementation."""
        node_id = self._new_node_id(operator)
        config = {
            "priority": float(priority),
            "base_node_id": base.node_id if base else None,
            "companion_node_id": companion.node_id if companion else None,
            "materialized": bool(executed),
            "tune_depth": int((base.config or {}).get("tune_depth", 0)) if base else 0,
            "refine_depth": int((base.config or {}).get("refine_depth", 0)) if base else 0,
            "diversify_depth": int((base.config or {}).get("diversify_depth", 0)) if base else 0,
            "architecture_track": classify_architecture(plan),
            **self._council_node_config(plan),
        }
        node = NodeState(
            node_id=node_id,
            parent_id=parent.node_id if parent else None,
            node_type="planning",
            plan=plan,
            operator=operator,
            executed=executed,
            config=config,
            result={
                "status": "completed" if executed else "pending",
                "planning_only": True,
                **({"merge_sources": merge_sources} if merge_sources else {}),
            },
        )
        self.all_nodes[node_id] = node
        if parent is not None:
            parent.children_ids.append(node_id)
        summary = " ".join(plan.split())[:240]
        verb = "Created" if executed else "Queued"
        self._log(f"{verb} {self.node_label(node_id)} planning node ({operator}): {summary}")
        self._persist_tree_state()
        return node

    def _execute(
        self,
        plan: str,
        *,
        operator: str,
        parent: NodeState | None = None,
        base: NodeState | None = None,
        companion: NodeState | None = None,
    ) -> NodeState:
        """Execute one scientific experiment with failure isolation."""
        node_id = self._new_node_id(operator)
        base_config = dict(base.config or {}) if base else {}
        config = {
            "base_node_id": base.node_id if base else None,
            "companion_node_id": companion.node_id if companion else None,
            "tune_depth": int(base_config.get("tune_depth", 0)) + (operator == "tune"),
            "refine_depth": int(base_config.get("refine_depth", 0)) + (operator == "refine"),
            "diversify_depth": int(base_config.get("diversify_depth", 0)) + (operator == "diversify"),
            "architecture_track": classify_architecture(plan),
            **self._council_node_config(plan),
        }
        if base_config.get("hypothesis_id") and not config.get("hypothesis_id"):
            config["hypothesis_id"] = base_config["hypothesis_id"]
        node = NodeState(
            node_id=node_id,
            parent_id=parent.node_id if parent else None,
            node_type="implementation",
            plan=plan,
            operator=operator,
            config=config,
        )
        self.all_nodes[node_id] = node
        if parent is not None:
            parent.children_ids.append(node_id)

        self.implementation_attempts += 1
        if operator == "tune":
            self.tuning_attempts += 1
        display_name = self.node_label(node_id)
        accounting = (
            f"free tuning {self.tuning_attempts}/{self.tuning_attempt_limit}"
            if operator == "tune"
            else f"idea budget {self.experiments_executed}/{self.total_budget}"
        )
        self._log(
            f"Starting {operator} {display_name}; attempt {self.implementation_attempts}/"
            f"{self.attempt_limit}, {accounting}."
        )
        try:
            result = self.implementation_agent.run(
                self.run_root / node_id,
                plan,
                self.task_dir,
                self.task_analysis,
                parent_code=base.code if base else None,
                companion_code=companion.code if companion else None,
                operator=operator,
                node_label=display_name,
                max_debug_attempts=5,
                council_brief=self.council_brief,
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "score": None,
                "metric": self.metric_name,
                "direction": self.metric_direction,
                "output": None,
                "diagnostics": f"Unhandled node error: {type(exc).__name__}: {exc}",
                "attempts": 1,
                "operator": operator,
                "code": base.code if base else "",
            }
            self._log(f"{display_name} isolated an unhandled error: {exc}")

        node.executed = True
        node.code = str(result.pop("code", ""))
        node.result = result
        completed = result.get("status") == "completed" and result.get("score") is not None
        if completed:
            try:
                score = float(result["score"])
            except (TypeError, ValueError):
                score = math.nan
            if not math.isfinite(score):
                result.update(
                    status="failed",
                    score=None,
                    diagnostics=(
                        str(result.get("diagnostics", ""))
                        + "\nGenerated implementation returned a non-finite score."
                    ).strip(),
                )
                completed = False

        if completed:
            score = float(result["score"])
            reward = self._score_to_reward(score)
            result["reward"] = reward
            budget_charged = operator != "tune"
            result["budget_charged"] = budget_charged
            self.completed_implementations += 1
            if budget_charged:
                self.experiments_executed += 1
            self.scheduler.current_step = self.completed_implementations
            self.scheduler.backpropagate(node_id, reward, self.all_nodes)
            previous_best = self.best_node_id
            if self.best_node_id is None:
                self.best_node_id = node_id
            else:
                best_score = float(self.all_nodes[self.best_node_id].result["score"])
                if self._better(score, best_score):
                    self.best_node_id = node_id
            self._log(
                f"Completed {display_name}: score={score:.8g}; "
                f"idea budget={self.experiments_executed}/{self.total_budget}; "
                f"budget charged={'yes' if budget_charged else 'no (tuning)'}; "
                f"output={result.get('output')}."
            )
            if self.best_node_id != previous_best:
                self._log(f"New measured baseline: {display_name} with score={score:.8g}.")
        else:
            result["budget_charged"] = False
            diagnostic = " ".join(str(result.get("diagnostics", "unknown error")).split())[-500:]
            self.scheduler.backpropagate(node_id, -1.0, self.all_nodes)
            self._log(f"{display_name} failed; the idea budget is preserved: {diagnostic}")
        self._persist_tree_state()
        return node

    def _mark_pruned(self, node: NodeState, reason: str) -> None:
        node.result = dict(node.result or {})
        node.result["pruned"] = True
        node.result["pruned_reason"] = reason
        node.config = dict(node.config or {})
        node.config["pruned_reason"] = reason
        self._log(f"Pruned {self.node_label(node.node_id)}: {reason}")

    def _assess_child(self, child: NodeState, base: NodeState | None) -> bool:
        """Return whether a completed child deserves descendants."""
        if not child.result or child.result.get("status") != "completed":
            return False
        if base is None or not base.result or base.result.get("score") is None:
            return True
        child_score = float(child.result["score"])
        base_score = float(base.result["score"])
        if not self._improved(child_score, base_score):
            tune_depth = int((child.config or {}).get("tune_depth", 0))
            if (
                child.operator != "tune"
                and tune_depth < self.max_tune_depth
                and self._can_attempt("tune")
            ):
                child.result["awaiting_rescue_tune"] = True
                self._log(
                    f"{self.node_label(child.node_id)} underperformed "
                    f"{self.node_label(base.node_id)} ({child_score:.8g} vs "
                    f"{base_score:.8g}); queuing one focused rescue tune before pruning."
                )
                self._create_planning_node(
                    f"Rescue-tune {self.node_label(child.node_id)} before deciding whether "
                    "to prune this measured branch.",
                    operator="tune",
                    parent=child,
                    executed=False,
                    priority=0.9,
                    base=child,
                )
                return False
            self._mark_pruned(
                child,
                f"score {child_score:.8g} did not improve parent {base_score:.8g}",
            )
            if child.operator == "tune" and (
                bool(base.result.get("awaiting_rescue_tune"))
                or (
                    base.node_id != self.best_node_id
                    and not self._high_performer(base_score)
                )
            ):
                self._mark_pruned(
                    base,
                    "focused rescue tuning did not make this branch competitive",
                )
            return False
        if child.operator == "tune" and base.result.get("awaiting_rescue_tune"):
            base.result.pop("awaiting_rescue_tune", None)
            self._mark_pruned(
                base,
                f"superseded by tuned descendant {self.node_label(child.node_id)}",
            )
        if not self._high_performer(child_score):
            self._mark_pruned(
                child,
                "the improvement remained outside the competitive band of the active baseline",
            )
            if (
                child.operator == "tune"
                and base.node_id != self.best_node_id
                and not self._high_performer(base_score)
                and not base.result.get("pruned")
            ):
                self._mark_pruned(
                    base,
                    "focused rescue tuning did not make this branch competitive",
                )
            return False
        self._log(
            f"{self.node_label(child.node_id)} improved {self.node_label(base.node_id)} "
            f"from {base_score:.8g} to {child_score:.8g}."
        )
        return True

    def _pending_action(self, base: NodeState, operator: str) -> NodeState | None:
        for child_id in base.children_ids:
            child = self.all_nodes.get(child_id)
            if (
                child is not None
                and child.node_type == "planning"
                and child.operator == operator
                and not child.executed
            ):
                return child
        return None

    def _spawn_follow_up_nodes(self, node: NodeState) -> None:
        """Create lazy, score-prioritized actions around a measured implementation."""
        if (
            node.node_id in self._expanded_nodes
            or not node.result
            or node.result.get("status") != "completed"
            or node.result.get("pruned")
        ):
            return
        self._expanded_nodes.add(node.node_id)
        high = self._high_performer(float(node.result["score"]))
        tune_depth = int((node.config or {}).get("tune_depth", 0))
        if not high:
            actions = [("tune", 0.75)] if tune_depth < self.max_tune_depth else []
        elif node.operator == "architect":
            actions = [("tune", 0.62), ("refine", 0.3), ("diversify", 0.1)]
        elif node.operator == "refine":
            actions = [("tune", 0.52), ("refine", 0.28), ("diversify", 0.14)]
        elif node.operator == "tune":
            actions = [("refine", 0.45), ("tune", 0.22), ("diversify", 0.16)]
        else:
            actions = [("refine", 0.46), ("tune", 0.42), ("diversify", 0.18)]

        for operator, priority in actions:
            if operator == "tune" and tune_depth >= self.max_tune_depth:
                continue
            if not self._can_attempt(operator):
                continue
            self._create_planning_node(
                f"Lazy {operator} of {self.node_label(node.node_id)}; materialized only if selected.",
                operator=operator,
                parent=node,
                executed=False,
                priority=priority,
                base=node,
            )

    def _prune_stale_frontier(self) -> None:
        """Drop non-tuning actions from lineages displaced by a stronger baseline."""
        def has_improving_descendant(base: NodeState) -> bool:
            queue = list(base.children_ids)
            seen: set[str] = set()
            base_score = float(base.result["score"])
            while queue:
                node_id = queue.pop(0)
                if node_id in seen:
                    continue
                seen.add(node_id)
                descendant = self.all_nodes.get(node_id)
                if descendant is None:
                    continue
                result = descendant.result or {}
                if (
                    descendant.node_type == "implementation"
                    and result.get("status") == "completed"
                    and result.get("score") is not None
                    and not result.get("pruned")
                    and self._improved(float(result["score"]), base_score)
                ):
                    return True
                queue.extend(descendant.children_ids)
            return False

        for node in self.all_nodes.values():
            if node.node_type != "planning" or node.executed:
                continue
            base = self.all_nodes.get(str((node.config or {}).get("base_node_id")))
            if base is None or not base.result or base.result.get("score") is None:
                continue
            weak = bool(base.result.get("pruned")) or not self._high_performer(
                float(base.result["score"])
            )
            superseded = weak and has_improving_descendant(base)
            if weak and (node.operator != "tune" or superseded):
                node.executed = True
                node.result = {
                    "status": "pruned",
                    "planning_only": True,
                    "pruned": True,
                    "pruned_reason": (
                        "an improving descendant superseded this action"
                        if superseded
                        else "a stronger measured baseline displaced this lineage"
                    ),
                }
                node.config["pruned_reason"] = node.result["pruned_reason"]
            elif weak and node.operator == "tune":
                node.config["priority"] = max(
                    0.75, float(node.config.get("priority", 0.0) or 0.0)
                )

    def _execute_planning_action(self, planning: NodeState) -> NodeState | None:
        if not self._can_attempt(planning.operator) or planning.executed:
            return None
        base = self.all_nodes.get(str((planning.config or {}).get("base_node_id")))
        companion = self.all_nodes.get(
            str((planning.config or {}).get("companion_node_id"))
        )
        if base is None or not base.result or base.result.get("status") != "completed":
            planning.executed = True
            planning.result = {
                "status": "pruned",
                "planning_only": True,
                "pruned": True,
                "pruned_reason": "the measured parent is unavailable",
            }
            self._persist_tree_state()
            return None

        score = float(base.result["score"])
        try:
            if planning.operator == "tune":
                self._log(f"Invoking tuner for {self.node_label(base.node_id)} at score={score:.8g}.")
                plan = self.technique_agent.propose_tuning(
                    self.task_analysis,
                    base.plan or "",
                    score,
                    str(base.result.get("diagnostics", "")),
                )
            elif planning.operator == "architect":
                measured_context = "\n".join(
                    f"- {self.node_label(node.node_id)}: operator={node.operator}; "
                    f"architecture_track={(node.config or {}).get('architecture_track')}; "
                    f"score={(node.result or {}).get('score')}; "
                    f"plan_summary={self._clean_plan_text(node.plan)[:350]}"
                    for node in self.all_nodes.values()
                    if node.node_type == "implementation"
                    and node.result
                    and node.result.get("status") == "completed"
                )
                plan = self.technique_agent.propose_architecture_exploration(
                    self.task_analysis,
                    base.plan or "",
                    score,
                    measured_alternatives=measured_context,
                    plateau_evidence=json.dumps(self._plateau_state(), default=str),
                    require_custom=True,
                )
            else:
                measured_context = "\n".join(
                    f"- {self.node_label(node.node_id)}: operator={node.operator}; "
                    f"score={(node.result or {}).get('score')}; "
                    f"pruned={bool((node.result or {}).get('pruned'))}; "
                    f"plan_summary={self._clean_plan_text(node.plan)[:300]}"
                    for node in list(self.all_nodes.values())[-16:]
                    if node.node_type == "implementation"
                    and node.result
                    and node.result.get("status") == "completed"
                )
                plan = self.technique_agent.propose_follow_up(
                    self.task_analysis,
                    planning.operator or "refine",
                    base.plan or "",
                    score,
                    str(base.result.get("diagnostics", "")),
                    measured_context,
                )
        except Exception as exc:
            self._log(
                f"Planning service failed for {self.node_label(planning.node_id)}; "
                f"using a parent-preserving fallback: {exc}"
            )
            plan = (
                f"Preserve the complete working implementation of {self.node_label(base.node_id)} "
                f"as a control. Apply one bounded {planning.operator or 'refine'} change, use "
                "the identical local split and score, and retain the parent whenever the "
                "candidate does not improve."
            )
        planning.plan = plan
        planning.executed = True
        planning.config["materialized"] = True
        planning.config["architecture_track"] = classify_architecture(plan)
        planning.result = {"status": "completed", "planning_only": True}
        self._log(
            f"Materialized {self.node_label(planning.node_id)} ({planning.operator}) "
            f"from {self.node_label(base.node_id)}."
        )
        self._persist_tree_state()
        child = self._execute(
            plan,
            operator=planning.operator or "refine",
            parent=planning,
            base=base,
            companion=companion,
        )
        if child.result is not None and companion is not None:
            child.result["merged_with"] = companion.node_id
        if self._assess_child(child, base):
            self._spawn_follow_up_nodes(child)
        self._persist_tree_state()
        return child

    def _tune(self, node: NodeState) -> NodeState | None:
        if not self._can_attempt("tune") or not node.result:
            return None
        planning = self._pending_action(node, "tune")
        if planning is None:
            if int((node.config or {}).get("tune_depth", 0)) >= self.max_tune_depth:
                return None
            planning = self._create_planning_node(
                f"Lazy tune of {self.node_label(node.node_id)}; materialized only if selected.",
                operator="tune",
                parent=node,
                executed=False,
                priority=0.75,
                base=node,
            )
        return self._execute_planning_action(planning)

    def _architect(self, node: NodeState, *, trigger: str) -> NodeState | None:
        """Measure one custom neural counterfactual against the selected control."""
        if not self._can_attempt("architect") or not node.result:
            return None
        planning = self._pending_action(node, "architect")
        if planning is None:
            planning = self._create_planning_node(
                "Design a bounded task-invented neural architecture from observed evidence; "
                "materialize only when selected by the architecture coverage policy.",
                operator="architect",
                parent=node,
                executed=False,
                priority=1.25,
                base=node,
            )
        planning.config["architecture_trigger"] = trigger[:1000]
        planning.config["architecture_track"] = "custom_neural"
        return self._execute_planning_action(planning)

    def _merge_two_nodes(self, first: NodeState, second: NodeState, custom_plan: str = "") -> NodeState | None:
        max_merges = max(2, self.total_budget // 2)
        if not self._can_attempt("merge") or self._merge_attempts >= max_merges:
            return None
        pair = frozenset((first.node_id, second.node_id))
        if pair in self._merged_pairs:
            return None
        self._merged_pairs.add(pair)
        self._merge_attempts += 1
        self._log(
            f"Merging high-performing {self.node_label(first.node_id)} ({first.result.get('score')}) "
            f"and {self.node_label(second.node_id)} ({second.result.get('score')})."
        )
        if custom_plan and len(custom_plan.split()) >= 8:
            plan = custom_plan
        else:
            try:
                plan = self.technique_agent.propose_merge(
                    self.task_analysis,
                    first.plan or "",
                    second.plan or "",
                )
            except Exception as exc:
                self._log(f"Merge planning failed; using blend fallback: {exc}")
                plan = (
                    "Combine the two measured implementations with a validation-selected blend "
                    "or consensus, using the stronger parent unchanged whenever the merge does "
                    "not improve the shared local score."
                )
            plan += (
                f"\nParent A local score: {first.result.get('score')}. "
                f"Parent B local score: {second.result.get('score')}. Preserve or beat the stronger parent."
            )
        planning = self._create_planning_node(
            plan,
            operator="merge",
            parent=first,
            executed=True,
            priority=0.5,
            base=first,
            companion=second,
            merge_sources=[first.node_id, second.node_id],
        )
        merged = self._execute(
            plan,
            operator="merge",
            parent=planning,
            base=first,
            companion=second,
        )
        if merged.result is not None:
            merged.result["merged_with"] = second.node_id
        if self._assess_child(merged, first):
            self._spawn_follow_up_nodes(merged)
        self._persist_tree_state()
        return merged

    def _refine(self, node: NodeState, custom_plan: str = "") -> NodeState | None:
        if not self._can_attempt("refine") or not node.result:
            return None
        planning = self._pending_action(node, "refine")
        if planning is None:
            planning = self._create_planning_node(
                f"Refine of {self.node_label(node.node_id)}; materialized by decision.",
                operator="refine",
                parent=node,
                executed=False,
                priority=0.45,
                base=node,
            )
        if custom_plan:
            planning.plan = custom_plan
        return self._execute_planning_action(planning)

    def _merge_high_performers(self) -> NodeState | None:
        strongest_by_branch: dict[str, NodeState] = {}
        for node in self._successful_nodes():
            if self._high_performer(float(node.result["score"])):
                strongest_by_branch.setdefault(self._branch_root_id(node), node)
        candidates = list(strongest_by_branch.values())
        if len(candidates) < 2:
            return None
        return self._merge_two_nodes(candidates[0], candidates[1])

    def _run_root_plan(self, plan: str, *, replacement: bool = False) -> NodeState:
        planning = self._create_planning_node(
            plan,
            operator="root",
            parent=self.all_nodes["root"],
            executed=True,
            priority=0.0,
        )
        if replacement:
            planning.config["replacement"] = True
        root = self._execute(plan, operator="root", parent=planning)
        if root.result and root.result.get("status") == "completed":
            self._spawn_follow_up_nodes(root)
        return root

    def _run_recovery(self) -> NodeState | None:
        if not self._can_attempt("recovery"):
            return None
        recent_failures = "\n\n".join(
            str(node.result.get("diagnostics", ""))[-2500:]
            for node in list(self.all_nodes.values())[-6:]
            if node.result and node.result.get("status") == "failed"
        )
        plan = (
            "Write a conservative end-to-end implementation using direct reads of the exact "
            "inventory paths. Prefer the simplest broadly installed algorithm suitable for "
            "the observed values, include internal fallbacks, and always write the requested "
            "deliverable. Avoid the failed API patterns below.\n\n"
            f"Recent failure evidence:\n{recent_failures[-6000:]}"
        )
        planning = self._create_planning_node(
            plan,
            operator="recovery",
            parent=self.all_nodes["root"],
            executed=True,
            priority=0.8,
        )
        recovered = self._execute(plan, operator="recovery", parent=planning)
        if recovered.result and recovered.result.get("status") == "completed":
            self._spawn_follow_up_nodes(recovered)
        return recovered

    def run_tree_search(self) -> str | None:
        """Run council-directed planning and measured implementation search."""
        self._log("=" * 62)
        self._log(f"Starting adaptive method tree search for {self.task_name}.")
        self._log("=" * 62)

        self._prepare_research_council()

        if self.council_brief is not None:
            candidate_count = min(
                self.total_budget,
                max(
                    self.initial_fanout,
                    len(self.council_brief.selected_portfolio),
                ),
            )
        else:
            candidate_count = min(3, max(2, self.initial_fanout + 1))
        try:
            plans = self.technique_agent.generate_initial_approaches(
                self.task_analysis, candidate_count
            )
        except Exception as exc:
            self._log(f"Initial planning failed; using bounded direct root plans: {exc}")
            plans = [
                (
                    "Build a deterministic, resource-bounded end-to-end baseline from the exact "
                    "inventory paths, use an honest local score, and write the requested output."
                ),
                (
                    "Build a complementary dependency-light pipeline from the exact inventory, "
                    "using the same local score and output layout as the primary baseline."
                ),
                (
                    "Build a conservative recovery pipeline with broad input and library fallbacks "
                    "while preserving the shared score and requested output."
                ),
            ][:candidate_count]
        primary = list(plans[: self.initial_fanout])
        self._backup_plans = list(plans[self.initial_fanout :])
        self._log(
            f"Received {len(plans)} root plans: {len(primary)} primary and "
            f"{len(self._backup_plans)} recovery backups."
        )
        for index, plan in enumerate(plans, start=1):
            role = "primary" if index <= len(primary) else "backup"
            self._log(f"Plan {index} [{role}]: {' '.join(plan.split())[:240]}")

        root_nodes: list[NodeState] = []
        for plan in primary:
            candidate_plan: str | None = plan
            replacement = False
            while candidate_plan is not None and self._can_attempt("root"):
                root = self._run_root_plan(candidate_plan, replacement=replacement)
                if root.result and root.result.get("status") == "completed":
                    root_nodes.append(root)
                    break
                candidate_plan = self._backup_plans.pop(0) if self._backup_plans else None
                replacement = True
                if candidate_plan is not None:
                    self._log("Promoting a backup root because the prior implementation failed.")

        while self.best_node_id is None and self._can_attempt("recovery"):
            if self._backup_plans:
                root = self._run_root_plan(self._backup_plans.pop(0), replacement=True)
                if root.result and root.result.get("status") == "completed":
                    root_nodes.append(root)
            else:
                self._log("No runnable baseline yet; invoking failure-informed recovery.")
                self._run_recovery()

        # Give every displaced root exactly one model-locked rescue tune, then
        # prune it if the measured result remains weak.
        self._prune_stale_frontier()
        for root in root_nodes:
            if self._can_attempt("tune") and not self._high_performer(float(root.result["score"])):
                self._log(f"Running mandatory rescue tuning for weak {self.node_label(root.node_id)}.")
                self._tune(root)

        action_guard = self.attempt_limit * 4 + 16
        actions = 0
        while self._can_continue_search() and actions < action_guard:
            actions += 1
            self._prune_stale_frontier()

            # Collect executed node results for LLM evaluation
            nodes_history = []
            for nid, node in self.all_nodes.items():
                if nid == "root" or not node.executed:
                    continue
                res = node.result or {}
                nodes_history.append(
                    {
                        "node_id": nid,
                        "operator": node.operator or "root",
                        "node_type": node.node_type,
                        "status": res.get("status", "completed"),
                        "score": res.get("score"),
                        "architecture_track": (node.config or {}).get(
                            "architecture_track"
                        ),
                        "modality_ablation_scores": res.get(
                            "modality_ablation_scores"
                        ),
                        "plan_summary": self._clean_plan_text(node.plan)[:400],
                    }
                )

            best_score = (
                float(self.all_nodes[self.best_node_id].result["score"])
                if self.best_node_id
                and self.all_nodes[self.best_node_id].result
                and self.all_nodes[self.best_node_id].result.get("score") is not None
                else None
            )

            experiments_remaining = max(0, self.total_budget - self.experiments_executed)
            plateau_state = self._plateau_state()
            architecture_coverage = self._architecture_coverage()

            architecture_trigger = self._architecture_intervention_reason(
                experiments_remaining
            )
            if architecture_trigger and self.best_node_id is not None:
                self._log(
                    "Architecture coverage intervention before merge/finalize: "
                    f"{architecture_trigger}."
                )
                architecture_node = self._architect(
                    self.all_nodes[self.best_node_id], trigger=architecture_trigger
                )
                if architecture_node is not None:
                    continue

            # Analyze state with LLM Manager Agent and decide next action
            decision = self.technique_agent.decide_next_step(
                self.task_analysis,
                nodes_history=nodes_history,
                best_node_id=self.best_node_id,
                best_score=best_score,
                experiments_remaining=experiments_remaining,
                plateau_state=plateau_state,
                architecture_coverage=architecture_coverage,
            )

            action = decision.get("action", "diversify")
            target_ids = decision.get("target_node_ids", [])
            reasoning = decision.get("reasoning", "")
            custom_plan = decision.get("plan", "")

            self._log(
                f"ManagerAgent LLM Decision: action='{action}', targets={target_ids}. Reasoning: {reasoning}"
            )

            if action == "finalize":
                self._log("ManagerAgent LLM decided to finalize search.")
                break

            if action == "merge" and self._can_attempt("merge"):
                target_nodes = [
                    self.all_nodes[tid]
                    for tid in target_ids
                    if tid in self.all_nodes
                    and self.all_nodes[tid].executed
                    and self.all_nodes[tid].result
                    and self.all_nodes[tid].result.get("status") == "completed"
                ]
                if len(target_nodes) < 2:
                    succ = self._successful_nodes()
                    if len(succ) >= 2:
                        target_nodes = succ[:2]
                if len(target_nodes) >= 2:
                    merged = self._merge_two_nodes(target_nodes[0], target_nodes[1], custom_plan=custom_plan)
                    if merged is not None:
                        continue

            if (
                action == "architect"
                and self.best_node_id is not None
                and self._can_attempt("architect")
            ):
                architected = self._architect(
                    self.all_nodes[self.best_node_id],
                    trigger=str(reasoning or "LLM identified a missing architecture experiment"),
                )
                if architected is not None:
                    continue

            if action in {"tune", "refine"} and target_ids:
                target_id = target_ids[0]
                if target_id in self.all_nodes and self.all_nodes[target_id].executed:
                    target_node = self.all_nodes[target_id]
                    if action == "tune" and self._can_attempt("tune"):
                        res = self._tune(target_node)
                        if res is not None:
                            continue
                    elif action == "refine" and self._can_attempt("refine"):
                        res = self._refine(target_node, custom_plan=custom_plan)
                        if res is not None:
                            continue

            # Heuristic selection fallback via scheduler
            frontier_scores = self.scheduler.frontier_scores("root", self.all_nodes)
            eligible_ids = {
                node_id
                for node_id in frontier_scores
                if self._can_attempt(self.all_nodes[node_id].operator)
            }
            selected_id = self.scheduler.select_next_node(
                "root", self.all_nodes, eligible_node_ids=eligible_ids
            )
            if selected_id is None:
                if self.best_node_id is None and self._can_attempt("recovery"):
                    self._run_recovery()
                    continue
                self._log(
                    "Eligible frontier exhausted; stopping without expanding weak branches "
                    "or exceeding the new-idea budget."
                )
                break
            selected = self.all_nodes[selected_id]
            self._log(
                f"Selected {self.node_label(selected_id)} ({selected.operator}) from "
                f"{len(eligible_ids)} eligible pending actions; priority score="
                f"{frontier_scores[selected_id]:.5f}."
            )
            self._execute_planning_action(selected)

        if actions >= action_guard:
            self._log("Planning-action safety guard reached; finalized the strongest runnable node.")
        if self.best_node_id is None:
            self._log(
                f"Tree search ended without a runnable implementation after "
                f"{self.implementation_attempts} isolated attempts."
            )
        else:
            score = self.all_nodes[self.best_node_id].result["score"]
            self._log(
                f"Tree search finished: best={self.node_label(self.best_node_id)}; "
                f"{self.metric_name}={score}; idea budget={self.experiments_executed}/"
                f"{self.total_budget}; completed implementations="
                f"{self.completed_implementations}; free tuning attempts="
                f"{self.tuning_attempts}; total attempts={self.implementation_attempts}."
            )
        self._persist_tree_state()
        return self.best_node_id

    def generate_final_submission(self, best_node_id: str) -> bool:
        node = self.all_nodes.get(best_node_id)
        if node is None or not node.result or not node.result.get("output"):
            return False
        validation = self.submission_validator.validate(
            str(node.result["output"]),
            self.task_analysis,
            allowed_root=self.run_root / node.node_id,
        )
        node.result["final_submission_validation"] = validation.to_dict()
        if not validation.valid or validation.output_path is None:
            self._log(
                "Final output validation failed: "
                + "; ".join(validation.errors)
            )
            self._persist_tree_state()
            return False
        self.final_output_path = self.aggregator_agent.materialize(
            validation.output_path, self.run_root
        )
        self._log(f"Final output copied to {self.final_output_path}.")
        self._persist_tree_state()
        return True

    @staticmethod
    def _clean_plan_text(text: str | None) -> str:
        if not text:
            return ""
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            while line.startswith("#"):
                line = line.lstrip("#").strip()
            line = line.replace("**", "").replace("`", "")
            if line.startswith("- "):
                line = line[2:].strip()
            if line:
                lines.append(line)
        return " ".join(lines)

    def save_tree_image(self, output_path: Path | None = None) -> Path:
        """
        Generates and saves a large, fully readable image of the method exploration tree.
        Box sizes dynamically scale to fit their text content.
        """
        output = Path(output_path or (self.run_root / "method_tree.png"))
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            import os
            import tempfile
            import textwrap

            plot_cache = Path(tempfile.gettempdir()) / "aibuildai-matplotlib-cache"
            plot_cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
            os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache / "xdg"))
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            if not self.all_nodes:
                fig, ax = plt.subplots(figsize=(10, 3), dpi=140)
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"{self.task_name}\nNo implementation nodes were created",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                fig.savefig(output, bbox_inches="tight")
                plt.close(fig)
                self._log(f"Saved method tree image to {output}.")
                return output

            WRAP_WIDTH = 35          # characters per line before wrapping
            MAX_DESC_CHARS = 200     # truncate descriptions longer than this
            LINE_HEIGHT = 0.35       # vertical space per wrapped line (in data coords)
            BOX_WIDTH = 4.5          # fixed width for all boxes
            BOX_PAD_V = 0.5          # vertical padding inside box (title area + bottom)
            LEAF_H_SPACE = 5.5       # horizontal space allocated to each leaf node
            MIN_V_GAP = 3.5          # minimum vertical gap between depth levels
            TITLE_FONT = 10
            DESC_FONT = 9

            def get_node_display(node_id: str, node: NodeState) -> tuple[str, list[str], str, str, float]:
                if node_id == "root":
                    title = "Search Root"
                    desc = "Virtual orchestration node"
                    color, border = "#E0E0E0", "#616161"
                elif node.node_type in ("technique", "planning"):
                    node_kind = "Technique" if node.node_type == "technique" else "Planning"
                    title = f"{self.node_label(node_id)} ({node_kind})"
                    if not node.executed:
                        desc = f"PENDING — not executed within budget\n{self._clean_plan_text(node.plan)}"
                        color, border = "#FFF8E1", "#F9A825"
                    else:
                        tech_record = (node.config or {}).get("technique_record", {})
                        tech_status = tech_record.get("status", "completed")
                        artifact_id = tech_record.get("artifact_id")
                        if tech_status == "pool_hit":
                            desc = f"Pool hit: {artifact_id}"
                        elif tech_status == "pool_added":
                            desc = f"Web artifact added to pool: {artifact_id}"
                        elif tech_status == "bootstrap_failed":
                            candidate = tech_record.get("candidate_artifact", {}).get("artifact_id")
                            desc = f"Candidate failed verification: {candidate}\nFallback plan retained"
                        else:
                            desc = self._clean_plan_text(node.plan or tech_record.get("plan", f"{node_kind} completed"))
                        color, border = "#E3F2FD", "#1565C0"
                else:  # implementation
                    res = node.result or {}
                    score = res.get("score")
                    status = str(res.get("status") or ("completed" if node.executed else "pending"))
                    title = f"{self.node_label(node_id)} (Implementation)"
                    if not node.executed or status == "pending":
                        desc = "PENDING — not executed within budget"
                        color, border = "#FFF8E1", "#F9A825"
                    elif status == "failed" or score is None:
                        desc = "FAILED / Crashed"
                        color, border = "#FFEBEE", "#C62828"
                    else:
                        tech_record = node.config.get("technique_record", {}) if node.config else {}
                        artifact_id = tech_record.get("artifact_id")
                        op = node.operator or 'root'
                        fid = getattr(node, 'fidelity', None)
                        op_fid = f"{op} / {fid}" if fid else op
                        if artifact_id:
                            desc = f"{op_fid}\nUse: {artifact_id}\nScore: {float(score):.5f}"
                        elif tech_record.get("status") == "bootstrap_failed":
                            desc = f"{op_fid}\nUse: Self-contained fallback\nScore: {float(score):.5f}"
                        else:
                            desc = f"{op_fid}\nScore: {float(score):.5f}"
                        color, border = "#E8F5E9", "#2E7D32"

                linewidth = 2.0
                if node_id == self.best_node_id:
                    title = "★ " + title
                    border = "#1B5E20"
                    linewidth = 3.0

                if len(desc) > MAX_DESC_CHARS:
                    desc = desc[:MAX_DESC_CHARS] + "..."

                desc_lines = desc.split("\n")
                wrapped_lines: list[str] = []
                for d_line in desc_lines:
                    wrapped = textwrap.wrap(d_line, width=WRAP_WIDTH)
                    if wrapped:
                        wrapped_lines.extend(wrapped)
                    else:
                        wrapped_lines.append("")
                return title, wrapped_lines, color, border, linewidth

            node_heights: dict[str, float] = {}
            for nid, node in self.all_nodes.items():
                _, wrapped_lines, _, _, _ = get_node_display(nid, node)
                node_heights[nid] = BOX_PAD_V + len(wrapped_lines) * LINE_HEIGHT

            # Node depth calculation
            node_depths: dict[str, int] = {}
            def get_depth(nid: str) -> int:
                if nid in node_depths:
                    return node_depths[nid]
                parent = self.all_nodes[nid].parent_id if nid in self.all_nodes else None
                if parent is None or parent not in self.all_nodes:
                    d = 0
                else:
                    d = get_depth(parent) + 1
                node_depths[nid] = d
                return d

            for nid in self.all_nodes:
                get_depth(nid)

            max_depth = max(node_depths.values()) if node_depths else 0
            depth_max_h: dict[int, float] = {}
            for nid, d in node_depths.items():
                depth_max_h[d] = max(depth_max_h.get(d, 0.0), node_heights[nid])

            depth_y: dict[int, float] = {0: 0.0}
            for d in range(1, max_depth + 1):
                gap = max(MIN_V_GAP, (depth_max_h[d - 1] / 2.0) + (depth_max_h[d] / 2.0) + 1.2)
                depth_y[d] = depth_y[d - 1] - gap

            def compute_layout(
                node_id: str,
                x_left: float = 0.0,
                visited: set[str] | None = None,
            ) -> tuple[dict[str, tuple[float, float]], float]:
                if visited is None:
                    visited = set()
                if node_id in visited:
                    y = depth_y[node_depths.get(node_id, 0)]
                    return {node_id: (x_left, y)}, LEAF_H_SPACE
                visited.add(node_id)

                node = self.all_nodes[node_id]
                valid_children = [cid for cid in node.children_ids if cid in self.all_nodes]
                y = depth_y[node_depths.get(node_id, 0)]

                if not valid_children:
                    return {node_id: (x_left, y)}, LEAF_H_SPACE

                coords: dict[str, tuple[float, float]] = {}
                current_x = x_left
                child_widths: list[float] = []
                for child_id in valid_children:
                    child_coords, child_width = compute_layout(child_id, current_x, visited)
                    coords.update(child_coords)
                    child_widths.append(child_width)
                    current_x += child_width

                child_xs = [coords[cid][0] for cid in valid_children]
                x = sum(child_xs) / len(child_xs)
                coords[node_id] = (x, y)
                return coords, sum(child_widths)

            root_id = "root"
            if root_id not in self.all_nodes:
                roots = [
                    nid for nid, node in self.all_nodes.items()
                    if node.parent_id is None
                ]
                if not roots:
                    raise ValueError("No root node found for tree visualization")
                root_id = roots[0]

            positions, _ = compute_layout(root_id)
            disconnected = [nid for nid in self.all_nodes if nid not in positions]
            for index, nid in enumerate(disconnected, start=1):
                positions[nid] = (index * LEAF_H_SPACE, MIN_V_GAP)

            xs = [pos[0] for pos in positions.values()]
            ys = [pos[1] for pos in positions.values()]
            x_span = max(xs) - min(xs) if xs else 0.0
            y_span = max(ys) - min(ys) if ys else 0.0

            fig_width = max(20.0, x_span + 8.0)
            fig_height = max(12.0, y_span + 6.0)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
            ax.axis("off")

            # 1. Draw edges
            for nid, (x, y) in positions.items():
                node = self.all_nodes[nid]
                h = node_heights[nid]
                for child_id in node.children_ids:
                    if child_id in positions:
                        cx, cy = positions[child_id]
                        ch = node_heights[child_id]
                        ax.plot(
                            [x, cx],
                            [y - h / 2.0, cy + ch / 2.0],
                            color="#9E9E9E",
                            linestyle="-",
                            linewidth=1.5,
                            zorder=1,
                        )
                merged_with = (node.result or {}).get("merged_with")
                if merged_with in positions:
                    source_x, source_y = positions[str(merged_with)]
                    ax.plot(
                        [source_x, x],
                        [source_y, y],
                        color="#6F42C1",
                        linewidth=1.4,
                        linestyle="--",
                        zorder=1,
                    )

            # 2. Draw nodes
            for nid, (x, y) in positions.items():
                node = self.all_nodes[nid]
                title, wrapped_lines, color, border, linewidth = get_node_display(nid, node)
                h = node_heights[nid]

                rect = patches.FancyBboxPatch(
                    (x - BOX_WIDTH / 2.0, y - h / 2.0),
                    BOX_WIDTH,
                    h,
                    boxstyle="round,pad=0.1",
                    linewidth=linewidth,
                    edgecolor=border,
                    facecolor=color,
                    zorder=2,
                )
                ax.add_patch(rect)

                title_y = y + h / 2.0 - 0.3
                ax.text(
                    x,
                    title_y,
                    title,
                    ha="center",
                    va="center",
                    fontsize=TITLE_FONT,
                    fontweight="bold",
                    color="#212121",
                    zorder=3,
                )

                for i, line in enumerate(wrapped_lines):
                    line_y = title_y - 0.35 - i * LINE_HEIGHT
                    ax.text(
                        x,
                        line_y,
                        line,
                        ha="center",
                        va="center",
                        fontsize=DESC_FONT,
                        color="#424242",
                        zorder=3,
                    )

            ax.set_xlim(min(xs) - BOX_WIDTH, max(xs) + BOX_WIDTH)
            max_h = max(node_heights.values(), default=1.0)
            ax.set_ylim(min(ys) - max_h - 1.0, max(ys) + max_h + 1.0)
            ax.set_title(
                f"Method Exploration Tree — {self.task_name}",
                fontsize=16,
                fontweight="bold",
                pad=25,
            )
            fig.tight_layout()
            fig.savefig(output, bbox_inches="tight")
            plt.close(fig)
            self._log(f"Saved method tree image to {output}.")
            return output
        except Exception as exc:
            try:
                from PIL import Image, ImageDraw

                rows = []
                for node in self.all_nodes.values():
                    result = node.result or {}
                    rows.append(
                        f"{self.node_label(node.node_id)} | {node.node_type}/{node.operator or 'root'} "
                        f"| {result.get('status', 'pending')} | score={result.get('score', '—')}"
                    )
                image_height = max(500, 100 + 30 * len(rows))
                fallback = Image.new("RGB", (1800, image_height), "white")
                drawing = ImageDraw.Draw(fallback)
                drawing.text(
                    (40, 30),
                    f"Method Exploration Tree — {self.task_name}\nRenderer fallback: {exc}",
                    fill="black",
                )
                for index, row in enumerate(rows):
                    drawing.text((60, 100 + index * 30), row[:220], fill="#303030")
                fallback.save(output, format="PNG")
            except Exception:
                import base64

                one_pixel_png = (
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
                    "/x8AAusB9Y9Z4m8AAAAASUVORK5CYII="
                )
                output.write_bytes(base64.b64decode(one_pixel_png))
            self._log(f"Tree renderer was unavailable ({exc}); wrote readable fallback to {output}.")
            return output

    def finalize_run_artifacts(self) -> dict[str, Path]:
        """Refresh requested run summaries regardless of search outcome."""
        tree_state = self._persist_tree_state()
        method_tree = self.save_tree_image(self.run_root / "method_tree.png")
        return {"tree_state": tree_state, "method_tree": method_tree}
