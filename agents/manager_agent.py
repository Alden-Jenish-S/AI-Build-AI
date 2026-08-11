"""Lean score-driven manager with tuning, pruning, and mid-search merging."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime_utils import (
    absolute_path_without_symlink_resolution,
    run_supervised_process,
    sanitized_subprocess_env,
    validate_path_component,
)
from search_evidence import (
    family_fingerprint as _family_fingerprint,
    pearson_correlation,
    relative_noise_floor,
    score_noise_estimate,
    signature_from_result,
)
from tree.node import NodeState
from tree.scheduler import UCB1Scheduler

from .aggregator_agent import AggregatorAgent
from .architecture_policy import classify_architecture, coverage_from_tracks
from .council import CouncilBrief, CouncilCoordinator
from .implementation_agent import ImplementationAgent
from .submission_validator import SubmissionValidator
from .task_analyzer import TaskAnalysis, TaskAnalyzer
from .technique_agent import TechniqueAgent
from .modality_policy import transfer_learning_applicable


class ManagerAgent:
    """Build only on runnable implementations and stop weak branches early.

    Class-level defaults keep partially-constructed instances (unit tests,
    partial state restore) functional; __init__ overrides them from the env.
    """

    improvement_noise_k = 0.35
    abort_iterative_enabled = True
    abort_margin_std = 2.0
    complementarity_weight = 0.4
    diversify_probe_enabled = True
    final_ensemble_enabled = True
    max_architect_iterations = 2
    probe_attempt_limit = 3
    probe_attempts = 0
    ensemble_attempts = 0
    council_brief = None

    def __init__(
        self,
        task_name: str,
        total_budget: int = 10,
        venv_path: str | None = None,
        model_name: str | None = None,
        resume: bool = False,
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
        self._resumed = bool(resume) and (self.run_root / "tree_state.json").is_file()
        selected_python = absolute_path_without_symlink_resolution(venv_path) if venv_path else Path(sys.executable)
        self.python = str(selected_python if selected_python.is_file() else Path(sys.executable))
        self.task_analyzer = TaskAnalyzer(model_name=model_name)
        self.task_analysis: TaskAnalysis = self.task_analyzer.analyze(self.task_dir)

        # Do not displace a usable prior run until the new task can at least be
        # inventoried successfully. On explicit resume the run is kept in place.
        if not self._resumed:
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
        self._dirty_node_ids: set[str] = set()
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
        # Cheap screening probes are also budget-free but bounded; they let the
        # search rank candidates before spending a full idea on them.
        self.probe_attempt_limit = max(
            3,
            int(
                self.total_budget
                * self._env_float("AIBUILDAI_PROBE_MULTIPLIER", 2.0)
            ),
        )
        self.probe_attempts = 0
        # Final OOF ensemble is a single budget-free finalization step.
        self.ensemble_attempts = 0
        # Variance-aware decision policies (mode-agnostic: fold scores, repeated
        # seeds, or a relative floor when neither exists).
        self.improvement_noise_k = max(
            0.0, self._env_float("AIBUILDAI_IMPROVEMENT_NOISE_K", 0.35)
        )
        self.abort_iterative_enabled = self._env_enabled(
            "AIBUILDAI_ABORT_ITERATIVE", default=True
        )
        self.abort_margin_std = max(
            0.5, self._env_float("AIBUILDAI_ABORT_MARGIN_STD", 2.0)
        )
        self.complementarity_weight = max(
            0.0, self._env_float("AIBUILDAI_COMPLEMENTARITY_WEIGHT", 0.4)
        )
        self.diversify_probe_enabled = self._env_enabled(
            "AIBUILDAI_DIVERSIFY_PROBE", default=True
        )
        self.final_ensemble_enabled = self._env_enabled(
            "AIBUILDAI_FINAL_ENSEMBLE", default=True
        )
        self.max_architect_iterations = max(
            1, self._env_int("AIBUILDAI_MAX_ARCHITECT_ITERATIONS", 2)
        )
        self._log(
            f"Prepared task '{self.task_name}' with {len(self.task_analysis.files)} files; "
            f"metric={self.metric_name} ({self.metric_direction}); budget={self.total_budget}."
        )
        if self._resumed:
            self._restore_tree_state()
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
        self._log(f"Archived the previous run as {destination.name}.")

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
                    "architecture_trigger", "probe", "family_fingerprint",
                    "architect_count",
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
        dirty = self._dirty_node_ids
        if dirty:
            for node_id in dirty:
                node = self.all_nodes.get(node_id)
                if node is not None and node_id != "root":
                    self._persist_node(node)
            self._dirty_node_ids.clear()
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
            "probe_attempts": self.probe_attempts,
            "probe_attempt_limit": self.probe_attempt_limit,
            "ensemble_attempts": self.ensemble_attempts,
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

    def _restore_tree_state(self) -> None:
        """Rebuild mutable state from the on-disk tree_state.json snapshot.

        Node code is reloaded from each node's `algorithm.py` (never from the
        snapshot), and rewards are re-propagated from stored scores so the
        scheduler lineage statistics match the measured history.
        """
        path = self.run_root / "tree_state.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._log(f"Could not resume from {path.name}: {exc}; starting fresh.")
            self._resumed = False
            return
        nodes = payload.get("nodes")
        if not isinstance(nodes, dict) or not nodes:
            self._log(f"No nodes in {path.name}; starting fresh.")
            self._resumed = False
            return

        restored: dict[str, NodeState] = {}
        for node_id, raw in nodes.items():
            if not isinstance(raw, dict):
                continue
            code: str | None = None
            if node_id != "root":
                source = self.run_root / node_id / "algorithm.py"
                if source.is_file():
                    try:
                        code = source.read_text(encoding="utf-8")
                    except OSError:
                        code = None
            restored[node_id] = NodeState(
                node_id=node_id,
                parent_id=raw.get("parent_id"),
                node_type=raw.get("node_type") or "implementation",
                plan=raw.get("plan"),
                code=code,
                result=raw.get("result"),
                executed=bool(raw.get("executed")),
                operator=raw.get("operator"),
                config=dict(raw.get("config") or {}),
                visits=int(raw.get("visits", 0)),
                total_reward=float(raw.get("total_reward", 0.0)),
                children_ids=list(raw.get("children_ids") or []),
            )
        restored.setdefault("root", self.all_nodes["root"])
        self.all_nodes = restored
        self._dirty_node_ids.update(
            node_id for node_id in restored if node_id != "root"
        )
        self.experiments_executed = max(0, int(payload.get("budget_used", 0)))
        self.completed_implementations = max(
            0, int(payload.get("completed_implementations", 0))
        )
        self.implementation_attempts = max(0, int(payload.get("implementation_attempts", 0)))
        self.tuning_attempts = max(0, int(payload.get("tuning_attempts", 0)))
        self.probe_attempts = max(0, int(payload.get("probe_attempts", 0)))
        self.ensemble_attempts = max(0, int(payload.get("ensemble_attempts", 0)))
        self._node_counter = max(
            (
                int(match.group(1))
                for node_id in restored
                if (match := re.fullmatch(r"node(\d+)", node_id))
            ),
            default=0,
        )
        self._expanded_nodes = {
            node_id for node_id, node in restored.items() if node.children_ids
        }
        merged_pairs: set[frozenset[str]] = set()
        merge_attempts = 0
        for node in restored.values():
            if node.operator == "merge" and node.executed:
                merge_attempts += 1
                base = (node.config or {}).get("base_node_id")
                companion = (node.config or {}).get("companion_node_id")
                if base and companion:
                    merged_pairs.add(frozenset({base, companion}))
        self._merged_pairs = merged_pairs
        self._merge_attempts = merge_attempts
        stored_best = payload.get("best_node_id")
        self.best_node_id = (
            stored_best if stored_best in restored else self._pick_best_from_restored()
        )
        final_output = payload.get("final_output")
        output_path = Path(final_output) if final_output else None
        self.final_output_path = output_path if output_path and output_path.is_file() else None

        # Re-propagate rewards so scheduler lineage statistics match history.
        self.scheduler = UCB1Scheduler(self.total_budget)
        for node in restored.values():
            if node.node_type != "implementation" or not node.executed:
                continue
            result = node.result or {}
            score_value = result.get("score")
            if result.get("status") == "completed" and score_value is not None:
                try:
                    reward = self._score_to_reward(float(score_value))
                except (TypeError, ValueError):
                    reward = -1.0
            elif result.get("status") == "truncated" and score_value is not None:
                reward = -0.2
            else:
                reward = -1.0
            if result.get("probe"):
                reward *= 0.5
            self.scheduler.backpropagate(node.node_id, reward, restored)
        self._resumed = True
        self._log(
            f"Resumed {len(restored)} nodes from {path.name}; "
            f"best={self.node_label(self.best_node_id)}; "
            f"budget={self.experiments_executed}/{self.total_budget}."
        )

    def _pick_best_from_restored(self) -> str | None:
        best_id: str | None = None
        best_score: float | None = None
        for node in self.all_nodes.values():
            result = node.result or {}
            if (
                node.node_type != "implementation"
                or result.get("status") != "completed"
                or result.get("score") is None
            ):
                continue
            try:
                score = float(result["score"])
            except (TypeError, ValueError):
                continue
            if best_score is None or self._better(score, best_score):
                best_score = score
                best_id = node.node_id
        return best_id

    def _new_node_id(self, operator: str) -> str:
        self._node_counter += 1
        return f"node{self._node_counter}"

    def _mark_dirty(self, *node_ids: str | None) -> None:
        """Schedule per-node state files for the next incremental persist."""
        self._dirty_node_ids.update(
            node_id for node_id in node_ids if node_id is not None
        )

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

    def _improved(self, candidate: float, parent: float, noise: float | None = None) -> bool:
        tolerance = max(1e-9, abs(parent) * 1e-6)
        if noise is not None and noise > 0.0:
            tolerance = max(tolerance, self.improvement_noise_k * noise)
        return (
            candidate < parent - tolerance
            if self.metric_direction == "minimize"
            else candidate > parent + tolerance
        )

    def _noise_for(self, node: NodeState | None) -> float:
        """Estimate the evaluation noise of a measured node.

        Fold dispersion, then repeated-seed dispersion, then a conservative
        relative floor for holdout/task-native tasks without either.
        """
        result = (node.result or {}) if node is not None else {}
        score_value = result.get("score")
        if score_value is None:
            return 0.0
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            return 0.0
        estimate = score_noise_estimate(result)
        if estimate is not None:
            return max(estimate, 1e-8)
        return relative_noise_floor(score)

    def _score_to_reward(self, score: float) -> float:
        oriented = -float(score) if self.metric_direction == "minimize" else float(score)
        return oriented / (1.0 + abs(oriented))

    def _can_attempt(self, operator: str | None = None) -> bool:
        """Return whether an action can run under its own accounting rule."""
        if self.implementation_attempts >= self.attempt_limit:
            return False
        if operator == "tune":
            return self.tuning_attempts < self.tuning_attempt_limit
        if operator == "probe":
            return self.probe_attempts < self.probe_attempt_limit
        if operator == "ensemble":
            return self.ensemble_attempts < 1
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
        best_node = self.all_nodes[self.best_node_id]
        best = float(best_node.result["score"])
        band = max(
            0.01,
            abs(best) * 0.05,
            2.0 * self._noise_for(best_node),
        )
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
                and not result.get("probe")
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
            result = node.result or {}
            if result.get("status") != "completed" or result.get("score") is None:
                continue
            if result.get("probe"):
                continue
            # A completed custom-network measurement counts as coverage; a failed
            # or invalidated architect attempt must not disable the intervention.
            if node.operator == "architect":
                dedicated_attempted = True
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
        # Noise-aware material-gain detection: an improvement must exceed both
        # the configured relative tolerance and the candidate's own evaluation
        # noise (fold/seed dispersion, or a relative floor), otherwise the
        # measurement is treated as saturation. This prevents the search from
        # chasing noise on every task type (RL, tiny-N gene data, holdouts...).
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
                and not result.get("probe")
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
            candidate_noise = self._noise_for(self.all_nodes.get(node_id))
            threshold = max(
                1e-9,
                abs(best) * self.plateau_relative_gain,
                self.improvement_noise_k * candidate_noise,
            )
            gain = best - candidate if self.metric_direction == "minimize" else candidate - best
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

    def _architecture_revision_reason(
        self, experiments_remaining: int
    ) -> str | None:
        """Trigger a residual-driven revision when a custom network already
        improved the control but overall progress plateaued again."""
        if (
            not self.architecture_exploration_enabled
            or self.total_budget < self.architecture_min_budget
            or experiments_remaining <= 0
            or self.best_node_id is None
        ):
            return None
        architecture_nodes = [
            node
            for node in self.all_nodes.values()
            if node.node_type == "implementation"
            and node.executed
            and node.operator == "architect"
            and (node.result or {}).get("status") == "completed"
            and not (node.result or {}).get("probe")
        ]
        if not architecture_nodes:
            return None
        if len(architecture_nodes) >= self.max_architect_iterations:
            return None
        last = max(
            architecture_nodes,
            key=lambda node: float((node.result or {}).get("score", float("-inf"))),
        )
        base = self.all_nodes.get(str((last.config or {}).get("base_node_id")))
        if (
            base is None
            or not base.result
            or base.result.get("score") is None
        ):
            return None
        if not self._improved(
            float(last.result["score"]),
            float(base.result["score"]),
            noise=self._noise_for(base),
        ):
            return None
        plateau = self._plateau_state()
        if not plateau.get("plateaued"):
            return None
        return (
            f"architecture revision {len(architecture_nodes)} of "
            f"{self.max_architect_iterations}: the custom network "
            f"{self.node_label(last.node_id)} improved its control, but measured "
            "progress plateaued again; run a residual-error-driven revision"
        )

    def _architect_residual_evidence(self) -> str:
        """Summarize the latest custom-network measurement for the next revision."""
        lines: list[str] = []
        for node in self.all_nodes.values():
            result = node.result or {}
            if (
                node.node_type != "implementation"
                or not node.executed
                or node.operator != "architect"
                or result.get("status") != "completed"
                or result.get("probe")
            ):
                continue
            base = self.all_nodes.get(str((node.config or {}).get("base_node_id")))
            base_score = (
                (base.result or {}).get("score") if base is not None else None
            )
            lines.append(
                f"- {self.node_label(node.node_id)}: score={result.get('score')} "
                f"(control {base.node_id if base else 'none'}="
                f"{base_score}); plan: {self._clean_plan_text(node.plan)[:400]}; "
                f"diagnostics tail: {str(result.get('diagnostics', ''))[-1200:]}"
            )
        return "\n".join(lines[-2:])

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
        self._mark_dirty(node_id, parent.node_id if parent else None)
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
        probe: bool = False,
    ) -> NodeState:
        """Execute one scientific experiment with failure isolation.

        ``probe`` marks a cheap screening run: it never consumes the idea
        budget, never updates the measured baseline, and its score is only
        used to decide whether to promote the plan to a full run.
        """
        node_id = self._new_node_id(operator)
        base_config = dict(base.config or {}) if base else {}
        config = {
            "base_node_id": base.node_id if base else None,
            "companion_node_id": companion.node_id if companion else None,
            "tune_depth": int(base_config.get("tune_depth", 0)) + (operator == "tune"),
            "refine_depth": int(base_config.get("refine_depth", 0)) + (operator == "refine"),
            "diversify_depth": int(base_config.get("diversify_depth", 0)) + (operator == "diversify"),
            "architecture_track": classify_architecture(plan),
            "architect_count": int(base_config.get("architect_count", 0))
            + (1 if operator == "architect" else 0),
            "family_fingerprint": _family_fingerprint(plan),
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
        self._mark_dirty(node_id, parent.node_id if parent else None)

        self.implementation_attempts += 1
        if operator == "tune":
            self.tuning_attempts += 1
        if operator == "probe":
            self.probe_attempts += 1
        if operator == "ensemble":
            self.ensemble_attempts += 1
        display_name = self.node_label(node_id)
        accounting = (
            f"free tuning {self.tuning_attempts}/{self.tuning_attempt_limit}"
            if operator == "tune"
            else f"free probe {self.probe_attempts}/{self.probe_attempt_limit}"
            if operator == "probe"
            else f"idea budget {self.experiments_executed}/{self.total_budget}"
        )
        self._log(
            f"Starting {operator} {display_name}; attempt {self.implementation_attempts}/"
            f"{self.attempt_limit}, {accounting}."
        )
        abort_context = None
        if (
            not probe
            and self.abort_iterative_enabled
            and self.best_node_id is not None
        ):
            best = self.all_nodes[self.best_node_id]
            if best.result and best.result.get("score") is not None:
                try:
                    best_score = float(best.result["score"])
                    noise = self._noise_for(best)
                    abort_context = {
                        "best_score": best_score,
                        "direction": self.metric_direction,
                        "margin": max(1e-9, noise * self.abort_margin_std),
                        "patience": 2,
                    }
                except (TypeError, ValueError):
                    abort_context = None
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
                max_debug_attempts=3 if probe else 5,
                council_brief=self.council_brief,
                probe=probe,
                abort_context=abort_context,
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

        truncated = result.get("status") == "truncated" and result.get("score") is not None
        if truncated:
            result["budget_charged"] = False
            result["reward"] = -0.2
            self.scheduler.backpropagate(node_id, -0.2, self.all_nodes)
            self._log(
                f"{display_name} stopped early (truncated) with score="
                f"{float(result['score']):.8g}; the idea budget is preserved."
            )
            self._persist_tree_state()
            return node

        if probe or operator == "ensemble":
            if completed:
                score = float(result["score"])
                reward = self._score_to_reward(score) * 0.5
                result["reward"] = reward
                result["probe"] = bool(probe)
                result["budget_charged"] = False
                self.scheduler.backpropagate(node_id, reward, self.all_nodes)
                self._log(
                    f"Completed {display_name}: {'probe' if probe else 'ensemble'} "
                    f"score={score:.8g}; no idea budget charged."
                )
            else:
                result["probe"] = bool(probe)
                result["budget_charged"] = False
                diagnostic = (
                    " ".join(str(result.get("diagnostics", "unknown error")).split())[-500:]
                )
                self.scheduler.backpropagate(node_id, -1.0, self.all_nodes)
                self._log(
                    f"{display_name} ({'probe' if probe else 'ensemble'}) failed; "
                    f"budget preserved: {diagnostic}"
                )
            self._persist_tree_state()
            return node

        if completed:
            score = float(result["score"])
            reward = self._score_to_reward(score)
            result["reward"] = reward
            budget_charged = operator not in {"tune", "probe", "ensemble"}
            result["budget_charged"] = budget_charged
            self.completed_implementations += 1
            if budget_charged:
                self.experiments_executed += 1
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
                f"budget charged={'yes' if budget_charged else 'no (tuning/probe)'}; "
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
        self._mark_dirty(node.node_id)
        self._log(f"Pruned {self.node_label(node.node_id)}: {reason}")

    def _assess_child(self, child: NodeState, base: NodeState | None) -> bool:
        """Return whether a completed child deserves descendants."""
        if not child.result or child.result.get("status") != "completed":
            return False
        if base is None or not base.result or base.result.get("score") is None:
            return True
        child_score = float(child.result["score"])
        base_score = float(base.result["score"])
        if not self._improved(
            child_score, base_score, noise=self._noise_for(base)
        ):
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
            if int((node.config or {}).get("architect_count", 0)) < self.max_architect_iterations:
                actions = [("architect", 0.5), *actions]
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
            base_noise = self._noise_for(base)
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
                    and not result.get("probe")
                    and self._improved(
                        float(result["score"]), base_score, noise=base_noise
                    )
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
                    and node.executed
                    and (node.result or {}).get("status") == "completed"
                    and not (node.result or {}).get("probe")
                )
                plan = self.technique_agent.propose_architecture_exploration(
                    self.task_analysis,
                    base.plan or "",
                    score,
                    measured_alternatives=measured_context,
                    plateau_evidence=json.dumps(self._plateau_state(), default=str),
                    require_custom=True,
                    residual_evidence=self._architect_residual_evidence(),
                )
            elif planning.operator == "transfer":
                measured_context = "\n".join(
                    f"- {self.node_label(node.node_id)}: operator={node.operator}; "
                    f"score={(node.result or {}).get('score')}; "
                    f"pruned={bool((node.result or {}).get('pruned'))}; "
                    f"plan_summary={self._clean_plan_text(node.plan)[:300]}"
                    for node in self.all_nodes.values()
                    if node.node_type == "implementation"
                    and node.executed
                    and (node.result or {}).get("status") == "completed"
                    and not (node.result or {}).get("probe")
                )
                plan = self.technique_agent.propose_transfer_exploration(
                    self.task_analysis,
                    base.plan or "",
                    score,
                    measured_alternatives=measured_context,
                    plateau_evidence=json.dumps(self._plateau_state(), default=str),
                    residual_evidence=self._architect_residual_evidence(),
                )
            else:
                measured_context = "\n".join(
                    f"- {self.node_label(node.node_id)}: operator={node.operator}; "
                    f"score={(node.result or {}).get('score')}; "
                    f"pruned={bool((node.result or {}).get('pruned'))}; "
                    f"plan_summary={self._clean_plan_text(node.plan)[:300]}"
                    for node in list(self.all_nodes.values())[-16:]
                    if node.node_type == "implementation"
                    and node.executed
                    and (node.result or {}).get("status") == "completed"
                    and not (node.result or {}).get("probe")
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

        if planning.operator == "diversify":
            guarded = self._diversify_gate(plan, base, planning)
            if guarded is None:
                planning.result = {
                    "status": "pruned",
                    "planning_only": True,
                    "pruned": True,
                    "pruned_reason": (
                        "diversify proposal duplicated a measured family or failed "
                        "its cheap screening probe"
                    ),
                }
                planning.config["pruned_reason"] = planning.result["pruned_reason"]
                self._log(
                    f"Diverted {self.node_label(planning.node_id)}: "
                    f"{planning.result['pruned_reason']}."
                )
                self._persist_tree_state()
                return None
            plan = guarded
            planning.plan = plan
            planning.config["architecture_track"] = classify_architecture(plan)

        planning.config["family_fingerprint"] = _family_fingerprint(plan)
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
        if not (child.result or {}).get("probe") and self._assess_child(child, base):
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

    def _architect(self, node: NodeState, *, trigger: str, operator: str = "architect") -> NodeState | None:
        """Measure one custom neural or transfer counterfactual against the selected control."""
        if not self._can_attempt(operator) or not node.result:
            return None
        planning = self._pending_action(node, operator)
        if planning is None:
            planning = self._create_planning_node(
                "Design a bounded task-invented neural architecture or pretrained transfer model from observed evidence; "
                "materialize only when selected by the architecture coverage policy.",
                operator=operator,
                parent=node,
                executed=False,
                priority=1.25,
                base=node,
            )
        planning.config["architecture_trigger"] = trigger[:1000]
        planning.config["architecture_track"] = "custom_neural" if operator == "architect" else "established_neural"
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
        pair = self._merge_pair(candidates)
        if pair is None:
            return None
        return self._merge_two_nodes(pair[0], pair[1])

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

    def _run_probe(
        self,
        plan: str,
        *,
        base: NodeState | None = None,
        parent: NodeState | None = None,
    ) -> NodeState | None:
        """Run one cheap screening implementation; never charges the idea budget."""
        if not self._can_attempt("probe"):
            return None
        planning = self._create_planning_node(
            plan,
            operator="probe",
            parent=(parent or self.all_nodes["root"]),
            executed=True,
            priority=0.0,
            base=base,
        )
        child = self._execute(
            plan,
            operator="probe",
            parent=planning,
            base=base,
            probe=True,
        )
        return child

    def _promote_probe(self, probe_node: NodeState, plan: str) -> NodeState:
        """Promote a passed cheap probe to a full budgeted implementation."""
        planning = self._create_planning_node(
            plan
            + "\nThis run replaces a successful cheap screening probe. Use the FULL "
            "dataset and full iterations/epochs for this implementation while "
            "preserving the same split, metric, paths, and output schema.",
            operator="root",
            parent=probe_node,
            executed=True,
            priority=0.0,
            base=probe_node,
        )
        root = self._execute(plan, operator="root", parent=planning, base=probe_node)
        if root.result and root.result.get("status") == "completed":
            self._spawn_follow_up_nodes(root)
        return root

    def _family_collisions(self, fingerprint: str) -> list[str]:
        """Return measured nodes sharing the same model-family fingerprint."""
        if not fingerprint:
            return []
        hits: list[str] = []
        for node in self.all_nodes.values():
            if node.node_type != "implementation" or not node.executed:
                continue
            result = node.result or {}
            if result.get("status") not in {"completed", "truncated"}:
                continue
            if result.get("probe"):
                continue
            node_fingerprint = (node.config or {}).get("family_fingerprint")
            if not node_fingerprint:
                node_fingerprint = _family_fingerprint(node.plan or "")
            if node_fingerprint and node_fingerprint == fingerprint:
                hits.append(self.node_label(node.node_id))
        return hits

    def _diversify_gate(
        self,
        plan: str,
        base: NodeState,
        planning: NodeState,
    ) -> str | None:
        """Enforce family diversity and cheap screening before a full run."""
        fingerprint = _family_fingerprint(plan)
        collisions = self._family_collisions(fingerprint)
        if collisions:
            guard_message = (
                "FAMILY GUARD (hard constraint): your proposal repeats an already "
                f"measured model-family fingerprint '{fingerprint}' seen in nodes "
                f"{', '.join(collisions[:6])}. Propose a genuinely different model family."
            )
            self._log(
                f"Diversify family collision detected; requesting a re-plan: "
                f"{guard_message[:300]}"
            )
            try:
                re_plan = self.technique_agent.propose_follow_up(
                    self.task_analysis,
                    "diversify",
                    base.plan or "",
                    float(base.result["score"]),
                    str(base.result.get("diagnostics", "")),
                    search_context=guard_message,
                    avoid_families=guard_message,
                )
            except Exception as exc:
                self._log(f"Diversify re-plan failed: {exc}")
                re_plan = ""
            if (
                re_plan
                and _family_fingerprint(re_plan)
                and not self._family_collisions(_family_fingerprint(re_plan))
            ):
                plan = re_plan
                fingerprint = _family_fingerprint(re_plan)
                self._log(
                    "Diversify re-plan passed the family guard "
                    f"(fingerprint '{fingerprint}')."
                )
            else:
                self._log(
                    "Diversify proposal could not escape the measured family set; "
                    "discarding the action."
                )
                return None
        if self.diversify_probe_enabled and self._can_attempt("probe"):
            probe_node = self._run_probe(plan, base=base, parent=planning)
            if (
                probe_node is None
                or not probe_node.result
                or probe_node.result.get("status") != "completed"
            ):
                self._log(
                    f"Diversify {self.node_label(planning.node_id)} failed its cheap "
                    "probe; skipping the full run."
                )
                return None
            probe_score = float(probe_node.result["score"])
            if not self._improved(
                probe_score, float(base.result["score"]), noise=self._noise_for(base)
            ):
                self._log(
                    f"Diversify probe score {probe_score:.8g} did not beat base "
                    f"{base.result.get('score')} within evaluation noise; skipping "
                    "the full run and saving compute."
                )
                return None
            self._log(
                f"Diversify probe {probe_score:.8g} cleared the base; running the "
                "full implementation."
            )
        return plan

    def _signature_provider(self, node: NodeState) -> list[float] | None:
        """Prediction signature of a pending action's measured parent."""
        base = self.all_nodes.get(str((node.config or {}).get("base_node_id")))
        if base is None:
            return None
        return signature_from_result(base.result)

    def _incumbent_signature(self) -> list[float] | None:
        if self.best_node_id is None:
            return None
        return signature_from_result(self.all_nodes[self.best_node_id].result)

    def _pair_correlation(self, first: NodeState, second: NodeState) -> float:
        first_signature = signature_from_result(first.result)
        second_signature = signature_from_result(second.result)
        if first_signature is None or second_signature is None:
            return 0.0
        return pearson_correlation(first_signature, second_signature)

    def _merge_pair(
        self, candidates: list[NodeState]
    ) -> tuple[NodeState, NodeState] | None:
        """Pick the pair with the best score and prediction-complementarity
        tradeoff; falls back to the top two scores when no signatures exist."""
        pool = list(candidates)
        if len(pool) < 2:
            return None
        pool.sort(
            key=lambda node: float(node.result["score"]),
            reverse=self.metric_direction != "minimize",
        )
        pool = pool[:8]
        best_key = float("-inf")
        best_pair: tuple[NodeState, NodeState] | None = None
        for index in range(len(pool)):
            for other in range(index + 1, len(pool)):
                first = pool[index]
                second = pool[other]
                correlation = self._pair_correlation(first, second)
                score_sum = float(first.result["score"]) + float(second.result["score"])
                key = score_sum * (
                    1.0 + self.complementarity_weight * (1.0 - abs(correlation))
                )
                if key > best_key:
                    best_key = key
                    best_pair = (first, second)
        return best_pair

    def _plan_initial_roots(self) -> None:
        """Probe a broad candidate portfolio, then promote the strongest
        measured plans to full implementations (successive halving)."""
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
        if self._can_attempt("probe"):
            self._log(
                f"Probe-first: cheap screening of {len(plans)} candidate root plans "
                "before spending full budget."
            )
            probed: list[tuple[NodeState, str]] = []
            for root_plan in plans:
                if not self._can_attempt("probe"):
                    break
                probe_node = self._run_probe(root_plan)
                if (
                    probe_node is not None
                    and probe_node.result
                    and probe_node.result.get("status") == "completed"
                ):
                    probed.append((probe_node, root_plan))
            if probed:
                probed.sort(
                    key=lambda pair: float(pair[0].result["score"]),
                    reverse=self.metric_direction != "minimize",
                )
                promote_count = min(self.initial_fanout, len(probed))
                self._log(
                    f"Probe screening produced {len(probed)} runnable candidates; "
                    f"promoting the top {promote_count} to full implementations."
                )
                for probe_node, candidate_plan in probed[:promote_count]:
                    if not self._can_attempt("root"):
                        break
                    root = self._promote_probe(probe_node, candidate_plan)
                    if root.result and root.result.get("status") == "completed":
                        root_nodes.append(root)
        if not root_nodes:
            # All probes failed (or probing was unavailable); fall back to the
            # direct full baseline path so a screening failure never blocks.
            self._log(
                "No probe passed; falling back to direct full root implementations."
            )
            for plan in primary:
                candidate_plan: str | None = plan
                replacement = False
                while candidate_plan is not None and self._can_attempt("root"):
                    root = self._run_root_plan(candidate_plan, replacement=replacement)
                    if root.result and root.result.get("status") == "completed":
                        root_nodes.append(root)
                        break
                    candidate_plan = (
                        self._backup_plans.pop(0) if self._backup_plans else None
                    )
                    replacement = True
                    if candidate_plan is not None:
                        self._log(
                            "Promoting a backup root because the prior implementation failed."
                        )

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

    def run_tree_search(self) -> str | None:
        """Run council-directed planning and measured implementation search."""
        self._log("=" * 62)
        self._log(f"Starting adaptive method tree search for {self.task_name}.")
        self._log("=" * 62)

        if self._resumed:
            # The council brief is not serialized for reload; re-running the
            # council on resume would duplicate costly research work. Existing
            # nodes already carry their protocol hashes in config.
            self._log(
                "Skipping initial root planning; executing the restored frontier instead."
            )
        else:
            self._prepare_research_council()
            self._plan_initial_roots()

        action_guard = self.attempt_limit * 4 + 16
        actions = 0
        while self._can_continue_search() and actions < action_guard:
            actions += 1
            self._prune_stale_frontier()

            # Collect executed node results for LLM evaluation (cheap screening
            # probes are internal evidence and are excluded from the LLM history).
            nodes_history = []
            for nid, node in self.all_nodes.items():
                if nid == "root" or not node.executed:
                    continue
                res = node.result or {}
                if res.get("probe"):
                    continue
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

            architecture_trigger = (
                self._architecture_intervention_reason(experiments_remaining)
                or self._architecture_revision_reason(experiments_remaining)
            )
            if architecture_trigger and self.best_node_id is not None:
                self._log(
                    "Architecture coverage intervention before merge/finalize: "
                    f"{architecture_trigger}."
                )
                op = "transfer" if transfer_learning_applicable(self.task_analysis.modalities) else "architect"
                architecture_node = self._architect(
                    self.all_nodes[self.best_node_id], trigger=architecture_trigger, operator=op
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
                    and not self.all_nodes[tid].result.get("probe")
                ]
                if len(target_nodes) < 2:
                    succ = self._successful_nodes()
                    pair = self._merge_pair(succ)
                    if pair is not None:
                        target_nodes = [pair[0], pair[1]]
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
                    operator="architect",
                )
                if architected is not None:
                    continue

            if (
                action == "transfer"
                and self.best_node_id is not None
                and self._can_attempt("transfer")
            ):
                transfered = self._architect(
                    self.all_nodes[self.best_node_id],
                    trigger=str(reasoning or "LLM identified a missing transfer experiment"),
                    operator="transfer",
                )
                if transfered is not None:
                    continue

            if action in {"tune", "refine"} and target_ids:
                target_id = target_ids[0]
                target_node = self.all_nodes.get(target_id)
                if (
                    target_node is not None
                    and target_node.executed
                    and not (target_node.result or {}).get("probe")
                ):
                    if action == "tune" and self._can_attempt("tune"):
                        res = self._tune(target_node)
                        if res is not None:
                            continue
                    elif action == "refine" and self._can_attempt("refine"):
                        res = self._refine(target_node, custom_plan=custom_plan)
                        if res is not None:
                            continue

            # Heuristic selection fallback via scheduler (lineage UCB plus a
            # prediction-complementarity bonus toward the incumbent signature).
            frontier_scores = self.scheduler.frontier_scores(
                "root",
                self.all_nodes,
                best_signature=self._incumbent_signature(),
                signature_provider=self._signature_provider,
                complementarity_weight=self.complementarity_weight,
            )
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
                f"{self.tuning_attempts}; screening probes="
                f"{self.probe_attempts}; total attempts={self.implementation_attempts}."
            )
        self._persist_tree_state()
        return self.best_node_id

    def _verify_final_node(self, node: NodeState) -> None:
        """Re-run the winning program once and compare the reproduced score.

        The stored score is self-reported by the generated program; re-running it
        against the same node inputs checks that the reported evaluation is
        reproducible. A mismatch only warns and preserves the validated deliverable.
        """
        if not self._env_enabled("AIBUILDAI_FINAL_VERIFY", default=True):
            return
        try:
            node_dir = self.run_root / node.node_id
            source = node_dir / "algorithm.py"
            if not source.is_file():
                return
            child_env = sanitized_subprocess_env()
            child_env.update(
                {
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": os.getenv("AIBUILDAI_MODEL_THREADS", "4"),
                    "MKL_NUM_THREADS": os.getenv("AIBUILDAI_MODEL_THREADS", "4"),
                }
            )
            for proxy in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                child_env[proxy] = "http://127.0.0.1:9"
            child_env.pop("NO_PROXY", None)
            child_env.pop("no_proxy", None)
            self._log(
                f"Re-running {self.node_label(node.node_id)} once to verify the reported score."
            )
            completed = run_supervised_process(
                [self.python, str(source)],
                cwd=node_dir,
                env=child_env,
                stall_seconds=1800.0,
                hard_limit_seconds=7200.0,
                activity_root=node_dir,
                label=f"Final verification of {self.node_label(node.node_id)}",
            )
            payload = {}
            result_path = node_dir / "result.json"
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            verification: dict[str, Any] = {
                "returncode": completed.returncode,
                "elapsed_seconds": round(completed.elapsed_seconds, 3),
                "termination_reason": completed.termination_reason,
            }
            if not isinstance(payload, dict) or payload.get("score") is None:
                verification["outcome"] = "no_reproducible_score"
                self._log("Final verification produced no score; keeping the stored deliverable.")
            else:
                try:
                    new_score = float(payload["score"])
                    verification["reported_score"] = new_score
                except (TypeError, ValueError):
                    new_score = math.nan
                stored_score = float(node.result["score"])
                tolerance = max(1e-4, abs(stored_score) * 0.01)
                if math.isfinite(new_score) and abs(new_score - stored_score) <= tolerance:
                    verification["outcome"] = "consistent"
                    self._log(
                        f"Final verification reproduced score {new_score:.8g} "
                        f"(stored {stored_score:.8g})."
                    )
                else:
                    verification["outcome"] = "inconsistent"
                    self._log(
                        f"WARNING: final verification reproduced score {new_score:.8g} "
                        f"but the stored score is {stored_score:.8g}; keeping the stored "
                        "validated deliverable."
                    )
            node.result = dict(node.result or {})
            node.result["final_verification"] = verification
            self._mark_dirty(node.node_id)
            self._persist_tree_state()
        except Exception as exc:
            self._log(
                f"Final verification was skipped: {type(exc).__name__}: {exc}"
            )

    def _ensemble_candidates(self) -> list[NodeState]:
        """Completed measured nodes that stored OOF predictions on disk."""
        candidates: list[NodeState] = []
        for node in self._successful_nodes():
            if node.operator == "ensemble":
                continue
            oof_path = (node.result or {}).get("oof_predictions")
            if oof_path and Path(str(oof_path)).is_file():
                candidates.append(node)
        return candidates

    def _build_final_ensemble(self) -> NodeState | None:
        """Blend the stored OOF predictions of the strongest measured nodes."""
        if not self._can_attempt("ensemble"):
            return None
        candidates = self._ensemble_candidates()
        if len(candidates) < 2:
            self._log(
                "Final ensemble skipped: fewer than two measured nodes stored "
                "OOF predictions (expected when tasks have no labeled validation set)."
            )
            return None
        top = candidates[:5]
        first = top[0]
        pair = self._merge_pair(top)
        second = pair[1] if pair is not None and pair[1] is not first else top[1]
        path_lines = []
        for node in top:
            path_lines.append(
                f"- ../{node.node_id}/oof_predictions.npz (arrays `oof_pred`, "
                "`oof_index`, `test_pred`, `test_index`)"
            )
        plan = (
            "FINAL ENSEMBLE STEP (mandatory):\n"
            "- Each measured parent below saved out-of-fold and test predictions:\n"
            + "\n".join(path_lines)
            + "\n- Load every parent's `oof_pred`/`oof_index` and the labels for those "
            "validation rows from the council-approved inputs exactly as the parents did "
            "(inspect the parent code).\n"
            "- Optimize non-negative blend weights over the parents on the shared metric "
            f"({self.metric_name} {self.metric_direction}) with a bounded search "
            "(random/grid over the simplex, at most 300 evaluations, early stopping), "
            "preferring simpler weight vectors.\n"
            "- Compare the winning blend against every parent on the identical validation "
            "rows; if it does not beat the best parent, emit the best parent's `test_pred` "
            "unchanged.\n"
            f"- Apply the chosen weights to the parents' `test_pred`/`test_index` and write "
            "the requested deliverable with the same sample schema (inspect each "
            "`../{node_id}/submission*` directory for the exact contract).\n"
            "- Write `result.json` with the shared `evaluation_protocol_hash`, `fold_scores`, "
            "`validation_sample_count`, and the chosen score."
        )
        self._log(
            f"Building final OOF ensemble over {len(top)} stored prediction sets."
        )
        planning = self._create_planning_node(
            plan,
            operator="ensemble",
            parent=first,
            executed=True,
            priority=0.0,
            base=first,
            companion=second,
            merge_sources=[node.node_id for node in top],
        )
        child = self._execute(
            plan,
            operator="ensemble",
            parent=planning,
            base=first,
            companion=second,
        )
        self._persist_tree_state()
        return child

    def generate_final_submission(self, best_node_id: str) -> bool:
        node = self.all_nodes.get(best_node_id)
        if node is None or not node.result or not node.result.get("output"):
            return False
        chosen = node
        if (
            self.final_ensemble_enabled
            and self.council_brief is not None
            and self.council_brief.evaluation_protocol.mode == "cross_validation"
        ):
            try:
                ensemble_node = self._build_final_ensemble()
            except Exception as exc:
                self._log(f"Final ensemble failed and was skipped: {exc}")
                ensemble_node = None
            if (
                ensemble_node is not None
                and ensemble_node.result
                and ensemble_node.result.get("status") == "completed"
                and ensemble_node.result.get("score") is not None
            ):
                ensemble_score = float(ensemble_node.result["score"])
                best_score = float(node.result["score"])
                if self._improved(
                    ensemble_score, best_score, noise=self._noise_for(node)
                ):
                    chosen = ensemble_node
                    self.best_node_id = ensemble_node.node_id
                    self._log(
                        f"Final ensemble {self.node_label(ensemble_node.node_id)} "
                        f"improved the best deliverable ({best_score:.8g} -> "
                        f"{ensemble_score:.8g}); using its output."
                    )
                else:
                    self._log(
                        f"Final ensemble scored {ensemble_score:.8g} without beating "
                        f"the best node ({best_score:.8g}); keeping the best deliverable."
                    )
        self._verify_final_node(chosen)
        validation = self.submission_validator.validate(
            str(chosen.result["output"]),
            self.task_analysis,
            allowed_root=self.run_root / chosen.node_id,
        )
        chosen.result["final_submission_validation"] = validation.to_dict()
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
                elif node.node_type == "planning":
                    title = f"{self.node_label(node_id)} (Planning)"
                    if not node.executed:
                        desc = f"PENDING — not executed within budget\n{self._clean_plan_text(node.plan)}"
                        color, border = "#FFF8E1", "#F9A825"
                    else:
                        desc = self._clean_plan_text(node.plan)
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
                        op = node.operator or "root"
                        desc = f"{op}\nScore: {float(score):.5f}"
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
