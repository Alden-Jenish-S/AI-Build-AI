from __future__ import annotations

import os
import copy
import json
import math
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from packaging.requirements import Requirement
from evaluation.metrics import default_metric_for_problem
from evaluation.prediction_io import load_prediction_table
from evaluation.submission import validate_submission_file
from tree.node import NodeState
from tree.scheduler import UCB1Scheduler
from tree.global_memory import GlobalMemory
from .technique_agent import TechniqueAgent
from .implementation_agent import ImplementationAgent
from .aggregator_agent import AggregatorAgent
from .setup_agent import SetupAgent
from .task_analyzer import TaskAnalyzer
from .modality_scaffold import task_loader_source, write_runtime_data_contract
from memory_pool.builder.l2_builder import L2Builder
from memory_pool.query_tool import normalize_resource_profile
from search import (
    ArtifactRecord,
    DiversityController,
    EvidenceService,
    InformationGainStrategy,
    PromotionController,
    ProvenanceGraph,
    PruningPolicy,
    TuningCoordinator,
    TuningKnowledgeBase,
)
from runtime_utils import (
    absolute_path_without_symlink_resolution,
    detect_available_accelerators,
    infer_task_type,
    select_preferred_accelerator,
    validate_path_component,
)

class ManagerAgent:
    def __init__(
        self,
        task_name: str,
        total_budget: int = 10,
        venv_path: str | None = None,
        model_name: str = None,
        run_suffix: str = None,
    ):
        self.task_name = validate_path_component(task_name, "task_name")
        if run_suffix is not None:
            run_suffix = validate_path_component(run_suffix, "run_suffix")
        if not isinstance(total_budget, int) or isinstance(total_budget, bool) or total_budget < 1:
            raise ValueError(f"total_budget must be a positive integer, got {total_budget!r}")
        self.total_budget = total_budget
        self.model_name = model_name
        
        # Directories
        self.project_root = Path(__file__).resolve().parent.parent
        requirements_file = self.project_root / "requirements.txt"
        self.allowed_dependencies = []
        if requirements_file.is_file():
            for raw_line in requirements_file.read_text(
                encoding="utf-8"
            ).splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    self.allowed_dependencies.append(Requirement(line).name)
                except ValueError:
                    continue
        
        import sys
        import subprocess
        if venv_path is None:
            self.venv_path = sys.executable
        else:
            resolved_venv = Path(venv_path)
            if not resolved_venv.is_absolute():
                resolved_venv = self.project_root / resolved_venv
            resolved_path = str(
                absolute_path_without_symlink_resolution(resolved_venv)
            )

            # Check if the explicitly selected interpreter is functional.
            use_fallback = True
            if Path(resolved_path).exists():
                try:
                    res = subprocess.run(
                        [resolved_path, "-c", "import sys; print('ok')"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if res.returncode == 0 and "ok" in res.stdout:
                        use_fallback = False
                except Exception:
                    pass

            if use_fallback:
                print(
                    f"ManagerAgent WARNING: Specified python path "
                    f"'{resolved_path}' is invalid or non-functional. Falling "
                    f"back to active running interpreter: {sys.executable}"
                )
                self.venv_path = sys.executable
            else:
                self.venv_path = resolved_path
        
        self.task_dir = self.project_root / "tasks" / self.task_name
        if not self.task_dir.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {self.task_dir}")
        self.task_spec = TaskAnalyzer().resolve(self.task_dir)
        self.modality = self.task_spec.modality
        self.component_modalities = self.task_spec.component_modalities
        self.output_type = self.task_spec.output.type
        if run_suffix:
            self.run_root = self.project_root / "runs" / self.task_name / run_suffix
        else:
            self.run_root = self.project_root / "runs" / self.task_name
        self.run_root.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.scheduler = UCB1Scheduler(total_budget=total_budget)
        self.global_memory = GlobalMemory()
        self.setup_agent = SetupAgent(venv_python_path=self.venv_path)
        self.technique_agent = TechniqueAgent(model_name=self.model_name)
        self.implementation_agent = ImplementationAgent(venv_python_path=self.venv_path, model_name=self.model_name)
        self.aggregator_agent = AggregatorAgent()
        self.evidence_service = EvidenceService()
        self.pruning_policy = PruningPolicy()
        self.promotion_controller = PromotionController()
        self.information_gain_strategy = InformationGainStrategy()
        self.diversity_controller = DiversityController()
        self.provenance_graph = ProvenanceGraph(
            self.run_root / "provenance_graph.json"
        )
        self.tuning_coordinator = TuningCoordinator(
            TuningKnowledgeBase(
                self.run_root / "tuning_history.jsonl"
            )
        )
        self._scheduled_merge_pairs: set[tuple[str, str]] = set()
        
        # State tracker
        self.all_nodes: Dict[str, NodeState] = {}
        self.node_counter = 0
        self.experiments_executed = 0
        self.implementation_attempts = 0

        # Load task config for metric direction and the renewable progress lease.
        self.metric_direction = self.task_spec.metric_direction
        self.metric_name = self.task_spec.primary_metric
        self.progress_stall_seconds = 1800
        self.enable_multi_fidelity = True
        self.ensemble_top_k = 3
        self.ensemble_strategy = "auto"
        self.uncertainty_weight = 1.0
        self.max_l1_categories = 8
        self.max_artifact_candidates = 5
        self.max_fine_tune_rounds = 2
        self.max_debug_attempts = 3
        self.max_parallel_root_nodes = max(
            1, min(3, int(os.cpu_count() or 1))
        )
        self.enforce_evaluation_contract = True
        self.enable_executable_artifacts = (
            os.environ.get(
                "METHOD_TREE_ENABLE_EXECUTABLE_ARTIFACTS", "0"
            )
            == "1"
        )
        self.accelerator_allowlist = None
        self.accelerator_preference = "auto"
        self.available_accelerators = {"cpu"}
        self.preferred_accelerator = "cpu"
        self.available_ram_gb = self._available_ram_gb()

        config_file = self.task_dir / "task_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    task_config = json.load(f)
                self.progress_stall_seconds = task_config.get(
                    "progress_stall_seconds", 1800
                )
                self.enable_multi_fidelity = bool(
                    task_config.get("enable_multi_fidelity", True)
                )
                self.ensemble_top_k = max(1, int(task_config.get("ensemble_top_k", 3)))
                self.ensemble_strategy = task_config.get(
                    "ensemble_strategy", "auto"
                )
                self.uncertainty_weight = max(
                    0.0, float(task_config.get("uncertainty_weight", 1.0))
                )
                self.max_l1_categories = max(
                    1, int(task_config.get("max_l1_categories", 8))
                )
                self.max_artifact_candidates = max(
                    1, int(task_config.get("max_artifact_candidates", 5))
                )
                self.max_fine_tune_rounds = max(
                    0, int(task_config.get("max_fine_tune_rounds", 2))
                )
                self.max_debug_attempts = max(
                    0, int(task_config.get("max_debug_attempts", 3))
                )
                self.max_parallel_root_nodes = max(
                    1,
                    int(
                        task_config.get(
                            "max_parallel_root_nodes",
                            self.max_parallel_root_nodes,
                        )
                    ),
                )
                self.enable_executable_artifacts = bool(
                    task_config.get(
                        "enable_executable_artifacts",
                        self.enable_executable_artifacts,
                    )
                )
                resource_limits = task_config.get("resource_limits", {})
                if isinstance(resource_limits, dict):
                    accelerators = resource_limits.get("accelerators")
                    if isinstance(accelerators, list) and accelerators:
                        allowed_accelerators = {
                            str(item).lower() for item in accelerators
                        }
                        if "gpu" in allowed_accelerators:
                            allowed_accelerators.update({"cuda", "mps"})
                        allowed_accelerators.add("cpu")
                        self.accelerator_allowlist = allowed_accelerators
                    self.accelerator_preference = resource_limits.get(
                        "preferred_accelerator", "auto"
                    )
                    if resource_limits.get("max_ram_gb") is not None:
                        configured_ram_gb = max(
                            0.0, float(resource_limits["max_ram_gb"])
                        )
                        if self.available_ram_gb > 0:
                            self.available_ram_gb = min(
                                self.available_ram_gb, configured_ram_gb
                            )
                        else:
                            self.available_ram_gb = configured_ram_gb
            except Exception as e:
                print(f"ManagerAgent WARNING: Failed to parse task_config.json: {e}")
        self._refresh_accelerator_state()
        if self.metric_direction not in {"maximize", "minimize"}:
            raise ValueError(
                f"task_config metric_direction must be 'maximize' or 'minimize', "
                f"got {self.metric_direction!r}"
            )
        if (
            not isinstance(self.progress_stall_seconds, (int, float))
            or isinstance(self.progress_stall_seconds, bool)
            or not math.isfinite(self.progress_stall_seconds)
            or self.progress_stall_seconds <= 0
        ):
            raise ValueError(
                "task_config progress_stall_seconds must be positive and finite, "
                f"got {self.progress_stall_seconds!r}"
            )
        if self.ensemble_strategy not in {"auto", "average", "rank_average"}:
            raise ValueError(
                "task_config ensemble_strategy must be 'auto', 'average', or "
                "'rank_average'"
            )
        self.technique_agent.max_l1_categories = self.max_l1_categories
        self.technique_agent.max_artifact_candidates = self.max_artifact_candidates
        print(
            "ManagerAgent resources: "
            f"accelerators={sorted(self.available_accelerators)}, "
            f"selected={self.preferred_accelerator}, "
            f"ram_gb={self.available_ram_gb:.1f}"
        )
        
        # Load clean task description for web search queries (Bug 3: prevents branch bias leakage)
        self.task_description = (
            f"{self.modality} {self.task_spec.problem_type} ML task: "
            f"{task_name}"
        )
        desc_file = self.task_dir / "task_description.md"
        if desc_file.exists():
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    self.task_description = f.read().strip()
            except Exception:
                pass
        self.task_type = self.task_spec.problem_type
        self.task_description = (
            "Canonical task context: "
            f"modality={self.modality}; "
            f"component_modalities={list(self.component_modalities)}; "
            f"problem_type={self.task_type}; "
            f"output_type={self.output_type}; "
            f"primary_metric={self.metric_name} "
            f"({self.metric_direction}).\n"
            + self.task_description
        )
                
        print(
            "ManagerAgent initialized: "
            f"direction={self.metric_direction}, "
            "runtime_limit=None, "
            f"progress_stall_seconds={self.progress_stall_seconds}"
        )

    @staticmethod
    def _available_ram_gb() -> float:
        try:
            return (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            ) / (1024 ** 3)
        except (AttributeError, OSError, ValueError):
            return 0.0

    def _resolved_metric_name(self) -> str:
        """Return the task metric for normal and lightweight test instances."""
        metric_name = getattr(self, "metric_name", None)
        if metric_name:
            return str(metric_name)
        task_spec = getattr(self, "task_spec", None)
        if task_spec is not None:
            return str(task_spec.primary_metric)
        return default_metric_for_problem(
            getattr(self, "task_type", "supervised"),
            getattr(self, "output_type", None),
        )

    def _refresh_accelerator_state(self) -> None:
        """Re-probe the selected interpreter after dependency installation."""
        available = detect_available_accelerators(self.venv_path)
        allowlist = getattr(self, "accelerator_allowlist", None)
        if allowlist:
            available &= set(allowlist)
            available.add("cpu")
        self.available_accelerators = available
        self.preferred_accelerator = select_preferred_accelerator(
            available, getattr(self, "accelerator_preference", "auto")
        )

    @staticmethod
    def _initial_fanout_for_budget(total_budget: int) -> int:
        return min(3, max(1, int(total_budget) // 3))

    @staticmethod
    def _initial_candidate_count_for_budget(total_budget: int) -> int:
        """Include one bounded recovery approach for very small searches."""
        return min(3, max(2, int(total_budget)))

    def _spawn_root_approach(
        self, root_id: str, approach: dict, *, replacement: bool = False
    ) -> str:
        """Materialize one primary or backup root approach."""
        name = approach.get("name", "Branch_Plan")
        plan = approach.get("plan", "")
        node_id = self.get_new_node_id()
        child_node = NodeState(
            node_id=node_id,
            parent_id=root_id,
            node_type="technique",
            plan=plan,
            operator="root",
            fidelity="screen" if self.enable_multi_fidelity else "full",
            config={
                "priority": 0.0,
                "allowed_scopes": ["full_pipeline", "model_family"],
                "replacement_branch": replacement,
            },
        )
        self.all_nodes[node_id] = child_node
        self.all_nodes[root_id].children_ids.append(node_id)
        self._persist_node(node_id)
        label = "replacement branch" if replacement else "branch"
        print(
            f"ManagerAgent: Spawned {label} {node_id}: {name} "
            f"(Plan: {plan[:60]}...)"
        )
        return node_id

    def _promote_backup_approach(self, root_id: str) -> str | None:
        """Replace a failed root experiment without spending planning budget."""
        backups = getattr(self, "_backup_initial_approaches", [])
        if not backups:
            return None
        approach = backups.pop(0)
        node_id = self._spawn_root_approach(
            root_id, approach, replacement=True
        )
        self.initial_fanout += 1
        self.scheduler.set_warmup_budget(self.initial_fanout)
        self._trace_search(
            {
                "event": "backup_root_promoted",
                "node_id": node_id,
                "remaining_backups": len(backups),
            }
        )
        return node_id

    def _prepare_run_root(self) -> None:
        """Start with an empty run directory while preserving prior attempts."""
        self.run_root.mkdir(parents=True, exist_ok=True)
        if not any(self.run_root.iterdir()):
            return

        archive_root = self.run_root.parent / "archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived_run = archive_root / f"{self.run_root.name}_{timestamp}"
        shutil.move(str(self.run_root), str(archived_run))
        self.run_root.mkdir(parents=True, exist_ok=True)
        print(f"ManagerAgent: Archived previous run at {archived_run}")

    def _ensure_search_services(self) -> None:
        """Provide policy services for normal construction and focused unit tests."""
        if not hasattr(self, "evidence_service"):
            self.evidence_service = EvidenceService()
        if not hasattr(self, "pruning_policy"):
            self.pruning_policy = PruningPolicy()
        if not hasattr(self, "promotion_controller"):
            self.promotion_controller = PromotionController()
        if not hasattr(self, "information_gain_strategy"):
            self.information_gain_strategy = InformationGainStrategy()
        if not hasattr(self, "diversity_controller"):
            self.diversity_controller = DiversityController()
        if not hasattr(self, "provenance_graph"):
            self.provenance_graph = ProvenanceGraph(
                Path(self.run_root) / "provenance_graph.json"
            )
        if not hasattr(self, "tuning_coordinator"):
            self.tuning_coordinator = TuningCoordinator(
                TuningKnowledgeBase(
                    Path(self.run_root) / "tuning_history.jsonl"
                )
            )
        if not hasattr(self, "_scheduled_merge_pairs"):
            self._scheduled_merge_pairs = set()

    def _root_parallel_capacity(self) -> int:
        """Return a conservative process count for independent root runs."""
        requested = max(
            1, int(getattr(self, "max_parallel_root_nodes", 1))
        )
        # A single logical CUDA/MPS accelerator must not receive concurrent
        # training jobs. The resource detector currently reports accelerator
        # classes, not distinct device IDs, so serialize accelerator work.
        if getattr(self, "preferred_accelerator", "cpu") != "cpu":
            return 1
        cpu_capacity = max(1, int(os.cpu_count() or 1))
        ram_gb = float(getattr(self, "available_ram_gb", 0.0) or 0.0)
        ram_capacity = max(1, int(ram_gb // 4.0)) if ram_gb else 1
        return max(1, min(requested, cpu_capacity, ram_capacity))

    def _run_implementation_payload(self, node: NodeState) -> dict:
        """Run one implementation without mutating ManagerAgent search state."""
        node_dir = self.run_root / node.node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        config = node.config or {}
        return self.implementation_agent.run(
            node_dir,
            config.get("technique_record", {}),
            self.task_dir,
            task_assets_dir=self.run_root,
            stall_seconds=self.progress_stall_seconds,
            metric_direction=self.metric_direction,
            base_algorithm_path=config.get("base_code_path"),
            parent_node_dir=(
                self.run_root / config["base_node_id"]
                if config.get("base_node_id")
                else None
            ),
            fidelity=node.fidelity,
            operator=node.operator,
            enforce_evaluation_contract=True,
            accelerator=self.preferred_accelerator,
            available_accelerators=set(self.available_accelerators),
            tuning_context=config.get("tuning_context"),
            max_debug_attempts=getattr(self, "max_debug_attempts", 3),
            metric_name=self.metric_name,
            evaluation_mode=config.get("evaluation_mode"),
            parallel_processes=max(
                1, int(config.get("_parallel_root_workers", 1))
            ),
        )

    def _execute_initial_root_batch(
        self,
        root_id: str,
        l1_index: dict,
        l1_path: Path,
    ) -> int:
        """Resolve initial techniques, then run eligible CPU roots concurrently."""
        technique_ids = [
            node_id
            for node_id, state in list(self.all_nodes.items())
            if state.parent_id == root_id
            and state.node_type == "technique"
            and not state.executed
        ]
        for node_id in technique_ids:
            self._execute_node(
                self.all_nodes[node_id], node_id, root_id, l1_index, l1_path
            )

        remaining_budget = max(
            0, self.total_budget - self.experiments_executed
        )
        candidates = [
            state
            for state in self.all_nodes.values()
            if state.node_type == "implementation"
            and state.operator == "root"
            and not state.executed
            and not (
                (state.config or {}).get("technique_record", {}).get(
                    "model_card"
                )
            )
        ][:remaining_budget]
        workers = min(self._root_parallel_capacity(), len(candidates))
        if workers < 2:
            return 0

        batch_ids = [node.node_id for node in candidates]
        self._trace_search(
            {
                "event": "parallel_root_batch_started",
                "node_ids": batch_ids,
                "workers": workers,
                "accelerator": self.preferred_accelerator,
            }
        )
        results: dict[str, dict] = {}
        for node in candidates:
            node.config = dict(node.config or {})
            node.config["_parallel_root_workers"] = workers
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="root-experiment",
        ) as executor:
            future_nodes = {}
            for node in candidates:
                self.implementation_attempts = (
                    getattr(self, "implementation_attempts", 0) + 1
                )
                self._trace_search(
                    {
                        "event": "implementation_attempt_started",
                        "node_id": node.node_id,
                        "implementation_attempt": self.implementation_attempts,
                        "fidelity": node.fidelity,
                        "parallel_root_batch": True,
                    }
                )
                future_nodes[
                    executor.submit(self._run_implementation_payload, node)
                ] = node.node_id
            for future in as_completed(future_nodes):
                node_id = future_nodes[future]
                try:
                    results[node_id] = future.result()
                except Exception as exc:
                    results[node_id] = {
                        "score": None,
                        "status": "failed",
                        "diagnostics": f"Parallel root exception: {exc}",
                    }

        completed = 0
        for node in candidates:
            node.config = dict(node.config or {})
            node.config.pop("_parallel_root_workers", None)
            node.config["_precomputed_result"] = results[node.node_id]
            if self._execute_node(
                node, node.node_id, root_id, l1_index, l1_path
            ):
                self.experiments_executed += 1
                completed += 1
        self._trace_search(
            {
                "event": "parallel_root_batch_completed",
                "node_ids": batch_ids,
                "completed_experiments": completed,
            }
        )
        self._persist_tree_state()
        return completed

    def get_new_node_id(self) -> str:
        self.node_counter += 1
        return f"node_{self.node_counter}"

    def _node_payload(self, node: NodeState) -> Dict[str, Any]:
        """Return the durable, compact representation used by files and plots."""
        result = None
        if node.result:
            result = {
                "score": node.result.get("score"),
                "status": node.result.get("status"),
                "reward": node.result.get("reward"),
                "raw_reward": node.result.get("raw_reward"),
                "uncertainty_penalty": node.result.get("uncertainty_penalty"),
                "elapsed_seconds": node.result.get("elapsed_seconds"),
                "validation": node.result.get("validation", {}),
                "oof_path": node.result.get("oof_path"),
                "validation_path": node.result.get("validation_path"),
                "evaluation_mode": node.result.get("evaluation_mode"),
                "evaluation_policy": node.result.get("evaluation_policy"),
                "error_analysis": node.result.get("error_analysis"),
                "error_analysis_path": node.result.get(
                    "error_analysis_path"
                ),
                "duplicate_of": node.result.get("duplicate_of"),
                "code_fingerprint": node.result.get(
                    "code_fingerprint"
                ),
                "tuning": node.result.get("tuning"),
                "merge": node.result.get("merge"),
                "statistical_evidence": node.result.get(
                    "statistical_evidence"
                ),
                "pruning_decision": node.result.get("pruning_decision"),
                "promotion_decision": node.result.get("promotion_decision"),
                "artifact_repair": node.result.get("artifact_repair"),
                "artifact_variant": node.result.get("artifact_variant"),
                "no_effect_reason": node.result.get("no_effect_reason"),
                "deduplicated_outputs": node.result.get("deduplicated_outputs", []),
            }
            diagnostics = node.result.get("diagnostics")
            if diagnostics:
                result["diagnostics_tail"] = str(diagnostics)[-4000:]
        config = dict(node.config or {})
        if config.get("technique_record"):
            config["technique_record"] = self._compact_technique_record(
                config["technique_record"]
            )
        if config.get("locked_technique_record"):
            config["locked_technique_record"] = self._compact_technique_record(
                config["locked_technique_record"]
            )
        return {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "node_type": node.node_type,
            "plan": node.plan,
            "code": node.code,
            "config": config or None,
            "result": result,
            "executed": node.executed,
            "status": (
                "pending"
                if not node.executed
                else (result or {}).get("status", "completed")
            ),
            "visits": node.visits,
            "total_reward": node.total_reward,
            "operator": node.operator,
            "fidelity": node.fidelity,
            "children_ids": list(node.children_ids),
        }

    @staticmethod
    def _compact_technique_record(record: dict) -> dict:
        """Keep durable state useful without embedding source code and long logs."""
        compact = dict(record or {})
        raw_outline = compact.pop("raw_outline", None)
        if raw_outline:
            compact["raw_outline_sha256"] = hashlib.sha256(
                str(raw_outline).encode("utf-8")
            ).hexdigest()
        for key in ("model_card",):
            card = compact.get(key)
            if isinstance(card, dict):
                card = dict(card)
                card.pop("code_content", None)
                if card.get("verification_log"):
                    card["verification_log_tail"] = str(card.pop("verification_log"))[-2000:]
                compact[key] = card
        candidate = compact.get("candidate_artifact")
        if isinstance(candidate, dict) and isinstance(candidate.get("model_card"), dict):
            candidate = dict(candidate)
            card = dict(candidate["model_card"])
            card.pop("code_content", None)
            if card.get("verification_log"):
                card["verification_log_tail"] = str(card.pop("verification_log"))[-2000:]
            candidate["model_card"] = card
            compact["candidate_artifact"] = candidate
        return compact

    def _trace_search(self, event: dict) -> None:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_step": getattr(self, "experiments_executed", 0),
            **event,
        }
        with open(self.run_root / "search_trace.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    @staticmethod
    def _sha256_file(path: Path) -> str | None:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _no_effect_reason(self, node: NodeState, node_dir: Path) -> str | None:
        """Detect nodes whose measured predictions reproduce an earlier node."""
        base_node_id = (node.config or {}).get("base_node_id")
        candidate_ids = []
        if base_node_id:
            candidate_ids.append(str(base_node_id))
        candidate_ids.extend(
            node_id
            for node_id, state in getattr(self, "all_nodes", {}).items()
            if node_id not in {node.node_id, base_node_id, "root"}
            and state.node_type == "implementation"
            and state.result
            and state.result.get("status") == "completed"
        )
        candidate_ids = list(dict.fromkeys(candidate_ids))
        child_oof = next(
            (
                path
                for path in (
                    node_dir / "oof_predictions.npz",
                    node_dir / "oof_predictions.csv",
                )
                if path.is_file()
            ),
            node_dir / "oof_predictions.npz",
        )
        child_hash = self._sha256_file(child_oof)
        for candidate_id in candidate_ids:
            candidate_root = self.run_root / candidate_id
            candidate_oof = next(
                (
                    path
                    for path in (
                        candidate_root / "oof_predictions.npz",
                        candidate_root / "oof_predictions.csv",
                    )
                    if path.is_file()
                ),
                candidate_root / "oof_predictions.npz",
            )
            candidate_hash = self._sha256_file(candidate_oof)
            if child_hash and child_hash == candidate_hash:
                return (
                    "OOF predictions are byte-identical to measured node "
                    f"{candidate_id}"
                )
        try:
            import numpy as np
            from evaluation.prediction_io import legacy_prediction_payload

            def same_oof(reference_path: Path, measured_path: Path) -> bool:
                try:
                    reference = load_prediction_table(
                        reference_path.with_suffix("")
                    )
                    measured = load_prediction_table(
                        measured_path.with_suffix("")
                    )
                except FileNotFoundError:
                    return False
                if (
                    "row_id" not in reference
                    or "row_id" not in measured
                    or reference["row_id"].duplicated().any()
                    or measured["row_id"].duplicated().any()
                ):
                    return False
                if set(reference["row_id"]) != set(measured["row_id"]):
                    return False
                left = reference.set_index("row_id").sort_index().reset_index()
                right = measured.set_index("row_id").sort_index().reset_index()
                left_values, left_classes = legacy_prediction_payload(left)
                right_values, right_classes = legacy_prediction_payload(right)
                return bool(
                    left_classes == right_classes
                    and np.asarray(left_values).shape
                    == np.asarray(right_values).shape
                    and
                    np.allclose(
                        np.asarray(left_values, dtype=float),
                        np.asarray(right_values, dtype=float),
                        rtol=1e-12,
                        atol=1e-12,
                    )
                )

            for candidate_id in candidate_ids:
                if same_oof(
                    self.run_root / candidate_id / "oof_predictions",
                    child_oof,
                ):
                    return (
                        "OOF predictions are numerically identical to measured "
                        f"node {candidate_id}"
                    )
        except Exception:
            pass
        return None

    def _deduplicate_node_outputs(self, node: NodeState, node_dir: Path) -> list[str]:
        """Hard-link exact parent duplicates while retaining node-local paths."""
        base_node_id = (node.config or {}).get("base_node_id")
        if not base_node_id:
            return []
        parent_dir = self.run_root / base_node_id
        relative_paths = (
            Path("oof_predictions.npz"),
            Path("oof_predictions.csv"),
            Path("submission") / "submission.csv",
        )
        deduplicated = []
        for relative in relative_paths:
            parent_path, child_path = parent_dir / relative, node_dir / relative
            parent_hash = self._sha256_file(parent_path)
            if not parent_hash or parent_hash != self._sha256_file(child_path):
                continue
            child_path.unlink()
            try:
                os.link(parent_path, child_path)
            except OSError:
                shutil.copy2(parent_path, child_path)
            deduplicated.append(str(relative))
        return deduplicated

    def _persist_node(self, node_id: str) -> None:
        """Persist every agent node, including pending technique/frontier nodes."""
        if node_id == "root" or node_id not in self.all_nodes:
            return
        node = self.all_nodes[node_id]
        node_dir = self.run_root / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        with open(node_dir / "node_state.json", "w", encoding="utf-8") as f:
            json.dump(self._node_payload(node), f, indent=2, default=str)
        # tree_state.json is the canonical graph snapshot and node_state.json is
        # the compact crash-recovery shard. Separate technique_plan,
        # technique_record, and raw_outline files duplicated the same payload
        # without adding recovery information.

    def _persist_tree_state(self) -> None:
        """Write the canonical tree used to generate method_tree.png."""
        payload = {
            "task_name": self.task_name,
            "metric_direction": self.metric_direction,
            "metric_name": self._resolved_metric_name(),
            "budget": self.total_budget,
            "budget_unit": "completed_evaluation_experiment",
            "initial_fanout": getattr(self, "initial_fanout", None),
            "ucb_eligible_budget": max(
                0, self.total_budget - getattr(self, "initial_fanout", 0)
            ),
            "experiments_executed": getattr(self, "experiments_executed", 0),
            "implementation_attempts": getattr(
                self, "implementation_attempts", 0
            ),
            "max_fine_tune_rounds": getattr(self, "max_fine_tune_rounds", 2),
            "best_node_id": getattr(self, "best_node_id", None),
            "final_submission_status": getattr(
                self, "final_submission_status", "not_attempted"
            ),
            "final_submission_validation": getattr(
                self, "final_submission_validation", None
            ),
            "provenance_graph": "provenance_graph.json",
            "execution_graph": "single_parent_tree",
            "merge_operator": "merge_ensemble",
            "resource_capacity": {
                "accelerators": sorted(
                    getattr(self, "available_accelerators", {"cpu"})
                ),
                "preferred_accelerator": getattr(
                    self, "preferred_accelerator", "cpu"
                ),
                "ram_gb": getattr(self, "available_ram_gb", 0.0),
            },
            "nodes": {
                node_id: self._node_payload(node)
                for node_id, node in self.all_nodes.items()
            },
        }
        with open(self.run_root / "tree_state.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def _feasibility_reason(
        self, model_card: dict | None, check_accelerator: bool = True
    ) -> str | None:
        """Reject statically incompatible artifacts before charging an experiment."""
        if not model_card:
            return None
        profile = normalize_resource_profile(model_card)
        accelerator = profile["accelerator"]
        if (
            check_accelerator
            and accelerator == "gpu"
            and not ({"cuda", "mps", "gpu"} & self.available_accelerators)
        ):
            return "artifact requires a GPU but this run exposes CPU only"
        if (
            check_accelerator
            and accelerator in {"cuda", "mps"}
            and accelerator not in self.available_accelerators
        ):
            return f"artifact requires {accelerator.upper()} but that accelerator is unavailable"
        if (
            self.available_ram_gb > 0
            and profile["min_ram_gb"] > self.available_ram_gb
        ):
            return (
                f"artifact requires {profile['min_ram_gb']:.1f} GB RAM but only "
                f"{self.available_ram_gb:.1f} GB is available"
            )
        return None

    @staticmethod
    def _operator_compatibility_reason(
        model_card: dict | None, operator: str | None
    ) -> str | None:
        if not model_card or not operator:
            return None
        capabilities = model_card.get("capabilities")
        interface = model_card.get("interface", {})
        interface_text = json.dumps(interface, default=str).lower()
        description = str(model_card.get("description", "")).lower()
        if (
            ("oof" in interface_text or "out-of-fold" in description)
            and (
                not isinstance(capabilities, dict)
                or capabilities.get("accepts_harness_fold_ids") is not True
            )
        ):
            return (
                "OOF-producing artifact cannot accept harness fold_ids; "
                "its internal split would invalidate evaluation"
            )
        if (
            ("oof" in interface_text or "out-of-fold" in description)
            and isinstance(capabilities, dict)
            and capabilities.get("refits_full_training_data") is not True
        ):
            return (
                "OOF-producing artifact does not declare a full-training "
                "refit for final predictions"
            )
        if operator in {"root", "promote"}:
            return None
        if not isinstance(capabilities, dict):
            return None  # legacy cards remain usable; no-effect detection is the backstop
        supported = capabilities.get("supported_operators")
        if isinstance(supported, list) and operator not in supported:
            return f"artifact does not declare support for the {operator!r} operator"
        if operator == "tune" and not capabilities.get("tunable_parameters"):
            return "artifact exposes no declared tunable parameters"
        return None

    def run_tree_search(self) -> str:
        """
        Run the budget-scaled method tree directly on the task.

        Task profiling and data-loader scaffolding are prepared before root
        branches; no preliminary model is generated or executed.
        """
        self._prepare_run_root()
        task_analysis = TaskAnalyzer().analyze(
            self.task_dir,
            output_dir=self.run_root,
            include_index=True,
        )
        diagnostics = task_analysis.profile.get("diagnostics", {})
        self.technique_agent.set_dataset_directives(
            diagnostics.get("synthesized_directives", [])
            if isinstance(diagnostics, dict)
            else []
        )
        (self.run_root / "task_dataloader.py").write_text(
            task_loader_source(), encoding="utf-8"
        )
        write_runtime_data_contract(
            self.run_root / "dataset_index.jsonl",
            self.run_root / "runtime_data_contract.json",
            task_spec_path=self.run_root / "resolved_task_spec.json",
            task_dir=self.task_dir,
        )
        self._ensure_search_services()
        self._scheduled_merge_pairs.clear()
        print(f"\n==========================================")
        print(f"ManagerAgent: Starting Tree Search for {self.task_name}")
        print(f"==========================================")
        
        # Set task run folder in SetupAgent so it logs dependencies there
        self.setup_agent.set_task_run_dir(self.run_root)
        
        # 1. Create root virtual node
        root_id = "root"
        self.all_nodes[root_id] = NodeState(
            node_id=root_id,
            parent_id=None,
            node_type="technique",
            plan="Root virtual node",
            executed=True
        )
        
        # 2. Scale forced root coverage with the experiment budget.
        # If ideation fails, stop clearly instead of silently biasing the run with canned defaults.
        self.initial_fanout = self._initial_fanout_for_budget(self.total_budget)
        self.scheduler.set_warmup_budget(self.initial_fanout)
        # Keep a pre-generated recovery plan even for a one-experiment search.
        candidate_count = self._initial_candidate_count_for_budget(
            self.total_budget
        )
        dynamic_approaches = self.technique_agent.generate_initial_approaches(
            self.task_description, count=candidate_count
        )
        print("ManagerAgent: LLM initial branch ideas:")
        for idx, app in enumerate(dynamic_approaches, start=1):
            role = "primary" if idx <= self.initial_fanout else "backup"
            print(
                f"  {idx}. {app.get('name', 'unnamed_branch')} [{role}]: "
                f"{app.get('plan', '')}"
            )

        primary_approaches = dynamic_approaches[: self.initial_fanout]
        self._backup_initial_approaches = list(
            dynamic_approaches[self.initial_fanout :]
        )
        for app in primary_approaches:
            self._spawn_root_approach(root_id, app)

        self._persist_tree_state()

        # Load L1 index for the technique agents
        l1_path = self.project_root / "memory_pool" / "l1_index.json"
        if self.enable_executable_artifacts:
            with open(l1_path, 'r', encoding='utf-8') as f:
                l1_index = json.load(f)
        else:
            l1_index = {}

        # Resolve the forced initial coverage before adaptive UCB expansion.
        # Independent, self-contained CPU roots launch their model subprocesses
        # concurrently; result integration remains serialized in the manager.
        self._execute_initial_root_batch(
            root_id, l1_index, l1_path
        )
            
        # 3. Main search loop. Planning and broken generated scripts do not
        # consume scientific budget; completed evaluation runs do.
        action_count = 0
        action_guard = self.total_budget * 8 + 16
        while self.experiments_executed < self.total_budget:
            self.scheduler.current_step = self.experiments_executed
            action_count += 1
            if action_count > action_guard:
                print("ManagerAgent: Planning-action guard reached; stopping safely.")
                break
            print(
                f"\n--- Search Action {action_count} "
                f"(experiments {self.experiments_executed}/{self.total_budget}) ---"
            )
            
            # Select node to execute/expand using UCB1 scheduler
            frontier_scores = self.scheduler.frontier_scores(root_id, self.all_nodes)
            selected_id = max(frontier_scores, key=frontier_scores.get) if frontier_scores else None
            if selected_id is None:
                print("ManagerAgent: Search tree is exhausted; stopping early.")
                break
            node = self.all_nodes[selected_id]
            self._trace_search(
                {
                    "event": "selection",
                    "selected_node_id": selected_id,
                    "selected_node_type": node.node_type,
                    "exploration_constant": self.scheduler.get_exploration_constant(
                        self.experiments_executed
                    ),
                    "frontier_scores": frontier_scores,
                }
            )
            print(f"ManagerAgent: Selected Node {selected_id} (Type: {node.node_type})")
            
            # Bug 2 fix: wrap step body in try/except for node-level failure isolation
            try:
                attempted_experiment = self._execute_node(
                    node, selected_id, root_id, l1_index, l1_path
                )
                if attempted_experiment:
                    self.experiments_executed += 1
            except Exception as e:
                print(f"ManagerAgent: ERROR in node {selected_id}: {e}")
                import traceback
                traceback.print_exc()
                # Mark node as executed-but-failed so we don't re-select it
                node.executed = True
                node.result = {
                    "score": None,
                    "status": "failed",
                    "diagnostics": f"Node exception: {e}",
                }
                if node.node_type == "implementation":
                    self.scheduler.backpropagate(selected_id, -1.0, self.all_nodes)
                    if node.operator == "root":
                        replacement_id = self._promote_backup_approach(root_id)
                        if replacement_id:
                            print(
                                "ManagerAgent: Promoted a backup approach after "
                                "a pre-execution implementation failure."
                            )
                self._persist_node(selected_id)
                self._persist_tree_state()
                continue
                
        # Compare candidates at the highest completed fidelity. This prevents a
        # noisy screening score from displacing a rigorously evaluated candidate.
        fidelity_rank = {"screen": 0, "medium": 1, "full": 2}
        successful_nodes = [
            (nid, state)
            for nid, state in self.all_nodes.items()
            if state.node_type == "implementation"
            and state.result
            and state.result.get("score") is not None
            and state.result.get("status") == "completed"
        ]
        max_completed_fidelity = max(
            (fidelity_rank.get(state.fidelity, 0) for _, state in successful_nodes),
            default=-1,
        )
        best_node_id = None
        best_score = -float('inf') if self.metric_direction == "maximize" else float('inf')
        for nid, nstate in successful_nodes:
            if fidelity_rank.get(nstate.fidelity, 0) != max_completed_fidelity:
                continue
            score = nstate.result.get("score")
            if score is not None:
                if self.metric_direction == "maximize":
                    if score > best_score:
                        best_score = score
                        best_node_id = nid
                else:
                    if score < best_score:
                        best_score = score
                        best_node_id = nid
                    
        if best_node_id:
            self.best_node_id = best_node_id
            print(
                f"\nManagerAgent: Search finished after {self.experiments_executed} experiments. "
                f"Best Node: {best_node_id} (Score: {best_score:.5f}, "
                f"Fidelity: {self.all_nodes[best_node_id].fidelity})"
            )
        else:
            self.best_node_id = None
            print(f"\nManagerAgent: Tree Search finished! No successful implementation nodes found.")
            
        # Save final method tree image
        try:
            for node_id in self.all_nodes:
                self._persist_node(node_id)
            self._persist_tree_state()
            tree_img_path = self.run_root / "method_tree.png"
            self.save_tree_image(tree_img_path)
        except Exception as e:
            print(f"ManagerAgent WARNING: Failed to generate method tree image: {e}")
            
        return best_node_id

    def _conservative_score(self, score: float, cv_std: float = 0.0) -> float:
        """Discount a point estimate by its measured fold uncertainty."""
        score = float(score)
        try:
            uncertainty = max(0.0, float(cv_std or 0.0)) * getattr(
                self, "uncertainty_weight", 1.0
            )
        except (TypeError, ValueError):
            uncertainty = 0.0
        return (
            score - uncertainty
            if self.metric_direction == "maximize"
            else score + uncertainty
        )

    def _improves_on_score(
        self, score: float, comparison_score: float, cv_std: float = 0.0
    ) -> bool:
        conservative = self._conservative_score(score, cv_std)
        return (
            conservative > float(comparison_score)
            if self.metric_direction == "maximize"
            else conservative < float(comparison_score)
        )

    def _score_to_reward(self, score: float, cv_std: float = 0.0) -> float:
        """Map the task metric monotonically into a bounded scheduler reward."""
        conservative_score = self._conservative_score(score, cv_std)
        oriented = (
            conservative_score
            if self.metric_direction == "maximize"
            else -conservative_score
        )
        return oriented / (1.0 + abs(oriented))

    def _relative_improvement(
        self, score: float, reference_score: float
    ) -> float:
        """Normalize a candidate's change from a measured parent/reference."""
        score = float(score)
        reference_score = float(reference_score)
        if (
            self.metric_direction == "maximize"
            and 0.0 <= reference_score < 1.0
            and 0.0 <= score <= 1.0
        ):
            return (score - reference_score) / max(
                1.0 - reference_score, 1e-12
            )
        if self.metric_direction == "maximize":
            return (score - reference_score) / max(
                abs(reference_score), 1e-12
            )
        return (reference_score - score) / max(
            abs(reference_score), 1e-12
        )

    @staticmethod
    def _next_fidelity(fidelity: str) -> str:
        return {"screen": "medium", "medium": "full", "full": "full"}.get(
            fidelity, "full"
        )

    def _reference_result_for_node(self, node: NodeState) -> dict | None:
        """Return a measured parent or the strongest comparable prior method."""
        base_node_id = (node.config or {}).get("base_node_id")
        parent = self.all_nodes.get(base_node_id)
        if (
            parent is not None
            and parent.result
            and parent.result.get("score") is not None
            and parent.result.get("status") == "completed"
        ):
            return parent.result
        candidates = [
            state.result
            for state in self.all_nodes.values()
            if state.node_id != node.node_id
            and state.node_type == "implementation"
            and state.fidelity == node.fidelity
            and state.result
            and state.result.get("score") is not None
            and state.result.get("status") == "completed"
        ]
        if not candidates:
            return None
        selector = max if self.metric_direction == "maximize" else min
        return selector(candidates, key=lambda item: float(item["score"]))

    def _evidence_for_result(
        self, node: NodeState, reference: dict | None = None
    ):
        self._ensure_search_services()
        candidate = node.result or {}
        if candidate.get("score") is None:
            return None
        reference = reference or self._reference_result_for_node(node)
        if not reference or reference.get("score") is None:
            return None
        try:
            return self.evidence_service.compare(
                candidate,
                reference,
                direction=self.metric_direction,
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _model_family_for_node(node: NodeState) -> str:
        config = node.config or {}
        record = config.get("technique_record") or {}
        card = record.get("model_card") or (
            record.get("candidate_artifact") or {}
        ).get("model_card") or {}
        return str(
            card.get("artifact_id")
            or record.get("artifact_id")
            or next(iter((node.result or {}).get("implementation_families", [])), "")
            or "unknown_model_family"
        )

    def _record_tuning_history(self, node: NodeState) -> None:
        """Persist completed tuning evidence inside the current run only."""
        if node.operator != "tune" or not node.result:
            return
        tuning = node.result.get("tuning") or {}
        parameters = tuning.get("hyperparameters")
        if not isinstance(parameters, dict) or not parameters:
            return
        self._ensure_search_services()
        context = (node.config or {}).get("tuning_context") or {}
        validation = node.result.get("validation") or {}
        score = node.result.get("score")
        if score is None:
            return
        parent_score = context.get("parent_score")
        relative_improvement = (
            self._relative_improvement(score, parent_score)
            if parent_score is not None
            else 0.0
        )
        self.tuning_coordinator.record(
            trial_id=node.node_id,
            task_name=self.task_name,
            model_family=context.get("model_family")
            or self._model_family_for_node(node),
            search_space_version=context.get("search_space_version")
            or self.tuning_coordinator.search_space_version(
                context.get("tunable_parameters", [])
            ),
            parameters=parameters,
            score=float(score),
            normalized_improvement=relative_improvement,
            metric_name=self._resolved_metric_name(),
            metric_direction=self.metric_direction,
            fidelity=node.fidelity,
            dataset_fingerprint=validation.get("fold_assignment_sha256"),
            uncertainty=validation.get("cv_std"),
            elapsed_seconds=node.result.get("elapsed_seconds"),
            trial_count=tuning.get("tuning_trials") or 1,
            modality=self.modality,
            problem_type=self.task_type,
            output_type=self.output_type,
            accelerator_class=str(
                node.result.get("accelerator") or self.preferred_accelerator
            ),
        )

    def _record_node_provenance(self, node: NodeState) -> None:
        """Record artifact dependencies without adding scheduler parents."""
        if not node.result or node.result.get("status") != "completed":
            return
        self._ensure_search_services()
        config = node.config or {}
        validation = node.result.get("validation") or {}
        input_artifacts = list(config.get("input_artifact_ids") or [])
        base_node_id = config.get("base_node_id")
        if not input_artifacts and base_node_id:
            input_artifacts = [f"{base_node_id}:predictions"]
        parameters = (
            (node.result.get("tuning") or {}).get("hyperparameters")
            or config.get("parameters")
        )
        parameter_hash = (
            hashlib.sha256(
                json.dumps(
                    parameters, sort_keys=True, default=str
                ).encode("utf-8")
            ).hexdigest()
            if parameters
            else None
        )
        relation = (
            "ensembled_from"
            if node.operator == "merge_ensemble"
            else ("tuned_from" if node.operator == "tune" else "derived_from")
        )
        self.provenance_graph.record(
            ArtifactRecord(
                artifact_id=f"{node.node_id}:predictions",
                artifact_type="prediction_bundle",
                produced_by_node_id=node.node_id,
                dataset_fingerprint=validation.get(
                    "fold_assignment_sha256"
                ),
                code_hash=(
                    self._sha256_file(Path(node.code)) if node.code else None
                ),
                parameter_hash=parameter_hash,
                metadata={
                    "operator": node.operator,
                    "fidelity": node.fidelity,
                    "score": node.result.get("score"),
                    "oof_path": node.result.get("oof_path"),
                    "validation_path": node.result.get(
                        "validation_path"
                    ),
                    "evaluation_mode": node.result.get(
                        "evaluation_mode"
                    ),
                    "prediction_bundle": node.result.get(
                        "prediction_bundle"
                    ),
                    "model_bundle": node.result.get("model_bundle"),
                    "compatibility_key": node.result.get(
                        "compatibility_key"
                    ),
                    "modality": getattr(self, "modality", "tabular"),
                    "output_type": getattr(
                        self, "output_type", "class_probabilities"
                    ),
                },
            ),
            sources=[
                (artifact_id, relation) for artifact_id in input_artifacts
            ],
        )

    def _spawn_merge_ensemble_slot(
        self, node: NodeState, node_id: str, evidence
    ) -> None:
        """Create one manager-owned, OOF-backed ensemble action."""
        if (
            evidence is None
            or self.experiments_executed + 1 >= self.total_budget
            or getattr(self, "task_type", None) == "unsupervised_clustering"
            or not (node.result or {}).get("oof_path")
        ):
            return
        self._ensure_search_services()
        assessment = self.diversity_controller.best_partner(
            node_id=node_id,
            node_fidelity=node.fidelity,
            all_nodes=self.all_nodes,
            run_root=Path(self.run_root),
            metric_name=self._resolved_metric_name(),
            evidence=evidence,
            excluded_pairs=self._scheduled_merge_pairs,
            strategy=getattr(self, "ensemble_strategy", "auto"),
        )
        if assessment is None or assessment.utility <= 0.0:
            return
        partner_id = assessment.partner_node_id
        pair = tuple(sorted((node_id, partner_id)))
        self._scheduled_merge_pairs.add(pair)
        merge_id = self.get_new_node_id()
        input_node_ids = [node_id, partner_id]
        child = NodeState(
            node_id=merge_id,
            parent_id=node_id,
            node_type="implementation",
            plan="Ensemble two measured node models using the best OOF-backed strategy.",
            operator="merge_ensemble",
            fidelity=node.fidelity,
            config={
                "priority": assessment.utility,
                "input_node_ids": input_node_ids,
                "input_artifact_ids": [
                    f"{source_id}:predictions"
                    for source_id in input_node_ids
                ],
                "diversity_assessment": assessment.to_dict(),
                "requested_strategy": getattr(
                    self, "ensemble_strategy", "auto"
                ),
                "planned_strategy": assessment.strategy,
                "planned_weights": list(assessment.weights),
                "manager_owned_merge": True,
                "raw_code_fusion": False,
            },
        )
        self.all_nodes[merge_id] = child
        node.children_ids.append(merge_id)
        self._persist_node(merge_id)
        self._trace_search(
            {
                "event": "merge_ensemble_scheduled",
                "node_id": merge_id,
                "execution_parent_id": node_id,
                "input_node_ids": input_node_ids,
                "assessment": assessment.to_dict(),
            }
        )

    def _record_artifact_validation(
        self,
        tech_record: dict,
        node_id: str,
        score: Any,
        status: str,
        reward: Any,
        fidelity: str,
        elapsed_seconds: Any = None,
    ) -> None:
        """Compatibility no-op: run evidence belongs only to the run root.

        Node results are already persisted in ``tree_state.json`` and
        ``search_trace.jsonl``.  Never attach task names, node IDs, scores, or
        timings to shared strategy cards under ``memory_pool/l2_store``.
        """
        return

    @staticmethod
    def _dependency_fallback_record(tech_record: dict, error: object) -> dict:
        """Preserve branch intent when an optional artifact cannot be installed."""
        fallback = copy.deepcopy(tech_record or {})
        card = fallback.get("model_card") or (
            fallback.get("candidate_artifact", {}).get("model_card")
        ) or {}
        artifact_id = card.get("artifact_id") or fallback.get("artifact_id")
        modality = fallback.get("modality") or "tabular"
        original_plan = fallback.get(
            "plan", f"Build a robust {modality} pipeline."
        )
        fallback["unavailable_artifact"] = {
            "artifact_id": artifact_id,
            "category": card.get("category") or fallback.get("category"),
            "reason": str(error),
            "dependencies": list(card.get("dependencies", [])),
        }
        for key in (
            "model_card",
            "candidate_artifact",
            "artifact_id",
            "category",
            "scope",
        ):
            fallback.pop(key, None)
        fallback["status"] = "dependency_fallback"
        fallback["plan"] = (
            "The selected optional artifact could not be installed. Implement a "
            "dependency-light, self-contained equivalent of the INTENDED model family/architecture using only libraries that "
            "are already importable in the selected interpreter. Do not import the "
            f"unavailable artifact {artifact_id or '<unknown>'!r} or any of its "
            f"unavailable dependencies {card.get('dependencies', [])!r}. Do NOT revert to or re-execute an already executed "
            f"parent model architecture. Preserve the original branch intent: {original_plan}"
        )
        return fallback

    def _spawn_follow_up_nodes(self, node: NodeState, node_id: str) -> None:
        """Create policy-approved virtual actions outside the scheduler."""
        if self.experiments_executed + 1 >= self.total_budget or not node.code:
            return
        self._ensure_search_services()
        result = node.result or {}
        score = result.get("score")
        validation = result.get("validation") or {}
        evidence = self._evidence_for_result(node)
        if evidence is not None:
            result["statistical_evidence"] = evidence.to_dict()
            pruning = self.pruning_policy.decide(evidence)
            result["pruning_decision"] = pruning.to_dict()
            if pruning.prune:
                self._trace_search(
                    {
                        "event": "statistical_prune",
                        "node_id": node_id,
                        "evidence": evidence.to_dict(),
                        "decision": pruning.to_dict(),
                    }
                )
                self._persist_node(node_id)
                self._persist_tree_state()
                return

        repaired_artifact = result.get("artifact_repair") or {}
        artifact_variant = (
            repaired_artifact
            if repaired_artifact.get("verified")
            else (node.config or {}).get("artifact_variant")
        )
        model_card = ((node.config or {}).get("technique_record") or {}).get(
            "model_card", {}
        )
        model_capabilities = (
            model_card.get("capabilities", {})
            if isinstance(model_card, dict)
            else {}
        )
        tunable_parameters_declared = (
            isinstance(model_capabilities, dict)
            and "tunable_parameters" in model_capabilities
        )
        tunable_parameters = (
            model_capabilities.get("tunable_parameters", [])
            if tunable_parameters_declared
            else []
        )
        fine_tune_depth = int((node.config or {}).get("fine_tune_depth", 0) or 0)
        tuning_evidence = evidence
        if node.operator == "tune":
            previous_node = self.all_nodes.get(
                (node.config or {}).get("base_node_id")
            )
            if previous_node and previous_node.result:
                tuning_evidence = self._evidence_for_result(
                    node, previous_node.result
                )
        tune_eligible = (
            tuning_evidence is not None
            and tuning_evidence.probability_material_improvement
            >= self.promotion_controller.boundary
            and fine_tune_depth
            < getattr(self, "max_fine_tune_rounds", 2)
        )
        if tunable_parameters_declared and not tunable_parameters:
            tune_eligible = False

        base_utility = (
            self.information_gain_strategy.utility(evidence)
            if evidence is not None
            else 0.0
        )
        operator_priorities = {
            "refine": base_utility,
            "diversify": base_utility,
        }
        promotion = (
            self.promotion_controller.decide(
                evidence, current_fidelity=node.fidelity
            )
            if evidence is not None and self.enable_multi_fidelity
            else None
        )
        if promotion is not None:
            result["promotion_decision"] = promotion.to_dict()
            if promotion.promote:
                operator_priorities["promote"] = promotion.utility
        parent_technique_record = (node.config or {}).get(
            "technique_record"
        ) or {}
        parent_artifact_id = parent_technique_record.get("artifact_id") or (
            parent_technique_record.get("model_card") or {}
        ).get("artifact_id")
        if tune_eligible:
            operator_priorities["tune"] = self.information_gain_strategy.utility(
                tuning_evidence
            )
        for operator, priority in operator_priorities.items():
            child_fidelity = (
                promotion.target_fidelity
                if operator == "promote" and promotion is not None
                else node.fidelity
            )
            is_fine_tune = operator == "tune"
            is_promotion = operator == "promote"
            tuning_context = None
            if is_fine_tune:
                model_family = self._model_family_for_node(node)
                history_context = self.tuning_coordinator.build_context(
                    task_name=self.task_name,
                    model_family=model_family,
                    tunable_parameters=tunable_parameters,
                    metric_name=self._resolved_metric_name(),
                    metric_direction=self.metric_direction,
                    dataset_fingerprint=validation.get(
                        "fold_assignment_sha256"
                    ),
                    modality=getattr(self, "modality", "tabular"),
                    problem_type=getattr(self, "problem_type", getattr(self, "task_type", "supervised")),
                    output_type=getattr(self, "output_type", "class_probabilities" if getattr(self, "task_type", "") == "classification" else "scalar_predictions"),
                    accelerator_class=getattr(self, "preferred_accelerator", "auto"),
                )
                tuning_context = {
                    "trigger": "posterior_material_improvement",
                    "parent_node_id": node_id,
                    "parent_score": score,
                    "parent_cv_std": validation.get(
                        "score_std", validation.get("cv_std", 0.0)
                    ),
                    "metric_direction": self.metric_direction,
                    "comparison_fidelity": node.fidelity,
                    "fine_tune_round": fine_tune_depth + 1,
                    "artifact_variant": artifact_variant,
                    "evaluation_mode": (
                        None
                        if operator == "diversify"
                        else (node.result or {}).get("evaluation_mode")
                    ),
                    "tunable_parameters_declared": tunable_parameters_declared,
                    "tunable_parameters": tunable_parameters,
                    "model_family": model_family,
                    "dataset_fingerprint": validation.get(
                        "fold_assignment_sha256"
                    ),
                    "statistical_evidence": (
                        tuning_evidence.to_dict()
                        if tuning_evidence is not None
                        else None
                    ),
                    **history_context,
                }
            new_id = self.get_new_node_id()
            child = NodeState(
                node_id=new_id,
                parent_id=node_id,
                node_type="technique",
                plan=f"Lazy {operator} slot; materialized only if selected.",
                operator=operator,
                fidelity=child_fidelity,
                config={
                    "base_node_id": node_id,
                    "base_code_path": node.code,
                    "priority": priority,
                    "priority_locked": is_fine_tune or is_promotion,
                    "lazy_proposal": True,
                    "materialized": False,
                    "fine_tune_triggered": is_fine_tune,
                    "preserve_parent_technique": is_promotion,
                    "fine_tune_depth": (
                        fine_tune_depth + 1 if is_fine_tune else fine_tune_depth
                    ),
                    "tuning_context": tuning_context,
                    "evaluation_mode": (
                        None
                        if operator == "diversify"
                        else (node.result or {}).get("evaluation_mode")
                    ),
                    "policy_evidence": (
                        evidence.to_dict() if evidence is not None else None
                    ),
                    "promotion_decision": (
                        promotion.to_dict() if is_promotion else None
                    ),
                    "artifact_variant": artifact_variant,
                    "locked_technique_record": (
                        copy.deepcopy((node.config or {}).get("technique_record"))
                        if is_fine_tune or is_promotion
                        else None
                    ),
                    "allowed_scopes": (
                        ["full_pipeline", "model_family"]
                        if operator == "diversify"
                        else ["full_pipeline", "model_family", "component"]
                    ),
                    "excluded_artifact_ids": (
                        [parent_artifact_id]
                        if operator == "diversify" and parent_artifact_id
                        else []
                    ),
                    "raw_code_fusion": False,
                },
            )
            self.all_nodes[new_id] = child
            node.children_ids.append(new_id)
            self._persist_node(new_id)
            print(
                f"ManagerAgent: Spawned virtual {operator} slot {new_id} at "
                f"{child_fidelity} fidelity"
            )
            if is_fine_tune:
                self._trace_search(
                    {
                        "event": "fine_tune_scheduled",
                        "node_id": new_id,
                        "parent_node_id": node_id,
                        "parent_score": score,
                        "fine_tune_round": fine_tune_depth + 1,
                        "reused_trial_count": len(
                            (tuning_context or {}).get("reused_trials", [])
                        ),
                    }
                )
            elif is_promotion:
                self._trace_search(
                    {
                        "event": "promotion_scheduled",
                        "node_id": new_id,
                        "parent_node_id": node_id,
                        "from_fidelity": node.fidelity,
                        "to_fidelity": child_fidelity,
                        "decision": promotion.to_dict(),
                    }
                )
        self._spawn_merge_ensemble_slot(node, node_id, evidence)
        self._persist_node(node_id)
        self._persist_tree_state()

    def _materialize_lazy_proposal(self, node: NodeState) -> None:
        """Spend one proposal-generation call only after the scheduler selects a slot."""
        config = dict(node.config or {})
        if not config.get("lazy_proposal") or config.get("materialized"):
            return
        if config.get("fine_tune_triggered"):
            context = config.get("tuning_context") or {}
            node.plan = (
                "Fine-tune the measured winning parent without changing its model family, "
                "features, preprocessing, folds, or output contract. Use the parent settings "
                "as a control, run a bounded pruned search over its existing hyperparameters, "
                "and apply early stopping. Trigger context: "
                + json.dumps(context, default=str)
            )
            config["proposal_name"] = "measured_parent_fine_tune"
            config["materialized"] = True
            node.config = config
            self._persist_node(node.node_id)
            return
        if config.get("preserve_parent_technique"):
            node.plan = (
                "Promote the measured parent unchanged to the scheduled fidelity. "
                "Preserve its model family, preprocessing, features, and output "
                "contract; modify only evaluation-fidelity resource settings."
            )
            config["proposal_name"] = "evidence_guided_promotion"
            config["materialized"] = True
            node.config = config
            self._persist_node(node.node_id)
            return
        base_node_id = config.get("base_node_id")
        parent = self.all_nodes.get(base_node_id)
        try:
            parent_code = Path(config["base_code_path"]).read_text(encoding="utf-8")
        except Exception:
            parent_code = ""
        memory_context = {
            "parent": self.global_memory.records.get(base_node_id, {}),
            "recent_experiments": list(self.global_memory.records.items())[-8:],
        }
        try:
            proposal = self.technique_agent.generate_follow_up_approach(
                operator=node.operator,
                task_description=self.task_description,
                parent_code=parent_code,
                parent_result=(parent.result if parent else {}) or {},
                global_memory_context=memory_context,
            )
        except Exception as exc:
            print(f"ManagerAgent WARNING: Lazy proposal generation failed: {exc}")
            fallback_plans = {
                "refine": "Refine the measured parent by changing only its weakest validated component.",
                "tune": "Tune the measured parent's existing model family with a compact pruned search.",
                "diversify": "Add a feasible complementary model while preserving the measured parent for blending.",
            }
            proposal = {
                "name": f"fallback_{node.operator}",
                "plan": fallback_plans[node.operator],
                "operator": node.operator,
                "priority": config.get("priority", 0.0),
            }
        node.plan = proposal["plan"]
        proposed_priority = proposal.get("priority", config.get("priority", 0.0))
        config["priority"] = (
            max(float(config.get("priority", 0.0)), float(proposed_priority))
            if config.get("priority_locked")
            else proposed_priority
        )
        config["proposal_name"] = proposal.get("name")
        config["materialized"] = True
        node.config = config
        self._persist_node(node.node_id)

    def _execute_ensemble_node(
        self, node: NodeState, selected_id: str
    ) -> bool:
        """Have ManagerAgent ensemble two measured node prediction artifacts."""
        self._ensure_search_services()
        node_dir = Path(self.run_root) / selected_id
        source_node_ids = list(
            (node.config or {}).get("input_node_ids") or []
        )
        res = self.aggregator_agent.merge_nodes(
            Path(self.run_root),
            source_node_ids,
            node_dir,
            metric_name=self._resolved_metric_name(),
            strategy=(node.config or {}).get("requested_strategy", "auto"),
        )
        if res is None:
            node.executed = True
            node.result = {
                "score": None,
                "status": "skipped_no_effect",
                "reward": None,
                "diagnostics": (
                    "No OOF-backed multi-model ensemble improved on the "
                    "strongest source node."
                ),
            }
            self._trace_search(
                {
                    "event": "merge_ensemble_skipped",
                    "node_id": selected_id,
                    "input_node_ids": source_node_ids,
                    "reason": "no_effective_multi_model_plan",
                }
            )
            self._persist_node(selected_id)
            self._persist_tree_state()
            return False
        score = float(res["score"])
        validation = res.get("validation") or {}
        validation["fidelity"] = node.fidelity
        if source_node_ids:
            source_validation = (
                self.all_nodes[source_node_ids[0]].result or {}
            ).get("validation") or {}
            for key in (
                "fold_assignment_sha256",
                "row_count",
                "source_row_count",
            ):
                if source_validation.get(key) is not None:
                    validation[key] = source_validation[key]
        cv_std = validation.get("cv_std", 0.0)
        raw_reward = self._score_to_reward(score, 0.0)
        reward = self._score_to_reward(score, cv_std)
        node.code = res.get("code_path")
        node.result = {
            "score": score,
            "status": "completed",
            "reward": reward,
            "raw_reward": raw_reward,
            "uncertainty_penalty": raw_reward - reward,
            "diagnostics": res.get("diagnostics"),
            "elapsed_seconds": res.get("elapsed_seconds"),
            "validation": validation,
            "oof_path": res.get("oof_path"),
            "prediction_bundle": res.get("prediction_bundle"),
            "model_bundle": res.get("model_bundle"),
            "compatibility_key": res.get("compatibility_key"),
            "merge": res.get("merge"),
        }
        node.executed = True
        self.global_memory.record_implementation(
            selected_id,
            {
                "node_id": selected_id,
                "operator": "merge_ensemble",
                "fidelity": node.fidelity,
                "input_node_ids": source_node_ids,
                "validation": validation,
                "reward": reward,
            },
            score,
            "completed",
        )
        self.scheduler.backpropagate(selected_id, reward, self.all_nodes)
        self._record_node_provenance(node)
        self._trace_search(
            {
                "event": "merge_ensemble_completed",
                "node_id": selected_id,
                "input_node_ids": source_node_ids,
                "score": score,
                "reward": reward,
                "fidelity": node.fidelity,
            }
        )
        self._persist_node(selected_id)
        self._persist_tree_state()
        print(
            f"ManagerAgent: Ensemble Node {selected_id} completed. "
            f"Score: {score:.5f} (Reward: {reward:.5f})"
        )
        return True

    def _execute_node(self, node, selected_id, root_id, l1_index, l1_path):
        """Executes a single node (technique or implementation). Extracted for try/except isolation."""
        if (
            node.node_type == "implementation"
            and node.operator == "merge_ensemble"
        ):
            return self._execute_ensemble_node(node, selected_id)
        if node.node_type == "technique":
            print(f"ManagerAgent: Running Technique Agent on {selected_id}...")

            self._materialize_lazy_proposal(node)

            if (node.config or {}).get("fine_tune_triggered") or (
                node.config or {}
            ).get("preserve_parent_technique"):
                # Tuning and promotion are locked to the measured parent model;
                # querying the pool here could silently replace the comparison.
                tech_record = copy.deepcopy(
                    (node.config or {}).get("locked_technique_record") or {}
                )
                tech_record["plan"] = node.plan
                if (node.config or {}).get("fine_tune_triggered"):
                    tech_record["fine_tune"] = True
                    tech_record["tuning_context"] = (node.config or {}).get(
                        "tuning_context"
                    )
                    tech_record.setdefault("status", "self_contained_fine_tune")
                else:
                    tech_record["promotion"] = True
                    tech_record.setdefault(
                        "status", "self_contained_promotion"
                    )
            else:
                # Pool additions from earlier nodes in this same run must be visible.
                if self.enable_executable_artifacts:
                    with open(l1_path, 'r', encoding='utf-8') as f:
                        l1_index = json.load(f)
                else:
                    l1_index = {}

                context = self.global_memory.get_default_context(
                    selected_id, self.all_nodes
                )
                try:
                    tech_record = self.technique_agent.run(
                        task_description=self.task_description,
                        branch_plan=node.plan,
                        global_memory_context=context,
                        l1_index=l1_index,
                        available_accelerators=set(self.available_accelerators),
                        preferred_accelerator=self.preferred_accelerator,
                        available_dependencies=set(self.allowed_dependencies),
                        allowed_scopes=set(
                            (node.config or {}).get("allowed_scopes", [])
                        ) or None,
                        excluded_artifact_ids=set(
                            (node.config or {}).get(
                                "excluded_artifact_ids", []
                            )
                        ),
                        task_spec=self.task_spec.to_dict(),
                        enable_executable_artifacts=(
                            self.enable_executable_artifacts
                        ),
                    )
                except Exception as exc:
                    # Planning is not an experiment. Preserve the branch intent and
                    # let ImplementationAgent attempt a dependency-light pipeline
                    # instead of exhausting the tree on a provider-side LLM error.
                    tech_record = {
                        "status": "self_contained_fallback",
                        "plan": (
                            "Technique planning failed before selecting an artifact. "
                            "Implement a robust self-contained version of the branch "
                            "using only already importable project libraries. Preserve "
                            f"this branch intent: {node.plan}"
                        ),
                        "planning_error": str(exc),
                    }
                    self._trace_search(
                        {
                            "event": "technique_planning_fallback",
                            "node_id": selected_id,
                            "reason": str(exc),
                        }
                    )
                    print(
                        "ManagerAgent WARNING: Technique planning failed; "
                        "continuing with a self-contained implementation fallback: "
                        f"{exc}"
                    )
            planning_config = dict(node.config or {})
            artifact_variant = planning_config.get("artifact_variant")
            selected_card = tech_record.get("model_card") or (
                tech_record.get("candidate_artifact", {}).get("model_card")
            )
            selected_artifact_id = (
                selected_card.get("artifact_id")
                if isinstance(selected_card, dict)
                else None
            )
            if (
                artifact_variant
                and selected_artifact_id
                and selected_artifact_id
                != artifact_variant.get("artifact_id")
            ):
                # This branch selected a different artifact, so evidence belongs
                # to that new artifact rather than the inherited local variant.
                artifact_variant = None
            planning_config["artifact_variant"] = artifact_variant
            planning_config["technique_record"] = tech_record
            node.config = planning_config
            self._persist_node(selected_id)
            
            # Pre-allocate child Implementation node ID and create its directory
            child_id = self.get_new_node_id()
            node_dir = self.run_root / child_id
            node_dir.mkdir(parents=True, exist_ok=True)
            child_node = NodeState(
                node_id=child_id,
                parent_id=selected_id,
                node_type="implementation",
                operator=node.operator,
                fidelity=node.fidelity,
                config={
                    "technique_record": tech_record,
                    "base_node_id": planning_config.get("base_node_id"),
                    "base_code_path": planning_config.get("base_code_path"),
                    "priority": planning_config.get("priority", 0.0),
                    "fine_tune_triggered": planning_config.get(
                        "fine_tune_triggered", False
                    ),
                    "preserve_parent_technique": planning_config.get(
                        "preserve_parent_technique", False
                    ),
                    "fine_tune_depth": planning_config.get("fine_tune_depth", 0),
                    "tuning_context": planning_config.get("tuning_context"),
                    "policy_evidence": planning_config.get("policy_evidence"),
                    "promotion_decision": planning_config.get(
                        "promotion_decision"
                    ),
                    "artifact_variant": planning_config.get("artifact_variant"),
                },
            )
            self.all_nodes[child_id] = child_node
            node.children_ids.append(child_id)
            self._persist_node(child_id)
            
            # If pool miss, dynamically build and verify locally in child's node_dir!
            if tech_record.get("status") == "pool_miss":
                print("ManagerAgent: Bootstrapping new technique from web search outline (local build)...")
                builder = L2Builder(
                    project_root=self.project_root,
                    model_name=self.model_name,
                    venv_path=self.venv_path,
                    preferred_accelerator=self.preferred_accelerator,
                )
                raw_outline = tech_record.get("raw_outline", "")
                
                # Build locally (commit=False, target_dir=node_dir)
                success, category, artifact_id, model_card = builder.build_from_source(
                    "web_search_dynamic", raw_outline, commit=False, target_dir=node_dir
                )
                
                if success:
                    print(f"ManagerAgent: Successfully verified local artifact {artifact_id} in category '{category}'!")
                    tech_record["artifact_id"] = artifact_id
                    tech_record["category"] = category
                    tech_record["model_card"] = model_card
                    tech_record["plan"] = f"Import and use local bootstrapped artifact {artifact_id} from category {category}."
                    tech_record["status"] = "local_verified"

                    # Keep dynamically generated artifacts node-local. A task
                    # execution must never mutate the shared global pool.
                else:
                    tech_record["status"] = "bootstrap_failed"
                    tech_record["candidate_artifact"] = {
                        "category": category,
                        "artifact_id": artifact_id,
                        "model_card": model_card,
                    }
                    tech_record["plan"] = (
                        "Preserve the original experimental intent while omitting only the "
                        "unavailable artifact. Implement the closest feasible subset of this "
                        f"plan: {node.plan}. The failed artifact was "
                        f"{artifact_id or '<unknown>'}."
                    )
                    print(
                        "ManagerAgent WARNING: Web-derived artifact failed verification; "
                        "preserved it in the node directory and will use a self-contained fallback."
                    )

            # Failed candidates are diagnostic records only. They must not
            # constrain the self-contained recovery implementation.
            feasibility_card = tech_record.get("model_card")
            # Accelerator libraries may not exist until dependency setup runs.
            # Check static RAM constraints now and accelerator feasibility
            # after installing into and re-probing the selected interpreter.
            feasibility_reason = self._feasibility_reason(
                feasibility_card, check_accelerator=False
            )
            compatibility_reason = (
                None
                if (node.config or {}).get("fine_tune_triggered")
                else self._operator_compatibility_reason(
                    feasibility_card, node.operator
                )
            )
            feasibility_reason = feasibility_reason or compatibility_reason
            if feasibility_reason:
                tech_record["prior_status"] = tech_record.get("status")
                skip_status = "incompatible" if compatibility_reason else "infeasible"
                tech_record["feasibility_status"] = skip_status
                tech_record["status"] = skip_status
                tech_record["feasibility_reason"] = feasibility_reason
                child_node.executed = True
                child_node.result = {
                    "score": None,
                    "status": f"skipped_{skip_status}",
                    "reward": None,
                    "diagnostics": feasibility_reason,
                }
                if node.parent_id == root_id:
                    self.initial_fanout = max(0, self.initial_fanout - 1)
                    self.scheduler.set_warmup_budget(self.initial_fanout)
                print(
                    f"ManagerAgent: Skipping {child_id} before experiment budget: "
                    f"{feasibility_reason}"
                )
                self._trace_search(
                    {
                        "event": "skipped_infeasible",
                        "technique_node_id": selected_id,
                        "implementation_node_id": child_id,
                        "reason": feasibility_reason,
                        "verification_status": tech_record.get("prior_status"),
                    }
                )

            planning_config["technique_record"] = tech_record
            node.config = planning_config
            child_node.config["technique_record"] = tech_record
            
            # Record to global memory
            self.global_memory.record_technique(selected_id, tech_record.get("plan", ""), "succeeded")
            
            # Mark technique node as executed
            node.executed = True
            
            self._persist_node(selected_id)
            self._persist_node(child_id)
            self._persist_tree_state()
            if feasibility_reason:
                return False
            print(f"ManagerAgent: Technique Node {selected_id} resolved. Spawned Implementation Node {child_id}")
            return False
            
        elif node.node_type == "implementation":
            print(f"ManagerAgent: Running Implementation Agent on {selected_id}...")
            node.config = dict(node.config or {})
            precomputed_result = node.config.pop(
                "_precomputed_result", None
            )
            tech_record = node.config.get("technique_record", {})
            
            # Install dependencies via Setup Agent
            model_card = tech_record.get("model_card")
            try:
                requirements_file = self.project_root / "requirements.txt"
                if model_card:
                    # Every selected artifact, including a verified pool hit, is
                    # resolved to the exact human-controlled project requirement.
                    # This prevents a bare card dependency such as `torch` from
                    # accepting or installing an incompatible arbitrary version.
                    self.setup_agent.install_allowlisted_dependencies(
                        [model_card], requirements_file
                    )
            except Exception as exc:
                if node.operator in {"tune", "promote"} or (
                    node.config or {}
                ).get("fine_tune_triggered") or (
                    node.config or {}
                ).get("preserve_parent_technique"):
                    # Tuning and promotion are model-locked; replacing the
                    # artifact would invalidate the comparison.
                    node.executed = True
                    node.result = {
                        "score": None,
                        "status": "skipped_dependency_setup",
                        "reward": None,
                        "diagnostics": str(exc),
                    }
                    self._trace_search(
                        {
                            "event": "skipped_dependency_setup",
                            "node_id": selected_id,
                            "reason": str(exc),
                        }
                    )
                    self._persist_node(selected_id)
                    self._persist_tree_state()
                    print(
                        f"ManagerAgent: Skipping locked {node.operator} node {selected_id}; "
                        f"dependency setup failed: {exc}"
                    )
                    return False

                # For ordinary branches, an unavailable optional package should
                # not exhaust the tree. Remove the unusable artifact contract and
                # let the implementation agent attempt the closest core-library
                # equivalent as a normal, budgeted experiment.
                tech_record = self._dependency_fallback_record(tech_record, exc)
                node.config = dict(node.config or {})
                node.config["technique_record"] = tech_record
                node.config["artifact_variant"] = None
                model_card = None
                self._trace_search(
                    {
                        "event": "dependency_fallback",
                        "node_id": selected_id,
                        "reason": str(exc),
                        "unavailable_artifact": tech_record.get(
                            "unavailable_artifact"
                        ),
                    }
                )
                self._persist_node(selected_id)
                self._persist_tree_state()
                print(
                    f"ManagerAgent WARNING: Dependency setup failed for {selected_id}; "
                    "continuing with a dependency-light self-contained fallback: "
                    f"{exc}"
                )

            # Installing Torch or another backend can reveal CUDA/MPS support that
            # was invisible when the manager started. Refresh before choosing the
            # node device, then apply accelerator-only feasibility without charging
            # the experiment budget on failure.
            self._refresh_accelerator_state()
            accelerator_reason = self._feasibility_reason(model_card)
            if accelerator_reason:
                node.executed = True
                node.result = {
                    "score": None,
                    "status": "skipped_infeasible",
                    "reward": None,
                    "diagnostics": accelerator_reason,
                }
                self._trace_search(
                    {
                        "event": "skipped_infeasible_after_dependency_setup",
                        "node_id": selected_id,
                        "reason": accelerator_reason,
                    }
                )
                self._persist_node(selected_id)
                self._persist_tree_state()
                print(
                    f"ManagerAgent: Skipping {selected_id} before experiment budget; "
                    f"post-setup feasibility failed: {accelerator_reason}"
                )
                return False
                
            # Run implementation script in its own run folder
            node_dir = self.run_root / selected_id
            node_dir.mkdir(parents=True, exist_ok=True)
            if precomputed_result is None:
                self.implementation_attempts = (
                    getattr(self, "implementation_attempts", 0) + 1
                )
                self._trace_search(
                    {
                        "event": "implementation_attempt_started",
                        "node_id": selected_id,
                        "implementation_attempt": self.implementation_attempts,
                        "fidelity": node.fidelity,
                    }
                )
                res = self._run_implementation_payload(node)
            else:
                res = precomputed_result
            
            # Bug 1 fix: Handle execution failures properly
            score = res.get("score")  # Will be None on failure
            status = res.get("status", "completed")
            repaired_variant = res.get("artifact_repair") or {}
            artifact_variant = (
                repaired_variant
                if repaired_variant.get("verified")
                else (node.config or {}).get("artifact_variant")
            )
            if artifact_variant:
                node.config = dict(node.config or {})
                node.config["artifact_variant"] = artifact_variant
            
            # Record result to NodeState
            node.code = res.get("code_path")
            validation = res.get("validation", {})
            cv_std = (
                validation.get(
                    "score_std", validation.get("cv_std", 0.0)
                )
                if validation
                else 0.0
            )
            raw_reward = (
                self._score_to_reward(score, 0.0) if score is not None else -1.0
            )
            reward = (
                self._score_to_reward(score, cv_std) if score is not None else -1.0
            )
            node.result = {
                "score": score,
                "status": status,
                "reward": reward,
                "raw_reward": raw_reward,
                "uncertainty_penalty": raw_reward - reward,
                "diagnostics": res.get("diagnostics"),
                "elapsed_seconds": res.get("elapsed_seconds"),
                "accelerator": res.get(
                    "accelerator", self.preferred_accelerator
                ),
                "validation": validation,
                "oof_path": res.get("oof_path"),
                "validation_path": res.get("validation_path"),
                "evaluation_mode": res.get("evaluation_mode"),
                "evaluation_policy": res.get("evaluation_policy"),
                "error_analysis": res.get("error_analysis"),
                "error_analysis_path": res.get("error_analysis_path"),
                "duplicate_of": res.get("duplicate_of"),
                "code_fingerprint": res.get("code_fingerprint"),
                "prediction_bundle": res.get("prediction_bundle"),
                "model_bundle": res.get("model_bundle"),
                "compatibility_key": res.get("compatibility_key"),
                "tuning": res.get("tuning"),
                "artifact_repair": res.get("artifact_repair"),
                "artifact_variant": artifact_variant,
                "implementation_families": res.get(
                    "implementation_families", []
                ),
            }
            node.executed = True

            if status == "skipped_duplicate_pre_execution":
                self.scheduler.backpropagate(
                    selected_id, -1.0, self.all_nodes
                )
                self.global_memory.record_implementation(
                    selected_id,
                    {
                        "node_id": selected_id,
                        "duplicate_of": res.get("duplicate_of"),
                        "code_fingerprint": res.get("code_fingerprint"),
                    },
                    0.0,
                    status,
                )
                self._trace_search(
                    {
                        "event": "duplicate_pruned_pre_execution",
                        "node_id": selected_id,
                        "duplicate_of": res.get("duplicate_of"),
                        "code_fingerprint": res.get("code_fingerprint"),
                    }
                )
                if node.operator == "root":
                    self._promote_backup_approach(root_id)
                self._persist_node(selected_id)
                self._persist_tree_state()
                print(
                    "ManagerAgent: Pruned duplicate implementation "
                    f"{selected_id} before training; matches "
                    f"{res.get('duplicate_of')}."
                )
                return False
            
            if status == "failed":
                print(f"ManagerAgent: Implementation Node {selected_id} FAILED. No score produced.")
                # Record failure to global memory
                self.global_memory.record_implementation(selected_id, {"node_id": selected_id}, 0.0, "failed")
                # Backpropagate zero reward
                self.scheduler.backpropagate(selected_id, -1.0, self.all_nodes)
                if node.operator == "root":
                    replacement_id = self._promote_backup_approach(root_id)
                    if replacement_id:
                        print(
                            "ManagerAgent: Promoted a backup approach so the "
                            "remaining experiment budget can still be used."
                        )
                self._persist_node(selected_id)
                self._persist_tree_state()
                # Do NOT spawn follow-up technique nodes from failed implementations
                # Broken generated code is a technical failure, not a completed
                # scientific experiment. A finite backup list plus action_guard
                # bounds recovery without consuming the user's search budget.
                return False

            no_effect_reason = self._no_effect_reason(node, node_dir)
            deduplicated_outputs = self._deduplicate_node_outputs(node, node_dir)
            if no_effect_reason:
                status = "no_effect"
                reward = min(reward, -0.10)
                node.result.update(
                    {
                        "status": status,
                        "reward": reward,
                        "uncertainty_penalty": raw_reward - reward,
                        "no_effect_reason": no_effect_reason,
                        "deduplicated_outputs": deduplicated_outputs,
                    }
                )
                self.global_memory.record_implementation(
                    selected_id,
                    {"node_id": selected_id, "reason": no_effect_reason},
                    score,
                    status,
                )
                self.scheduler.backpropagate(selected_id, reward, self.all_nodes)
                self._trace_search(
                    {
                        "event": "no_effect",
                        "node_id": selected_id,
                        "reason": no_effect_reason,
                    }
                )
                self._persist_node(selected_id)
                self._persist_tree_state()
                print(f"ManagerAgent: Implementation Node {selected_id} had no new effect.")
                return True

            if deduplicated_outputs:
                node.result["deduplicated_outputs"] = deduplicated_outputs

            # Record to global memory
            self.global_memory.record_implementation(
                selected_id,
                {
                    "node_id": selected_id,
                    "operator": node.operator,
                    "fidelity": node.fidelity,
                    "base_node_id": (node.config or {}).get("base_node_id"),
                    "validation": res.get("validation", {}),
                    "elapsed_seconds": res.get("elapsed_seconds"),
                    "reward": reward,
                    "tuning": res.get("tuning"),
                    "artifact_repair": res.get("artifact_repair"),
                    "artifact_variant": artifact_variant,
                    "implementation_families": res.get(
                        "implementation_families", []
                    ),
                },
                score,
                "completed",
            )
            
            # Normalize reward for UCB1 backpropagation.
            self.scheduler.backpropagate(selected_id, reward, self.all_nodes)
            self._record_tuning_history(node)
            self._record_node_provenance(node)
            self._trace_search(
                {
                    "event": "experiment_completed",
                    "node_id": selected_id,
                    "score": score,
                    "reward": reward,
                    "fidelity": node.fidelity,
                    "tuning": res.get("tuning"),
                    "artifact_repair": res.get("artifact_repair"),
                    "artifact_variant": artifact_variant,
                    "implementation_families": res.get(
                        "implementation_families", []
                    ),
                }
            )
            self._persist_node(selected_id)
            self._persist_tree_state()
            print(
                f"ManagerAgent: Implementation Node {selected_id} completed. "
                f"Score: {score:.5f} (Reward: {reward:.5f}, Fidelity: {node.fidelity})"
            )
            self._spawn_follow_up_nodes(node, selected_id)
            return True

    def generate_final_submission(self, best_node_id: str):
        """Validate, optionally ensemble, and persist the final submission."""
        self.best_node_id = best_node_id or None
        self.final_submission_status = "failed"
        self.final_submission_validation = None

        def fail(message: str) -> bool:
            print(message)
            self.final_submission_status = "failed"
            try:
                self._persist_tree_state()
            except Exception:
                pass
            return False

        if not best_node_id:
            return fail(
                "ManagerAgent: No best node found to generate final submission."
            )

        best_node_dir = self.run_root / best_node_id
        generated_sub_path = best_node_dir / "submission" / "submission.csv"
        if not generated_sub_path.exists():
            return fail(
                "ManagerAgent WARNING: Generated submission file not found at "
                f"{generated_sub_path}"
            )

        # Ensemble only candidates evaluated at the same fidelity as the selected
        # best node. This avoids mixing cheap screening predictions with full runs.
        best_fidelity = self.all_nodes[best_node_id].fidelity
        ensemble_candidates = [
            {
                "node_id": node_id,
                "score": state.result.get("score"),
            }
            for node_id, state in self.all_nodes.items()
            if state.node_type == "implementation"
            and state.fidelity == best_fidelity
            and state.result
            and state.result.get("score") is not None
            and state.result.get("status") == "completed"
            and (self.run_root / node_id / "submission" / "submission.csv").is_file()
        ]
        ensemble_path = self.run_root / "ensemble_submission.csv"
        selected_ensemble_nodes = self.aggregator_agent.aggregate_ranked_candidates(
            self.run_root,
            ensemble_candidates,
            ensemble_path,
            maximize=self.metric_direction == "maximize",
            top_k=self.ensemble_top_k,
            strategy=self.ensemble_strategy,
            metric_name=self.metric_name,
        )
        if selected_ensemble_nodes:
            generated_sub_path = ensemble_path
            ensemble_manifest = dict(
                self.aggregator_agent.last_ensemble_manifest
            )
            ensemble_manifest.update(
                {
                    "fidelity": best_fidelity,
                    "node_ids": selected_ensemble_nodes,
                }
            )
            with open(self.run_root / "ensemble_manifest.json", "w", encoding="utf-8") as f:
                json.dump(ensemble_manifest, f, indent=2)

        # The task directory is read-only; final predictions belong to the run.
        run_output_path = self.run_root / "submission.csv"
        task_spec = getattr(self, "task_spec", None)
        if task_spec is None:
            classification_metric = self.metric_name in {
                "accuracy",
                "balanced_accuracy",
                "log_loss",
                "cross_entropy",
            } or "auc" in str(self.metric_name)
            task_spec = {
                "problem_type": (
                    "classification" if classification_metric else "regression"
                ),
                "output": {
                    "type": (
                        "class_probabilities"
                        if classification_metric
                        else "continuous"
                    )
                },
                "inputs": {},
            }
        try:
            final_frame, validation = validate_submission_file(
                generated_sub_path,
                task_dir=self.task_dir,
                task_spec=task_spec,
                normalize_probabilities=True,
            )
            final_frame.to_csv(run_output_path, index=False)
            self.final_submission_status = "completed"
            self.final_submission_validation = {
                **validation,
                "source_path": str(generated_sub_path),
                "output_path": str(run_output_path),
                "ensemble_node_ids": selected_ensemble_nodes,
            }
            try:
                self._persist_tree_state()
            except Exception:
                pass
            print(
                "ManagerAgent: Validated final submission saved to "
                f"{run_output_path}"
            )
            return True
        except Exception as exc:
            return fail(
                "ManagerAgent ERROR: Refusing invalid final submission: "
                f"{exc}"
            )

    def save_tree_image(self, output_path: Path):
        """
        Generates and saves a large, fully readable image of the method exploration tree.
        Box sizes dynamically scale to fit their text content.
        """
        if not self.all_nodes:
            print("ManagerAgent: No nodes to plot.")
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import textwrap

        WRAP_WIDTH = 35          # characters per line before wrapping
        MAX_DESC_CHARS = 200     # truncate descriptions longer than this
        LINE_HEIGHT = 0.35       # vertical space per wrapped line (in data coords)
        BOX_WIDTH = 4.5          # fixed width for all boxes
        BOX_PAD_V = 0.5          # vertical padding inside box (title area + bottom)
        LEAF_H_SPACE = 5.5       # horizontal space allocated to each leaf node
        MIN_V_GAP = 3.5          # minimum vertical gap between depth levels
        TITLE_FONT = 10
        DESC_FONT = 9

        # --- Helper: prepare display text for a node and compute wrapped line count ---
        def get_node_display(node_id, node):
            if node_id == "root":
                title = "Search Root"
                desc = "Virtual orchestration node"
                color, border = "#E0E0E0", "#616161"
            elif node.node_type == "technique":
                title = f"{node_id} (Technique)"
                if not node.executed:
                    desc = f"PENDING — not executed within budget\n{node.plan or ''}"
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
                        desc = node.plan or tech_record.get("plan", "Technique completed")
                    color, border = "#E3F2FD", "#1565C0"
            else:  # implementation
                res = node.result or {}
                score = res.get("score")
                status = res.get("status", "completed")
                title = f"{node_id} (Implementation)"
                if not node.executed:
                    desc = "PENDING — not executed within budget"
                    color, border = "#FFF8E1", "#F9A825"
                elif status == "failed" or score is None:
                    desc = "FAILED / Crashed"
                    color, border = "#FFEBEE", "#C62828"
                else:
                    tech_record = node.config.get("technique_record", {}) if node.config else {}
                    artifact_id = tech_record.get("artifact_id")
                    if artifact_id:
                        desc = (
                            f"{node.operator or 'root'} / {node.fidelity}\n"
                            f"Use: {artifact_id}\nScore: {score:.5f}"
                        )
                    elif tech_record.get("status") == "bootstrap_failed":
                        desc = (
                            f"{node.operator or 'root'} / {node.fidelity}\n"
                            f"Use: Self-contained fallback\nScore: {score:.5f}"
                        )
                    else:
                        desc = (
                            f"{node.operator or 'root'} / {node.fidelity}\n"
                            f"Score: {score:.5f}"
                        )
                    color, border = "#E8F5E9", "#2E7D32"

            # Truncate very long descriptions
            if len(desc) > MAX_DESC_CHARS:
                desc = desc[:MAX_DESC_CHARS] + "..."

            desc_lines = desc.split("\n")
            wrapped_lines = []
            for d_line in desc_lines:
                wrapped = textwrap.wrap(d_line, width=WRAP_WIDTH)
                if wrapped:
                    wrapped_lines.extend(wrapped)
                else:
                    wrapped_lines.append("")
            return title, wrapped_lines, color, border

        # --- Compute how tall each node's box is ---
        node_heights = {}
        for nid, node in self.all_nodes.items():
            _, wrapped_lines, _, _ = get_node_display(nid, node)
            node_heights[nid] = BOX_PAD_V + len(wrapped_lines) * LINE_HEIGHT

        # --- Layout: assign (x, y) coordinates ---
        def compute_layout(node_id, depth=0, x_left=0.0):
            node = self.all_nodes[node_id]
            valid_children = [cid for cid in node.children_ids if cid in self.all_nodes]

            if not valid_children:
                # Leaf node
                y = -depth * MIN_V_GAP
                return {node_id: (x_left, y)}, LEAF_H_SPACE

            coords = {}
            current_x = x_left
            child_widths = []
            for child_id in valid_children:
                child_coords, child_width = compute_layout(child_id, depth + 1, current_x)
                coords.update(child_coords)
                child_widths.append(child_width)
                current_x += child_width

            # Center parent over its children
            child_xs = [coords[cid][0] for cid in valid_children]
            x = sum(child_xs) / len(child_xs)
            y = -depth * MIN_V_GAP
            coords[node_id] = (x, y)
            return coords, sum(child_widths)

        try:
            # Find root
            root_id = "root"
            if root_id not in self.all_nodes:
                roots = [nid for nid, n in self.all_nodes.items() if n.parent_id is None]
                if not roots:
                    print("ManagerAgent WARNING: No root node found for tree visualization.")
                    return
                root_id = roots[0]

            coords, _ = compute_layout(root_id)

            # Figure sizing
            xs = [c[0] for c in coords.values()]
            ys = [c[1] for c in coords.values()]
            x_span = (max(xs) - min(xs)) if xs else 0
            y_span = (max(ys) - min(ys)) if ys else 0

            fig_width = max(20, x_span + 8)
            fig_height = max(12, y_span + 6)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
            ax.axis('off')

            # 1. Draw edges
            for nid, (x, y) in coords.items():
                node = self.all_nodes[nid]
                h = node_heights[nid]
                for child_id in node.children_ids:
                    if child_id in coords:
                        cx, cy = coords[child_id]
                        ch = node_heights[child_id]
                        # Line from bottom of parent box to top of child box
                        ax.plot(
                            [x, cx],
                            [y - h / 2, cy + ch / 2],
                            color='#9E9E9E', linestyle='-', linewidth=1.5, zorder=1
                        )

            # 2. Draw nodes
            for nid, (x, y) in coords.items():
                node = self.all_nodes[nid]
                title, wrapped_lines, color, border = get_node_display(nid, node)
                h = node_heights[nid]

                # Draw box
                rect = patches.FancyBboxPatch(
                    (x - BOX_WIDTH / 2, y - h / 2),
                    BOX_WIDTH, h,
                    boxstyle="round,pad=0.1",
                    linewidth=2.0,
                    edgecolor=border,
                    facecolor=color,
                    zorder=2
                )
                ax.add_patch(rect)

                # Title text (top of box)
                title_y = y + h / 2 - 0.3
                ax.text(
                    x, title_y, title,
                    ha='center', va='center',
                    fontsize=TITLE_FONT, fontweight='bold',
                    color='#212121', zorder=3
                )

                # Description text (below title, one line at a time)
                for i, line in enumerate(wrapped_lines):
                    line_y = title_y - 0.35 - i * LINE_HEIGHT
                    ax.text(
                        x, line_y, line,
                        ha='center', va='center',
                        fontsize=DESC_FONT,
                        color='#424242', zorder=3
                    )

            # Axis limits
            ax.set_xlim(min(xs) - BOX_WIDTH, max(xs) + BOX_WIDTH)
            max_h = max(node_heights.values()) if node_heights else 1
            ax.set_ylim(min(ys) - max_h - 1, max(ys) + max_h + 1)

            plt.title(
                f"Method Exploration Tree — {self.task_name}",
                fontsize=16, fontweight='bold', pad=25
            )
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            print(f"ManagerAgent: Saved method tree image to {output_path}")
        except Exception as e:
            print(f"ManagerAgent ERROR: Failed to generate method tree image: {e}")
            import traceback
            traceback.print_exc()
