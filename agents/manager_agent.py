import os
import json
import math
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from tree.node import NodeState
from tree.scheduler import UCB1Scheduler
from tree.global_memory import GlobalMemory
from .initial_agent import InitialAgent
from .technique_agent import TechniqueAgent
from .implementation_agent import ImplementationAgent
from .aggregator_agent import AggregatorAgent
from .setup_agent import SetupAgent
from memory_pool.builder.l2_builder import L2Builder
from memory_pool.query_tool import normalize_resource_profile
from runtime_utils import (
    detect_available_accelerators,
    select_preferred_accelerator,
    validate_path_component,
)

class ManagerAgent:
    def __init__(self, task_name: str, total_budget: int = 10, venv_path: str = "./.venv/bin/python", baseline_score: float = None, model_name: str = None, run_suffix: str = None):
        self.task_name = validate_path_component(task_name, "task_name")
        if run_suffix is not None:
            run_suffix = validate_path_component(run_suffix, "run_suffix")
        if not isinstance(total_budget, int) or isinstance(total_budget, bool) or total_budget < 1:
            raise ValueError(f"total_budget must be a positive integer, got {total_budget!r}")
        self.total_budget = total_budget
        if baseline_score is not None:
            baseline_score = float(baseline_score)
            if not math.isfinite(baseline_score):
                raise ValueError("baseline_score must be finite")
        self.baseline_score = baseline_score
        self.model_name = model_name
        
        # Directories
        self.project_root = Path(__file__).resolve().parent.parent
        
        import sys
        import subprocess
        resolved_venv = Path(venv_path)
        if not resolved_venv.is_absolute():
            resolved_venv = self.project_root / resolved_venv
        resolved_path = str(resolved_venv.resolve())
        
        # Check if the resolved venv python is fully functional
        use_fallback = True
        if Path(resolved_path).exists():
            try:
                res = subprocess.run([resolved_path, "-c", "import sys; print('ok')"], capture_output=True, text=True, timeout=5)
                if res.returncode == 0 and "ok" in res.stdout:
                    use_fallback = False
            except Exception:
                pass
                
        if use_fallback:
            print(f"ManagerAgent WARNING: Specified python path '{resolved_path}' is invalid or non-functional. Falling back to active running interpreter: {sys.executable}")
            self.venv_path = sys.executable
        else:
            self.venv_path = resolved_path
        
        self.task_dir = self.project_root / "tasks" / self.task_name
        if not self.task_dir.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {self.task_dir}")
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
        
        # State tracker
        self.all_nodes: Dict[str, NodeState] = {}
        self.node_counter = 0
        self.experiments_executed = 0

        # Load task config for metric direction and timeout
        self.metric_direction = "maximize"
        self.metric_name = "score"
        self.subprocess_timeout = 300
        self.enable_multi_fidelity = True
        self.ensemble_top_k = 3
        self.ensemble_strategy = "rank_average"
        self.uncertainty_weight = 1.0
        self.max_l1_categories = 8
        self.max_artifact_candidates = 5
        self.enforce_evaluation_contract = True
        self.available_accelerators = detect_available_accelerators(self.venv_path)
        self.preferred_accelerator = "cpu"
        self.available_ram_gb = self._available_ram_gb()

        accelerator_preference = "auto"
        config_file = self.task_dir / "task_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    task_config = json.load(f)
                self.metric_direction = task_config.get("metric_direction", "maximize")
                self.metric_name = task_config.get("metric_name", "score")
                self.subprocess_timeout = task_config.get("subprocess_timeout", 300)
                self.enable_multi_fidelity = bool(
                    task_config.get("enable_multi_fidelity", True)
                )
                self.ensemble_top_k = max(1, int(task_config.get("ensemble_top_k", 3)))
                self.ensemble_strategy = task_config.get(
                    "ensemble_strategy", "rank_average"
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
                        self.available_accelerators &= allowed_accelerators
                        self.available_accelerators.add("cpu")
                    accelerator_preference = resource_limits.get(
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
        self.preferred_accelerator = select_preferred_accelerator(
            self.available_accelerators, accelerator_preference
        )
        if self.metric_direction not in {"maximize", "minimize"}:
            raise ValueError(
                f"task_config metric_direction must be 'maximize' or 'minimize', "
                f"got {self.metric_direction!r}"
            )
        if (
            not isinstance(self.subprocess_timeout, (int, float))
            or isinstance(self.subprocess_timeout, bool)
            or not math.isfinite(self.subprocess_timeout)
            or self.subprocess_timeout <= 0
        ):
            raise ValueError(
                f"task_config subprocess_timeout must be positive and finite, "
                f"got {self.subprocess_timeout!r}"
            )
        if self.ensemble_strategy not in {"average", "rank_average"}:
            raise ValueError(
                "task_config ensemble_strategy must be 'average' or 'rank_average'"
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
        self.task_description = f"Tabular ML task: {task_name}"
        desc_file = self.task_dir / "task_description.md"
        if desc_file.exists():
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    self.task_description = f.read().strip()
            except Exception:
                pass
                
        print(f"ManagerAgent initialized: direction={self.metric_direction}, timeout={self.subprocess_timeout}, baseline_score={self.baseline_score}")

    @staticmethod
    def _available_ram_gb() -> float:
        try:
            return (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            ) / (1024 ** 3)
        except (AttributeError, OSError, ValueError):
            return 0.0

    @staticmethod
    def _initial_fanout_for_budget(total_budget: int) -> int:
        return min(3, max(1, int(total_budget) // 3))

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
        """Detect descendants whose generated predictions exactly reproduce a parent."""
        base_node_id = (node.config or {}).get("base_node_id")
        if not base_node_id:
            return None
        parent_dir = self.run_root / base_node_id
        pairs = (
            (parent_dir / "oof_predictions.csv", node_dir / "oof_predictions.csv", "OOF"),
            (
                parent_dir / "submission" / "submission.csv",
                node_dir / "submission" / "submission.csv",
                "submission",
            ),
        )
        matches = [
            label
            for parent_path, child_path, label in pairs
            if self._sha256_file(parent_path)
            and self._sha256_file(parent_path) == self._sha256_file(child_path)
        ]
        if "OOF" in matches and "submission" in matches:
            return "child OOF predictions and submission are byte-identical to the measured parent"
        try:
            import numpy as np
            import pandas as pd

            def same_predictions(parent_path: Path, child_path: Path) -> bool:
                if not parent_path.is_file() or not child_path.is_file():
                    return False
                parent_frame, child_frame = pd.read_csv(parent_path), pd.read_csv(child_path)
                if list(parent_frame.columns) != list(child_frame.columns) or len(parent_frame) != len(child_frame):
                    return False
                id_column = parent_frame.columns[0]
                if set(parent_frame[id_column]) != set(child_frame[id_column]):
                    return False
                prediction_columns = list(parent_frame.columns[1:])
                left = parent_frame.set_index(id_column).sort_index()[prediction_columns]
                right = child_frame.set_index(id_column).sort_index()[prediction_columns]
                return bool(
                    np.allclose(
                        left.to_numpy(dtype=float),
                        right.to_numpy(dtype=float),
                        rtol=1e-12,
                        atol=1e-12,
                    )
                )

            if all(same_predictions(parent_path, child_path) for parent_path, child_path, _ in pairs):
                return "child OOF predictions and submission are numerically identical to the measured parent"
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
        if node.node_type == "technique":
            with open(node_dir / "technique_plan.md", "w", encoding="utf-8") as f:
                f.write((node.plan or "") + "\n")
        technique_record = (node.config or {}).get("technique_record")
        if technique_record:
            raw_outline = technique_record.get("raw_outline")
            if raw_outline:
                (node_dir / "raw_outline.md").write_text(
                    str(raw_outline), encoding="utf-8"
                )
            with open(node_dir / "technique_record.json", "w", encoding="utf-8") as f:
                persisted_record = self._compact_technique_record(technique_record)
                if raw_outline:
                    persisted_record["raw_outline_path"] = "raw_outline.md"
                json.dump(
                    persisted_record,
                    f,
                    indent=2,
                    default=str,
                )

    def _persist_tree_state(self) -> None:
        """Write the canonical tree used to generate method_tree.png."""
        payload = {
            "task_name": self.task_name,
            "metric_direction": self.metric_direction,
            "metric_name": getattr(self, "metric_name", "score"),
            "baseline_score": self.baseline_score,
            "budget": self.total_budget,
            "budget_unit": "executed_implementation_experiment",
            "initial_fanout": getattr(self, "initial_fanout", None),
            "ucb_eligible_budget": max(
                0, self.total_budget - getattr(self, "initial_fanout", 0)
            ),
            "experiments_executed": getattr(self, "experiments_executed", 0),
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

    def initialize_task(self, temperature: float = 0.2):
        """
        Deletes any pre-existing initial algorithms/dataloaders and calls the 
        InitialAgent to generate them fresh from scratch based on the task description.
        """
        print(f"ManagerAgent: Dynamically generating initial task baselines for {self.task_name}...")
        
        loader_path = self.task_dir / "initial_dataloader.py"
        algo_path = self.task_dir / "initial_algorithm.py"

        original_files = {
            path: path.read_bytes() for path in (loader_path, algo_path) if path.exists()
        }
        try:
            for path in (loader_path, algo_path):
                if path.exists() or path.is_symlink():
                    path.unlink()

            initial_agent = InitialAgent(model_name=self.model_name)
            initial_agent.generate_initial_code(self.task_dir, temperature=temperature)
            if not loader_path.is_file() or not algo_path.is_file():
                raise RuntimeError("InitialAgent did not produce both required baseline files")
        except Exception:
            for path in (loader_path, algo_path):
                if path.exists() or path.is_symlink():
                    path.unlink()
                if path in original_files:
                    path.write_bytes(original_files[path])
            raise
        print("ManagerAgent: Baseline generated successfully!")

    def _feasibility_reason(self, model_card: dict | None) -> str | None:
        """Reject statically incompatible artifacts before charging an experiment."""
        if not model_card:
            return None
        profile = normalize_resource_profile(model_card)
        accelerator = profile["accelerator"]
        if accelerator == "gpu" and not ({"cuda", "mps", "gpu"} & self.available_accelerators):
            return "artifact requires a GPU but this run exposes CPU only"
        if accelerator in {"cuda", "mps"} and accelerator not in self.available_accelerators:
            return f"artifact requires {accelerator.upper()} but that accelerator is unavailable"
        if (
            self.available_ram_gb > 0
            and profile["min_ram_gb"] > self.available_ram_gb
        ):
            return (
                f"artifact requires {profile['min_ram_gb']:.1f} GB RAM but only "
                f"{self.available_ram_gb:.1f} GB is available"
            )
        if (
            profile["estimated_runtime_seconds"] > 0
            and profile["estimated_runtime_seconds"] > self.subprocess_timeout
        ):
            return (
                f"artifact estimates {profile['estimated_runtime_seconds']:.0f}s but "
                f"the task timeout is {self.subprocess_timeout:.0f}s"
            )
        return None

    @staticmethod
    def _operator_compatibility_reason(
        model_card: dict | None, operator: str | None
    ) -> str | None:
        if not model_card or not operator or operator == "initial":
            return None
        capabilities = model_card.get("capabilities")
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
        Runs the budget-scaled tree search on the task.
        Returns the best leaf node ID.
        Note: initialize_task() must be called by the caller before this method.
        """
        self._prepare_run_root()
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
        dynamic_approaches = self.technique_agent.generate_initial_approaches(
            self.task_description, count=self.initial_fanout
        )
        print("ManagerAgent: LLM initial branch ideas:")
        for idx, app in enumerate(dynamic_approaches, start=1):
            print(f"  {idx}. {app.get('name', 'unnamed_branch')}: {app.get('plan', '')}")
        
        root_node = self.all_nodes[root_id]
        
        for app in dynamic_approaches:
            name = app.get("name", "Branch_Plan")
            plan = app.get("plan", "")
            node_id = self.get_new_node_id()
            child_node = NodeState(
                node_id=node_id,
                parent_id=root_id,
                node_type="technique",
                plan=plan,
                operator="initial",
                fidelity="screen" if self.enable_multi_fidelity else "full",
                config={
                    "priority": 0.0,
                    "allowed_scopes": ["full_pipeline", "model_family"],
                },
            )
            self.all_nodes[node_id] = child_node
            root_node.children_ids.append(node_id)
            self._persist_node(node_id)
            print(f"ManagerAgent: Spawned branch {node_id}: {name} (Plan: {plan[:60]}...)")

        self._persist_tree_state()

        # Load L1 index for the technique agents
        l1_path = self.project_root / "memory_pool" / "l1_index.json"
        with open(l1_path, 'r', encoding='utf-8') as f:
            l1_index = json.load(f)
            
        # 3. Main search loop. Planning/technique resolution does not consume the
        # experiment budget; only an attempted implementation run does.
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
                    self.experiments_executed += 1
                    self.scheduler.backpropagate(selected_id, -1.0, self.all_nodes)
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
            print(
                f"\nManagerAgent: Search finished after {self.experiments_executed} experiments. "
                f"Best Node: {best_node_id} (Score: {best_score:.5f}, "
                f"Fidelity: {self.all_nodes[best_node_id].fidelity})"
            )
        else:
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

    def _score_to_reward(self, score: float, cv_std: float = 0.0) -> float:
        """Normalize a conservative, uncertainty-discounted metric around baseline."""
        score = float(score)
        try:
            uncertainty = max(0.0, float(cv_std or 0.0)) * getattr(
                self, "uncertainty_weight", 1.0
            )
        except (TypeError, ValueError):
            uncertainty = 0.0
        conservative_score = (
            score - uncertainty
            if self.metric_direction == "maximize"
            else score + uncertainty
        )
        if self.baseline_score is None:
            reward = (
                conservative_score
                if self.metric_direction == "maximize"
                else -conservative_score
            )
        elif self.metric_direction == "maximize":
            # For bounded scores such as AUC, use relative error reduction. Fall
            # back to baseline-relative change for unbounded metrics.
            if 0.0 <= self.baseline_score < 1.0 and 0.0 <= conservative_score <= 1.0:
                reward = (conservative_score - self.baseline_score) / max(
                    1.0 - self.baseline_score, 1e-12
                )
            else:
                reward = (conservative_score - self.baseline_score) / max(
                    abs(self.baseline_score), 1e-12
                )
        else:
            reward = (self.baseline_score - conservative_score) / max(
                abs(self.baseline_score), 1e-12
            )
        return max(-1.0, min(1.0, reward))

    @staticmethod
    def _next_fidelity(fidelity: str) -> str:
        return {"screen": "medium", "medium": "full", "full": "full"}.get(
            fidelity, "full"
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
        category = tech_record.get("category")
        artifact_id = tech_record.get("artifact_id")
        if not category or not artifact_id or tech_record.get("status") not in {
            "pool_hit", "pool_added"
        }:
            return
        L2Builder(
            project_root=self.project_root,
            model_name=self.model_name,
            venv_path=self.venv_path,
        ).record_task_validation(
            category,
            artifact_id,
            {
                "task_name": self.task_name,
                "node_id": node_id,
                "score": score,
                "metric_direction": self.metric_direction,
                "status": status,
                "reward": reward,
                "fidelity": fidelity,
                "elapsed_seconds": elapsed_seconds,
                "improved_over_baseline": (
                    None
                    if score is None or self.baseline_score is None
                    else (
                        score > self.baseline_score
                        if self.metric_direction == "maximize"
                        else score < self.baseline_score
                    )
                ),
            },
        )

    def _spawn_follow_up_nodes(self, node: NodeState, node_id: str) -> None:
        """Create cheap virtual operator slots; materialize only the selected slot."""
        if self.experiments_executed + 1 >= self.total_budget or not node.code:
            return
        operator_priorities = {"refine": 0.02, "tune": 0.01, "diversify": 0.0}
        for operator, priority in operator_priorities.items():
            child_fidelity = (
                self._next_fidelity(node.fidelity)
                if self.enable_multi_fidelity and operator in {"refine", "tune"}
                else node.fidelity
            )
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
                    "lazy_proposal": True,
                    "materialized": False,
                    "allowed_scopes": (
                        ["full_pipeline", "model_family"]
                        if operator == "diversify"
                        else ["full_pipeline", "model_family", "component"]
                    ),
                },
            )
            self.all_nodes[new_id] = child
            node.children_ids.append(new_id)
            self._persist_node(new_id)
            print(
                f"ManagerAgent: Spawned virtual {operator} slot {new_id} at "
                f"{child_fidelity} fidelity"
            )
        self._persist_node(node_id)
        self._persist_tree_state()

    def _materialize_lazy_proposal(self, node: NodeState) -> None:
        """Spend one proposal-generation call only after the scheduler selects a slot."""
        config = dict(node.config or {})
        if not config.get("lazy_proposal") or config.get("materialized"):
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
        config["priority"] = proposal.get("priority", config.get("priority", 0.0))
        config["proposal_name"] = proposal.get("name")
        config["materialized"] = True
        node.config = config
        self._persist_node(node.node_id)

    def _execute_node(self, node, selected_id, root_id, l1_index, l1_path):
        """Executes a single node (technique or implementation). Extracted for try/except isolation."""
        if node.node_type == "technique":
            print(f"ManagerAgent: Running Technique Agent on {selected_id}...")

            self._materialize_lazy_proposal(node)

            # Pool additions from earlier nodes in this same run must be visible.
            with open(l1_path, 'r', encoding='utf-8') as f:
                l1_index = json.load(f)
            
            # Fetch sibling/parent records from global memory
            context = self.global_memory.get_default_context(selected_id, self.all_nodes)
            
            # Bug 3 fix: pass clean task_description separately from branch plan
            tech_record = self.technique_agent.run(
                task_description=self.task_description,
                branch_plan=node.plan,
                global_memory_context=context,
                l1_index=l1_index,
                available_accelerators=set(self.available_accelerators),
                preferred_accelerator=self.preferred_accelerator,
                allowed_scopes=set(
                    (node.config or {}).get("allowed_scopes", [])
                ) or None,
            )
            planning_config = dict(node.config or {})
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

                    # A verified reusable method belongs in the memory pool even
                    # before it beats this task's baseline. Task performance is
                    # recorded separately after implementation.
                    use_pool = os.environ.get("ABLATION_USE_POOL", "1") != "0"
                    if use_pool:
                        committed = builder.commit_artifact(
                            category=category,
                            artifact_id=artifact_id,
                            local_code_file=node_dir / f"{artifact_id}.py",
                            local_card_file=node_dir / f"{artifact_id}.json",
                        )
                        tech_record["pool_committed"] = committed
                        if committed:
                            tech_record["status"] = "pool_added"
                            print(
                                f"ManagerAgent: Added verified web artifact {artifact_id} "
                                "to the global memory pool."
                            )
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

            feasibility_card = tech_record.get("model_card")
            if not feasibility_card:
                feasibility_card = (
                    tech_record.get("candidate_artifact", {}).get("model_card")
                )
            feasibility_reason = self._feasibility_reason(feasibility_card)
            compatibility_reason = self._operator_compatibility_reason(
                feasibility_card, node.operator
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
            tech_record = (node.config or {}).get("technique_record", {})
            
            # Install dependencies via Setup Agent
            model_card = tech_record.get("model_card")
            if model_card:
                self.setup_agent.install_dependencies([model_card])
                
            # Run implementation script in its own run folder
            node_dir = self.run_root / selected_id
            node_dir.mkdir(parents=True, exist_ok=True)
            
            res = self.implementation_agent.run(
                node_dir,
                tech_record,
                self.task_dir,
                timeout=self.subprocess_timeout,
                metric_direction=self.metric_direction,
                base_algorithm_path=(node.config or {}).get("base_code_path"),
                parent_node_dir=(
                    self.run_root / node.config["base_node_id"]
                    if (node.config or {}).get("base_node_id")
                    else None
                ),
                fidelity=node.fidelity,
                operator=node.operator,
                enforce_evaluation_contract=True,
                accelerator=self.preferred_accelerator,
                available_accelerators=set(self.available_accelerators),
            )
            
            # Bug 1 fix: Handle execution failures properly
            score = res.get("score")  # Will be None on failure
            status = res.get("status", "completed")
            
            # Record result to NodeState
            node.code = res.get("code_path")
            validation = res.get("validation", {})
            cv_std = validation.get("cv_std", 0.0) if validation else 0.0
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
            }
            node.executed = True
            
            if status == "failed":
                print(f"ManagerAgent: Implementation Node {selected_id} FAILED. No score produced.")
                self._record_artifact_validation(
                    tech_record,
                    selected_id,
                    None,
                    "failed",
                    -1.0,
                    node.fidelity,
                    res.get("elapsed_seconds"),
                )
                # Record failure to global memory
                self.global_memory.record_implementation(selected_id, {"node_id": selected_id}, 0.0, "failed")
                # Backpropagate zero reward
                self.scheduler.backpropagate(selected_id, -1.0, self.all_nodes)
                self._persist_node(selected_id)
                self._persist_tree_state()
                # Do NOT spawn follow-up technique nodes from failed implementations
                return True

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
                self._record_artifact_validation(
                    tech_record,
                    selected_id,
                    score,
                    status,
                    reward,
                    node.fidelity,
                    res.get("elapsed_seconds"),
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

            self._record_artifact_validation(
                tech_record,
                selected_id,
                score,
                status,
                reward,
                node.fidelity,
                res.get("elapsed_seconds"),
            )
            
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
                },
                score,
                "completed",
            )
            
            # Normalize reward for UCB1 backpropagation.
            self.scheduler.backpropagate(selected_id, reward, self.all_nodes)
            self._trace_search(
                {
                    "event": "experiment_completed",
                    "node_id": selected_id,
                    "score": score,
                    "reward": reward,
                    "fidelity": node.fidelity,
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
        """
        Locates the submission file of the best node, aligns it with the sample_submission.csv 
        in the task directory, and saves the final submission to the run and task folders.
        """
        if not best_node_id:
            print("ManagerAgent: No best node found to generate final submission.")
            return False
            
        import pandas as pd
        
        best_node_dir = self.run_root / best_node_id
        generated_sub_path = best_node_dir / "submission" / "submission.csv"
        sample_sub_path = self.task_dir / "sample_submission.csv"
        
        if not generated_sub_path.exists():
            print(f"ManagerAgent WARNING: Generated submission file not found at {generated_sub_path}")
            return False

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
            with open(self.run_root / "ensemble_manifest.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "strategy": self.ensemble_strategy,
                        "fidelity": best_fidelity,
                        "node_ids": selected_ensemble_nodes,
                    },
                    f,
                    indent=2,
                )
            
        # Target output paths
        task_output_path = self.task_dir / "submission.csv"
        run_output_path = self.run_root / "submission.csv"
        
        if sample_sub_path.exists():
            print(f"ManagerAgent: Formatting final submission based on sample submission: {sample_sub_path.name}")
            try:
                sample_df = pd.read_csv(sample_sub_path)
                generated_df = pd.read_csv(generated_sub_path)
                if sample_df.empty or generated_df.empty:
                    raise ValueError("submission files must not be empty")
                
                id_col = sample_df.columns[0]
                prediction_cols = list(sample_df.columns[1:])
                if not prediction_cols:
                    raise ValueError("sample submission has no prediction columns")
                if sample_df[id_col].duplicated().any():
                    raise ValueError(f"sample submission contains duplicate {id_col!r} values")
                missing_prediction_cols = [
                    col for col in prediction_cols if col not in generated_df.columns
                ]
                if missing_prediction_cols:
                    raise ValueError(
                        f"generated submission is missing columns: {missing_prediction_cols}"
                    )
                
                if id_col in generated_df.columns:
                    if generated_df[id_col].duplicated().any():
                        raise ValueError(f"generated submission contains duplicate {id_col!r} values")
                    if set(generated_df[id_col]) != set(sample_df[id_col]):
                        raise ValueError("generated IDs do not exactly match sample submission IDs")
                    aligned = generated_df.set_index(id_col).reindex(sample_df[id_col])
                    final_df = sample_df[[id_col]].copy()
                    for col in prediction_cols:
                        final_df[col] = aligned[col].to_numpy()
                else:
                    if len(generated_df) != len(sample_df):
                        raise ValueError(
                            "generated submission has no ID column and its row count differs from the sample"
                        )
                    final_df = sample_df[[id_col]].copy()
                    for col in prediction_cols:
                        final_df[col] = generated_df[col].to_numpy()

                if final_df[prediction_cols].isnull().any().any():
                    raise ValueError("generated predictions contain missing values")

                final_df.to_csv(task_output_path, index=False)
                final_df.to_csv(run_output_path, index=False)
                print(f"ManagerAgent: Aligned final submission saved to {task_output_path} and {run_output_path}")
                return True
            except Exception as e:
                print(f"ManagerAgent ERROR: Refusing invalid final submission: {e}")
                return False
        else:
            print("ManagerAgent: No sample submission found. Copying best node submission directly.")
            try:
                generated_df = pd.read_csv(generated_sub_path)
                if generated_df.empty or generated_df.isnull().any().any():
                    raise ValueError("generated submission is empty or contains missing values")
                generated_df.to_csv(task_output_path, index=False)
                generated_df.to_csv(run_output_path, index=False)
                return True
            except Exception as e:
                print(f"ManagerAgent ERROR: Refusing invalid final submission: {e}")
                return False

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
                            f"{node.operator or 'initial'} / {node.fidelity}\n"
                            f"Use: {artifact_id}\nScore: {score:.5f}"
                        )
                    elif tech_record.get("status") == "bootstrap_failed":
                        desc = (
                            f"{node.operator or 'initial'} / {node.fidelity}\n"
                            f"Use: Self-contained fallback\nScore: {score:.5f}"
                        )
                    else:
                        desc = (
                            f"{node.operator or 'initial'} / {node.fidelity}\n"
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
