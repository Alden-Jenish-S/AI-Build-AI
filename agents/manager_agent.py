import os
import json
import math
from pathlib import Path
from typing import Dict, Any, List
from tree.node import NodeState
from tree.scheduler import UCB1Scheduler
from tree.global_memory import GlobalMemory
from .initial_agent import InitialAgent
from .technique_agent import TechniqueAgent
from .implementation_agent import ImplementationAgent
from .setup_agent import SetupAgent
from memory_pool.builder.l2_builder import L2Builder
from runtime_utils import validate_path_component

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
        
        # State tracker
        self.all_nodes: Dict[str, NodeState] = {}
        self.node_counter = 0

        # Load task config for metric direction and timeout
        self.metric_direction = "maximize"
        self.subprocess_timeout = 300
        config_file = self.task_dir / "task_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    task_config = json.load(f)
                self.metric_direction = task_config.get("metric_direction", "maximize")
                self.subprocess_timeout = task_config.get("subprocess_timeout", 300)
            except Exception as e:
                print(f"ManagerAgent WARNING: Failed to parse task_config.json: {e}")
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
            }
            diagnostics = node.result.get("diagnostics")
            if diagnostics:
                result["diagnostics_tail"] = str(diagnostics)[-4000:]
        return {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "node_type": node.node_type,
            "plan": node.plan,
            "code": node.code,
            "config": node.config,
            "result": result,
            "executed": node.executed,
            "status": (
                "pending"
                if not node.executed
                else (result or {}).get("status", "completed")
            ),
            "visits": node.visits,
            "total_reward": node.total_reward,
            "children_ids": list(node.children_ids),
        }

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
            with open(node_dir / "technique_record.json", "w", encoding="utf-8") as f:
                json.dump(technique_record, f, indent=2, default=str)

    def _persist_tree_state(self) -> None:
        """Write the canonical tree used to generate method_tree.png."""
        payload = {
            "task_name": self.task_name,
            "metric_direction": self.metric_direction,
            "baseline_score": self.baseline_score,
            "budget": self.total_budget,
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

    def run_tree_search(self) -> str:
        """
        Runs the full 3-branch tree search on the task.
        Returns the best leaf node ID.
        Note: initialize_task() must be called by the caller before this method.
        """
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
        
        # 2. Spawn 3 initial strategic branches authored by the LLM.
        # If ideation fails, stop clearly instead of silently biasing the run with canned defaults.
        dynamic_approaches = self.technique_agent.generate_initial_approaches(self.task_description)
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
                plan=plan
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
            
        # 3. Main Tree Search Loop
        for step in range(self.total_budget):
            self.scheduler.current_step = step
            print(f"\n--- Search Step {step + 1}/{self.total_budget} ---")
            
            # Select node to execute/expand using UCB1 scheduler
            selected_id = self.scheduler.select_next_node(root_id, self.all_nodes)
            if selected_id is None:
                print("ManagerAgent: Search tree is exhausted; stopping early.")
                break
            node = self.all_nodes[selected_id]
            print(f"ManagerAgent: Selected Node {selected_id} (Type: {node.node_type})")
            
            # Bug 2 fix: wrap step body in try/except for node-level failure isolation
            try:
                self._execute_node(node, selected_id, root_id, l1_index, l1_path)
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
                # Backpropagate zero reward so UCB1 stats stay consistent
                self.scheduler.backpropagate(selected_id, 0.0, self.all_nodes)
                self._persist_node(selected_id)
                self._persist_tree_state()
                continue
                
        # Find best leaf node (skip failed nodes with score=None)
        best_node_id = None
        best_score = -float('inf') if self.metric_direction == "maximize" else float('inf')
        for nid, nstate in self.all_nodes.items():
            if nstate.node_type == "implementation" and nstate.result:
                score = nstate.result.get("score")
                if score is None:
                    continue  # Skip failed nodes
                if self.metric_direction == "maximize":
                    if score > best_score:
                        best_score = score
                        best_node_id = nid
                else:
                    if score < best_score:
                        best_score = score
                        best_node_id = nid
                    
        if best_node_id:
            print(f"\nManagerAgent: Tree Search finished! Best Node: {best_node_id} (Score: {best_score:.5f})")
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

    def _execute_node(self, node, selected_id, root_id, l1_index, l1_path):
        """Executes a single node (technique or implementation). Extracted for try/except isolation."""
        if node.node_type == "technique":
            print(f"ManagerAgent: Running Technique Agent on {selected_id}...")

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
                l1_index=l1_index
            )
            node.config = {"technique_record": tech_record}
            self._persist_node(selected_id)
            
            # Pre-allocate child Implementation node ID and create its directory
            child_id = self.get_new_node_id()
            node_dir = self.run_root / child_id
            node_dir.mkdir(parents=True, exist_ok=True)
            child_node = NodeState(
                node_id=child_id,
                parent_id=selected_id,
                node_type="implementation",
                config={"technique_record": tech_record}
            )
            self.all_nodes[child_id] = child_node
            node.children_ids.append(child_id)
            self._persist_node(child_id)
            
            # If pool miss, dynamically build and verify locally in child's node_dir!
            if tech_record.get("status") == "pool_miss":
                print("ManagerAgent: Bootstrapping new technique from web search outline (local build)...")
                builder = L2Builder(project_root=self.project_root, model_name=self.model_name, venv_path=self.venv_path)
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
                        "Use a self-contained baseline improvement because the web-derived "
                        f"artifact {artifact_id or '<unknown>'} failed verification."
                    )
                    print(
                        "ManagerAgent WARNING: Web-derived artifact failed verification; "
                        "preserved it in the node directory and will use a self-contained fallback."
                    )

            node.config = {"technique_record": tech_record}
            child_node.config = {"technique_record": tech_record}
            
            # Record to global memory
            self.global_memory.record_technique(selected_id, tech_record.get("plan", ""), "succeeded")
            
            # Mark technique node as executed
            node.executed = True
            
            # Reward for technique node is epsilon (0.01)
            self.scheduler.backpropagate(selected_id, 0.01, self.all_nodes)
            self._persist_node(selected_id)
            self._persist_node(child_id)
            self._persist_tree_state()
            print(f"ManagerAgent: Technique Node {selected_id} resolved. Spawned Implementation Node {child_id}")
            
        elif node.node_type == "implementation":
            print(f"ManagerAgent: Running Implementation Agent on {selected_id}...")
            tech_record = node.config.get("technique_record", {})
            
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
                metric_direction=self.metric_direction
            )
            
            # Bug 1 fix: Handle execution failures properly
            score = res.get("score")  # Will be None on failure
            status = res.get("status", "completed")
            
            # Record result to NodeState
            node.code = res.get("code_path")
            node.result = {"score": score, "status": status, "diagnostics": res.get("diagnostics")}
            node.executed = True
            
            if status == "failed":
                print(f"ManagerAgent: Implementation Node {selected_id} FAILED. No score produced.")
                if tech_record.get("pool_committed"):
                    L2Builder(
                        project_root=self.project_root,
                        model_name=self.model_name,
                        venv_path=self.venv_path,
                    ).record_task_validation(
                        tech_record.get("category"),
                        tech_record.get("artifact_id"),
                        {
                            "task_name": self.task_name,
                            "node_id": selected_id,
                            "score": None,
                            "metric_direction": self.metric_direction,
                            "status": "failed",
                        },
                    )
                # Record failure to global memory
                self.global_memory.record_implementation(selected_id, {"node_id": selected_id}, 0.0, "failed")
                # Backpropagate zero reward
                self.scheduler.backpropagate(selected_id, 0.0, self.all_nodes)
                self._persist_node(selected_id)
                self._persist_tree_state()
                # Do NOT spawn follow-up technique nodes from failed implementations
                return

            if tech_record.get("pool_committed"):
                L2Builder(
                    project_root=self.project_root,
                    model_name=self.model_name,
                    venv_path=self.venv_path,
                ).record_task_validation(
                    tech_record.get("category"),
                    tech_record.get("artifact_id"),
                    {
                        "task_name": self.task_name,
                        "node_id": selected_id,
                        "score": score,
                        "metric_direction": self.metric_direction,
                        "status": status,
                        "improved_over_baseline": (
                            None
                            if self.baseline_score is None
                            else (
                                score > self.baseline_score
                                if self.metric_direction == "maximize"
                                else score < self.baseline_score
                            )
                        ),
                    },
                )
            
            # Record to global memory
            self.global_memory.record_implementation(selected_id, {"node_id": selected_id}, score, "completed")
            
            # Normalize reward for UCB1 backpropagation.
            normalized_reward = score if self.metric_direction == "maximize" else -score
            self.scheduler.backpropagate(selected_id, normalized_reward, self.all_nodes)
            self._persist_node(selected_id)
            self._persist_tree_state()
            print(f"ManagerAgent: Implementation Node {selected_id} completed. Score: {score:.5f} (Normalized Reward: {normalized_reward:.5f})")
            
            # Spawn a new technique child so tree keeps growing
            new_tech_id = self.get_new_node_id()
            
            # Read parent code to let the LLM analyze it and suggest an advanced improvement
            parent_code = ""
            if node.code:
                try:
                    with open(node.code, 'r', encoding='utf-8') as f:
                        parent_code = f.read()
                except Exception:
                    pass
            
            proposal_plan = ""
            try:
                from agents.llm_utils import call_llm
                improvement_system = (
                    "You are an expert ML research scientist. Propose a specific, advanced follow-up improvement "
                    "plan to improve the performance of our current machine learning pipeline.\n"
                    "Look at the parent node's validation score and its python code.\n"
                    "Propose a concrete, advanced technique (such as a specialized loss function, a custom layer/interaction, "
                    "an attention mechanism, or a targeted regularization) that is complementary to the current code.\n"
                    "Keep your proposal to a single, clear, actionable sentence of at most 15 words. Do NOT write any introduction or code."
                )
                improvement_user = f"""
Parent Node ID: {selected_id}
Parent Validation Score: {score:.5f}

Current Python Code:
```python
{parent_code}
```

Propose a single advanced technique to improve it.
"""
                proposal_plan = call_llm(improvement_system, improvement_user, model=self.technique_agent.model_name).strip()
            except Exception as le:
                print(f"ManagerAgent WARNING: Failed to call LLM for improvement proposal: {le}")
                
            if not proposal_plan:
                proposal_plan = f"Continue improving from {selected_id} (previous score={score:.4f}). Try a different or complementary technique."
                
            new_tech_node = NodeState(
                node_id=new_tech_id,
                parent_id=selected_id,
                node_type="technique",
                plan=proposal_plan
            )
            self.all_nodes[new_tech_id] = new_tech_node
            node.children_ids.append(new_tech_id)
            self._persist_node(selected_id)
            self._persist_node(new_tech_id)
            self._persist_tree_state()
            print(f"ManagerAgent: Spawned follow-up Technique Node {new_tech_id} with plan: '{proposal_plan}'")

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
                        desc = f"Use: {artifact_id}\nScore: {score:.5f}"
                    elif tech_record.get("status") == "bootstrap_failed":
                        desc = f"Use: Self-contained fallback\nScore: {score:.5f}"
                    else:
                        desc = f"Score: {score:.5f}"
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
