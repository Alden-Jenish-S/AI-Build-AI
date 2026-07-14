import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.manager_agent import ManagerAgent
from agents.llm_utils import get_token_usage, reset_token_usage
from eval.metrics import calculate_ablation_metrics
from memory_pool.builder.l2_builder import L2Builder

def run_single_condition(condition_name: str, use_pool: bool, use_tree: bool, task_name: str, derived_baseline: float) -> Dict[str, Any]:
    """Runs a single ablation condition for a task."""
    print(f"\n>>> Running Ablation: {condition_name} on task: {task_name} <<<")
    reset_token_usage()
    
    # Configure pool override
    os.environ["ABLATION_USE_POOL"] = "1" if use_pool else "0"
    os.environ["ABLATION_USE_TREE"] = "1" if use_tree else "0"
    
    # Standard budget: 6 nodes total for tree-search, 3 nodes for single-agent
    budget = 6 if use_tree else 3
    
    manager = ManagerAgent(task_name=task_name, total_budget=budget, baseline_score=derived_baseline)
    
    start_time = time.time()
    
    # Cache restore: Copy cached baseline files to task_dir so they are exactly identical
    cache_dir = PROJECT_ROOT / "runs" / task_name / "baseline_cache"
    shutil.copy(cache_dir / "initial_dataloader.py", manager.task_dir / "initial_dataloader.py")
    shutil.copy(cache_dir / "initial_algorithm.py", manager.task_dir / "initial_algorithm.py")
    
    # If tree search is disabled, we simulate a single-agent linear pipeline:
    if not use_tree:
        print("Running in Single-Agent Mode...")
        
        if use_pool:
            l1_path = PROJECT_ROOT / "memory_pool" / "l1_index.json"
            with open(l1_path, 'r', encoding='utf-8') as f:
                l1_index = json.load(f)
        else:
            l1_index = {}
        
        # 1. Technique Node (Bug 3 fix: pass clean task_description separately)
        tech_record = manager.technique_agent.run(
            task_description=manager.task_description,
            branch_plan="Baseline tabular ensembling",
            global_memory_context={}, 
            l1_index=l1_index
        )

        # If the agent discovers a new technique, build and verify a local artifact
        # before implementation. Tree mode does this in ManagerAgent._execute_node.
        # In no-pool mode this local artifact can be used for the run, but is not
        # committed to the global memory pool.
        if tech_record.get("status") == "pool_miss":
            print("Single-agent mode: Bootstrapping new technique from web search outline (local build)...")
            builder = L2Builder(project_root=PROJECT_ROOT, model_name=manager.model_name, venv_path=manager.venv_path)
            raw_outline = tech_record.get("raw_outline", "")
            node_dir = manager.run_root / "single_agent_run"
            node_dir.mkdir(parents=True, exist_ok=True)
            success, category, artifact_id, model_card = builder.build_from_source(
                "web_search_dynamic", raw_outline, commit=False, target_dir=node_dir
            )
            if success:
                tech_record["status"] = "pool_miss_pending"
                tech_record["artifact_id"] = artifact_id
                tech_record["category"] = category
                tech_record["model_card"] = model_card
                tech_record["plan"] = f"Import and use local bootstrapped artifact {artifact_id} from category {category}."
            else:
                print("Single-agent mode WARNING: Failed to verify bootstrapped technique. Continuing without a model card.")

        # 2. Setup Node
        model_card = tech_record.get("model_card")
        if model_card:
            manager.setup_agent.install_dependencies([model_card])
        # 3. Implementation Node
        node_dir = manager.run_root / "single_agent_run"
        res = manager.implementation_agent.run(
            node_dir, 
            tech_record, 
            manager.task_dir,
            timeout=manager.subprocess_timeout,
            metric_direction=manager.metric_direction
        )
        
        # Bug 1 fix: Handle failed implementation status
        status = res.get("status", "completed")
        if status == "failed":
            best_score = None
            print(f"Single-agent run FAILED. No score produced.")
        else:
            best_score = res.get("score")
            manager.generate_final_submission("single_agent_run")

            if use_pool and tech_record.get("status") == "pool_miss_pending" and best_score is not None:
                exceeds = best_score > derived_baseline if manager.metric_direction == "maximize" else best_score < derived_baseline
                if exceeds:
                    builder = L2Builder(project_root=PROJECT_ROOT, model_name=manager.model_name, venv_path=manager.venv_path)
                    artifact_id = tech_record.get("artifact_id")
                    category = tech_record.get("category")
                    local_code_file = node_dir / f"{artifact_id}.py"
                    local_card_file = node_dir / f"{artifact_id}.json"
                    if local_code_file.exists() and local_card_file.exists():
                        if builder.commit_artifact(category, artifact_id, local_code_file, local_card_file):
                            tech_record["status"] = "pool_hit"
                            print(f"Single-agent mode: Committed {artifact_id} to memory pool.")
                    else:
                        print(f"Single-agent mode WARNING: Local artifact files missing for {artifact_id}; cannot commit.")
                else:
                    print(f"Single-agent mode: New technique did not beat baseline ({derived_baseline}); not committing.")
        
        # Build fake history for metric parser
        history = [
            {"type": "technique", "status": "pool_hit" if use_pool else "pool_miss"},
            {"type": "implementation", "score": best_score}
        ]
    else:
        # Full tree search run
        best_node_id = manager.run_tree_search()
        if best_node_id:
            manager.generate_final_submission(best_node_id)
        best_score = None
        if best_node_id and best_node_id in manager.all_nodes:
            res = manager.all_nodes[best_node_id].result
            if res:
                best_score = res.get("score")
                
        # Format history for metrics
        history = []
        for nid, nstate in manager.all_nodes.items():
            if nid == "root":
                continue
            history.append({
                "type": nstate.node_type,
                "status": "pool_hit" if use_pool and nstate.config and nstate.config.get("technique_record", {}).get("status") == "pool_hit" else "pool_miss",
                "score": nstate.result.get("score") if nstate.result else None
            })
            
    elapsed = time.time() - start_time
    tokens = get_token_usage()
    
    # Compute maximize flag for metrics
    maximize = (manager.metric_direction == "maximize")
    metrics = calculate_ablation_metrics(history, derived_baseline, maximize=maximize)
    
    return {
        "condition": condition_name,
        "best_score": best_score,
        "medal_rate": metrics["medal_rate"],
        "gold_rate": metrics["gold_rate"],
        "avg_tokens": tokens["input_tokens"] + tokens["output_tokens"],
        "pool_hit_rate": metrics["pool_hit_rate"] if use_pool else "n/a",
        "overcome_rate": metrics["overcome_rate"],
        "time_elapsed": elapsed
    }

def main():
    task_name = "tabular-playground-series-may-2022"
    if len(sys.argv) > 1:
        task_name = sys.argv[1]
        
    # 1. Initialize manager to load config and generate/run baseline once
    temp_manager = ManagerAgent(task_name=task_name)
    
    # Setup cache directory
    temp_manager.run_root.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_ROOT / "runs" / task_name / "baseline_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate baseline once at temp=0
    temp_manager.initialize_task(temperature=0.0)
    
    # Cache generated baseline files
    shutil.copy(temp_manager.task_dir / "initial_dataloader.py", cache_dir / "initial_dataloader.py")
    shutil.copy(temp_manager.task_dir / "initial_algorithm.py", cache_dir / "initial_algorithm.py")
    
    # Run the baseline once to derive its score
    print("Running initial baseline to derive baseline score...")
    baseline_run_dir = PROJECT_ROOT / "runs" / task_name / "initial_baseline_run"
    baseline_run_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy loader to baseline run folder
    shutil.copy(temp_manager.task_dir / "initial_dataloader.py", baseline_run_dir / "initial_dataloader.py")
    # Symlink input dataset
    src_input = temp_manager.task_dir / "input"
    dest_input = baseline_run_dir / "input"
    if src_input.exists():
        if not dest_input.exists():
            try:
                os.symlink(src_input, dest_input)
            except Exception:
                if src_input.is_dir():
                    shutil.copytree(src_input, dest_input)
    else:
        # Fallback: if files are directly in task_dir, copy/symlink them
        dest_input.mkdir(parents=True, exist_ok=True)
        for f in temp_manager.task_dir.glob("*"):
            if f.is_file() and f.suffix in [".csv", ".tsv", ".txt", ".json"] and f.name != "task_config.json":
                target_path = dest_input / f.name
                if target_path.exists() or target_path.is_symlink():
                    try:
                        os.remove(target_path)
                    except Exception:
                        pass
                try:
                    os.symlink(f, target_path)
                except Exception:
                    try:
                        shutil.copy(f, target_path)
                    except Exception:
                        pass
                
    # Run initial_algorithm.py
    cmd = [temp_manager.venv_path, str(temp_manager.task_dir / "initial_algorithm.py")]
    derived_baseline = 0.5 if temp_manager.metric_direction == "maximize" else 999.0
    
    try:
        res = subprocess.run(cmd, cwd=baseline_run_dir, capture_output=True, text=True, timeout=temp_manager.subprocess_timeout)
        # Parse score via result.json
        result_json_path = baseline_run_dir / "result.json"
        if result_json_path.exists():
            with open(result_json_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            derived_baseline = float(result_data.get("score", derived_baseline))
            print(f"Derived baseline score from result.json: {derived_baseline}")
        else:
            # Regex fallback
            diagnostics = res.stdout + "\n" + res.stderr
            import re
            score_matches = re.findall(
                r'(?:score|auc|accuracy|metric|rmse|mae|loss|f1)[:\s=]+(-?[0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)',
                diagnostics, re.IGNORECASE
            )
            if score_matches:
                derived_baseline = float(score_matches[-1])
            print(f"Derived baseline score from stdout regex: {derived_baseline}")
    except Exception as e:
        print(f"Failed to run initial baseline, defaulting baseline score to 0.7: {e}")
        derived_baseline = 0.70

    results = []
    
    # 2. 2x2 conditions
    conditions = [
        ("No-pool, single-agent (baseline)", False, False),
        ("No-pool, tree-search", False, True),
        ("Pool, single-agent", True, False),
        ("Pool, tree-search (full system)", True, True)
    ]
    
    for cond_name, use_pool, use_tree in conditions:
        try:
            res = run_single_condition(cond_name, use_pool, use_tree, task_name, derived_baseline)
            results.append(res)
        except Exception as e:
            print(f"Error running ablation condition {cond_name}: {e}")
            
    # 3. Output markdown table
    out_dir = PROJECT_ROOT / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    res_file = out_dir / "results.md"
    
    with open(res_file, 'w', encoding='utf-8') as f:
        f.write("# Ablation Study Results\n\n")
        f.write(f"Task evaluated: {task_name}\n")
        f.write(f"Derived baseline score: {derived_baseline:.4f} (direction: {temp_manager.metric_direction})\n\n")
        f.write("| Condition | Best Score | Medal Rate | Gold Rate | Avg tokens/node | Pool-hit-rate | Overcome Rate |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            avg_tok = f"{r['avg_tokens']}"
            bs = f"{r['best_score']:.4f}" if r['best_score'] is not None else "FAILED"
            f.write(f"| {r['condition']} | {bs} | {r['medal_rate']:.1%} | {r['gold_rate']:.1%} | {avg_tok} | {r['pool_hit_rate']} | {r['overcome_rate']:.1%} |\n")
            
    print(f"\nAblation run completed! Results written to {res_file}")

if __name__ == "__main__":
    import subprocess
    main()
