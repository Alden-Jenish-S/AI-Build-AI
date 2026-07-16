import re
import os
import sys
import json
import math
import signal
import subprocess
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional
from .llm_utils import call_llm
from .validation_guard import inspect_generated_code
from evaluation_contract import FIDELITY_PROFILES, validate_evaluation_outputs
from runtime_utils import (
    accelerator_subprocess_env,
    resolve_within,
    validate_storage_identifier,
)

class ImplementationAgent:
    def __init__(self, venv_python_path: str = "./.venv/bin/python", model_name: str = None):
        import sys
        resolved_path = str(Path(venv_python_path).resolve())
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
            print(f"ImplementationAgent WARNING: Specified python path '{resolved_path}' is invalid or non-functional. Falling back to active running interpreter: {sys.executable}")
            self.venv_python = sys.executable
        else:
            self.venv_python = resolved_path
            
        self.model_name = model_name
        self.project_root = Path(__file__).resolve().parent.parent

    def _inherit_parent_workspace(self, parent_node_dir: Path, node_dir: Path) -> list[str]:
        """Seed a child with reusable parent artifacts without copying stale outputs/data."""
        inherited = []
        if not parent_node_dir or not parent_node_dir.is_dir():
            return inherited
        excluded = {
            "algorithm.py",
            "result.json",
            "node_state.json",
            "technique_record.json",
            "error.log",
            "oof_predictions.csv",
            "evaluation_manifest.json",
            "fold_assignments.csv",
        }
        allowed_suffixes = {
            ".py", ".json", ".yaml", ".yml", ".txt", ".pkl", ".joblib",
            ".cbm", ".bin", ".pt", ".pth",
        }
        for source in parent_node_dir.iterdir():
            if (
                not source.is_file()
                or source.is_symlink()
                or source.name in excluded
                or source.suffix.lower() not in allowed_suffixes
            ):
                continue
            destination = node_dir / source.name
            if destination.exists():
                continue
            shutil.copy2(source, destination)
            inherited.append(source.name)
        return inherited

    def run(
        self,
        node_dir: Path,
        technique_record: dict,
        task_dir: Path,
        timeout: int = 300,
        metric_direction: str = "maximize",
        base_algorithm_path: Optional[Path] = None,
        parent_node_dir: Optional[Path] = None,
        fidelity: str = "full",
        operator: Optional[str] = None,
        enforce_evaluation_contract: bool = False,
        accelerator: str = "cpu",
        available_accelerators: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        """
        1. Reads task skeletons from task_dir.
        2. Calls LLM to generate updated code wiring the artifact in.
        3. Writes it to node_dir / "algorithm.py".
        4. Runs it and parses result.json for the evaluation metric score.
        
        Args:
            node_dir: Directory for this node's run outputs
            technique_record: Dict from TechniqueAgent with plan/model_card
            task_dir: Task directory with initial code and data
            timeout: Subprocess timeout in seconds (default 300)
            metric_direction: "maximize" or "minimize" — used in the prompt
        """
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"timeout must be a positive number, got {timeout!r}")
        if metric_direction not in {"maximize", "minimize"}:
            raise ValueError(
                f"metric_direction must be 'maximize' or 'minimize', got {metric_direction!r}"
            )
        accelerator = str(accelerator).lower()
        if accelerator not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported accelerator: {accelerator!r}")
        exposed_accelerators = {
            str(item).lower() for item in (available_accelerators or {"cpu"})
        }
        exposed_accelerators.add("cpu")
        if accelerator not in exposed_accelerators:
            raise ValueError(
                f"Selected accelerator {accelerator!r} is not exposed by this run"
            )
        node_dir.mkdir(parents=True, exist_ok=True)
        run_started = time.monotonic()
        execution_resource = {
            "selected_accelerator": accelerator,
            "available_accelerators": sorted(exposed_accelerators),
            "environment_variable": "AIBUILDAI_ACCELERATOR",
            "fallback": "cpu",
        }
        with open(node_dir / "execution_resource.json", "w", encoding="utf-8") as f:
            json.dump(execution_resource, f, indent=2)
            f.write("\n")
        inherited_files = self._inherit_parent_workspace(
            Path(parent_node_dir) if parent_node_dir else None, node_dir
        )
        
        # Copy dataloader to node_dir so imports work locally
        src_loader = task_dir / "initial_dataloader.py"
        dest_loader = node_dir / "initial_dataloader.py"
        if src_loader.exists():
            shutil.copy(src_loader, dest_loader)
        contract_source = self.project_root / "evaluation_contract.py"
        if contract_source.is_file():
            shutil.copy2(contract_source, node_dir / "evaluation_contract.py")
            
        # Copy the memory pool technique python file to node_dir so it can be imported (Fixes ModuleNotFoundError)
        model_card = technique_record.get("model_card", {})
        if model_card and "category" in model_card and "code_path" in model_card:
            try:
                category = validate_storage_identifier(model_card["category"], "category")
                artifact_id = validate_storage_identifier(
                    model_card.get("artifact_id"), "artifact_id"
                )
                code_name = model_card["code_path"]
                if code_name != f"{artifact_id}.py":
                    raise ValueError("code_path must match '<artifact_id>.py'")
                store = self.project_root / "memory_pool" / "l2_store"
                src_tech = resolve_within(store, category, code_name)
                dest_tech = resolve_within(node_dir, code_name)
                if src_tech.is_file():
                    shutil.copy(src_tech, dest_tech)
                    print(f"ImplementationAgent: Copied technique from {src_tech} to {dest_tech}")
                elif not dest_tech.is_file():
                    print(f"ImplementationAgent WARNING: Technique code file not found at {src_tech}")
            except ValueError as exc:
                raise ValueError(f"Unsafe or invalid model card: {exc}") from exc

        # Symlink input directory so dataloader can find datasets (Fixes FileNotFoundError)
        src_input = task_dir / "input"
        dest_input = node_dir / "input"
        if src_input.exists():
            if not dest_input.exists():
                try:
                    os.symlink(src_input, dest_input)
                except Exception:
                    if src_input.is_dir():
                        shutil.copytree(src_input, dest_input)
        else:
            # Fallback: if data files are directly in task_dir, copy/symlink them under 'input'
            if not dest_input.exists():
                dest_input.mkdir(parents=True, exist_ok=True)
                for f in task_dir.glob("*"):
                    if (
                        f.is_file()
                        and f.suffix in [".csv", ".tsv", ".txt", ".json"]
                        and f.name not in {"task_config.json", "submission.csv"}
                    ):
                        try:
                            os.symlink(f, dest_input / f.name)
                        except Exception:
                            shutil.copy(f, dest_input / f.name)
                print(f"ImplementationAgent: Prepared 'input' folder fallback at {dest_input}")

        # Descendants evolve the measured parent implementation. Only root-level
        # candidates start from the generated baseline.
        src_algo = Path(base_algorithm_path) if base_algorithm_path else (
            task_dir / "initial_algorithm.py"
        )
        if not src_algo.is_file():
            raise FileNotFoundError(f"Base algorithm does not exist: {src_algo}")
        with open(src_algo, 'r', encoding='utf-8') as f:
            original_code = f.read()
            
        # Load task config for metric info
        config_file = task_dir / "task_config.json"
        metric_name = "score"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                task_config = json.load(f)
            metric_name = task_config.get("metric_name", "score")
            metric_direction = task_config.get("metric_direction", metric_direction)
            
        # Never execute an artifact that failed verification.
        if model_card and model_card.get("verified") is not True:
            raise ValueError(
                f"Model card {model_card.get('artifact_id', '<unknown>')!r} is not verified"
            )
            
        # Get dataset analysis report to pass to LLM
        dataset_snapshot = ""
        report_file = task_dir / "dataset_analysis_report.txt"
        try:
            if report_file.exists():
                with open(report_file, 'r', encoding='utf-8') as f:
                    analysis_report = f.read()
            else:
                from .data_analyzer import run_dataset_analysis
                print(f"ImplementationAgent: Checking/running dataset analysis fallback for {task_dir.name}...")
                analysis_report = run_dataset_analysis(task_dir)
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(analysis_report)
                    
            dataset_snapshot = (
                "=== Dataset Analysis & Schema Report ===\n"
                f"{analysis_report}\n"
                "========================================\n"
            )
        except Exception as e:
            print(f"ImplementationAgent WARNING: Could not get dataset analysis report: {e}")
            dataset_snapshot = ""

        # Read the technique source code if it exists (local or global)
        technique_code = ""
        if model_card and "code_path" in model_card:
            category = model_card.get("category", "")
            code_name = model_card["code_path"]
            # Look in node_dir (local) or global store
            tech_file = node_dir / code_name
            if not tech_file.exists():
                tech_file = Path(self.project_root) / "memory_pool" / "l2_store" / category / code_name
            if tech_file.exists():
                try:
                    with open(tech_file, 'r', encoding='utf-8') as f:
                        technique_code = f.read()
                except Exception as e:
                    print(f"ImplementationAgent WARNING: Could not read technique code: {e}")

        tech_code_str = ""
        if technique_code:
            tech_code_str = f"Chosen Technique Source Code:\n```python\n{technique_code}\n```"

        # Call LLM to generate glue code. The artifact file is copied beside
        # algorithm.py, so importing an invented memory_pool package is incorrect.
        if model_card and technique_code:
            module_name = Path(model_card["code_path"]).stem
            entrypoint_signature = model_card.get("interface", {}).get("entrypoint", "")
            entrypoint_name = entrypoint_signature.split("(", 1)[0].strip()
            integration_instruction = (
                f"Import the verified local artifact exactly from module {module_name!r}; for example, "
                f"`from {module_name} import {entrypoint_name}`. Do not import from a `memory_pool` package "
                "and do not reimplement the artifact's internal logic."
            )

        else:
            integration_instruction = (
                "No verified artifact is available for this node. Improve the supplied parent baseline directly "
                "according to the technique plan, keep the script self-contained, and do not import any "
                "`memory_pool` module or imaginary artifact."
            )

        if fidelity not in FIDELITY_PROFILES:
            raise ValueError("fidelity must be 'screen', 'medium', or 'full'")
        fidelity_profile = FIDELITY_PROFILES[fidelity]
        operator_instruction = {
            "refine": "Modify only the highest-impact relevant component; preserve working parent behavior elsewhere.",
            "tune": "Preserve the model family and run a compact Optuna search with pruning and a fixed seed.",
            "diversify": "Favor a sound model or representation whose errors are likely less correlated with the parent.",
            "promote": "Preserve the parent method and evaluate it more rigorously at the requested fidelity.",
        }.get(operator or "", "Apply the requested technique as a focused change to the parent pipeline.")

        system_prompt = (
            "You are the Implementation Agent. Produce a complete executable revision of the supplied parent algorithm. "
            f"{integration_instruction} Ensure the output is valid Python code wrapped in a ```python block.\n"
            f"Search operator: {operator or 'initial'}. {operator_instruction}\n"
            f"Evaluation fidelity: {fidelity} ({json.dumps(fidelity_profile)}). These limits are mandatory.\n"
            f"Execution accelerator: {accelerator}; available={sorted(exposed_accelerators)}. "
            "The subprocess also exposes this value as AIBUILDAI_ACCELERATOR. When the selected accelerator is "
            "CUDA or MPS, use the framework-native GPU/device option for every compatible training component "
            "and move neural-network models and tensors to that device. Check that the framework backend is "
            "usable and fall back to CPU only when that model/library has no working backend for the selected "
            "accelerator. When the technique permits equivalent model families, prefer a GPU-capable learner "
            "over a CPU-only one. Do not send small preprocessing or unsupported scikit-learn estimators to a GPU. "
            "After training, report os.environ.get('AIBUILDAI_ACTUAL_ACCELERATOR', selected_device) in "
            "result.json; GPU-aware pool artifacts update this variable when they fall back.\n"
            "Import `prepare_evaluation_data` from the local `evaluation_contract` module and call "
            f"`X_eval, y_eval, row_ids, fold_ids, evaluation_meta = prepare_evaluation_data(train_data, '{fidelity}')`. "
            "Use X_eval/y_eval for every training and validation operation, use the supplied fold_ids exactly, "
            "and train final test-prediction models on all X_eval rows. Save one OOF prediction per scheduled "
            "row to oof_predictions.csv with columns row_id,target,prediction. Do not create a separate split.\n"
            "Fit every imputer, encoder, scaler, feature selector, and target-dependent transform on training folds only. "
            "Never use test-set statistics or target values in feature generation. Use stratification for classification.\n"
            "When practical, save validation predictions to 'oof_predictions.csv' with columns row_id,target,prediction.\n"
            f"IMPORTANT: At the END of your script, write a JSON file 'result.json' in the current directory:\n"
            f'  import json; json.dump({{"score": <float>, "metric": "{metric_name}", "direction": "{metric_direction}", '
            f'"cv_mean": <float>, "cv_std": <float>, "folds": <int>, "fidelity": "{fidelity}", '
            f'"accelerator": <actual "cpu"|"cuda"|"mps">}}, open("result.json", "w"))\n'
            "The score must be the cross-validation mean when CV is used, otherwise the held-out validation score."
        )
        
        technique_plan = technique_record.get("plan", "")
        
        prompt_model_card = dict(model_card or {})
        prompt_model_card.pop("code_content", None)
        if prompt_model_card.get("verification_log"):
            prompt_model_card["verification_log"] = str(
                prompt_model_card["verification_log"]
            )[-1200:]
        user_prompt = f"""
            Parent Code ({'measured ancestor' if base_algorithm_path else 'generated baseline'}):
            ```python
            {original_code}
            ```

            Chosen Technique Plan:
            {technique_plan}

            Model Card details:
            {json.dumps(prompt_model_card, indent=2) if prompt_model_card else "None"}

            {tech_code_str}

            {dataset_snapshot}

            {integration_instruction}
            Inherited reusable files: {inherited_files}
            Return the complete updated file content.
            """
        response = call_llm(system_prompt, user_prompt, model=self.model_name)
        
        # Clean markdown code block formatting
        clean_code = response
        if "```python" in response:
            clean_code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            clean_code = response.split("```")[1].split("```")[0]

        guard_issues = inspect_generated_code(clean_code)
        if guard_issues:
            print(
                "ImplementationAgent: Validation guard found leakage risks; "
                "requesting a pre-execution repair."
            )
            repair_response = call_llm(
                "You are an ML validation-safety reviewer. Repair every listed leakage defect while "
                "preserving the intended model and focused change. Fit preprocessing only on training "
                "folds. Return the complete corrected Python file in a ```python block.",
                f"""
Leakage defects:
{json.dumps(guard_issues, indent=2)}

Code:
```python
{clean_code}
```
""",
                model=self.model_name,
            )
            if "```python" in repair_response:
                clean_code = repair_response.split("```python", 1)[1].split("```", 1)[0]
            elif "```" in repair_response:
                clean_code = repair_response.split("```", 1)[1].split("```", 1)[0]
            else:
                clean_code = repair_response
            remaining_issues = inspect_generated_code(clean_code)
            if remaining_issues:
                raise ValueError(
                    "Generated implementation failed leakage guard after repair: "
                    + "; ".join(remaining_issues)
                )
            
        dest_code_file = node_dir / "algorithm.py"
        with open(dest_code_file, 'w', encoding='utf-8') as f:
            f.write(clean_code.strip())
            
        print(f"ImplementationAgent: Wrote glue code to {dest_code_file}")
                    
        cmd = [self.venv_python, "algorithm.py"]
        
        # Debug/Coder retry loop
        max_attempts = 3
        attempt = 0
        score = None
        score_source = "none"
        status = "completed"
        stdout = ""
        stderr = ""
        exit_code = 0
        timeout_kill = False
        diagnostics = ""
        result_data: Dict[str, Any] = {}
        
        while attempt < max_attempts:
            if attempt > 0:
                print(f"ImplementationAgent: Debug Attempt {attempt}/{max_attempts - 1} — Previous execution failed. Invoking coder debugging agent...")
                debug_system_prompt = (
                    "You are a Coder/Debugging Agent. The previous implementation of the machine learning code failed with an error.\n"
                    "Review the original baseline code, your previous generated code, and the error traceback/stderr output.\n"
                    "Identify the issue, fix it, and write the corrected code.\n"
                    f"{integration_instruction}\n"
                    "CRITICAL: Start your Python code response with a brief comment block explaining the cause of the error and how you are fixing it. "
                    "This helps trace execution logical errors correctly.\n"
                    "Ensure the output is valid Python code wrapped in a ```python block.\n"
                    "Fit preprocessing on training folds only; never derive preprocessing statistics from test data.\n"
                    "Keep using evaluation_contract.prepare_evaluation_data and its supplied fold_ids exactly; "
                    "write complete OOF predictions for the scheduled rows.\n"
                    f"The selected accelerator is {accelerator}, exposed as AIBUILDAI_ACCELERATOR. Preserve "
                    "framework-native GPU configuration when supported and retain a safe CPU fallback. Report "
                    "AIBUILDAI_ACTUAL_ACCELERATOR after training.\n"
                    f"IMPORTANT: At the END of your script, write a JSON file 'result.json' in the current directory:\n"
                    f'  import json; json.dump({{"score": <float>, "metric": "{metric_name}", "direction": "{metric_direction}", '
                    f'"cv_mean": <float>, "cv_std": <float>, "folds": <int>, "fidelity": "{fidelity}", '
                    f'"accelerator": <actual "cpu"|"cuda"|"mps">}}, open("result.json", "w"))\n'
                    "The score should be the final validation metric value."
                )
                
                debug_user_prompt = f"""
                Your Previous Generated Code (which failed):
                ```python
                {clean_code}
                ```

                Subprocess Exit Code: {exit_code}
                Subprocess Timeout: {timeout_kill}

                Traceback / Error Output (stderr):
                ```
                {stderr}
                ```

                Stdout output (if any):
                ```
                {stdout}
                ```

                Artifact interface metadata:
                {json.dumps(prompt_model_card.get('interface', {}), indent=2)}

                Required evaluation profile:
                {json.dumps(fidelity_profile)}

                Please debug and fix the code. Follow the integration instruction exactly and ensure all variables/datasets are loaded properly. Return the complete corrected code file.
                """
                response = call_llm(debug_system_prompt, debug_user_prompt, model=self.model_name)
                
                # Clean markdown code block formatting
                clean_code = response
                if "```python" in response:
                    clean_code = response.split("```python")[1].split("```")[0]
                elif "```" in response:
                    clean_code = response.split("```")[1].split("```")[0]

                debug_guard_issues = inspect_generated_code(clean_code)
                if debug_guard_issues:
                    stderr = (
                        "Static leakage guard rejected the debug revision:\n"
                        + "\n".join(debug_guard_issues)
                    )
                    stdout = ""
                    exit_code = -2
                    timeout_kill = False
                    attempt += 1
                    continue

                with open(dest_code_file, 'w', encoding='utf-8') as f:
                    f.write(clean_code.strip())
                print(f"ImplementationAgent: Wrote revised glue code to {dest_code_file}")

            # Execute with activity-based watchdog timeout
            timeout_kill = False
            stdout_lines = []
            stderr_lines = []
            
            import fcntl

            # Do not accept stale outputs from an earlier run in the same node folder.
            result_json_path = node_dir / "result.json"
            submission_path = node_dir / "submission" / "submission.csv"
            for stale_output in (
                result_json_path,
                submission_path,
                node_dir / "oof_predictions.csv",
                node_dir / "evaluation_manifest.json",
                node_dir / "fold_assignments.csv",
            ):
                if stale_output.exists() or stale_output.is_symlink():
                    stale_output.unlink()
            
            remaining_deadline = float(timeout) - (time.monotonic() - run_started)
            if remaining_deadline <= 0:
                exit_code = -1
                timeout_kill = True
                stderr = "overall experiment deadline exhausted before this attempt"
                break

            proc = subprocess.Popen(
                cmd,
                cwd=node_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=accelerator_subprocess_env(accelerator),
                start_new_session=True,
            )

            def terminate_process_group() -> None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    return
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            
            # Make stdout/stderr non-blocking
            for fd in (proc.stdout, proc.stderr):
                fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                
            start_time = time.time()
            last_output_time = time.time()
            hard_timeout = remaining_deadline
            inactivity_timeout = min(1200.0, hard_timeout)
            
            while True:
                ret = proc.poll()
                
                # Read non-blocking stdout bytes
                try:
                    out = proc.stdout.read()
                    if out:
                        decoded_out = out.decode('utf-8', errors='replace')
                        stdout_lines.append(decoded_out)
                        last_output_time = time.time()
                        sys.stdout.write(decoded_out)
                        sys.stdout.flush()
                except IOError:
                    pass
                    
                # Read non-blocking stderr bytes
                try:
                    err = proc.stderr.read()
                    if err:
                        decoded_err = err.decode('utf-8', errors='replace')
                        stderr_lines.append(decoded_err)
                        last_output_time = time.time()
                        sys.stderr.write(decoded_err)
                        sys.stderr.flush()
                except IOError:
                    pass
                    
                if ret is not None:
                    # Flush remaining buffers
                    try:
                        out = proc.stdout.read()
                        if out:
                            decoded_out = out.decode('utf-8', errors='replace')
                            stdout_lines.append(decoded_out)
                    except IOError:
                        pass
                    try:
                        err = proc.stderr.read()
                        if err:
                            decoded_err = err.decode('utf-8', errors='replace')
                            stderr_lines.append(decoded_err)
                    except IOError:
                        pass
                    exit_code = ret
                    break
                    
                now = time.time()
                elapsed = now - start_time
                inactive_elapsed = now - last_output_time
                
                if inactive_elapsed > inactivity_timeout:
                    print(f"\nImplementationAgent: INACTIVITY TIMEOUT after {inactivity_timeout}s with no output. Killing process.")
                    terminate_process_group()
                    exit_code = -1
                    timeout_kill = True
                    break
                    
                if elapsed > hard_timeout:
                    print(f"\nImplementationAgent: HARD TIMEOUT ceiling of {hard_timeout}s reached. Killing process.")
                    terminate_process_group()
                    exit_code = -1
                    timeout_kill = True
                    break
                    
                time.sleep(0.1)

            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)
            proc.stdout.close()
            proc.stderr.close()
                
            # Preserve individual failed-attempt diagnostics. error.log is reserved
            # for the final failed state so recovered nodes are not mislabeled.
            diagnostics = stdout + "\n" + stderr
            if stderr.strip():
                attempt_log_path = node_dir / f"attempt_{attempt + 1}.log"
                with open(attempt_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"Attempt={attempt + 1}\nexit_code={exit_code}\ntimeout_kill={timeout_kill}\n\n")
                    f.write("=== STDERR ===\n")
                    f.write(stderr)
                    f.write("\n\n=== STDOUT ===\n")
                    f.write(stdout)
            
            # Parse score — prefer structured result.json, fall back to stdout regex
            score = None
            score_source = "none"
            status = "completed"
            
            # Strategy 1: Parse result.json (structured contract)
            result_data = {}
            if exit_code == 0 and result_json_path.exists():
                try:
                    with open(result_json_path, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    parsed_score = result_data.get("score")
                    if parsed_score is not None:
                        score = float(parsed_score)
                        if not math.isfinite(score):
                            raise ValueError("score must be finite")
                        declared_direction = result_data.get("direction")
                        if declared_direction and declared_direction != metric_direction:
                            raise ValueError(
                                f"result direction {declared_direction!r} does not match {metric_direction!r}"
                            )
                        declared_metric = result_data.get("metric")
                        if declared_metric and declared_metric != metric_name:
                            raise ValueError(
                                f"result metric {declared_metric!r} does not match {metric_name!r}"
                            )
                        declared_fidelity = result_data.get("fidelity")
                        if declared_fidelity and declared_fidelity != fidelity:
                            raise ValueError(
                                f"result fidelity {declared_fidelity!r} does not match {fidelity!r}"
                            )
                        declared_accelerator = result_data.get("accelerator")
                        if enforce_evaluation_contract and declared_accelerator is None:
                            raise ValueError(
                                "result must declare the accelerator actually used"
                            )
                        if declared_accelerator is not None:
                            declared_accelerator = str(declared_accelerator).lower()
                            if declared_accelerator not in {"cpu", "cuda", "mps"}:
                                raise ValueError(
                                    "result accelerator must be cpu, cuda, or mps"
                                )
                            if declared_accelerator not in {"cpu", accelerator}:
                                raise ValueError(
                                    f"result claims {declared_accelerator!r}, but this node "
                                    f"selected {accelerator!r}"
                                )
                            result_data["accelerator"] = declared_accelerator
                        if result_data.get("cv_std") is not None:
                            cv_std = float(result_data["cv_std"])
                            if not math.isfinite(cv_std) or cv_std < 0:
                                raise ValueError("cv_std must be finite and non-negative")
                        if result_data.get("folds") is not None:
                            folds = int(result_data["folds"])
                            if folds < 1:
                                raise ValueError("folds must be positive")
                        if enforce_evaluation_contract:
                            contract_validation = validate_evaluation_outputs(
                                node_dir, fidelity, metric_name
                            )
                            result_data.update(contract_validation)
                            score = float(contract_validation["cv_mean"])
                            result_data["score"] = score
                            result_data["metric"] = metric_name
                            result_data["direction"] = metric_direction
                            with open(result_json_path, "w", encoding="utf-8") as f:
                                json.dump(result_data, f, indent=2)
                        score_source = "result.json"
                        print(f"ImplementationAgent: Score from result.json: {score} "
                              f"(metric={result_data.get('metric', '?')}, direction={result_data.get('direction', '?')})")
                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                    OverflowError,
                    OSError,
                ) as e:
                    score = None
                    score_source = "none"
                    stderr = (stderr + "\nResult contract error: " + str(e)).strip()
                    diagnostics = stdout + "\n" + stderr
                    print(f"ImplementationAgent: WARNING: result.json exists but couldn't parse: {e}")
            
            # Strategy 2: Regex fallback on stdout (handles negatives and scientific notation)
            if (
                score is None
                and exit_code == 0
                and not timeout_kill
                and not enforce_evaluation_contract
            ):
                # Match patterns like: "Score: 0.93245", "AUC: -0.123", "accuracy = 9.5e-3"
                score_matches = re.findall(
                    r'(?:score|auc|accuracy|metric|rmse|mae|loss|f1)[:\s=]+(-?[0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)',
                    diagnostics, re.IGNORECASE
                )
                if score_matches:
                    try:
                        score = float(score_matches[-1])
                        if not math.isfinite(score):
                            raise ValueError("score must be finite")
                        score_source = "stdout_regex"
                        print(f"ImplementationAgent: Score from stdout regex (fallback): {score}")
                    except ValueError:
                        score = None
                        pass
            
            # If successful execution and a score was found, stop debugging
            if exit_code == 0 and score is not None:
                break
                
            attempt += 1
        
        # A crashing process is always a failed run, even if it wrote a partial score.
        if exit_code != 0:
            status = "failed"
            score = None
            score_source = "none"
            if timeout_kill:
                print(f"ImplementationAgent: FAILED — timeout after {timeout}s (exit_code={exit_code})")
            else:
                print(f"ImplementationAgent: FAILED — subprocess crashed (exit_code={exit_code}). "
                      f"See {node_dir / 'error.log'} for details.")
        elif score is None:
            # exit_code == 0 but no score produced — unusual but not a crash
            status = "failed"
            print(f"ImplementationAgent: FAILED — subprocess exited cleanly but produced no score.")
        else:
            (node_dir / "error.log").unlink(missing_ok=True)
            print(f"ImplementationAgent: Execution completed — exit_code={exit_code}, score={score}, "
                  f"source={score_source}, timeout_kill={timeout_kill}")

        if status == "failed":
            with open(node_dir / "error.log", "w", encoding="utf-8") as f:
                f.write(stderr or diagnostics or "experiment failed without diagnostics")

        actual_accelerator = (
            result_data.get("accelerator", accelerator)
            if status == "completed"
            else None
        )
        execution_resource["reported_accelerator"] = actual_accelerator
        with open(node_dir / "execution_resource.json", "w", encoding="utf-8") as f:
            json.dump(execution_resource, f, indent=2)
            f.write("\n")
        
        return {
            "status": status,
            "score": score,
            "score_source": score_source,
            "exit_code": exit_code,
            "timeout_kill": timeout_kill,
            "stdout": stdout,
            "stderr": stderr,
            "diagnostics": diagnostics,
            "code_path": str(dest_code_file),
            "base_code_path": str(src_algo),
            "parent_node_dir": str(parent_node_dir) if parent_node_dir else None,
            "inherited_files": inherited_files,
            "operator": operator,
            "fidelity": fidelity,
            "accelerator": actual_accelerator,
            "selected_accelerator": accelerator,
            "elapsed_seconds": time.monotonic() - run_started,
            "validation": {
                key: result_data.get(key)
                for key in (
                    "cv_mean", "cv_std", "folds", "fold_scores", "seed",
                    "fidelity", "row_count", "source_row_count",
                    "fold_assignment_sha256",
                )
                if result_data.get(key) is not None
            },
            "oof_path": (
                str(node_dir / "oof_predictions.csv")
                if (node_dir / "oof_predictions.csv").is_file()
                else None
            ),
        }
