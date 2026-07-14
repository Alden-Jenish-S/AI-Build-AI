import re
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any
from .llm_utils import call_llm

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

    def run(self, node_dir: Path, technique_record: dict, task_dir: Path,
            timeout: int = 300, metric_direction: str = "maximize") -> Dict[str, Any]:
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
        node_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy dataloader to node_dir so imports work locally
        src_loader = task_dir / "initial_dataloader.py"
        dest_loader = node_dir / "initial_dataloader.py"
        if src_loader.exists():
            import shutil
            shutil.copy(src_loader, dest_loader)
            
        # Copy the memory pool technique python file to node_dir so it can be imported (Fixes ModuleNotFoundError)
        model_card = technique_record.get("model_card", {})
        if model_card and "category" in model_card and "code_path" in model_card:
            category = model_card["category"]
            code_name = model_card["code_path"]
            src_tech = Path(self.project_root) / "memory_pool" / "l2_store" / category / code_name
            dest_tech = node_dir / code_name
            if src_tech.exists():
                import shutil
                shutil.copy(src_tech, dest_tech)
                print(f"ImplementationAgent: Copied technique from {src_tech} to {dest_tech}")
            else:
                print(f"ImplementationAgent WARNING: Technique code file not found at {src_tech}")

        # Symlink input directory so dataloader can find datasets (Fixes FileNotFoundError)
        src_input = task_dir / "input"
        dest_input = node_dir / "input"
        if src_input.exists():
            if not dest_input.exists():
                try:
                    os.symlink(src_input, dest_input)
                except Exception:
                    import shutil
                    if src_input.is_dir():
                        shutil.copytree(src_input, dest_input)
        else:
            # Fallback: if data files are directly in task_dir, copy/symlink them under 'input'
            if not dest_input.exists():
                dest_input.mkdir(parents=True, exist_ok=True)
                import shutil
                for f in task_dir.glob("*"):
                    if f.is_file() and f.suffix in [".csv", ".tsv", ".txt", ".json"] and f.name != "task_config.json":
                        try:
                            os.symlink(f, dest_input / f.name)
                        except Exception:
                            shutil.copy(f, dest_input / f.name)
                print(f"ImplementationAgent: Prepared 'input' folder fallback at {dest_input}")

        # Read the initial algorithm skeleton
        src_algo = task_dir / "initial_algorithm.py"
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
            
        # Check verified flag on model card
        if model_card and model_card.get("verified") is False:
            print(f"ImplementationAgent WARNING: Model card '{model_card.get('artifact_id', '?')}' is NOT verified. Proceeding with caution.")
            
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

        # Call LLM to generate glue code
        system_prompt = (
            "You are the Implementation Agent. Modify the original algorithm code to integrate the selected "
            "memory pool technique. You must directly import the technique from the memory pool rather than "
            "regenerating its logic. Ensure the output is valid Python code wrapped in a ```python block.\n"
            f"IMPORTANT: At the END of your script, write a JSON file 'result.json' in the current directory:\n"
            f'  import json; json.dump({{"score": <float>, "metric": "{metric_name}", "direction": "{metric_direction}"}}, open("result.json", "w"))\n'
            "The score should be the final validation metric value."
        )
        
        technique_plan = technique_record.get("plan", "")
        
        user_prompt = f"""
            Original Code:
            ```python
            {original_code}
            ```

            Chosen Technique Plan:
            {technique_plan}

            Model Card details:
            {json.dumps(model_card, indent=2) if model_card else "None"}

            {tech_code_str}

            {dataset_snapshot}

            Please modify the code to import and use the artifact. Return the complete updated file content.
            """
        response = call_llm(system_prompt, user_prompt, model=self.model_name)
        
        # Clean markdown code block formatting
        clean_code = response
        if "```python" in response:
            clean_code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            clean_code = response.split("```")[1].split("```")[0]
            
        dest_code_file = node_dir / "algorithm.py"
        with open(dest_code_file, 'w', encoding='utf-8') as f:
            f.write(clean_code.strip())
            
        print(f"ImplementationAgent: Wrote glue code to {dest_code_file}")
                    
        cmd = [self.venv_python, "algorithm.py"]
        
        # Debug/Coder retry loop
        max_attempts = 5
        attempt = 0
        score = None
        score_source = "none"
        status = "completed"
        stdout = ""
        stderr = ""
        exit_code = 0
        timeout_kill = False
        diagnostics = ""
        
        while attempt < max_attempts:
            if attempt > 0:
                print(f"ImplementationAgent: Debug Attempt {attempt}/{max_attempts - 1} — Previous execution failed. Invoking coder debugging agent...")
                debug_system_prompt = (
                    "You are a Coder/Debugging Agent. The previous implementation of the machine learning code failed with an error.\n"
                    "Review the original baseline code, your previous generated code, and the error traceback/stderr output.\n"
                    "Identify the issue, fix it, and write the corrected code.\n"
                    "CRITICAL: Start your Python code response with a brief comment block explaining the cause of the error and how you are fixing it. "
                    "This helps trace execution logical errors correctly.\n"
                    "Ensure the output is valid Python code wrapped in a ```python block.\n"
                    f"IMPORTANT: At the END of your script, write a JSON file 'result.json' in the current directory:\n"
                    f'  import json; json.dump({{"score": <float>, "metric": "{metric_name}", "direction": "{metric_direction}"}}, open("result.json", "w"))\n'
                    "The score should be the final validation metric value."
                )
                
                debug_user_prompt = f"""
                Original Baseline Code:
                ```python
                {original_code}
                ```

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

                {tech_code_str}

                {dataset_snapshot}

                Please debug and fix the code. Ensure the imported technique is used correctly and all variables/datasets are loaded properly. Return the complete corrected code file.
                """
                response = call_llm(debug_system_prompt, debug_user_prompt, model=self.model_name)
                
                # Clean markdown code block formatting
                clean_code = response
                if "```python" in response:
                    clean_code = response.split("```python")[1].split("```")[0]
                elif "```" in response:
                    clean_code = response.split("```")[1].split("```")[0]
                    
                with open(dest_code_file, 'w', encoding='utf-8') as f:
                    f.write(clean_code.strip())
                print(f"ImplementationAgent: Wrote revised glue code to {dest_code_file}")

            # Execute with activity-based watchdog timeout
            timeout_kill = False
            stdout_lines = []
            stderr_lines = []
            
            import time
            import fcntl
            import os
            
            proc = subprocess.Popen(
                cmd,
                cwd=node_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Make stdout/stderr non-blocking
            for fd in (proc.stdout, proc.stderr):
                fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                
            start_time = time.time()
            last_output_time = time.time()
            inactivity_timeout = 120.0
            hard_timeout = 1800.0  # 30-minute absolute ceiling
            
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
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    exit_code = -1
                    timeout_kill = True
                    break
                    
                if elapsed > hard_timeout:
                    print(f"\nImplementationAgent: HARD TIMEOUT ceiling of {hard_timeout}s reached. Killing process.")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    exit_code = -1
                    timeout_kill = True
                    break
                    
                time.sleep(0.1)
                
            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)
                
            # Save stderr to error.log for debugging
            diagnostics = stdout + "\n" + stderr
            if stderr.strip():
                error_log_path = node_dir / "error.log"
                with open(error_log_path, 'w', encoding='utf-8') as f:
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
            result_json_path = node_dir / "result.json"
            if result_json_path.exists():
                try:
                    with open(result_json_path, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    parsed_score = result_data.get("score")
                    if parsed_score is not None:
                        score = float(parsed_score)
                        score_source = "result.json"
                        print(f"ImplementationAgent: Score from result.json: {score} "
                              f"(metric={result_data.get('metric', '?')}, direction={result_data.get('direction', '?')})")
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"ImplementationAgent: WARNING: result.json exists but couldn't parse: {e}")
            
            # Strategy 2: Regex fallback on stdout (handles negatives and scientific notation)
            if score is None and not timeout_kill:
                # Match patterns like: "Score: 0.93245", "AUC: -0.123", "accuracy = 9.5e-3"
                score_matches = re.findall(
                    r'(?:score|auc|accuracy|metric|rmse|mae|loss|f1)[:\s=]+(-?[0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)',
                    diagnostics, re.IGNORECASE
                )
                if score_matches:
                    try:
                        score = float(score_matches[-1])
                        score_source = "stdout_regex"
                        print(f"ImplementationAgent: Score from stdout regex (fallback): {score}")
                    except ValueError:
                        pass
            
            # If successful execution and a score was found, stop debugging
            if exit_code == 0 and score is not None:
                break
                
            attempt += 1
        
        # Determine final status: if exit_code != 0 AND no score was produced, it's a failure
        if exit_code != 0 and score is None:
            status = "failed"
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
            print(f"ImplementationAgent: Execution completed — exit_code={exit_code}, score={score}, "
                  f"source={score_source}, timeout_kill={timeout_kill}")
        
        return {
            "status": status,
            "score": score,
            "score_source": score_source,
            "exit_code": exit_code,
            "timeout_kill": timeout_kill,
            "stdout": stdout,
            "stderr": stderr,
            "diagnostics": diagnostics,
            "code_path": str(dest_code_file)
        }
