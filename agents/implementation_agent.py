"""Generate, run, and repair one self-contained task implementation."""

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

from runtime_utils import expose_task_data, run_supervised_process, sanitized_subprocess_env

from .llm_utils import call_llm
from .submission_validator import SubmissionValidator
from .task_analyzer import TaskAnalysis
from .web_search import search_web


def _python_source(response: str) -> str:
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return max(fenced, key=len).strip() + "\n"
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    text = re.sub(r"(?s)<thinking>.*?</thinking>", "", text).strip()
    return text.strip() + "\n"


def _tail(text: str, limit: int = 7000) -> str:
    return text if len(text) <= limit else text[-limit:]


def _compact_code(code: str, limit: int) -> str:
    """Retain imports/setup and the final pipeline/output logic."""
    if len(code) <= limit:
        return code
    head = max(2000, limit // 3)
    return code[:head] + "\n# ... middle omitted from prompt ...\n" + code[-(limit - head):]


class ImplementationAgent:
    """One compact code-writing call followed by bounded execution repairs."""

    def __init__(
        self,
        venv_python_path: str | None = None,
        model_name: str | None = None,
        submission_validator: SubmissionValidator | None = None,
    ) -> None:
        self.python = str(venv_python_path or sys.executable)
        self.model_name = model_name
        self.submission_validator = submission_validator or SubmissionValidator()

    @staticmethod
    def _log(node_name: str, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{timestamp}] ImplementationAgent[{node_name}]: {message}", flush=True)

    @staticmethod
    def _output_path(node_dir: Path) -> Path | None:
        preferred = node_dir / "submission" / "submission.csv"
        root_csv = node_dir / "submission.csv"
        if preferred.is_file():
            return preferred
        if root_csv.is_file():
            preferred.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root_csv, preferred)
            return preferred
        submission_dir = node_dir / "submission"
        if submission_dir.is_dir():
            files = [path for path in sorted(submission_dir.rglob("*")) if path.is_file()]
            if files:
                return submission_dir
        return None

    @staticmethod
    def _read_result(node_dir: Path) -> dict[str, Any]:
        path = node_dir / "result.json"
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _score(payload: dict[str, Any], stdout: str, direction: str) -> float:
        try:
            score = float(payload.get("score"))
            if math.isfinite(score):
                return score
        except (TypeError, ValueError):
            pass
        matches = re.findall(
            r"(?i)(?:score|auc|accuracy|f1|dice|iou|silhouette|rmse|mae|loss)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
            stdout,
        )
        if matches:
            try:
                score = float(matches[-1])
                if math.isfinite(score):
                    return score
            except ValueError:
                pass
        return 1e30 if direction == "minimize" else -1e30

    def _prompt(
        self,
        analysis: TaskAnalysis,
        plan: str,
        parent_code: str | None,
        companion_code: str | None,
        candidate_code: str | None,
        feedback: str,
        web_notes: str,
    ) -> str:
        parent = ""
        if parent_code:
            parent = (
                "\nA working parent implementation is included below. Preserve its working "
                "paths and output behavior; edit it narrowly for this plan.\n"
                f"```python\n{_compact_code(parent_code, 20000)}\n```\n"
            )
        companion = ""
        if companion_code:
            companion = (
                "\nA second measured implementation is included for the selected merge. "
                "Use only complementary logic from it and compare the merged candidate against "
                "both measured parents.\n"
                f"```python\n{_compact_code(companion_code, 14000)}\n```\n"
            )
        repair = ""
        if feedback:
            repair = (
                "\nThe previous execution did not finish correctly. Repair the actual error; "
                "do not invent imports from evaluation, contracts, loaders, or this agent system.\n"
                + (
                    "Previous attempted program:\n"
                    f"```python\n{_compact_code(candidate_code, 16000)}\n```\n"
                    if candidate_code
                    else ""
                )
                + f"Execution output:\n{feedback}\n"
            )
        research = f"\nRelevant web-search notes & literature insights:\n{web_notes[:5000]}\n" if web_notes else ""
        runtime_paths = [f"input/{item['path']}" for item in analysis.files]
        displayed_paths: list[str] = []
        displayed_chars = 0
        for path in runtime_paths:
            rendered = f"- `{path}`"
            if displayed_paths and displayed_chars + len(rendered) + 1 > 6000:
                break
            displayed_paths.append(rendered)
            displayed_chars += len(rendered) + 1
        exact_paths = "\n".join(displayed_paths)
        if len(runtime_paths) > len(displayed_paths):
            exact_paths += (
                f"\n- … {len(runtime_paths) - len(displayed_paths)} more paths are described "
                "by the inventory above"
            )
        return f"""
Write one complete, self-contained Python program for this task.

{analysis.prompt_context(20000)}

Implementation plan:
{plan[:8000]}
{parent}{companion}{repair}{research}
Exact runtime input paths available under input/:
{exact_paths}

CRITICAL RULES & AGENT REASONING WORKFLOW:

1. THINK BEFORE YOU DECIDE & WRITE CODE:
   Begin your response with a <thinking> ... </thinking> block where you step-by-step:
   a. Analyze the task goal, score metric ({analysis.metric} {analysis.direction}), and target deliverable.
   b. Resource & File Inspection: Dynamically inspect the actual files and schemas listed under `input/`. Do NOT assume hardcoded file names, columns, or data structures. Check available resources at runtime.
   c. Model & Architecture Suitability: Verify if the proposed approach actually suits the observed data modalities (tabular, vision, text, audio, time-series, multimodal).
   d. Custom Task-Tailored Engineering: Do not limit yourself to standard library `.fit()` / `.predict()` wrapper calls. Implement custom PyTorch `nn.Module` blocks, custom data augmentation functions (text noise/masking/synonym swapping, tabular noise injection/mixup, custom transforms), metric-aligned loss functions/objectives (custom GBDT objective, focal/ranking/asymmetric loss), or custom feature engineering transformers whenever appropriate for the task.
   e. Lightweight vs Heavy Models: Keep in mind that well-tailored compact models with custom modules, custom augmentations, or domain features often outperform heavy generic off-the-shelf models. Choose the optimal tradeoff for validation score and execution efficiency.
   f. If repairing a failed run or tuning, analyze the execution traceback, diagnose why it failed, and verify if fixing/tuning the current model will improve performance vs refactoring the approach.
   g. Outline your end-to-end data pipeline, validation split strategy, model training, and exact output generation.

2. RUNTIME & SUBMISSION CONSTRAINTS:
   - The program runs from an isolated node directory. Every task-owned file is available under `input/<inventory path>`.
   - Do NOT hardcode file names or column names. Inspect actual files and dataframes dynamically at runtime from `input/`.
   - Treat `input/` as immutable: never write, rename, move, or delete files in `input/`.
   - Do not import repository-internal modules (`evaluation`, `contracts`, `agents`, `core`, `modalities`). Use standard installed Python packages only.
   - Choose an honest local validation strategy matching the metric ({analysis.metric} {analysis.direction}).
   - Bound memory and runtime. Include graceful fallbacks if optional hardware/accelerators are unavailable.
   - Produce the exact requested deliverable under `submission/`. For sample CSV deliverables, preserve row order and exact column headers.
   - A task-provided sample output is a structural contract, regardless of its file type. Without a sample, native non-empty files or directory bundles are allowed.
   - The agent validates the produced artifact independently. A successful process or self-reported score cannot make an invalid output eligible.
   - At the end, write a small `result.json` object containing `score`, `metric`, `direction`, `output`, and `diagnostics`.

3. OUTPUT FORMAT:
   First return your step-by-step reasoning in a <thinking> ... </thinking> block.
   Then return your complete Python code in ONE fenced block ```python ... ```.
""".strip()

    @staticmethod
    def _web_query(stderr: str) -> str:
        meaningful = [line.strip() for line in stderr.splitlines() if line.strip()]
        last = meaningful[-1] if meaningful else stderr.strip()
        return f"Python debugging {last}"[:99]

    def run(
        self,
        node_dir: Path,
        plan: str,
        task_dir: Path,
        task_analysis: TaskAnalysis,
        *,
        parent_code: str | None = None,
        companion_code: str | None = None,
        operator: str = "root",
        node_label: str | None = None,
        max_debug_attempts: int = 4,
        stall_seconds: float = 1200.0,
        hard_limit_seconds: float | None = None,
    ) -> dict[str, Any]:
        node_dir = Path(node_dir)
        display_name = node_label or node_dir.name
        node_dir.mkdir(parents=True, exist_ok=True)
        linked_inputs = expose_task_data(Path(task_dir), node_dir)
        self._log(display_name, f"Exposed {len(linked_inputs)} task files under input/.")
        source_file = node_dir / "algorithm.py"
        feedback = ""
        web_notes = ""
        last_code = parent_code or ""
        last_attempt_code = ""
        diagnostics: list[str] = []

        for attempt in range(1, max(1, int(max_debug_attempts)) + 1):
            self._log(
                display_name,
                f"Generating implementation attempt {attempt}/{max_debug_attempts} ({operator}).",
            )
            prompt = self._prompt(
                task_analysis,
                plan,
                parent_code,
                companion_code,
                last_attempt_code if feedback else None,
                feedback,
                web_notes,
            )
            try:
                response = call_llm(
                    "You write reliable, direct, high-performing SOTA implementations from observed task files.",
                    prompt,
                    model=self.model_name,
                    temperature=0.1 if attempt > 1 else 0.2,
                )
                code = _python_source(response)
                if len(code) < 80:
                    raise ValueError("code response was empty or too short")
                last_code = code
                last_attempt_code = code
                self._log(display_name, f"Received {len(code)} characters of executable code.")
            except Exception as exc:
                diagnostics.append(f"attempt {attempt}: LLM response error: {exc}")
                self._log(display_name, f"LLM attempt {attempt} failed: {exc}")
                (node_dir / f"attempt_{attempt}.log").write_text(
                    f"LLM response error: {type(exc).__name__}: {exc}\n",
                    encoding="utf-8",
                )
                feedback = _tail(str(exc))
                continue

            source_file.write_text(code, encoding="utf-8")
            # A repaired attempt must prove its own deliverable instead of
            # inheriting stale output from a crashed prior attempt.
            if attempt > 1:
                for stale_file in (node_dir / "result.json", node_dir / "submission.csv"):
                    try:
                        stale_file.unlink()
                    except FileNotFoundError:
                        pass
                stale_submission = node_dir / "submission"
                if stale_submission.is_dir():
                    shutil.rmtree(stale_submission)
            child_env = sanitized_subprocess_env()
            child_env.update(
                {
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": os.getenv("AIBUILDAI_MODEL_THREADS", "4"),
                    "MKL_NUM_THREADS": os.getenv("AIBUILDAI_MODEL_THREADS", "4"),
                }
            )
            try:
                self._log(display_name, f"Executing {source_file.name}; child logs follow.")
                completed = run_supervised_process(
                    [self.python, source_file.name],
                    cwd=node_dir,
                    env=child_env,
                    stall_seconds=stall_seconds,
                    hard_limit_seconds=hard_limit_seconds,
                    activity_root=node_dir,
                    stdout_stream=sys.stdout,
                    stderr_stream=sys.stderr,
                    label=f"Implementation {display_name} attempt {attempt}",
                )
            except Exception as exc:
                feedback = _tail(f"Process launch failed: {exc}")
                diagnostics.append(f"attempt {attempt}: {feedback}")
                (node_dir / f"attempt_{attempt}.log").write_text(
                    feedback + "\n", encoding="utf-8"
                )
                self._log(display_name, feedback)
                continue

            output = self._output_path(node_dir)
            payload = self._read_result(node_dir)
            combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
            (node_dir / f"attempt_{attempt}.log").write_text(
                "\n".join(
                    (
                        f"returncode: {completed.returncode}",
                        f"elapsed_seconds: {completed.elapsed_seconds:.3f}",
                        "",
                        "STDOUT:",
                        _tail(completed.stdout, 20000),
                        "",
                        "STDERR:",
                        _tail(completed.stderr, 20000),
                        "",
                    )
                ),
                encoding="utf-8",
            )
            diagnostics.append(
                f"attempt {attempt}: exit={completed.returncode}; elapsed={completed.elapsed_seconds:.1f}s\n{_tail(combined, 3000)}"
            )
            self._log(
                display_name,
                f"Attempt {attempt} exited {completed.returncode} after {completed.elapsed_seconds:.1f}s.",
            )

            if completed.returncode == 0 and output is not None:
                validation = self.submission_validator.validate(
                    output,
                    task_analysis,
                    allowed_root=node_dir,
                )
                if not validation.valid:
                    feedback = _tail(validation.feedback())
                    diagnostics.append(
                        f"attempt {attempt}: output validation failed\n{feedback}"
                    )
                    with (node_dir / f"attempt_{attempt}.log").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(f"\nSUBMISSION VALIDATION:\n{feedback}\n")
                    self._log(display_name, "Generated output failed submission validation.")
                    for message in validation.errors:
                        self._log(display_name, f"Validation error: {message}")
                    continue
                output = validation.output_path
                score = self._score(payload, completed.stdout, task_analysis.direction)
                self._log(display_name, f"Accepted runnable output {output}; score={score:.8g}.")
                return {
                    "status": "completed",
                    "score": score,
                    "metric": str(payload.get("metric") or task_analysis.metric),
                    "direction": str(payload.get("direction") or task_analysis.direction),
                    "output": str(output),
                    "diagnostics": "\n\n".join(diagnostics)[-10000:],
                    "attempts": attempt,
                    "operator": operator,
                    "submission_validation": validation.to_dict(),
                    "code": last_code,
                }

            reason = (
                f"exit code {completed.returncode}; "
                + ("no deliverable was written under submission/." if output is None else "deliverable exists but execution failed.")
            )
            feedback = _tail(f"{reason}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
            if attempt < max_debug_attempts and os.getenv("AIBUILDAI_WEB_SEARCH", "1").lower() not in {"0", "false", "no"}:
                try:
                    self._log(display_name, "Searching the web for the concrete execution error.")
                    web_notes = search_web(self._web_query(combined or reason))
                except Exception as exc:
                    web_notes = f"Web search was unavailable: {exc}"
                    self._log(display_name, web_notes)

        self._log(display_name, f"All {max(1, int(max_debug_attempts))} repair attempts were exhausted.")
        return {
            "status": "failed",
            "score": None,
            "metric": task_analysis.metric,
            "direction": task_analysis.direction,
            "output": None,
            "diagnostics": "\n\n".join(diagnostics)[-12000:] or feedback,
            "attempts": max(1, int(max_debug_attempts)),
            "operator": operator,
            "code": last_code,
        }
