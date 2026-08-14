"""Adaptive local evidence collection and isolated council script execution."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import logging
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from runtime_utils import expose_task_data, run_supervised_process, sanitized_subprocess_env

from ..llm_utils import call_llm
from ..modality_policy import predictive_modality_inventory
from ..task_analyzer import TaskAnalysis
from .contracts import content_hash


_TABLE_SUFFIXES = {".csv", ".tsv", ".parquet", ".feather"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_AUDIO_SUFFIXES = {".wav", ".flac", ".ogg"}
_PROHIBITED_NAME_PATTERNS = (
    re.compile(r"(?:^|[_-])test[_-]?(?:labels?|targets?|truth|answers?)(?:[_-]|$)", re.I),
    re.compile(r"(?:^|[_-])holdout[_-]?(?:labels?|targets?|truth|answers?)(?:[_-]|$)", re.I),
    re.compile(r"(?:^|[_-])ground[_-]?truth(?:[_-]|$)", re.I),
    re.compile(r"(?:^|[_-])answer[_-]?key(?:[_-]|$)", re.I),
    re.compile(r"(?:^|[_-])grader[_-]?(?:output|labels?|truth)(?:[_-]|$)", re.I),
    re.compile(r"(?:^|[_-])(?:labels?|targets?|truth|answers?)[_-]test(?:[_-]|$)", re.I),
    re.compile(r"(?:^|[_-])(?:test[_-]?y|y[_-]?test)(?:[_-]|$)", re.I),
    re.compile(r"^(?:solution|answers?|answer[_-]?key)$", re.I),
)


def classify_input_access(analysis: TaskAnalysis) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    """Default-deny obvious answer artifacts while retaining ordinary task data."""
    allowed: list[str] = []
    prohibited: list[dict[str, str]] = []
    for item in analysis.files:
        relative = str(item.get("path") or "")
        stem = Path(relative).stem
        lower = relative.casefold()
        reason = ""
        if any(pattern.search(stem) for pattern in _PROHIBITED_NAME_PATTERNS):
            reason = "looks like held-out labels, ground truth, or a benchmark answer artifact"
        elif any(
            marker in lower
            for marker in (
                "/private_test/labels",
                "/hidden_test/labels",
                "grading/answers",
                "grader/answers",
            )
        ):
            reason = "path indicates private evaluation answers"
        if reason:
            prohibited.append({"path": relative, "reason": reason})
        else:
            allowed.append(relative)
    return tuple(allowed), tuple(prohibited)


def _safe_number(value: object) -> float | int | None:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else round(number, 8)


def _bounded_frame(path: Path, max_rows: int):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(
            path,
            sep="\t" if suffix == ".tsv" else ",",
            nrows=max_rows,
            low_memory=False,
        )
    if suffix == ".parquet":
        import pyarrow.parquet as parquet

        batch = next(parquet.ParquetFile(path).iter_batches(batch_size=max_rows), None)
        return batch.to_pandas() if batch is not None else pd.DataFrame()
    if suffix == ".feather":
        import pyarrow.feather as feather

        return feather.read_table(path, memory_map=True).slice(0, max_rows).to_pandas()
    raise ValueError(f"unsupported table format: {suffix}")


def _table_diagnostics(
    root: Path,
    analysis: TaskAnalysis,
    allowed: set[str],
    *,
    max_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    tables: list[dict[str, Any]] = []
    frames: dict[str, Any] = {}
    target_names = {
        value.strip()
        for value in str(analysis.target or "").split(",")
        if value.strip() and "target variables" not in value
    }
    for item in analysis.files:
        relative = str(item.get("path") or "")
        if relative not in allowed or Path(relative).suffix.lower() not in _TABLE_SUFFIXES:
            continue
        path = root / relative
        try:
            frame = _bounded_frame(path, max_rows)
        except Exception as exc:
            tables.append(
                {
                    "path": relative,
                    "status": "unreadable",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            )
            continue
        frames[relative] = frame
        dtypes = Counter(
            "numeric"
            if pd.api.types.is_numeric_dtype(frame[column])
            else "datetime"
            if pd.api.types.is_datetime64_any_dtype(frame[column])
            else "text_or_categorical"
            for column in frame.columns
        )
        missing = frame.isna().mean().sort_values(ascending=False)
        cardinalities = frame.nunique(dropna=True).sort_values(ascending=False)
        stem = Path(relative).stem.casefold()
        is_output_template = any(
            marker in stem for marker in ("sample_submission", "sample_output", "output_template")
        )
        target_candidates = (
            []
            if is_output_template
            else [column for column in frame.columns if str(column) in target_names]
        )
        target_summary: dict[str, Any] = {}
        for column in target_candidates[:20]:
            series = frame[column]
            clean = series.dropna()
            counts = clean.astype(str).value_counts(normalize=True).head(20)
            target_summary[str(column)] = {
                "observed": int(len(clean)),
                "missing_fraction": _safe_number(series.isna().mean()),
                "unique": int(clean.nunique()),
                "top_distribution": {
                    str(key): _safe_number(value) for key, value in counts.items()
                },
                "numeric_quantiles": (
                    {
                        str(key): _safe_number(value)
                        for key, value in pd.to_numeric(clean, errors="coerce")
                        .quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
                        .items()
                    }
                    if pd.api.types.is_numeric_dtype(series) and len(clean)
                    else {}
                ),
            }
        tables.append(
            {
                "path": relative,
                "status": "profiled",
                "sampled_rows": int(len(frame)),
                "sample_limit": max_rows,
                "column_count": int(len(frame.columns)),
                "dtype_families": dict(dtypes),
                "duplicate_row_fraction": _safe_number(frame.duplicated().mean()) if len(frame) else 0,
                "top_missing_fraction": {
                    str(key): _safe_number(value) for key, value in missing.head(20).items()
                },
                "top_cardinalities": {
                    str(key): int(value) for key, value in cardinalities.head(30).items()
                },
                "constant_columns": [
                    str(column) for column, count in cardinalities.items() if int(count) <= 1
                ][:100],
                "target_summary": target_summary,
            }
        )

    drift: dict[str, Any] = {"available": False}
    train_path = next((path for path in frames if "train" in Path(path).stem.casefold()), None)
    test_path = next((path for path in frames if "test" in Path(path).stem.casefold()), None)
    if train_path and test_path:
        train = frames[train_path]
        test = frames[test_path]
        common = [column for column in train.columns if column in test.columns]
        numeric = [
            column
            for column in common
            if pd.api.types.is_numeric_dtype(train[column])
            and pd.api.types.is_numeric_dtype(test[column])
        ]
        mean_shift = []
        for column in numeric[:256]:
            left = pd.to_numeric(train[column], errors="coerce")
            right = pd.to_numeric(test[column], errors="coerce")
            scale = float(left.std())
            if not math.isfinite(scale) or scale <= 1e-12:
                continue
            shift = abs(float(left.mean()) - float(right.mean())) / scale
            if math.isfinite(shift):
                mean_shift.append((str(column), shift))
        mean_shift.sort(key=lambda pair: pair[1], reverse=True)
        drift = {
            "available": True,
            "train_path": train_path,
            "test_path": test_path,
            "common_columns": len(common),
            "numeric_columns_checked": min(256, len(numeric)),
            "largest_standardized_mean_shifts": [
                {"column": column, "absolute_standardized_shift": round(value, 8)}
                for column, value in mean_shift[:30]
            ],
        }
    return tables, drift


def _media_diagnostics(root: Path, analysis: TaskAnalysis, allowed: set[str]) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    audio: list[dict[str, Any]] = []
    for item in analysis.files:
        relative = str(item.get("path") or "")
        if relative not in allowed:
            continue
        path = root / relative
        suffix = path.suffix.lower()
        if suffix in _IMAGE_SUFFIXES and len(images) < 40:
            try:
                from PIL import Image

                with Image.open(path) as image:
                    images.append(
                        {
                            "path": relative,
                            "width": int(image.width),
                            "height": int(image.height),
                            "mode": str(image.mode),
                            "format": str(image.format),
                        }
                    )
            except Exception as exc:
                images.append({"path": relative, "error": str(exc)[:300]})
        elif suffix in _AUDIO_SUFFIXES and len(audio) < 30:
            try:
                # pyrefly: ignore [missing-import]
                import soundfile

                info = soundfile.info(path)
                audio.append(
                    {
                        "path": relative,
                        "frames": int(info.frames),
                        "sample_rate": int(info.samplerate),
                        "channels": int(info.channels),
                        "duration_seconds": _safe_number(info.duration),
                    }
                )
            except Exception as exc:
                audio.append({"path": relative, "error": str(exc)[:300]})
    return {
        "image_sample": images,
        "image_shapes": dict(
            Counter(
                f"{item.get('width')}x{item.get('height')}:{item.get('mode')}"
                for item in images
                if item.get("width")
            )
        ),
        "audio_sample": audio,
    }


def _resource_inventory() -> dict[str, Any]:
    packages = (
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "catboost",
        "lightgbm",
        "xgboost",
        "torch",
        "torchvision",
        "transformers",
        "timm",
        "albumentations",
        "librosa",
        "soundfile",
        "optuna",
        "PIL",
        "imageio",
        "pyarrow",
    )
    memory_bytes = None
    try:
        memory_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    gpu: dict[str, Any] = {"torch_available": importlib.util.find_spec("torch") is not None}
    if gpu["torch_available"]:
        try:
            import torch

            gpu.update(
                cuda_available=bool(torch.cuda.is_available()),
                device_count=int(torch.cuda.device_count()),
                devices=[
                    {
                        "name": torch.cuda.get_device_name(index),
                        "memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                    }
                    for index in range(torch.cuda.device_count())
                ],
            )
        except Exception as exc:
            gpu["inspection_error"] = str(exc)[:300]
    return {
        "cpu_count": os.cpu_count(),
        "physical_memory_bytes": memory_bytes,
        "packages_available": {
            name: importlib.util.find_spec(name) is not None for name in packages
        },
        "gpu": gpu,
    }


def build_problem_fingerprint(
    analysis: TaskAnalysis, diagnostics: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a de-identified research description with no task/file/column names."""
    kinds = Counter(str(item.get("kind") or "unknown") for item in analysis.files)
    modality_inventory = predictive_modality_inventory(analysis.files)
    extensions = Counter(str(item.get("extension") or "none") for item in analysis.files)
    table_profiles = diagnostics.get("tables", [])
    table_shapes = [
        {
            "rows_observed": item.get("sampled_rows"),
            "columns": item.get("column_count"),
            "dtype_families": item.get("dtype_families", {}),
            "maximum_missing_fraction": max(
                (
                    float(value)
                    for value in item.get("top_missing_fraction", {}).values()
                    if value is not None
                ),
                default=0.0,
            ),
            "maximum_cardinality_observed": max(
                (int(value) for value in item.get("top_cardinalities", {}).values()),
                default=0,
            ),
            "target_topologies": [
                {
                    "observed": target.get("observed"),
                    "unique": target.get("unique"),
                    "missing_fraction": target.get("missing_fraction"),
                    "largest_class_fraction": max(
                        (
                            float(value)
                            for value in target.get("top_distribution", {}).values()
                            if value is not None
                        ),
                        default=None,
                    ),
                }
                for target in item.get("target_summary", {}).values()
            ],
        }
        for item in table_profiles
        if item.get("status") == "profiled"
    ]
    return {
        "objective": "supervised or task-native prediction from provided local data",
        "metric": analysis.metric,
        "direction": analysis.direction,
        "file_count": len(analysis.files),
        "data_kinds": dict(kinds),
        "predictive_modalities": modality_inventory["modalities"],
        "modality_file_counts": modality_inventory["file_counts"],
        "is_multimodal": modality_inventory["is_multimodal"],
        "extensions": dict(extensions),
        "table_profiles": table_shapes,
        "image_shape_distribution": diagnostics.get("media", {}).get("image_shapes", {}),
        "audio_files_profiled": len(diagnostics.get("media", {}).get("audio_sample", [])),
        "resource_inventory": diagnostics.get("resources", {}),
        "train_test_shift_summary": {
            key: value
            for key, value in diagnostics.get("train_test_shift", {}).items()
            if key not in {"train_path", "test_path", "largest_standardized_mean_shifts"}
        },
        "research_constraint": (
            "Use only general methods and primary literature. Do not search for this dataset, "
            "competition, notebooks, winning solutions, or copied implementation code."
        ),
    }


def collect_base_diagnostics(
    analysis: TaskAnalysis, *, max_table_rows: int = 50_000
) -> tuple[dict[str, Any], tuple[str, ...], tuple[dict[str, str], ...]]:
    root = Path(analysis.task_dir).resolve()
    allowed, prohibited = classify_input_access(analysis)
    tables, drift = _table_diagnostics(
        root, analysis, set(allowed), max_rows=max(1_000, int(max_table_rows))
    )
    diagnostics = {
        "schema_version": 1,
        "analysis_kind": "bounded adaptive preflight",
        "limits": {"maximum_rows_per_table": max_table_rows},
        "tables": tables,
        "train_test_shift": drift,
        "media": _media_diagnostics(root, analysis, set(allowed)),
        "resources": _resource_inventory(),
        "prohibited_inputs": list(prohibited),
    }
    diagnostics["diagnostics_hash"] = content_hash(diagnostics)
    return diagnostics, allowed, prohibited


_SAFE_STDLIB_ROOTS = {
    "os",
    "sys",
    "time",
    "datetime",
    "warnings",
    "functools",
    "operator",
    "typing",
    "contextlib",
    "collections",
    "itertools",
    "math",
    "statistics",
    "hashlib",
    "json",
    "csv",
    "re",
    "random",
    "pathlib",
}
_DISALLOWED_CALL_NAMES = {
    "compile",
    "eval",
    "exec",
    "__import__",
    "breakpoint",
    "help",
    "input",
}
_DISALLOWED_ATTRIBUTES = {
    "chmod",
    "connect",
    "create_connection",
    "fork",
    "kill",
    "popen",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "spawnl",
    "spawnlp",
    "spawnv",
    "spawnvp",
    "system",
    "to_csv",
    "to_feather",
    "to_json",
    "to_parquet",
    "to_pickle",
    "touch",
    "unlink",
    "write_bytes",
    "write_text",
}


def _python_source(response: str) -> str:
    fenced = re.findall(
        r"```(?:python|py)?\s*\n(.*?)```", response, flags=re.DOTALL | re.IGNORECASE
    )
    if fenced:
        return max(fenced, key=len).strip() + "\n"
    return re.sub(r"(?s)<thinking>.*?</thinking>", "", response).strip() + "\n"


def validate_diagnostic_script(source: str, available_packages: list[str] | None = None) -> list[str]:
    errors: list[str] = []
    
    allowed_roots = set(_SAFE_STDLIB_ROOTS)
    if available_packages:
        allowed_roots.update(available_packages)
    else:
        # Fallback if no packages are provided
        allowed_roots.update({
            "numpy", "pandas", "scipy", "sklearn", "PIL", "imageio", "soundfile", "pyarrow", "torch", "lightgbm"
        })

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"invalid Python syntax: {exc}"]
    assignments: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(value)

    def safe_output_target(node: ast.AST | None, seen: set[str] | None = None) -> bool:
        if isinstance(node, ast.Constant):
            return node.value == "analysis_result.json"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Path"
            and len(node.args) == 1
        ):
            return safe_output_target(node.args[0], seen)
        if isinstance(node, ast.Name):
            visited = set(seen or ())
            if node.id in visited:
                return False
            visited.add(node.id)
            values = assignments.get(node.id, [])
            return bool(values) and all(safe_output_target(value, visited) for value in values)
        return False

    class ResourceBoundVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.loop_depth = 0

        def visit_For(self, node: ast.For) -> None:
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        visit_AsyncFor = visit_For

        def visit_While(self, node: ast.While) -> None:
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_Call(self, node: ast.Call) -> None:
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if name in {"GridSearchCV", "RandomizedSearchCV"}:
                errors.append(
                    f"{name} is too open-ended for a bounded council diagnostic"
                )
            if name in {"cross_val_score", "cross_validate"} and self.loop_depth:
                errors.append(
                    f"{name} cannot run inside a loop; bound the diagnostic to at most "
                    "two model configurations on one shared split/fold set"
                )
            if name == "fit" and self.loop_depth >= 3:
                errors.append(
                    "model fitting inside three nested loops is too expensive for a council "
                    "diagnostic; flatten the comparison to at most 20 total fits"
                )
            for keyword in node.keywords:
                if keyword.arg == "n_jobs":
                    value: object = None
                    if isinstance(keyword.value, ast.Constant):
                        value = keyword.value.value
                    elif (
                        isinstance(keyword.value, ast.UnaryOp)
                        and isinstance(keyword.value.op, ast.USub)
                        and isinstance(keyword.value.operand, ast.Constant)
                    ):
                        try:
                            value = -keyword.value.operand.value
                        except TypeError:
                            value = "dynamic"
                    else:
                        value = "dynamic"
                    if value not in {None, 0, 1}:
                        errors.append("council diagnostics must use n_jobs=1")
            self.generic_visit(node)

    ResourceBoundVisitor().visit(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            blocked = roots - allowed_roots
            if blocked:
                errors.append("disallowed import(s): " + ", ".join(sorted(blocked)))
        elif isinstance(node, ast.ImportFrom):
            root = str(node.module or "").split(".", 1)[0]
            if not root or root not in allowed_roots:
                errors.append(f"disallowed import: {node.module!r}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _DISALLOWED_CALL_NAMES:
                errors.append(f"disallowed call: {node.func.id}")
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr.casefold() in _DISALLOWED_ATTRIBUTES
                and node.func.attr.casefold() not in {"write_text", "write_bytes"}
            ):
                errors.append(f"disallowed call attribute: {node.func.attr}")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr.casefold() in {"write_text", "write_bytes"}
            ):
                owner = node.func.value
                safe_output = safe_output_target(owner)
                if not safe_output:
                    errors.append(
                        f"{node.func.attr} is allowed only on Path('analysis_result.json')"
                    )
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "open"
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "open"
            ):
                mode: object = None
                mode_index = 1 if isinstance(node.func, ast.Name) else 0
                if len(node.args) > mode_index:
                    mode_node = node.args[mode_index]
                    if isinstance(mode_node, ast.Constant):
                        mode = mode_node.value
                for keyword in node.keywords:
                    if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                        mode = keyword.value.value
                if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
                    target: object = None
                    if isinstance(node.func, ast.Name) and node.args:
                        if safe_output_target(node.args[0]):
                            target = "analysis_result.json"
                    elif isinstance(node.func, ast.Attribute):
                        owner = node.func.value
                        if safe_output_target(owner):
                            target = "analysis_result.json"
                    if str(target or "") != "analysis_result.json":
                        errors.append(
                            "write-mode open is allowed only for analysis_result.json"
                        )
    return list(dict.fromkeys(errors))


class DiagnosticScriptRunner:
    """Ask a council member for one focused analysis and execute it with bounds."""

    def __init__(self, python: str, model_name: str | None = None) -> None:
        self.python = str(python)
        self.model_name = model_name

    def _generate(
        self,
        mandate: str,
        analysis: TaskAnalysis,
        diagnostics: Mapping[str, Any],
        *,
        previous_code: str = "",
        feedback: str = "",
    ) -> str:
        repair = ""
        if feedback:
            repair = f"""
The previous diagnostic failed. Repair the concrete failure without broadening
the mandate or weakening the validation method.

Previous code:
```python
{previous_code[-12000:]}
```

Failure evidence:
{feedback[-5000:]}
"""
        prompt = f"""
You are the local evidence investigator for an ML research council.

Member mandate:
{mandate[:3000]}

Task inventory (local only):
{analysis.prompt_context(10000)}

Existing bounded diagnostics:
{json.dumps(diagnostics, indent=2, default=str)[:12000]}

Write one focused, bounded, read-only Python diagnostic program that resolves the
highest-value factual uncertainty in the mandate. Inputs are under input/. The
program must not use the network, subprocesses, shell commands, dynamic imports,
or modify input files. It may inspect at most 100,000 table rows per file and at
most 100 media files. It must finish by writing `analysis_result.json` containing
a JSON object with keys: question, method, findings, limitations, suggested_next_questions.
Optionally include a top-level `measured_baseline` record: {{"model": str, "model_family": str, "score": float, "mode": str, "folds": int, "split_strategy": str, "sample_size": int, "limitations": str}}.
When the investigation compares predictive models, include simple regularized
linear models as credible candidates when applicable. Put estimator-specific
preprocessing (scaling, encoding, imputation, feature selection) inside each
validation fold with a Pipeline; do not handicap one family by applying the
preprocessing preferred by another. The `measured_baseline` must be the best
actually measured candidate according to the task metric direction, not the
most complex model or the model named first in the mandate. State its exact
preprocessing and key hyperparameters in `limitations` or the method text so a
downstream implementation can reproduce it.
The entire script may perform at most 20 model fits, with at most two credible
model configurations evaluated on one shared split or at most five folds. Do not
put `cross_val_score`, `cross_validate`, or model fitting inside a candidate,
feature-subset, seed, or hyperparameter loop. Use `n_jobs=1`; expensive tuning and
feature-subset sweeps belong in later search nodes, not council diagnostics.
Findings must be measurements, never model recommendations. Write no other file;
use either `open("analysis_result.json", "w")` or the direct literal
`Path("analysis_result.json").write_text(...)`. Return only Python code.

CRITICAL AST VALIDATION RULES:
- Do NOT use `to_csv`, `to_pickle`, `to_parquet`, etc.
- Do NOT write submission files or save models to disk.
- If you violate these rules, the sandbox will REJECT your script.
{repair}
""".strip()
        response = call_llm(
            "You write safe, resource-bounded data investigation scripts for senior ML researchers.",
            prompt,
            model=self.model_name,
            temperature=0.0,
        )
        return _python_source(response)

    def run(
        self,
        member_id: str,
        mandate: str,
        analysis: TaskAnalysis,
        diagnostics: Mapping[str, Any],
        council_dir: Path,
        allowed_paths: tuple[str, ...],
    ) -> dict[str, Any]:
        work_dir = Path(council_dir) / "work" / member_id
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        available_packages = diagnostics.get("resources", {}).get("packages_available", [])
        source_path = work_dir / "diagnostic.py"
        try:
            copy_limit = max(
                0, int(os.getenv("AIBUILDAI_COUNCIL_COPY_LIMIT_BYTES", "33554432"))
            )
        except ValueError:
            copy_limit = 33_554_432
        approved_size = sum(
            (Path(analysis.task_dir) / relative).stat().st_size
            for relative in allowed_paths
            if (Path(analysis.task_dir) / relative).is_file()
        )
        isolated_copies = approved_size <= copy_limit
        expose_task_data(
            analysis.task_dir,
            work_dir,
            allowed_paths=allowed_paths,
            copy_files=isolated_copies,
        )
        env = sanitized_subprocess_env()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "ALL_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
            }
        )
        result_path = work_dir / "analysis_result.json"
        try:
            max_attempts = max(
                1, min(5, int(os.getenv("AIBUILDAI_COUNCIL_DIAGNOSTIC_ATTEMPTS", "3")))
            )
        except ValueError:
            max_attempts = 3
        try:
            attempt_hard_limit = max(
                30.0,
                float(os.getenv("AIBUILDAI_COUNCIL_DIAGNOSTIC_HARD_LIMIT_SECONDS", "120")),
            )
        except ValueError:
            attempt_hard_limit = 120.0
        try:
            total_execution_limit = max(
                attempt_hard_limit,
                float(os.getenv("AIBUILDAI_COUNCIL_DIAGNOSTIC_TOTAL_SECONDS", "240")),
            )
        except ValueError:
            total_execution_limit = 240.0
        previous_code = ""
        feedback = ""
        execution_seconds = 0.0
        last_failure: dict[str, Any] = {
            "member_id": member_id,
            "status": "generation_failed",
            "error": "no diagnostic attempt completed",
        }
        logs: list[str] = []
        for attempt in range(1, max_attempts + 1):
            if execution_seconds >= total_execution_limit:
                last_failure["budget_exhausted"] = True
                break
            try:
                code = self._generate(
                    mandate,
                    analysis,
                    diagnostics,
                    previous_code=previous_code,
                    feedback=feedback,
                )
            except Exception as exc:
                feedback = f"LLM generation failed: {type(exc).__name__}: {exc}"
                logs.append(f"attempt {attempt}: {feedback}")
                last_failure = {
                    "member_id": member_id,
                    "status": "generation_failed",
                    "error": feedback[:1000],
                }
                continue
            previous_code = code
            source_path.write_text(code, encoding="utf-8")
            errors = validate_diagnostic_script(code, available_packages)
            if errors:
                feedback = "AST validation failed:\n- " + "\n- ".join(errors)
                logs.append(f"attempt {attempt}: {feedback}")
                last_failure = {
                    "member_id": member_id,
                    "status": "rejected",
                    "errors": errors,
                    "script": str(source_path),
                }
                continue
            try:
                result_path.unlink()
            except FileNotFoundError:
                pass
            completed = run_supervised_process(
                [self.python, source_path.name],
                cwd=work_dir,
                env=env,
                stall_seconds=90,
                hard_limit_seconds=min(
                    attempt_hard_limit,
                    max(1.0, total_execution_limit - execution_seconds),
                ),
                activity_root=work_dir,
                label=f"Council diagnostic {member_id} attempt {attempt}",
            )
            execution_seconds += completed.elapsed_seconds
            attempt_log = (
                f"attempt {attempt}; returncode: {completed.returncode}; "
                f"termination_reason: {completed.termination_reason or 'process_exit'}; "
                f"elapsed_seconds: {completed.elapsed_seconds:.2f}\n\n"
                f"STDOUT:\n{completed.stdout[-12000:]}\n\n"
                f"STDERR:\n{completed.stderr[-12000:]}\n"
            )
            logs.append(attempt_log)
            if completed.returncode != 0 or not result_path.is_file():
                feedback = (
                    f"Execution failed with return code {completed.returncode}; "
                    f"termination reason={completed.termination_reason or 'process_exit'}.\n"
                    f"STDOUT:\n{completed.stdout[-2000:]}\n"
                    f"STDERR:\n{completed.stderr[-3000:]}"
                )
                if completed.hard_limit_reached:
                    feedback += (
                        "\nThe diagnostic exceeded its hard runtime budget. Remove model/feature/"
                        "seed sweeps, use n_jobs=1, and reduce the repair to at most two model "
                        "configurations and 20 total fits. Do not merely increase a timeout."
                    )
                last_failure = {
                    "member_id": member_id,
                    "status": "execution_failed",
                    "returncode": completed.returncode,
                    "stderr": completed.stderr[-2000:],
                    "script": str(source_path),
                    "attempts": attempt,
                    "termination_reason": completed.termination_reason,
                    "execution_seconds": execution_seconds,
                }
                continue
            if result_path.stat().st_size > 1_000_000:
                feedback = "analysis_result.json exceeded 1 MB"
                last_failure = {
                    "member_id": member_id,
                    "status": "rejected",
                    "errors": [feedback],
                    "script": str(source_path),
                    "attempts": attempt,
                }
                continue
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("analysis result must be a JSON object")
                mb = payload.get("measured_baseline")
                if mb is not None:
                    if not isinstance(mb, dict):
                        logging.warning("measured_baseline is not a dict; dropping it.")
                        payload.pop("measured_baseline")
                    else:
                        score = mb.get("score")
                        if not isinstance(score, (int, float)) or not math.isfinite(score):
                            logging.warning("measured_baseline has invalid score; dropping it.")
                            payload.pop("measured_baseline")
                        else:
                            mb["metric"] = analysis.metric
                            mb["direction"] = analysis.direction
                            mb["source"] = member_id
            except Exception as exc:
                feedback = f"Invalid analysis_result.json: {type(exc).__name__}: {exc}"
                last_failure = {
                    "member_id": member_id,
                    "status": "invalid_result",
                    "error": feedback,
                    "script": str(source_path),
                    "attempts": attempt,
                }
                continue
            (work_dir / "diagnostic.log").write_text(
                "\n\n".join(logs), encoding="utf-8"
            )
            return {
                "member_id": member_id,
                "status": "completed",
                "script": str(source_path),
                "result": payload,
                "elapsed_seconds": completed.elapsed_seconds,
                "attempts": attempt,
                "input_exposure": (
                    "read_only_copies" if isolated_copies else "allowlisted_links"
                ),
                "approved_input_bytes": approved_size,
            }
        (work_dir / "diagnostic.log").write_text(
            "\n\n".join(logs), encoding="utf-8"
        )
        return last_failure
