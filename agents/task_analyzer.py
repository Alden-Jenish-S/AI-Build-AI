"""Small, read-only task-directory analyzer.

The analyzer deliberately describes observed data and the requested output.  It
does not prescribe a model family, assess data quality, estimate complexity, or
build runtime/evaluation contracts.
"""

from __future__ import annotations

import csv
import json
import mimetypes
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_TABLE_EXTENSIONS = {".csv", ".tsv", ".parquet", ".feather"}
_TEXT_EXTENSIONS = {
    ".md", ".txt", ".rst", ".yaml", ".yml", ".toml", ".ini", ".cfg"
}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg"}
_ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z"}
_DESCRIPTION_NAMES = {
    "readme", "task_description", "description", "overview", "instructions", "task"
}


def _truncate(value: object, limit: int = 100) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _goal_from_description(description: str) -> str:
    """Prefer explicit goal/task sentences over introductory background."""
    normalized = description.replace("\r", "\n")
    units = [
        " ".join(unit.split())
        for unit in re.split(r"(?:\n+|(?<=[.!?])\s+)", normalized)
        if unit.strip()
    ]
    patterns = (
        r"\b(?:your|the)\s+(?:task|goal)\b",
        r"\btasked\s+to\b",
        r"\byou\s+will\s+be\s+predicting\b",
        r"\byou\s+must\s+(?:predict|submit|produce|generate)\b",
        r"\bgoal\s+of\s+(?:the|this)\b",
    )
    selected = []
    seen = set()
    for unit in units:
        if any(re.search(pattern, unit, re.IGNORECASE) for pattern in patterns):
            if unit not in seen:
                seen.add(unit)
                selected.append(unit)
    if selected:
        return _truncate(" ".join(selected[:2]), 400)
    return _truncate(" ".join(description.split()[:40]), 400)


def _kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _TABLE_EXTENSIONS:
        return "table"
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return "structured_text"
    if suffix in _TEXT_EXTENSIONS:
        return "text"
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _AUDIO_EXTENSIONS:
        return "audio"
    if suffix in _VIDEO_EXTENSIONS:
        return "video"
    if suffix in _ARCHIVE_EXTENSIONS:
        return "archive"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed.split("/", 1)[0] if guessed else "binary_or_unknown"


def _line_count(path: Path) -> int | None:
    """Count rows without loading a delimited file into memory."""
    try:
        count = 0
        final_byte = b""
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                count += block.count(b"\n")
                final_byte = block[-1:]
        if final_byte and final_byte != b"\n":
            count += 1
        return max(0, count - 1)
    except OSError:
        return None


def _delimited_profile(path: Path) -> dict[str, Any]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    profile: dict[str, Any] = {"rows": _line_count(path)}
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            reader = csv.reader(stream, delimiter=delimiter)
            columns = next(reader, [])
            samples = []
            for _, row in zip(range(3), reader):
                samples.append(
                    {
                        str(column): _truncate(row[index] if index < len(row) else "")
                        for index, column in enumerate(columns[:30])
                    }
                )
        inferred_types: dict[str, str] = {}
        for index, column in enumerate(columns):
            values = [row.get(str(column), "") for row in samples]
            nonempty = [str(value) for value in values if str(value).strip()]
            observed = "empty"
            if nonempty:
                try:
                    for value in nonempty:
                        int(value)
                    observed = "integer"
                except ValueError:
                    try:
                        for value in nonempty:
                            float(value)
                        observed = "number"
                    except ValueError:
                        lowered = {value.lower() for value in nonempty}
                        observed = "boolean" if lowered <= {"true", "false", "yes", "no"} else "text"
            inferred_types[str(column)] = observed
        profile.update(
            columns=[str(item) for item in columns],
            dtypes=inferred_types,
            sample=samples,
        )
    except (OSError, csv.Error) as exc:
        profile["note"] = f"Could not preview delimited text: {_truncate(exc)}"
    return profile


def _columnar_profile(path: Path) -> dict[str, Any]:
    try:
        import pyarrow as pa

        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as parquet

            reader = parquet.ParquetFile(path)
            schema = reader.schema_arrow
            batch = next(reader.iter_batches(batch_size=3), None)
            rows = int(reader.metadata.num_rows)
        else:
            import pyarrow.ipc as ipc

            source = pa.memory_map(str(path), "r")
            reader = ipc.open_file(source)
            schema = reader.schema
            rows = 0
            batch = None
            for index in range(reader.num_record_batches):
                current_batch = reader.get_batch(index)
                rows += len(current_batch)
                if batch is None:
                    batch = current_batch.slice(0, 3)
        sample = batch.to_pylist() if batch is not None else []
        return {
            "rows": rows,
            "columns": [str(field.name) for field in schema],
            "dtypes": {str(field.name): str(field.type) for field in schema},
            "sample": [
                {str(key): _truncate(value) for key, value in record.items()}
                for record in sample[:3]
            ],
        }
    except Exception as exc:
        return {"note": f"Could not preview columnar table: {_truncate(exc)}"}


def _structured_profile(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            records: list[object] = []
            rows = 0
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if line.strip():
                        rows += 1
                        if len(records) < 3:
                            records.append(json.loads(line))
            keys = sorted({str(key) for record in records if isinstance(record, dict) for key in record})
            return {"rows": rows, "keys": keys, "sample": records}
        if path.stat().st_size > 2_000_000:
            return {"note": "JSON is larger than the bounded analysis preview."}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {"shape": "object", "keys": [str(key) for key in list(payload)[:100]]}
        if isinstance(payload, list):
            sample = payload[:3]
            keys = sorted({str(key) for record in sample if isinstance(record, dict) for key in record})
            return {"shape": "array", "rows": len(payload), "keys": keys, "sample": sample}
        return {"shape": type(payload).__name__, "sample": _truncate(payload)}
    except Exception as exc:
        return {"note": f"Could not preview JSON: {_truncate(exc)}"}


@dataclass
class TaskAnalysis:
    task_name: str
    task_dir: Path
    goal: str
    target: str
    expected_output: str
    metric: str = "task score"
    direction: str = "maximize"
    files: list[dict[str, Any]] = field(default_factory=list)
    folders: list[dict[str, Any]] = field(default_factory=list)
    description: str = ""
    submission: dict[str, Any] | None = None
    task_facts: dict[str, Any] = field(default_factory=dict)

    @property
    def report(self) -> str:
        lines = [
            f"# Task inventory: {self.task_name}",
            "",
            f"Goal: {self.goal}",
            f"Target: {self.target}",
            f"Expected output: {self.expected_output}",
            f"Score: {self.metric} ({self.direction})",
            "",
            "## Files & Data Modalities",
            "",
        ]
        displayed_files = self.files
        if len(self.files) > 250:
            displayed_files = []
            seen_groups: Counter[tuple[str, str]] = Counter()
            for item in self.files:
                parent = str(Path(str(item["path"])).parent)
                group = (parent, str(item["kind"]))
                if parent == "." or seen_groups[group] < 3:
                    displayed_files.append(item)
                    seen_groups[group] += 1
                if len(displayed_files) >= 250:
                    break
            lines.append(
                f"Showing {len(displayed_files)} representative paths; folder counts below cover all {len(self.files)} files."
            )
            lines.append("")
        for item in displayed_files:
            detail = item.get("profile", {})
            columns = detail.get("columns") or detail.get("keys") or []
            if columns:
                if len(columns) > 8:
                    suffix = f"; {len(columns)} columns ({', '.join(map(str, columns[:4]))} ... {', '.join(map(str, columns[-2:]))})"
                else:
                    suffix = f"; columns: {', '.join(map(str, columns))}"
            else:
                suffix = ""
            rows = detail.get("rows")
            row_text = f"; rows: {rows}" if rows is not None else ""
            lines.append(
                f"- `{item['path']}` — {item['kind']}, {item['bytes']} bytes{row_text}{suffix}"
            )
            dtypes = detail.get("dtypes", {})
            if isinstance(dtypes, dict) and dtypes:
                by_type: dict[str, list[str]] = defaultdict(list)
                for col, dt in dtypes.items():
                    by_type[str(dt)].append(str(col))
                type_parts = []
                for dt, cols in by_type.items():
                    if len(cols) <= 4:
                        type_parts.append(f"{dt}: {', '.join(cols)}")
                    else:
                        type_parts.append(f"{dt} ({len(cols)} cols e.g. {', '.join(cols[:3])}...)")
                lines.append(f"  - observed types: {'; '.join(type_parts)}")
            samples = detail.get("sample", [])
            if isinstance(samples, list) and samples and isinstance(samples[0], dict):
                sample_dict = samples[0]
                if len(sample_dict) > 6:
                    keys = list(sample_dict.keys())[:4]
                    preview = {k: sample_dict[k] for k in keys}
                    try:
                        ex_str = json.dumps(preview, ensure_ascii=False, default=str)[:-1] + ", ...}"
                    except Exception:
                        ex_str = str(preview)
                else:
                    try:
                        ex_str = json.dumps(sample_dict, ensure_ascii=False, default=str)
                    except Exception:
                        ex_str = str(sample_dict)
                lines.append(f"  - sample preview: {_truncate(ex_str, 200)}")
        if self.folders:
            lines.extend(("", "## Folders", ""))
            displayed_folders = self.folders[:300]
            for folder in displayed_folders:
                kinds = ", ".join(
                    f"{key}: {value}" for key, value in folder.get("kinds", {}).items()
                )
                lines.append(
                    f"- `{folder['path']}` — {folder['file_count']} files"
                    + (f" ({kinds})" if kinds else "")
                )
            if len(self.folders) > len(displayed_folders):
                lines.append(
                    f"- … {len(self.folders) - len(displayed_folders)} additional folders"
                )
        if self.description:
            lines.extend(("", "## Provided instructions", "", self.description.strip()))
        if self.task_facts:
            lines.extend(("", "## Explicit task facts", ""))
            for key, value in self.task_facts.items():
                lines.append(f"- {key}: {_truncate(value, 800)}")
        return "\n".join(lines).strip() + "\n"

    def prompt_context(self, max_chars: int = 24000) -> str:
        """Compact, human-readable context passed to code-writing calls."""
        text = self.report
        return text if len(text) <= max_chars else text[:max_chars] + "\n[Inventory truncated]\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "goal": self.goal,
            "target": self.target,
            "expected_output": self.expected_output,
            "metric": self.metric,
            "direction": self.direction,
            "files": self.files,
            "folders": self.folders,
            "description": self.description,
            "submission": self.submission,
            "task_facts": self.task_facts,
        }


class TaskAnalyzer:
    """Inventory arbitrary task files without enforcing a modality schema."""

    def __init__(self, *args: object, model_name: str | None = None, **kwargs: object) -> None:
        self.model_name = model_name

    def analyze(self, task_dir: Path) -> TaskAnalysis:
        root = Path(task_dir).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {root}")

        paths: list[Path] = []
        ignored_dirs = {".git", ".venv", "__pycache__", "runs", "node_modules", ".tempmediaStorage", "submission"}
        for current, directory_names, file_names in os.walk(
            root, followlinks=False, onerror=lambda _error: None
        ):
            directory_names[:] = [
                d for d in directory_names if d not in ignored_dirs and not d.endswith("-previous") and not re.match(r"^node_?\d+$", d)
            ]
            directory_names.sort()
            for name in sorted(file_names):
                if name in {".DS_Store", "tree_state.json", "task_analysis.md", "method_tree.png"}:
                    continue
                candidate = Path(current) / name
                try:
                    if candidate.is_file():
                        paths.append(candidate)
                except OSError:
                    continue
        files: list[dict[str, Any]] = []
        description_parts: list[str] = []
        task_facts: dict[str, Any] = {}
        by_folder: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for path in paths:
            try:
                file_size = path.stat().st_size
            except OSError:
                continue
            relative = path.relative_to(root).as_posix()
            kind = _kind(path)
            profile: dict[str, Any] = {}
            if path.suffix.lower() in {".csv", ".tsv"}:
                profile = _delimited_profile(path)
            elif path.suffix.lower() in {".parquet", ".feather"}:
                profile = _columnar_profile(path)
            elif path.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
                profile = _structured_profile(path)
                if path.name.lower() in {"task_config.json", "config.json"} and file_size <= 2_000_000:
                    try:
                        configured = json.loads(path.read_text(encoding="utf-8"))
                        if isinstance(configured, dict):
                            for key in (
                                "goal", "objective", "description", "instructions",
                                "target", "target_column", "label_column", "output",
                                "submission", "output_contract", "metric", "primary_metric", "direction",
                            ):
                                if configured.get(key) is not None:
                                    task_facts[key] = configured[key]
                    except (OSError, json.JSONDecodeError):
                        pass
            stem = path.stem.lower().replace("-", "_")
            if (
                kind == "text"
                and (stem in _DESCRIPTION_NAMES or "description" in stem or "instruction" in stem)
                and sum(map(len, description_parts)) < 12000
            ):
                try:
                    text_content = path.read_text(encoding="utf-8", errors="replace")[:6000].strip()
                    if text_content and text_content not in description_parts:
                        description_parts.append(text_content)
                except OSError:
                    pass
            item = {
                "path": relative,
                "kind": kind,
                "extension": path.suffix.lower() or "none",
                "bytes": file_size,
                "profile": profile,
            }
            files.append(item)
            parent = path.parent.relative_to(root).as_posix()
            if parent != ".":
                parts = parent.split("/")
                for index in range(1, len(parts) + 1):
                    by_folder["/".join(parts[:index])].append(item)

        folders = []
        for relative, children in sorted(by_folder.items()):
            folders.append(
                {
                    "path": relative,
                    "file_count": len(children),
                    "kinds": dict(sorted(Counter(item["kind"] for item in children).items())),
                }
            )

        description = "\n\n".join(description_parts).strip()
        tables = [item for item in files if item["kind"] == "table"]
        def output_reference_priority(item: dict[str, Any]) -> tuple[int, str]:
            stem = Path(str(item["path"])).stem.lower().replace("-", "_")
            if stem == "sample_submission":
                return (0, str(item["path"]))
            if "sample" in stem and ("submission" in stem or "output" in stem):
                return (1, str(item["path"]))
            if "template" in stem and ("submission" in stem or "output" in stem):
                return (2, str(item["path"]))
            if "submission" in stem:
                return (3, str(item["path"]))
            return (99, str(item["path"]))

        reference_candidates = [
            item for item in files if output_reference_priority(item)[0] < 99
        ]
        submission = min(
            reference_candidates,
            key=output_reference_priority,
            default=None,
        )
        train = next((item for item in tables if "train" in Path(item["path"]).stem.lower()), None)
        test = next((item for item in tables if "test" in Path(item["path"]).stem.lower()), None)
        data = next((item for item in tables if Path(item["path"]).stem.lower() == "data"), None)

        submission_columns = list((submission or {}).get("profile", {}).get("columns", []))
        train_columns = list((train or {}).get("profile", {}).get("columns", []))
        test_columns = set((test or {}).get("profile", {}).get("columns", []))
        target_candidates = (
            [column for column in train_columns if column not in test_columns]
            if train is not None and test is not None
            else []
        )

        configured_target = task_facts.get("target") or task_facts.get("target_column") or task_facts.get("label_column")
        if configured_target is not None:
            target = _truncate(configured_target, 1000)
        elif target_candidates:
            if len(target_candidates) > 8:
                target = f"{len(target_candidates)} target variables ({', '.join(map(str, target_candidates[:3]))} ... {', '.join(map(str, target_candidates[-2:]))})"
            else:
                target = ", ".join(map(str, target_candidates))
        elif submission_columns:
            output_columns = submission_columns[1:] if len(submission_columns) > 1 else submission_columns
            if len(output_columns) > 8:
                target = f"{len(output_columns)} target variables ({', '.join(map(str, output_columns[:3]))} ... {', '.join(map(str, output_columns[-2:]))})"
            else:
                target = ", ".join(map(str, output_columns)) or "the value requested by the submission template"
        elif data is not None:
            target = "derive the requested output for the rows/items in the provided data"
        else:
            target = "produce the result described by the task instructions"

        if submission is not None and submission.get("kind") == "table":
            rows = submission.get("profile", {}).get("rows")
            if len(submission_columns) > 10:
                col_str = f"{len(submission_columns)} columns ({', '.join(map(str, submission_columns[:3]))} ... {', '.join(map(str, submission_columns[-2:]))})"
            else:
                col_str = f"columns {submission_columns}"
            expected_output = (
                f"Write `submission/submission.csv` with {col_str}"
                + (f" and {rows} data rows" if rows is not None else "")
                + ", following the provided sample submission."
            )
        elif submission is not None:
            expected_output = (
                "Write the requested deliverable under `submission/`, matching the observed "
                f"sample output `{submission['path']}` ({submission['kind']}, "
                f"{submission['extension']}), and record its path in `result.json`."
            )
        else:
            expected_output = (
                "Write the requested deliverable under `submission/` and record its path "
                "in `result.json`."
            )
            configured_output = task_facts.get("output") or task_facts.get("submission")
            if configured_output is not None:
                expected_output += f" Explicit output instruction: {_truncate(configured_output, 1000)}"

        configured_goal = (
            task_facts.get("goal") or task_facts.get("objective")
            or task_facts.get("instructions") or task_facts.get("description")
        )
        goal = (
            _truncate(configured_goal, 1400)
            if configured_goal is not None
            else _goal_from_description(description)
            if description
            else f"Use the files in this task to predict or produce {target}."
        )

        lowered = description.lower()
        metric = "task score"
        metric_candidates = (
            (r"\badjusted\s+rand(?:\s+index)?\b", "adjusted Rand index"),
            (r"\b(?:roc[\s_-]*auc|area\s+under\s+the\s+roc)\b", "ROC AUC"),
            (r"\bmean\s+average\s+precision\b", "mean average precision"),
            (r"\b(?:log\s*loss|logarithmic\s+loss|cross[\s_-]*entropy)\b", "log loss"),
            (r"\b(?:rmse|root\s+mean\s+squared\s+error)\b", "RMSE"),
            (r"\b(?:mae|mean\s+absolute\s+error)\b", "MAE"),
            (r"\baccuracy\b", "accuracy"),
            (r"\bf1(?:[\s_-]*score)?\b", "F1"),
            (r"\bdice(?:\s+(?:coefficient|score))?\b", "Dice"),
            (r"\b(?:iou|intersection\s+over\s+union)\b", "IoU"),
        )
        for pattern, name in metric_candidates:
            if re.search(pattern, lowered):
                metric = name
                break
        configured_metric = task_facts.get("primary_metric") or task_facts.get("metric")
        if configured_metric is not None:
            metric = _truncate(configured_metric, 200)
        configured_direction = str(task_facts.get("direction", "")).strip().lower()
        direction = (
            configured_direction if configured_direction in {"maximize", "minimize"}
            else "minimize" if any(word in metric.lower() for word in ("loss", "rmse", "mae", "error"))
            else "maximize"
        )

        return TaskAnalysis(
            task_name=root.name,
            task_dir=root,
            goal=goal,
            target=target,
            expected_output=expected_output,
            metric=metric,
            direction=direction,
            files=files,
            folders=folders,
            description=description,
            submission=submission,
            task_facts=task_facts,
        )

    def resolve(self, task_dir: Path) -> TaskAnalysis:
        """Compatibility alias for callers that previously resolved a spec."""
        return self.analyze(task_dir)
