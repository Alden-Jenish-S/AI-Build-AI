"""Dataset discovery and profiling for generated task loaders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_utils import infer_task_type, task_data_files


def _read_config(task_dir: Path) -> dict:
    config_file = Path(task_dir) / "task_config.json"
    if not config_file.is_file():
        return {}
    with open(config_file, "r", encoding="utf-8") as file:
        loaded = json.load(file)
    return loaded if isinstance(loaded, dict) else {}


def _description(task_dir: Path) -> str:
    path = Path(task_dir) / "task_description.md"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _relative_name(task_dir: Path, path: Path) -> str:
    # Data discovery currently scans the flat task root or its flat input/
    # directory. Expose the basename so generated loaders can always resolve it
    # directly beneath their run-local ./input link.
    return path.name


def discover_dataset_layout(task_dir: Path) -> dict:
    """Inspect available files and infer roles without requiring fixed basenames.

    Explicit ``task_config.json`` entries win. Supported keys are
    ``train_file``, ``test_file``, ``data_file``, ``sample_submission_file``,
    ``target_column``, and ``task_type``.
    """
    task_dir = Path(task_dir)
    config = _read_config(task_dir)
    description = _description(task_dir)
    files = task_data_files(task_dir)
    tabular = [
        path for path in files if path.suffix.lower() in {".csv", ".tsv"}
    ]
    explicit_names = {
        "train": config.get("train_file"),
        "test": config.get("test_file"),
        "data": config.get("data_file"),
        "sample_submission": config.get("sample_submission_file"),
    }
    by_name = {path.name: path for path in tabular}
    by_relative = {_relative_name(task_dir, path): path for path in tabular}

    roles: dict[str, Path] = {}
    for role, configured_name in explicit_names.items():
        if not configured_name:
            continue
        configured_path = by_relative.get(str(configured_name)) or by_name.get(
            Path(str(configured_name)).name
        )
        if configured_path is None:
            raise FileNotFoundError(
                f"Configured {role}_file {configured_name!r} is not among the "
                f"discovered data files: {sorted(by_relative)}"
            )
        roles[role] = configured_path

    # Conventional names are only one signal, not a requirement.
    lowered = {path.name.lower(): path for path in tabular}
    roles.setdefault("train", lowered.get("train.csv"))
    roles.setdefault("test", lowered.get("test.csv"))
    roles.setdefault(
        "sample_submission", lowered.get("sample_submission.csv")
    )
    roles = {role: path for role, path in roles.items() if path is not None}

    sample_path = roles.get("sample_submission")
    if sample_path is None:
        submission_candidates = [
            path
            for path in tabular
            if "submission" in path.stem.lower()
            or "prediction_template" in path.stem.lower()
        ]
        if len(submission_candidates) == 1:
            sample_path = submission_candidates[0]
            roles["sample_submission"] = sample_path

    task_type = infer_task_type(description, config.get("task_type"))
    unused = [path for path in tabular if path not in set(roles.values())]
    if task_type == "unsupervised_clustering":
        if "data" not in roles:
            feature_candidates = [
                path for path in unused if path != sample_path
            ]
            if len(feature_candidates) == 1:
                roles["data"] = feature_candidates[0]
            elif "data.csv" in lowered:
                roles["data"] = lowered["data.csv"]
    elif "train" not in roles:
        # Schema comparison handles nonstandard supervised names. The training
        # table is the table with one or more columns absent from a second table.
        headers = {}
        for path in unused:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            try:
                headers[path] = set(
                    pd.read_csv(path, sep=sep, nrows=0).columns
                )
            except Exception:
                continue
        pairs = [
            (left, right, headers[left] - headers[right])
            for left in headers
            for right in headers
            if left != right and headers[left] - headers[right]
        ]
        if pairs:
            left, right, _ = min(pairs, key=lambda item: len(item[2]))
            roles["train"], roles["test"] = left, right
        elif len(unused) == 1:
            roles["train"] = unused[0]

    inventory = []
    for path in files:
        item = {
            "path": _relative_name(task_dir, path),
            "name": path.name,
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "role": next(
                (role for role, role_path in roles.items() if role_path == path),
                "auxiliary",
            ),
        }
        if path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            try:
                sample = pd.read_csv(path, sep=sep, nrows=5)
                item["columns"] = sample.columns.tolist()
                item["sample_rows"] = len(sample)
            except Exception as exc:
                item["read_error"] = str(exc)
        inventory.append(item)

    return {
        "task_type": task_type,
        "target_column": config.get("target_column"),
        "roles": {
            role: _relative_name(task_dir, path)
            for role, path in roles.items()
        },
        "inventory": inventory,
    }


def _role_path(task_dir: Path, relative: str | None) -> Path | None:
    if not relative:
        return None
    path = Path(task_dir) / relative
    if path.is_file():
        return path
    # When discovery uses the task's input directory, a caller may pass that
    # directory itself. Fall back to its basename without assuming a role name.
    candidate = Path(task_dir) / "input" / Path(relative).name
    return candidate if candidate.is_file() else None


def run_dataset_analysis(task_dir: Path) -> str:
    """Profile discovered task files and emit a loader-oriented schema report."""
    task_dir = Path(task_dir)
    analysis_report = ["=== AUTOMATIC DATASET ANALYSIS REPORT ==="]
    try:
        layout = discover_dataset_layout(task_dir)
        analysis_report.append(
            "Discovered Data Layout (machine-readable):\n"
            + json.dumps(layout, indent=2)
        )
        analysis_report.append(
            f"Inferred task type: {layout['task_type']}"
        )
        analysis_report.append(
            "Resolved file roles: "
            + json.dumps(layout["roles"], sort_keys=True)
        )

        role = (
            "data"
            if layout["task_type"] == "unsupervised_clustering"
            else "train"
        )
        primary = _role_path(task_dir, layout["roles"].get(role))
        if primary is None:
            discovered = [item["path"] for item in layout["inventory"]]
            return (
                "\n".join(analysis_report)
                + "\nDataset Analysis: could not resolve a primary feature table "
                + f"from discovered files {discovered}. Set data_file/train_file "
                + "in task_config.json when the layout is ambiguous."
            )

        print(f"DataAnalyzer: Inspecting {primary.name}...")
        sep = "\t" if primary.suffix.lower() == ".tsv" else ","
        file_size_bytes = primary.stat().st_size
        large_file = file_size_bytes > 300 * 1024 * 1024
        df = pd.read_csv(
            primary, sep=sep, nrows=100_000 if large_file else None
        )
        if df.empty:
            raise ValueError(f"{primary.name} contains no data rows")
        analysis_report.append(
            (
                f"Profiled the first {len(df)} rows of {primary.name} "
                f"({file_size_bytes / (1024**2):.1f} MB)."
                if large_file
                else f"Loaded {primary.name} completely: {len(df)} rows."
            )
        )

        target_col = layout.get("target_column")
        test_path = _role_path(task_dir, layout["roles"].get("test"))
        test_sample = None
        if test_path is not None:
            test_sep = "\t" if test_path.suffix.lower() == ".tsv" else ","
            test_sample = pd.read_csv(test_path, sep=test_sep, nrows=5)
        if layout["task_type"] != "unsupervised_clustering":
            if target_col not in df.columns:
                train_only = (
                    [
                        column
                        for column in df.columns
                        if column not in test_sample.columns
                    ]
                    if test_sample is not None
                    else []
                )
                common = [
                    column
                    for column in df.columns
                    if column.lower()
                    in {"target", "label", "class", "cover_type"}
                ]
                if len(train_only) == 1:
                    target_col = train_only[0]
                elif common:
                    target_col = common[0]
            if target_col in df.columns:
                analysis_report.append(
                    f"Inferred Target Column: '{target_col}'"
                )
            else:
                analysis_report.append(
                    "WARNING: No target column could be inferred. Set "
                    "target_column in task_config.json if this is supervised."
                )
                target_col = None
        else:
            target_col = None
            analysis_report.append(
                "Target Column: none (unlabeled transductive clustering). "
                "The primary data table is both the fitting population and the "
                "submission population."
            )

        sample_path = _role_path(
            task_dir, layout["roles"].get("sample_submission")
        )
        if sample_path is not None:
            sample_sep = (
                "\t" if sample_path.suffix.lower() == ".tsv" else ","
            )
            sample_columns = pd.read_csv(
                sample_path, sep=sample_sep, nrows=0
            ).columns.tolist()
            analysis_report.append(
                f"Sample Submission Columns: {sample_columns}"
            )

        num_rows = len(df)
        analysis_report.append("\n1. Features and Column Types:")
        for column in df.columns:
            null_count = int(df[column].isnull().sum())
            analysis_report.append(
                f"  - '{column}': type={df[column].dtype}, "
                f"unique_values={df[column].nunique()}, "
                f"nulls={null_count} ({null_count / num_rows * 100:.2f}%)"
            )

        drop_suggestions = []
        for column in df.columns:
            if column == target_col:
                continue
            if df[column].nunique() == 1:
                drop_suggestions.append(
                    f"  - '{column}' (constant column with single value)"
                )
            elif column.lower() in {"id", "uuid", "index"} or (
                df[column].nunique() == num_rows
                and pd.api.types.is_integer_dtype(df[column].dtype)
            ):
                drop_suggestions.append(
                    f"  - '{column}' (looks like an ID or sequential index column)"
                )
            elif df[column].isnull().mean() > 0.90:
                drop_suggestions.append(
                    f"  - '{column}' (extremely high missing rate: "
                    f"{df[column].isnull().mean() * 100:.1f}%)"
                )
        analysis_report.append("\n2. Suggested Features to Drop:")
        analysis_report.extend(drop_suggestions or ["  - None detected"])

        missing = [column for column in df if df[column].isnull().any()]
        analysis_report.append("\n3. Missing Value Analysis:")
        analysis_report.extend(
            [
                f"  - '{column}' has {int(df[column].isnull().sum())} missing "
                f"values ({df[column].isnull().mean() * 100:.2f}%)"
                for column in missing
            ]
            or ["  - No missing values detected in the primary table."]
        )

        if target_col is not None:
            target = (
                pd.read_csv(primary, sep=sep, usecols=[target_col])[target_col]
                if large_file
                else df[target_col]
            ).dropna()
            threshold = max(20, int(np.sqrt(max(len(target), 1))))
            is_regression = layout["task_type"] == "regression" or (
                layout["task_type"] == "supervised"
                and pd.api.types.is_numeric_dtype(target.dtype)
                and target.nunique() > threshold
            )
            analysis_report.append(
                f"\n4. Target Distribution for '{target_col}':"
            )
            if is_regression:
                analysis_report.append(
                    "  Inferred task type: regression (continuous numeric target)"
                )
                analysis_report.append(
                    "  Target summary: "
                    + target.describe(
                        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
                    ).to_string()
                )
            else:
                counts = target.value_counts()
                analysis_report.append(
                    "  Inferred task type: classification"
                )
                analysis_report.append("  Class Counts:")
                analysis_report.extend(
                    [
                        f"    - Class {value}: {count} instances "
                        f"({count / max(len(target), 1) * 100:.4f}%)"
                        for value, count in counts.head(50).items()
                    ]
                )
                rare = counts[counts < 10]
                if not rare.empty:
                    analysis_report.append(
                        "\n!!! CRITICAL INCONSISTENCY DETECTED !!!"
                    )
                    analysis_report.append(
                        f"  {len(rare)} target classes have fewer than 10 instances."
                    )
                    if (rare < 2).any():
                        analysis_report.append(
                            "  WARNING: singleton classes make ordinary stratified "
                            "splits invalid; use a non-stratified safe fallback."
                        )
        else:
            analysis_report.append(
                "\n4. Unsupervised Contract:\n"
                "  - Do not invent labels or treat the last feature as a target.\n"
                "  - Fit clustering on the discovered primary data table.\n"
                "  - Predict one cluster label for every sample-submission row.\n"
                "  - Use the harness silhouette proxy only for internal search; "
                "the competition Adjusted Rand Index remains externally scored."
            )
    except Exception as exc:
        analysis_report.append(f"Error during dataset analysis: {exc}")
    analysis_report.append("=========================================")
    return "\n".join(analysis_report)
