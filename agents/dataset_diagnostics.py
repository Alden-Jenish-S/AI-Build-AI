"""Bounded, modality-neutral dataset diagnostics for agent planning."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from core.contracts import TaskSpec
from core.runtime_contracts import DatasetBundle
from modalities.common import resolve_task_source


_MAX_PROFILE_ROWS = 100_000
_MAX_DISTRIBUTION_ROWS = 20_000
_MAX_PAIRWISE_FEATURES = 128
_MAX_REPORTED_PAIRS = 100
_CORRELATION_THRESHOLD = 0.85


def _safe_source(task_dir: Path, source: object) -> Path | None:
    root = Path(task_dir).resolve()
    raw = Path(str(source or ""))
    candidates = (
        (raw,) if raw.is_absolute() else (root / raw, root / "input" / raw)
    )
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved != root and root not in resolved.parents:
            continue
        if resolved.is_file():
            return resolved
    return None


def _read_table(path: Path) -> pd.DataFrame | None:
    try:
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            return pd.read_csv(
                path,
                sep="\t" if suffix == ".tsv" else ",",
                nrows=_MAX_PROFILE_ROWS,
            )
        if suffix == ".parquet":
            # Avoid pandas engines that materialize the entire parquet file.
            # PyArrow batches keep profiling bounded on very large datasets.
            import pyarrow.parquet as parquet

            batches = parquet.ParquetFile(path).iter_batches(
                batch_size=_MAX_PROFILE_ROWS
            )
            first_batch = next(batches, None)
            return (
                first_batch.to_pandas()
                if first_batch is not None
                else pd.DataFrame()
            )
    except Exception:
        return None
    return None


def _diagnostic_frames(
    task_dir: Path, task: TaskSpec
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, bool]:
    """Resolve bounded train/test tables without assuming a modality."""
    cached: dict[Path, pd.DataFrame | None] = {}
    entries: list[tuple[str, Path, Mapping[str, object]]] = []
    for item in task.inputs.values():
        path = _safe_source(task_dir, item.source)
        if path is None or path.suffix.lower() not in {
            ".csv",
            ".tsv",
            ".parquet",
        }:
            continue
        entries.append((str(item.role).lower(), path, item.options))
        if path not in cached:
            cached[path] = _read_table(path)

    target_path = (
        _safe_source(task_dir, task.target.source)
        if task.target is not None and task.target.source
        else None
    )
    if target_path is not None and target_path not in cached:
        cached[target_path] = _read_table(target_path)

    train_path = next(
        (path for role, path, _ in entries if role in {"train", "data"}),
        None,
    )
    test_path = next(
        (path for role, path, _ in entries if role == "test"), None
    )
    target_field = task.target.field if task.target is not None else None
    if train_path is None and target_path in cached:
        train_path = target_path
    if train_path is None and target_field:
        train_path = next(
            (
                path
                for path, frame in cached.items()
                if frame is not None and target_field in frame.columns
            ),
            None,
        )
    if train_path is None and cached:
        train_path = next(iter(cached))

    train = cached.get(train_path) if train_path is not None else None
    test = cached.get(test_path) if test_path is not None else None
    sampled = bool(train is not None and len(train) >= _MAX_PROFILE_ROWS)
    if train is None:
        return None, test, sampled

    # A modality manifest often stores both splits in one table. Resolve the
    # split generically from its declared field, a conventional field, or
    # target availability.
    if test is None:
        split_fields = []
        for _, path, options in entries:
            if path != train_path:
                continue
            configured = options.get("split_field")
            if configured:
                split_fields.append(str(configured))
        split_fields.extend(("split", "set", "partition"))
        split_field = next(
            (field for field in split_fields if field in train.columns), None
        )
        if split_field is not None:
            values = train[split_field].astype(str).str.strip().str.lower()
            test_mask = values.isin({"test", "testing", "holdout"})
            train_mask = values.isin({"train", "training"})
            if test_mask.any() and train_mask.any():
                test = train.loc[test_mask].copy()
                train = train.loc[train_mask].copy()
        elif target_field and target_field in train.columns:
            test_mask = train[target_field].isna()
            if test_mask.any() and (~test_mask).any():
                test = train.loc[test_mask].copy()
                train = train.loc[~test_mask].copy()
    return train, test, sampled


def _feature_columns(task: TaskSpec, train: pd.DataFrame) -> list[object]:
    excluded = {
        field
        for field in (
            task.target.field if task.target is not None else None,
            task.sample_id_field,
            task.entity_id_field,
            task.group_id_field,
        )
        if field
    }
    excluded.update(
        column
        for column in train.columns
        if str(column).lower() in {"split", "set", "partition"}
    )
    return [column for column in train.columns if column not in excluded]


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _target_values(
    task: TaskSpec,
    train: pd.DataFrame | None,
    bundle: DatasetBundle | None,
) -> pd.Series:
    if (
        train is not None
        and task.target is not None
        and task.target.field in train.columns
    ):
        return train[task.target.field]
    if bundle is not None:
        return pd.Series(
            [record.target for record in bundle.train_records], dtype=object
        )
    return pd.Series(dtype=object)


def _target_topology(task: TaskSpec, values: pd.Series) -> dict[str, object]:
    total = int(len(values))
    clean = values.dropna()
    result: dict[str, object] = {
        "available": bool(len(clean)),
        "observed_count": total,
        "missing_count": int(values.isna().sum()),
        "missing_percentage": round(
            100.0 * float(values.isna().mean()), 6
        )
        if total
        else None,
        "target_skewness": None,
        "quantiles": {},
        "class_distribution": {},
        "minority_class_fraction": None,
        "minority_to_majority_ratio": None,
        "majority_to_minority_ratio": None,
    }
    if clean.empty:
        result["reason"] = "no supervised target values were available"
        return result

    target_type = str(task.target.type or "") if task.target is not None else ""
    if target_type.endswith("_path"):
        result.update(
            {
                "topology": "structured_file_references",
                "target_storage": target_type,
                "unique_reference_count": int(clean.astype(str).nunique()),
                "reason": (
                    "class balance and numeric skew require decoding the "
                    "structured targets and are not inferred from path names"
                ),
            }
        )
        return result

    is_regression = task.problem_type == "regression"
    if is_regression:
        numeric = pd.to_numeric(clean, errors="coerce").dropna().astype(float)
        if numeric.empty:
            result["reason"] = "regression target could not be converted to numeric"
            return result
        quantiles = numeric.quantile(
            [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
        )
        result["target_skewness"] = _finite(numeric.skew())
        result["quantiles"] = {
            name: _finite(quantiles.iloc[index])
            for index, name in enumerate(
                ("min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max")
            )
        }
        result["topology"] = "continuous"
        return result

    counts = clean.astype(str).value_counts(dropna=False)
    minimum = int(counts.min())
    maximum = int(counts.max())
    result.update(
        {
            "topology": "categorical",
            "class_count": int(len(counts)),
            "class_distribution": {
                str(label): int(count)
                for label, count in counts.head(200).items()
            },
            "minority_class_fraction": _finite(minimum / len(clean)),
            "minority_to_majority_ratio": _finite(
                minimum / maximum if maximum else None
            ),
            "majority_to_minority_ratio": _finite(
                maximum / minimum if minimum else None
            ),
            "target_skewness_reason": (
                "skewness is not defined for nominal class labels"
            ),
        }
    )
    return result


def _bundle_diagnostic_frames(
    bundle: DatasetBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build bounded feature tables while preserving bundle-owned splits."""
    def rows(records: tuple) -> pd.DataFrame:
        flattened = []
        for record in records[:_MAX_PROFILE_ROWS]:
            row: dict[str, object] = {}
            for input_name, value in record.inputs.items():
                if isinstance(value, Mapping):
                    for field, item in value.items():
                        row[f"{input_name}__{field}"] = item
                elif isinstance(value, (bool, int, float, np.generic)):
                    row[input_name] = value
                elif value is None:
                    row[input_name] = None
            flattened.append(row)
        return pd.DataFrame(flattened)

    return rows(bundle.train_records), rows(bundle.test_records)


def _structured_target_signals(
    task_dir: Path,
    task: TaskSpec,
    bundle: DatasetBundle | None,
) -> dict[str, object]:
    if (
        bundle is None
        or task.target is None
        or task.target.type != "mask_path"
    ):
        return {}
    records = bundle.train_records
    if not records:
        return {}
    sample_count = min(len(records), 256)
    indices = np.linspace(0, len(records) - 1, sample_count, dtype=int)
    foreground_fractions = []
    dimensions: dict[str, int] = {}
    unreadable = 0
    try:
        from PIL import Image
    except ImportError:
        return {
            "mask_diagnostics_available": False,
            "reason": "Pillow is unavailable",
        }
    for index in indices:
        reference = records[int(index)].target
        try:
            path = resolve_task_source(task_dir, str(reference))
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L"))
        except (FileNotFoundError, OSError, ValueError):
            unreadable += 1
            continue
        dimensions[f"{mask.shape[1]}x{mask.shape[0]}"] = (
            dimensions.get(f"{mask.shape[1]}x{mask.shape[0]}", 0) + 1
        )
        foreground_fractions.append(float(np.mean(mask > 0)))
    values = np.asarray(foreground_fractions, dtype=float)
    return {
        "mask_diagnostics_available": bool(len(values)),
        "mask_sample_count": int(len(values)),
        "mask_unreadable_count": int(unreadable),
        "mask_dimensions": dimensions,
        "empty_mask_fraction": (
            float(np.mean(values == 0.0)) if len(values) else None
        ),
        "foreground_fraction_quantiles": (
            {
                name: float(value)
                for name, value in zip(
                    ("min", "q25", "median", "q75", "max"),
                    np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]),
                )
            }
            if len(values)
            else {}
        ),
    }


def _deterministic_numeric_sample(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    numeric = numeric[np.isfinite(numeric)]
    if len(numeric) > _MAX_DISTRIBUTION_ROWS:
        indices = np.linspace(
            0, len(numeric) - 1, _MAX_DISTRIBUTION_ROWS, dtype=int
        )
        numeric = numeric[indices]
    return numeric


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float | None:
    if not len(left) or not len(right):
        return None
    left = np.sort(left)
    right = np.sort(right)
    points = np.sort(np.concatenate((left, right)))
    left_cdf = np.searchsorted(left, points, side="right") / len(left)
    right_cdf = np.searchsorted(right, points, side="right") / len(right)
    return _finite(np.max(np.abs(left_cdf - right_cdf)))


def _data_quality_and_drift(
    features: list[object],
    train: pd.DataFrame | None,
    test: pd.DataFrame | None,
) -> dict[str, object]:
    if train is None:
        return {
            "available": False,
            "reason": "no tabular or manifest feature table was available",
            "missingness_percentages": {},
            "train_test_ks_drift": {},
        }
    missingness = {}
    for column in features:
        entry: dict[str, float | None] = {
            "train": round(100.0 * float(train[column].isna().mean()), 6)
        }
        entry["test"] = (
            round(100.0 * float(test[column].isna().mean()), 6)
            if test is not None and column in test.columns and len(test)
            else None
        )
        missingness[str(column)] = entry

    drift: dict[str, object] = {}
    common = [
        column
        for column in features
        if test is not None
        and column in test.columns
        and pd.api.types.is_numeric_dtype(train[column])
        and pd.api.types.is_numeric_dtype(test[column])
    ]
    truncated = len(common) > _MAX_PAIRWISE_FEATURES
    if truncated:
        ranked = sorted(
            common,
            key=lambda column: _finite(
                pd.to_numeric(train[column], errors="coerce").var()
            )
            or 0.0,
            reverse=True,
        )
        common = ranked[:_MAX_PAIRWISE_FEATURES]
    for column in common:
        left = _deterministic_numeric_sample(train[column])
        right = _deterministic_numeric_sample(test[column])
        statistic = _ks_statistic(left, right)
        if statistic is None:
            continue
        critical = 1.36 * math.sqrt(
            (len(left) + len(right)) / (len(left) * len(right))
        )
        actionable_threshold = max(0.10, critical)
        drift[str(column)] = {
            "statistic": statistic,
            "actionable_threshold": _finite(actionable_threshold),
            "drift_detected": bool(statistic > actionable_threshold),
            "train_count": int(len(left)),
            "test_count": int(len(right)),
        }
    detected = [
        name
        for name, details in drift.items()
        if isinstance(details, Mapping) and details.get("drift_detected")
    ]
    return {
        "available": True,
        "missingness_percentages": missingness,
        "train_test_ks_drift": {
            "available": test is not None and bool(len(test)),
            "method": "two-sample Kolmogorov-Smirnov statistic",
            "numeric_features_evaluated": len(drift),
            "numeric_features_truncated": truncated,
            "per_feature": drift,
            "drifted_features": detected,
            "max_statistic": max(
                (
                    float(details["statistic"])
                    for details in drift.values()
                    if isinstance(details, Mapping)
                ),
                default=None,
            ),
        },
    }


def _complexity(
    features: list[object],
    train: pd.DataFrame | None,
    *,
    sampled: bool,
) -> dict[str, object]:
    if train is None:
        return {
            "available": False,
            "sample_count": None,
            "feature_count": None,
            "feature_to_sample_ratio": None,
            "high_collinearity": {},
        }
    sample_count = int(len(train))
    feature_count = int(len(features))
    numeric = [
        column
        for column in features
        if pd.api.types.is_numeric_dtype(train[column])
    ]
    truncated = len(numeric) > _MAX_PAIRWISE_FEATURES
    if truncated:
        numeric = sorted(
            numeric,
            key=lambda column: _finite(
                pd.to_numeric(train[column], errors="coerce").var()
            )
            or 0.0,
            reverse=True,
        )[:_MAX_PAIRWISE_FEATURES]
    pairs: list[dict[str, object]] = []
    if len(numeric) >= 2:
        correlation = (
            train[numeric]
            .head(_MAX_DISTRIBUTION_ROWS)
            .apply(pd.to_numeric, errors="coerce")
            .corr()
            .abs()
        )
        for left_index, left in enumerate(numeric):
            for right in numeric[left_index + 1 :]:
                value = _finite(correlation.loc[left, right])
                if value is not None and value > _CORRELATION_THRESHOLD:
                    pairs.append(
                        {
                            "left": str(left),
                            "right": str(right),
                            "absolute_correlation": value,
                        }
                    )
    pairs.sort(
        key=lambda item: float(item["absolute_correlation"]), reverse=True
    )
    pair_count = len(pairs)
    return {
        "available": True,
        "sample_count": sample_count,
        "sample_count_is_sampled": sampled,
        "feature_count": feature_count,
        "feature_to_sample_ratio": _finite(
            feature_count / sample_count if sample_count else None
        ),
        "high_collinearity": {
            "threshold": _CORRELATION_THRESHOLD,
            "numeric_features_evaluated": len(numeric),
            "numeric_features_truncated": truncated,
            "pair_count": pair_count,
            "pairs": pairs[:_MAX_REPORTED_PAIRS],
            "pairs_truncated": pair_count > _MAX_REPORTED_PAIRS,
        },
    }


def _synthesize_directives(
    task: TaskSpec,
    target: Mapping[str, object],
    quality: Mapping[str, object],
    complexity: Mapping[str, object],
) -> list[str]:
    directives: list[str] = []
    if target.get("topology") == "structured_file_references":
        directives.append(
            "Decode structured target references lazily after the harness split, "
            "preserve spatial/temporal alignment with each input, and never treat "
            "target path strings as class labels."
        )
    empty_mask_fraction = _finite(target.get("empty_mask_fraction"))
    if empty_mask_fraction is not None and empty_mask_fraction >= 0.10:
        directives.append(
            f"Empty masks comprise about {empty_mask_fraction:.1%} of the bounded "
            "target sample; preserve empty examples in validation and compare an "
            "explicit empty-mask gate or calibrated minimum-component rule."
        )
    class_count = target.get("class_count")
    if class_count is not None and int(class_count) < 2:
        directives.append(
            "Only one target class was observed; supervised class validation "
            "is not identifiable until label extraction or split construction "
            "is corrected."
        )
    target_missing = _finite(target.get("missing_percentage"))
    if target_missing is not None and target_missing > 0:
        directives.append(
            f"The target is missing for {target_missing:.2f}% of profiled "
            "training rows; never impute labels, and separate unlabeled rows "
            "from supervised fitting and validation."
        )
    if task.group_id_field:
        directives.append(
            f"Use group-aware validation keyed by {task.group_id_field!r}; "
            "never split one group across training and validation."
        )
    elif task.time_field:
        directives.append(
            f"Use chronology-aware validation keyed by {task.time_field!r}; "
            "do not use random folds across future observations."
        )

    imbalance = _finite(target.get("minority_to_majority_ratio"))
    if imbalance is not None and imbalance < 0.20:
        split_kind = "stratified group" if task.group_id_field else "stratified"
        directives.append(
            f"Severe class imbalance detected (minority/majority={imbalance:.4f}); "
            f"use {split_kind} splitting and compare class weighting or calibrated "
            "sampling using the task metric."
        )
    skewness = _finite(target.get("target_skewness"))
    quantiles = target.get("quantiles")
    minimum = (
        _finite(quantiles.get("min"))
        if isinstance(quantiles, Mapping)
        else None
    )
    if skewness is not None and abs(skewness) >= 1.0:
        transform = (
            "a leakage-safe log1p target transform with inverse transformation"
            if minimum is not None and minimum >= 0
            else "a leakage-safe Yeo-Johnson or robust target transformation"
        )
        directives.append(
            f"The regression target is strongly skewed (skewness={skewness:.3f}); "
            f"evaluate {transform}, scoring predictions on the original target scale."
        )

    missingness = quality.get("missingness_percentages")
    if isinstance(missingness, Mapping):
        affected = [
            str(name)
            for name, values in missingness.items()
            if isinstance(values, Mapping)
            and max(
                (
                    _finite(values.get("train")) or 0.0,
                    _finite(values.get("test")) or 0.0,
                )
            )
            >= 5.0
        ]
        if affected:
            directives.append(
                "Material missingness is present; fit imputation inside each "
                "training fold and test missingness indicators for: "
                + ", ".join(affected[:12])
                + ("." if len(affected) <= 12 else ", and other affected features.")
            )

    drift = quality.get("train_test_ks_drift")
    drifted = (
        list(drift.get("drifted_features", []))
        if isinstance(drift, Mapping)
        else []
    )
    if drifted:
        directives.append(
            "Train/test numeric drift is material for "
            + ", ".join(str(item) for item in drifted[:12])
            + "; prefer drift-robust transforms or regularization, validate "
            "the suspected shift explicitly, and never fit transforms on test rows."
        )

    collinearity = complexity.get("high_collinearity")
    pair_count = (
        int(collinearity.get("pair_count", 0) or 0)
        if isinstance(collinearity, Mapping)
        else 0
    )
    if pair_count:
        directives.append(
            f"Detected {pair_count} highly collinear numeric feature pair(s) "
            f"(|r|>{_CORRELATION_THRESHOLD}); evaluate fold-local redundancy "
            "removal, regularization, or stable dimensionality reduction."
        )

    ratio = _finite(complexity.get("feature_to_sample_ratio"))
    if ratio is not None and ratio >= 0.5:
        directives.append(
            f"The feature-to-sample ratio is high (D/N={ratio:.4f}); use "
            "strong regularization and perform feature selection or dimensionality "
            "reduction strictly inside validation folds."
        )
    if not directives:
        directives.append(
            "No dominant pathology was detected in the bounded profile; use "
            "leakage-safe preprocessing and let validation evidence determine "
            "whether additional robustness mechanisms are warranted."
        )
    return directives


def build_dataset_diagnostics(
    task_dir: Path,
    task: TaskSpec,
    *,
    bundle: DatasetBundle | None = None,
) -> dict[str, object]:
    """Compute a stable diagnostic schema for any registered modality."""
    train, test, sampled = _diagnostic_frames(Path(task_dir), task)
    if bundle is not None and (
        train is None
        or (
            task.target is not None
            and str(task.target.type or "").endswith("_path")
        )
    ):
        train, test = _bundle_diagnostic_frames(bundle)
        sampled = len(bundle.train_records) > _MAX_PROFILE_ROWS
    features = _feature_columns(task, train) if train is not None else []
    target = _target_topology(task, _target_values(task, train, bundle))
    target.update(_structured_target_signals(Path(task_dir), task, bundle))
    quality = _data_quality_and_drift(features, train, test)
    complexity = _complexity(features, train, sampled=sampled)
    return {
        "schema_version": 1,
        "profiling_limits": {
            "maximum_table_rows": _MAX_PROFILE_ROWS,
            "maximum_distribution_rows": _MAX_DISTRIBUTION_ROWS,
            "maximum_pairwise_numeric_features": _MAX_PAIRWISE_FEATURES,
        },
        "target_topology": target,
        "data_quality_and_drift": quality,
        "complexity_metrics": complexity,
        "synthesized_directives": _synthesize_directives(
            task, target, quality, complexity
        ),
    }


def render_dataset_analysis_markdown(
    task: TaskSpec, profile: Mapping[str, object]
) -> str:
    """Render concise LLM context without duplicating the full JSON profile."""
    diagnostics = profile.get("diagnostics", {})
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    target = diagnostics.get("target_topology", {})
    quality = diagnostics.get("data_quality_and_drift", {})
    complexity = diagnostics.get("complexity_metrics", {})
    target = target if isinstance(target, Mapping) else {}
    quality = quality if isinstance(quality, Mapping) else {}
    complexity = complexity if isinstance(complexity, Mapping) else {}
    drift = quality.get("train_test_ks_drift", {})
    drift = drift if isinstance(drift, Mapping) else {}
    collinearity = complexity.get("high_collinearity", {})
    collinearity = (
        collinearity if isinstance(collinearity, Mapping) else {}
    )
    missingness = quality.get("missingness_percentages", {})
    missingness = missingness if isinstance(missingness, Mapping) else {}
    missing_features = sorted(
        (
            (
                str(name),
                max(
                    _finite(values.get("train")) or 0.0,
                    _finite(values.get("test")) or 0.0,
                ),
            )
            for name, values in missingness.items()
            if isinstance(values, Mapping)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    lines = [
        "# Dataset Analysis",
        "",
        "## Task contract",
        "",
        f"- Modality: `{task.modality}`",
        f"- Problem type: `{task.problem_type}`",
        f"- Output type: `{task.output.type}`",
        f"- Primary metric: `{task.primary_metric}` ({task.metric_direction})",
        "",
        "## Target topology",
        "",
        f"- Observed targets: {target.get('observed_count')}",
        f"- Skewness: {target.get('target_skewness')}",
        f"- Quantiles: {target.get('quantiles', {})}",
        f"- Minority/majority ratio: {target.get('minority_to_majority_ratio')}",
        "",
        "## Data quality and drift",
        "",
        "- Highest missingness: "
        + (", ".join(f"{name}={value:.2f}%" for name, value in missing_features[:20]) or "none detected"),
        f"- Maximum train/test KS statistic: {drift.get('max_statistic')}",
        f"- Drifted numeric features: {drift.get('drifted_features', [])}",
        "",
        "## Complexity",
        "",
        f"- Sample count: {complexity.get('sample_count')}",
        f"- Feature count: {complexity.get('feature_count')}",
        f"- Feature-to-sample ratio (D/N): {complexity.get('feature_to_sample_ratio')}",
        f"- Highly collinear pair count (|r|>{_CORRELATION_THRESHOLD}): {collinearity.get('pair_count')}",
        "",
        "## Synthesized modeling directives",
        "",
    ]
    directives = diagnostics.get("synthesized_directives", [])
    lines.extend(f"- {directive}" for directive in directives)
    return "\n".join(lines).rstrip() + "\n"
