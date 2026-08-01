"""Typed prediction payload persistence with legacy scalar exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from core.runtime_contracts import PredictionBundle, SplitPlan


def _storage_array(value: np.ndarray) -> np.ndarray:
    """Avoid pickled object arrays for ordinary string labels/text."""
    array = np.asarray(value)
    if array.dtype != object:
        return array
    flattened = array.reshape(-1).tolist()
    if all(
        item is None
        or isinstance(item, (str, bytes, bool, int, float, np.generic))
        for item in flattened
    ):
        return np.asarray(
            ["" if item is None else str(item) for item in flattened],
            dtype=str,
        ).reshape(array.shape)
    return array


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _ragged_array(values: Sequence[Any]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = list(values)
    return result


def legacy_prediction_payload(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract scalar predictions or a probability matrix from legacy OOF."""
    if "prediction" in frame.columns:
        return frame["prediction"].to_numpy(), ()
    columns = [
        column
        for column in frame.columns
        if str(column).startswith(("prediction::", "prediction_"))
    ]
    if not columns:
        raise ValueError(
            "OOF output must contain 'prediction' or "
            "'prediction::<class>' columns"
        )
    predictions = frame[columns].to_numpy(dtype=float)
    class_names = tuple(
        str(column).split("::", 1)[1]
        if str(column).startswith("prediction::")
        else str(column)[len("prediction_") :]
        for column in columns
    )
    if class_names == tuple(str(index) for index in range(len(class_names))):
        class_names = ()
    return predictions, class_names


def write_legacy_oof(
    path: str | Path,
    *,
    sample_ids: Sequence[object],
    targets: np.ndarray,
    predictions: np.ndarray,
    fold_ids: np.ndarray | None = None,
    class_names: Sequence[object] = (),
) -> None:
    """Persist the backward-compatible scalar or probability OOF schema."""
    values = np.asarray(predictions)
    data: dict[str, object] = {
        "row_id": list(sample_ids),
        "target": np.asarray(targets).reshape(-1),
    }
    if values.ndim == 1:
        data["prediction"] = values
    elif values.ndim == 2:
        names = (
            tuple(str(item) for item in class_names)
            if class_names
            else tuple(str(index) for index in range(values.shape[1]))
        )
        if len(names) != values.shape[1]:
            raise ValueError("class names do not align with probability columns")
        for index, name in enumerate(names):
            data[f"prediction::{name}"] = values[:, index]
    else:
        raise ValueError("legacy OOF supports scalar or matrix predictions")
    if fold_ids is not None:
        data["fold_id"] = np.asarray(fold_ids).reshape(-1)
    pd.DataFrame(data).to_csv(path, index=False)


def write_prediction_bundle(
    output_dir: str | Path,
    *,
    task_fingerprint: str,
    split_plan: SplitPlan,
    output_type: str,
    sample_ids: Sequence[str],
    predictions: np.ndarray,
    targets: np.ndarray | None = None,
    fold_ids: np.ndarray | None = None,
    class_names: Sequence[str] = (),
    metadata: dict | None = None,
    write_legacy_csv: bool = True,
) -> PredictionBundle:
    """Write an aligned compressed payload and its validated manifest."""
    root = Path(output_dir)
    prediction_dir = root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    ids = tuple(str(item) for item in sample_ids)
    values = _storage_array(np.asarray(predictions))
    if values.ndim < 1 or values.shape[0] != len(ids):
        raise ValueError(
            "prediction samples must occupy axis zero and align with sample IDs"
        )
    if np.issubdtype(values.dtype, np.number) and not np.isfinite(
        values.astype(float, copy=False)
    ).all():
        raise ValueError("prediction payload contains non-finite values")
    target_values = (
        None
        if targets is None
        else _storage_array(np.asarray(targets))
    )
    if target_values is not None and len(target_values) != len(ids):
        raise ValueError("targets do not align with prediction sample IDs")
    if fold_ids is None:
        folds = np.asarray(
            [split_plan.assignments[sample_id] for sample_id in ids],
            dtype=np.int16,
        )
    else:
        folds = np.asarray(fold_ids, dtype=np.int16)
    if len(folds) != len(ids):
        raise ValueError("fold IDs do not align with predictions")

    use_json = values.dtype == object or (
        target_values is not None and target_values.dtype == object
    )
    if use_json:
        payload_path = prediction_dir / "payload.json"
        payload = {
            "predictions": _json_value(values),
            "fold_ids": folds.tolist(),
        }
        if target_values is not None:
            payload["targets"] = _json_value(target_values)
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        payload_format = "json"
    else:
        payload_path = prediction_dir / "payload.npz"
        payload = {
            "predictions": values,
            "fold_ids": folds,
        }
        if target_values is not None:
            payload["targets"] = target_values
        np.savez_compressed(payload_path, **payload)
        payload_format = "npz"
    bundle = PredictionBundle(
        task_fingerprint=task_fingerprint,
        split_fingerprint=split_plan.split_fingerprint,
        output_type=output_type,
        sample_ids=ids,
        payload_path=payload_path.name,
        payload_format=payload_format,
        class_names=tuple(str(item) for item in class_names),
        target_path=payload_path.name if target_values is not None else None,
        fold_ids_path=payload_path.name,
        metadata=metadata or {},
    )
    (prediction_dir / "manifest.json").write_text(
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if (
        write_legacy_csv
        and values.ndim in {1, 2}
        and target_values is not None
        and target_values.ndim == 1
    ):
        write_legacy_oof(
            root / "oof_predictions.csv",
            sample_ids=ids,
            targets=target_values,
            predictions=values,
            fold_ids=folds,
            class_names=class_names,
        )
    return bundle


def load_prediction_bundle(
    manifest_path: str | Path,
) -> tuple[PredictionBundle, np.ndarray, np.ndarray | None, np.ndarray]:
    """Load and revalidate a typed prediction payload."""
    manifest_path = Path(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = PredictionBundle.from_dict(raw)
    payload_path = manifest_path.parent / bundle.payload_path
    if bundle.payload_format not in {"json", "npz"} or not payload_path.is_file():
        raise ValueError(
            f"unsupported or missing prediction payload: {payload_path}"
        )
    if bundle.payload_format == "npz":
        with np.load(payload_path, allow_pickle=False) as payload:
            predictions = payload["predictions"]
            targets = payload["targets"] if "targets" in payload.files else None
            fold_ids = payload["fold_ids"]
    else:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        raw_predictions = payload["predictions"]
        raw_targets = payload.get("targets")
        try:
            predictions = np.asarray(raw_predictions)
        except ValueError:
            predictions = _ragged_array(raw_predictions)
        if raw_targets is None:
            targets = None
        else:
            try:
                targets = np.asarray(raw_targets)
            except ValueError:
                targets = _ragged_array(raw_targets)
        fold_ids = np.asarray(payload["fold_ids"], dtype=np.int16)
    if len(predictions) != len(bundle.sample_ids):
        raise ValueError("prediction payload does not align with its manifest")
    if targets is not None and len(targets) != len(bundle.sample_ids):
        raise ValueError("target payload does not align with its manifest")
    if len(fold_ids) != len(bundle.sample_ids):
        raise ValueError("fold payload does not align with its manifest")
    if np.issubdtype(np.asarray(predictions).dtype, np.number) and not np.isfinite(
        np.asarray(predictions, dtype=float)
    ).all():
        raise ValueError("prediction payload contains non-finite values")
    return bundle, predictions, targets, fold_ids
