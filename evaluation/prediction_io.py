"""Typed prediction payload persistence with legacy scalar exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from core.runtime_contracts import PredictionBundle, SplitPlan


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
    values = np.asarray(predictions)
    if values.ndim not in {1, 2} or values.shape[0] != len(ids):
        raise ValueError(
            "prediction samples must occupy axis zero and align with sample IDs"
        )
    if not np.isfinite(values.astype(float)).all():
        raise ValueError("prediction payload contains non-finite values")
    target_values = None if targets is None else np.asarray(targets)
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

    payload_path = prediction_dir / "payload.npz"
    payload = {
        "predictions": values,
        "fold_ids": folds,
    }
    if target_values is not None:
        payload["targets"] = target_values
    np.savez_compressed(payload_path, **payload)
    bundle = PredictionBundle(
        task_fingerprint=task_fingerprint,
        split_fingerprint=split_plan.split_fingerprint,
        output_type=output_type,
        sample_ids=ids,
        payload_path=payload_path.name,
        payload_format="npz",
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
        and values.ndim == 1
        and target_values is not None
    ):
        pd.DataFrame(
            {
                "row_id": ids,
                "target": target_values,
                "prediction": values,
                "fold_id": folds,
            }
        ).to_csv(root / "oof_predictions.csv", index=False)
    return bundle


def load_prediction_bundle(
    manifest_path: str | Path,
) -> tuple[PredictionBundle, np.ndarray, np.ndarray | None, np.ndarray]:
    """Load and revalidate a typed prediction payload."""
    manifest_path = Path(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = PredictionBundle.from_dict(raw)
    payload_path = manifest_path.parent / bundle.payload_path
    if bundle.payload_format != "npz" or not payload_path.is_file():
        raise ValueError(
            f"unsupported or missing prediction payload: {payload_path}"
        )
    with np.load(payload_path, allow_pickle=False) as payload:
        predictions = payload["predictions"]
        targets = payload["targets"] if "targets" in payload.files else None
        fold_ids = payload["fold_ids"]
    if len(predictions) != len(bundle.sample_ids):
        raise ValueError("prediction payload does not align with its manifest")
    if targets is not None and len(targets) != len(bundle.sample_ids):
        raise ValueError("target payload does not align with its manifest")
    if len(fold_ids) != len(bundle.sample_ids):
        raise ValueError("fold payload does not align with its manifest")
    if not np.isfinite(np.asarray(predictions, dtype=float)).all():
        raise ValueError("prediction payload contains non-finite values")
    return bundle, predictions, targets, fold_ids
