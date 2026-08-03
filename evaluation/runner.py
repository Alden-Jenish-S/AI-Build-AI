"""Harness-owned modality-neutral evaluation lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.runtime_contracts import DatasetBundle, PredictionBundle, SplitPlan
from .fidelity import get_fidelity_profile
from .metrics import metric_value, resolve_metric_name
from .prediction_io import load_prediction_bundle, write_assignment_table
from .splitters import create_split_plan


def prepare_evaluation_bundle(
    bundle: DatasetBundle,
    fidelity: str,
    *,
    seed: int = 42,
    output_dir: str | Path = ".",
) -> tuple[tuple, SplitPlan, dict]:
    """Select samples, assign folds, and persist a protocol manifest."""
    profile = get_fidelity_profile(bundle.task.modality, fidelity)
    records, plan = create_split_plan(bundle, profile, seed=seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_assignment_table(
        output / "fold_assignments.npz",
        sample_ids=[record.sample_id for record in records],
        fold_ids=np.asarray(
            [plan.assignments[record.sample_id] for record in records],
            dtype=np.int16,
        ),
    )
    manifest = {
        "protocol_version": 2,
        "task_id": bundle.task.task_id,
        "modality": bundle.task.modality,
        "component_modalities": list(bundle.task.component_modalities),
        "problem_type": bundle.task.problem_type,
        "output_type": bundle.task.output.type,
        "task_fingerprint": bundle.dataset_fingerprint,
        "split_fingerprint": plan.split_fingerprint,
        "fidelity": profile.to_dict(),
        "source_sample_count": len(bundle.train_records),
        "sample_count": len(records),
        "split_strategy": plan.strategy,
        "leakage_unit": plan.leakage_unit,
        "primary_metric": bundle.task.primary_metric,
        "metric_direction": bundle.task.metric_direction,
    }
    (output / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records, plan, manifest


def evaluate_prediction_bundle(
    manifest_path: str | Path,
    metric_name: str,
) -> dict[str, object]:
    """Recompute fold metrics from a typed prediction bundle."""
    bundle, predictions, targets, fold_ids = load_prediction_bundle(
        manifest_path
    )
    if targets is None:
        raise ValueError("supervised evaluation requires target payloads")
    resolved_metric = resolve_metric_name(
        metric_name,
        problem_type=bundle.metadata.get("problem_type"),
        output_type=bundle.output_type,
    )
    fold_scores = [
        metric_value(
            resolved_metric,
            targets[fold_ids == fold],
            predictions[fold_ids == fold],
            class_names=bundle.class_names,
        )
        for fold in sorted(np.unique(fold_ids))
    ]
    return {
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "folds": len(fold_scores),
        "fold_scores": [float(value) for value in fold_scores],
        "task_fingerprint": bundle.task_fingerprint,
        "split_fingerprint": bundle.split_fingerprint,
        "compatibility_key": bundle.compatibility_key,
        "output_type": bundle.output_type,
        "metric": resolved_metric,
    }
