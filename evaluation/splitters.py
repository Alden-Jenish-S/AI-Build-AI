"""Deterministic sample/group/entity split planning."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from core.runtime_contracts import (
    DatasetBundle,
    FidelityProfile,
    SampleRecord,
    SplitPlan,
)


def _classification(bundle: DatasetBundle) -> bool:
    return bundle.task.problem_type in {
        "classification",
        "multilabel_classification",
    }


def _select_records(
    bundle: DatasetBundle,
    fidelity: FidelityProfile,
    seed: int,
) -> tuple[SampleRecord, ...]:
    records = bundle.train_records
    fraction = fidelity.sample_fraction
    if fraction >= 1 or len(records) <= fidelity.folds:
        return records
    target_size = max(fidelity.folds, int(len(records) * fraction))
    rng = np.random.default_rng(seed)

    groups = [
        record.group_id or record.entity_id
        for record in records
    ]
    if all(group is not None for group in groups):
        unique_groups = np.asarray(sorted(set(str(group) for group in groups)))
        rng.shuffle(unique_groups)
        selected_groups: set[str] = set()
        selected_count = 0
        counts = Counter(str(group) for group in groups)
        for group in unique_groups:
            selected_groups.add(str(group))
            selected_count += counts[str(group)]
            if selected_count >= target_size:
                break
        selected = tuple(
            record
            for record, group in zip(records, groups)
            if str(group) in selected_groups
        )
    else:
        indices = np.sort(
            rng.choice(len(records), size=target_size, replace=False)
        )
        selected = tuple(records[int(index)] for index in indices)
    if len(selected) < fidelity.folds:
        raise ValueError(
            "selected fidelity population is smaller than its fold count"
        )
    return selected


def create_split_plan(
    bundle: DatasetBundle,
    fidelity: FidelityProfile,
    *,
    seed: int = 42,
) -> tuple[tuple[SampleRecord, ...], SplitPlan]:
    """Select fidelity rows and assign folds without crossing leakage units."""
    records = _select_records(bundle, fidelity, seed)
    sample_ids = np.asarray([record.sample_id for record in records])
    targets = np.asarray([record.target for record in records])
    groups = np.asarray(
        [
            record.group_id or record.entity_id or record.sample_id
            for record in records
        ]
    )
    explicit_groups = any(
        record.group_id is not None or record.entity_id is not None
        for record in records
    )
    classification = _classification(bundle)
    folds = fidelity.folds
    if explicit_groups and len(set(groups.tolist())) < folds:
        raise ValueError(
            "group/entity count is smaller than the requested fold count"
        )

    if explicit_groups and classification:
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=seed
        )
        iterator = splitter.split(sample_ids, targets, groups)
        strategy = "stratified_group"
    elif explicit_groups:
        splitter = GroupKFold(n_splits=folds)
        iterator = splitter.split(sample_ids, targets, groups)
        strategy = "group"
    elif classification and min(Counter(targets.tolist()).values()) >= folds:
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed
        )
        iterator = splitter.split(sample_ids, targets)
        strategy = "stratified"
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        iterator = splitter.split(sample_ids)
        strategy = "kfold"

    assignments: dict[str, int] = {}
    for fold, (_, validation_indices) in enumerate(iterator):
        for index in validation_indices:
            assignments[str(sample_ids[index])] = fold
    if len(assignments) != len(records):
        raise RuntimeError("failed to assign every selected sample to a fold")

    if explicit_groups:
        group_folds: dict[str, set[int]] = {}
        for record in records:
            group = str(record.group_id or record.entity_id)
            group_folds.setdefault(group, set()).add(
                assignments[record.sample_id]
            )
        if any(len(group_assignment) != 1 for group_assignment in group_folds.values()):
            raise RuntimeError("a group/entity crossed evaluation folds")

    leakage_unit = (
        "group_id"
        if any(record.group_id is not None for record in records)
        else "entity_id"
        if any(record.entity_id is not None for record in records)
        else "sample_id"
    )
    return records, SplitPlan(
        assignments=assignments,
        strategy=strategy,
        seed=seed,
        leakage_unit=leakage_unit,
        group_field=bundle.task.group_id_field,
    )
