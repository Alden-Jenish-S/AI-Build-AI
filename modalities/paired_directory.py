"""High-confidence discovery for directory-backed structured targets.

Many vision and media datasets encode supervision by pairing files rather than
by placing a scalar target column in a table.  This module recognizes common
split/input/target directory conventions without guessing from arbitrary file
names.  Ambiguous layouts intentionally return ``None`` so callers can request
an explicit ``task_config.json`` instead of constructing the wrong contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

from core.contracts import TaskSpec
from core.runtime_contracts import DatasetBundle, SampleRecord
from .common import (
    canonical_input_reference,
    json_safe,
    records_to_bundle,
    task_input_root,
)


_MEDIA_EXTENSIONS = {
    "audio": frozenset({".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}),
    "image": frozenset(
        {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    ),
    "video": frozenset(
        {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
    ),
}
_TABLE_EXTENSIONS = frozenset({".csv", ".tsv"})
_ID_FIELDS = (
    "id",
    "sample_id",
    "image_id",
    "audio_id",
    "video_id",
    "file_id",
    "filename",
)


def _normalized_name(path: Path) -> str:
    return path.name.strip().lower().replace("-", "_").replace(" ", "_")


def _relative_source(task_dir: Path, path: Path) -> str:
    return Path(path).resolve().relative_to(task_input_root(task_dir)).as_posix()


def _files(directory: Path, extensions: Iterable[str]) -> tuple[Path, ...]:
    allowed = frozenset(extensions)
    return tuple(
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in allowed
    )


def _file_key(directory: Path, path: Path) -> str:
    return path.relative_to(directory).with_suffix("").as_posix()


def _candidate_directories(root: Path) -> tuple[Path, ...]:
    candidates = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        try:
            depth = len(path.relative_to(root).parts)
        except ValueError:
            continue
        if depth <= 4:
            candidates.append(path)
    return tuple(sorted(candidates))


def _observed_directory_files(directory: Path) -> tuple[Path, ...]:
    """Return one homogeneous leaf collection without interpreting its name."""
    return tuple(path for path in sorted(directory.iterdir()) if path.is_file())


def _storage_family(path: Path) -> str | None:
    """Identify decodable storage from bytes/extensions, not directory labels."""
    suffix = path.suffix.lower()
    for family, extensions in _MEDIA_EXTENSIONS.items():
        if suffix in extensions:
            return family
    if suffix in {".csv", ".json", ".jsonl", ".md", ".txt", ".tsv", ".xml"}:
        return "structured_text"
    return None


def _directory_collection(
    directory: Path,
) -> tuple[str, dict[str, Path]] | None:
    files = _observed_directory_files(directory)
    if not files:
        return None
    sampled_families = [_storage_family(path) for path in files[:20]]
    families = {family for family in sampled_families if family is not None}
    if len(families) != 1 or any(family is None for family in sampled_families):
        return None
    family = next(iter(families))
    mapping: dict[str, Path] = {}
    for path in files:
        if _storage_family(path) != family:
            return None
        key = _file_key(directory, path)
        if key in mapping:
            return None
        mapping[key] = path
    return family, mapping


def _raster_target_score(paths: Iterable[Path]) -> float | None:
    """Measure whether observed raster values look target-like.

    A low score means few unique values relative to the array size. This is an
    observation about the actual files and does not depend on directory names.
    """
    scores = []
    for path in list(paths)[:8]:
        try:
            with Image.open(path) as image:
                values = np.asarray(image)
        except Exception:
            continue
        if values.size == 0:
            continue
        unique = np.unique(values.reshape(-1)[:100_000]).size
        scores.append(unique / min(values.size, 100_000))
    return float(np.median(scores)) if scores else None


def _table_id_sets(root: Path) -> list[tuple[Path, str, set[str], tuple[str, ...]]]:
    result = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TABLE_EXTENSIONS:
            continue
        try:
            frame = _table(path)
        except Exception:
            continue
        columns = tuple(str(column) for column in frame.columns)
        for column in columns:
            values = {
                Path(str(value).strip()).stem
                for value in frame[column].dropna().tolist()
                if str(value).strip()
            }
            if values:
                result.append((path, column, values, columns))
    return result


def _name_tokens(value: object) -> set[str]:
    normalized = str(value).lower().replace("-", "_").replace(" ", "_")
    tokens = {token for token in normalized.split("_") if token}
    tokens.update(token[:-1] for token in list(tokens) if token.endswith("s"))
    return tokens


def _unique_file_map(directory: Path, extensions: Iterable[str]) -> dict[str, Path] | None:
    result: dict[str, Path] = {}
    for path in _files(directory, extensions):
        key = _file_key(directory, path)
        if key in result:
            return None
        result[key] = path
    return result or None


def _table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")


@dataclass(frozen=True)
class PairedDirectoryLayout:
    """Resolved file pairs and optional entity-aligned tabular metadata."""

    input_modality: str
    problem_type: str
    output_type: str
    target_type: str
    target_field: str
    train_input_dir: Path
    target_dir: Path
    test_input_dir: Path | None
    train_pairs: tuple[tuple[str, Path, Path], ...]
    test_items: tuple[tuple[str, Path], ...]
    metadata_path: Path | None = None
    metadata_id_field: str | None = None
    metadata_feature_fields: tuple[str, ...] = ()
    submission_id_field: str | None = None
    submission_columns: tuple[str, ...] = ()

    @property
    def modality(self) -> str:
        return "multimodal" if self.metadata_path is not None else self.input_modality


def _submission_schema(
    root: Path, test_keys: set[str]
) -> tuple[str | None, tuple[str, ...], Path | None]:
    """Find the output template by its alignment to observed inference IDs."""
    ranked = []
    if not test_keys:
        return None, (), None
    for path, field, values, columns in _table_id_sets(root):
        if len(columns) < 2:
            continue
        overlap = len(values & test_keys)
        coverage = overlap / len(test_keys)
        if coverage < 0.90:
            continue
        row_distance = abs(len(values) - len(test_keys)) / max(len(test_keys), 1)
        ranked.append((coverage, -row_distance, path, field, columns))
    if not ranked:
        return None, (), None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, path, field, columns = ranked[0]
    return field, columns, path


def _metadata_table(
    root: Path,
    sample_keys: set[str],
    *,
    excluded_path: Path | None = None,
) -> tuple[Path, str, tuple[str, ...]] | None:
    """Find one unambiguous auxiliary table covering paired sample IDs."""
    best: tuple[float, Path, str, tuple[str, ...]] | None = None
    ambiguous = False
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in _TABLE_EXTENSIONS
            or (excluded_path is not None and path == excluded_path)
        ):
            continue
        try:
            frame = _table(path)
        except Exception:
            continue
        if frame.empty:
            continue
        ordered = [
            *[field for field in _ID_FIELDS if field in frame.columns],
            *[str(column) for column in frame.columns if str(column) not in _ID_FIELDS],
        ]
        for field in ordered:
            values = {
                Path(str(value).strip()).stem
                for value in frame[field].dropna().tolist()
                if str(value).strip()
            }
            coverage = len(values & sample_keys) / max(len(sample_keys), 1)
            features = tuple(str(column) for column in frame.columns if str(column) != field)
            if coverage < 0.90 or not features:
                continue
            candidate = (coverage, path, field, features)
            if best is None or coverage > best[0] + 1e-12:
                best = candidate
                ambiguous = False
            elif best is not None and abs(coverage - best[0]) <= 1e-12 and path != best[1]:
                ambiguous = True
            break
    if best is None or ambiguous:
        return None
    return best[1], best[2], best[3]


@lru_cache(maxsize=64)
def discover_paired_directory_layout(task_dir: Path) -> PairedDirectoryLayout | None:
    """Discover paired supervision from content and filename-stem alignment.

    Directory basenames are never assigned train/test/target semantics. The
    resolver first observes homogeneous file collections, finds near-complete
    stem pairing, distinguishes inputs from targets using decoded content, and
    aligns the remaining collection to table IDs when one is present.
    """
    task_dir = Path(task_dir)
    if (task_dir / "task_config.json").is_file():
        return None
    root = task_input_root(task_dir)
    directories = _candidate_directories(root)
    collections: list[tuple[str, Path, dict[str, Path]]] = []
    for directory in directories:
        observed = _directory_collection(directory)
        if observed is not None:
            family, mapping = observed
            collections.append((family, directory, mapping))

    pair_candidates = []
    for left_index, (left_family, left_dir, left_files) in enumerate(collections):
        for right_family, right_dir, right_files in collections[left_index + 1 :]:
            if left_family != right_family:
                continue
            if left_family != "image":
                # Other paired target encodings require a generated task-native
                # adapter; do not guess their semantics here.
                continue
            shared = set(left_files) & set(right_files)
            coverage = len(shared) / max(len(left_files), 1)
            reverse_coverage = len(shared) / max(len(right_files), 1)
            if len(shared) < 2 or coverage < 0.95 or reverse_coverage < 0.95:
                continue
            left_score = _raster_target_score(left_files.values())
            right_score = _raster_target_score(right_files.values())
            pair_candidates.append(
                (
                    min(coverage, reverse_coverage),
                    len(shared),
                    left_dir,
                    right_dir,
                    left_files,
                    right_files,
                    left_score,
                    right_score,
                )
            )
    if not pair_candidates:
        return None
    pair_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = pair_candidates[0]
    if len(pair_candidates) > 1 and pair_candidates[1][:2] == best[:2]:
        return None
    (
        _,
        _,
        left_dir,
        right_dir,
        left_files,
        right_files,
        left_score,
        right_score,
    ) = best

    # Locate an inference collection from its lack of train-stem overlap and
    # its alignment with any table column. This does not rely on split names.
    table_sets = _table_id_sets(root)
    inference_candidates = []
    paired_stems = set(left_files) | set(right_files)
    for family, directory, mapping in collections:
        if directory in {left_dir, right_dir} or family != "image":
            continue
        keys = {Path(key).name for key in mapping}
        overlap_with_pair = len(keys & {Path(key).name for key in paired_stems})
        if overlap_with_pair / max(len(keys), 1) > 0.05:
            continue
        table_coverage = max(
            (
                len(keys & values) / max(len(keys), 1)
                for _, _, values, _ in table_sets
            ),
            default=0.0,
        )
        inference_candidates.append((table_coverage, len(mapping), directory, mapping))
    inference_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    test_input_dir = inference_candidates[0][2] if inference_candidates else None
    test_mapping = inference_candidates[0][3] if inference_candidates else {}
    test_items = tuple(sorted(test_mapping.items()))
    test_stems = {Path(key).name for key in test_mapping}
    submission_id, submission_columns, submission_path = _submission_schema(
        root, test_stems
    )

    # For raster/raster pairs, decoded value cardinality normally identifies the
    # target. If tiny fixtures are indistinguishable, use vocabulary from the
    # observed output template as a task-local tie breaker.
    choose_left_target = False
    choose_right_target = False
    if left_score is not None and right_score is not None:
        choose_left_target = left_score < right_score * 0.75
        choose_right_target = right_score < left_score * 0.75
    if not choose_left_target and not choose_right_target and submission_columns:
        output_tokens = set().union(
            *(_name_tokens(column) for column in submission_columns[1:])
        )
        left_overlap = len(_name_tokens(left_dir.name) & output_tokens)
        right_overlap = len(_name_tokens(right_dir.name) & output_tokens)
        choose_left_target = left_overlap > right_overlap
        choose_right_target = right_overlap > left_overlap
    if (
        not choose_left_target
        and not choose_right_target
        and test_input_dir is not None
    ):
        # The inference collection must use the same representation as the
        # training input. Compare observed path tokens symmetrically; no token
        # is assigned a predefined meaning.
        test_tokens = set(test_input_dir.relative_to(root).parts)
        left_similarity = len(set(left_dir.relative_to(root).parts) & test_tokens)
        right_similarity = len(set(right_dir.relative_to(root).parts) & test_tokens)
        choose_left_target = right_similarity > left_similarity
        choose_right_target = left_similarity > right_similarity
    if choose_left_target == choose_right_target:
        return None
    if choose_left_target:
        target_dir, targets = left_dir, left_files
        input_dir, inputs = right_dir, right_files
    else:
        target_dir, targets = right_dir, right_files
        input_dir, inputs = left_dir, left_files

    shared_keys = sorted(set(inputs) & set(targets))
    all_stems = {Path(key).name for key in shared_keys} | test_stems
    metadata = _metadata_table(
        root, all_stems, excluded_path=submission_path
    )
    return PairedDirectoryLayout(
        input_modality="image",
        problem_type="segmentation",
        output_type="masks",
        target_type="mask_path",
        target_field="mask_path",
        train_input_dir=input_dir,
        target_dir=target_dir,
        test_input_dir=test_input_dir,
        train_pairs=tuple((key, inputs[key], targets[key]) for key in shared_keys),
        test_items=test_items,
        metadata_path=metadata[0] if metadata else None,
        metadata_id_field=metadata[1] if metadata else None,
        metadata_feature_fields=metadata[2] if metadata else (),
        submission_id_field=submission_id,
        submission_columns=submission_columns,
    )


def paired_layout_config(task_dir: Path, layout: PairedDirectoryLayout) -> dict[str, object]:
    """Translate a discovered layout into the canonical schema-v2 contract."""
    media_options: dict[str, object] = {
        "auto_paired_target": True,
        "target_source": _relative_source(task_dir, layout.target_dir),
    }
    if layout.test_input_dir is not None:
        media_options["test_source"] = _relative_source(task_dir, layout.test_input_dir)
    inputs: dict[str, object] = {
        layout.input_modality: {
            "modality": layout.input_modality,
            "role": "train",
            "source": _relative_source(task_dir, layout.train_input_dir),
            "format": "directory",
            **media_options,
        }
    }
    components = [layout.input_modality]
    if layout.metadata_path is not None:
        inputs["metadata"] = {
            "modality": "tabular",
            "role": "auxiliary",
            "source": _relative_source(task_dir, layout.metadata_path),
            "format": layout.metadata_path.suffix.lower().lstrip("."),
            "required": False,
            "auto_paired_metadata": True,
            "join_field": layout.metadata_id_field,
            "feature_fields": list(layout.metadata_feature_fields),
        }
        components.append("tabular")
    output_options: dict[str, object] = {"structured": True}
    if layout.output_type == "masks":
        output_options.update({"binary": True, "target_threshold": 0})
    if layout.submission_columns:
        output_options.update(
            {
                "submission_id_column": layout.submission_columns[0],
                "submission_prediction_columns": list(layout.submission_columns[1:]),
            }
        )
        if any("rle" in column.lower() or "encoded" in column.lower() for column in layout.submission_columns[1:]):
            output_options["submission_encoding"] = "run_length_encoding"
            description_path = Path(task_dir) / "task_description.md"
            description = (
                description_path.read_text(encoding="utf-8").lower()
                if description_path.is_file()
                else ""
            )
            if "top to bottom" in description and "left to right" in description:
                output_options["rle_flatten_order"] = "column_major"
            if "one-indexed" in description or "one indexed" in description:
                output_options["rle_index_base"] = 1
            output_options["rle_pair_format"] = "start_length"
    config: dict[str, object] = {
        "schema_version": 2,
        "modality": layout.modality,
        "component_modalities": components,
        "problem_type": layout.problem_type,
        "inputs": inputs,
        "sample_id_field": layout.submission_id_field or layout.metadata_id_field or "sample_id",
        "entity_id_field": layout.submission_id_field or layout.metadata_id_field or "sample_id",
        "target": {
            "source": _relative_source(task_dir, layout.target_dir),
            "field": layout.target_field,
            "type": layout.target_type,
        },
        "output": {"type": layout.output_type, "options": output_options},
    }
    if layout.problem_type == "segmentation":
        config.update({"metrics": ["dice"], "primary_metric": "dice"})
    return config


def build_paired_directory_bundle(
    task_dir: Path,
    task_spec: TaskSpec,
    layout: PairedDirectoryLayout | None = None,
) -> DatasetBundle:
    """Build lazy path records for a paired directory task."""
    resolved = layout or discover_paired_directory_layout(task_dir)
    if resolved is None:
        raise ValueError("paired directory layout is no longer resolvable")
    metadata_by_id: dict[str, dict[str, object]] = {}
    if resolved.metadata_path is not None and resolved.metadata_id_field is not None:
        frame = _table(resolved.metadata_path)
        for _, series in frame.iterrows():
            raw_id = series.get(resolved.metadata_id_field)
            if raw_id is None or pd.isna(raw_id):
                continue
            key = Path(str(raw_id).strip()).stem
            if not key:
                continue
            metadata_by_id[key] = {
                field: json_safe(series.get(field))
                for field in resolved.metadata_feature_fields
            }

    media_name = next(
        name
        for name, spec in task_spec.inputs.items()
        if spec.modality == resolved.input_modality
        and bool(spec.options.get("auto_paired_target"))
    )
    records: list[SampleRecord] = []
    for key, input_path, target_path in resolved.train_pairs:
        sample_id = key
        inputs: dict[str, object] = {
            media_name: canonical_input_reference(task_dir, input_path)
        }
        if "metadata" in task_spec.inputs:
            inputs["metadata"] = metadata_by_id.get(Path(sample_id).name)
        records.append(
            SampleRecord(
                sample_id=sample_id,
                entity_id=sample_id,
                group_id=sample_id,
                inputs=inputs,
                target=canonical_input_reference(task_dir, target_path),
                split="train",
            )
        )
    for key, input_path in resolved.test_items:
        sample_id = key
        inputs = {media_name: canonical_input_reference(task_dir, input_path)}
        if "metadata" in task_spec.inputs:
            inputs["metadata"] = metadata_by_id.get(Path(sample_id).name)
        records.append(
            SampleRecord(
                sample_id=sample_id,
                entity_id=sample_id,
                group_id=sample_id,
                inputs=inputs,
                target=None,
                split="test",
            )
        )
    return records_to_bundle(
        task_spec,
        records,
        metadata={
            "index_source": "paired_directories",
            "pairing_key": "relative_path_without_extension",
            "target_storage": resolved.target_type,
            "train_pair_count": len(resolved.train_pairs),
            "test_input_count": len(resolved.test_items),
            "metadata_coverage": (
                sum(Path(key).name in metadata_by_id for key, _, _ in resolved.train_pairs)
                + sum(Path(key).name in metadata_by_id for key, _ in resolved.test_items)
            ),
        },
    )
