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

import pandas as pd

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
_MEDIA_DIRECTORY_NAMES = {
    "audio": frozenset({"audio", "audios", "recordings", "sounds", "waveforms"}),
    "image": frozenset({"image", "images", "imgs", "photos", "pictures"}),
    "video": frozenset({"clips", "movies", "video", "videos"}),
}
_TRAIN_NAMES = frozenset({"train", "training"})
_TEST_NAMES = frozenset({"eval", "evaluation", "inference", "predict", "test", "testing"})
_TARGET_KINDS = {
    "segmentation": {
        "names": frozenset(
            {
                "mask",
                "masks",
                "segmentation",
                "segmentations",
                "semantic_masks",
                "instance_masks",
            }
        ),
        "extensions": _MEDIA_EXTENSIONS["image"],
        "output_type": "masks",
        "target_type": "mask_path",
        "field": "mask_path",
    },
    "detection": {
        "names": frozenset(
            {"annotation", "annotations", "bbox", "bboxes", "box", "boxes"}
        ),
        "extensions": frozenset({".csv", ".json", ".jsonl", ".txt", ".xml"}),
        "output_type": "boxes",
        "target_type": "annotation_path",
        "field": "annotation_path",
    },
    "captioning": {
        "names": frozenset({"caption", "captions", "descriptions"}),
        "extensions": frozenset({".json", ".jsonl", ".md", ".txt"}),
        "output_type": "text",
        "target_type": "text_path",
        "field": "text_path",
    },
    "temporal_localization": {
        "names": frozenset(
            {"segments", "temporal_annotations", "timestamps", "time_spans"}
        ),
        "extensions": frozenset({".csv", ".json", ".jsonl", ".txt"}),
        "output_type": "boxes",
        "target_type": "temporal_annotation_path",
        "field": "temporal_annotation_path",
    },
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


def _split_kind(path: Path, root: Path) -> str | None:
    parts = {
        part.lower().replace("-", "_")
        for part in path.relative_to(root).parts[:-1]
    }
    if parts & _TRAIN_NAMES:
        return "train"
    if parts & _TEST_NAMES:
        return "test"
    return None


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


def _submission_schema(root: Path) -> tuple[str | None, tuple[str, ...]]:
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _TABLE_EXTENSIONS
        and "submission" in path.name.lower()
    )
    for path in candidates:
        try:
            frame = _table(path)
        except Exception:
            continue
        columns = tuple(str(column) for column in frame.columns)
        if len(columns) >= 2:
            return columns[0], columns
    return None, ()


def _metadata_table(
    root: Path,
    sample_keys: set[str],
) -> tuple[Path, str, tuple[str, ...]] | None:
    """Find one unambiguous auxiliary table covering paired sample IDs."""
    best: tuple[float, Path, str, tuple[str, ...]] | None = None
    ambiguous = False
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in _TABLE_EXTENSIONS
            or "submission" in path.name.lower()
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
    """Discover a common paired-file supervised task with strict safeguards."""
    task_dir = Path(task_dir)
    if (task_dir / "task_config.json").is_file():
        return None
    root = task_input_root(task_dir)
    directories = _candidate_directories(root)
    media_candidates: list[tuple[str, Path, dict[str, Path]]] = []
    target_candidates: list[tuple[str, Path, dict[str, Path]]] = []
    for directory in directories:
        name = _normalized_name(directory)
        split = _split_kind(directory, root)
        if split == "train":
            for modality, names in _MEDIA_DIRECTORY_NAMES.items():
                if name not in names:
                    continue
                mapping = _unique_file_map(directory, _MEDIA_EXTENSIONS[modality])
                if mapping:
                    media_candidates.append((modality, directory, mapping))
            for problem_type, target_kind in _TARGET_KINDS.items():
                if name not in target_kind["names"]:
                    continue
                mapping = _unique_file_map(directory, target_kind["extensions"])
                if mapping:
                    target_candidates.append((problem_type, directory, mapping))

    matches: list[
        tuple[float, int, str, str, Path, Path, dict[str, Path], dict[str, Path]]
    ] = []
    for modality, input_dir, inputs in media_candidates:
        for problem_type, target_dir, targets in target_candidates:
            shared = set(inputs) & set(targets)
            coverage = len(shared) / max(len(inputs), 1)
            reverse_coverage = len(shared) / max(len(targets), 1)
            if len(shared) < 2 or coverage < 0.95 or reverse_coverage < 0.95:
                continue
            # Masks and boxes are meaningful for images/video; captions may
            # pair with any media type.  Reject nonsensical combinations.
            if problem_type in {"segmentation", "detection"} and modality not in {"image", "video"}:
                continue
            if problem_type == "temporal_localization" and modality not in {"audio", "video"}:
                continue
            matches.append(
                (
                    min(coverage, reverse_coverage),
                    len(shared),
                    modality,
                    problem_type,
                    input_dir,
                    target_dir,
                    inputs,
                    targets,
                )
            )
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = matches[0]
    if len(matches) > 1 and matches[1][:2] == best[:2] and matches[1][4:6] != best[4:6]:
        return None
    _, _, modality, problem_type, input_dir, target_dir, inputs, targets = best
    shared_keys = sorted(set(inputs) & set(targets))

    test_candidates: list[tuple[float, Path, dict[str, Path]]] = []
    for directory in directories:
        if _split_kind(directory, root) != "test":
            continue
        if _normalized_name(directory) not in _MEDIA_DIRECTORY_NAMES[modality]:
            continue
        mapping = _unique_file_map(directory, _MEDIA_EXTENSIONS[modality])
        if mapping:
            name_match = float(_normalized_name(directory) == _normalized_name(input_dir))
            test_candidates.append((name_match, directory, mapping))
    test_input_dir = None
    test_items: tuple[tuple[str, Path], ...] = ()
    if test_candidates:
        test_candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
        _, test_input_dir, test_mapping = test_candidates[0]
        test_items = tuple(sorted(test_mapping.items()))

    all_stems = {Path(key).name for key in shared_keys}
    all_stems.update(Path(key).name for key, _ in test_items)
    metadata = _metadata_table(root, all_stems)
    submission_id, submission_columns = _submission_schema(root)
    kind = _TARGET_KINDS[problem_type]
    return PairedDirectoryLayout(
        input_modality=modality,
        problem_type=problem_type,
        output_type=str(kind["output_type"]),
        target_type=str(kind["target_type"]),
        target_field=str(kind["field"]),
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
