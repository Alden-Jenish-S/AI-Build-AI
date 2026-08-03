"""Content-first inventory and verification for task-owned files.

The inventory deliberately describes observations rather than assigning a task
to a predefined data family.  Adapters may use the observations to construct a
runtime contract, but the method tree is not allowed to start until that
contract is checked against the files that actually exist.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping


_SAMPLE_BYTES = 64 * 1024
_MAX_GROUP_EXAMPLES = 5
_MAX_TEXT_DOCUMENTS = 12
_MAX_TEXT_CHARS = 16_000


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _content_kind(path: Path) -> str:
    """Return an observed storage signature without inferring an ML modality."""
    try:
        with path.open("rb") as stream:
            head = stream.read(64)
    except OSError:
        return "unreadable"
    signatures = (
        (b"\x89PNG\r\n\x1a\n", "png"),
        (b"\xff\xd8\xff", "jpeg"),
        (b"GIF87a", "gif"),
        (b"GIF89a", "gif"),
        (b"II*\x00", "tiff"),
        (b"MM\x00*", "tiff"),
        (b"RIFF", "riff"),
        (b"fLaC", "flac"),
        (b"OggS", "ogg"),
        (b"PAR1", "parquet"),
        (b"\x93NUMPY", "npy"),
        (b"PK\x03\x04", "zip_container"),
        (b"%PDF", "pdf"),
    )
    for signature, name in signatures:
        if head.startswith(signature):
            if name == "riff" and head[8:12] == b"WAVE":
                return "wave"
            return name
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "iso_media"
    if not head:
        return "empty"
    if b"\x00" not in head:
        try:
            head.decode("utf-8")
            return "utf8_text"
        except UnicodeDecodeError:
            pass
    return "binary"


def _tabular_preview(path: Path) -> dict[str, object] | None:
    """Describe delimiter-structured text by inspecting its content."""
    if _content_kind(path) != "utf8_text":
        return None
    try:
        sample = path.read_text(encoding="utf-8", errors="strict")[:_SAMPLE_BYTES]
    except (OSError, UnicodeError):
        return None
    lines = [line for line in sample.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    # JSON records are row-oriented too, but feeding their commas to
    # csv.Sniffer invents bogus column names. Detect them from syntax first.
    try:
        json_rows = [json.loads(line) for line in lines[:6]]
    except (json.JSONDecodeError, TypeError):
        json_rows = []
    if json_rows and all(isinstance(row, Mapping) for row in json_rows):
        columns = list(dict.fromkeys(
            str(key) for row in json_rows for key in row.keys()
        ))
        return {
            "delimiter": "json_lines",
            "columns": columns,
            "sample_rows": [
                [str(row.get(column, "")) for column in columns]
                for row in json_rows[:5]
            ],
        }
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:20]))
        rows = list(csv.reader(lines[:6], dialect=dialect))
    except csv.Error:
        return None
    if not rows or len(rows[0]) < 2:
        return None
    width = len(rows[0])
    if any(len(row) != width for row in rows[1:]):
        return None
    return {
        "delimiter": dialect.delimiter,
        "columns": [str(value) for value in rows[0]],
        "sample_rows": [[str(value) for value in row] for row in rows[1:]],
    }


def _document_preview(path: Path) -> str | None:
    if _content_kind(path) != "utf8_text":
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError):
        return None
    # Delimited data belongs in table_summaries, not in the task narrative.
    if _tabular_preview(path) is not None:
        return None
    stripped = text.strip()
    return stripped[:_MAX_TEXT_CHARS] if stripped else None


def _stem_relationships(
    directory_stems: Mapping[str, set[str]],
) -> list[dict[str, object]]:
    relationships: list[dict[str, object]] = []
    directories = sorted(directory_stems)
    for index, left in enumerate(directories):
        left_stems = directory_stems[left]
        if len(left_stems) < 2:
            continue
        for right in directories[index + 1 :]:
            right_stems = directory_stems[right]
            if len(right_stems) < 2:
                continue
            shared = left_stems & right_stems
            if len(shared) < 2:
                continue
            left_coverage = len(shared) / len(left_stems)
            right_coverage = len(shared) / len(right_stems)
            if max(left_coverage, right_coverage) < 0.25:
                continue
            relationships.append(
                {
                    "left": left,
                    "right": right,
                    "shared_stems": len(shared),
                    "left_count": len(left_stems),
                    "right_count": len(right_stems),
                    "left_coverage": round(left_coverage, 6),
                    "right_coverage": round(right_coverage, 6),
                    "examples": sorted(shared)[:_MAX_GROUP_EXAMPLES],
                }
            )
    relationships.sort(
        key=lambda item: (
            min(float(item["left_coverage"]), float(item["right_coverage"])),
            int(item["shared_stems"]),
        ),
        reverse=True,
    )
    return relationships[:100]


def build_task_inventory(task_dir: Path) -> dict[str, object]:
    """Build a bounded neutral inventory of every task-owned file."""
    root = Path(task_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"task directory does not exist: {root}")
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    if not files:
        raise ValueError(f"task directory contains no files: {root}")

    grouped: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    directory_stems: dict[str, set[str]] = defaultdict(set)
    total_bytes = 0
    fingerprint_rows = []
    for path in files:
        stat = path.stat()
        relative = _relative(root, path)
        parent = Path(relative).parent.as_posix()
        suffix = path.suffix.lower() or "[none]"
        # Read only one representative per directory/suffix while grouping;
        # content signatures for the remaining representatives are added below.
        grouped[(parent, suffix, "pending")].append(path)
        directory_stems[parent].add(path.stem)
        total_bytes += int(stat.st_size)
        fingerprint_rows.append((relative, int(stat.st_size), int(stat.st_mtime_ns)))

    groups: list[dict[str, object]] = []
    for (parent, suffix, _), paths in sorted(grouped.items()):
        representatives = paths[:_MAX_GROUP_EXAMPLES]
        kinds = Counter(_content_kind(path) for path in representatives)
        groups.append(
            {
                "directory": parent,
                "suffix": suffix,
                "observed_content_kinds": dict(sorted(kinds.items())),
                "file_count": len(paths),
                "total_bytes": sum(path.stat().st_size for path in paths),
                "examples": [_relative(root, path) for path in representatives],
            }
        )

    table_summaries = []
    for path in files:
        preview = _tabular_preview(path)
        if preview is None:
            continue
        table_summaries.append(
            {
                "path": _relative(root, path),
                "size_bytes": path.stat().st_size,
                **preview,
            }
        )

    documents = []
    for path in sorted(files, key=lambda item: (item.stat().st_size, str(item))):
        if len(documents) >= _MAX_TEXT_DOCUMENTS:
            break
        preview = _document_preview(path)
        if preview is None:
            continue
        documents.append({"path": _relative(root, path), "text": preview})

    payload: dict[str, object] = {
        "schema_version": 1,
        "task_id": root.name,
        "total_files": len(files),
        "total_bytes": total_bytes,
        "top_level_entries": sorted(path.name for path in root.iterdir()),
        "file_groups": groups,
        "table_summaries": table_summaries,
        "text_documents": documents,
        "stem_relationships": _stem_relationships(directory_stems),
    }
    payload["inventory_fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _resolve_source(root: Path, source: object) -> Path | None:
    if source is None or not str(source).strip():
        return None
    raw = Path(str(source))
    candidates = (raw,) if raw.is_absolute() else (root / raw, root / "input" / raw)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved != root and root not in resolved.parents:
            continue
        if resolved.exists():
            return resolved
    return None


def _declared_paths(task_spec: object) -> set[str]:
    mapping = task_spec.to_dict() if hasattr(task_spec, "to_dict") else dict(task_spec)
    found: set[str] = set()

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and (
            key in {"source", "test_source", "target_source", "metadata_source"}
            or key.endswith("_path")
            or key.endswith("_dir")
        ):
            found.add(value)

    visit(mapping)
    return found


def verify_task_contract(
    task_dir: Path,
    task_spec: object,
    inventory: Mapping[str, object],
    *,
    bundle: object | None = None,
) -> dict[str, object]:
    """Check that a resolved contract accounts for the observed task data."""
    root = Path(task_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    mapping = task_spec.to_dict() if hasattr(task_spec, "to_dict") else dict(task_spec)
    inputs = mapping.get("inputs", {})
    if not isinstance(inputs, Mapping) or not inputs:
        errors.append("resolved task declares no inputs")
        inputs = {}
    for name, raw in inputs.items():
        raw = raw if isinstance(raw, Mapping) else {}
        source = raw.get("source")
        resolved = _resolve_source(root, source)
        if resolved is None:
            if raw.get("required", True):
                errors.append(f"required input {name!r} does not exist: {source!r}")
            else:
                warnings.append(f"optional input {name!r} was not found: {source!r}")
        elif resolved.is_dir() and not any(path.is_file() for path in resolved.rglob("*")):
            errors.append(f"required input {name!r} is an empty directory: {source!r}")

    target = mapping.get("target")
    if isinstance(target, Mapping) and target.get("source"):
        resolved_target = _resolve_source(root, target.get("source"))
        if resolved_target is None:
            errors.append(f"declared target source does not exist: {target.get('source')!r}")
        elif target.get("field") and resolved_target.is_file():
            preview = _tabular_preview(resolved_target)
            if preview is not None and str(target["field"]) not in preview["columns"]:
                errors.append(
                    f"declared target field {target['field']!r} is absent from "
                    f"{target.get('source')!r}"
                )

    declared = []
    for source in _declared_paths(task_spec):
        resolved = _resolve_source(root, source)
        if resolved is not None:
            declared.append(resolved)

    def claimed(relative_directory: str) -> bool:
        directory = (root / relative_directory).resolve()
        return any(
            path == directory
            or path in directory.parents
            or directory in path.parents
            for path in declared
        )

    groups = list(inventory.get("file_groups", []))
    meaningful = [
        group
        for group in groups
        if isinstance(group, Mapping)
        and int(group.get("file_count", 0)) >= 2
        and not str(group.get("directory", "")).startswith(".")
    ]
    unclaimed = [
        {
            "directory": group.get("directory"),
            "suffix": group.get("suffix"),
            "file_count": group.get("file_count"),
        }
        for group in meaningful
        if not claimed(str(group.get("directory", ".")))
    ]
    unclaimed_count = sum(int(group["file_count"]) for group in unclaimed)
    total_files = max(1, int(inventory.get("total_files", 0)))
    if unclaimed_count >= 10 and unclaimed_count / total_files >= 0.10:
        errors.append(
            "resolved task ignores substantial observed data groups: "
            + json.dumps(unclaimed[:10], sort_keys=True)
        )
    elif unclaimed:
        warnings.append(
            "some observed file groups are not referenced by the runtime contract: "
            + json.dumps(unclaimed[:10], sort_keys=True)
        )

    train_count = None
    test_count = None
    if bundle is not None:
        train_records = tuple(getattr(bundle, "train_records", ()) or ())
        test_records = tuple(getattr(bundle, "test_records", ()) or ())
        train_count, test_count = len(train_records), len(test_records)
        if not train_records:
            errors.append("resolved dataset bundle contains no training records")
        target_required = mapping.get("target") is not None
        if target_required and any(record.target is None for record in train_records):
            errors.append("resolved training records are missing declared targets")

    return {
        "schema_version": 1,
        "verified": not errors,
        "errors": errors,
        "warnings": warnings,
        "inventory_fingerprint": inventory.get("inventory_fingerprint"),
        "declared_sources": sorted(
            _relative(root, path) if path != root else "." for path in set(declared)
        ),
        "unclaimed_file_groups": unclaimed,
        "train_record_count": train_count,
        "test_record_count": test_count,
    }
