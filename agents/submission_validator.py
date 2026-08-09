"""Format-adaptive validation for generated task deliverables.

The validator deliberately does not impose one submission format.  It derives
constraints from an optional task configuration and an observed sample output,
then falls back to safe structural checks for arbitrary files or directories.
Unknown non-empty formats are accepted with a warning so new task families can
work before a specialized validator is registered.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .task_analyzer import TaskAnalysis


ValidatorFunction = Callable[[Path, Path | None], tuple[list[str], list[str], dict[str, Any]]]

_REFERENCE_NAMES = (
    "sample_submission",
    "submission_sample",
    "sample_output",
    "output_sample",
    "example_output",
    "output_example",
    "submission_template",
    "output_template",
)
_IDENTIFIER_NAMES = {
    "id", "ids", "index", "key", "row_id", "sample_id", "record_id",
    "entity_id", "image_id", "sequence_id", "episode_id",
}
_JSON_SUFFIXES = {".json", ".geojson"}
_JSONL_SUFFIXES = {".jsonl", ".ndjson"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
_NUMPY_SUFFIXES = {".npy", ".npz"}
_ARCHIVE_SUFFIXES = {".zip"}
_TEXT_SUFFIXES = {
    ".txt", ".md", ".rst", ".fasta", ".fa", ".faa", ".fna", ".fastq",
    ".fq", ".gff", ".gff3", ".gtf", ".vcf", ".pdb", ".cif", ".mmcif",
}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _normalized_suffix(path: Path) -> str:
    return path.suffix.lower()


def _is_identifier(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return normalized in _IDENTIFIER_NAMES or normalized.endswith("_id")


def _hash_value(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8", errors="replace")).digest(), "big")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    output_path: Path | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: Mapping[str, Any] = field(default_factory=dict)

    def feedback(self) -> str:
        lines = ["Submission validation failed:"]
        lines.extend(f"- {message}" for message in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {message}" for message in self.warnings)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "output_path": str(self.output_path) if self.output_path else None,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": dict(self.checks),
        }


@dataclass(frozen=True)
class _DelimitedProfile:
    columns: tuple[str, ...]
    rows: int
    numeric_columns: tuple[int, ...]
    identifier_column: int | None
    identifier_fingerprint: tuple[int, int, int] | None
    identifier_order_hash: str | None


class SubmissionValidator:
    """Validate a task output using evidence rather than a fixed file format."""

    def __init__(
        self,
        *,
        max_files: int = 100_000,
        max_semantic_files: int = 256,
        max_parse_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.max_files = max(1, int(max_files))
        self.max_semantic_files = max(1, int(max_semantic_files))
        self.max_parse_bytes = max(1024, int(max_parse_bytes))
        self._custom: dict[str, ValidatorFunction] = {}
        self._delimited_cache: dict[tuple[str, int, int], _DelimitedProfile] = {}

    def register(self, suffix: str, validator: ValidatorFunction) -> None:
        """Register a domain validator without changing the core validator."""
        normalized = str(suffix).strip().lower()
        if not normalized.startswith(".") or len(normalized) < 2:
            raise ValueError("validator suffix must look like '.ext'")
        if not callable(validator):
            raise TypeError("validator must be callable")
        self._custom[normalized] = validator

    def validate(
        self,
        output: str | Path | None,
        analysis: TaskAnalysis,
        *,
        allowed_root: str | Path | None = None,
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        checks: dict[str, Any] = {}
        if output is None:
            return ValidationResult(False, None, ("No output artifact was provided.",))

        supplied = Path(output)
        if not supplied.exists() and not supplied.is_symlink():
            return ValidationResult(False, None, (f"Output artifact does not exist: {supplied}",))
        if supplied.is_symlink():
            return ValidationResult(False, None, ("The top-level output artifact cannot be a symbolic link.",))

        try:
            resolved = supplied.resolve(strict=True)
        except OSError as exc:
            return ValidationResult(False, None, (f"Could not resolve output artifact: {exc}",))
        if allowed_root is not None:
            allowed = Path(allowed_root).resolve(strict=True)
            if not _inside(resolved, allowed):
                return ValidationResult(
                    False,
                    None,
                    (f"Output escapes the node directory: {resolved}",),
                )

        constraints = self._constraints(analysis)
        reference = self._reference_path(analysis, constraints, warnings)
        checks["reference"] = str(reference) if reference else None
        checks["constraints"] = constraints

        if resolved.is_dir():
            files = self._directory_files(resolved, errors)
            checks["artifact_kind"] = "directory"
            checks["file_count"] = len(files)
            checks["total_bytes"] = sum(path.stat().st_size for path in files if path.exists())
            self._apply_directory_constraints(resolved, files, constraints, errors)
            if reference is not None and reference.is_dir():
                self._validate_directory_reference(
                    resolved,
                    files,
                    reference,
                    constraints,
                    errors,
                    warnings,
                    checks,
                )
            selected = self._select_primary_file(files, reference, constraints, errors, warnings)
            if selected is not None:
                checks["selected_primary_file"] = str(selected)
                self._validate_file(selected, reference, constraints, errors, warnings, checks)
                final_path = selected
            else:
                semantic_files = self._semantic_sample(files)
                if len(semantic_files) < len(files):
                    warnings.append(
                        f"Semantically inspected {len(semantic_files)} of {len(files)} output "
                        "files; every path still received structural validation."
                    )
                for path in semantic_files:
                    self._validate_file(path, None, {}, errors, warnings, checks, nested=True)
                final_path = resolved
        elif resolved.is_file():
            checks["artifact_kind"] = "file"
            checks["file_count"] = 1
            checks["total_bytes"] = resolved.stat().st_size
            self._apply_file_constraints(resolved, constraints, errors)
            self._validate_file(resolved, reference, constraints, errors, warnings, checks)
            final_path = resolved
        else:
            errors.append("Output must be a regular file or directory.")
            final_path = None

        return ValidationResult(
            not errors,
            final_path if not errors else None,
            tuple(dict.fromkeys(errors)),
            tuple(dict.fromkeys(warnings)),
            checks,
        )

    @staticmethod
    def _constraints(analysis: TaskAnalysis) -> dict[str, Any]:
        constraints: dict[str, Any] = {}
        for key in ("output", "submission", "output_contract"):
            value = analysis.task_facts.get(key)
            if isinstance(value, Mapping):
                constraints.update({str(name): item for name, item in value.items()})
        nested = constraints.get("contract")
        if isinstance(nested, Mapping):
            constraints.update({str(name): item for name, item in nested.items()})
        return constraints

    def _reference_path(
        self,
        analysis: TaskAnalysis,
        constraints: Mapping[str, Any],
        warnings: list[str],
    ) -> Path | None:
        task_root = analysis.task_dir.resolve()
        for key in ("reference", "sample", "template", "sample_submission"):
            configured = constraints.get(key)
            if isinstance(configured, str) and configured.strip():
                candidate = (task_root / configured).resolve()
                if not _inside(candidate, task_root):
                    warnings.append(f"Ignored output reference outside the task directory: {configured}")
                    continue
                if candidate.exists():
                    return candidate
                warnings.append(f"Configured output reference was not found: {configured}")

        submission = analysis.submission or {}
        relative = submission.get("path") if isinstance(submission, Mapping) else None
        if isinstance(relative, str):
            candidate = (task_root / relative).resolve()
            if _inside(candidate, task_root) and candidate.exists():
                return candidate

        candidates: list[tuple[int, Path]] = []
        for current, directory_names, file_names in os.walk(task_root, followlinks=False):
            directory_names[:] = [name for name in directory_names if name not in {".git", ".venv", "runs", "submission"}]
            for name in file_names:
                path = Path(current) / name
                stem = path.stem.lower().replace("-", "_")
                try:
                    rank = _REFERENCE_NAMES.index(stem)
                except ValueError:
                    if "sample" in stem and ("output" in stem or "submission" in stem):
                        rank = len(_REFERENCE_NAMES)
                    else:
                        continue
                candidates.append((rank, path))
            for name in directory_names:
                normalized = name.lower().replace("-", "_")
                if normalized in _REFERENCE_NAMES:
                    candidates.append((_REFERENCE_NAMES.index(normalized), Path(current) / name))
        return min(candidates, default=(0, None), key=lambda item: (item[0], str(item[1])))[1]

    def _directory_files(self, root: Path, errors: list[str]) -> list[Path]:
        files: list[Path] = []
        for path in sorted(root.rglob("*")):
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                errors.append(f"Could not inspect output path {path}: {exc}")
                continue
            if stat.S_ISLNK(mode):
                errors.append(f"Output directories cannot contain symbolic links: {path.relative_to(root)}")
            elif stat.S_ISREG(mode):
                files.append(path)
                if len(files) > self.max_files:
                    errors.append(f"Output contains more than {self.max_files} files.")
                    break
            elif not stat.S_ISDIR(mode):
                errors.append(f"Output contains a non-regular filesystem entry: {path.relative_to(root)}")
        if not files:
            errors.append("Output directory contains no files.")
        if files and all(path.stat().st_size == 0 for path in files):
            errors.append("Every file in the output directory is empty.")
        return files

    @staticmethod
    def _coerce_int(value: object, name: str, errors: list[str]) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            errors.append(f"Output constraint {name!r} must be an integer.")
            return None
        if parsed < 0:
            errors.append(f"Output constraint {name!r} cannot be negative.")
            return None
        return parsed

    def _apply_directory_constraints(
        self,
        root: Path,
        files: list[Path],
        constraints: Mapping[str, Any],
        errors: list[str],
    ) -> None:
        kind = str(constraints.get("kind", "")).strip().lower()
        if kind in {"file", "regular_file"} and not constraints.get("primary_file"):
            errors.append(
                "Task configuration requires a file output, but the generated artifact is a directory."
            )
        minimum = self._coerce_int(constraints.get("min_files"), "min_files", errors)
        maximum = self._coerce_int(constraints.get("max_files"), "max_files", errors)
        if minimum is not None and len(files) < minimum:
            errors.append(f"Output has {len(files)} files; at least {minimum} are required.")
        if maximum is not None and len(files) > maximum:
            errors.append(f"Output has {len(files)} files; at most {maximum} are allowed.")
        required = constraints.get("required_files", [])
        if isinstance(required, list):
            for name in required:
                relative = Path(str(name))
                if relative.is_absolute() or ".." in relative.parts:
                    errors.append(f"Invalid required output path in task configuration: {name}")
                elif not (root / relative).is_file():
                    errors.append(f"Required output file is missing: {relative.as_posix()}")

    def _apply_file_constraints(
        self,
        path: Path,
        constraints: Mapping[str, Any],
        errors: list[str],
    ) -> None:
        kind = str(constraints.get("kind", "")).strip().lower()
        if kind in {"directory", "folder", "bundle"}:
            errors.append(
                "Task configuration requires a directory output, but the generated artifact is a file."
            )
        if path.stat().st_size == 0:
            errors.append("Output file is empty.")
        extension = constraints.get("extension")
        if isinstance(extension, str) and extension.strip():
            expected = extension.strip().lower()
            expected = expected if expected.startswith(".") else f".{expected}"
            if path.suffix.lower() != expected:
                errors.append(f"Output extension {path.suffix!r} does not match required {expected!r}.")
        minimum = self._coerce_int(constraints.get("min_bytes"), "min_bytes", errors)
        maximum = self._coerce_int(constraints.get("max_bytes"), "max_bytes", errors)
        size = path.stat().st_size
        if minimum is not None and size < minimum:
            errors.append(f"Output has {size} bytes; at least {minimum} are required.")
        if maximum is not None and size > maximum:
            errors.append(f"Output has {size} bytes; at most {maximum} are allowed.")

    def _validate_directory_reference(
        self,
        output_root: Path,
        output_files: list[Path],
        reference_root: Path,
        constraints: Mapping[str, Any],
        errors: list[str],
        warnings: list[str],
        checks: dict[str, Any],
    ) -> None:
        reference_files: list[Path] = []
        for path in sorted(reference_root.rglob("*")):
            if path.is_symlink():
                warnings.append(
                    "Ignored symbolic link in sample output directory: "
                    f"{path.relative_to(reference_root)}"
                )
            elif path.is_file():
                reference_files.append(path)
        reference_suffixes = sorted(
            {path.suffix.lower() or "[no extension]" for path in reference_files}
        )
        output_suffixes = {
            path.suffix.lower() or "[no extension]" for path in output_files
        }
        missing_suffixes = [
            suffix for suffix in reference_suffixes if suffix not in output_suffixes
        ]
        if missing_suffixes:
            errors.append(
                "Output directory is missing file types present in the sample: "
                f"{missing_suffixes}"
            )
        if constraints.get("match_reference_paths") is True:
            expected = {
                path.relative_to(reference_root).as_posix()
                for path in reference_files
            }
            observed = {
                path.relative_to(output_root).as_posix()
                for path in output_files
            }
            if expected != observed:
                errors.append(
                    "Output directory paths do not match the configured sample layout; "
                    f"missing={sorted(expected - observed)[:20]}, "
                    f"extra={sorted(observed - expected)[:20]}."
                )
        checks["directory_reference"] = {
            "path": str(reference_root),
            "file_count": len(reference_files),
            "file_types": reference_suffixes,
        }

    def _select_primary_file(
        self,
        files: list[Path],
        reference: Path | None,
        constraints: Mapping[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> Path | None:
        if not files:
            return None
        configured_primary = constraints.get("primary_file")
        if isinstance(configured_primary, str) and configured_primary.strip():
            matches = [
                path for path in files
                if path.name == configured_primary
                or path.as_posix().endswith(f"/{configured_primary}")
            ]
            if len(matches) == 1:
                return matches[0]
            errors.append(
                f"Configured primary output file was not found uniquely: {configured_primary}"
            )
            return None
        required_extension = constraints.get("extension")
        suffix = None
        if isinstance(required_extension, str) and required_extension.strip():
            suffix = required_extension.strip().lower()
            suffix = suffix if suffix.startswith(".") else f".{suffix}"
        elif reference is not None and reference.is_file():
            suffix = reference.suffix.lower()
        if not suffix:
            return None
        candidates = [path for path in files if path.suffix.lower() == suffix]
        if len(candidates) == 1:
            return candidates[0]
        preferred = [path for path in candidates if path.name.lower() in {"submission.csv", "submission.json", "output.json"}]
        if len(preferred) == 1:
            return preferred[0]
        if len(candidates) > 1:
            errors.append(f"Output directory contains multiple possible primary {suffix} files; declare one in task_config.json.")
        elif reference is not None and reference.is_file():
            errors.append(f"Output directory contains no file matching reference extension {suffix}.")
        else:
            warnings.append(f"No output file matches configured extension {suffix}.")
        return None

    def _semantic_sample(self, files: list[Path]) -> list[Path]:
        if len(files) <= self.max_semantic_files:
            return files
        # Validation is deterministic and covers both ends of large collections.
        head = self.max_semantic_files // 2
        tail = self.max_semantic_files - head
        return files[:head] + files[-tail:]

    def _validate_file(
        self,
        path: Path,
        reference: Path | None,
        constraints: Mapping[str, Any],
        errors: list[str],
        warnings: list[str],
        checks: dict[str, Any],
        *,
        nested: bool = False,
    ) -> None:
        if path.stat().st_size == 0:
            errors.append(f"Output file is empty: {path.name}")
            return
        suffix = _normalized_suffix(path)
        reference_file = reference if reference is not None and reference.is_file() else None
        if reference_file is not None and reference_file.suffix.lower() != suffix:
            errors.append(
                f"Output type {suffix or '[no extension]'} does not match the sample output type "
                f"{reference_file.suffix.lower() or '[no extension]'}."
            )
            return

        custom = self._custom.get(suffix)
        if custom is not None:
            custom_errors, custom_warnings, custom_checks = custom(path, reference_file)
            errors.extend(custom_errors)
            warnings.extend(custom_warnings)
            checks.setdefault("custom", {})[suffix] = custom_checks
        elif suffix in {".csv", ".tsv"}:
            self._validate_delimited(path, reference_file, constraints, errors, warnings, checks)
        elif suffix in _JSON_SUFFIXES:
            self._validate_json(path, reference_file, constraints, errors, warnings, checks)
        elif suffix in _JSONL_SUFFIXES:
            self._validate_jsonl(path, reference_file, constraints, errors, warnings, checks)
        elif suffix in _NUMPY_SUFFIXES:
            self._validate_numpy(path, reference_file, errors, warnings, checks)
        elif suffix in _IMAGE_SUFFIXES:
            self._validate_image(path, errors, warnings, checks)
        elif suffix in _ARCHIVE_SUFFIXES:
            self._validate_zip(path, errors, warnings, checks)
        elif suffix in _TEXT_SUFFIXES:
            self._validate_text(path, errors)
        elif not nested:
            warnings.append(
                f"No semantic validator is registered for {suffix or 'this extensionless format'}; "
                "the non-empty regular file was accepted structurally."
            )

    @staticmethod
    def _raise_csv_limit() -> None:
        limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(limit)
                return
            except OverflowError:
                limit //= 10

    def _profile_delimited(
        self,
        path: Path,
        *,
        delimiter: str,
        numeric_hint: tuple[int, ...] = (),
        identifier_hint: str | None = None,
    ) -> _DelimitedProfile:
        stat_result = path.stat()
        cache_key = (str(path.resolve()), stat_result.st_size, stat_result.st_mtime_ns)
        if not numeric_hint and identifier_hint is None and cache_key in self._delimited_cache:
            return self._delimited_cache[cache_key]
        self._raise_csv_limit()
        with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
            reader = csv.reader(stream, delimiter=delimiter, strict=True)
            columns = tuple(str(value) for value in next(reader, []))
            if not columns or all(not value.strip() for value in columns):
                raise ValueError("delimited output has no header")
            if len(set(columns)) != len(columns):
                raise ValueError("delimited output contains duplicate column names")
            if any(index < 0 or index >= len(columns) for index in numeric_hint):
                raise ValueError(
                    "delimited output has fewer columns than the numeric sample contract"
                )
            identifier_column = None
            if identifier_hint and identifier_hint in columns:
                identifier_column = columns.index(identifier_hint)
            elif columns and _is_identifier(columns[0]):
                identifier_column = 0
            numeric_candidates = set(numeric_hint)
            observed: dict[int, list[str]] = {index: [] for index in range(len(columns))}
            rows = 0
            fingerprint_count = 0
            fingerprint_sum = 0
            fingerprint_xor = 0
            order_hash = hashlib.sha256()
            modulus = 1 << 256
            for row in reader:
                if not row:
                    continue
                rows += 1
                if len(row) != len(columns):
                    raise ValueError(
                        f"row {rows + 1} has {len(row)} values; expected {len(columns)}"
                    )
                if rows <= 50 and not numeric_hint:
                    for index, value in enumerate(row):
                        if value.strip():
                            observed[index].append(value.strip())
                if identifier_column is not None:
                    value = row[identifier_column]
                    digest = _hash_value(value)
                    fingerprint_count += 1
                    fingerprint_sum = (fingerprint_sum + digest) % modulus
                    fingerprint_xor ^= digest
                    encoded = value.encode("utf-8", errors="replace")
                    order_hash.update(len(encoded).to_bytes(8, "big"))
                    order_hash.update(encoded)
                for index in numeric_hint:
                    value = row[index].strip()
                    if not value:
                        raise ValueError(f"row {rows + 1}, column {columns[index]!r} is empty")
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError(f"row {rows + 1}, column {columns[index]!r} is not finite")
            if not numeric_hint:
                for index, values in observed.items():
                    if index == identifier_column or not values:
                        continue
                    try:
                        numbers = [float(value) for value in values]
                    except ValueError:
                        continue
                    if all(math.isfinite(value) for value in numbers):
                        numeric_candidates.add(index)
            profile = _DelimitedProfile(
                columns=columns,
                rows=rows,
                numeric_columns=tuple(sorted(numeric_candidates)),
                identifier_column=identifier_column,
                identifier_fingerprint=(fingerprint_count, fingerprint_sum, fingerprint_xor)
                if identifier_column is not None else None,
                identifier_order_hash=order_hash.hexdigest() if identifier_column is not None else None,
            )
        if not numeric_hint and identifier_hint is None:
            self._delimited_cache[cache_key] = profile
        return profile

    def _validate_delimited(
        self,
        path: Path,
        reference: Path | None,
        constraints: Mapping[str, Any],
        errors: list[str],
        warnings: list[str],
        checks: dict[str, Any],
    ) -> None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        reference_profile = None
        try:
            if reference is not None:
                reference_profile = self._profile_delimited(reference, delimiter=delimiter)
            identifier_hint = constraints.get("identifier_column")
            if not isinstance(identifier_hint, str):
                identifier_hint = None
            numeric_hint = reference_profile.numeric_columns if reference_profile else ()
            candidate = self._profile_delimited(
                path,
                delimiter=delimiter,
                numeric_hint=numeric_hint,
                identifier_hint=identifier_hint,
            )
            if reference_profile is None and candidate.numeric_columns:
                candidate = self._profile_delimited(
                    path,
                    delimiter=delimiter,
                    numeric_hint=candidate.numeric_columns,
                    identifier_hint=identifier_hint,
                )
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            errors.append(f"Invalid delimited output {path.name}: {exc}")
            return

        expected_columns = constraints.get("columns")
        if isinstance(expected_columns, list):
            configured_columns = tuple(str(value) for value in expected_columns)
        else:
            configured_columns = None
        authoritative_columns = configured_columns or (
            reference_profile.columns if reference_profile else None
        )
        if authoritative_columns is not None and candidate.columns != authoritative_columns:
            errors.append(
                f"Output columns do not match the expected order. Expected {list(authoritative_columns)}, "
                f"received {list(candidate.columns)}."
            )

        configured_rows = self._coerce_int(constraints.get("row_count"), "row_count", errors)
        reference_is_complete = bool(
            reference is not None and "submission" in reference.stem.lower()
        )
        expected_rows = configured_rows
        if expected_rows is None and reference_profile is not None and reference_is_complete:
            expected_rows = reference_profile.rows
        if expected_rows is not None and candidate.rows != expected_rows:
            errors.append(f"Output has {candidate.rows} data rows; expected {expected_rows}.")
        if candidate.rows == 0 and constraints.get("allow_empty_rows") is not True:
            errors.append("Delimited output contains no data rows.")

        if (
            reference_profile is not None
            and reference_is_complete
            and reference_profile.identifier_column is not None
            and candidate.identifier_column is not None
            and reference_profile.identifier_fingerprint != candidate.identifier_fingerprint
        ):
            errors.append("Output identifiers do not match the identifiers in the sample submission.")
        if (
            constraints.get("id_order_required") is True
            and reference_profile is not None
            and reference_profile.identifier_order_hash != candidate.identifier_order_hash
        ):
            errors.append("Output identifier order does not match the required sample order.")
        checks["delimited"] = {
            "path": str(path),
            "columns": list(candidate.columns),
            "rows": candidate.rows,
            "numeric_columns": [candidate.columns[index] for index in candidate.numeric_columns],
            "identifier_column": candidate.columns[candidate.identifier_column]
            if candidate.identifier_column is not None else None,
        }

    def _load_json(self, path: Path) -> Any:
        if path.stat().st_size > self.max_parse_bytes:
            raise OverflowError("JSON exceeds the bounded semantic parsing limit")
        return json.loads(path.read_text(encoding="utf-8"))

    def _validate_json(
        self,
        path: Path,
        reference: Path | None,
        constraints: Mapping[str, Any],
        errors: list[str],
        warnings: list[str],
        checks: dict[str, Any],
    ) -> None:
        try:
            payload = self._load_json(path)
        except OverflowError as exc:
            warnings.append(f"{path.name}: {exc}; accepted using structural checks only.")
            return
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid JSON output {path.name}: {exc}")
            return
        if payload in ({}, []) and constraints.get("allow_empty") is not True:
            errors.append("JSON output is structurally empty.")
        if reference is not None:
            try:
                sample = self._load_json(reference)
            except (OSError, UnicodeError, json.JSONDecodeError, OverflowError):
                sample = None
            if sample is not None and type(payload) is not type(sample):
                errors.append(
                    f"JSON top-level type {type(payload).__name__} does not match sample type "
                    f"{type(sample).__name__}."
                )
            if isinstance(sample, dict) and isinstance(payload, dict):
                missing = sorted(set(sample) - set(payload))
                if missing:
                    errors.append(f"JSON output is missing sample keys: {missing}")
            if (
                isinstance(sample, list)
                and isinstance(payload, list)
                and "submission" in reference.stem.lower()
                and len(payload) != len(sample)
            ):
                errors.append(f"JSON output has {len(payload)} items; expected {len(sample)}.")
        checks["json"] = {"path": str(path), "top_level_type": type(payload).__name__}

    def _validate_jsonl(
        self,
        path: Path,
        reference: Path | None,
        constraints: Mapping[str, Any],
        errors: list[str],
        warnings: list[str],
        checks: dict[str, Any],
    ) -> None:
        def inspect(target: Path) -> tuple[int, str | None]:
            count = 0
            shape = None
            with target.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    count += 1
                    current = type(value).__name__
                    shape = shape or current
                    if current != shape:
                        raise ValueError(f"line {line_number} changes top-level value type")
            return count, shape

        try:
            rows, shape = inspect(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"Invalid JSON Lines output {path.name}: {exc}")
            return
        if rows == 0 and constraints.get("allow_empty_rows") is not True:
            errors.append("JSON Lines output contains no records.")
        if reference is not None and "submission" in reference.stem.lower():
            try:
                expected_rows, expected_shape = inspect(reference)
                if rows != expected_rows:
                    errors.append(f"JSON Lines output has {rows} records; expected {expected_rows}.")
                if expected_shape and shape != expected_shape:
                    errors.append(f"JSON Lines record type {shape} does not match sample type {expected_shape}.")
            except Exception as exc:
                warnings.append(f"Could not inspect JSON Lines reference: {exc}")
        checks["jsonl"] = {"path": str(path), "rows": rows, "record_type": shape}

    def _validate_numpy(
        self,
        path: Path,
        reference: Path | None,
        errors: list[str],
        warnings: list[str],
        checks: dict[str, Any],
    ) -> None:
        if path.stat().st_size > self.max_parse_bytes:
            warnings.append(
                f"{path.name}: NumPy artifact exceeds the bounded semantic parsing limit; "
                "accepted using structural checks only."
            )
            return
        try:
            import numpy as np

            loaded = np.load(path, allow_pickle=False, mmap_mode="r" if path.suffix.lower() == ".npy" else None)
            if path.suffix.lower() == ".npz":
                names = list(loaded.files)
                shapes = {name: list(loaded[name].shape) for name in names}
                if not names or all(loaded[name].size == 0 for name in names):
                    errors.append("NPZ output contains no non-empty arrays.")
                loaded.close()
            else:
                names = None
                shapes = {"array": list(loaded.shape)}
                if loaded.size == 0:
                    errors.append("NumPy output array is empty.")
            checks["numpy"] = {"path": str(path), "arrays": names, "shapes": shapes}
        except ImportError:
            warnings.append("NumPy is unavailable; array output received structural validation only.")
        except Exception as exc:
            errors.append(f"Invalid NumPy output {path.name}: {exc}")

    @staticmethod
    def _validate_image(
        path: Path,
        errors: list[str],
        warnings: list[str],
        checks: dict[str, Any],
    ) -> None:
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
                size = image.size
                image_format = image.format
            checks.setdefault("images", []).append(
                {"path": str(path), "size": list(size), "format": image_format}
            )
        except ImportError:
            warnings.append("Pillow is unavailable; image output received structural validation only.")
        except Exception as exc:
            errors.append(f"Invalid image output {path.name}: {exc}")

    def _validate_zip(
        self,
        path: Path,
        errors: list[str],
        warnings: list[str],
        checks: dict[str, Any],
    ) -> None:
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if not names:
                    errors.append("ZIP output contains no entries.")
                for name in names:
                    member = Path(name)
                    if member.is_absolute() or ".." in member.parts:
                        errors.append(f"ZIP output contains an unsafe member path: {name}")
                        break
                uncompressed_bytes = sum(info.file_size for info in infos)
                if uncompressed_bytes <= self.max_parse_bytes:
                    corrupt = archive.testzip()
                    if corrupt:
                        errors.append(f"ZIP output contains a corrupt member: {corrupt}")
                else:
                    warnings.append(
                        f"{path.name}: archive expands beyond the bounded semantic parsing "
                        "limit; member headers were validated without full decompression."
                    )
                checks["zip"] = {
                    "path": str(path),
                    "entries": len(names),
                    "uncompressed_bytes": uncompressed_bytes,
                }
        except zipfile.BadZipFile as exc:
            errors.append(f"Invalid ZIP output {path.name}: {exc}")

    @staticmethod
    def _validate_text(path: Path, errors: list[str]) -> None:
        try:
            with path.open("r", encoding="utf-8", errors="strict") as stream:
                preview = stream.read(8192)
            if not preview.strip() and path.stat().st_size <= 8192:
                errors.append(f"Text output contains no meaningful content: {path.name}")
        except (OSError, UnicodeError) as exc:
            errors.append(f"Invalid UTF-8 text output {path.name}: {exc}")
