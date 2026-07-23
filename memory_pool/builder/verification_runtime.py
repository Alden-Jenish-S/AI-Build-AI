"""Contract-driven synthetic runtime used by ``sandbox_verifier.py``."""

from __future__ import annotations

import csv
import importlib
import inspect
import json
import os
import pickle
import struct
import sys
import wave
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_MODALITY_ALIASES = {
    "array": "array",
    "audio": "audio",
    "categorical": "tabular",
    "document": "text",
    "graph": "graph",
    "image": "image",
    "images": "image",
    "missing": "tabular",
    "multimodal": "multimodal",
    "nlp": "text",
    "numeric": "tabular",
    "sequence": "timeseries",
    "tabular": "tabular",
    "temporal": "timeseries",
    "text": "text",
    "time_series": "timeseries",
    "timeseries": "timeseries",
    "token_ids": "text",
    "tokens": "text",
    "video": "video",
    "vision": "image",
    "waveform": "audio",
}
_ALIGNING_OUTPUT_KINDS = {
    "embeddings",
    "features",
    "labels",
    "logits",
    "masks",
    "predictions",
    "probabilities",
    "scores",
    "transformed_features",
}


@dataclass
class FixtureSet:
    modality: str
    train: Any
    test: Any
    target: Any
    train_size: int
    test_size: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    feature_columns: list[str]
    train_with_target: Any
    predictions: list[np.ndarray]


def _bounded_size(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)


def _normalized_modality(
    interface: Mapping[str, Any], capabilities: Mapping[str, Any]
) -> tuple[str, dict[str, Any], bool]:
    raw_contract = interface.get("input_contract")
    input_contract = dict(raw_contract) if isinstance(raw_contract, dict) else {}
    explicit = bool(input_contract)
    candidates: list[str] = []
    for value in (
        input_contract.get("modality"),
        capabilities.get("modality"),
        capabilities.get("data_modality"),
    ):
        if value:
            candidates.append(str(value).strip().lower())
    for key in ("modalities", "input_types"):
        values = capabilities.get(key, [])
        if isinstance(values, list):
            candidates.extend(str(item).strip().lower() for item in values)
    for candidate in candidates:
        normalized = candidate.replace("-", "_").replace(" ", "_")
        if normalized in _MODALITY_ALIASES:
            return _MODALITY_ALIASES[normalized], input_contract, explicit
    return "tabular", input_contract, explicit


def _sample_shape(
    contract: Mapping[str, Any], default: tuple[int, ...]
) -> tuple[int, ...]:
    raw = contract.get("sample_shape")
    if not isinstance(raw, list) or not raw:
        return default
    shape = []
    for dimension in raw[:4]:
        try:
            parsed = int(dimension)
        except (TypeError, ValueError):
            return default
        if parsed < 1:
            return default
        shape.append(min(parsed, 64))
    return tuple(shape)


def _vary_rows(array: np.ndarray) -> np.ndarray:
    """Ensure synthetic samples are observably different along axis zero."""
    result = np.asarray(array).copy()
    if len(result) > 1:
        offsets = np.linspace(0.0, 0.5, len(result), dtype=np.float32)
        result = result + offsets.reshape((len(result),) + (1,) * (result.ndim - 1))
    return result


def _as_container(value: Any, container: str, dependencies: list[str]) -> Any:
    normalized = container.lower().replace("-", "_")
    if normalized in {"numpy", "ndarray", "array"}:
        return np.asarray(value)
    if normalized in {"list", "python_list"}:
        return list(value)
    if normalized in {"tensor", "torch", "torch_tensor"}:
        if not any(
            item.lower().startswith(("torch", "pytorch"))
            for item in dependencies
        ):
            raise ValueError(
                "input_contract requests a torch tensor without declaring torch"
            )
        import torch

        return torch.as_tensor(np.asarray(value))
    if normalized == "dict":
        return {"values": value}
    return value


def _write_png(path: Path, image: np.ndarray) -> None:
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        color_type = 0
    elif pixels.ndim == 3 and pixels.shape[2] in (3, 4):
        color_type = 2 if pixels.shape[2] == 3 else 6
    else:
        raise ValueError("PNG fixtures require HxW, HxWx3, or HxWx4 samples")
    normalized = pixels.astype(float)
    normalized -= normalized.min()
    peak = normalized.max()
    if peak > 0:
        normalized /= peak
    raw = (normalized * 255).astype(np.uint8)
    scanlines = b"".join(b"\x00" + row.tobytes() for row in raw)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    header = struct.pack(
        ">IIBBBBB",
        raw.shape[1],
        raw.shape[0],
        8,
        color_type,
        0,
        0,
        0,
    )
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def _write_sample(path: Path, modality: str, sample: Any) -> None:
    extension = path.suffix.lower()
    array = np.asarray(sample) if not isinstance(sample, str) else None
    if extension == ".npy":
        np.save(path, array)
    elif extension == ".npz":
        np.savez(path, values=array)
    elif extension in {".pkl", ".pickle"}:
        with open(path, "wb") as stream:
            pickle.dump(sample, stream)
    elif extension in {".txt", ".text"}:
        path.write_text(
            sample if isinstance(sample, str) else json.dumps(np.asarray(sample).tolist()),
            encoding="utf-8",
        )
    elif extension == ".json":
        path.write_text(json.dumps(sample if isinstance(sample, str) else array.tolist()))
    elif extension == ".jsonl":
        values = sample if isinstance(sample, list) else [sample]
        path.write_text(
            "".join(json.dumps(item) + "\n" for item in values),
            encoding="utf-8",
        )
    elif extension in {".csv", ".tsv"}:
        delimiter = "\t" if extension == ".tsv" else ","
        rows = np.atleast_2d(array)
        with open(path, "w", newline="", encoding="utf-8") as stream:
            csv.writer(stream, delimiter=delimiter).writerows(rows.tolist())
    elif extension == ".wav":
        audio = np.asarray(sample, dtype=float).reshape(-1)
        audio = np.clip(audio, -1.0, 1.0)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes((audio * 32767).astype("<i2").tobytes())
    elif extension in {".ppm", ".pgm"}:
        image = np.asarray(sample)
        if image.ndim == 3:
            image = image[..., :3]
            magic = b"P6"
        else:
            magic = b"P5"
        normalized = image.astype(float)
        normalized -= normalized.min()
        peak = normalized.max()
        if peak > 0:
            normalized /= peak
        pixels = (normalized * 255).astype(np.uint8)
        path.write_bytes(
            magic
            + f"\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode()
            + pixels.tobytes()
        )
    elif extension == ".png":
        _write_png(path, np.asarray(sample))
    else:
        raise ValueError(
            f"unsupported synthetic fixture file type {extension!r} for "
            f"modality {modality!r}; use csv, tsv, json, jsonl, txt, npy, "
            "npz, pickle, wav, ppm, pgm, or png"
        )


def _materialize_files(
    work_dir: Path,
    modality: str,
    split: str,
    values: Any,
    contract: Mapping[str, Any],
) -> Any:
    container = str(contract.get("container", "paths")).lower()
    default_extension = {
        "audio": ".wav",
        "image": ".png",
        "tabular": ".csv",
        "text": ".txt",
    }.get(modality, ".npy")
    extension = str(
        contract.get("file_extension")
        or contract.get("file_type")
        or default_extension
    ).lower()
    if not extension.startswith("."):
        extension = "." + extension
    split_dir = work_dir / f"{split}_fixtures"
    split_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        is_dataframe = isinstance(values, pd.DataFrame)
    except ImportError:
        is_dataframe = False
    if is_dataframe:
        samples = values.to_numpy().tolist()
    else:
        samples = list(values) if not isinstance(values, str) else [values]
    if container in {"file", "single_file"}:
        path = split_dir / f"{split}{extension}"
        if is_dataframe and extension in {".csv", ".tsv"}:
            values.to_csv(
                path,
                index=False,
                sep="\t" if extension == ".tsv" else ",",
            )
        elif is_dataframe and extension == ".json":
            values.to_json(path, orient="records")
        elif is_dataframe and extension == ".jsonl":
            values.to_json(path, orient="records", lines=True)
        elif is_dataframe and extension in {".pkl", ".pickle"}:
            values.to_pickle(path)
        else:
            _write_sample(path, modality, samples)
        return str(path)
    paths = []
    for index, sample in enumerate(samples):
        path = split_dir / f"{split}_{index:03d}{extension}"
        _write_sample(path, modality, sample)
        paths.append(str(path))
    if container in {"directory", "dir"}:
        return str(split_dir)
    return paths


def _build_fixture(
    modality: str,
    contract: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    dependencies: list[str],
    work_dir: Path,
    legacy_contract: bool,
) -> FixtureSet:
    rng = np.random.default_rng(42)
    train_size = _bounded_size(contract.get("train_size"), 65, 12, 128)
    test_size = _bounded_size(contract.get("test_size"), 30, 4, 64)
    input_types = {
        str(item).lower()
        for item in capabilities.get("input_types", [])
        if isinstance(item, str)
    }
    default_container = {
        "graph": "list",
        "multimodal": "dict",
        "tabular": "dataframe",
        "text": "list",
    }.get(modality, "numpy")
    container = str(
        contract.get("container", default_container)
    ).lower()

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    feature_columns: list[str] = []
    train_with_target = None

    if modality == "tabular":
        import pandas as pd

        mixed = bool({"categorical", "missing"} & input_types)
        if legacy_contract and any(
            item.lower().startswith(
                ("torch", "pytorch", "tensorflow", "keras", "jax", "flax")
            )
            for item in dependencies
        ):
            mixed = True
        numeric_columns = [
            "num1",
            "num2",
            "num3",
            "num4",
            "all_missing_num",
        ]
        categorical_columns = ["cat1", "cat2"] if mixed else []
        train_frame = pd.DataFrame(
            {
                "num1": rng.normal(size=train_size),
                "num2": rng.normal(size=train_size) * 10,
                "num3": rng.random(train_size),
                "num4": rng.normal(size=train_size),
                "all_missing_num": np.full(train_size, np.nan),
            }
        )
        test_frame = pd.DataFrame(
            {
                "num1": rng.normal(size=test_size),
                "num2": rng.normal(size=test_size) * 10,
                "num3": rng.random(test_size),
                "num4": rng.normal(size=test_size),
                "all_missing_num": rng.normal(size=test_size),
            }
        )
        if mixed:
            train_frame["cat1"] = rng.choice(["A", "B", "C"], train_size)
            train_frame["cat2"] = rng.choice(["X", "Y"], train_size)
            test_frame["cat1"] = rng.choice(["A", "B", "C"], test_size)
            test_frame["cat2"] = rng.choice(["X", "Y"], test_size)
            train_frame.loc[:5, "num3"] = np.nan
            train_frame.loc[6:10, "cat2"] = np.nan
            test_frame.loc[:3, "num3"] = np.nan
            test_frame.loc[:3, "cat2"] = np.nan
            test_frame.loc[4:7, "cat1"] = "UNSEEN"
        else:
            numeric_columns = ["num1", "num2", "num4"]
            train_frame = train_frame[numeric_columns].copy()
            test_frame = test_frame[numeric_columns].copy()
        feature_columns = list(train_frame.columns)
        if container in {"numpy", "ndarray", "array"}:
            available_numeric = [
                column
                for column in numeric_columns
                if column != "all_missing_num"
            ]
            train = train_frame[available_numeric].to_numpy(dtype=np.float32)
            test = test_frame[available_numeric].to_numpy(dtype=np.float32)
        else:
            train, test = train_frame, test_frame
    elif modality == "text":
        train = [
            f"training document {index} about topic {index % 3}"
            for index in range(train_size)
        ]
        test = [
            f"evaluation document {index} with signal {index % 5}"
            for index in range(test_size)
        ]
        if str(contract.get("representation", "")).lower() in {
            "token_ids",
            "tokens",
        }:
            width = _sample_shape(contract, (12,))[0]
            train = rng.integers(1, 100, size=(train_size, width))
            test = rng.integers(1, 100, size=(test_size, width))
        else:
            train = _as_container(train, container, dependencies)
            test = _as_container(test, container, dependencies)
    elif modality == "image":
        shape = _sample_shape(contract, (16, 16, 3))
        train = _vary_rows(rng.random((train_size, *shape), dtype=np.float32))
        test = _vary_rows(rng.random((test_size, *shape), dtype=np.float32))
        train = _as_container(train, container, dependencies)
        test = _as_container(test, container, dependencies)
    elif modality == "video":
        shape = _sample_shape(contract, (4, 12, 12, 3))
        train = _vary_rows(rng.random((train_size, *shape), dtype=np.float32))
        test = _vary_rows(rng.random((test_size, *shape), dtype=np.float32))
        train = _as_container(train, container, dependencies)
        test = _as_container(test, container, dependencies)
    elif modality == "audio":
        shape = _sample_shape(contract, (512,))
        train = _vary_rows(rng.normal(size=(train_size, *shape)).astype(np.float32))
        test = _vary_rows(rng.normal(size=(test_size, *shape)).astype(np.float32))
        train = _as_container(train, container, dependencies)
        test = _as_container(test, container, dependencies)
    elif modality == "timeseries":
        shape = _sample_shape(contract, (24, 4))
        train = _vary_rows(rng.normal(size=(train_size, *shape)).astype(np.float32))
        test = _vary_rows(rng.normal(size=(test_size, *shape)).astype(np.float32))
        train = _as_container(train, container, dependencies)
        test = _as_container(test, container, dependencies)
    elif modality == "graph":
        def graphs(count: int) -> list[dict[str, np.ndarray]]:
            result = []
            for index in range(count):
                nodes = 5 + index % 3
                source = np.arange(nodes, dtype=np.int64)
                target = np.roll(source, -1)
                result.append(
                    {
                        "x": rng.normal(size=(nodes, 4)).astype(np.float32),
                        "edge_index": np.stack([source, target]),
                    }
                )
            return result

        train, test = graphs(train_size), graphs(test_size)
    elif modality == "multimodal":
        train = {
            "text": [f"sample {index}" for index in range(train_size)],
            "image": rng.random((train_size, 8, 8, 3), dtype=np.float32),
        }
        test = {
            "text": [f"test {index}" for index in range(test_size)],
            "image": rng.random((test_size, 8, 8, 3), dtype=np.float32),
        }
    else:
        shape = _sample_shape(contract, (8,))
        train = _vary_rows(rng.normal(size=(train_size, *shape)).astype(np.float32))
        test = _vary_rows(rng.normal(size=(test_size, *shape)).astype(np.float32))
        train = _as_container(train, container, dependencies)
        test = _as_container(test, container, dependencies)

    target_types = {
        str(item).lower()
        for item in capabilities.get("target_types", [])
        if isinstance(item, str)
    }
    if "regression" in target_types:
        target: Any = np.linspace(-1.0, 1.0, train_size, dtype=np.float32)
    elif "multiclass_classification" in target_types:
        target = np.arange(train_size, dtype=np.int64) % 3
    elif "multilabel_classification" in target_types:
        target = rng.integers(0, 2, size=(train_size, 3), dtype=np.int64)
    else:
        target = np.arange(train_size, dtype=np.int64) % 2
    target_encoding = str(contract.get("target_encoding", "")).lower()
    legacy_string_target = legacy_contract and any(
        item.lower().startswith(
            ("torch", "pytorch", "tensorflow", "keras", "jax", "flax")
        )
        for item in dependencies
    ) and "binary_classification" in target_types
    if target_encoding in {"string", "labels"} or legacy_string_target:
        target = np.where(np.asarray(target) == 1, "positive", "negative")

    if modality == "tabular":
        try:
            import pandas as pd

            is_dataframe = isinstance(train, pd.DataFrame)
        except ImportError:
            is_dataframe = False
        if is_dataframe:
            train_with_target = train.copy()
            train_with_target["target"] = target

    if container in {"paths", "file_paths", "files", "directory", "dir", "file", "single_file"}:
        train = _materialize_files(work_dir, modality, "train", train, contract)
        test = _materialize_files(work_dir, modality, "test", test, contract)

    predictions = [
        np.linspace(0.05, 0.95, test_size, dtype=np.float32),
        np.linspace(0.9, 0.1, test_size, dtype=np.float32),
    ]
    return FixtureSet(
        modality=modality,
        train=train,
        test=test,
        target=target,
        train_size=train_size,
        test_size=test_size,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        feature_columns=feature_columns,
        train_with_target=train_with_target,
        predictions=predictions,
    )


def _explicit_role(
    parameter_roles: Mapping[str, Any], parameter_name: str
) -> str | None:
    declared = parameter_roles.get(parameter_name)
    if isinstance(declared, str):
        return declared.lower()
    if isinstance(declared, dict) and isinstance(declared.get("role"), str):
        return declared["role"].lower()
    return None


def _role_value(role: str, fixture: FixtureSet) -> Any:
    normalized = role.lower().replace("-", "_")
    if "." in normalized:
        base_role, component = normalized.split(".", 1)
        container = (
            fixture.train
            if base_role == "train"
            else fixture.test
            if base_role == "test"
            else None
        )
        if isinstance(container, Mapping) and component in container:
            return container[component]
        raise ValueError(
            f"parameter role {role!r} does not match a declared multimodal "
            "fixture component"
        )
    mapping = {
        "categorical_columns": fixture.categorical_columns,
        "cat_columns": fixture.categorical_columns,
        "feature_columns": fixture.feature_columns,
        "features": fixture.feature_columns,
        "numeric_columns": fixture.numeric_columns,
        "predictions": fixture.predictions,
        "prediction_ensemble": fixture.predictions,
        "target": fixture.target,
        "target_column": "target",
        "test": fixture.test,
        "test_data": fixture.test,
        "train": fixture.train,
        "train_data": fixture.train,
        "train_table_with_target": fixture.train_with_target,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported parameter role {role!r}")
    return mapping[normalized]


def _build_arguments(
    entrypoint: Any,
    fixture: FixtureSet,
    input_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    signature = inspect.signature(entrypoint)
    parameter_roles = input_contract.get("parameter_roles", {})
    if not isinstance(parameter_roles, dict):
        raise ValueError("interface.input_contract.parameter_roles must be an object")
    kwargs: dict[str, Any] = {}
    assigned_roles: dict[str, str] = {}
    positional_required_index = 0

    for parameter_name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue
        lowered = parameter_name.lower()
        role = _explicit_role(parameter_roles, parameter_name)
        if role:
            kwargs[parameter_name] = _role_value(role, fixture)
            assigned_roles[parameter_name] = role
            continue
        if lowered in {"x_train", "x_tr", "train_data"}:
            role = "train"
        elif lowered == "train_df":
            role = (
                "train_table_with_target"
                if fixture.train_with_target is not None
                else "train"
            )
        elif lowered in {"x_test", "x_te", "test_data", "test_df"}:
            role = "test"
        elif lowered in {"y", "y_train", "y_tr", "labels", "targets"}:
            role = "target"
        elif lowered in {"preds", "predictions", "preds_list"}:
            role = "predictions"
        elif lowered in {"target", "target_col", "target_column", "label_col"}:
            role = "target_column"
        elif (
            "numeric" in lowered
            or "num_col" in lowered
            or lowered in {"num_cols", "numerical_cols"}
        ):
            role = "numeric_columns"
        elif "cat" in lowered and (
            "col" in lowered or "feature" in lowered
        ):
            role = "categorical_columns"
        elif "col" in lowered or lowered in {"feature_names", "features"}:
            role = "feature_columns"
        elif "test" in lowered or "eval" in lowered or "infer" in lowered:
            role = "test"
        elif "train" in lowered:
            role = "train"
        elif any(
            token in lowered
            for token in (
                "audio",
                "document",
                "graph",
                "image",
                "input",
                "sequence",
                "series",
                "text",
                "video",
                "waveform",
            )
        ):
            role = "train" if positional_required_index == 0 else "test"
        elif "fold" in lowered or "split" in lowered:
            kwargs[parameter_name] = 3
            continue
        elif "classif" in lowered:
            kwargs[parameter_name] = True
            continue
        elif "weight" in lowered:
            if parameter.default is inspect.Parameter.empty:
                kwargs[parameter_name] = [0.5, 0.5]
            continue
        elif lowered in {"epochs", "num_epochs", "n_epochs"}:
            kwargs[parameter_name] = 2
            continue
        elif "patience" in lowered:
            kwargs[parameter_name] = 1
            continue
        elif lowered == "batch_size":
            kwargs[parameter_name] = 16
            continue
        elif lowered == "device":
            kwargs[parameter_name] = "cpu"
            continue
        elif parameter.default is not inspect.Parameter.empty:
            continue
        elif positional_required_index == 0:
            role = "train"
        elif positional_required_index == 1:
            role = "test"
        elif positional_required_index == 2:
            role = "target"
        else:
            raise TypeError(
                f"cannot synthesize required parameter {parameter_name!r}; "
                "declare interface.input_contract.parameter_roles"
            )
        kwargs[parameter_name] = _role_value(role, fixture)
        assigned_roles[parameter_name] = role
        positional_required_index += 1
    return kwargs, assigned_roles


def _safe_length(value: Any) -> int | None:
    if isinstance(value, Mapping):
        lengths = {_safe_length(item) for item in value.values()}
        lengths.discard(None)
        return next(iter(lengths)) if len(lengths) == 1 else None
    if isinstance(value, (str, bytes, Path)):
        return None
    try:
        return len(value)
    except (TypeError, AttributeError):
        return None


def _as_array(value: Any) -> np.ndarray | None:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return np.asarray(value.detach().cpu().numpy())
        except Exception:
            return None
    try:
        import pandas as pd

        if isinstance(value, (pd.DataFrame, pd.Series)):
            return np.asarray(value)
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)) and value:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if array.dtype != object or all(
            isinstance(item, (str, bytes, int, float, bool, np.generic))
            for item in value
        ):
            return array
    return None


def _inspect_outputs(value: Any) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    visited: set[int] = set()

    def inspect_value(item: Any) -> None:
        item_id = id(item)
        if item_id in visited:
            return
        visited.add(item_id)
        array = _as_array(item)
        if array is not None and array.ndim > 0:
            observation: dict[str, Any] = {
                "shape": tuple(int(part) for part in array.shape),
                "numeric": bool(np.issubdtype(array.dtype, np.number)),
            }
            if observation["numeric"]:
                if not np.isfinite(array).all():
                    raise AssertionError(
                        "entrypoint returned NaN or infinite values"
                    )
                numeric = array.astype(float)
                observation.update(
                    {
                        "minimum": float(np.min(numeric)),
                        "maximum": float(np.max(numeric)),
                        "row_variance": float(
                            np.max(np.var(numeric, axis=0))
                            if numeric.ndim > 1
                            else np.var(numeric)
                        ),
                    }
                )
            else:
                try:
                    observation["unique"] = int(len(np.unique(array)))
                except TypeError:
                    observation["unique"] = len(
                        {repr(value) for value in array.reshape(-1)}
                    )
            observations.append(observation)
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                inspect_value(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                inspect_value(nested)

    inspect_value(value)
    return observations


def _expected_alignment(
    output_contract: Mapping[str, Any],
    fixture: FixtureSet,
    kwargs: Mapping[str, Any],
    assigned_roles: Mapping[str, str],
    observations: list[dict[str, Any]],
) -> tuple[int | None, str]:
    aligned_to = str(output_contract.get("aligned_to") or "").strip()
    if aligned_to in kwargs:
        aligned_role = assigned_roles.get(aligned_to, "")
        if aligned_role in {"test", "test_data"} or aligned_role.startswith(
            "test."
        ):
            return fixture.test_size, aligned_to
        if aligned_role in {
            "train",
            "train_data",
            "train_table_with_target",
        } or aligned_role.startswith("train."):
            return fixture.train_size, aligned_to
        return _safe_length(kwargs[aligned_to]), aligned_to
    lowered = aligned_to.lower()
    if "train" in lowered:
        return fixture.train_size, aligned_to or "train"
    if any(token in lowered for token in ("test", "eval", "infer")):
        return fixture.test_size, aligned_to or "test"
    if lowered in {"predictions", "preds", "preds_list"}:
        return fixture.test_size, aligned_to
    kind = str(output_contract.get("kind") or "").lower()
    if kind in _ALIGNING_OUTPUT_KINDS:
        return fixture.test_size, aligned_to or "test"
    has_test_argument = any(
        role in {"test", "test_data"} or role.startswith("test.")
        for role in assigned_roles.values()
    )
    if not output_contract and has_test_argument and observations:
        return fixture.test_size, "X_test"
    return None, aligned_to or "none"


def _validate_output(
    result: Any,
    output_contract: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    fixture: FixtureSet,
    kwargs: Mapping[str, Any],
    assigned_roles: Mapping[str, str],
) -> None:
    if result is None:
        raise AssertionError("entrypoint returned None")
    observations = _inspect_outputs(result)
    expected_length, alignment_label = _expected_alignment(
        output_contract,
        fixture,
        kwargs,
        assigned_roles,
        observations,
    )
    print(f"Function returned type: {type(result).__name__}")
    print(f"Observed output shapes: {[item['shape'] for item in observations]}")
    if expected_length is not None:
        aligned = [
            item
            for item in observations
            if item["shape"] and item["shape"][0] == expected_length
        ]
        if not aligned:
            raise AssertionError(
                f"no returned output aligns with {alignment_label} length "
                f"{expected_length}; observed shapes="
                f"{[item['shape'] for item in observations]}"
            )
    else:
        aligned = observations

    kind = str(output_contract.get("kind") or "").lower()
    value_type = str(output_contract.get("value_type") or "").lower()
    if kind in {"predictions", "probabilities", "scores", "labels", "logits"}:
        if not aligned:
            raise AssertionError(
                "prediction-like output must be an array, tensor, Series, "
                "DataFrame, or scalar sequence"
            )
        prediction_observations = [
            item for item in aligned if len(item["shape"]) >= 1
        ]
        if not prediction_observations:
            raise AssertionError("prediction-like output has no sample axis")
        varied = []
        for item in prediction_observations:
            if item["numeric"]:
                varied.append(item.get("row_variance", 0.0) > 1e-12)
            else:
                varied.append(item.get("unique", 0) > 1)
        if not any(varied):
            raise AssertionError(
                "prediction output is constant on varied test rows/synthetic "
                "samples"
            )
        if value_type in {"probability", "probabilities"}:
            numeric_predictions = [
                item for item in prediction_observations if item["numeric"]
            ]
            if not numeric_predictions or not any(
                item["minimum"] >= -1e-12
                and item["maximum"] <= 1.0 + 1e-12
                for item in numeric_predictions
            ):
                raise AssertionError(
                    "probability predictions must be numeric and stay within [0, 1]"
                )

    target_types = capabilities.get("target_types", [])
    if (
        "multiclass_classification" in target_types
        and kind in {"predictions", "probabilities", "logits"}
        and aligned
        and all(len(item["shape"]) > 2 for item in aligned)
    ):
        raise AssertionError(
            "multiclass prediction output must expose samples on axis zero "
            "and classes on axis one"
        )


def run_request(request_path: Path) -> None:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    artifact_dir = Path(request["artifact_dir"])
    module_name = str(request["module_name"])
    entrypoint_name = str(request["entrypoint_name"])
    interface = request.get("interface", {})
    capabilities = request.get("capabilities", {})
    dependencies = [
        str(item) for item in request.get("dependencies", [])
    ]
    if not isinstance(interface, dict) or not isinstance(capabilities, dict):
        raise ValueError("verification request contracts must be objects")

    modality, input_contract, explicit_contract = _normalized_modality(
        interface, capabilities
    )
    fixture = _build_fixture(
        modality,
        input_contract,
        capabilities,
        dependencies,
        Path(request_path).parent,
        legacy_contract=not explicit_contract,
    )
    sys.path.insert(0, str(artifact_dir))
    module = importlib.import_module(module_name)
    entrypoint = getattr(module, entrypoint_name)
    kwargs, assigned_roles = _build_arguments(
        entrypoint, fixture, input_contract
    )
    result = entrypoint(**kwargs)
    output_contract = interface.get("output_contract", {})
    if not isinstance(output_contract, dict):
        raise ValueError("interface.output_contract must be an object")
    _validate_output(
        result,
        output_contract,
        capabilities,
        fixture,
        kwargs,
        assigned_roles,
    )
    print(f"VERIFICATION_MODALITY={modality}")
    print(
        "VERIFICATION_CONTRACT="
        + ("declared" if explicit_contract else "legacy_inferred")
    )
    print("SUCCESS")


def main() -> None:
    os.environ["AIBUILDAI_VERIFY"] = "1"
    os.environ["AIBUILDAI_ACCELERATOR"] = "cpu"
    os.environ["AIBUILDAI_MAX_EPOCHS"] = "2"
    os.environ["AIBUILDAI_EARLY_STOPPING_PATIENCE"] = "1"
    if len(sys.argv) != 2:
        raise SystemExit("usage: verification_runtime.py <request.json>")
    try:
        run_request(Path(sys.argv[1]))
    except Exception:
        import traceback

        print("FAILURE:")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
