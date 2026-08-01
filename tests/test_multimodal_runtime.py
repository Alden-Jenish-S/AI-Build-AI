import base64
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from agents.task_analyzer import TaskAnalyzer
from agents.modality_scaffold import build_runtime_data_contract
from agents.validation_guard import inspect_generated_code
from core.runtime_contracts import FidelityProfile
from evaluation.prediction_io import (
    load_prediction_bundle,
    write_prediction_bundle,
)
from evaluation.runner import evaluate_prediction_bundle
from evaluation.splitters import create_split_plan
from memory_pool.builder.verification_runtime import (
    _build_arguments,
    _build_fixture,
)
from modalities import build_default_registry
from runtime_utils import expose_task_data, task_data_files


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
    "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _write_wav(path: Path, frequency: int = 440) -> None:
    rate = 8000
    time = np.arange(rate // 20) / rate
    signal = (0.2 * np.sin(2 * np.pi * frequency * time) * 32767).astype(
        "<i2"
    )
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(signal.tobytes())


def _base_config(modality: str, input_config: dict) -> dict:
    return {
        "schema_version": 2,
        "modality": modality,
        "problem_type": "classification",
        "inputs": {modality: input_config},
        "sample_id_field": "sample_id",
        "target": {"source": "input/metadata.csv", "field": "label"},
        "output": {"type": "class_probabilities"},
        "metrics": [{"name": "accuracy", "direction": "maximize"}],
        "primary_metric": "accuracy",
    }


class MediaAdapterTests(unittest.TestCase):
    def test_runtime_contract_exposes_flattened_components_and_numpy_arrays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "dataset_index.jsonl"
            records = [
                {
                    "sample_id": "train-1",
                    "split": "train",
                    "inputs": {
                        "image": "input/images/1.png",
                        "metadata": {"width": 3.0, "color": "green"},
                    },
                    "target": "oak",
                },
                {
                    "sample_id": "test-1",
                    "split": "test",
                    "inputs": {
                        "image": "input/images/2.png",
                        "metadata": {"width": 4.0, "color": "red"},
                    },
                },
            ]
            index_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n"
            )

            contract = build_runtime_data_contract(index_path)

            self.assertEqual(contract["feature_container"], "pandas.DataFrame")
            self.assertEqual(
                contract["components"]["metadata"]["column_prefix"],
                "metadata__",
            )
            self.assertEqual(
                contract["components"]["image"]["columns"], ["image"]
            )
            self.assertEqual(
                contract["array_variables"]["row_ids"], "numpy.ndarray"
            )

    def test_validation_guard_rejects_runtime_container_mismatches(self):
        runtime_contract = {
            "components": {
                "metadata": {
                    "storage": "flattened_columns",
                    "column_prefix": "metadata__",
                }
            },
            "array_variables": {
                "y_eval": "numpy.ndarray",
                "row_ids": "numpy.ndarray",
            },
        }
        issues = inspect_generated_code(
            """
y_train = y_eval[mask]
selected_y = y_train.iloc[rows]
selected_ids = row_ids.iloc[rows]
metadata = X_train["metadata"]
""",
            runtime_contract=runtime_contract,
        )
        self.assertEqual(len(issues), 3)
        self.assertTrue(any("NumPy array" in issue for issue in issues))
        self.assertTrue(any("metadata__" in issue for issue in issues))

    def test_multimodal_verifier_resolves_bare_and_component_path_roles(self):
        def entrypoint(train_csv_path, test_csv_path, image_dir):
            return train_csv_path, test_csv_path, image_dir

        contract = {
            "container": "paths",
            "parameter_roles": {
                "train_csv_path": "train",
                "test_csv_path": "test",
                "image_dir": "train.image",
            },
        }
        capabilities = {"input_types": ["image", "tabular"]}
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = _build_fixture(
                "multimodal",
                contract,
                capabilities,
                [],
                Path(temp_dir),
                legacy_contract=False,
            )
            kwargs, roles = _build_arguments(
                entrypoint, fixture, contract
            )

            self.assertTrue(Path(kwargs["train_csv_path"]).is_file())
            self.assertEqual(Path(kwargs["train_csv_path"]).suffix, ".csv")
            self.assertTrue(Path(kwargs["test_csv_path"]).is_file())
            self.assertTrue(Path(kwargs["image_dir"]).is_dir())
            self.assertEqual(roles["image_dir"], "train.image")

    def test_image_audio_and_video_adapters_build_lazy_indices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = build_default_registry()
            cases = []

            image_task = root / "image_task"
            (image_task / "input" / "images").mkdir(parents=True)
            image_rows = []
            for index in range(6):
                path = image_task / "input" / "images" / f"{index}.png"
                path.write_bytes(_PNG_1X1)
                image_rows.append(
                    {
                        "sample_id": f"img-{index}",
                        "image_path": f"images/{index}.png",
                        "label": index % 2,
                        "split": "train" if index < 4 else "test",
                    }
                )
            pd.DataFrame(image_rows).to_csv(
                image_task / "input" / "metadata.csv", index=False
            )
            image_config = _base_config(
                "image",
                {
                    "source": "input/images",
                    "format": "directory",
                    "manifest": "input/metadata.csv",
                    "path_field": "image_path",
                },
            )
            (image_task / "task_config.json").write_text(
                json.dumps(image_config)
            )
            cases.append(("image", image_task, 4, 2))

            audio_task = root / "audio_task"
            (audio_task / "input" / "audio").mkdir(parents=True)
            audio_rows = []
            for index in range(6):
                _write_wav(
                    audio_task / "input" / "audio" / f"{index}.wav",
                    220 + index * 10,
                )
                audio_rows.append(
                    {
                        "sample_id": f"aud-{index}",
                        "audio_path": f"audio/{index}.wav",
                        "label": index % 2,
                        "split": "train" if index < 4 else "test",
                    }
                )
            pd.DataFrame(audio_rows).to_csv(
                audio_task / "input" / "metadata.csv", index=False
            )
            audio_config = _base_config(
                "audio",
                {
                    "source": "input/metadata.csv",
                    "format": "file_manifest",
                    "path_field": "audio_path",
                },
            )
            (audio_task / "task_config.json").write_text(
                json.dumps(audio_config)
            )
            cases.append(("audio", audio_task, 4, 2))

            video_task = root / "video_task"
            (video_task / "input" / "videos").mkdir(parents=True)
            video_rows = []
            for index in range(6):
                (
                    video_task / "input" / "videos" / f"{index}.mp4"
                ).write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes([index]) * 32)
                video_rows.append(
                    {
                        "sample_id": f"vid-{index}",
                        "video_path": f"videos/{index}.mp4",
                        "label": index % 2,
                        "split": "train" if index < 4 else "test",
                    }
                )
            pd.DataFrame(video_rows).to_csv(
                video_task / "input" / "metadata.csv", index=False
            )
            video_config = _base_config(
                "video",
                {
                    "source": "input/metadata.csv",
                    "format": "file_manifest",
                    "path_field": "video_path",
                },
            )
            (video_task / "task_config.json").write_text(
                json.dumps(video_config)
            )
            cases.append(("video", video_task, 4, 2))

            for modality, task_dir, train_count, test_count in cases:
                with self.subTest(modality=modality):
                    adapter = registry.get(modality)
                    spec = adapter.discover(task_dir)
                    bundle = adapter.build_bundle(task_dir, spec)
                    profile = adapter.profile(task_dir, spec)
                    self.assertEqual(len(bundle.train_records), train_count)
                    self.assertEqual(len(bundle.test_records), test_count)
                    self.assertEqual(profile["modality"], modality)
                    self.assertTrue(bundle.dataset_fingerprint)
                    self.assertTrue(
                        str(
                            next(
                                iter(bundle.train_records[0].inputs.values())
                            )
                        ).startswith("input/")
                    )

    def test_recursive_media_inputs_are_exposed_as_file_links(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            media = task_dir / "input" / "images" / "nested"
            media.mkdir(parents=True)
            (media / "sample.png").write_bytes(_PNG_1X1)
            (task_dir / "input" / "labels.csv").write_text(
                "sample_id,image_path,label\n1,images/nested/sample.png,0\n"
            )
            run_dir = root / "run"

            discovered = task_data_files(task_dir)
            linked = expose_task_data(task_dir, run_dir)

            self.assertIn(media / "sample.png", discovered)
            target = run_dir / "input" / "images" / "nested" / "sample.png"
            self.assertIn(target, linked)
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), (media / "sample.png").resolve())


class MultimodalEvaluationTests(unittest.TestCase):
    def _task(self, root: Path) -> Path:
        task_dir = root / "multimodal_task"
        (task_dir / "input" / "images").mkdir(parents=True)
        rows = []
        for index in range(8):
            (task_dir / "input" / "images" / f"{index}.png").write_bytes(
                _PNG_1X1
            )
            rows.append(
                {
                    "entity_id": f"entity-{index}",
                    "image_path": f"images/{index}.png",
                    "age": 20 + index,
                    "device": "mobile" if index % 2 else "desktop",
                    "label": index % 2,
                    "split": "train",
                }
            )
        rows.extend(
            [
                {
                    "entity_id": "test-0",
                    "image_path": "images/0.png",
                    "age": 40,
                    "device": "mobile",
                    "label": None,
                    "split": "test",
                },
                {
                    "entity_id": "test-1",
                    "image_path": "images/1.png",
                    "age": 41,
                    "device": "desktop",
                    "label": None,
                    "split": "test",
                },
            ]
        )
        pd.DataFrame(rows).to_csv(
            task_dir / "input" / "entities.csv", index=False
        )
        config = {
            "schema_version": 2,
            "modality": "multimodal",
            "component_modalities": ["image", "tabular"],
            "problem_type": "classification",
            "inputs": {
                "image": {
                    "modality": "image",
                    "source": "input/entities.csv",
                    "path_field": "image_path",
                    "split_field": "split",
                },
                "metadata": {
                    "modality": "tabular",
                    "source": "input/entities.csv",
                    "feature_fields": ["age", "device"],
                    "split_field": "split",
                },
            },
            "sample_id_field": "entity_id",
            "entity_id_field": "entity_id",
            "target": {
                "source": "input/entities.csv",
                "field": "label",
            },
            "output": {"type": "class_probabilities"},
            "metrics": [
                {"name": "accuracy", "direction": "maximize"}
            ],
            "primary_metric": "accuracy",
        }
        (task_dir / "task_config.json").write_text(json.dumps(config))
        return task_dir

    def test_entity_aligned_bundle_split_and_prediction_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = self._task(root)
            output_dir = root / "run"
            analysis = TaskAnalyzer().analyze(
                task_dir,
                output_dir=output_dir,
                include_index=True,
            )
            bundle = analysis.bundle
            self.assertIsNotNone(bundle)
            self.assertEqual(len(bundle.train_records), 8)
            self.assertEqual(len(bundle.test_records), 2)
            self.assertEqual(
                set(bundle.train_records[0].inputs), {"image", "metadata"}
            )
            self.assertTrue((output_dir / "dataset_index.jsonl").is_file())

            fidelity = FidelityProfile(
                name="test",
                sample_fraction=1.0,
                folds=2,
                max_epochs=1,
                max_trials=1,
                early_stopping_patience=1,
                max_estimator_iterations=10,
            )
            records, split_plan = create_split_plan(
                bundle, fidelity, seed=7
            )
            for record in records:
                self.assertEqual(
                    split_plan.assignments[record.sample_id],
                    split_plan.assignments[record.entity_id],
                )
            targets = np.asarray(
                [int(record.target) for record in records]
            )
            predictions = targets.astype(float)
            prediction_bundle = write_prediction_bundle(
                output_dir,
                task_fingerprint=bundle.dataset_fingerprint,
                split_plan=split_plan,
                output_type="class_probabilities",
                sample_ids=[record.sample_id for record in records],
                predictions=predictions,
                targets=targets,
            )
            loaded, loaded_predictions, loaded_targets, fold_ids = (
                load_prediction_bundle(
                    output_dir / "predictions" / "manifest.json"
                )
            )
            self.assertEqual(
                loaded.compatibility_key,
                prediction_bundle.compatibility_key,
            )
            np.testing.assert_array_equal(
                loaded_predictions, predictions
            )
            np.testing.assert_array_equal(loaded_targets, targets)
            self.assertEqual(len(fold_ids), 8)
            result = evaluate_prediction_bundle(
                output_dir / "predictions" / "manifest.json",
                "accuracy",
            )
            self.assertEqual(result["cv_mean"], 1.0)

    def test_validation_guard_rejects_entity_resplit_and_eval_augmentation(self):
        task_spec = {
            "modality": "multimodal",
            "entity_id_field": "entity_id",
        }
        issues = inspect_generated_code(
            """
train_rows, valid_rows = train_test_split(rows)
valid_rows = random_augment(valid_rows)
""",
            task_spec=task_spec,
        )
        self.assertTrue(
            any("group/entity-sensitive" in issue for issue in issues)
        )
        self.assertTrue(
            any("training-only" in issue for issue in issues)
        )


if __name__ == "__main__":
    unittest.main()
