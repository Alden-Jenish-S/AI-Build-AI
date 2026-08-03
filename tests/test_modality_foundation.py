import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agents.data_analyzer import (
    discover_dataset_layout,
    run_dataset_analysis,
)
from agents.task_analyzer import TaskAnalyzer
from core.contracts import TaskSpec
from core.modality_registry import ModalityRegistry
from modalities import build_default_registry
from modalities.tabular import TabularAdapter


class TaskContractTests(unittest.TestCase):
    def test_legacy_tabular_config_translates_to_canonical_contract(self):
        spec = TaskSpec.from_mapping(
            "legacy_regression",
            {
                "task_type": "regression",
                "target_column": "price",
                "metric_name": "rmse",
                "metric_direction": "minimize",
                "resource_limits": {"max_ram_gb": 8},
            },
            legacy_roles={
                "train": "train.csv",
                "test": "test.csv",
            },
        )

        self.assertEqual(spec.modality, "tabular")
        self.assertEqual(spec.component_modalities, ("tabular",))
        self.assertEqual(spec.problem_type, "regression")
        self.assertEqual(spec.target.field, "price")
        self.assertEqual(spec.output.type, "continuous")
        self.assertEqual(spec.primary_metric, "rmse")
        self.assertEqual(spec.metric_direction, "minimize")
        self.assertEqual(spec.resource_limits.max_ram_gb, 8)
        self.assertEqual(spec.inputs["train"].source, "train.csv")

    def test_modality_and_problem_type_are_independent(self):
        spec = TaskSpec.from_mapping(
            "image_regression",
            {
                "schema_version": 2,
                "modality": "image",
                "problem_type": "regression",
                "inputs": {
                    "image": {
                        "source": "input/images.csv",
                        "format": "file_manifest",
                        "path_field": "path",
                    }
                },
                "target": {"field": "value"},
                "output": {"type": "continuous"},
                "metrics": [
                    {"name": "mae", "direction": "minimize"}
                ],
                "primary_metric": "mae",
            },
        )

        self.assertEqual(spec.modality, "image")
        self.assertEqual(spec.problem_type, "regression")
        self.assertEqual(spec.inputs["image"].options["path_field"], "path")
        self.assertEqual(spec.output.type, "continuous")
        self.assertEqual(
            TaskSpec.from_mapping(
                spec.task_id, spec.to_dict()
            ).to_dict(),
            spec.to_dict(),
        )

    def test_multimodal_contract_requires_multiple_inputs(self):
        with self.assertRaisesRegex(
            ValueError, "at least two named inputs"
        ):
            TaskSpec.from_mapping(
                "invalid_multimodal",
                {
                    "schema_version": 2,
                    "modality": "multimodal",
                    "component_modalities": ["image", "tabular"],
                    "problem_type": "classification",
                    "inputs": {
                        "image": {
                            "modality": "image",
                            "source": "images.csv",
                        }
                    },
                    "target": {"field": "label"},
                    "metrics": [
                        {"name": "accuracy", "direction": "maximize"}
                    ],
                },
            )

    def test_valid_multimodal_contract_preserves_component_modalities(self):
        spec = TaskSpec.from_mapping(
            "image_metadata",
            {
                "schema_version": 2,
                "modality": "multimodal",
                "component_modalities": ["image", "tabular"],
                "problem_type": "classification",
                "inputs": {
                    "image": {
                        "modality": "image",
                        "source": "entities.csv",
                        "path_field": "image_path",
                    },
                    "metadata": {
                        "modality": "tabular",
                        "source": "entities.csv",
                        "feature_fields": ["age", "location"],
                    },
                },
                "target": {"field": "label"},
                "sample_id_field": "entity_id",
                "entity_id_field": "entity_id",
                "metrics": [
                    {"name": "roc_auc", "direction": "maximize"}
                ],
            },
        )

        self.assertEqual(spec.modality, "multimodal")
        self.assertEqual(
            spec.component_modalities, ("image", "tabular")
        )
        self.assertEqual(spec.inputs["image"].modality, "image")
        self.assertEqual(spec.inputs["metadata"].modality, "tabular")


class RegistryTests(unittest.TestCase):
    def test_default_registry_contains_tabular_adapter(self):
        registry = build_default_registry()
        self.assertEqual(
            tuple(registry.names()),
            (
                "audio",
                "image",
                "multimodal",
                "tabular",
                "text",
                "video",
            ),
        )
        self.assertIsInstance(registry.get("tabular"), TabularAdapter)

    def test_duplicate_registration_and_unknown_lookup_are_explicit(self):
        registry = ModalityRegistry()
        registry.register("tabular", TabularAdapter())
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("tabular", TabularAdapter())
        with self.assertRaisesRegex(LookupError, "no adapter registered"):
            registry.get("image")


class TabularAdapterTests(unittest.TestCase):
    @staticmethod
    def _write_task(task_dir: Path) -> None:
        (task_dir / "task_description.md").write_text(
            "Predict the continuous sale price using root mean squared error.",
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "row_id": [1, 2, 3, 4],
                "feature": [10.0, 11.0, 12.0, 13.0],
                "price": [100.0, 110.0, 120.0, 130.0],
            }
        ).to_csv(task_dir / "observations_labeled.csv", index=False)
        pd.DataFrame(
            {
                "row_id": [5, 6],
                "feature": [14.0, 15.0],
            }
        ).to_csv(task_dir / "observations_holdout.csv", index=False)
        (task_dir / "task_config.json").write_text(
            json.dumps(
                {
                    "task_type": "regression",
                    "train_file": "observations_labeled.csv",
                    "test_file": "observations_holdout.csv",
                    "metric_name": "rmse",
                    "metric_direction": "minimize",
                }
            ),
            encoding="utf-8",
        )

    def test_legacy_wrapper_delegates_to_tabular_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "task"
            task_dir.mkdir()
            self._write_task(task_dir)

            legacy_layout = discover_dataset_layout(task_dir)
            adapter_layout = TabularAdapter().discover_layout(task_dir)
            self.assertEqual(legacy_layout, adapter_layout)
            self.assertIn(
                "Inferred Target Column: 'price'",
                run_dataset_analysis(task_dir),
            )

    def test_task_analyzer_persists_resolved_contract_outside_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            output_dir = root / "run" / "task_assets"
            task_dir.mkdir()
            self._write_task(task_dir)
            original_files = {
                path.name for path in task_dir.iterdir()
            }

            analysis = TaskAnalyzer().analyze(
                task_dir, output_dir=output_dir, include_index=True
            )

            self.assertEqual(analysis.task_spec.modality, "tabular")
            self.assertEqual(analysis.task_spec.problem_type, "regression")
            self.assertEqual(analysis.task_spec.target.field, "price")
            self.assertEqual(analysis.profile["sample_count"], 4)
            self.assertEqual(analysis.profile["feature_count"], 2)
            self.assertEqual(
                original_files,
                {path.name for path in task_dir.iterdir()},
            )
            resolved = json.loads(
                (output_dir / "resolved_task_spec.json").read_text()
            )
            self.assertEqual(resolved["target"]["field"], "price")
            self.assertTrue(
                (output_dir / "dataset_profile.json").is_file()
            )
            self.assertTrue(
                (output_dir / "dataset_analysis.md").is_file()
            )
            self.assertFalse(
                (output_dir / "dataset_analysis_report.txt").exists()
            )
            self.assertFalse(
                (output_dir / "dataset_index_manifest.json").exists()
            )
            self.assertEqual(
                analysis.profile["dataset_index"]["storage"],
                "direct_tabular",
            )
            diagnostics = analysis.profile["diagnostics"]
            self.assertIn("target_topology", diagnostics)
            self.assertIn("data_quality_and_drift", diagnostics)
            self.assertIn("complexity_metrics", diagnostics)
            self.assertTrue(diagnostics["synthesized_directives"])
            self.assertEqual(
                diagnostics["target_topology"]["quantiles"]["min"],
                100.0,
            )
            self.assertEqual(
                diagnostics["target_topology"]["quantiles"]["max"],
                130.0,
            )
            self.assertIsNotNone(
                diagnostics["complexity_metrics"][
                    "feature_to_sample_ratio"
                ]
            )

    def test_task_analyzer_detects_imbalance_drift_and_collinearity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            output_dir = root / "run"
            task_dir.mkdir()
            train = pd.DataFrame(
                {
                    "id": range(100),
                    "x": range(100),
                    "x_duplicate": [2 * value for value in range(100)],
                    "partly_missing": [None] * 20 + list(range(80)),
                    "label": [0] * 95 + [1] * 5,
                }
            )
            test = pd.DataFrame(
                {
                    "id": range(100, 140),
                    "x": range(300, 340),
                    "x_duplicate": [2 * value for value in range(300, 340)],
                    "partly_missing": list(range(40)),
                }
            )
            train.to_csv(task_dir / "train.csv", index=False)
            test.to_csv(task_dir / "test.csv", index=False)
            (task_dir / "task_config.json").write_text(
                json.dumps(
                    {
                        "task_type": "classification",
                        "target_column": "label",
                        "sample_id_field": "id",
                        "train_file": "train.csv",
                        "test_file": "test.csv",
                    }
                )
            )

            diagnostics = TaskAnalyzer().analyze(
                task_dir, output_dir=output_dir, include_index=True
            ).profile["diagnostics"]

            target = diagnostics["target_topology"]
            self.assertAlmostEqual(
                target["minority_to_majority_ratio"], 5 / 95
            )
            quality = diagnostics["data_quality_and_drift"]
            self.assertEqual(
                quality["missingness_percentages"]["partly_missing"]["train"],
                20.0,
            )
            self.assertIn(
                "x", quality["train_test_ks_drift"]["drifted_features"]
            )
            collinearity = diagnostics["complexity_metrics"][
                "high_collinearity"
            ]
            self.assertGreaterEqual(collinearity["pair_count"], 1)
            directives = "\n".join(diagnostics["synthesized_directives"])
            self.assertIn("class imbalance", directives)
            self.assertIn("numeric drift", directives)
            self.assertIn("collinear", directives)

    def test_task_analyzer_rejects_output_inside_read_only_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "task"
            task_dir.mkdir()
            self._write_task(task_dir)

            with self.assertRaisesRegex(ValueError, "outside"):
                TaskAnalyzer().analyze(
                    task_dir, output_dir=task_dir / "generated"
                )


if __name__ == "__main__":
    unittest.main()
