import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from agents.aggregator_agent import AggregatorAgent
from agents.task_analyzer import TaskAnalyzer
from core.contracts import TaskSpec
from core.runtime_contracts import SplitPlan
from evaluation.metrics import (
    infer_metric_from_description,
    metric_value,
    resolve_metric_name,
)
from evaluation.policy import select_evaluation_policy
from evaluation.prediction_io import (
    load_assignment_table,
    load_prediction_table,
    write_assignment_table,
    write_prediction_bundle,
    write_prediction_table,
)
from evaluation.runner import evaluate_prediction_bundle
from evaluation_contract import (
    evaluate_clustering_predictions,
    prepare_final_training_data,
    prepare_evaluation_data,
    prepare_holdout_evaluation_data,
    validate_evaluation_outputs,
    write_structured_predictions,
)
from evaluation.submission import (
    validate_node_submission,
    validate_submission_file,
)


class MetricResolutionTests(unittest.TestCase):
    def test_description_metric_inference_recognizes_concrete_metrics(self):
        self.assertEqual(
            infer_metric_from_description(
                "Submissions are evaluated using multi-class logarithmic loss."
            ),
            "log_loss",
        )
        self.assertEqual(
            infer_metric_from_description(
                "The score is root mean squared error (RMSE)."
            ),
            "rmse",
        )
        self.assertIsNone(
            infer_metric_from_description("Build the best possible model.")
        )
        self.assertEqual(
            infer_metric_from_description(
                "Segmentation masks are evaluated using mean average precision "
                "at intersection over union (IoU) thresholds from 0.5 to 0.95."
            ),
            "segmentation_average_precision",
        )

    def test_placeholder_metrics_resolve_from_problem_and_output(self):
        expected = {
            "classification": ("accuracy", "maximize"),
            "regression": ("rmse", "minimize"),
            "segmentation": ("dice", "maximize"),
            "detection": ("box_iou", "maximize"),
            "retrieval": ("ndcg@10", "maximize"),
            "captioning": ("token_f1", "maximize"),
            "temporal_localization": ("temporal_iou", "maximize"),
        }
        for problem_type, (metric, direction) in expected.items():
            with self.subTest(problem_type=problem_type):
                spec = TaskSpec.from_mapping(
                    problem_type,
                    {
                        "schema_version": 1,
                        "modality": "tabular",
                        "problem_type": problem_type,
                        "inputs": {"train": {"source": "train.csv"}},
                        "target": "target",
                        "metric_name": "score",
                    },
                )
                self.assertEqual(spec.primary_metric, metric)
                self.assertEqual(spec.metric_direction, direction)

    def test_schema_v2_metrics_do_not_fall_back_to_legacy_score(self):
        spec = TaskSpec.from_mapping(
            "image_task",
            {
                "schema_version": 2,
                "modality": "image",
                "problem_type": "classification",
                "inputs": {
                    "image": {
                        "source": "input/manifest.csv",
                        "path_field": "image_path",
                    }
                },
                "target": {"field": "label"},
                "metrics": [
                    {"name": "f1_macro", "direction": "maximize"}
                ],
                "primary_metric": "f1_macro",
            },
        )
        self.assertEqual(spec.primary_metric, "f1_macro")
        self.assertEqual(resolve_metric_name("score", problem_type="regression"), "rmse")


class EvaluationPolicyTests(unittest.TestCase):
    def test_binary_prediction_and_assignment_tables_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_prediction_table(
                root / "oof_predictions.npz",
                sample_ids=["a", "b"],
                targets=np.asarray([0, 1]),
                predictions=np.asarray([[0.8, 0.2], [0.1, 0.9]]),
                fold_ids=np.asarray([0, 1]),
                class_names=["no", "yes"],
            )
            write_assignment_table(
                root / "fold_assignments.npz",
                sample_ids=["a", "b"],
                fold_ids=np.asarray([0, 1]),
            )

            predictions = load_prediction_table(
                root / "oof_predictions"
            )
            assignments = load_assignment_table(
                root / "fold_assignments"
            )

            self.assertEqual(
                predictions.columns.tolist(),
                [
                    "row_id",
                    "target",
                    "prediction::no",
                    "prediction::yes",
                    "fold_id",
                ],
            )
            self.assertEqual(assignments["row_id"].tolist(), ["a", "b"])
            self.assertFalse((root / "oof_predictions.csv").exists())
            self.assertFalse((root / "fold_assignments.csv").exists())

    @staticmethod
    def _task(problem_type="classification"):
        return TaskSpec.from_mapping(
            "policy_task",
            {
                "schema_version": 1,
                "modality": "tabular",
                "problem_type": problem_type,
                "inputs": {"train": {"source": "train.csv"}},
                "target": "target",
                "metric_name": (
                    "rmse" if problem_type == "regression" else "accuracy"
                ),
            },
        )

    def test_policy_uses_oof_only_for_cv_suitable_models(self):
        task = self._task("regression")
        classical = select_evaluation_policy(
            task, {"plan": "Train a LightGBM gradient boosting model."}
        )
        deep = select_evaluation_policy(
            task, {"plan": "Train a PyTorch neural transformer."}
        )
        unknown = select_evaluation_policy(
            task, {"plan": "Use a custom expensive prediction method."}
        )
        self.assertEqual(classical.mode, "cross_validation")
        self.assertTrue(classical.requires_oof)
        self.assertEqual(deep.mode, "holdout")
        self.assertFalse(deep.requires_oof)
        self.assertEqual(unknown.mode, "holdout")

    def test_policy_honors_capabilities_and_unsupervised_objective(self):
        task = self._task("classification")
        explicit_holdout = select_evaluation_policy(
            task,
            {},
            {
                "capabilities": {
                    "evaluation_mode": "holdout",
                    "accepts_harness_fold_ids": True,
                }
            },
        )
        self.assertEqual(explicit_holdout.mode, "holdout")
        clustering = self._task("unsupervised_clustering")
        native = select_evaluation_policy(
            clustering,
            {"plan": "Random forest OOF"},
        )
        self.assertEqual(native.mode, "task_native")
        self.assertFalse(native.requires_oof)

    def test_holdout_is_scored_without_oof_or_generated_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            train_data = {
                "X": pd.DataFrame({"feature": range(100)}),
                "y": np.asarray([0, 1] * 50),
                "task_type": "classification",
            }
            (
                X_train,
                y_train,
                X_valid,
                y_valid,
                validation_ids,
                metadata,
            ) = prepare_holdout_evaluation_data(
                train_data, "full", output_dir=temp_dir
            )
            self.assertEqual(metadata["evaluation_mode"], "holdout")
            self.assertEqual(len(X_train) + len(X_valid), 100)
            self.assertEqual(len(y_train), len(X_train))
            self.assertFalse(
                (Path(temp_dir) / "oof_predictions.csv").exists()
            )
            pd.DataFrame(
                {
                    "row_id": validation_ids,
                    # Generated targets are ignored in favor of harness proof.
                    "target": 1 - y_valid,
                    "prediction": y_valid,
                }
            ).to_csv(
                Path(temp_dir) / "validation_predictions.csv",
                index=False,
            )
            validated = validate_evaluation_outputs(
                temp_dir,
                "full",
                "accuracy",
                expected_evaluation_mode="holdout",
            )
            self.assertEqual(validated["score"], 1.0)
            self.assertEqual(validated["evaluation_mode"], "holdout")
            self.assertEqual(validated["folds"], 1)
            self.assertNotIn("cv_mean", validated)

    def test_unsupervised_task_native_evaluation_does_not_write_oof(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            values = np.concatenate(
                [np.linspace(-2.0, -1.0, 30), np.linspace(1.0, 2.0, 30)]
            )
            train_data = {
                "X": pd.DataFrame({"feature": values}),
                "y": None,
                "task_type": "unsupervised_clustering",
            }
            X_eval, _, row_ids, fold_ids, metadata = (
                prepare_evaluation_data(
                    train_data,
                    "full",
                    output_dir=temp_dir,
                    evaluation_mode="task_native",
                )
            )
            labels = (X_eval["feature"].to_numpy() > 0).astype(int)
            evaluate_clustering_predictions(
                X_eval,
                labels,
                row_ids,
                fold_ids,
                fidelity=metadata["fidelity"],
                output_dir=temp_dir,
            )
            self.assertTrue(
                (Path(temp_dir) / "validation_predictions.npz").is_file()
            )
            self.assertFalse(
                (Path(temp_dir) / "oof_predictions.csv").exists()
            )
            validated = validate_evaluation_outputs(
                temp_dir,
                "full",
                "adjusted_rand_index",
                expected_evaluation_mode="task_native",
            )
            self.assertEqual(
                validated["evaluation_mode"], "task_native"
            )


class OutputAwareMetricTests(unittest.TestCase):
    def test_scalar_probability_and_structured_metrics(self):
        self.assertEqual(
            metric_value(
                "accuracy",
                np.asarray(["oak", "maple"]),
                np.asarray([[0.9, 0.1], [0.2, 0.8]]),
                class_names=("oak", "maple"),
            ),
            1.0,
        )
        self.assertAlmostEqual(
            metric_value(
                "rmse",
                np.asarray([1.0, 2.0]),
                np.asarray([1.0, 4.0]),
            ),
            2.0**0.5,
        )
        mask = np.asarray([[[0, 1], [1, 0]]])
        self.assertEqual(metric_value("dice", mask, mask), 1.0)
        self.assertEqual(
            metric_value("segmentation_average_precision", mask, mask),
            1.0,
        )
        self.assertEqual(
            metric_value(
                "segmentation_average_precision",
                mask,
                mask.astype(float)[:, None, :, :],
            ),
            1.0,
        )
        self.assertEqual(
            metric_value(
                "box_iou",
                np.asarray([[0.0, 0.0, 2.0, 2.0]]),
                np.asarray([[0.0, 0.0, 1.0, 2.0]]),
            ),
            0.5,
        )
        self.assertEqual(
            metric_value(
                "temporal_iou",
                np.asarray([[0.0, 4.0]]),
                np.asarray([[1.0, 3.0]]),
            ),
            0.5,
        )
        self.assertEqual(
            metric_value(
                "token_f1",
                np.asarray(["green leaf"]),
                np.asarray(["leaf green"]),
            ),
            1.0,
        )

    def test_ranked_item_metric_accepts_ragged_payloads(self):
        targets = np.empty(2, dtype=object)
        targets[:] = [["a"], ["b"]]
        predictions = np.empty(2, dtype=object)
        predictions[:] = [["a", "c"], ["c", "b"]]
        self.assertGreater(metric_value("ndcg@2", targets, predictions), 0.8)

    def test_typed_text_and_ragged_predictions_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            split = SplitPlan(
                assignments={"a": 0, "b": 1},
                strategy="test",
                seed=1,
            )
            targets = np.empty(2, dtype=object)
            targets[:] = [["x"], ["y"]]
            predictions = np.empty(2, dtype=object)
            predictions[:] = [["x", "z"], ["z", "y"]]
            write_prediction_bundle(
                output,
                task_fingerprint="task",
                split_plan=split,
                output_type="ranked_items",
                sample_ids=["a", "b"],
                predictions=predictions,
                targets=targets,
                metadata={"problem_type": "retrieval"},
                write_legacy_csv=False,
            )
            result = evaluate_prediction_bundle(
                output / "predictions" / "manifest.json",
                "score",
            )
            self.assertEqual(result["metric"], "ndcg@10")
            self.assertEqual(result["folds"], 2)

    def test_ranked_aggregation_preserves_best_structured_submission(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for node_id, values in (
                ("best", ["", "1 2"]),
                ("other", ["3 1", ""]),
            ):
                submission_dir = root / node_id / "submission"
                submission_dir.mkdir(parents=True)
                pd.DataFrame(
                    {"id": ["a", "b"], "rle_mask": values}
                ).to_csv(submission_dir / "submission.csv", index=False)
            destination = root / "ensemble.csv"
            selected = AggregatorAgent().aggregate_ranked_candidates(
                root,
                [
                    {"node_id": "best", "score": 0.9},
                    {"node_id": "other", "score": 0.8},
                ],
                destination,
                metric_name="segmentation_average_precision",
            )
            self.assertEqual(selected, ["best"])
            copied = pd.read_csv(destination, keep_default_na=False)
            self.assertEqual(copied["rle_mask"].tolist(), ["", "1 2"])

    def test_probability_matrix_oof_can_be_scored_and_blended(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for node_id, first_class in (
                ("one", [0.9, 0.2, 0.8, 0.1]),
                ("two", [0.7, 0.4, 0.6, 0.3]),
            ):
                node = root / node_id
                node.mkdir()
                first = np.asarray(first_class)
                pd.DataFrame(
                    {
                        "row_id": [1, 2, 3, 4],
                        "target": ["a", "b", "a", "b"],
                        "prediction::a": first,
                        "prediction::b": 1.0 - first,
                    }
                ).to_csv(node / "oof_predictions.csv", index=False)
            plan = AggregatorAgent()._oof_plan(
                root,
                ["one", "two"],
                "accuracy",
                strategy="average",
            )
            self.assertIsNotNone(plan)
            self.assertEqual(plan["class_names"], ["a", "b"])
            self.assertEqual(plan["ensemble_oof_score"], 1.0)


class LegacyInferenceAndValidationTests(unittest.TestCase):
    def test_configless_table_and_media_are_joined_as_multimodal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "leaf-classification"
            images = task_dir / "images"
            images.mkdir(parents=True)
            train = pd.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "feature": [0.1, 0.2, 0.3, 0.4],
                    "species": ["a", "b", "a", "b"],
                }
            )
            test = pd.DataFrame(
                {"id": [5, 6], "feature": [0.5, 0.6]}
            )
            train.to_csv(task_dir / "train.csv", index=False)
            test.to_csv(task_dir / "test.csv", index=False)
            pd.DataFrame(
                {"id": [5, 6], "a": [0.5, 0.5], "b": [0.5, 0.5]}
            ).to_csv(task_dir / "sample_submission.csv", index=False)
            (task_dir / "task_description.md").write_text(
                "Submissions are evaluated using multi-class logarithmic loss.\n"
            )
            for sample_id in range(1, 7):
                (images / f"{sample_id}.jpg").write_bytes(b"image")

            analysis = TaskAnalyzer().analyze(
                task_dir, include_index=True
            )
            self.assertEqual(analysis.task_spec.modality, "multimodal")
            self.assertEqual(
                analysis.task_spec.component_modalities,
                ("image", "tabular"),
            )
            self.assertEqual(analysis.task_spec.problem_type, "classification")
            self.assertEqual(analysis.task_spec.primary_metric, "log_loss")
            self.assertEqual(analysis.task_spec.metric_direction, "minimize")
            self.assertEqual(
                analysis.task_spec.output.class_names, ("a", "b")
            )
            self.assertEqual(analysis.task_spec.sample_id_field, "id")
            self.assertEqual(
                analysis.task_spec.output.options[
                    "submission_prediction_columns"
                ],
                ["a", "b"],
            )
            self.assertEqual(len(analysis.bundle.train_records), 4)
            self.assertEqual(len(analysis.bundle.test_records), 2)
            self.assertEqual(
                [record.sample_id for record in analysis.bundle.test_records],
                ["5", "6"],
            )
            self.assertIn(
                "image", analysis.bundle.train_records[0].inputs
            )
            self.assertFalse(
                any(
                    isinstance(component, dict) and "id" in component
                    for component in analysis.bundle.train_records[0].inputs.values()
                )
            )

    def test_high_cardinality_multiclass_uses_accuracy_without_warnings(self):
        classes = np.repeat(np.arange(99), 10)
        train_data = {
            "X_full": pd.DataFrame(
                {"feature": np.arange(len(classes), dtype=float)}
            ),
            "y_full": classes,
            "row_ids_full": np.arange(len(classes)),
            "task_type": "classification",
            "output_type": "class_probabilities",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _, targets, row_ids, fold_ids, metadata = (
                    prepare_evaluation_data(
                        train_data,
                        "screen",
                        output_dir=temp_dir,
                    )
                )
                pd.DataFrame(
                    {
                        "row_id": row_ids,
                        "target": targets,
                        "prediction": targets,
                        "fold_id": fold_ids,
                    }
                ).to_csv(
                    Path(temp_dir) / "oof_predictions.csv",
                    index=False,
                )
                result = validate_evaluation_outputs(
                    temp_dir, "screen", "score"
                )
            self.assertTrue(metadata["classification"])
            self.assertEqual(len(row_ids), 99 * 4)
            self.assertGreater(metadata["effective_data_fraction"], 0.25)
            self.assertEqual(result["metric"], "accuracy")
            self.assertEqual(result["cv_mean"], 1.0)
            self.assertFalse(
                any(
                    "unique classes is greater than 50%" in str(item.message)
                    for item in caught
                )
            )

    def test_explicit_metric_is_not_overridden_by_description(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            pd.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "feature": [0.1, 0.2, 0.3, 0.4],
                    "target": [0, 1, 0, 1],
                }
            ).to_csv(task_dir / "train.csv", index=False)
            (task_dir / "task_description.md").write_text(
                "The public discussion also mentions log loss.\n"
            )
            (task_dir / "task_config.json").write_text(
                """{
  "problem_type": "classification",
  "target_column": "target",
  "metric_name": "balanced_accuracy"
}"""
            )
            spec = TaskAnalyzer().resolve(task_dir)
            self.assertEqual(spec.primary_metric, "balanced_accuracy")

    def test_full_refit_manifest_and_submission_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            node_dir = root / "node"
            task_dir.mkdir()
            (node_dir / "submission").mkdir(parents=True)
            pd.DataFrame(
                {"id": [10, 11], "a": [0.5, 0.5], "b": [0.5, 0.5]}
            ).to_csv(task_dir / "sample_submission.csv", index=False)
            spec = TaskSpec.from_mapping(
                "task",
                {
                    "problem_type": "classification",
                    "inputs": {
                        "train": {"source": "train.csv"},
                        "test": {
                            "source": "test.csv",
                            "required": False,
                        },
                        "sample_submission": {
                            "source": "sample_submission.csv",
                            "required": False,
                        },
                    },
                    "target": "target",
                    "output": {
                        "type": "class_probabilities",
                        "class_names": ["a", "b"],
                    },
                    "metrics": ["log_loss"],
                },
            )
            train_data = {
                "X_full": pd.DataFrame({"feature": range(6)}),
                "y_full": np.asarray(["a", "b", "a", "b", "a", "b"]),
                "row_ids_full": np.arange(6),
                "task_type": "classification",
            }
            test_data = {
                "X_test": pd.DataFrame({"feature": [10, 11]}),
                "test_ids": np.asarray([10, 11]),
            }
            prepare_final_training_data(
                train_data, test_data, output_dir=node_dir
            )
            # Generated code may attempt to replace the public manifest; the
            # harness-owned proof remains authoritative.
            (node_dir / "final_training_manifest.json").write_text(
                '{"train_row_ids": [0, 1, 2, 3, 4, 5]}'
            )
            pd.DataFrame(
                {
                    "id": [11.0, 10.0],
                    "a": [0.1, 0.8],
                    "b": [0.1, 0.8],
                }
            ).to_csv(
                node_dir / "submission" / "submission.csv", index=False
            )
            validation = validate_node_submission(
                node_dir, task_dir=task_dir, task_spec=spec
            )
            self.assertEqual(validation["full_training_row_count"], 6)
            final = pd.read_csv(
                node_dir / "submission" / "submission.csv"
            )
            self.assertEqual(final["id"].tolist(), [10, 11])
            np.testing.assert_allclose(
                final[["a", "b"]].sum(axis=1), 1.0
            )
            canonical_manifest = json.loads(
                (node_dir / "final_training_manifest.json").read_text()
            )
            self.assertTrue(
                canonical_manifest["used_full_training_data"]
            )

    def test_scalar_submission_repairs_numeric_ids_and_column_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            node_dir = root / "node"
            task_dir.mkdir()
            (node_dir / "submission").mkdir(parents=True)
            pd.DataFrame(
                {"id": [250000, 250001], "loss": [0.0, 0.0]}
            ).to_csv(task_dir / "sample_submission.csv", index=False)
            spec = TaskSpec.from_mapping(
                "task",
                {
                    "problem_type": "regression",
                    "inputs": {
                        "train": {"source": "train.csv"},
                        "test": {"source": "test.csv"},
                        "sample_submission": {
                            "source": "sample_submission.csv"
                        },
                    },
                    "target": "loss",
                    "metrics": ["rmse"],
                },
            )
            prepare_final_training_data(
                {
                    "X_full": pd.DataFrame({"x": [1.0, 2.0]}),
                    "y_full": np.asarray([1.0, 2.0]),
                    "row_ids_full": np.asarray([0, 1]),
                },
                {
                    "X_test": pd.DataFrame({"x": [3.0, 4.0]}),
                    "test_ids": np.asarray([250000.0, 250001.0]),
                },
                output_dir=node_dir,
            )
            pd.DataFrame(
                {
                    "id": [250001.0, 250000.0],
                    "prediction": [2.0, 1.0],
                }
            ).to_csv(
                node_dir / "submission" / "submission.csv", index=False
            )
            validation = validate_node_submission(
                node_dir, task_dir=task_dir, task_spec=spec
            )
            self.assertEqual(
                validation["schema_renames"], {"prediction": "loss"}
            )
            final = pd.read_csv(
                node_dir / "submission" / "submission.csv"
            )
            self.assertEqual(final["id"].tolist(), [250000, 250001])
            self.assertEqual(final["loss"].tolist(), [1.0, 2.0])

    def test_rle_submission_accepts_empty_masks_and_rejects_bad_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task"
            task_dir.mkdir()
            template = task_dir / "sample_submission.csv"
            pd.DataFrame(
                {"id": ["a", "b"], "rle_mask": ["", ""]}
            ).to_csv(template, index=False)
            spec = TaskSpec.from_mapping(
                "segmentation",
                {
                    "schema_version": 2,
                    "modality": "image",
                    "problem_type": "segmentation",
                    "inputs": {
                        "image": {
                            "modality": "image",
                            "source": "train/images",
                        },
                        "sample_submission": {
                            "modality": "image",
                            "source": "sample_submission.csv",
                            "required": False,
                        },
                    },
                    "target": {
                        "source": "train/masks",
                        "field": "mask_path",
                        "type": "mask_path",
                    },
                    "output": {
                        "type": "masks",
                        "options": {
                            "submission_encoding": "run_length_encoding",
                            "rle_index_base": 1,
                        },
                    },
                    "metrics": ["dice"],
                },
            )
            submission = root / "submission.csv"
            pd.DataFrame(
                {"id": ["b", "a"], "rle_mask": ["", "1 2 5 1"]}
            ).to_csv(submission, index=False)
            frame, _ = validate_submission_file(
                submission, task_dir=task_dir, task_spec=spec
            )
            self.assertEqual(frame["rle_mask"].tolist(), ["1 2 5 1", ""])
            pd.DataFrame(
                {"id": ["a", "b"], "rle_mask": ["1 3 3 1", ""]}
            ).to_csv(submission, index=False)
            with self.assertRaisesRegex(ValueError, "non-overlapping"):
                validate_submission_file(
                    submission, task_dir=task_dir, task_spec=spec
                )

    def test_legacy_oof_id_alias_is_canonicalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            train_data = {
                "X_full": pd.DataFrame({"x": range(12)}),
                "y_full": np.asarray([float(value) for value in range(12)]),
                "row_ids_full": np.arange(12),
                "task_type": "regression",
                "output_type": "continuous",
            }
            _, targets, row_ids, fold_ids, _ = prepare_evaluation_data(
                train_data, "full", output_dir=temp_dir
            )
            pd.DataFrame(
                {
                    "id": row_ids.astype(float),
                    "target": targets,
                    "prediction": targets,
                    "fold_id": fold_ids,
                }
            ).to_csv(
                Path(temp_dir) / "oof_predictions.csv", index=False
            )
            result = validate_evaluation_outputs(
                temp_dir, "full", "rmse"
            )
            self.assertEqual(result["cv_mean"], 0.0)
            self.assertIn(
                "row_id",
                pd.read_csv(
                    Path(temp_dir) / "oof_predictions.csv", nrows=0
                ).columns,
            )
            without_folds = pd.read_csv(
                Path(temp_dir) / "oof_predictions.csv"
            ).drop(columns=["fold_id"])
            without_folds.to_csv(
                Path(temp_dir) / "oof_predictions.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "fold_id"):
                validate_evaluation_outputs(temp_dir, "full", "rmse")


if __name__ == "__main__":
    unittest.main()
