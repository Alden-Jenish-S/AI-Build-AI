from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agents.aggregator_agent import AggregatorAgent
from agents.implementation_agent import ImplementationAgent
from agents.manager_agent import ManagerAgent
from agents.submission_validator import SubmissionValidator
from agents.task_analyzer import TaskAnalysis, TaskAnalyzer
from runtime_utils import SupervisedProcessResult
from tree.node import NodeState


class SubmissionValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.task = self.root / "task"
        self.node = self.root / "node"
        self.task.mkdir()
        self.node.mkdir()
        self.validator = SubmissionValidator(max_semantic_files=4)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def analysis(
        self,
        *,
        submission: dict | None = None,
        task_facts: dict | None = None,
    ) -> TaskAnalysis:
        return TaskAnalysis(
            task_name="task",
            task_dir=self.task,
            goal="produce the requested output",
            target="task result",
            expected_output="output under submission/",
            submission=submission,
            task_facts=task_facts or {},
        )

    def write_sample_csv(self, text: str = "id,prediction\na,0.5\nb,0.5\n") -> dict:
        sample = self.task / "sample_submission.csv"
        sample.write_text(text, encoding="utf-8")
        return {"path": sample.name, "kind": "table", "extension": ".csv"}

    def test_csv_sample_enforces_columns_rows_and_identifier_set(self) -> None:
        submission = self.write_sample_csv()
        output = self.node / "answer.csv"
        # Identifier order is not assumed unless a task configuration requires it.
        output.write_text("id,prediction\nb,0.2\na,0.8\n", encoding="utf-8")
        result = self.validator.validate(
            output, self.analysis(submission=submission), allowed_root=self.node
        )
        self.assertTrue(result.valid, result.feedback())

        output.write_text("id,prediction\na,0.2\nc,0.8\n", encoding="utf-8")
        result = self.validator.validate(
            output, self.analysis(submission=submission), allowed_root=self.node
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("identifiers" in error for error in result.errors))

    def test_csv_sample_rejects_wrong_schema_and_nonfinite_predictions(self) -> None:
        submission = self.write_sample_csv()
        output = self.node / "answer.csv"
        output.write_text("id,score\na,0.2\nb,0.8\n", encoding="utf-8")
        result = self.validator.validate(
            output, self.analysis(submission=submission), allowed_root=self.node
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("columns" in error for error in result.errors))

        output.write_text("id,prediction\na,nan\nb,0.8\n", encoding="utf-8")
        result = self.validator.validate(
            output, self.analysis(submission=submission), allowed_root=self.node
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("not finite" in error for error in result.errors))

    def test_text_columns_can_contain_empty_task_native_values(self) -> None:
        submission = self.write_sample_csv("id,rle_mask\na,1 2\nb,\n")
        output = self.node / "answer.csv"
        output.write_text("id,rle_mask\na,1 3\nb,\n", encoding="utf-8")
        result = self.validator.validate(
            output, self.analysis(submission=submission), allowed_root=self.node
        )
        self.assertTrue(result.valid, result.feedback())

    def test_sample_free_unknown_file_is_accepted_structurally(self) -> None:
        output = self.node / "policy.ckpt"
        output.write_bytes(b"opaque-domain-specific-artifact")
        result = self.validator.validate(output, self.analysis(), allowed_root=self.node)
        self.assertTrue(result.valid, result.feedback())
        self.assertTrue(any("No semantic validator" in warning for warning in result.warnings))

    def test_sample_free_directory_is_accepted_and_empty_directory_is_rejected(self) -> None:
        output = self.node / "bundle"
        output.mkdir()
        (output / "weights.bin").write_bytes(b"weights")
        result = self.validator.validate(output, self.analysis(), allowed_root=self.node)
        self.assertTrue(result.valid, result.feedback())

        (output / "weights.bin").unlink()
        result = self.validator.validate(output, self.analysis(), allowed_root=self.node)
        self.assertFalse(result.valid)
        self.assertTrue(any("contains no files" in error for error in result.errors))

    def test_known_format_is_semantically_checked_without_a_sample(self) -> None:
        output = self.node / "records.json"
        output.write_text("{not-json", encoding="utf-8")
        result = self.validator.validate(output, self.analysis(), allowed_root=self.node)
        self.assertFalse(result.valid)
        self.assertTrue(any("Invalid JSON" in error for error in result.errors))

        output.write_text(json.dumps({"answer": [1, 2, 3]}), encoding="utf-8")
        result = self.validator.validate(output, self.analysis(), allowed_root=self.node)
        self.assertTrue(result.valid, result.feedback())

    def test_task_config_can_define_constraints_without_a_sample(self) -> None:
        output = self.node / "records.jsonl"
        output.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
        analysis = self.analysis(
            task_facts={
                "output": {
                    "extension": ".jsonl",
                    "min_bytes": 5,
                    "max_bytes": 1000,
                }
            }
        )
        result = self.validator.validate(output, analysis, allowed_root=self.node)
        self.assertTrue(result.valid, result.feedback())

    def test_output_cannot_escape_node_or_use_symlinks(self) -> None:
        outside = self.root / "outside.bin"
        outside.write_bytes(b"outside")
        result = self.validator.validate(outside, self.analysis(), allowed_root=self.node)
        self.assertFalse(result.valid)
        self.assertTrue(any("escapes" in error for error in result.errors))

        link = self.node / "linked.bin"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("symbolic links are unavailable")
        result = self.validator.validate(link, self.analysis(), allowed_root=self.node)
        self.assertFalse(result.valid)
        self.assertTrue(any("symbolic link" in error for error in result.errors))

    def test_directory_can_select_one_primary_file_from_a_sample(self) -> None:
        submission = self.write_sample_csv()
        output = self.node / "submission"
        output.mkdir()
        candidate = output / "predictions.csv"
        candidate.write_text("id,prediction\na,0.2\nb,0.8\n", encoding="utf-8")
        (output / "model.bin").write_bytes(b"model")
        result = self.validator.validate(
            output, self.analysis(submission=submission), allowed_root=self.node
        )
        self.assertTrue(result.valid, result.feedback())
        self.assertEqual(candidate.resolve(), result.output_path)

    def test_sample_directory_constrains_file_types_without_fixing_names(self) -> None:
        reference = self.task / "sample_output"
        reference.mkdir()
        (reference / "example.mask").write_bytes(b"mask")
        output = self.node / "submission"
        output.mkdir()
        (output / "prediction.mask").write_bytes(b"prediction")
        result = self.validator.validate(output, self.analysis(), allowed_root=self.node)
        self.assertTrue(result.valid, result.feedback())
        self.assertEqual([".mask"], result.checks["directory_reference"]["file_types"])

        (output / "prediction.mask").unlink()
        (output / "prediction.txt").write_text("prediction", encoding="utf-8")
        result = self.validator.validate(output, self.analysis(), allowed_root=self.node)
        self.assertFalse(result.valid)
        self.assertTrue(any("missing file types" in error for error in result.errors))

    def test_custom_validator_supports_unseen_domain_formats(self) -> None:
        output = self.node / "structure.xyzq"
        output.write_bytes(b"domain payload")

        def custom(path: Path, reference: Path | None):
            self.assertIsNone(reference)
            return [], ["domain validator ran"], {"bytes": path.stat().st_size}

        self.validator.register(".xyzq", custom)
        result = self.validator.validate(output, self.analysis(), allowed_root=self.node)
        self.assertTrue(result.valid, result.feedback())
        self.assertIn(".xyzq", result.checks["custom"])

    def test_invalid_generated_output_enters_repair_loop(self) -> None:
        submission = self.write_sample_csv()
        analysis = self.analysis(submission=submission)
        executions = 0

        def execute(_command, *, cwd: Path, **_kwargs):
            nonlocal executions
            executions += 1
            output_dir = Path(cwd) / "submission"
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / "submission.csv"
            output.write_text(
                "id,wrong\na,0.2\nb,0.8\n"
                if executions == 1
                else "id,prediction\na,0.2\nb,0.8\n",
                encoding="utf-8",
            )
            (Path(cwd) / "result.json").write_text(
                json.dumps({"score": 0.8, "output": "submission/submission.csv"}),
                encoding="utf-8",
            )
            return SupervisedProcessResult(
                args=("python", "algorithm.py"),
                returncode=0,
                stdout="score=0.8\n",
                stderr="",
                elapsed_seconds=0.01,
                stalled=False,
                hard_limit_reached=False,
                termination_reason=None,
                progress_events=1,
                last_progress_source="process_output",
                last_progress_age_seconds=0.0,
            )

        generated_code = "\n".join(
            ["from pathlib import Path", "# generated implementation"]
            + [f"VALUE_{index} = {index}" for index in range(20)]
        )
        agent = ImplementationAgent(submission_validator=self.validator)
        with (
            patch("agents.implementation_agent.call_llm", return_value=generated_code),
            patch("agents.implementation_agent.run_supervised_process", side_effect=execute),
        ):
            result = agent.run(
                self.node,
                "write predictions",
                self.task,
                analysis,
                max_debug_attempts=2,
                stall_seconds=1,
            )
        self.assertEqual("completed", result["status"])
        self.assertEqual(2, result["attempts"])
        self.assertTrue(result["submission_validation"]["valid"])
        self.assertIn(
            "SUBMISSION VALIDATION",
            (self.node / "attempt_1.log").read_text(encoding="utf-8"),
        )

    def test_verified_numeric_candidate_can_defer_submission(self) -> None:
        submission = self.write_sample_csv(
            "id,class_0,class_1\na,0.5,0.5\nb,0.5,0.5\n"
        )
        analysis = self.analysis(submission=submission)
        analysis.metric = "log loss"
        analysis.direction = "minimize"

        def execute(_command, *, cwd: Path, **_kwargs):
            cwd = Path(cwd)
            (cwd / "submission").mkdir()
            (cwd / "submission" / "submission.csv").write_text(
                "id,class_0,class_1\na,0.7,0.3\nb,0.4,0.6\n",
                encoding="utf-8",
            )
            np.savez_compressed(
                cwd / "oof_predictions.npz",
                oof_pred=np.asarray([[0.8, 0.2], [0.2, 0.8]]),
                oof_target=np.asarray([0, 1]),
                oof_index=np.asarray([0, 1]),
                oof_fold=np.asarray([0, 1]),
                test_pred=np.asarray([[0.7, 0.3], [0.4, 0.6]]),
                test_index=np.asarray(["a", "b"]),
            )
            (cwd / "result.json").write_text(
                json.dumps(
                    {
                        "score": 0.9,
                        "metric": "log loss",
                        "direction": "minimize",
                        "output": "submission/submission.csv",
                    }
                ),
                encoding="utf-8",
            )
            return SupervisedProcessResult(
                args=("python", "algorithm.py"),
                returncode=0,
                stdout="score=0.9\n",
                stderr="",
                elapsed_seconds=0.01,
                stalled=False,
                hard_limit_reached=False,
                termination_reason=None,
                progress_events=1,
                last_progress_source="process_output",
                last_progress_age_seconds=0.0,
            )

        generated_code = "\n".join(
            ["from pathlib import Path", "# generated implementation"]
            + [f"VALUE_{index} = {index}" for index in range(20)]
        )
        agent = ImplementationAgent(submission_validator=self.validator)
        with (
            patch("agents.implementation_agent.call_llm", return_value=generated_code),
            patch("agents.implementation_agent.run_supervised_process", side_effect=execute),
        ):
            result = agent.run(
                self.node,
                "write prediction evidence",
                self.task,
                analysis,
                max_debug_attempts=1,
                stall_seconds=1,
            )
        self.assertEqual("completed", result["status"])
        self.assertIsNone(result["output"])
        self.assertAlmostEqual(result["score"], -np.log(0.8))
        self.assertTrue(result["prediction_evidence"]["centrally_scored"])
        self.assertFalse((self.node / "submission").exists())

    def test_deferred_table_materializer_aligns_identifiers(self) -> None:
        submission = self.write_sample_csv(
            "id,class_0,class_1\na,0.5,0.5\nb,0.5,0.5\n"
        )
        manager = object.__new__(ManagerAgent)
        manager.task_dir = self.task
        manager.task_analysis = self.analysis(submission=submission)
        output = manager._materialize_prediction_table(
            np.asarray([[0.2, 0.8], [0.7, 0.3]]),
            np.asarray(["b", "a"]),
            self.node,
        )
        self.assertEqual(
            "id,class_0,class_1\na,0.7,0.3\nb,0.2,0.8\n",
            output.read_text(encoding="utf-8"),
        )

    def test_winner_is_revalidated_before_final_materialization(self) -> None:
        submission = self.write_sample_csv()
        run_root = self.root / "run"
        node_dir = run_root / "node1"
        node_dir.mkdir(parents=True)
        output = node_dir / "submission.csv"
        output.write_text("id,wrong\na,0.2\nb,0.8\n", encoding="utf-8")

        manager = object.__new__(ManagerAgent)
        manager.run_root = run_root
        manager.task_analysis = self.analysis(submission=submission)
        manager.submission_validator = self.validator
        manager.aggregator_agent = AggregatorAgent()
        manager.final_output_path = None
        manager.all_nodes = {
            "node1": NodeState(
                node_id="node1",
                parent_id="root",
                result={"status": "completed", "output": str(output), "score": 0.8},
                executed=True,
            )
        }
        manager._persist_tree_state = lambda: run_root / "tree_state.json"

        self.assertFalse(manager.generate_final_submission("node1"))
        self.assertFalse((run_root / "submission.csv").is_file())
        output.write_text("id,prediction\na,0.2\nb,0.8\n", encoding="utf-8")
        self.assertTrue(manager.generate_final_submission("node1"))
        self.assertTrue((run_root / "submission.csv").is_file())
        self.assertFalse(output.exists())

    def test_task_analyzer_discovers_non_csv_sample_output(self) -> None:
        (self.task / "sample_output.json").write_text('{"predictions": []}', encoding="utf-8")
        (self.task / "README.md").write_text("Your task is to produce predictions.", encoding="utf-8")
        analysis = TaskAnalyzer().analyze(self.task)
        self.assertIsNotNone(analysis.submission)
        self.assertEqual("sample_output.json", analysis.submission["path"])
        self.assertIn("matching the observed sample output", analysis.expected_output)

    def test_task_analyzer_does_not_convert_tsv_output_contract_to_csv(self) -> None:
        (self.task / "sample_output.tsv").write_text(
            "id\tprediction\na\t0.5\n", encoding="utf-8"
        )
        analysis = TaskAnalyzer().analyze(self.task)
        self.assertEqual(".tsv", analysis.submission["extension"])
        self.assertIn("`.tsv`", analysis.expected_output)
        self.assertNotIn("submission.csv", analysis.expected_output)

    def test_task_analyzer_surfaces_ambiguous_metric_and_output_contracts(self) -> None:
        (self.task / "README.md").write_text(
            "Generate a biologically plausible protein candidate.", encoding="utf-8"
        )
        analysis = TaskAnalyzer().analyze(self.task)
        self.assertEqual("task score", analysis.metric)
        self.assertTrue(any("scalar metric" in item for item in analysis.contract_warnings))
        self.assertTrue(any("structural validation" in item for item in analysis.contract_warnings))
        self.assertIn("Contract warnings", analysis.report)

    def test_task_analyzer_recognizes_structure_metric_and_sample_output(self) -> None:
        (self.task / "README.md").write_text(
            "Predict the protein structure; evaluation uses RMSD.", encoding="utf-8"
        )
        (self.task / "sample_output.pdb").write_text(
            "ATOM      1  CA  GLY A   1       0.0 0.0 0.0\n", encoding="utf-8"
        )
        analysis = TaskAnalyzer().analyze(self.task)
        self.assertEqual("RMSD", analysis.metric)
        self.assertEqual("minimize", analysis.direction)
        self.assertEqual("sample_output.pdb", analysis.submission["path"])


if __name__ == "__main__":
    unittest.main()
