from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from agents.task_analyzer import TaskAnalyzer
from agents.task_inventory import build_task_inventory
from agents.technique_agent import TechniqueAgent
from core.contracts import TaskSpec
from core.runtime_contracts import DatasetBundle, SampleRecord
from evaluation.fidelity import get_fidelity_profile


class ContentFirstTaskDiscoveryTests(unittest.TestCase):
    def test_paired_roles_come_from_content_and_alignment_not_directory_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "opaque_task"
            observed_inputs = task_dir / "part_a" / "collection_q"
            observed_targets = task_dir / "part_a" / "collection_r"
            inference_inputs = task_dir / "part_b" / "collection_q"
            for directory in (
                observed_inputs,
                observed_targets,
                inference_inputs,
            ):
                directory.mkdir(parents=True)

            rng = np.random.default_rng(7)
            for index in range(6):
                sample_id = f"entity_{index}"
                Image.fromarray(
                    rng.integers(0, 256, size=(12, 12), dtype=np.uint8)
                ).save(observed_inputs / f"{sample_id}.png")
                Image.fromarray(
                    np.where(
                        np.indices((12, 12))[0] <= index,
                        255,
                        0,
                    ).astype(np.uint8)
                ).save(observed_targets / f"{sample_id}.png")
            inference_ids = []
            for index in range(3):
                sample_id = f"future_{index}"
                inference_ids.append(sample_id)
                Image.fromarray(
                    rng.integers(0, 256, size=(12, 12), dtype=np.uint8)
                ).save(inference_inputs / f"{sample_id}.png")
            pd.DataFrame(
                {"key": inference_ids, "encoded": [""] * len(inference_ids)}
            ).to_csv(task_dir / "format_reference.csv", index=False)

            analysis = TaskAnalyzer().analyze(task_dir, include_index=True)

            self.assertTrue(analysis.verification["verified"])
            self.assertEqual(analysis.task_spec.problem_type, "segmentation")
            self.assertEqual(
                analysis.task_spec.inputs["image"].source,
                "part_a/collection_q",
            )
            self.assertEqual(
                analysis.task_spec.target.source,
                "part_a/collection_r",
            )
            self.assertEqual(len(analysis.bundle.train_records), 6)
            self.assertEqual(len(analysis.bundle.test_records), 3)

    def test_unknown_file_collection_is_not_silently_treated_as_a_table(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "mixed_task"
            task_dir.mkdir()
            pd.DataFrame(
                {"key": [1, 2, 3], "feature": [2, 3, 4], "outcome": [0, 1, 0]}
            ).to_csv(task_dir / "records_a.csv", index=False)
            pd.DataFrame(
                {"key": [4, 5], "feature": [5, 6]}
            ).to_csv(task_dir / "records_b.csv", index=False)
            opaque = task_dir / "payloads"
            opaque.mkdir()
            for index in range(12):
                (opaque / f"{index}.blob").write_bytes(bytes([index, 0, 255]))

            inventory = build_task_inventory(task_dir)
            self.assertEqual(inventory["total_files"], 14)
            with self.assertRaisesRegex(ValueError, "refusing to assume"):
                TaskAnalyzer().resolve(task_dir)

    def test_contract_identifiers_are_extensible_not_allowlisted(self):
        spec = TaskSpec.from_mapping(
            "chemistry",
            {
                "schema_version": 2,
                "modality": "molecular-graph",
                "problem_type": "property-ranking",
                "inputs": {
                    "structures": {
                        "modality": "molecular-graph",
                        "source": "structures.sdf",
                        "format": "sdf",
                    }
                },
                "target": {"source": "assays.dat", "field": "response"},
                "output": {"type": "ordered-candidates"},
                "metrics": [
                    {
                        "name": "domain-utility",
                        "direction": "maximize",
                    }
                ],
            },
        )
        self.assertEqual(spec.modality, "molecular_graph")
        self.assertEqual(spec.problem_type, "property_ranking")
        self.assertEqual(spec.output.type, "ordered_candidates")
        self.assertEqual(spec.primary_metric, "domain_utility")

    def test_explicit_unknown_task_kind_uses_contract_driven_indexing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "novel_task"
            task_dir.mkdir()
            (task_dir / "observations.jsonl").write_text(
                "\n".join(
                    json.dumps({"key": key, "payload": value, "response": value + 1})
                    for key, value in (("a", 2), ("b", 5), ("c", 8))
                )
                + "\n",
                encoding="utf-8",
            )
            (task_dir / "queries.jsonl").write_text(
                "\n".join(
                    json.dumps({"key": key, "payload": value})
                    for key, value in (("d", 3), ("e", 7))
                )
                + "\n",
                encoding="utf-8",
            )
            (task_dir / "task_config.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "modality": "event-sequence-with-context",
                        "problem_type": "next-state-utility",
                        "inputs": {
                            "history": {
                                "role": "train",
                                "source": "observations.jsonl",
                                "format": "jsonl",
                                "id_field": "key",
                            },
                            "future": {
                                "role": "test",
                                "source": "queries.jsonl",
                                "format": "jsonl",
                                "id_field": "key",
                            },
                        },
                        "target": {
                            "source": "observations.jsonl",
                            "field": "response",
                        },
                        "sample_id_field": "key",
                        "output": {"type": "task-native-value"},
                        "metrics": [
                            {"name": "absolute-utility-error", "direction": "minimize"}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            analysis = TaskAnalyzer().analyze(task_dir, include_index=True)

            self.assertTrue(analysis.verification["verified"])
            self.assertEqual(analysis.task_spec.modality, "event_sequence_with_context")
            self.assertEqual(len(analysis.bundle.train_records), 3)
            self.assertEqual(len(analysis.bundle.test_records), 2)
            self.assertEqual(analysis.bundle.train_records[0].target, 3)

    def test_analysis_agent_resolves_unknown_files_before_profiling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "unclassified_task"
            task_dir.mkdir()
            (task_dir / "task_description.md").write_text(
                "Estimate response for each query; minimize mean absolute error.",
                encoding="utf-8",
            )
            (task_dir / "observed.jsonl").write_text(
                '\n'.join(
                    json.dumps({"key": key, "signal": value, "response": value + 1})
                    for key, value in (("a", 1), ("b", 4), ("c", 9))
                ) + '\n',
                encoding="utf-8",
            )
            (task_dir / "requested.jsonl").write_text(
                '\n'.join(
                    json.dumps({"key": key, "signal": value})
                    for key, value in (("d", 2), ("e", 6))
                ) + '\n',
                encoding="utf-8",
            )
            response = {
                "resolved": True,
                "summary": "The response field exists only for observed examples.",
                "evidence": ["observed.jsonl has response; requested.jsonl does not"],
                "uncertainties": [],
                "contract": {
                    "task_kind": "event-stream",
                    "objective": "response-estimation",
                    "inputs": [
                        {
                            "name": "observed_events",
                            "role": "train",
                            "source": "observed.jsonl",
                            "format": "jsonl",
                            "id_field": "key",
                            "path_field": "",
                            "required": True,
                            "evidence": "contains signal and response",
                        },
                        {
                            "name": "requested_events",
                            "role": "test",
                            "source": "requested.jsonl",
                            "format": "jsonl",
                            "id_field": "key",
                            "path_field": "",
                            "required": True,
                            "evidence": "contains signal without response",
                        },
                    ],
                    "target": {
                        "present": True,
                        "source": "observed.jsonl",
                        "field": "response",
                        "format": "jsonl",
                        "id_field": "key",
                        "alignment": "id",
                        "evidence": "explicit response values",
                    },
                    "sample_id_field": "key",
                    "output": {
                        "type": "response-value",
                        "shape": "one value per requested key",
                        "encoding": "number",
                    },
                    "metrics": [{"name": "mae", "direction": "minimize"}],
                    "primary_metric": "mae",
                },
            }
            with patch("agents.task_analyzer.call_llm_json", return_value=response) as llm:
                analysis = TaskAnalyzer(
                    model_name="mock",
                    enable_agent_resolution=True,
                ).analyze(task_dir, include_index=True)

            self.assertTrue(llm.called)
            self.assertTrue(analysis.verification["verified"])
            self.assertEqual(analysis.task_spec.modality, "event_stream")
            self.assertEqual(len(analysis.bundle.train_records), 3)
            self.assertEqual(len(analysis.bundle.test_records), 2)

    def test_targetless_task_is_not_rejected_by_objective_name(self):
        spec = TaskSpec.from_mapping(
            "representation",
            {
                "schema_version": 2,
                "modality": "opaque_vectors",
                "problem_type": "contrastive_representation_learning",
                "inputs": {
                    "examples": {
                        "role": "train",
                        "source": "vectors.bin",
                        "format": "binary",
                    }
                },
                "output": {"type": "embeddings"},
                "metrics": [
                    {"name": "neighborhood_consistency", "direction": "maximize"}
                ],
            },
        )
        bundle = DatasetBundle(
            task=spec,
            train_records=(
                SampleRecord(
                    sample_id="x",
                    inputs={"examples": "input/vectors.bin"},
                    target=None,
                ),
            ),
        )
        self.assertIsNone(bundle.train_records[0].target)

    def test_fidelity_does_not_inject_data_family_assumptions(self):
        first = get_fidelity_profile("unseen_binary_archive", "screen").to_dict()
        second = get_fidelity_profile("image", "screen").to_dict()
        self.assertEqual(first, second)
        self.assertNotIn("spatial_size", second)
        self.assertNotIn("audio_sample_rate", second)
        self.assertNotIn("video_frames", second)


class EvidenceGroundedPlanningTests(unittest.TestCase):
    def test_initial_planning_prompt_contains_verified_task_evidence(self):
        agent = TechniqueAgent()
        agent.set_task_evidence(
            {
                "verification": {"verified": True},
                "observed_file_groups": [
                    {"directory": "opaque/a", "file_count": 17}
                ],
            }
        )
        captured = {}

        def fake_call(system, user, **kwargs):
            captured["system"] = system
            return [{"name": "observed_pipeline", "plan": "Use opaque/a."}]

        with patch("agents.technique_agent.call_llm_json", side_effect=fake_call):
            approaches = agent.generate_initial_approaches("task narrative", count=1)

        self.assertEqual(len(approaches), 1)
        self.assertIn("opaque/a", captured["system"])
        self.assertIn("Do not assume conventional filenames", captured["system"])

    def test_artifact_selection_is_not_prefiltered_by_task_category(self):
        agent = TechniqueAgent()
        agent.set_task_evidence(
            {"verification": {"verified": True}, "inputs": ["opaque.store"]}
        )
        artifact = {
            "artifact_id": "generic_callable",
            "category": "reusable_methods",
            "description": "Callable representation learner with an adaptable interface.",
            "interface": {"entrypoint": "fit_predict"},
            "capabilities": {
                "modalities": ["a_different_historical_label"],
                "target_types": ["a_different_historical_objective"],
            },
            "verified": True,
            "scope": "model_family",
        }

        def fake_query(category, artifact_id=None):
            if artifact_id is None:
                return {"artifacts": [artifact]}
            return artifact

        with patch("agents.technique_agent.query", side_effect=fake_query), patch(
            "agents.technique_agent.call_llm",
            side_effect=["reusable_methods", "generic_callable"],
        ):
            result = agent.run(
                "derive a method from the observed task files",
                "learn a representation justified by observed relationships",
                {},
                {
                    "reusable_methods": {
                        "description": "Reusable callable learning methods"
                    }
                },
                task_spec={
                    "modality": "never_seen_before",
                    "problem_type": "custom_objective",
                    "output": {"type": "custom_output"},
                },
                enable_executable_artifacts=True,
            )

        self.assertEqual(result["status"], "pool_hit")
        self.assertEqual(result["artifact_id"], "generic_callable")


if __name__ == "__main__":
    unittest.main()
