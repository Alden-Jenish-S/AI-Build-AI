"""Regression tests for modality fidelity: the system must work only with the
modalities the task actually provides and must never fabricate a second
modality (e.g. treat a label-only CSV as "tabular" on an image task)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.implementation_agent import _python_source
from agents.modality_policy import predictive_modality_inventory
from agents.task_analyzer import TaskAnalysis, TaskAnalyzer


def _table(path: str, columns: list[str]) -> dict:
    return {
        "path": path,
        "kind": "table",
        "extension": ".csv",
        "bytes": 100,
        "profile": {"columns": columns, "rows": 10, "dtypes": {}, "sample": []},
    }


class ModalityDisciplineTests(unittest.TestCase):
    def test_image_task_with_label_only_csv_is_not_multimodal(self) -> None:
        files = [
            {"path": "images/train/1.jpg", "kind": "image"},
            _table("train.csv", ["image_id", "healthy", "rust", "scab"]),
            _table("test.csv", ["image_id"]),
            _table(
                "sample_submission.csv",
                ["image_id", "healthy", "rust", "scab"],
            ),
        ]
        inventory = predictive_modality_inventory(files)
        self.assertEqual(["image"], inventory["modalities"])
        self.assertFalse(inventory["is_multimodal"])

    def test_feature_table_still_counts_as_tabular_modality(self) -> None:
        files = [
            {"path": "images/train/1.jpg", "kind": "image"},
            _table("train.csv", ["id", "f1", "f2", "target"]),
            _table("sample_submission.csv", ["id", "target"]),
        ]
        inventory = predictive_modality_inventory(files)
        self.assertEqual(["image", "tabular"], inventory["modalities"])
        self.assertTrue(inventory["is_multimodal"])

    def test_id_and_target_only_table_is_not_tabular_without_submission(self) -> None:
        files = [
            {"path": "images/1.jpg", "kind": "image"},
            _table("train.csv", ["image_id", "label"]),
        ]
        inventory = predictive_modality_inventory(files)
        self.assertEqual(["image"], inventory["modalities"])

    def test_identifier_only_table_is_not_a_modality(self) -> None:
        files = [_table("train.csv", ["image_id"])]
        inventory = predictive_modality_inventory(files)
        self.assertEqual([], inventory["modalities"])

    def test_tables_without_column_profile_keep_conservative_default(self) -> None:
        files = [
            {"path": "train.csv", "kind": "table"},
            {"path": "sample_submission.csv", "kind": "table"},
        ]
        inventory = predictive_modality_inventory(files)
        self.assertEqual(["tabular"], inventory["modalities"])

    def test_task_analysis_modalities_populated_from_real_task_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = Path(temporary) / "image_only_task"
            images = task_dir / "images"
            images.mkdir(parents=True)
            (images / "leaf_1.jpg").write_bytes(b"\xff\xd8\xff\xe0")
            (task_dir / "train.csv").write_text(
                "image_id,healthy,rust,scab\nleaf_1,0,1,0\n"
            )
            (task_dir / "test.csv").write_text("image_id\nleaf_2\n")
            (task_dir / "sample_submission.csv").write_text(
                "image_id,healthy,rust,scab\nleaf_2,0.5,0.5,0.5\n"
            )

            analysis: TaskAnalysis = TaskAnalyzer().analyze(task_dir)
            self.assertEqual(["image"], analysis.modalities)
            self.assertEqual(["image"], analysis.to_dict()["modalities"])
            self.assertIn("Predictive modalities: image", analysis.report)

    def test_task_analysis_modalities_include_real_tabular_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = Path(temporary) / "mixed_task"
            images = task_dir / "images"
            images.mkdir(parents=True)
            (images / "leaf_1.jpg").write_bytes(b"\xff\xd8\xff\xe0")
            (task_dir / "train.csv").write_text(
                "image_id,f1,f2,target\nleaf_1,0.1,0.2,1\n"
            )
            (task_dir / "test.csv").write_text("image_id,f1,f2\nleaf_2,0.3,0.4\n")
            (task_dir / "sample_submission.csv").write_text(
                "image_id,target\nleaf_2,0.5\n"
            )

            analysis: TaskAnalysis = TaskAnalyzer().analyze(task_dir)
            self.assertEqual(["image", "tabular"], analysis.modalities)
            self.assertTrue(analysis.to_dict()["modalities"])


class PythonSourceExtractionTests(unittest.TestCase):
    def test_well_formed_fenced_block(self) -> None:
        response = (
            "Here is my reasoning.\n"
            "```python\nimport torch\nprint(torch.__version__)\n```\n"
            "Done."
        )
        self.assertEqual(
            "import torch\nprint(torch.__version__)\n", _python_source(response)
        )

    def test_unclosed_fence_is_recovered(self) -> None:
        response = (
            "Reasoning here.\n"
            "```python\nimport numpy as np\nx = np.arange(3)\n"
        )
        source = _python_source(response)
        self.assertIn("import numpy as np", source)
        self.assertIn("x = np.arange(3)", source)

    def test_leading_prose_inside_fence_is_dropped(self) -> None:
        response = (
            "```python\n"
            "- So effectively this is a pure image classification task.\n"
            "- The tabular modality here is just the image_id.\n"
            "import pandas as pd\n"
            "df = pd.DataFrame()\n"
            "```\n"
        )
        source = _python_source(response)
        self.assertIn("import pandas as pd", source)
        self.assertNotIn("effectively", source)

    def test_trailing_prose_inside_fence_is_dropped(self) -> None:
        response = (
            "```python\n"
            "import pandas as pd\n"
            "df = pd.DataFrame()\n"
            "I will use 5 epochs for the probe mode.\n"
            "```\n"
        )
        source = _python_source(response)
        self.assertIn("import pandas as pd", source)
        self.assertNotIn("epochs for the probe", source)

    def test_thinking_blocks_are_removed(self) -> None:
        response = (
            "<thinking>Let me think about this carefully.</thinking>\n"
            "```python\nprint(1)\n```\n"
        )
        self.assertEqual("print(1)\n", _python_source(response))


if __name__ == "__main__":
    unittest.main()
