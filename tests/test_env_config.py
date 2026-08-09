from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.env_config import load_project_environment


class EnvironmentConfigTests(unittest.TestCase):
    def test_project_dotenv_is_loaded_without_exposing_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret = "test-openalex-secret"
            (root / ".env").write_text(
                f'OPENALEX_API_KEY="{secret}" # local secret\n'
                "AIBUILDAI_COUNCIL_MAX_QUERIES=4\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "OPENALEX_API_KEY": "",
                    "AIBUILDAI_COUNCIL_MAX_QUERIES": "",
                },
                clear=False,
            ):
                # Empty exported variables are intentionally authoritative too;
                # remove them to represent an unset shell.
                os.environ.pop("OPENALEX_API_KEY")
                os.environ.pop("AIBUILDAI_COUNCIL_MAX_QUERIES")
                result = load_project_environment(root)
                self.assertEqual(secret, os.environ["OPENALEX_API_KEY"])
                self.assertEqual("4", os.environ["AIBUILDAI_COUNCIL_MAX_QUERIES"])
                self.assertEqual(
                    ("AIBUILDAI_COUNCIL_MAX_QUERIES", "OPENALEX_API_KEY"),
                    result.loaded_names,
                )
                self.assertNotIn(secret, repr(result))

    def test_exported_value_wins_and_openalex_alias_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text(
                "OPENALEX_API_KEY=file-value\nSETTING=file-value\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "OPENALEX_API_KEY": "exported-value",
                    "SETTING": "exported-value",
                },
                clear=False,
            ):
                result = load_project_environment(root)
                self.assertEqual("exported-value", os.environ["OPENALEX_API_KEY"])
                self.assertEqual("exported-value", os.environ["SETTING"])
                self.assertEqual((), result.loaded_names)

            with patch.dict(
                os.environ,
                {"OPENALEX_KEY": "alias-value"},
                clear=False,
            ):
                os.environ.pop("OPENALEX_API_KEY", None)
                result = load_project_environment(root / "missing")
                self.assertEqual("alias-value", os.environ["OPENALEX_API_KEY"])
                self.assertEqual("OPENALEX_KEY", result.openalex_alias)


if __name__ == "__main__":
    unittest.main()
