from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.llm_utils import call_llm_json, reset_token_usage


_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


class LLMJSONFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_token_usage()

    def tearDown(self) -> None:
        cache_path = os.environ.pop("AIBUILDAI_LLM_CAPABILITY_CACHE", None)
        if cache_path is not None:
            try:
                Path(cache_path).unlink(missing_ok=True)
            except OSError:
                pass

    def test_unsupported_response_format_gets_schema_visible_fallback_and_repair(self) -> None:
        unavailable = RuntimeError(
            "Error code: 400 - This response_format type is unavailable now"
        )
        with patch(
            "agents.llm_utils.call_llm",
            side_effect=[unavailable, '{"wrong":"field"}', '{"answer":"repaired"}'],
        ) as mocked:
            result = call_llm_json(
                "system",
                "produce an answer",
                schema=_SCHEMA,
                schema_name="answer",
            )
        self.assertEqual({"answer": "repaired"}, result)
        self.assertEqual(3, mocked.call_count)
        self.assertIsNotNone(mocked.call_args_list[0].kwargs.get("response_format"))
        self.assertIn("JSON Schema", mocked.call_args_list[1].args[1])
        self.assertIn("missing required field", mocked.call_args_list[2].args[1])

    def test_capability_is_remembered_for_later_council_calls(self) -> None:
        unavailable = RuntimeError(
            "Error code: 400 - This response_format type is unavailable now"
        )
        with patch(
            "agents.llm_utils.call_llm",
            side_effect=[unavailable, '{"answer":"first"}'],
        ):
            first = call_llm_json(
                "system", "first", schema=_SCHEMA, schema_name="first"
            )
        self.assertEqual({"answer": "first"}, first)

        with patch(
            "agents.llm_utils.call_llm", return_value='{"answer":"second"}'
        ) as mocked:
            second = call_llm_json(
                "system", "second", schema=_SCHEMA, schema_name="second"
            )
        self.assertEqual({"answer": "second"}, second)
        self.assertEqual(1, mocked.call_count)
        self.assertIsNone(mocked.call_args.kwargs.get("response_format"))

    def test_capability_persists_across_process_resets(self) -> None:
        unavailable = RuntimeError(
            "Error code: 400 - This response_format type is unavailable now"
        )
        original = {
            key: os.environ.get(key)
            for key in (
                "LLM_PROVIDER",
                "LLM_BASE_URL",
                "LLM_API_KEY",
                "LLM_MODEL",
            )
        }
        os.environ.update(
            {
                "LLM_PROVIDER": "unittest_provider",
                "LLM_BASE_URL": "https://example.invalid/v1",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "test-model",
            }
        )
        try:
            with tempfile.TemporaryDirectory() as temporary:
                cache_path = str(Path(temporary) / "llm_capabilities.json")
                os.environ["AIBUILDAI_LLM_CAPABILITY_CACHE"] = cache_path
                with patch(
                    "agents.llm_utils.call_llm",
                    side_effect=[unavailable, '{"answer":"first"}'],
                ):
                    call_llm_json(
                        "system", "probe", schema=_SCHEMA, schema_name="probe"
                    )
                self.assertTrue(Path(cache_path).is_file())
                # Simulate a fresh process: reset re-seeds the set from disk.
                reset_token_usage()
                with patch(
                    "agents.llm_utils.call_llm",
                    return_value='{"answer":"cached-away"}',
                ) as mocked:
                    result = call_llm_json(
                        "system", "again", schema=_SCHEMA, schema_name="again"
                    )
                self.assertEqual({"answer": "cached-away"}, result)
                # Structured output must be skipped entirely on the re-seeded
                # process: no double-send on the first structured call.
                self.assertIsNone(mocked.call_args.kwargs.get("response_format"))
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
