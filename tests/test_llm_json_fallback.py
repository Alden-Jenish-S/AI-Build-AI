from __future__ import annotations

import unittest
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


if __name__ == "__main__":
    unittest.main()
