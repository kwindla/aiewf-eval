#!/usr/bin/env python3

import json
import unittest

from scorer import score_message


def message(name=None, args=None, content=""):
    calls = []
    if name:
        calls.append(
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    return {"role": "assistant", "content": content, "tool_calls": calls}


class ScorerTest(unittest.TestCase):
    def test_turn12_correct(self):
        scored = score_message(
            12,
            message(
                "submit_session_suggestion",
                {
                    "name": "Jennifer Smith",
                    "suggestion_text": "A session on state machine abstractions for complex workflows.",
                },
            ),
        )
        self.assertTrue(scored["success"])

    def test_turn15_correct(self):
        scored = score_message(
            15,
            message(
                "submit_dietary_request",
                {"name": "Jennifer Smith", "dietary_preference": "vegan"},
            ),
        )
        self.assertTrue(scored["success"])

    def test_turn12_historical_short_variant_is_correct(self):
        scored = score_message(
            12,
            message(
                "submit_session_suggestion",
                {
                    "name": "Jennifer Smith",
                    "suggestion_text": "state machine abstractions for complex workflows",
                },
            ),
        )
        self.assertTrue(scored["success"])

    def test_turn12_negated_decoy_is_wrong(self):
        scored = score_message(
            12,
            message(
                "submit_session_suggestion",
                {
                    "name": "Jennifer Smith",
                    "suggestion_text": (
                        "Not a session on state machine abstractions for complex workflows"
                    ),
                },
            ),
        )
        self.assertFalse(scored["success"])
        self.assertEqual(scored["category"], "correct_tool_wrong_or_missing_argument")

    def test_turn15_negation_and_substring_collisions_are_wrong(self):
        for preference in ("not vegan", "non-vegan", "vegan-ish"):
            with self.subTest(preference=preference):
                scored = score_message(
                    15,
                    message(
                        "submit_dietary_request",
                        {"name": "Jennifer Smith", "dietary_preference": preference},
                    ),
                )
                self.assertFalse(scored["success"])

    def test_extra_and_missing_arguments_are_wrong(self):
        cases = (
            (12, "submit_session_suggestion", {"name": "Jennifer Smith"}),
            (
                12,
                "submit_session_suggestion",
                {
                    "name": "Jennifer Smith",
                    "suggestion_text": "A session on state machine abstractions for complex workflows.",
                    "session_id": "invented",
                },
            ),
            (15, "submit_dietary_request", {"name": "Jennifer Smith"}),
            (
                15,
                "submit_dietary_request",
                {
                    "name": "Jennifer Smith",
                    "dietary_preference": "vegan",
                    "notes": "invented",
                },
            ),
        )
        for turn, name, args in cases:
            with self.subTest(turn=turn, args=args):
                scored = score_message(turn, message(name, args))
                self.assertFalse(scored["success"])

    def test_malformed_arguments_are_model_failure(self):
        scored = score_message(
            15,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "submit_dietary_request",
                            "arguments": "{not json",
                        },
                    }
                ],
            },
        )
        self.assertFalse(scored["success"])
        self.assertEqual(scored["category"], "malformed_tool_call")

    def test_missing_message_is_response_parser_failure(self):
        scored = score_message(12, None)
        self.assertEqual(scored["category"], "response_parser_failure")

    def test_wrong_tool_is_failure(self):
        scored = score_message(
            15,
            message(
                "submit_session_suggestion",
                {
                    "name": "Jennifer Smith",
                    "suggestion_text": "A session on state machine abstractions for complex workflows.",
                },
            ),
        )
        self.assertFalse(scored["success"])
        self.assertEqual(scored["category"], "wrong_tool")

    def test_false_claim(self):
        scored = score_message(12, message(content="I've submitted that suggestion."))
        self.assertEqual(scored["category"], "no_tool_false_claim_of_completion")

    def test_confirmation(self):
        scored = score_message(15, message(content="Would you like me to submit it?"))
        self.assertEqual(scored["category"], "no_tool_redundant_confirmation_or_question")

    def test_multiple_calls_are_strict_failure(self):
        one = message(
            "submit_dietary_request",
            {"name": "Jennifer Smith", "dietary_preference": "vegan"},
        )["tool_calls"][0]
        scored = score_message(15, {"role": "assistant", "content": "", "tool_calls": [one, one]})
        self.assertFalse(scored["success"])
        self.assertEqual(scored["category"], "duplicate_or_multiple_tool_calls")


if __name__ == "__main__":
    unittest.main()
