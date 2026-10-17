import pytest

from multi_turn_eval.judging.claude_judge import extract_judge_result


def judge_root() -> dict:
    return {
        "phase1_analysis": [],
        "function_call_tracking": {},
        "realignment_notes": "",
        "final_judgments": [],
    }


def compact_root() -> str:
    return (
        '{"phase1_analysis":[],"function_call_tracking":{},'
        '"realignment_notes":"","final_judgments":[]}'
    )


@pytest.mark.parametrize(
    "prefix",
    [
        # Retained invalid attempt 1: two complete empty objects in prose.
        "Looking at the format {} and then {} before the result.\n",
        # Retained invalid attempt 2: one complete empty object in prose.
        "Looking at one format example {} before the result.\n",
        # Retained invalid attempt 3: the same one-object prefix class at a
        # different offset/length.
        "Looking at a longer structural preamble with one example {} before the result.\n",
    ],
)
def test_retained_invalid_structural_patterns_select_fenced_root(prefix: str) -> None:
    response = prefix + "```json\n" + compact_root() + "\n```"
    assert extract_judge_result(response) == judge_root()


def test_legacy_valid_result_is_unchanged() -> None:
    expected = judge_root()
    assert extract_judge_result("prose\n```json\n" + compact_root() + "\n```") == expected


def test_bom_and_fence_are_wrappers_not_payload() -> None:
    assert extract_judge_result("\ufeff```json\n" + compact_root() + "\n```") == judge_root()


def test_doubled_schema_results_fail_closed() -> None:
    with pytest.raises(ValueError, match="found 2"):
        extract_judge_result(compact_root() + "\n" + compact_root())


def test_unrelated_or_incomplete_objects_do_not_match() -> None:
    with pytest.raises(ValueError, match="found 0"):
        extract_judge_result('prefix {} {"final_judgments": []} suffix')
