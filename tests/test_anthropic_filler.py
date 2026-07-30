"""Focused coverage for request-local filler injection on Anthropic requests."""

from multi_turn_eval.services.anthropic_logged import (
    _apply_filler_anthropic,
    _apply_filler_anthropic_system,
)


def test_anthropic_prefix_fills_last_text_user_without_mutation(monkeypatch):
    monkeypatch.setenv("MTE_FILLER_DOTS", "3")
    monkeypatch.setenv("MTE_FILLER_TOKEN", "-")
    monkeypatch.setenv("MTE_FILLER_POSITION", "prefix")
    question = {"role": "user", "content": [{"type": "text", "text": "Book it."}]}
    tool_result = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"}],
    }
    messages = [question, tool_result]

    filled = _apply_filler_anthropic(messages)

    assert filled is not messages
    assert filled[0] is not question
    assert filled[0]["content"] is not question["content"]
    assert filled[0]["content"][0]["text"] == "- - - Book it."
    assert filled[1] is tool_result
    assert question["content"][0]["text"] == "Book it."


def test_anthropic_system_filler_handles_text_blocks_without_mutation(monkeypatch):
    monkeypatch.setenv("MTE_FILLER_DOTS", "2")
    monkeypatch.setenv("MTE_FILLER_TOKEN", ".")
    monkeypatch.setenv("MTE_FILLER_POSITION", "system")
    system = [{"type": "text", "text": "Use tools."}]
    messages = [{"role": "user", "content": "Hello"}]

    filled_messages = _apply_filler_anthropic(messages)
    filled_system = _apply_filler_anthropic_system(system)

    assert filled_messages is messages
    assert filled_system is not system
    assert filled_system[0] is not system[0]
    assert filled_system[0]["text"] == "Use tools. . ."
    assert system[0]["text"] == "Use tools."


def test_anthropic_unset_filler_is_identity_noop(monkeypatch):
    monkeypatch.delenv("MTE_FILLER_DOTS", raising=False)
    messages = [{"role": "user", "content": "Hello"}]
    system = "Use tools."

    assert _apply_filler_anthropic(messages) is messages
    assert _apply_filler_anthropic_system(system) is system
