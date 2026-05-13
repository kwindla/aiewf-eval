import hashlib
import inspect
import json

import pytest

from benchmarks._shared.tools import ToolsSchemaForTest
from multi_turn_eval.services.nemotron_audio_in import NemotronAudioInLLMService
from pipecat.processors.aggregators.llm_context import LLMContext


def _service(**kwargs):
    return NemotronAudioInLLMService(
        api_key=None,
        base_url="http://example.test/v1",
        **kwargs,
    )


def _user_message(content="hello"):
    return {"role": "user", "content": content}


def _payload_from_context(service, context_messages, context=None):
    canonical = service._canonical_messages_from_context(context_messages)
    assert canonical is not None
    payload, full_messages = service._build_payload_from_messages(
        canonical,
        context=context,
    )
    return payload, full_messages


def test_input_audio_converts_to_audio_url():
    service = _service()

    converted = service._convert_context_content_part(
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
    )

    assert converted == {
        "type": "audio_url",
        "audio_url": {"url": "data:audio/wav;base64,AAAA"},
    }


def test_audio_url_passes_through():
    service = _service()
    original = {
        "type": "audio_url",
        "audio_url": {"url": "data:audio/wav;base64,AAAA"},
    }

    converted = service._convert_context_content_part(original)

    assert converted == original
    assert converted is not original
    assert converted["audio_url"] is not original["audio_url"]


def test_chat_template_kwargs_emitted():
    default_service = _service()
    default_payload, _ = _payload_from_context(default_service, [_user_message()])
    assert default_payload["chat_template_kwargs"] == {"enable_thinking": False}

    thinking_service = _service(chat_template_kwargs={"enable_thinking": True})
    thinking_payload, _ = _payload_from_context(thinking_service, [_user_message()])
    assert thinking_payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_can_generate_metrics_true():
    assert _service().can_generate_metrics() is True


def test_set_model_name():
    assert _service().model_name == "nemotron_3_nano_omni"


def test_cache_off_payload_omits_cache_fields():
    payload, _ = _payload_from_context(_service(), [_user_message()])

    assert payload["messages"] == [_user_message()]
    assert "conversation_id" not in payload
    assert "conversation_require_cache" not in payload


def test_cache_on_full_context_includes_id_no_require():
    service = _service(conversation_cache_enabled=True)
    service._conversation_id = "abc"

    payload, _ = _payload_from_context(service, [_user_message()])

    assert payload["conversation_id"] == "abc"
    assert "conversation_require_cache" not in payload


def test_suffix_only_payload_has_one_user_message_and_require_cache():
    service = _service(conversation_cache_enabled=True, suffix_only_conversation=True)
    service._conversation_id = "abc"
    service._conversation_cache_committed = True

    payload, _ = _payload_from_context(service, [_user_message("latest")])

    assert payload["messages"] == [_user_message("latest")]
    assert payload["conversation_require_cache"] is True


@pytest.mark.parametrize(
    (
        "cache_enabled",
        "suffix_only",
        "cache_committed",
        "conversation_id",
        "expected",
    ),
    [
        (False, True, True, "abc", False),
        (True, False, True, "abc", False),
        (True, True, False, "abc", False),
        (True, True, True, "abc", True),
    ],
)
def test_should_send_suffix_only_gating(
    cache_enabled,
    suffix_only,
    cache_committed,
    conversation_id,
    expected,
):
    service = _service(
        conversation_cache_enabled=cache_enabled,
        suffix_only_conversation=suffix_only,
    )
    service._conversation_cache_committed = cache_committed
    service._conversation_id = conversation_id

    assert service._should_send_suffix_only() is expected


def test_latest_user_message():
    service = _service()
    messages = [
        {"role": "system", "content": "system"},
        _user_message("first"),
        {"role": "assistant", "content": "assistant"},
        _user_message("second"),
    ]

    latest = service._latest_user_message(messages)

    assert latest == _user_message("second")
    assert latest is not messages[-1]
    assert service._latest_user_message([{"role": "assistant", "content": "none"}]) is None


def test_is_conversation_cache_miss():
    assert NemotronAudioInLLMService._is_conversation_cache_miss(
        409,
        json.dumps({"error": {"type": "ConversationCacheMissError"}}),
    )
    assert not NemotronAudioInLLMService._is_conversation_cache_miss(
        409,
        json.dumps({"error": {"type": "DifferentError"}}),
    )
    assert not NemotronAudioInLLMService._is_conversation_cache_miss(
        500,
        json.dumps({"error": {"type": "ConversationCacheMissError"}}),
    )
    assert not NemotronAudioInLLMService._is_conversation_cache_miss(409, "not json")


def test_tools_schema_conversion():
    context = LLMContext([_user_message()], tools=ToolsSchemaForTest)
    payload, _ = _payload_from_context(
        _service(),
        context.get_messages(),
        context=context,
    )

    tools = payload["tools"]
    assert isinstance(tools, list)
    assert len(tools) == len(ToolsSchemaForTest.standard_tools)
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "end_session"
    assert "parameters" in tools[0]["function"]


def test_not_given_omitted_from_payload():
    context = LLMContext([_user_message()])

    payload, _ = _payload_from_context(
        _service(),
        context.get_messages(),
        context=context,
    )
    serialized = json.dumps(payload)

    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "NOT_GIVEN" not in serialized


def test_merge_tool_call_delta_accumulates():
    tool_calls_by_index = {}

    NemotronAudioInLLMService._merge_tool_call_delta(
        tool_calls_by_index,
        [
            {
                "index": 0,
                "id": "call_abc",
                "type": "function",
                "function": {"name": "submit_", "arguments": '{"name":"'},
            }
        ],
    )
    NemotronAudioInLLMService._merge_tool_call_delta(
        tool_calls_by_index,
        [
            {
                "index": 0,
                "function": {
                    "name": "dietary_request",
                    "arguments": 'Ada","dietary_preference":"vegan"}',
                },
            }
        ],
    )

    finalized = NemotronAudioInLLMService._finalize_tool_calls(tool_calls_by_index)

    assert finalized == [
        {
            "id": "call_abc",
            "type": "function",
            "function": {
                "name": "submit_dietary_request",
                "arguments": '{"name":"Ada","dietary_preference":"vegan"}',
            },
        }
    ]


def test_finalize_synthesizes_missing_ids():
    finalized = NemotronAudioInLLMService._finalize_tool_calls(
        {
            0: {"type": "function", "function": {"name": "first", "arguments": "{}"}},
            1: {"type": "function", "function": {"name": "second", "arguments": "{}"}},
        }
    )

    assert [tool_call["id"] for tool_call in finalized] == ["call_0", "call_1"]


def test_trace_json_value_redacts_audio_url():
    service = _service()
    digest = hashlib.sha256("AAAA".encode("ascii")).hexdigest()

    redacted = service._trace_json_value(
        {"audio_url": {"url": "data:audio/wav;base64,AAAA"}}
    )

    assert redacted == {
        "audio_url": {
            "url": f"<data-audio-base64 sha256={digest} chars=4>",
        }
    }


def test_trace_json_value_recurses():
    service = _service()
    value = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"audio_url": {"url": "https://example.test/audio.wav"}},
        ],
        "count": 2,
    }

    assert service._trace_json_value(value) == value


def test_write_trace_file_creates_sorted_json(tmp_path):
    service = _service()
    service._trace_dir = tmp_path

    service._write_trace_file(
        trace_id="trace-1",
        phase="request",
        payload={"z": 1, "a": 2},
    )

    trace_path = tmp_path / "trace-1.request.json"
    text = trace_path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    pairs = json.loads(text, object_pairs_hook=list)

    assert parsed["trace_id"] == "trace-1"
    assert parsed["phase"] == "request"
    assert parsed["a"] == 2
    assert parsed["z"] == 1
    assert [key for key, _ in pairs] == sorted(parsed.keys())


def test_count_audio_parts():
    messages = [
        _user_message("plain"),
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,A"}},
                {"type": "text", "text": "hello"},
                {"type": "input_audio", "input_audio": {"data": "B", "format": "wav"}},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,C"}}
            ],
        },
    ]

    assert _service()._count_audio_parts(messages) == 2


def test_no_conversation_logical_checkpoint_token_count_ever():
    messages = [_user_message()]
    cache_off, _ = _payload_from_context(_service(), messages)

    cache_on_service = _service(conversation_cache_enabled=True)
    cache_on_service._conversation_id = "abc"
    cache_on_full, _ = _payload_from_context(cache_on_service, messages)

    suffix_service = _service(conversation_cache_enabled=True, suffix_only_conversation=True)
    suffix_service._conversation_id = "abc"
    suffix_service._conversation_cache_committed = True
    suffix_only, _ = _payload_from_context(suffix_service, [_user_message("latest")])

    for payload in (cache_off, cache_on_full, suffix_only):
        assert "conversation_logical_checkpoint_token_count" not in payload


def test_tool_round_carries_results_into_next_suffix():
    service = _service(conversation_cache_enabled=True, suffix_only_conversation=True)
    tool_result = {"role": "tool", "tool_call_id": "call_0", "content": "{}"}
    service._pending_tool_result_messages = [tool_result]
    service._conversation_cache_committed = True
    service._conversation_id = "x"
    service._canonical_messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
    ]

    canonical = service._canonical_messages_from_context(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u2"}]
    )
    assert canonical is not None
    payload, full_messages = service._build_payload_from_messages(
        canonical,
        context=LLMContext(),
    )

    assert full_messages == [
        *service._canonical_messages,
        tool_result,
        {"role": "user", "content": "u2"},
    ]
    assert payload["messages"] == [tool_result, {"role": "user", "content": "u2"}]
    assert payload["conversation_require_cache"] is True


def test_disable_guard_removed():
    source = inspect.getsource(NemotronAudioInLLMService)

    assert "_tool_calls_handled_this_request" not in source
