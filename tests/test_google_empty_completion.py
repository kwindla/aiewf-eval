"""The empty-completion diagnostic must surface the server-side verdict."""

import asyncio
from types import SimpleNamespace

from loguru import logger

from multi_turn_eval.services.google_logged import LoggedGoogleLLMService


def _usage(prompt=13482, candidates=0, thoughts=0):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=prompt + candidates,
        cached_content_token_count=0,
        thoughts_token_count=thoughts,
    )


def _service(chunks):
    service = object.__new__(LoggedGoogleLLMService)

    async def _async_noop(*args, **kwargs):
        return None

    async def _stream_content(context):
        async def _gen():
            for chunk in chunks:
                yield chunk

        return _gen()

    service._name = "LoggedGoogleLLMService#test"
    service.push_frame = _async_noop
    service.push_error = _async_noop
    service.start_ttfb_metrics = _async_noop
    service.stop_ttfb_metrics = _async_noop
    service.run_function_calls = _async_noop
    service.start_llm_usage_metrics = _async_noop
    service._stream_content = _stream_content
    return service


def _run_and_capture(service):
    records = []
    sink_id = logger.add(lambda message: records.append(str(message)), level="WARNING")
    try:
        asyncio.run(service._process_context(context=SimpleNamespace()))
    finally:
        logger.remove(sink_id)
    return records


def test_candidate_less_stream_logs_empty_completion_diagnostics():
    chunk = SimpleNamespace(
        usage_metadata=_usage(),
        prompt_feedback="BLOCK_REASON_PROBE",
        candidates=[],
    )
    records = _run_and_capture(_service([chunk]))

    matched = [r for r in records if "Empty completion from Gemini" in r]
    assert matched, records
    assert "BLOCK_REASON_PROBE" in matched[0]
    assert "candidates_seen=False" in matched[0]


def test_empty_candidate_stream_logs_finish_reason():
    chunk = SimpleNamespace(
        usage_metadata=_usage(),
        prompt_feedback=None,
        candidates=[
            SimpleNamespace(
                finish_reason="MALFORMED_FUNCTION_CALL",
                content=None,
                grounding_metadata=None,
            )
        ],
    )
    records = _run_and_capture(_service([chunk]))

    matched = [r for r in records if "Empty completion from Gemini" in r]
    assert matched, records
    assert "MALFORMED_FUNCTION_CALL" in matched[0]


def test_normal_text_stream_stays_quiet():
    chunk = SimpleNamespace(
        usage_metadata=_usage(candidates=12),
        prompt_feedback=None,
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text="Hello there!",
                            thought=None,
                            function_call=None,
                            inline_data=None,
                            thought_signature=None,
                        )
                    ]
                ),
                grounding_metadata=None,
            )
        ],
    )
    records = _run_and_capture(_service([chunk]))

    assert not [r for r in records if "Empty completion from Gemini" in r], records


def test_ttfat_timer_restarts_for_every_response_and_waits_for_visible_output():
    events = []
    service = _service([])

    def chunk(text, thought):
        return SimpleNamespace(
            usage_metadata=None,
            prompt_feedback=None,
            candidates=[SimpleNamespace(
                finish_reason=None,
                grounding_metadata=None,
                content=SimpleNamespace(parts=[SimpleNamespace(
                    text=text, thought=thought, function_call=None,
                    inline_data=None, thought_signature=None,
                )]),
            )],
        )

    async def start():
        events.append("start")

    async def stop():
        events.append("visible")

    async def stream(context):
        assert events[-1] == "start"
        async def chunks():
            yield chunk("Think before answering", True)
            assert events[-1] == "start"  # Thought output must not stop TTFAT.
            yield chunk("Here is the answer", False)
        return chunks()

    service.start_ttfb_metrics = start
    service.stop_ttfb_metrics = stop
    service._stream_content = stream

    async def run():
        await service._process_context(SimpleNamespace())
        await service._process_context(SimpleNamespace())

    asyncio.run(run())
    assert events == ["start", "visible", "start", "visible"]
