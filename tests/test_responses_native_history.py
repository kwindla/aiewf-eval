"""Regression coverage for provider-native history and external input boundaries."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from multi_turn_eval.services.openai_responses import OpenAIResponsesLLMService


def setup_service(stateful=False, extra=None):
    service = OpenAIResponsesLLMService(model='gpt-6-luna', api_key='test', stateful_responses_enabled=stateful)
    messages = [{'role': 'system', 'content': 'Help.'}, {'role': 'user', 'content': 'First'}]
    service._context_to_openai_params = lambda context: {'messages': messages}
    if extra is not None:
        original = service._setting
        service._setting = lambda key, default=None: extra if key == 'extra' else original(key, default)
    return service, messages


def request(service):
    return service._responses_request_params(SimpleNamespace())


def complete(service, output, id='resp-1'):
    service._remember_completed_response({'id': id, 'output': output})


def message(text, phase):
    return {'id': 'msg-' + phase, 'type': 'message', 'role': 'assistant', 'phase': phase,
            'status': 'completed', 'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}


def call(index):
    return {'type': 'function_call', 'id': f'fc-{index}', 'call_id': f'call-{index}',
            'name': 'vote', 'arguments': '{}', 'status': 'completed'}


def append_call_result(messages, index):
    messages.extend([
        {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'id': f'call-{index}',
         'function': {'name': 'vote', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': f'call-{index}', 'content': f'result-{index}'},
    ])


def test_manual_history_preserves_all_native_output_items_and_boundaries():
    service, messages = setup_service()
    initial = request(service)['input']
    output = [
        {'type': 'reasoning', 'id': 'rs-1', 'summary': [], 'encrypted_content': 'encrypted'},
        message('Commentary', 'commentary'), message('', 'final_answer'),
    ]
    frozen = deepcopy(output)
    complete(service, output)
    output[1]['phase'] = 'mutated'
    messages.extend([{'role': 'assistant', 'content': 'Commentary'}, {'role': 'user', 'content': 'Second'}])
    outgoing = request(service)
    assert outgoing['input'][:-1] == initial + frozen
    assert outgoing['input'][-1]['content'][0]['text'] == 'Second'
    assert 'previous_response_id' not in outgoing


@pytest.mark.parametrize('stateful', [False, True])
def test_parallel_results_survive_interleaved_assistant_messages_and_repeat_cycles(stateful):
    service, messages = setup_service(stateful)
    initial = request(service)['input']
    complete(service, [call(1), call(2)])
    append_call_result(messages, 1)
    append_call_result(messages, 2)
    expected = [{'type': 'function_call_output', 'call_id': f'call-{i}', 'output': f'result-{i}'} for i in [1, 2]]
    outgoing = request(service)
    assert outgoing['input'] == (expected if stateful else initial + [call(1), call(2)] + expected)
    assert outgoing.get('previous_response_id') == ('resp-1' if stateful else None)
    complete(service, [message('Done', 'final_answer')], 'resp-2')
    messages.append({'role': 'assistant', 'content': 'Done'})
    # Routine sanitizer removes empty assistant stubs before the next request.
    messages[:] = [m for m in messages if not (m['role'] == 'assistant' and not m.get('content'))]
    messages.append({'role': 'user', 'content': 'Third'})
    outgoing = request(service)
    if stateful:
        assert outgoing['previous_response_id'] == 'resp-2'
        assert len(outgoing['input']) == 1
    else:
        assert [i['call_id'] for i in outgoing['input'] if i['type'] == 'function_call_output'] == ['call-1', 'call-2']
        assert outgoing['input'][-2] == message('Done', 'final_answer')


def test_filler_uses_native_manual_history_and_keeps_earlier_users_clean(monkeypatch):
    monkeypatch.setenv('MTE_FILLER_DOTS', '3')
    service, messages = setup_service(True)
    assert request(service)['input'][-1]['content'][0]['text'] == 'First . . .'
    complete(service, [message('Hello', 'final_answer')])
    messages.extend([{'role': 'assistant', 'content': 'Hello'}, {'role': 'user', 'content': 'Second'}])
    outgoing = request(service)
    assert 'previous_response_id' not in outgoing
    assert outgoing['input'][1]['content'][0]['text'] == 'First'
    assert outgoing['input'][-1]['content'][0]['text'] == 'Second . . .'
    assert messages[-1]['content'] == 'Second'
    complete(service, [message('Again', 'final_answer')], 'resp-2')
    monkeypatch.delenv('MTE_FILLER_DOTS')
    messages.extend([{'role': 'assistant', 'content': 'Again'}, {'role': 'user', 'content': 'Third'}])
    assert 'previous_response_id' not in request(service)  # Flush server-retained filler.
    complete(service, [message('Clean', 'final_answer')], 'resp-3')
    messages.extend([{'role': 'assistant', 'content': 'Clean'}, {'role': 'user', 'content': 'Fourth'}])
    assert request(service)['previous_response_id'] == 'resp-3'


def test_store_false_replays_native_outputs():
    service, messages = setup_service(True, {'store': False})
    request(service)
    complete(service, [message('Hello', 'final_answer')])
    messages.extend([{'role': 'assistant', 'content': 'Hello'}, {'role': 'user', 'content': 'Second'}])
    outgoing = request(service)
    assert 'previous_response_id' not in outgoing
    assert outgoing['store'] is False
    assert outgoing['input'][-2] == message('Hello', 'final_answer')


def test_rewritten_external_history_fails_clearly_and_new_conversation_resets():
    service, messages = setup_service(True)
    request(service)
    complete(service, [message('Hello', 'final_answer')])
    messages.extend([{'role': 'assistant', 'content': 'Hello'}, {'role': 'user', 'content': 'Second'}])
    messages[1]['content'] = 'Rewritten'
    with pytest.raises(ValueError, match='external history was rewritten'):
        request(service)
    messages[:] = [{'role': 'user', 'content': 'New conversation'}]
    outgoing = request(service)
    assert 'previous_response_id' not in outgoing
    assert len(outgoing['input']) == 1


def test_uncompleted_request_does_not_advance_history_or_cloud_pointer():
    service, messages = setup_service(True)
    first = request(service)
    assert request(service) == first
    assert service._native_history == []
    assert service._previous_response_id is None


@pytest.mark.parametrize('terminal', ['response.failed', 'response.error', 'response.incomplete', 'error'])
def test_explicit_terminal_failure_never_executes_completed_tool_or_commits_history(terminal):
    import asyncio
    from unittest.mock import AsyncMock

    async def run():
        service, messages = setup_service(True)
        item = SimpleNamespace(**call(1))
        events = [SimpleNamespace(type='response.output_item.done', output_index=0, item=item),
                  SimpleNamespace(type=terminal)]
        class Stream:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def __aiter__(self): return self
            async def __anext__(self):
                if not events: raise StopAsyncIteration
                return events.pop(0)
        service._client = SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs: Stream()))
        for name in ['start_ttfb_metrics', 'stop_ttfb_metrics', 'push_frame', 'push_error', 'run_function_calls']:
            setattr(service, name, AsyncMock())
        await service._process_context(SimpleNamespace())
        service.push_error.assert_awaited_once()
        service.run_function_calls.assert_not_awaited()
        assert service._native_history == []
        assert service._previous_response_id is None
    asyncio.run(run())


def test_nonstored_response_is_never_used_as_a_cloud_parent():
    extra = {'store': False}
    service, messages = setup_service(True, extra)
    request(service)
    complete(service, [message('Hello', 'final_answer')])
    extra['store'] = True
    messages.extend([{'role': 'assistant', 'content': 'Hello'}, {'role': 'user', 'content': 'Second'}])
    outgoing = request(service)
    assert 'previous_response_id' not in outgoing
    assert outgoing['input'][-2] == message('Hello', 'final_answer')


def test_seeded_native_items_are_not_flattened():
    service, messages = setup_service()
    seeded = [message('Progress', 'commentary'), message('', 'final_answer'),
              {'type': 'reasoning', 'id': 'rs-1', 'summary': [], 'encrypted_content': 'encrypted'}]
    messages.extend(seeded)
    assert request(service)['input'][-3:] == seeded
