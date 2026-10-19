"""A provider interruption must never seed a new request with flattened output."""
import json
import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from openai import NOT_GIVEN
from multi_turn_eval.services.openai_responses import OpenAIResponsesLLMService
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('completed_before_error', [False, True])
def test_changed_context_after_provider_interruption_is_rejected(completed_before_error):
    service=OpenAIResponsesLLMService(model='gpt-6-luna',api_key='test',stateful_responses_enabled=True)
    messages=[{'role':'user','content':'First'}]
    service._context_to_openai_params=lambda context:{'messages':messages}
    if completed_before_error:
        service._responses_request_params(SimpleNamespace())
        service._remember_completed_response({'id':'resp-1','output':[{'type':'message','role':'assistant','phase':'final_answer','content':[{'type':'output_text','text':'Complete'}]}]})
        messages.extend([{'role':'assistant','content':'Complete'},{'role':'user','content':'Second'}])
    request=service._responses_request_params(SimpleNamespace())
    assert service._responses_request_params(SimpleNamespace())==request
    messages.extend([{'role':'assistant','content':'Partial provider text'},{'role':'user','content':'Next'}])
    with pytest.raises(RuntimeError,match='unfinished Responses request'):
        service._responses_request_params(SimpleNamespace())


@pytest.mark.parametrize('existing_include', [None, ['message.output_text.logprobs'], ['reasoning.encrypted_content']])
def test_unstored_reasoning_requests_encrypted_replay_without_mutating_settings(existing_include):
    service=OpenAIResponsesLLMService(model='gpt-6-luna',api_key='test',stateful_responses_enabled=True)
    extra={'store':False}
    if existing_include is not None:extra['include']=deepcopy(existing_include)
    before=deepcopy(extra);original=service._setting
    service._setting=lambda key,default=NOT_GIVEN:extra if key=='extra' else original(key,default)
    messages=[{'role':'user','content':'First'}]
    service._context_to_openai_params=lambda context:{'messages':messages}
    first=service._responses_request_params(SimpleNamespace())
    assert first['include'].count('reasoning.encrypted_content')==1
    assert set(existing_include or [])<=set(first['include']) and extra==before
    reasoning={'type':'reasoning','id':'rs-private','summary':[],'encrypted_content':'encrypted-replay'}
    service._remember_completed_response({'id':'resp-unstored','status':'completed','output':[reasoning]})
    messages.append({'role':'user','content':'Second'})
    second=service._responses_request_params(SimpleNamespace())
    assert 'previous_response_id' not in second and reasoning in second['input']
    assert second['include'].count('reasoning.encrypted_content')==1 and extra==before


@pytest.mark.parametrize('status', ['incomplete', 'failed'])
def test_nonstreaming_failure_preserves_completed_ledger_and_blocks_changed_context(status):
    service=OpenAIResponsesLLMService(model='gpt-6-luna',api_key='test',stateful_responses_enabled=True)
    messages=[{'role':'user','content':'First'}]
    service._context_to_openai_params=lambda context:{'messages':messages}
    service._responses_request_params(SimpleNamespace())
    service._remember_completed_response({'id':'resp-1','status':'completed','output':[
        {'type':'message','role':'assistant','phase':'final_answer','content':[{'type':'output_text','text':'Complete'}]}]})
    before=deepcopy(service._native_history)
    messages.extend([{'role':'assistant','content':'Complete'},{'role':'user','content':'Second'}])
    async def create(**kwargs):
        return SimpleNamespace(id='resp-unfinished',status=status,output=[])
    service._client=SimpleNamespace(responses=SimpleNamespace(create=create))
    with pytest.raises(RuntimeError,match='unfinished Responses result'):
        asyncio.run(service.run_inference(SimpleNamespace()))
    assert service._native_history==before and service._previous_response_id=='resp-1'
    messages.append({'role':'user','content':'Third'})
    with pytest.raises(RuntimeError,match='unfinished Responses request'):
        service._responses_request_params(SimpleNamespace())
