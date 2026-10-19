"""Register the approved low-reasoning model matrix; no network calls."""
import json, random
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SPECS=[
 ('luna','gpt-6-luna','openai','https://api.openai.com/v1','OPENAI_API_KEY',{'none':{'MTE_OPENAI_RESPONSES_REASONING_EFFORT':'none'},'low':{'MTE_OPENAI_RESPONSES_REASONING_EFFORT':'low'}}),
 ('sonnet','claude-sonnet-5-5','anthropic','https://api.anthropic.com','ANTHROPIC_API_KEY',{'between-tools':{'MTE_ANTHROPIC_THINKING':'between_tools'},'low':{'MTE_ANTHROPIC_THINKING':'adaptive'}}),
 ('haiku','claude-haiku-5-5','anthropic','https://api.anthropic.com','ANTHROPIC_API_KEY',{'off':{'MTE_ANTHROPIC_THINKING':'disabled'},'low':{'MTE_ANTHROPIC_THINKING':'adaptive'}}),
 ('spark13','meta/muse-spark-1.3','openrouter','https://openrouter.ai/api/v1','OPENROUTER_API_KEY',{'minimal':{'MTE_OPENROUTER_REASONING_EFFORT':'minimal'},'low':{'MTE_OPENROUTER_REASONING_EFFORT':'low'}}),
 ('spark-open','meta-models/muse-spark-open','baseten','https://inference.baseten.co/v1','BASETEN_API_KEY',{'low':{'MTE_BASETEN_REASONING_EFFORT':'low'}}),
 ('grok-voice','grok-voice-3.0','openai-realtime','wss://api.x.ai/v1/realtime','XAI_API_KEY',{'none':{'MTE_XAI_VOICE_REASONING_EFFORT':'none'}}),
]
SOURCE=['pyproject.toml','uv.lock','src/multi_turn_eval/pipelines/base.py','src/multi_turn_eval/pipelines/text.py','src/multi_turn_eval/pipelines/realtime.py','src/multi_turn_eval/pipelines/grok_realtime.py','src/multi_turn_eval/services/openai_responses.py','src/multi_turn_eval/services/anthropic_logged.py','src/multi_turn_eval/services/openrouter_logged.py','src/multi_turn_eval/services/lilac_logged.py','src/multi_turn_eval/services/baseten_logged.py','src/multi_turn_eval/recording/transcript_recorder.py','benchmarks/_shared/turns.py','benchmarks/_shared/tools.py','benchmarks/aiwf_medium_context/prompts/system.py','benchmarks/aiwf_medium_context/data/knowledge_base.txt','ops/recent-models-20261018/PROTOCOL.md']

def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')

def main():
 for name,model,service,url,key,arms in SPECS:
  directory=HERE/name;directory.mkdir(exist_ok=True)
  voice=name=='grok-voice'
  common={'MTE_ENABLE_RECOVERY':'0' if voice else '1','MTE_DEDUPE_TOOL_CALLS':'1','MTE_TOOL_RESULT_RUN_LLM':'0','MTE_TEXT_IDLE_TIMEOUT_SECS':'180'}
  if service=='anthropic':common.update(MTE_ANTHROPIC_EFFORT='low',MTE_ANTHROPIC_THINKING_DISPLAY='omitted',MTE_ANTHROPIC_MAX_TOKENS='16384')
  if service=='openai':common.update(MTE_OPENAI_SERVICE_TIER='priority')
  if service=='openrouter':common.update(MTE_OPENROUTER_PROVIDER='Meta',MTE_OPENROUTER_MAX_TOKENS='16384')
  if service=='baseten':common.update(MTE_BASETEN_MAX_TOKENS='16384',MTE_BASETEN_TEMPERATURE='1.0',MTE_BASETEN_TOP_P='1.0',MTE_BASETEN_MIN_REQUEST_INTERVAL_SECS='3',MTE_BASETEN_RATE_LIMIT_RETRIES='3',MTE_BASETEN_RATE_LIMIT_BACKOFF_SECS='10')
  config={'schema_version':1,'campaign_id':'recent-models-20261018-'+name,'benchmark':'aiwf_medium_context_speech_v2' if voice else 'aiwf_medium_context','model':model,'accepted_response_model_ids':[model],'service':service,'pipeline':'grok-realtime' if voice else 'text','endpoint':{'url':url,'request_env':'BASETEN_BASE_URL' if service=='baseten' else 'MTE_ENDPOINT_PROVENANCE'},'credential':{'request_env':key,'source_env':key,'source_file':'/home/khkramer/src/gb-benchmarks/.env' if service=='baseten' else '/home/khkramer/src/aiewf-eval/.env','source_file_key':key},'paths':{'campaign_artifact_dir':str((directory/'artifacts').relative_to(ROOT)),'run_output_root':'runs/recent-models-20261018/'+name},'schedule_path':str((directory/'schedule.tsv').relative_to(ROOT)),'source_integrity_paths':SOURCE,'target_eligible_runs':25 if voice else 30*len(arms),'fixed_scheduled_turns_per_conversation':30,'serving':{'verified':False},'eligibility':{'policy':'first_valid_response','missing_future_turns':'retain_as_fixed_denominator_failures'},'collection':{'provider_endpoint_concurrency':1,'max_attempts_per_slot_default':3,'timeout_seconds_default':1800},'unset_environment_prefixes':['MTE_'],'unset_environment':[],'common_environment':common,'arms':{}}
  for arm,env in arms.items():
   needles=[]
   if service=='openai':needles=['reasoning.effort='+arm+',']
   if service=='anthropic':needles=["'type': '"+env['MTE_ANTHROPIC_THINKING']+"'",'effort=low,']
   if service=='openrouter':needles=['reasoning_effort='+arm+',','provider=Meta']
   if service=='baseten':needles=['reasoning_effort=low,']
   if voice:needles=['session reasoning.effort=none']
   config['arms'][arm]={'environment':env,'provenance_log_needles':needles,'forbidden_log_needles':['MTE_FILLER_DOTS active']}
  if not (directory/'configuration.json').exists():dump(directory/'configuration.json',config)
  order=[];rng=random.Random(20261018)
  for i in range(25 if voice else 30):
   pair=list(arms);rng.shuffle(pair);order.extend(pair)
  if not (directory/'schedule.tsv').exists():(directory/'schedule.tsv').write_text('slot\tarm\n'+''.join(f'{i}\t{a}\n' for i,a in enumerate(order,1)))
 print('Registered 9 text configurations (270 slots) and 1 voice configuration (25 complete target)')
if __name__=='__main__':main()
