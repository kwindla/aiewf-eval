"""Read-only selected-capture transport inventory; no provider calls."""
from pathlib import Path
from collections import Counter
import json,re,hashlib,subprocess,csv
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];COHORT=ROOT/'ops/judge-refresh-opus55-20261017/cohorts'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prefix(p):
 if not p.exists():return ''
 with p.open(errors='replace') as f:return f.read(50000)
def log_info(p):
 s=prefix(p);lines=s.splitlines()
 transport='responses-custom' if 'OpenAIResponsesLLMService' in s else 'chat-completions' if 'OpenAILLMService' in s else 'other' if s else 'no-adjacent-log'
 classes=sorted(set(re.findall(r'([A-Za-z_][A-Za-z_0-9]*(?:LLMService|LiveService))#',s)))
 evid=[{'line':i+1,'text':v} for i,v in enumerate(lines) if any(t in v for t in ['Configured ','Using xAI Responses','stateful_responses=','MTE_FILLER_DOTS active','Recovery nudges enabled=','Tool result run_llm enabled='])][:12]
 return {'path':str(p),'exists':p.exists(),'prefix_sha256':hashlib.sha256(s.encode()).hexdigest() if s else None,'transport':transport,'classes':classes,'startup_evidence':evid}
text=(ROOT/'README.md').read_text();ranked=[];voice=[]
for l in text.splitlines():
 if not l.startswith('| '):continue
 cells=[c.strip() for c in l.strip('|').split('|')]
 if len(cells)==11 and cells[0]!='Model':ranked.append(cells[0])
 if len(cells)==12 and cells[0]!='Model':voice.append(cells[0])
rows=[];candidates=[]
for p in sorted(COHORT.glob('*.json')):
 d=json.loads(p.read_text())
 if d['label'] not in ranked:
  aliases={'nemotron-3-nano-30b-512':'nemotron-3-nano-30b (512, NVFP4)','nemotron-3-super-120b-512':'nemotron-3-super-120b (512, NVFP4)'}
  if d['key'] in aliases:d['label']=aliases[d['key']]
  else:continue
 captures=[]
 for x in d['inputs']:
  source=Path(x['source']);info=log_info(source.parent/'run.log')
  captures.append({'capture_id':x['capture_id'],'transcript':x['source'],'transcript_sha256':x['transcript_sha256'],'captured_at':x.get('captured_at'),'generation_configuration':x.get('generation_configuration'),'log':info})
 counts=dict(Counter(c['log']['transport'] for c in captures));affected=counts.get('responses-custom',0)>0
 rr={'key':d['key'],'label':d['label'],'provider':d['provider'],'mode':'text','selected_complete':len(captures),'cohort_manifest':str(p.relative_to(ROOT)),'cohort_manifest_sha256':digest(p),'transport_counts':counts,'captures':captures,'rerun_candidate':affected,'classification':'used-defective-path-impact-uncertain' if affected else 'not-custom-responses-adapter','adapter_source_hash':None,'adapter_source_hash_limit':'No per-capture source snapshot or commit identity in these runtime.json files. Transport is established by actual capture logs; source revision is not inferred from capture date.'}
 if affected:
  model=d.get('matching_rule',{}).get('family',d.get('configuration',{}).get('model'));eff=d.get('matching_rule',{}).get('match',{}).get('reasoning_effort',[])
  env={'MTE_ENABLE_RECOVERY':'1','MTE_DEDUPE_TOOL_CALLS':'1','MTE_TOOL_RESULT_RUN_LLM':'0','MTE_TEXT_IDLE_TIMEOUT_SECS':'180','MTE_OPENAI_SERVICE_TIER':'priority'}
  if eff:env['MTE_OPENAI_RESPONSES_REASONING_EFFORT']=eff[0]
  if 'dots96' in d['key']:env.update(MTE_FILLER_DOTS='96',MTE_FILLER_TOKEN='.',MTE_FILLER_POSITION='suffix')
  rr['reproduction']={'model':model,'service':'openai','pipeline':'text','benchmark':'aiwf_medium_context','endpoint':'https://api.openai.com/v1','assigned':30,'eligibility':'first_valid_response; retain early endings and fixed 900 denominator','environment':env,'sampling':'provider default; no added temperature/top_p','output_token_cap':'unset explicitly in fresh campaign','service_tier_note':'Fresh protocol explicitly requests priority. Historical unlogged tiers remain unknown; recorded fast/priority request+echo variation is preserved below.','historic_effort_values':sorted({v for c in captures for v in (c['generation_configuration'] or {}).get('reasoning_effort',[])}),'historic_service_tier_log_values':sorted({v for c in captures for v in (c['generation_configuration'] or {}).get('service_tier',[])}),'historic_filler_values':{json.dumps(v) for c in captures for v in (c['generation_configuration'] or {}).get('filler',[])}}
  rr['reproduction']['historic_filler_values']=sorted(rr['reproduction']['historic_filler_values']);rr['stateful_responses_enabled']=False
  rr['stateful_evidence_limit']='Historical snapshots of this adapter are stateless; exact per-capture adapter bytes are unavailable. No stateful configuration/log marker appears in these OpenAI captures.'
  candidates.append(rr['key'])
 rows.append(rr)
# New Luna has zero complete conversations and is outside the ranked table.
cfgpath=ROOT/'ops/recent-models-20261018/luna/configuration.json';cfg=json.loads(cfgpath.read_text());audit=json.loads((ROOT/'ops/recent-models-20261018/luna/completion-audit.json').read_text())
for arm in ['none','low']:
 cs=[c for c in audit['cases'] if c['arm']==arm]
 rr={'key':'luna-'+arm,'label':'gpt-6-luna ('+arm+')','provider':'OpenAI','mode':'text','assigned':30,'selected_complete':0,'classification':'confirmed-history-loss-and-premature-endings','causal_limit':'Controlled none-mode replay is 6 bad flattened versus 6 relevant native-output answers; low-specific causal mechanism and repaired full-conversation completion not yet measured.','rerun_candidate':True,'transport_counts':{'responses-custom':30},'stateful_responses_enabled':False,'configuration':str(cfgpath.relative_to(ROOT)),'adapter_source_hash':'edb632989dfb768c53e08a75eab75aef39659e34e95fe2282ab2e6dd1e0e8b48','adapter_snapshot':'ops/recent-models-20261018/runtime-snapshot/src/multi_turn_eval/services/openai_responses.py','captures':[{'capture_id':Path(c['source']).parent.name,'transcript':c['source'],'transcript_sha256':c['transcript_sha256'],'observed_turns':c['observed_scripted_turns'],'log':log_info(ROOT/Path(c['source']).parent/'run.log')} for c in cs],'reproduction':{'model':'gpt-6-luna','service':'openai','pipeline':'text','benchmark':'aiwf_medium_context','endpoint':cfg['endpoint']['url'],'assigned':30,'eligibility':'first_valid_response; retain early endings and fixed 900 denominator','environment':{**cfg['common_environment'],**cfg['arms'][arm]['environment']},'configuration':str(cfgpath.relative_to(ROOT))}}
 rows.append(rr);candidates.append(rr['key'])
versions=[]
for commit in subprocess.check_output(['git','log','--format=%h','--','src/multi_turn_eval/services/openai_responses.py'],cwd=ROOT,text=True).splitlines():
 s=subprocess.check_output(['git','show',commit+':src/multi_turn_eval/services/openai_responses.py'],cwd=ROOT)
 versions.append({'commit':commit,'sha256':hashlib.sha256(s).hexdigest(),'stateful_flag_available':b'stateful_responses_enabled' in s,'phase_key_present':b'"phase"' in s,'encrypted_content_key_present':b'encrypted_content' in s,'limitation':'Repository revision comparison only; not assigned to historical captures without a source manifest.'})
# Current recent Anthropic/Meta rows and fresh Ultra are separate prospective captures.
recent=json.loads((ROOT/'ops/recent-models-20261018/aggregate.json').read_text())
for item in recent['text_models']:
 if item['name']=='luna':continue
 configpath=ROOT/'ops/recent-models-20261018'/item['name']/'configuration.json';config=json.loads(configpath.read_text());canon=list(csv.DictReader((configpath.parent/'artifacts/canonical.tsv').open(),delimiter='\t'))
 captures=[]
 for cr in canon:
  if cr['arm']!=item['arm']:continue
  transcript=ROOT/cr['run_dir']/'transcript.jsonl'
  captures.append({'capture_id':Path(cr['run_dir']).name,'transcript':str(transcript),'transcript_sha256':digest(transcript),'log':log_info(transcript.parent/'run.log')})
 rows.append({'key':item['key'],'label':item['label'],'provider':item['provider'],'mode':'text','assigned':item['assigned'],'selected_complete':item['complete'],'classification':'not-custom-responses-adapter','rerun_candidate':False,'configuration':str(configpath.relative_to(ROOT)),'transport_counts':dict(Counter(c['log']['transport'] for c in captures)),'captures':captures})
ultrahome=ROOT/'ops/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018';ultracanon=list(csv.DictReader((ultrahome/'artifacts/canonical.tsv').open(),delimiter='\t'));ultracaptures=[]
for cr in ultracanon:
 if cr['arm']!='on':continue
 transcript=ROOT/cr['run_dir']/'transcript.jsonl'
 ultracaptures.append({'capture_id':Path(cr['run_dir']).name,'transcript':str(transcript),'transcript_sha256':digest(transcript),'log':log_info(transcript.parent/'run.log')})
rows.append({'key':'ultra-on','label':'nemotron-3-ultra (thinking on)','provider':'Baseten','mode':'text','selected_complete':30,'classification':'not-custom-responses-adapter','rerun_candidate':False,'configuration':str((ultrahome/'configuration.json').relative_to(ROOT)),'transport_counts':dict(Counter(c['log']['transport'] for c in ultracaptures)),'captures':ultracaptures})

# Preserve archived xAI status without reintroducing its retired row.
xpath=ROOT/'ops/xai-grok46/configuration.json';xcfg=json.loads(xpath.read_text())
xai={'key':'grok-4.6-low','current_readme':False,'status':'explicitly archived by user; not counted in current row reruns','configuration':str(xpath.relative_to(ROOT)),'source_manifest':'ops/xai-grok46/artifacts/source-sha256.txt','adapter_source_hash':'09044f551e57e176f719ba490b95a28ffbb532d5303132ffe4fec25c5a948423','adapter_snapshot_bytes_located':False,'stateful_responses_enabled':True,'provenance_basis':'Frozen configuration serving verification and required log needles explicitly record stateful Responses previous_response_id continuation.','reasoning_effort':'low','endpoint':xcfg['endpoint']['url'],'provider':'xAI','parallel_tool_cursor_risk':'Existing stateful helper slices after latest assistant; multiple assistant-call/result pairs can discard earlier results. Actual triggering in archived Grok cohort not proven; raw canonical run paths are no longer at the manifest locations.','source_limit':'Source SHA is known but exact matching source bytes have not been located; do not assign current snapshot as the historical adapter.'}
voice_profiles=[]
for p in sorted(COHORT.glob('voice-*.json')):
 d=json.loads(p.read_text());live='live_' in d['key']
 voice_profiles.append({'key':d['key'],'label':d['label'],'main_table':d['label'] in voice,'additional_experiment':d['key']=='voice-gemini38medium_new','classification':'not-custom-responses-adapter','rerun_candidate':False,'cohort_manifest':str(p.relative_to(ROOT)),'selected_complete':len(d['inputs']),'captures':[{'capture_id':x['capture_id'],'capture_path':x['capture_path'],'original_quality_input_source':x['original_quality_input_source'],'transcript_sha256':x['transcript_sha256']} for x in d['inputs']],'transport_note':'Upstream OpenAILiveLLMService with ResponsesDelegation. Uses the Responses endpoint but not this project custom adapter.' if live else 'Dedicated speech/realtime transport, not the project text Responses adapter.'})
voice_profiles.append({'key':'grok-voice-none','label':'Grok Voice 3.0 (reasoning off)','main_table':True,'classification':'not-custom-responses-adapter','rerun_candidate':False,'selected_complete':25,'configuration':'ops/recent-models-20261018/grok-voice/configuration.json','transport_note':'Dedicated XAIRealtimeLLMService over realtime WebSocket; no custom text Responses adapter.'})
out={'scope':'Current README text rows and unranked Luna configurations; archived xAI noted separately; other endpoint users distinguished from this custom adapter.','readme_sha256':digest(ROOT/'README.md'),'ranked_text_rows':len(ranked),'ranked_text_labels':ranked,'voice_labels':voice,'voice_profiles':voice_profiles,'rerun_candidate_keys':candidates,'rerun_candidate_count':len(candidates),'fresh_assigned_total':30*len(candidates),'selected_older_responses_captures':sum(r['transport_counts'].get('responses-custom',0) for r in rows if not r['key'].startswith('luna-')),'rows':rows,'repository_adapter_versions':versions,'archived_xai':xai,'existing_evidence_unchanged':True}
(HERE/'impact-inventory.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k in {'ranked_text_rows','rerun_candidate_count','fresh_assigned_total','selected_older_responses_captures','rerun_candidate_keys'}},indent=2))
