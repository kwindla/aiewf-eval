"""Recalculate quality and latency from every fresh capture and exact grade."""
import importlib.util,json,sys
from common import HERE,ROOT,NAMES,cfg,collect,dump,sha
import judge as driver
sys.path.insert(0,str(ROOT/'ops/recent-models-20261018'))
spec=importlib.util.spec_from_file_location('repair_analysis_helpers',ROOT/'ops/recent-models-20261018/analyze.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
from verify_capture_sources import captured_sources
sys.path.pop(0)
base=driver.base

def analyze():
 driver.verify()
 assert (HERE/'judging/COMPLETE.json').exists()
 with captured_sources(collect,HERE) as resolutions:cases=base.freeze()
 dump(HERE/'capture-source-resolution.json',{'all_captured_sources_verified':True,'archived_versions_used':resolutions})
 grades={};records={};complete=set();cost=0
 for c in cases:
  raw=(ROOT/c['source']).read_bytes();assert base.sha(raw)==c['transcript_sha256']
  rows=[json.loads(line) for line in raw.splitlines() if line.strip()];records[c['job_id']]=rows
  scheduled=[r for r in rows if not r.get('recovery_turn')]
  if len(scheduled)==30 and all(r.get('assistant_text','').strip() or r.get('tool_calls') for r in scheduled):complete.add(c['job_id'])
  assert base.valid(c),c['job_id']
  grade=json.loads(base.result_path(c).read_text());grades[c['job_id']]=grade;cost+=grade['metadata'].get('judge_cost_usd',0) or 0
 inventory=json.loads((HERE/'impact-inventory.json').read_text());labels={r['key']:r['label'] for r in inventory['rows']}
 models=[]
 for name in NAMES:
  selected=[c for c in cases if c['name']==name];done=[c for c in selected if c['job_id'] in complete]
  assigned_score=helpers.score(selected,grades);completed_score=helpers.score(done,grades)
  logs=[((ROOT/c['source']).parent/'run.log').read_text() for c in selected]
  model={'key':name,'name':name,'arm':'standard','label':labels[name],'provider':'OpenAI','assigned':len(selected),'complete':len(done),'incomplete':len(selected)-len(done),'shortfall':30-len(done),
         'all_assigned':assigned_score,'complete_only':completed_score,'strict_percent':completed_score['strict_percent'],'strict_ci95':completed_score['strict_ci95'],
         'dimension_error_percent':completed_score['dimension_error_percent'],'latency':helpers.latency([r for c in done for r in records[c['job_id']]]),
         'all_attempt_latency':helpers.latency([r for c in selected for r in records[c['job_id']]]),'cases':[c['job_id'] for c in selected],
         'observed_scripted_turns':sum(len(c['observed_turns']) for c in selected),'missing_future_turns':30*len(selected)-sum(len(c['observed_turns']) for c in selected),
         'requested_service_tier':'priority','effective_service_tiers':sorted({r['service_tier'] for c in selected for r in map(json.loads,((ROOT/c['source']).parent/'responses-history-audit.jsonl').read_text().splitlines()) if r['event']=='completed' and r.get('service_tier')})}
  assert model['assigned']==30
  models.append(model)
 events=[json.loads(line) for line in (HERE/'judging/attempts.jsonl').read_text().splitlines() if line.strip()]
 data={'at':collect.utc_now(),'judge_model':base.MODEL,'judge_effort':base.EFFORT,'all_quality_complete':len(grades)==len(cases)==360,'judged_conversations':len(grades),'successful_judge_cost_usd':cost,'judge_credit_cooldowns':sum(e['event']=='credit_cooldown' for e in events),'text_models':models,'voice_models':[],'cases':cases}
 dump(HERE/'aggregate.json',data)
 report='# Responses adapter repair and prospective refresh\n\n'
 report+=helpers.table(['Configuration','Complete / assigned','Fixed-900 pass','Completed-only pass','Completed TTFAT P50','P95','Effective tier'],[[m['label'],f"{m['complete']}/30",helpers.pct(m['all_assigned']['strict_percent']),helpers.pct(m['strict_percent']),helpers.ms(m['latency']['positive']['p50_ms']),helpers.ms(m['latency']['positive']['p95_ms']),', '.join(m['effective_service_tiers'])] for m in models])+'\n\n'
 report+='Terminal unanswered requests canceled at the registered idle timeout remain incomplete assigned conversations; they are not retried toward completion. Their earlier native histories remain valid.\n\n'
 report+='The fixed denominator retains early endings and counts missing future turns as failures. The README reports conditional accuracy and latency among complete conversations; this report also records completion reliability. Each row received 30 fixed assignments, without topping up incomplete conversations. Latency excludes scripted turn zero and includes recovery responses, matching the existing analysis. Conversation-cluster bootstrap intervals and all component counts are in [aggregate.json](aggregate.json).\n\n'
 report+='The adapter now preserves native phases, output boundaries, empty final answers and reasoning. Ordinary requests link previous_response_id; the dots arm replays native outputs so historical user messages remain filler-free. A request-boundary cursor preserves every parallel tool result. Explicit incomplete/error streams do not execute pending tools. This repair does not guarantee completion: the excluded none smoke still ended early, and exact native replay sometimes reproduced the early ending. A retrieval audit confirmed that the failed cloud request retained every prior native output, including commentary and the empty final-answer boundary. See [protocol](PROTOCOL.md), [smokes](luna-low/SMOKE.json), and [excluded native/cloud diagnostic](none-diagnostic.log).\n\n'
 replacements=json.loads((HERE/'OPERATIONAL-REPLACEMENTS.json').read_text())
 report+=f"Operational integrity: {len(replacements['replacements'])} assignments were retried after positive evidence of provider stream interruption. Original raw captures, original portable canonical manifests and original COLLECTED manifests remain unchanged in the audit archive. Replacements use the identical frozen runtime/settings and retain early model endings. See [amendment](OPERATIONAL-AMENDMENT.md) and [supersession ledger](OPERATIONAL-REPLACEMENTS.json).\n\n"
 restorations=json.loads((HERE/'PUBLIC-TEXT-RESTORATIONS.json').read_text())
 completed_restorations=sum(f"{e['name']}-standard-{e['slot']:03d}" in complete for e in restorations['entries'])
 report+=f"Public-text fidelity: {len(restorations['entries'])} ending records had public commentary omitted by the existing cancellation recorder. {completed_restorations} belong to complete conversations (the valid Terra medium closing response); the others are early endings. A derived judge view restores only output_text from stored responses whose public message and native function-call hashes exactly match the capture-time audit. Persisted-response retrieval omits encrypted reasoning; any such private-item hash remains explicitly unverified in the manifest. Originals are unchanged; private reasoning is excluded. See [restoration policy](PUBLIC-TEXT-AMENDMENT.md), [manifest](PUBLIC-TEXT-RESTORATIONS.json), and [Luna review](LUNA-COMPLETION-REVIEW.md).\n\n"
 report+='After generation, a defensive adapter guard rejects changed contexts after an unfinished request; identical-context retries remain allowed. It also rejects non-completed nonstreaming results before history commit and requests encrypted reasoning for unstored replay. All accepted captures used store=true, and this hardening does not alter their completed-response request paths; exact generation bytes remain in the runtime snapshot. See [post-capture hardening](POST-CAPTURE-HARDENING.json).\n\n'
 report+='Twelve configurations were conservatively refreshed based on actual historical adapter logs, not model names. Luna history loss was directly demonstrated; the other ten rows used the same adapter but their historical score impact is not established. These fresh cohorts are independent, and all now explicitly request Priority; some older tiers were mixed or unlogged. Therefore changes in scores or latency are not a controlled estimate of the adapter fix alone. Older raw captures and grades remain available.\n\n'
 report+=helpers.table(['Configuration','Observed scripted turns','Missing future turns','Observed timing P50','P95','N'],[[m['label'],m['observed_scripted_turns'],m['missing_future_turns'],helpers.ms(m['all_attempt_latency']['positive']['p50_ms']),helpers.ms(m['all_attempt_latency']['positive']['p95_ms']),m['all_attempt_latency']['positive']['n']] for m in models])+'\n\n'
 report+=f"All 360 grades use Opus 5.5 Medium and the unchanged text rubric. Successful judge cost: ${cost:.6f}. Anthropic credit cooldowns: {data['judge_credit_cooldowns']}; each waits at least 90 seconds before retrying. Private reasoning is excluded from grading evidence.\n\n"
 report+='[Frozen judge inputs](judging/inputs.json) · [Generation sources](runtime-sha256.json) · [Source verification](capture-source-resolution.json) · [Impact inventory](impact-inventory.json)\n'
 (HERE/'REPORT.md').write_text(report)
 dump(HERE/'ANALYSIS-COMPLETE.json',{'all_quality_complete':True,'aggregate_sha256':sha(HERE/'aggregate.json'),'report_sha256':sha(HERE/'REPORT.md')})
 print(json.dumps({'configurations':12,'assigned':360,'complete':sum(m['complete'] for m in models),'judge_cost_usd':cost}))
if __name__=='__main__':analyze()
