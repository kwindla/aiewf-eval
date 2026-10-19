"""Audit all registered Luna endings without making quality judgments."""
import hashlib,json,statistics,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from preflight import ROOT,cfg,collect,dump
c=cfg('luna');collect.verify_source_hashes(c)
_,rows=collect.validate_manifests(c,collect.validate_schedule(c));assert len(rows)==60
cases=[]
for row in rows:
 source=ROOT/row['run_dir']/'transcript.jsonl';raw=source.read_bytes();records=[json.loads(line) for line in raw.splitlines() if line.strip()];scheduled=[r for r in records if not r.get('recovery_turn')]
 last=scheduled[-1];cases.append({'slot':int(row['slot']),'arm':row['arm'],'source':str(source.relative_to(ROOT)),'transcript_sha256':hashlib.sha256(raw).hexdigest(),'observed_scripted_turns':len(scheduled),'complete':len(scheduled)==30,'last_user_turn_one_based':last['turn']+1,'last_user':last['user_text'],'last_tools':last['tool_calls'],'early_end_session':last['turn']<29 and any(t['name']=='end_session' for t in last['tool_calls'])})
summary={}
for mode in ['none','low']:
 selected=[r for r in cases if r['arm']==mode];lengths=[r['observed_scripted_turns'] for r in selected]
 summary[mode]={'assigned':len(selected),'complete':sum(r['complete'] for r in selected),'early_end_session':sum(r['early_end_session'] for r in selected),'observed_scripted_turns':sum(lengths),'missing_future_turns':900-sum(lengths),'median_observed_turns':statistics.median(lengths),'min_observed_turns':min(lengths),'max_observed_turns':max(lengths),'ending_turns_one_based':dict(sorted(Counter(r['last_user_turn_one_based'] for r in selected).items()))}
dump(HERE/'completion-audit.json',{'arms':summary,'cases':cases,'quality_judgments':'No grades inferred from completion; see the campaign aggregate.'})
text='# GPT-6 Luna completion audit\n\nBoth registered modes finished collection. This audit describes completion only; see the [campaign report](../REPORT.md) for quality scores. Every assigned conversation remains in the cohort.\n\n| Mode | Assigned | Complete | Early end_session | Observed turns | Missing future turns | Median observed length |\n|---|---:|---:|---:|---:|---:|---:|\n'
for mode,s in summary.items():text+=f"| {mode} | {s['assigned']} | {s['complete']} | {s['early_end_session']} | {s['observed_scripted_turns']} | {s['missing_future_turns']} | {s['median_observed_turns']:g} |\n"
text+='\nThe transcripts show the model calling end_session before the user ends the conversation. The off mode commonly ends around the attendance correction; low generally reaches much further into the conversation. See [all ending evidence](completion-audit.json) for exact questions, calls, source paths, and hashes.\n\nAn excluded full-harness diagnostic captured provider stream events and completed output items. All tool names matched, including the premature end_session. That check found no tool-name translation defect; see [stream audit](stream-audit-summary.json). No runtime source files, prompts, reasoning settings, or canonical captures were changed for that check.\n\nThese are completion results, not a judgment of every preceding answer. Final fixed-denominator accuracy will count missing future turns as failures; completed-only accuracy is unavailable with zero complete conversations.\n'
(HERE/'COMPLETION.md').write_text(text);print(json.dumps(summary))
