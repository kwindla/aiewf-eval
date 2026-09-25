"""Lossless Gemini spoken exchanges, with provider completion and input audit."""
import hashlib,json,re,sys
from pathlib import Path

def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l]
def normalized(text):return re.sub(r'\s+','',text)
def reconstruct(run):
 events=rows(run/'gemini-events.jsonl');original=rows(run/'transcript.jsonl');log=(run/'run.log').read_text();out=run/'reconstructed';out.mkdir(exist_ok=True)
 inputs=[(int(m[1]),float(m[2])-float(m[3]),float(m[2])) for m in re.finditer(r'\[USER_AUDIO_QUEUED\] turn=(\d+) predicted_end=([0-9.]+) duration=([0-9.]+)',log)]
 issues=[];mapping=[];records=[]
 if any(r.get('recovery_turn') for r in original):issues.append(dict(reason='Recovery occurred; not comparable to no-recovery cohort'))
 if [i for i,_,_ in inputs]!=list(range(len(inputs))):issues.append(dict(reason='Input sequence missing/repeated; do not repair silently'))
 setup=[e for e in events if e['type']=='setup'];meta=json.loads((run/'benchmark-provenance.json').read_text())
 for s in setup:
  c=s['config']
  if c.get('thinking_config',{}).get('thinking_level','').lower()!='medium':issues.append(dict(reason='Thinking setting mismatch'))
  instruction=c.get('system_instruction');text=instruction if isinstance(instruction,str) else ''.join(p.get('text','') for p in instruction.get('parts',[]))
  if hashlib.sha256(text.encode()).hexdigest()!=meta['prompt_sha256']:issues.append(dict(reason='Setup prompt mismatch'))
  for tool in c.get('tools',[]):
   for d in tool.get('function_declarations',[]):
    if d.get('behavior')!='NON_BLOCKING':issues.append(dict(reason='Tool not explicitly NON_BLOCKING',tool=d.get('name')))
 if not setup:issues.append(dict(reason='Missing setup evidence'))
 for row in original:
  idx=row['turn'];w=next((v for v in inputs if v[0]==idx),None)
  if not w:issues.append(dict(turn=idx,reason='No input window'));continue
  next_start=next((v[1] for v in inputs if v[0]==idx+1),float('inf'))
  window=[(n,e) for n,e in enumerate(events) if w[1]<=e['monotonic']<next_start]
  commits=[e for e in events if e['type']=='exchange_commit' and e['turn']==idx]
  # Input log prints only 3 decimals; commit JSON retains full precision.
  if any(e['monotonic']>next_start+.002 for e in commits):issues.append(dict(turn=idx,reason='Commit after next input window beyond timestamp rounding'))
  phases=[e for _,e in window if e['type']=='phase_boundary'];idles=[e for _,e in window if e['type']=='interaction_idle']
  fragments=[]
  for n,e in window:
   if e['type']!='server':continue
   sc=e['message'].get('server_content',{});t=sc.get('output_transcription',{}).get('text')
   if t:fragments.append(dict(line=n+1,monotonic=e['monotonic'],text=t))
  text=''.join(f['text'] for f in fragments).strip()
  if normalized(text)!=normalized(row['assistant_text']):issues.append(dict(turn=idx,reason='Raw spoken fragments differ from committed transcript; audio review required',raw=text,committed=row['assistant_text']))
  if len(commits)!=1:issues.append(dict(turn=idx,reason='Expected one exchange commit',count=len(commits)))
  if not idles:issues.append(dict(turn=idx,reason='No provider IDLE in exchange'))
  if commits:
   before=[e for e in phases if not e['input_open'] and e['monotonic']<=commits[0]['monotonic']]
   if not before or before[-1]['status']!='IDLE':issues.append(dict(turn=idx,reason='Committed without latest terminal IDLE'))
   if commits[0]['monotonic']<w[2]:issues.append(dict(turn=idx,reason='Committed before input ended'))
  r=dict(row);r['assistant_text']=text;records.append(r);mapping.append(dict(turn=idx,fragments=fragments,phases=phases,commits=commits,changed=text!=row['assistant_text']))
 # No repair, favorable reordering, backend/thought substitution, or tool reassignment.
 (out/'transcript.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
 result=dict(method='gemini-idle-exchanges-v1',source_transcript_sha256=hashlib.sha256((run/'transcript.jsonl').read_bytes()).hexdigest(),source_events_sha256=hashlib.sha256((run/'gemini-events.jsonl').read_bytes()).hexdigest(),exchanges=mapping,issues=issues,approved_for_judging=not issues,interim_phase_count=sum(e['type']=='phase_boundary' and e.get('status')=='IN_PROGRESS' for e in events))
 (out/'mapping.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(exchanges=len(records),issues=issues,interim_phase_count=result['interim_phase_count'])));return not issues
if __name__=='__main__':
 if not reconstruct(Path(sys.argv[1])):raise SystemExit('Exchange audit requires review')
