"""Find positive native-wire evidence of missing public recovery endings."""
from common import *

def main():
    evidence=[]
    checked=[]
    unavailable=[]
    for c in read(HERE/'inventory.json')['cases']:
        endings=[(i,r) for i,r in enumerate(records(c)) if r.get('recovery_turn') and any(t.get('name')=='end_session' for t in r.get('tool_calls',[]))]
        if not endings:
            continue
        wire=rootpath(c['source']).parent/'responses-history-audit.jsonl'
        if not wire.exists():
            unavailable.append({'job_id':c['job_id'],'reason':'No native Responses audit for this capture'})
            continue
        completed=[json.loads(line) for line in wire.read_text().splitlines() if line.strip()]
        completed=[e for e in completed if e['event']=='completed' and any(i.get('name')=='end_session' for i in e['output_items'])]
        assert len(endings)==len(completed)==1, ('Ambiguous ending correspondence',c['job_id'])
        position,record=endings[0]
        event=completed[0]
        item={'job_id':c['job_id'],'source_sha256':c['transcript_sha256'],'record_index':position,
              'wire_path':str(wire.relative_to(ROOT)),'wire_sha256':sha(wire),'provider_response_id':event['id'],
              'recorded_public_characters':len(record['assistant_text']),
              'native_public_characters':sum(i.get('content_characters',0) for i in event['output_items'] if i.get('role')=='assistant')}
        checked.append(item)
        if not record['assistant_text'].strip() and item['native_public_characters']:
            evidence.append(item)
    dump(HERE/'RECOVERY-ENDING-AUDIT.json',{'checked':checked,'positive_omissions':evidence,'wire_unavailable':unavailable,
         'interpretation':'Absence of native evidence does not establish an omission; empty tool-only responses remain valid evidence.'})
    dump(HERE/'RECOVERY-PUBLIC-TEXT-RESTORATIONS-v1.json',{'schema_version':1,'entries':[],
         'audit_sha256':sha(HERE/'RECOVERY-ENDING-AUDIT.json'),'note':'Original 34-entry restoration manifest preserved separately.'})
    print('Recovery endings:',len(checked),'with native wire;',len(unavailable),'without wire;',len(evidence),'positive omissions',flush=True)
    assert not evidence,'Positive omitted public recovery output must be restored before validation'

if __name__=='__main__':main()
