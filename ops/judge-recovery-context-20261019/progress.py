"""Read-only canonical progress and reported spend."""
import time
from common import *

def main():
    paths=list((HERE/'judging/canonical').glob('*/result.json'))
    events=[json.loads(line) for line in (HERE/'attempts.jsonl').read_text().splitlines() if line.strip()]
    canonical=[e for e in events if e.get('phase')=='canonical']
    start=min((e['at'] for e in canonical if e['event']=='start'),default=time.time())
    cost=sum(read(p)['metadata'].get('judge_cost_usd') or 0 for p in paths)
    elapsed=time.time()-start
    status={'graded':len(paths),'target':698,'remaining':698-len(paths),'elapsed_minutes':round(elapsed/60,1),
            'successful_canonical_cost_usd':round(cost,4),'operational_errors':sum(e['event']=='error' for e in canonical),
            'credit_failures':sum(e.get('credit_failure',False) for e in canonical),
            'rough_remaining_minutes':round(elapsed/max(len(paths),1)*(698-len(paths))/60) if paths else None,
            'complete':(HERE/'canonical-COMPLETE.json').exists()}
    dump(HERE/'STATUS.json',status)
    print(json.dumps(status),flush=True)

if __name__=='__main__':main()
