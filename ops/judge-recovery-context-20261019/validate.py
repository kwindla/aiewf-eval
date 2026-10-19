"""Apply the preregistered agreement, repeatability and development gates."""
from common import *

def result(phase,job):
    return read(HERE/'judging'/phase/job/'result.json')

def main():
    refs=read(HERE/'references/FROZEN.json')
    selection=read(HERE/'validation-selection.json')['primary']
    aliases={e['case']:e['job_id'] for e in selection}
    repetitions={phase:{alias:result(phase,job) for alias,job in aliases.items()} for phase in ['validation-A','validation-B']}
    report={'reference_counts':refs['counts'],'reference_files_sha256':sha(HERE/'references/FROZEN.json'),'repetitions':{}}
    passed=refs['minimum_gate_passed']
    for phase,grades in repetitions.items():
        matches=[]
        discrepancies=[]
        for row in refs['consensus']:
            judgment=grades[row['case']]['judgments'][str(row['turn'])]
            actual=judgment['scores'][row['dimension']]
            matches.append({**row,'match':actual==row['expected']})
            if actual!=row['expected']:
                discrepancies.append({**row,'actual':actual,'candidate_reasoning':judgment['reasoning']})
        counts={}
        for category in ['overall','recovery','clean','voice']:
            chosen=[e for e in matches if category=='overall' or e['category']==category]
            numerator=sum(e['match'] for e in chosen)
            counts[category]={'matched':numerator,'total':len(chosen),'agreement':numerator/len(chosen) if chosen else None}
        gate=counts['overall']['agreement']>=.95
        passed &= gate
        report['repetitions'][phase]={'counts':counts,'gate_passed':gate,'discrepancies':discrepancies}
    discordances=[]
    for alias,job in aliases.items():
        a,b=repetitions['validation-A'][alias],repetitions['validation-B'][alias]
        assert set(a['judgments'])==set(b['judgments'])==set(map(str,range(30)))
        for t in range(30):
            for d in DIMS:
                av,bv=a['judgments'][str(t)]['scores'][d],b['judgments'][str(t)]['scores'][d]
                if av!=bv:
                    discordances.append({'case':alias,'turn':t,'dimension':d,'A':av,'B':bv,
                                         'A_reasoning':a['judgments'][str(t)]['reasoning'],
                                         'B_reasoning':b['judgments'][str(t)]['reasoning']})
    report['repeatability']={'discordant':len(discordances),'dimensions':2160,'rate':len(discordances)/2160,'gate_passed':len(discordances)/2160<=.02,'discrepancies':discordances}
    passed &= report['repeatability']['gate_passed']
    development=[]
    for phase in ['development-A','development-B']:
        support=result(phase,'refresh--gemini-3.5-flash-023-68432ef89f55')['judgments']['19']
        development.append({'phase':phase,'case':'support','turn':19,'dimension':'instruction_following','expected':True,
                            'actual':support['scores']['instruction_following'],'reasoning':support['reasoning']})
        delayed=result(phase,'refresh--deepseek-v4-flash-0731-high-023-c4c7afd555bf')['judgments']
        for t in [15,24]:
            for d in ['tool_use_correct','instruction_following']:
                j=delayed[str(t)]
                development.append({'phase':phase,'case':'delayed','turn':t,'dimension':d,'expected':False,
                                    'actual':j['scores'][d],'reasoning':j['reasoning']})
    report['development']={'checks':development,'passed':all(e['actual']==e['expected'] for e in development)}
    passed &= report['development']['passed']
    report['passed']=bool(passed)
    report['limits']='Selected diagnostic sample with old-label stratification; independent model references, not human ground truth or population accuracy. Both repetitions diagnostic; canonical requests separately designated before execution.'
    dump(HERE/'VALIDATION.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['repetitions','development']}),flush=True)
    assert passed,'Validation gate failed; keep last verified publication and resolve before bulk'

if __name__=='__main__':main()
