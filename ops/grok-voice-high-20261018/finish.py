"""Publish only after all high grades and checks pass; retain a full audit."""
import json,subprocess,sys,time
from common import HERE,ROOT,OLD,sha,dump,verify_baseline

def main():
 while not (HERE/'judging/COMPLETE.json').exists():time.sleep(15)
 for name in ['analyze.py','publish.py']:subprocess.run([sys.executable,str(HERE/name)],cwd=ROOT,check=True)
 commands=[('tests',[sys.executable,'-m','pytest','-q','--disable-warnings']),('chart',[sys.executable,str(ROOT/'scripts/render_text_model_voice_readiness.py'),'--check']),('whitespace',['git','diff','--check'])]
 for name,cmd in commands:
  with (HERE/(name+'-final.log')).open('w') as out:subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
 published=[ROOT/'README.md',ROOT/'leaderboard-medium-context.md',ROOT/'docs/text-model-voice-readiness.svg',ROOT/'docs/text-cohort-sizes-2026-10-17.md',HERE/'PUBLICATION.json']
 before={str(p):p.read_bytes() for p in published};subprocess.run([sys.executable,str(HERE/'publish.py')],cwd=ROOT,check=True)
 assert all(p.read_bytes()==before[str(p)] for p in published),'Publisher is not idempotent'
 verify_baseline()
 a=json.loads((HERE/'aggregate.json').read_text());assert a['all_quality_complete'] and a['judged_conversations']==25
 sources=[HERE/p for p in ['common.py','smoke.py','collect.py','audio.py','judge.py','analyze.py','publish.py','finish.py','precompute.py','status.py','PROTOCOL.md']]+[OLD/p for p in ['judge_campaign.py','analyze.py','voice_waveform.py','verify_capture_sources.py']]+[ROOT/'scripts/analyze_turn_metrics.py']
 artifacts=[HERE/p for p in ['aggregate.json','comparison.json','REPORT.md','PUBLICATION.json','baseline-sha256.json','runtime-sha256.json','capture-source-resolution.json','judging/inputs.json','judging/driver-sha256.json','judging/COMPLETE.json','COLLECTION-COMPLETE.json']]
 dump(HERE/'VERIFICATION.json',{'all_checks_passed':True,'conversations':25,'all_quality_complete':True,'publisher_idempotent':True,'baseline_unchanged':True,'successful_judge_cost_usd':a['successful_judge_cost_usd'],'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'artifacts_sha256':{str(p.relative_to(HERE)):sha(p) for p in artifacts},'published_artifacts_sha256':{str(p.relative_to(ROOT)):sha(p) for p in published},'result_sha256':{str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'judging/results').glob('*.json'))},'audio_proof_sha256':{str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'grok-voice/audio').glob('*-source.json'))}})
 print('High campaign complete, published and verified',flush=True)
if __name__=='__main__':main()
