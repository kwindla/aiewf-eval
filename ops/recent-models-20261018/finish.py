"""Finish analysis/publication and verify the generated leaderboard artifacts."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 while not (HERE/'judging/COMPLETE.json').exists():time.sleep(15)
 assert json.loads((HERE/'POST-CAPTURE-COMPAT.json').read_text())['validated']
 for script in ['analyze.py','publish.py']:
  subprocess.run([sys.executable,str(HERE/script)],cwd=ROOT,check=True)
 checks=[('tests',[sys.executable,'-m','pytest','-q','--disable-warnings']),('chart',[sys.executable,str(ROOT/'scripts/render_text_model_voice_readiness.py'),'--check']),('whitespace',['git','diff','--check'])]
 for name,cmd in checks:
  with (HERE/(name+'-final.log')).open('w') as f:
   subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 data=json.loads((HERE/'aggregate.json').read_text());assert data['all_quality_complete'] and data['judged_conversations']==295
 files=['aggregate.json','REPORT.md','PUBLICATION.json','judging/inputs.json','judging/COMPLETE.json','runtime-sha256.json','capture-source-resolution.json','POST-CAPTURE-COMPAT.json']
 verification={'all_checks_passed':True,'conversations':295,'all_quality_complete':True,'successful_judge_cost_usd':data['successful_judge_cost_usd'],'artifacts_sha256':{p:sha(HERE/p) for p in files},'analysis_source_sha256':{p:sha(HERE/p) for p in ['analyze.py','publish.py','voice_waveform.py','verify_capture_sources.py','post_capture_compat.py','finish.py','PROTOCOL.md']},'published_artifacts_sha256':{p:sha(ROOT/p) for p in ['README.md','leaderboard-medium-context.md','docs/text-model-voice-readiness.svg','docs/text-cohort-sizes-2026-10-17.md']},'result_sha256':{str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'judging/results').glob('*.json'))}}
 (HERE/'VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n');print('Campaign complete and verified',flush=True)
if __name__=='__main__':main()
