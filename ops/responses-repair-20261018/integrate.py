"""Integrate only after both isolated capture and frozen voice verification finish."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from common import HERE,ROOT,sha,dump
MAIN=Path('/home/khkramer/src/aiewf-eval-wt/gemini38-flash')
TARGET=MAIN/'ops/responses-repair-20261018'
FILES=['src/multi_turn_eval/services/openai_responses.py','src/multi_turn_eval/pipelines/base.py','tests/test_openai_responses_routing.py','tests/test_text_model_voice_readiness.py']
EXTRA_TESTS=['tests/test_responses_native_history.py','tests/test_responses_repair_campaign.py','tests/test_responses_interrupted_context.py']
def execute(command,log):
 env=os.environ.copy();env['PYTHONPATH']=str(MAIN/'src')+os.pathsep+str(MAIN)
 with (TARGET/log).open('w') as out:subprocess.run(command,cwd=MAIN,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
def main():
 gate=MAIN/'ops/grok-voice-high-20261018/VERIFICATION.json'
 while not gate.exists() or not (HERE/'ANALYSIS-COMPLETE.json').exists():time.sleep(15)
 assert json.loads(gate.read_text())['all_checks_passed']
 data=json.loads((HERE/'aggregate.json').read_text());assert data['all_quality_complete'] and data['judged_conversations']==360
 expected=json.loads((HERE/'integration-prior-sha256.json').read_text())
 for name,digest in expected.items():assert sha(MAIN/name)==digest,'Integration held: existing work changed '+name
 # The main runtime is no longer participating in any high collection/judging.
 for name in FILES+EXTRA_TESTS:shutil.copy2(ROOT/name,MAIN/name)
 shutil.copytree(HERE,TARGET,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pid'))
 shutil.copytree(ROOT/'runs/responses-repair-20261018',MAIN/'runs/responses-repair-20261018',dirs_exist_ok=True)
 hardening=json.loads((TARGET/'POST-CAPTURE-HARDENING.json').read_text())
 adapter=MAIN/'src/multi_turn_eval/services/openai_responses.py'
 assert sha(adapter)==hardening['captured_adapter_sha256']
 hardened=TARGET/'post-capture/openai_responses.py';assert sha(hardened)==hardening['hardened_adapter_sha256']
 shutil.copy2(hardened,adapter)
 hardening['state']='applied_after_all_generation_and_judging';dump(TARGET/'POST-CAPTURE-HARDENING.json',hardening)
 # Preserve the publication layering of later approved prospective cohorts.
 publisher=MAIN/'ops/judge-refresh-opus55-20261017/publish.py';source=publisher.read_text()
 marker="    repair=ROOT/'ops/responses-repair-20261018/PUBLICATION.json'"
 if marker not in source:
  point=source.index("\n\nif __name__ == '__main__':")
  source=source[:point]+"\n"+marker+"\n    if repair.exists():\n        subprocess.run([sys.executable,str(repair.parent/'publish.py')],cwd=ROOT,check=True)\n"+source[point:]
  publisher.write_text(source)
 test=MAIN/'tests/test_judge_refresh_publication.py';source=test.read_text()
 old="    text=(ROOT/'README.md').read_text().split('Text mode models:\\n')[1].split('Speech-to-speech models:\\n')[0]"
 replacement="""    repair_path=ROOT/'ops/responses-repair-20261018/PUBLICATION.json'
    repair=json.loads(repair_path.read_text()) if repair_path.exists() else {}
    if repair:
        addition=json.loads((repair_path.parent/'aggregate.json').read_text())
        models=[m for m in models if m['label'] not in repair['replaced_labels']]
        models += [m for m in addition['text_models'] if m['label'] in repair['ranked_labels']]
"""+old
 assert old in source
 source=source.replace(old,replacement,1)
 source=source.replace("assert sum(m['shortfall']>0 for m in models)==13+recent.get('text_shortfalls_added',0)","assert sum(m['shortfall']>0 for m in models)==(repair['current_shortfalls'] if repair else 13+recent.get('text_shortfalls_added',0))")
 test.write_text(source)
 python=str(MAIN/'.venv/bin/python')
 execute([python,str(TARGET/'analyze.py')],'analysis-integrated.log')
 execute([python,str(TARGET/'publish.py')],'publication.log')
 finish_verification()
def finish_verification():
 data=json.loads((TARGET/'aggregate.json').read_text());assert data['all_quality_complete'] and data['judged_conversations']==360
 hardening=json.loads((TARGET/'POST-CAPTURE-HARDENING.json').read_text())
 assert hardening['state']=='applied_after_all_generation_and_judging'
 assert sha(MAIN/'src/multi_turn_eval/services/openai_responses.py')==hardening['hardened_adapter_sha256']
 assert json.loads((MAIN/'ops/grok-voice-high-20261018/VERIFICATION.json').read_text())['all_checks_passed']
 python=str(MAIN/'.venv/bin/python')
 execute([python,'-m','pytest','-q','--disable-warnings'],'tests-final.log')
 execute([python,str(MAIN/'scripts/render_text_model_voice_readiness.py'),'--check'],'chart-final.log')
 execute(['git','diff','--check'],'whitespace-final.log')
 published=[MAIN/p for p in ['README.md','leaderboard-medium-context.md','docs/text-model-voice-readiness.svg','docs/text-cohort-sizes-2026-10-17.md']]+[TARGET/'PUBLICATION.json']
 before={str(p):p.read_bytes() for p in published}
 execute([python,str(TARGET/'publish.py')],'publication-idempotence.log')
 assert all(p.read_bytes()==before[str(p)] for p in published),'Publisher is not idempotent'
 baseline=json.loads((MAIN/'ops/grok-voice-high-20261018/baseline-sha256.json').read_text())
 assert all(sha(MAIN/name)==digest for name,digest in baseline.items()),'Earlier study changed'
 sources=[MAIN/p for p in FILES+EXTRA_TESTS+['tests/test_judge_refresh_publication.py','ops/judge-refresh-opus55-20261017/publish.py']]+list(TARGET.glob('*.py'))
 artifacts=[TARGET/p for p in ['aggregate.json','REPORT.md','PUBLICATION.json','runtime-sha256.json','capture-source-resolution.json','judging/inputs.json','judging/driver-sha256.json','judging/COMPLETE.json','COLLECTION-COMPLETE.json','collection-driver-sha256.json','POST-CAPTURE-HARDENING.json','OPERATIONAL-REPLACEMENTS.json','OPERATIONAL-REPLACEMENTS-COMPLETE.json','POST-CAPTURE-HARDENING-TESTS.json','OPERATIONAL-AMENDMENT.md','ANALYSIS-COMPLETE.json','PUBLIC-TEXT-AMENDMENT.md','PUBLIC-TEXT-RESTORATIONS.json','PUBLIC-TEXT-FINAL-CHECK.json','JUDGING-SCHEDULE-AMENDMENT.md','LUNA-COMPLETION-REVIEW.md','LUNA-COMPLETION-FINAL-AUDIT.json','LUNA-COMPLETION-REVIEW-INTERIM.md']]
 dump(TARGET/'VERIFICATION.json',{'all_checks_passed':True,'assigned_conversations':360,'all_quality_complete':True,'main_runtime_integrated':True,'publisher_idempotent':True,'earlier_studies_unchanged':True,'successful_judge_cost_usd':data['successful_judge_cost_usd'],
      'source_sha256':{str(p.relative_to(MAIN)):sha(p) for p in sources},'artifact_sha256':{str(p.relative_to(TARGET)):sha(p) for p in artifacts},'published_artifact_sha256':{str(p.relative_to(MAIN)):sha(p) for p in published},
      'result_sha256':{str(p.relative_to(TARGET)):sha(p) for p in sorted((TARGET/'judging/results').glob('*.json'))}})
 (TARGET/'STATUS.md').write_text('# Responses adapter refresh complete\n\nAll 360 assigned captures rejudged; repaired runtime integrated; README, chart, full tests and publication idempotence verified.\n\n[Results](REPORT.md) · [Verification](VERIFICATION.json)\n')
 print('Responses repair and 360-conversation refresh integrated, published and verified',flush=True)
if __name__=='__main__':
 if sys.argv[1:]==['--verify-integrated']:finish_verification()
 else:main()
