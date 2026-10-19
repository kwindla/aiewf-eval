"""Reuse the audited voice analysis and compare independent high/none cohorts."""
import importlib.util,json,sys
from common import HERE,ROOT,OLD,RAW,cfg,collect,dump,sha,verify_baseline
from audio import measure,wave
import judge as driver

def analyze():
 driver.verify_driver();verify_baseline()
 cases=driver.base.freeze()
 for c in cases:measure(c['job_id'],(ROOT/c['source']).parent)
 # Reuse exactly the same score, bootstrap, waveform and turn-taking aggregation.
 sys.path.insert(0,str(OLD))
 spec=importlib.util.spec_from_file_location('grok_high_original_analysis',OLD/'analyze.py')
 original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
 sys.path.pop(0)
 original.HERE=HERE;original.ROOT=ROOT;original.cfg=cfg;original.collect=collect;original.judge=driver.base
 original.LABELS={('grok-voice','high'):'Grok Voice 3.0 (reasoning high)'}
 sys.modules['voice_waveform']=wave
 original.main()
 data=json.loads((HERE/'aggregate.json').read_text());assert data['all_quality_complete'] and data['judged_conversations']==25
 high=data['voice_models'][0];none=json.loads((OLD/'aggregate.json').read_text())['voice_models'][0]
 assert high['complete']==none['complete']==25 and high['turn_taking_scored']
 # The baseline's existing audio measurements are immutable and remain bound to their WAVs.
 for item in none['audio_inputs']:
  assert sha(ROOT/item['source_wav'])==item['wav_sha256']
  for path_key,hash_key in [('metrics','metrics_sha256'),('input_waveform_audit','input_waveform_audit_sha256')]:assert sha(ROOT/item[path_key])==item[hash_key]
 comparison={'judge_model':data['judge_model'],'judge_effort':data['judge_effort'],'none':none,'high':high,'strict_delta_percentage_points':high['strict_percent']-none['strict_percent'],'high_successful_judge_cost_usd':data['successful_judge_cost_usd'],'comparison_design':'Independent sequential cohorts; same tasks and inputs, not matched identical responses.','baseline_manifest_sha256':sha(HERE/'baseline-sha256.json')}
 dump(HERE/'comparison.json',comparison)
 def table(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join('---' for _ in header)+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
 def pct(v):return f'{v:.1f}%'
 def ms(v):return '—' if v is None else f'{v:.0f}ms'
 quality=[];timing=[]
 for m in [none,high]:
  den=30*m['complete'];counts=m['complete_only']['dimension_passes']
  quality.append([m['arm'],m['complete'],f"{m['failed_conversations']}/{m['eligible_attempts']}",pct(m['strict_percent']),f"{m['strict_ci95'][0]:.1f}–{m['strict_ci95'][1]:.1f}%",*[f'{counts[d]}/{den}' for d in driver.base.DIMS],f"{m['turn_taking_passes']}/{den}"])
  timing.append([m['arm'],ms(m['first_speech_positive']['p50_ms']),ms(m['first_speech_positive']['p95_ms']),ms(m['non_tool']['median_ms']),ms(m['tool']['mean_ms']),m['first_speech_positive']['n'],m['nonpositive_n'],m['missing_latency_n']])
 report='# Grok Voice 3.0: reasoning high versus none\n\n'+table(['Mode','Complete','Unsuccessful / attempts','Strict pass','95% conversation bootstrap CI','Tool use','Instructions','Grounding','Turn-taking'],quality)+'\n\n'
 report+=f"High changes strict pass by {comparison['strict_delta_percentage_points']:+.1f} percentage points in these independent cohorts. These are new high conversations compared with the existing none cohort; the baseline was neither regenerated nor rejudged. Each cohort has 25 complete conversations and 750 scripted turns. The speech-v2 tasks, audio inputs and Opus 5.5 Medium rubric are the same.\n\n"
 report+=table(['Mode','First-speech P50','P95','Non-tool median','Tool mean','Positive timing N','Nonpositive','Missing'],timing)+'\n\n'
 report+='Latency comes from the actual stereo recording, using the same speech analyzer and three-landmark input waveform correlation as the none study. All completed captures must pass structural and input-alignment checks; timing problems are held for review. All unsuccessful attempts are retained. Quality never determines capture selection. The bootstrap uses the unchanged 20,000-sample conversation method; intervals describe these samples.\n\n'
 report+='[Protocol](PROTOCOL.md) · [High aggregate](aggregate.json) · [Comparison data](comparison.json) · [Attempts](grok-voice/attempts.json) · [Frozen judge inputs](judging/inputs.json) · [None report](../recent-models-20261018/REPORT.md)\n\n'
 report+=f"High-only successful Opus judging cost: ${data['successful_judge_cost_usd']:.6f}. Judge credit cooldown events: {data['judge_credit_cooldowns']}; each waits at least 90 seconds. Private reasoning is not grading evidence.\n"
 failed=[a for a in json.loads((HERE/'grok-voice/attempts.json').read_text()) if not a['complete']]
 if failed:
  report+='\nUnsuccessful high captures:\n\n'+table(['Attempt','Recorded turns','Exit code','Validation error'],[[a['attempt'],a.get('turns','—'),a['exit_code'],a['validation_error'] or '—'] for a in failed])+'\n'
 (HERE/'REPORT.md').write_text(report)
 print(json.dumps({'strict_high':high['strict_percent'],'strict_none':none['strict_percent'],'delta_pp':comparison['strict_delta_percentage_points'],'high_cost_usd':data['successful_judge_cost_usd']}))
if __name__=='__main__':analyze()
