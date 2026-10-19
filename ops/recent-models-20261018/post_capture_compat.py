"""Repair the unset Sonnet 5.5 CLI mode only after capture and input freezing."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def dump(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def main():
 while not (HERE/'COLLECTION-COMPLETE.json').exists() or not (HERE/'judging/inputs.json').exists() or not (HERE/'judging/attempts.jsonl').exists():time.sleep(10)
 marker=HERE/'POST-CAPTURE-COMPAT.json';path=ROOT/'src/multi_turn_eval/pipelines/base.py'
 if marker.exists():
  record=json.loads(marker.read_text());assert sha(path.read_bytes())==record['after_sha256'],'Post-capture runtime changed unexpectedly'
  if record.get('validated'):return
 else:
  raw=path.read_bytes();expected=json.loads((HERE/'runtime-sha256.json').read_text())['src/multi_turn_eval/pipelines/base.py']
  assert sha(raw)==expected,'Do not overwrite a changed runtime source'
  anchor='            haiku55 = model_lower == "claude-haiku-5-5"\n'
  replacement=anchor+'''            # Pipecat 1.12 otherwise injects disabled for Sonnet 5+, which
            # Sonnet 5.5 rejects. Preserve the intended no-up-front-thinking
            # default with its supported replacement. Explicit modes/efforts win.
            if sonnet55 and not thinking_mode and not effort:
                thinking_mode = "between_tools"
'''
  text=raw.decode();assert text.count(anchor)==1
  updated=text.replace(anchor,replacement).encode();temp=path.with_suffix('.compat.tmp');temp.write_bytes(updated);temp.replace(path)
  record={'applied_after_capture':True,'judge_inputs_already_frozen':True,'judge_inputs_sha256':sha((HERE/'judging/inputs.json').read_bytes()),'before_sha256':expected,'after_sha256':sha(updated),'validated':False,'scope':'Unset Sonnet 5.5 mode only; every registered campaign mode was explicit and is unchanged.','captured_source_snapshot':'runtime-snapshot/src/multi_turn_eval/pipelines/base.py'}
  dump(marker,record)
 with (HERE/'post-capture-compat-tests.log').open('w') as out:
  subprocess.run([sys.executable,'-m','pytest','tests/test_recent_model_modes.py','tests/test_openai_responses_routing.py','tests/test_capture_source_verification.py','-q','--disable-warnings'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
 record['validated']=True;dump(marker,record)
 doc=HERE/'MODE-SETTINGS.md';text=doc.read_text()
 start=text.index('Sonnet 5.5 requires an explicit thinking mode with Pipecat 1.12:')
 end=text.index('\n\nFor example,',start)
 text=text[:start]+'''The registered modes are explicit. After all capture finished and judge inputs
were frozen, the factory's unset Sonnet 5.5 default was repaired: it now uses
`between_tools`, replacing Pipecat 1.12's rejected `disabled` default. Explicit
thinking modes and effort settings keep precedence. This compatibility repair
was not used to generate or reinterpret any campaign capture; original source
bytes remain in the verified runtime snapshot. See POST-CAPTURE-COMPAT.json.
`between_tools` has no display field; adaptive runs use `display=omitted`, with
signed thinking blocks preserved by the patched Anthropic adapter.'''+text[end:]
 doc.write_text(text)
 print('Post-capture Sonnet default compatibility fix applied and tested',flush=True)
if __name__=='__main__':main()
