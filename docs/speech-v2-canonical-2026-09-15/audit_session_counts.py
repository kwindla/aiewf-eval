"""Read-only audit of session-count answers across every arm; never edits scores."""
import json,re
from collections import Counter
from pathlib import Path
HERE=Path(__file__).parent
captures=[json.loads(l) for l in (HERE/'capture/attempts.jsonl').read_text().splitlines()]
rows=[]
number=r'(?:\d+|two|three|four|five|six|seven|eight|nine|ten)'
explicit=re.compile(r'\b'+number+r'\s+(?:(?:main|conference|scheduled|listed|MCP|track)\s+){0,3}sessions\b',re.I)
breakdown=re.compile(r'\b'+number+r'\s+(?:talks|workshops|keynotes)\b',re.I)
for capture in captures:
 if not capture.get('run_dir'):continue
 path=Path(capture['run_dir'])/'claude_judged.jsonl'
 if not path.exists():continue
 for line in path.read_text().splitlines():
  d=json.loads(line)
  if d['turn']!=4:continue
  text=d['assistant_text'];category='explicit_session_count' if explicit.search(text) else 'numeric_breakdown' if breakdown.search(text) else 'overview_or_other'
  rows.append(dict(arm=capture['arm'],attempt=capture['index'],run_dir=capture['run_dir'],category=category,assistant=text,scores=d['scores'],strict_pass=all(d['scores'][k] for k in ['tool_use_correct','instruction_following','kb_grounding']),reason=d['claude_reasoning']))
summary={}
for arm in ['openai21','grok','openai15','gemini31','gemini38']:
 selected=[r for r in rows if r['arm']==arm]
 summary[arm]=dict(judged=len(selected),categories=dict(Counter(r['category'] for r in selected)),strict_failures=sum(not r['strict_pass'] for r in selected),numeric_answers_passed=sum(r['strict_pass'] and r['category']!='overview_or_other' for r in selected),numeric_answers_instruction_passed=sum(r['scores']['instruction_following'] and r['category']!='overview_or_other' for r in selected))
(HERE/'session-count-audit.json').write_text(json.dumps(dict(note='Diagnostic pattern flags for manual review, not score corrections. No prompts, judge outputs, or scores changed.',summary=summary,answers=rows),indent=2))
print(json.dumps(summary,indent=2))
for r in rows:
 if r['strict_pass'] and r['category']!='overview_or_other':print('REVIEW',r['arm'],r['attempt'],r['assistant'],r['reason'])
