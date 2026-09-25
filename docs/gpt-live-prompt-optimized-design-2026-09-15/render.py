"""Render review-only prompts; imports no benchmark runner and makes no API calls."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
kb=ROOT/'benchmarks/aiwf_medium_context/data/knowledge_base.txt'
frontend=(HERE/'frontend.txt').read_text()
instructions=(HERE/'backend-instructions.txt').read_text()
backend=instructions+'\n# Knowledge base\n\n'+kb.read_text()
(HERE/'backend.txt').write_text(backend)
assert backend.endswith(kb.read_text())
assert kb.read_text() not in frontend
metadata=dict(status='draft-for-review; not registered or active',proposed_dataset_label='GPT-Live prompt-optimized',model='gpt-live-1',backend_model='gpt-5.6-terra',backend_reasoning='medium',voice='marin',delegation='responses',knowledge_base_sha256=hashlib.sha256(kb.read_bytes()).hexdigest(),frontend_sha256=hashlib.sha256(frontend.encode()).hexdigest(),backend_sha256=hashlib.sha256(backend.encode()).hexdigest(),frontend_words=len(frontend.split()),backend_instruction_words=len(instructions.split()),backend_total_words=len(backend.split()),sources=['https://developers.openai.com/api/docs/guides/live-prompting','https://developers.openai.com/api/docs/guides/live-delegation'],source_review_date='2026-09-15')
(HERE/'draft-manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))
