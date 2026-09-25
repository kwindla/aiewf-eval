"""Shared Opus 5 quality rubric on reconstructed spoken exchanges only."""
import asyncio,json,sys
from pathlib import Path
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
from multi_turn_eval.judging.claude_judge import judge_with_claude,load_transcript,write_outputs
async def main():
 load_dotenv();run=Path(sys.argv[1]);out=run/'reconstructed'
 assert json.loads((out/'mapping.json').read_text())['approved_for_judging']
 b=BenchmarkConfig();result=await judge_with_claude(out,expected_turns=b.turns,skip_turn_taking=True,judge_model='claude-opus-5',judge_system_prompt=b.judge_system_prompt)
 write_outputs(out,load_transcript(out),result['judgments'],result['summary'],result['model_name'],result.get('realignment_notes',''),result.get('function_tracking',{}),result.get('turn_taking_analysis'),judge_metadata=result.get('judge_metadata'))
 (out/'evaluation-scope.json').write_text(json.dumps(dict(quality_rubric='speech-v2-policy-compliance-1',audio_turn_taking='Not scored here; quantitative audio assessment is separate. Do not publish judge default turn_taking as audio success.',backend_text_substituted=False),indent=2)+'\n')
 print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':asyncio.run(main())
