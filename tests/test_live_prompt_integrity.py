from pathlib import Path


def test_optimized_prompts_keep_full_kb_only_in_backend_and_shared_judge():
    from benchmarks.aiwf_medium_context_speech_live_optimized.config import (
        BenchmarkConfig as Opt,
    )
    from benchmarks.aiwf_medium_context_speech_live.config import (
        BenchmarkConfig as Standard,
    )

    kb = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/aiwf_medium_context/data/knowledge_base.txt"
    ).read_text()
    assert Opt.backend_system_instruction.endswith(kb)
    assert kb not in Opt.frontend_system_instruction
    assert Opt.system_instruction == Standard.system_instruction
    assert Opt.judge_system_prompt == Standard.judge_system_prompt
    assert Opt.turns is Standard.turns and Opt.tools_schema is Standard.tools_schema
