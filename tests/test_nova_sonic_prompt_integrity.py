"""The Nova Sonic prompt-optimized row changes only the SYSTEM policy and tool descriptions."""
import hashlib
import re
from pathlib import Path

from benchmarks.aiwf_medium_context_speech_nova_sonic_optimized.config import BenchmarkConfig as Opt
from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig as Standard

POLICY_SHA256 = "f05ddfdd5966c2f934f2fe915f340a43aaec62ef5b44c9ed66d94cf99ca87925"


def test_optimized_policy_is_the_evaluated_bytes():
    assert hashlib.sha256(Opt.system_policy.encode()).hexdigest() == POLICY_SHA256


def test_knowledge_base_tasks_audio_and_judge_are_unchanged():
    kb = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/aiwf_medium_context/data/knowledge_base.txt"
    ).read_text()
    assert Opt.reference_context == Standard.reference_context
    assert Opt.reference_context.endswith(kb)
    assert kb not in Opt.system_policy
    # The Nova Sonic USER reference layout requires exactly policy + reference
    assert Opt.system_instruction == Opt.system_policy + Opt.reference_context
    assert Opt.turns is Standard.turns
    assert Opt.audio_dir == Standard.audio_dir
    assert Opt.judge_system_prompt == Standard.judge_system_prompt


def test_tools_differ_only_in_descriptions():
    optimized = Opt.tools_schema.standard_tools
    standard = Standard.tools_schema.standard_tools
    assert [t.name for t in optimized] == [t.name for t in standard]
    for opt, std in zip(optimized, standard):
        assert opt.required == std.required
        assert list(opt.properties) == list(std.properties)
        for name in opt.properties:
            assert opt.properties[name]["type"] == std.properties[name]["type"]


def test_policy_contains_no_scripted_answers():
    # Generic rules only: no session IDs, user names or session titles from the script
    assert not re.search(r"\b\d{6}\b", Opt.system_policy)
    for text in ["Jennifer", "Kirschner", "Vibe Coding", "telemetry", "vegan"]:
        assert text not in Opt.system_policy
        for tool in Opt.tools_schema.standard_tools:
            assert text not in tool.description
            assert all(text not in p["description"] for p in tool.properties.values())