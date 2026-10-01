"""Regular Gemini 3.8: standard prompt, no thinking config, synchronous completion."""

import argparse, asyncio, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from multi_turn_eval.pipelines.gemini_regular import RegularPipeline, RegularService
from multi_turn_eval.pipelines import gemini_exchanges, gemini_regular
from multi_turn_eval.pipelines.realtime import (
    GeminiLiveLLMServiceWithReconnection,
    GeminiVADParams,
)
from multi_turn_eval.cli import create_run_directory, setup_logging
from multi_turn_eval.recording.transcript_recorder import TranscriptRecorder
from multi_turn_eval.cli import load_benchmark
from dotenv import load_dotenv


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True)
    a = p.parse_args()
    load_dotenv()
    os.environ["MTE_ENABLE_RECOVERY"] = "0"
    b = load_benchmark(a.benchmark)()
    model = "gemini-3.8-live"
    out = create_run_directory(b.name, model)
    setup_logging(out)
    print("Output directory:", out, flush=True)
    meta = dict(
        model=model,
        thinking=None,
        prompt_sha256=hashlib.sha256(b.system_instruction.encode()).hexdigest(),
        judge_prompt_sha256=hashlib.sha256(b.judge_system_prompt.encode()).hexdigest(),
        startup_guard=True,
        completion="synchronous turnComplete plus playback; deferred end_session",
    )
    (out / "topup-provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "capture-snapshot.py").write_bytes(Path(__file__).read_bytes())
    (out / "adapter-snapshot.py").write_bytes(
        Path(gemini_exchanges.__file__).read_bytes()
    )
    (out / "regular-adapter-snapshot.py").write_bytes(
        Path(gemini_regular.__file__).read_bytes()
    )
    (out / "system-prompt.txt").write_text(b.system_instruction)
    recorder = TranscriptRecorder(out, model)
    pipeline = RegularPipeline(
        b,
        out / "gemini-events.jsonl",
        require_interaction_status=False,
        explicit_audio_activity=True,
        allow_turn_replay=False,
    )
    try:
        await asyncio.wait_for(
            pipeline.run(
                recorder=recorder,
                model=model,
                service_class=RegularService,
                service_name="gemini-live",
                thinking="default",
            ),
            900,
        )
    finally:
        recorder.close()
        pipeline.event_file.close()


if __name__ == "__main__":
    asyncio.run(main())
