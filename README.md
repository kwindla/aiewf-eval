# Multi-Turn Eval

A framework for evaluating multi-turn LLM conversations with support for text, realtime audio, and speech-to-speech models.

The two benchmarks here in this public repo are:

- `aiwf_long_context` - older long-context benchmark described [here](https://post-training.aitinkerers.org/p/your-conversation-is-out-of-distribution)
- `aiwf_medium_context` - newer medium-context benchmark

## aiwf_medium_context results summary for selected models

Text mode models:

| Model                     | Tool Use  | Instruction | KB Ground | Pass Rate | Median Rate | TTFB Med | TTFB P95 | TTFB Max |
|---------------------------|-----------|-------------|-----------|-----------|-------------|----------|----------|----------|
| gpt-5.1                   | 300/300   | 300/300     | 300/300   | 100.0%    | 100.0%      |  916ms   | 2011ms   | 5216ms   |
| gemini-3-flash-preview    | 300/300   | 300/300     | 300/300   | 100.0%    | 100.0%      | 1193ms   | 1635ms   | 6653ms   |
| claude-sonnet-4-5         | 300/300   | 300/300     | 300/300   | 100.0%    | 100.0%      | 2234ms   | 3062ms   | 5438ms   |
| gpt-4.1 †                 | 283/300   | 273/300     | 298/300   | 94.9%     | 97.8%       | 683ms    | 1052ms   | 3860ms   |
| gemini-2.5-flash †        | 275/300   | 268/300     | 300/300   | 93.7%     | 94.4%       |  594ms   | 1349ms   | 2104ms   |
| kimi-k2-instruct-0905 ‡   | 28/30     | 29/30       | 30/30     | 93.3%     | 93.3%       | 286ms    | 1774ms   | 2465ms   |
| nova-2-pro-preview        | 288/300   | 278/300     | 289/300   | 92.7%     | 93.3%       |  686ms   |  750ms   | 1459ms   |
| gpt-5-mini                | 271/300   | 272/300     | 289/300   | 92.4%     | 95.6%       | 6339ms   | 17845ms  | 27028ms  |
| gpt-4o-mini               | 271/300   | 262/300     | 293/300   | 91.8%     | 92.2%       |  760ms   | 1322ms   | 3256ms   |
| gpt-4o                    | 278/300   | 249/300     | 294/300   | 91.2%     | 95.6%       |  625ms   | 1222ms   | 13378ms  |
| nemotron-3-nano-30b-a3b * | 282/300   | 280/300     | 293/300   | 91.0%     | 93.3%       |  171ms   |  199ms   |  255ms   |
| gpt-oss-120b (groq)       | 272/300   | 270/300     | 298/300   | 89.3%     | 90.0%       |   98ms   |  226ms   | 2117ms   |
| gpt-5.2                   | 224/300   | 228/300     | 250/300   | 78.0%     | 92.2%       |  819ms   | 1483ms   | 1825ms   |
| claude-haiku-4-5          | 221/300   | 172/300     | 299/300   | 76.9%     | 75.6%       |  732ms   | 1334ms   | 4654ms   |


> - [ † ] - gpt-4.1 and gemini-2.5-flash are used for most production voice agents because they currently offer the best balance of overall intelligence and low TTFB.
> - [ * ] - Nemotron 3 Nano running in-cluster on NVIDIA Blackwell hardware
> - [ ‡ ] - Single-run result from `runs/aiwf_medium_context/20260217T124339_moonshotai_kimi-k2-instruct-0905_d7392630` (not yet a 10-run aggregate). Claude summary for this run: Tool Use 28/30, Instruction 29/30, KB Grounding 30/30, Turn-Taking 30/30.

Each conversation in this benchmark is 30 turns. Unless marked otherwise (for example `‡`), the scores above are aggregated across 10 runs for each model. **Pass Rate** means the percentage of total turns across all runs that the judge model scored as successful. Each run is also scored independently. **Median Rate** is the median individual run pass rate. Think of pass rate as the model's average performance, and the median rate as a way to measure the model's consistency. See gpt-5.2, for example, which has a pass rate of 78.0% but a median rate of 92.2%, indicating that while it performs well on average, it can have very poor runs that drag down the median conversation score.

TTFB is the number reported by the Pipecat service for each model. It is the time from the request to generate inference to the first byte of the response. An optimized speech-to-speech pipeline with typical network latencies should be able to achieve a total voice-to-voice latency of approximately LLM TTFB + 500ms. In general, a model with a TTFB above ~700ms is too slow for most voice agent use cases.
For the same `kimi-k2-instruct-0905` run above, end-to-end per-turn latency (`latency_ms` in `transcript.jsonl`) was: median 899ms, p95 3954ms, max 5249ms.

Speech-to-speech models:

| Model             | Tool Use    | Instruction | KB Ground      | Turn Ok    | Pass Rate    | Non-Tool V2V Med  | Non-Tool V2V Max  | Tool V2V Mean   | Silence Pad Mean |
|-------------------|-------------|-------------|----------|------------|--------------|-------------------|-------------------|--------------|----------------|
| ultravox-v0.7     | 293/300 | 294/300     | 298/300  | 300/300 |   97.7%  | 864ms         |  1888ms       | 2406ms     | 82ms         |
| gpt-realtime      | 271/300 | 260/300     | 300/300  | 296/300 |   86.7%  | 1536ms        |  4672ms       | 2199ms     | 341ms        |
| gemini-live       | 258/300 | 261/300     | 293/300  | 278/300 |   86.0%  | 2624ms        | 30000ms       | 4082ms     | 90ms         |
| nova-2-sonic      | 278/300 | 265/300     | 296/300  | *       |   *      | 1280ms        |  3232ms       | 1689ms     | 79ms         |
| grok-realtime     | 267/300 | 275/300     | 295/300  | *       |   *      | 1184ms        |  2016ms       | 1472ms     | 478ms        |

For speech-to-speech models, we measure voice-to-voice latency by analyzing the conversation recording. We measure the overall time from the end of the user's speech to the beginning of the model's speech response. This latency is different from the TTFB reported by the Pipecat service for these models, because all of these models were tested in server-side VAD mode (the server-side turn delay is opaque to the Pipecat pipeline), and all of the models send initial silence bytes before actual speech audio. (Text-to-speech models do this, too. The initial silence segments are typically between 150ms and 250ms for standalone TTS models.)

We also include a "Turn Ok" column for these models, which counts how often the model does not respond at all when we expect it or responds with control characters, API refusals, or generic errors.

The API for Grok is currently not production ready. Grok often enters into an unrecoverable state during a session.

The Nova 2 Sonic model performs very well on instruction following and tool calling but has a high rate of safety refusals for normal content. It also has a connection limit of 8 minutes. Fixes for both of these issues are in flight from AWS.

### Sample Recordings

Listen to full 30-turn benchmark conversations from each speech-to-speech model:

| Model         | Recording                                      |
|---------------|------------------------------------------------|
| Ultravox v0.7 | [ultravox-v0.7.mp3](samples/ultravox-v0.7.mp3) |
| GPT Realtime  | [gpt-realtime.mp3](samples/gpt-realtime.mp3)   |
| Grok Realtime | [grok-realtime.mp3](samples/grok-realtime.mp3) |
| Gemini Live   | [gemini-live.mp3](samples/gemini-live.mp3)     |
| Nova 2 Sonic  | [nova-2-sonic.mp3](samples/nova-2-sonic.mp3)   |

In the recordings, the left channel is the user's speech and the right channel is the model's speech. You'll hear a small audio tag (a short beep) at the start of each model speech segment. The benchmarking code adds this audio tag to make it easier to double-check track alignment and to provide a visual indicator of segment start for manually inspecting the recordings.

![Visually inspecting silence padding by looking at the audio waveform](samples/silence-padding-measurement.png)

*Visually inspecting silence padding by looking at the audio waveform. Here we're using the [Descript](https://descript.com/) editor.*

## Quick Start

```bash
# Install dependencies
uv sync

# List available benchmarks
uv run multi-turn-eval list-benchmarks

# Run a benchmark. Results will be saved to runs/aiwf_medium_context/<timestamp>_<model_name>
uv run multi-turn-eval run aiwf_medium_context --model claude-sonnet-4-5 --service anthropic

# Judge the results
uv run multi-turn-eval judge runs/aiwf_medium_context/<timestamp>_claude-sonnet-4-5
```

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd multi-turn-eval
uv sync
```

On Intel macOS (`x86_64`), `torch`/`torchaudio` wheels for the latest versions are not published. The project skips those dependencies on that platform so `uv sync` can complete normally.

## Environment Variables

Set the appropriate API keys for the services you want to use:

```bash
# Required for judging (Claude evaluates all benchmark results)
export ANTHROPIC_API_KEY=sk-ant-...

# Text-mode model services
export OPENAI_API_KEY=sk-...          # OpenAI (GPT-4o, GPT-4o-mini, etc.)
export GOOGLE_API_KEY=...             # Google (Gemini models)
export OPENROUTER_API_KEY=...         # OpenRouter (access to multiple providers)
export GROQ_API_KEY=...               # Groq
export CEREBRAS_API_KEY=...           # Cerebras

# Speech-to-speech model services
export OPENAI_API_KEY=sk-...          # OpenAI Realtime (gpt-realtime)
export GOOGLE_API_KEY=...             # Gemini Live (gemini-native-audio)
export ULTRAVOX_API_KEY=...           # Ultravox (ultravox-v0.7)
export XAI_API_KEY=...                # xAI Grok Realtime (grok-realtime)

# AWS Nova models (text and speech-to-speech)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...          # Optional: for temporary credentials
export AWS_REGION=us-east-1           # Optional: defaults to us-east-1
```

You can also create a `.env` file in the project root with these variables.

## CLI Commands

### Running Benchmarks

```bash
# Basic usage with text model
uv run multi-turn-eval run <benchmark> --model <model> --service <service>

# Examples:
uv run multi-turn-eval run aiwf_medium_context --model claude-sonnet-4-5 --service anthropic
uv run multi-turn-eval run aiwf_medium_context --model gpt-4o --service openai
uv run multi-turn-eval run aiwf_medium_context --model gemini-2.5-flash --service google

# Realtime audio models 
uv run multi-turn-eval run aiwf_medium_context --model gpt-realtime --service openai-realtime
uv run multi-turn-eval run aiwf_medium_context --model gemini-2.5-flash-native-audio-preview-12-2025 --service gemini-live
uv run multi-turn-eval run aiwf_medium_context --model ultravox-v0.7 --service ultravox-realtime

# Nova Sonic (no --service needed, pipeline creates its own LLM)
uv run multi-turn-eval run aiwf_medium_context --model amazon.nova-2-sonic-v1:0 --pipeline nova-sonic

# Grok (xAI) Realtime
uv run multi-turn-eval run aiwf_medium_context --model grok-realtime

# Debug with limited turns
uv run multi-turn-eval run aiwf_medium_context --model gpt-4o --service openai --only-turns 0,1,2

# Verbose logging
uv run multi-turn-eval run aiwf_medium_context --model gpt-4o --service openai --verbose

# Run multiple full 30-turn evaluations sequentially (e.g., 10 runs => 300 turns aggregate)
uv run multi-turn-eval run aiwf_medium_context --model moonshotai/kimi-k2-instruct-0905 --service openrouter --runs 10
```

Each run now also writes `native_summary.json` in the run directory (computed directly from `transcript.jsonl`, no judge required). When `--runs` is greater than 1, a single parent run directory is created with per-iteration outputs under `iterations/run_XX/`, plus `native_aggregate.json` in the parent directory.

### Judging Runs

After a benchmark run completes, judge the results using Claude:

```bash
# Judge a specific run
uv run multi-turn-eval judge runs/aiwf_medium_context/20251213T123456_claude-sonnet-4-5

# If the run was executed with --runs > 1, judge the parent run directory.
# The CLI judges each iterations/run_XX/ folder and writes an aggregate claude_summary.json at the parent.
uv run multi-turn-eval judge runs/aiwf_medium_context/20251213T123456_claude-sonnet-4-5

# Judge only specific iterations from a parent run (and still write aggregate summary at parent)
uv run multi-turn-eval judge runs/aiwf_medium_context/20251213T123456_claude-sonnet-4-5 --only-runs 1,3,9

# Judge with specific turns
uv run multi-turn-eval judge runs/aiwf_medium_context/20251213T123456_claude-sonnet-4-5 --only-turns 0,1,2

# Use a different judge model
uv run multi-turn-eval judge runs/aiwf_medium_context/20251213T123456_claude-sonnet-4-5 --judge-model claude-sonnet-4-5
```

Judge outputs (saved to the run directory):
- `claude_summary.json` - Score metrics
- `claude_analysis.md` - Human-readable report with failures
- `claude_judged.jsonl` - Per-turn judgments with reasoning

#### Judging Speech-to-Speech Runs

For speech-to-speech model runs (OpenAI Realtime, Gemini Live, Ultravox, etc.), the judge automatically performs **turn-taking analysis** when a `conversation.wav` file is present. This adds a 4th evaluation dimension:

| Dimension | Description |
|-----------|-------------|
| `turn_taking` | Audio timing correctness (pre-computed from WAV analysis) |
| `tool_use_correct` | Function calls match expected |
| `instruction_following` | Response follows system prompt |
| `kb_grounding` | Facts match knowledge base |

Turn-taking failures are detected when:
- **Missing timing data**: No user speech detected before bot response
- **Negative TTFB**: Bot started speaking before user finished
- **Empty response**: Model returned only control tokens (e.g., `<ctrl46>`)
- **No response**: Model never responded (15s timeout)
- **Alignment drift**: Audio tag positions drift >150ms from expected
- **Audio overlap**: User and bot speaking simultaneously

When turn-taking failures occur, the judge is more lenient on `instruction_following` since garbled audio may cause transcription issues.

```bash
# Judge a speech-to-speech run (turn-taking analysis runs automatically)
uv run multi-turn-eval judge runs/aiwf_medium_context/20260111T123456_gpt-realtime_abc123

# Skip turn-taking analysis
uv run multi-turn-eval judge runs/aiwf_medium_context/20260111T123456_gpt-realtime_abc123 --skip-turn-taking
```

Additional outputs for speech-to-speech runs:
- `claude_summary.json` includes `turn_taking_failures` list and `turn_taking_affected_instruction` count
- `claude_analysis.md` includes Turn-Taking Analysis section with per-turn issues
- `claude_judged.jsonl` includes `turn_taking_issues` array for failed turns

### Listing Options

```bash
# List available benchmarks
uv run multi-turn-eval list-benchmarks

# List available pipelines
uv run multi-turn-eval list-pipelines

# List service aliases
uv run multi-turn-eval list-aliases
```

## Service Aliases

For convenience, common service classes have short aliases:

| Alias | Service Class |
|-------|---------------|
| `openai` | `pipecat.services.openai.llm.OpenAILLMService` |
| `openai-realtime` | `pipecat.services.openai.realtime.llm.OpenAIRealtimeLLMService` |
| `anthropic` | `pipecat.services.anthropic.llm.AnthropicLLMService` |
| `google` | `pipecat.services.google.llm.GoogleLLMService` |
| `gemini-live` | `multi_turn_eval.pipelines.realtime.GeminiLiveLLMServiceWithReconnection` |
| `bedrock` | `pipecat.services.aws.llm.AWSBedrockLLMService` |
| `ultravox-realtime` | `pipecat.services.ultravox.llm.UltravoxRealtimeLLMService` |

You can also use fully-qualified class names:

```bash
uv run multi-turn-eval run aiwf_medium_context \
    --model gpt-4o \
    --service pipecat.services.openai.llm.OpenAILLMService
```

## Benchmarks

Benchmarks are located in `benchmarks/`. Each benchmark is a Python package with:
- `config.py` - Benchmark configuration (turns, tools, system instruction)
- `prompts/system.py` - System prompt with knowledge base
- `data/knowledge_base.txt` - Knowledge base content

### Available Benchmarks

| Benchmark | Description | Knowledge Base |
|-----------|-------------|----------------|
| `aiwf_long_context` | Long context benchmark | ~40K tokens |
| `aiwf_medium_context` | Medium context benchmark | ~12K tokens |

Both benchmarks share the same 30 turns, tools, and audio files. Only the knowledge base size differs.

## Pipelines

| Pipeline | Use Case | Auto-Detection Pattern |
|----------|----------|------------------------|
| `text` | Synchronous text LLMs | Default for all models |
| `realtime` | OpenAI Realtime, Gemini Live, Ultravox Realtime | `*realtime*`, `*native-audio*`, `*live*`, `*ultravox*` |
| `nova-sonic` | AWS Nova Sonic | `*nova-sonic*`, `*nova_sonic*` |

## Output Structure

Runs are saved to `runs/<benchmark>/<timestamp>_<model>/`:

```
runs/
└── aiwf_medium_context/
    └── 20251213T123456_claude-sonnet-4-5/
        ├── transcript.jsonl        # Turn-by-turn results
        ├── runtime.json            # Run metadata and metrics
        ├── run.log                 # Debug logs
        ├── claude_summary.json     # Judge summary (after judging)
        ├── claude_judged.jsonl     # Per-turn judgments (after judging)
        └── claude_analysis.md      # Human-readable analysis (after judging)
```

## Tested Models

| Model | Pipeline | Service |
|-------|----------|---------|
| `gpt-4o` | text | openai |
| `gpt-4o-mini` | text | openai |
| `gpt-realtime` | realtime | openai-realtime |
| `gemini-2.5-flash` | text | google |
| `gemini-2.5-flash-native-audio-preview-12-2025` | realtime | gemini-live |
| `ultravox-v0.7` | realtime | ultravox-realtime |
| `claude-sonnet-4-5` | text | anthropic |
| `claude-haiku-4-5` | text | anthropic |
| `amazon.nova-2-sonic-v1_0` | nova-sonic | (built-in) |

## Project Structure

```
multi-turn-eval/
├── src/multi_turn_eval/           # Main package
│   ├── cli.py                     # CLI entry point
│   ├── pipelines/                 # Pipeline implementations
│   │   ├── base.py                # Abstract base pipeline
│   │   ├── text.py                # Text pipeline
│   │   ├── realtime.py            # Realtime pipeline (OpenAI/Gemini)
│   │   └── nova_sonic.py          # Nova Sonic pipeline
│   ├── processors/                # Frame processors
│   │   ├── tool_call_recorder.py  # Records tool calls
│   │   └── tts_transcript.py      # TTS transcript handling
│   ├── transports/                # Input/output transports
│   │   ├── paced_input.py         # Paced audio input
│   │   └── null_audio_output.py   # Null audio sink
│   ├── recording/                 # Transcript recording
│   │   └── transcript_recorder.py # Records transcripts
│   └── judging/                   # Judge implementations
│       └── claude_judge.py        # Claude-based judging
│
├── benchmarks/                    # Benchmark definitions
│   ├── _shared/                   # Shared benchmark data
│   │   ├── turns.py               # 30 turns with golden data
│   │   ├── tools.py               # Tool/function definitions
│   │   └── audio/                 # Audio files for turns
│   ├── aiwf_long_context/         # Long context benchmark
│   └── aiwf_medium_context/       # Medium context benchmark
│
├── runs/                          # Output directory (gitignored)
├── scripts/                       # Utility scripts
└── pyproject.toml                 # Project configuration
```

## Using Pre-release Pipecat Versions

To use a git branch of pipecat instead of the PyPI release, edit `pyproject.toml`:

```toml
[tool.uv.sources]
pipecat-ai = { git = "https://github.com/pipecat-ai/pipecat.git", rev = "main" }
```

Then run `uv sync` to update.

## Evaluation Dimensions

The Claude judge evaluates each turn on three dimensions:

1. **tool_use_correct** - Did the assistant call the expected function with correct arguments?
2. **instruction_following** - Did the assistant answer the question or advance the task?
3. **kb_grounding** - Is the response factually consistent with the knowledge base?

## Comprehensive Turn Metrics Analysis

For detailed per-turn timing analysis of speech-to-speech models, use the comprehensive metrics script:

```bash
# Analyze a run with summary statistics
uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/<timestamp>_<model>

# Show per-turn breakdown table
uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/<timestamp>_<model> -v

# Output as JSON (for programmatic use)
uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/<timestamp>_<model> --json
```

### Metrics Explained

The script consolidates timing data from multiple sources and calculates the following metrics:

| Metric | Description | Calculation |
|--------|-------------|-------------|
| **Server TTFB** | Time from request to first byte from model | Read from `transcript.jsonl` (reported by Pipecat) |
| **Pipeline TTFB** | Time from user speech end to bot audio tag | `bot_tag_log_ms - user_end_ms` (Silero VAD) |
| **WAV V2V** | Voice-to-voice latency measured from audio | `bot_silero_start_ms - user_end_ms` (Silero VAD) |
| **Silent Pad (RMS)** | Silent padding before speech (RMS detection) | `bot_rms_onset_ms - bot_tag_log_ms` |
| **Silent Pad (VAD)** | Silent padding before speech (Silero VAD) | `bot_silero_start_ms - bot_tag_wav_ms` |
| **Tag Alignment** | Drift between log position and WAV detection | `bot_tag_log_ms - bot_tag_wav_ms` |

**Key metric relationships:**
- **WAV V2V = Pipeline TTFB + Silent Pad (VAD)** - The total voice-to-voice latency includes both the time waiting for audio to arrive and any initial silence in the audio stream
- **Pipeline TTFB** measures when audio starts arriving at the pipeline
- **Silent Pad** measures how much silence is at the beginning of the audio (most models send 40-120ms of silence before speech)

### Alignment Sanity Check

The script verifies that log-based timestamps match actual audio positions by detecting audio tags (2kHz tones) embedded in the WAV file:

- **Bot tags**: Inserted when bot audio arrives at the pipeline
- **Alignment OK**: Log and WAV positions match within ±20ms tolerance
- **Issues detected**: Missing tags, extra tags, or drift outside tolerance

### Output Files

When run with `--json`, the script outputs structured data that can be saved:

```bash
# Save metrics to JSON file
uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/<timestamp>_<model> --json > turn_metrics.json
```

### Claude Code Prompt for Batch Benchmarking

Use this prompt with Claude Code to run comprehensive benchmarks across multiple speech-to-speech models:

```
Run a full 30-turn test with all four speech-to-speech models: ultravox-v0.7,
gpt-realtime, grok-realtime, gemini-2.5-flash-native-audio-preview-12-2025.

For each model:
1. Run the 30-turn benchmark
2. Analyze using scripts/analyze_turn_metrics.py and save turn_metrics.json
3. Judge the model performance using the Claude judge

After completing all models, create a summary comparison table with these columns:
- Model
- Tool Use (X/30)
- Instruction (X/30)
- KB Ground (X/30)
- Turn Ok (X/30)
- Pass Rate
- Non-Tool V2V Median
- Non-Tool V2V Max
- Tool V2V Mean
- Silence Pad Mean

Separate metrics for tool-call turns vs non-tool-call turns in the analysis.
```

This will run all four models (which takes approximately 15-20 minutes each), analyze their timing metrics, judge their responses, and produce a comparison table.

## License

MIT
