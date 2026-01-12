# Multi-Turn Eval

A framework for evaluating multi-turn LLM conversations with support for text, realtime audio, and speech-to-speech models.

The two benchmarks here in this public repo are:

- `aiwf_long_context` - older long-context benchmark described [here](https://post-training.aitinkerers.org/p/your-conversation-is-out-of-distribution)
- `aiwf_medium_context` - newer medium-context benchmark

## aiwf_medium_context results summary for selected models

Text mode models:

```
| Model                     | Tool Use  | Instruction | KB Ground | Pass Rate | Median Rate | TTFB Med | TTFB P95 | TTFB Max |
|---------------------------|-----------|-------------|-----------|-----------|-------------|----------|----------|----------|
| gpt-5.1                   | 300/300   | 300/300     | 300/300   | 100.0%    | 100.0%      |  916ms   | 2011ms   | 5216ms   |
| gemini-3-flash-preview    | 300/300   | 300/300     | 300/300   | 100.0%    | 100.0%      | 1193ms   | 1635ms   | 6653ms   |
| claude-sonnet-4-5         | 300/300   | 300/300     | 300/300   | 100.0%    | 100.0%      | 2234ms   | 3062ms   | 5438ms   |
| gpt-4.1                   | 283/300   | 273/300     | 298/300   | 94.9%     | 97.8%       | 683ms    | 1052ms   | 3860ms   |
| gemini-2.5-flash          | 275/300   | 268/300     | 300/300   | 93.7%     | 94.4%       |  594ms   | 1349ms   | 2104ms   |
| nova-2-pro-preview        | 288/300   | 278/300     | 289/300   | 92.7%     | 93.3%       |  686ms   |  750ms   | 1459ms   |
| gpt-5-mini                | 271/300   | 272/300     | 289/300   | 92.4%     | 95.6%       | 6339ms   | 17845ms  | 27028ms  |
| gpt-4o-mini               | 271/300   | 262/300     | 293/300   | 91.8%     | 92.2%       |  760ms   | 1322ms   | 3256ms   |
| gpt-4o                    | 278/300   | 249/300     | 294/300   | 91.2%     | 95.6%       |  625ms   | 1222ms   | 13378ms  |
| nemotron-3-nano-30b-a3b * | 282/300   | 280/300     | 293/300   | 91.0%     | 93.3%       |  171ms   |  199ms   |  255ms   |
| gpt-oss-120b (groq)       | 272/300   | 270/300     | 298/300   | 89.3%     | 90.0%       |   98ms   |  226ms   | 2117ms   |
| gpt-5.2                   | 224/300   | 228/300     | 250/300   | 78.0%     | 92.2%       |  819ms   | 1483ms   | 1825ms   |
| claude-haiku-4-5          | 221/300   | 172/300     | 299/300   | 76.9%     | 75.6%       |  732ms   | 1334ms   | 4654ms   |

* [ Nemotron 3 Nano running in-cluster on NVIDIA Blackwell hardware ]
```

Each conversation in this benchmark is 30 turns. The scores above are aggregated across 10 runs for each model. **Pass Rate** means the percentage of total turns across all runs that the judge model scored as successful. Each run is also scored independently. **Median Rate** is the median individual run pass rate. Think of pass rate as the model's average performance, and the median rate as a way to measure the model's consistency. The older gemini-native-audio-release, for example, often gave very good performance (89.4% median rate), but was prone to poor runs (81.2% pass rate). The newer release is much more consistent (the overall pass rate is much closer to the median rate).

TTFB is the number reported by the Pipecat service for each model. It is the time from the request to generate inference to the first byte of the response. An optimized speech-to-speech pipeline with typical network latencies should be able to achieve a total voice-to-voice latency of approximately LLM TTFB + 500ms.

Speech-to-speech models:

```
-----------------------------------------------------------------------------------------------------------------------------------------
| Model             | Tool    | Instruction | KB       | Turn    | Pass     | Non-Tool V2V  | Non-Tool V2V  | Tool V2V   | Silence Pad  |
|                   | Use     |             | Ground   | Ok      | Rate     | Med           | Max           | Mean       | Mean         |
-----------------------------------------------------------------------------------------------------------------------------------------
| ultravox-v0.7     | 293/300 | 294/300     | 298/300  | 300/300 |   97.7%  | 864ms         | 1888ms        | 2406ms     | 82ms         |
-----------------------------------------------------------------------------------------------------------------------------------------
| gpt-realtime      | 271/300 | 260/300     | 300/300  | 296/300 |   86.7%  | 1536ms        | 4672ms        | 2199ms     | 341ms        |
-----------------------------------------------------------------------------------------------------------------------------------------
| gemini-live       | 258/300 | 261/300     | 293/300  | 278/300 |   86.0%  | 2624ms        | 61747ms       | 4082ms     | 90ms         |
-----------------------------------------------------------------------------------------------------------------------------------------
| * nova-2-sonic    | 278/300 | 265/300     | 296/300  | *       |  (93.2%) | *             | *             | *          | *            |
-----------------------------------------------------------------------------------------------------------------------------------------
| * grok-realtime   | 267/300 | 275/300     | 295/300  | *       |  (89.0%) | 1184ms        | 2016ms        | 1472ms     | 478ms        |
-----------------------------------------------------------------------------------------------------------------------------------------
```

Speech-to-speech models, which take audio as input and generate audio as output. For these models, we measure voice-to-voice latency by analyzing the saved audio files and measuring the time from the end of the user's audio to the beginning of the model's speech audio response. This latency is different from the TTFB reported by the Pipecat service for these models, because all of these models were tested in server-side VAD mode (so the server-side turn delay is opaque to the Pipecat pipeline), and all of the models send initial silence bytes before actual speech audio. (Text-to-speech models do this, too. The initial silence segments are typically between 150ms and 250ms for standalone TTS models.)

We also include a "Turn Ok" column for these models, which counts how often the model does not respond at all when we expect it to. This is a difficult metric to specify precisely. Are general API failures and disconnects turn failures? We're conservative, here, only including in turn failures model non-responsiveness or extremely slow response while the persistent session remains connected.

The APIs for the Grok and AWS Nova 2 Sonic models are currently unreliable enough that to use them in production would require a very large amount of defensive programming at the application level. These are the second and third best performing models, **when they complete a full 30-turn conversation**. But performance is unstable: the AWS model frequently emits content refusals for normal content and has an 8 minute connection limit; its context prefill often fails with errors; the Grok API returns errors for all actions in the middle of a session and may or may not recover.

## Model Notes


### Ultravox v0.7

**Strengths:**
- **Highest overall quality** (97.7% pass rate) - best tool use and instruction following
- **Perfect turn-taking** (300/300) - no timing anomalies across 300 turns
- **Fastest voice-to-voice latency** (864ms median) - most responsive for non-tool turns
- **Low silence padding** (82ms mean) - minimal delay before speech starts

**Weaknesses:**
- **Slower tool call responses** (2406ms mean) - highest latency when calling functions (local TTS processing)
- **Occasional end_session miss** - sometimes fails to call the session-ending function

### GPT-Realtime (OpenAI)

**Strengths:**
- **Perfect KB grounding** (300/300) - never hallucinates facts from the knowledge base
- **Excellent turn-taking** (296/300, 98.7%) - natural conversation flow
- **Consistent latency** (4672ms max vs 61747ms for Gemini) - fewer extreme outliers

**Weaknesses:**
- **Lower instruction following** (260/300, 86.7%) - sometimes ignores system prompt constraints
- **Mid-conversation tool misses** - often fails to call functions like dietary_request, vote_session
- **Moderate silence padding** (341ms mean) - noticeable pause before speech starts

### Grok-Realtime (xAI)

**Strengths:**
- **Fastest tool call responses** (1472ms mean) - best for function-calling use cases
- **Strong instruction following** (275/300, 91.7%) - second best after Ultravox
- **Consistent latency** (2016ms max) - most predictable response times
- **Good V2V median** (1184ms) - competitive with GPT-Realtime

**Weaknesses:**
- **30-minute session limit** - xAI terminates connections after 30 minutes. When hit, the pipeline can get stuck in an error loop (receiving error frames prevents inactivity timeout). Recommendation: implement session keepalive or pre-emptive refresh.
- **Server-side VAD disagreement** - xAI's VAD sometimes fails to detect user speech end, causing `missing_timing_data` or `negative_ttfb` issues
- **Highest silence padding** (478ms mean) - sends more "thinking" audio before speech
- **Turn-taking failures** (279/300, 93%) - timing anomalies more common than GPT/Ultravox

### Gemini Live (Google)

**Strengths:**
- **Lowest silence padding** (90ms mean) - most natural-sounding response starts
- **Good KB grounding** (293/300, 97.7%) - reliable factual accuracy

**Weaknesses:**
- **Session instability** - sessions occasionally disconnect mid-conversation, requiring up to 9 reconnection attempts in worst cases
- **Extreme latency outliers** (61747ms max) - reconnection attempts cause massive latency spikes (over 1 minute)
- **Highest median latency** (2624ms) - notably slower than other models even when stable
- **Lowest tool use accuracy** (258/300, 86%) - struggles with mid-conversation function calls
- **Turn-taking failures** (278/300, 92.7%) - greeting timing edge cases and reconnection issues
- **Empty response issue** - sometimes returns only control tokens, requiring retry

### Nova Sonic (AWS)

Nova Sonic has an **8-minute connection limit** and requires reloading conversation history after reconnection. Content refusals can occur early in conversations, causing the model to stop responding coherently for the remainder of the session. Performance when working is competitive, but reliability issues currently limit production use.

## Features

- **Multi-turn conversation evaluation** with configurable benchmarks
- **Three pipeline types**:
  - **Text** - For synchronous text LLMs (OpenAI, Anthropic, Google, Bedrock)
  - **Realtime** - For speech-to-speech models (OpenAI Realtime, Gemini Live)
  - **Nova Sonic** - For AWS Nova Sonic with automatic reconnection
- **Claude-based judging** with detailed per-turn analysis
- **Automatic metrics collection** (TTFB, token usage, latency)

## Quick Start

```bash
# Install dependencies
uv sync

# List available benchmarks
uv run multi-turn-eval list-benchmarks

# Run a benchmark with Claude
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

## Environment Variables

Set the appropriate API keys for the services you want to use:

```bash
# For Claude (Anthropic) - also required for judging
export ANTHROPIC_API_KEY=sk-ant-...

# For OpenAI models (GPT-4o, gpt-realtime, etc.)
export OPENAI_API_KEY=sk-...

# For Google/Gemini models
export GOOGLE_API_KEY=...

# For AWS Bedrock / Nova Sonic
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1

# For OpenRouter
export OPENROUTER_API_KEY=...

# For Ultravox Realtime
export ULTRAVOX_API_KEY=...
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
```

### Judging Runs

After a benchmark run completes, judge the results using Claude:

```bash
# Judge a specific run
uv run multi-turn-eval judge runs/aiwf_medium_context/20251213T123456_claude-sonnet-4-5

# Judge with specific turns
uv run multi-turn-eval judge runs/aiwf_medium_context/20251213T123456_claude-sonnet-4-5 --only-turns 0,1,2

# Use a different judge model
uv run multi-turn-eval judge runs/aiwf_medium_context/20251213T123456_claude-sonnet-4-5 --judge-model claude-sonnet-4-5
```

Judge outputs (saved to the run directory):
- `claude_summary.json` - Score metrics
- `claude_analysis.md` - Human-readable report with failures
- `claude_judged.jsonl` - Per-turn judgments with reasoning

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

## TTFB Analysis

For speech-to-speech models, you can analyze Time-to-First-Byte (TTFB) from the recorded audio using Silero VAD (neural network-based voice activity detection):

```bash
# Analyze TTFB for a realtime run
uv run python scripts/analyze_ttfb_silero.py runs/aiwf_medium_context/<timestamp>_<model>

# Show per-turn breakdown with tool call indicators
uv run python scripts/analyze_ttfb_silero.py runs/aiwf_medium_context/<timestamp>_<model> -v

# Output as JSON
uv run python scripts/analyze_ttfb_silero.py runs/aiwf_medium_context/<timestamp>_<model> --json

# Adjust silence gap threshold (default 2000ms)
uv run python scripts/analyze_ttfb_silero.py runs/aiwf_medium_context/<timestamp>_<model> --min-silence-ms 1500
```

### How It Works

The script:
- Uses Silero VAD for accurate speech boundary detection
- Analyzes the stereo `conversation.wav` (user on left channel, bot on right)
- Segments each track independently, then pairs by index
- Calculates TTFB as the gap between user speech end and bot speech start
- Reads `transcript.jsonl` to identify which turns involved tool calls
- Automatically skips initial bot greetings (for models like Gemini that speak first)

### Output

The analysis provides separate statistics for:

1. **Overall** - All turns combined
2. **Non-Tool Call Turns** - Turns where the model responded without calling a function
3. **Tool Call Turns** - Turns where the model called one or more tools before responding

Example output:
```
======================================================================
OVERALL STATISTICS (All Turns)
======================================================================
  Count:            30 turns
  Mean:           1227ms
  Median:         1124ms
  ...

----------------------------------------------------------------------
NON-TOOL CALL TURNS
----------------------------------------------------------------------
  Count:            27 turns
  Mean:           1090ms
  Median:          868ms
  ...

----------------------------------------------------------------------
TOOL CALL TURNS (turns: [11, 12, 29])
----------------------------------------------------------------------
  Count:             3 turns
  Mean:           1295ms
  ...
```

Tool call turns typically have higher TTFB since the model must process the tool call and response before generating audio.

### Notes

- **Initial bot greeting**: Some models (e.g., Gemini native audio) emit an initial greeting before the user speaks. The script automatically detects and skips this by checking if the first bot segment starts before the first user segment ends.
- **Segment mismatch**: If the number of user and bot segments don't match, the script pairs as many as possible and reports the mismatch.
- **Negative TTFB**: Indicates overlapping speech (bot started before user finished). This may indicate audio sync issues or interruptions.

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
