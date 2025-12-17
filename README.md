# Multi-Turn Eval

A framework for evaluating multi-turn LLM conversations with support for text, realtime audio, and speech-to-speech models.

The two benchmarks here in this public repo are:

- `aiwf_long_context` - older long-context benchmark described [here](https://post-training.aitinkerers.org/p/your-conversation-is-out-of-distribution)
- `aiwf_medium_context` - newer medium-context benchmark

## aiwf_medium_context results summary for selected models

Text mode models:

```
| Model                  | Tool Use  | Instruction | KB Ground | Pass Rate | Median Rate | TTFB Med | TTFB P95 | TTFB Max |
|------------------------|-----------|-------------|-----------|-----------|-------------|----------|----------|----------|
| gpt-5.1                | 300/300   | 300/300     | 300/300   | 100.0%    | 100.0%      | 916ms    | 2011ms   | 5216ms   |
| gemini-3-flash-preview | 300/300   | 300/300     | 300/300   | 100.0%    | 100.0%      | 1193ms   | 1635ms   | 6653ms   |
| claude-sonnet-4-5      | 300/300   | 300/300     | 300/300   | 100.0%    | 100.0%      | 2234ms   | 3062ms   | 5438ms   |
| gpt-4.1                | 283/300   | 273/300     | 298/300   | 94.9%     | 97.8%       | 683ms    | 1052ms   | 3860ms   |
| gemini-2.5-flash       | 275/300   | 268/300     | 300/300   | 93.7%     | 94.4%       | 594ms    | 1349ms   | 2104ms   |
| gpt-5-mini             | 271/300   | 272/300     | 289/300   | 92.4%     | 95.6%       | 6339ms   | 17845ms  | 27028ms  |
| gpt-4o-mini            | 271/300   | 262/300     | 293/300   | 91.8%     | 92.2%       | 760ms    | 1322ms   | 3256ms   |
| gpt-4o                 | 278/300   | 249/300     | 294/300   | 91.2%     | 95.6%       | 625ms    | 1222ms   | 13378ms  |
| gpt-5.2                | 224/300   | 228/300     | 250/300   | 78.0%     | 92.2%       | 819ms    | 1483ms   | 1825ms   |
| claude-haiku-4-5       | 221/300   | 172/300     | 299/300   | 76.9%     | 75.6%       | 732ms    | 1334ms   | 4654ms   |
```

Speech-to-speech models:

```
|   Model                         | Tool Use  | Instruction | KB Ground | Pass Rate | Median Rate | TTFB Med |
|---------------------------------|-----------|-------------|-----------|-----------|-------------|----------|
|   gpt-realtime                  | 267/300   | 265/300     | 300/300   | 92.4%     | 92.8%       | 818ms    |
|   gemini-native-audio-12-2025   | 253/300   | 259/300     | 286/300   | 88.7%     | 90.0%       | N/A      |
|   gemini-native-audio-09-2025   | 236/300   | 227/300     | 268/300   | 81.2%     | 89.4%       | 785ms    |
| * amazon.nova-2-sonic-v1:0      | 278/300   | 265/300     | 296/300   | 93.2%     | 95.6%       | *        |
```

Each conversation in this benchmark is 30 turns. The scores above are aggregated across 10 runs for each model. **Pass Rate** means the percentage of total turns across all runs that the judge model scored as successful. Each run is also scored independently. **Median Rate** is the median individual run pass rate. Think of pass rate as the model's average performance, and the median rate as a way to measure the model's consistency. The older gemini-native-audio-release, for example, often gave very good performance (89.4% median rate), but was prone to poor runs (81.2% pass rate). The newer release is much more consistent (the overall pass rate is much closer to the median rate).

The new AWS Nova 2 Sonic model is marked with an asterisk (*). It is the best speech-to-speech model in this benchmark, **when we complete a full 30-turn conversation**. But performance is unstable in a way that is not captured in this summary table: content refusals sometimes happen early in a conversation and the model never recovers; there is an 8m connection limit and reloading conversation history is fragile. This needs more investigation. Both of these may be Pipecat implementation issues. For the moment, we're ignoring incomplete runs and including complete-run numbers to show the model's promise. But we expect to see some changes to the implementation before it can be used in production (improvements to either in the Pipecat implementation, the AWS APIs, or both).


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

# Realtime audio models (pipeline auto-detected from model name)
uv run multi-turn-eval run aiwf_medium_context --model gpt-realtime --service openai-realtime
uv run multi-turn-eval run aiwf_medium_context --model gemini-2.5-flash-native-audio-preview-12-2025 --service gemini-live

# Nova Sonic (no --service needed, pipeline creates its own LLM)
uv run multi-turn-eval run aiwf_medium_context --model amazon.nova-2-sonic-v1:0 --pipeline nova-sonic

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
| `realtime` | OpenAI Realtime, Gemini Live | `*realtime*`, `*native-audio*`, `*live*` |
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

## License

MIT
