# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/multi_turn_eval/`. Start with `cli.py` for the entrypoint, then `pipelines/`, `processors/`, `transports/`, `recording/`, and `judging/` for execution flow. Benchmark definitions live in `benchmarks/`; each benchmark package includes `config.py`, `prompts/system.py`, and `data/knowledge_base.txt`. Utility scripts are in `scripts/`, reference material is in `docs/`, sample media is in `samples/`, and generated run artifacts go to `runs/` (treat as local output, not source).

## Build, Test, and Development Commands
Use `uv` with Python 3.12+.

- `uv sync` installs dependencies from `pyproject.toml` and `uv.lock`.
- `uv run multi-turn-eval list-benchmarks` shows available benchmark packages.
- `uv run multi-turn-eval run aiwf_medium_context --model gpt-4o --service openai --only-turns 0,1,2` runs a fast debug slice.
- `uv run multi-turn-eval judge runs/aiwf_medium_context/<timestamp>_<model>_<id>` scores a completed run.
- `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/<timestamp>_<model>_<id>` summarizes timing behavior for speech runs.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, type hints on public functions, and small, focused modules. Use `snake_case` for files, functions, and variables; `PascalCase` for classes; keep benchmark package names lowercase with underscores. There is no repo-level formatter or linter config checked in, so keep imports, docstrings, and logging consistent with nearby files before submitting changes.

## Testing Guidelines
There is no dedicated `tests/` directory today. Validate changes with the smallest relevant benchmark run, usually `--only-turns` against `aiwf_medium_context`, then re-run `judge` on the output directory. If you change timing, audio, or transcript logic, also run `scripts/analyze_turn_metrics.py` and note the affected run path in your PR.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `credit Modal for compute` and `re-score Nemotron 3 Nano with judging fixes`. Keep commit messages concise and specific to one logical change. PRs should explain the benchmark or pipeline area touched, list validation commands you ran, link any related issue, and include sample output paths or screenshots when changes affect reports, docs, or audio-analysis output.

## Configuration Tips
Keep API keys in a local `.env`; `load_dotenv()` is called from the CLI. Do not commit secrets, generated `runs/` data, or ad hoc benchmark outputs.
