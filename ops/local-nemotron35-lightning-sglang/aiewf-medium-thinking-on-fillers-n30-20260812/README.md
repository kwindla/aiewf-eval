# Nemotron 3.5 Lightning thinking-on filler campaign

This campaign tests 96 trailing dots and 96 trailing dashes with the successful
thinking-on/unbounded Nemotron 3.5 Lightning configuration. Each arm contains
30 complete assigned AIEWF conversations. The frozen no-filler comparison is
the 30-run `on-unbounded` arm from the completed binary campaign.

The two fresh arms alternate order within pairs and run strictly sequentially.
Every request uses `enable_thinking=true`, `force_nonempty_content=true`,
temperature 1.0, top-p 0.95, and no output or thinking cap. Filler is appended
only to a copy of the current final user message; persisted history and judged
transcripts remain filler-free. `/flush_cache` is called between conversations,
while normal Radix prefix reuse remains enabled within each conversation.

The first attempt with a valid model response is canonical. Model-caused early
termination or a later stall is retained, with missing future turns failing on
the fixed 900-turn denominator per arm.

Run collection from the repository root while the admitted SGLang endpoint is
available on port 8000:

```bash
LOCAL_NEMOTRON_API_KEY=dummy .venv/bin/python \
  ops/aiewf-campaign-template/collect.py \
  --config ops/local-nemotron35-lightning-sglang/aiewf-medium-thinking-on-fillers-n30-20260812/configuration.json \
  --execute
```

Then run `judge.sh` and `analyze.py` in this directory.
