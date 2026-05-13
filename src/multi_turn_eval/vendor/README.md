# Vendored upstream code

## `nemotron_omni.py`

A verbatim copy of upstream's
`../nemotron-nano-omni/src/nemotron_voice/services/nvidia/nemotron_omni.py`.

- Source repo: `nemotron-nano-omni` (local sibling working copy)
- Source branch: `khk/cache-refactor-5090`
- Pinned upstream commit: `6100582` ("Keep common vLLM patch separate from Spark patch")
- Vendored on: 2026-05-13

Do **not** edit this file in-tree. Refresh by re-copying from upstream.

### Refresh procedure

```bash
cd /home/khkramer/src/nemotron-nano-omni
git pull
git rev-parse --short HEAD  # record the new commit hash
cd /home/khkramer/src/aiewf-eval
cp ../nemotron-nano-omni/src/nemotron_voice/services/nvidia/nemotron_omni.py \
   src/multi_turn_eval/vendor/nemotron_omni.py
# update the "Pinned upstream commit" line above
uv run pytest tests -v
# run the audio-in smoke tests + statistical run before declaring the refresh good
```

### vLLM server-side patches

The conversation-cache contract requires matching engine patches on the
vLLM server. As of upstream commit `6100582` those live in
`../nemotron-nano-omni/platforms/common/patches/vllm-nemotron-omni-conversation-cache.patch`
(plus a Spark-specific overlay). Confirm with the server admin that the
current patch set is applied before running cache-on benchmarks.
