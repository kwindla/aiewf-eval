# BaseTen Qwen3-8B benchmark deployment

This Truss deploys the official `Qwen/Qwen3-8B` BF16 checkpoint at the pinned
revision in `config.yaml`. It is the dedicated serving route for the clean
BaseTen replacement cohort in the focused 96-dot stability campaign.

The deployment is intentionally configured for the benchmark rather than as a
general public endpoint:

- one H100 (80 GiB VRAM);
- vLLM 0.25.1 pinned by image digest;
- 32,768-token maximum context;
- automatic prefix caching;
- Qwen/Hermes automatic tool-call parsing;
- thinking disabled by default, with request-level overrides still possible;
- official BF16 weights, with no weight quantization.

The campaign client sends the non-thinking sampling settings recommended by the
Qwen model card: temperature 0.7, top-p 0.8, top-k 20, and min-p 0 (the vLLM
default). It records the content-aware per-turn TTFAT from the streaming API.

No API key or endpoint identifier belongs in this directory. Deployment and
autoscaling identifiers are recorded in the campaign state after a successful
push.
