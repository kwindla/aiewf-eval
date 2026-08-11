# Local Gemma 4 31B BF16-KV N=120 extension

This campaign adds 120 conversations to the immutable 30-conversation local
NVFP4-weights + compact BF16-KV cohort. The pooled analysis therefore uses 150
conversations and 4,500 fixed-denominator turns.

```bash
.venv/bin/python ops/local-gemma4-31b-nvfp4-sglang/run_extension.py \
  --kv-cache bf16
```
