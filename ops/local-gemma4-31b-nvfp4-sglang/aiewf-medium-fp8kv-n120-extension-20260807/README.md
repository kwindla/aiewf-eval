# Local Gemma 4 31B FP8-KV N=120 extension

This campaign adds 120 conversations to the immutable 30-conversation local
NVFP4-weights + FP8-KV cohort. The pooled analysis therefore uses 150
conversations and 4,500 fixed-denominator turns.

```bash
.venv/bin/python ops/local-gemma4-31b-nvfp4-sglang/run_extension.py \
  --kv-cache fp8
```
