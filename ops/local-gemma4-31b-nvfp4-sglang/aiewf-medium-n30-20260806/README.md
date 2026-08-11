# Local RTX 5090 Gemma 4 31B NVFP4 AIEWF campaign

This is the frozen 30-conversation, no-filler, thinking-off comparison cohort
for `RedHatAI/gemma-4-31B-it-NVFP4`. It matches the BaseTen BF16 campaign's
sampling parameters and fixed 900-turn denominator. The only intended model
quality change is the NVFP4 checkpoint; MTP is disabled.

After the serving gates in the parent directory pass, execute or resume with:

```bash
.venv/bin/python \
  ops/local-gemma4-31b-nvfp4-sglang/aiewf-medium-n30-20260806/run_campaign.py
```

The wrapper owns the Docker server lifecycle and stops it in a `finally`
block. A model-caused short conversation after its first valid response stays
in the cohort; infrastructure attempts with no valid response are replaced.
