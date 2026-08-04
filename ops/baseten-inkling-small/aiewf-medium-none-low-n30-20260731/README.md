# Inkling Small BaseTen AIEWF campaign

This bundle runs 30 valid `reasoning_effort=none` and 30 valid
`reasoning_effort=low` conversations against BaseTen's shared Model API model
`thinkingmachines/inkling-small`.

The 60 assignments form 30 temporal pairs and run strictly sequentially. Five
six-pair blocks each contain three `none`-first and three `low`-first pairs. The
order was frozen with seed `20260731` before any campaign outcome existed.

Both arms pin temperature 1.0 and 16,384 maximum completion tokens. Thinking is
controlled only by the top-level `reasoning_effort`; the vLLM-style
`enable_thinking` knob and all filler variables are removed from the child
environment.

An attempt becomes canonical after its first valid model response. Early model
termination, later malformed output, and later idle timeouts remain measured
outcomes. Only objective provider or transport failures with zero valid model
responses may be replaced, up to four attempts per slot.

The collector is read-only by default:

```bash
.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-low-n30-20260731/collect.py
```

After both excluded smoke gates pass, execute the campaign with `--execute`.
The collector loads only `BASETEN_API_KEY` from the sibling `gb-benchmarks/.env`
when it is absent from the process environment. It never prints the key.

This is a serverless Model API campaign; there is no deployment to tear down.
