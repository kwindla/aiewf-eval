# BaseTen Kimi K2.6 thinking-on AIEWF campaign

This bundle collects 30 complete, valid AIEWF medium-context conversations
against BaseTen's `moonshotai/Kimi-K2.6` Model API route. The request explicitly
sends `extra_body.chat_template_args.enable_thinking=true` while omitting
`reasoning_effort`. It uses no filler, temperature 1.0, top-p 0.95, and
`max_tokens=8192`, matching the portable sampling signature of the historical
Cerebras thinking-on campaign.

The exact full-conversation smoke at
`runs/aiwf_medium_context/20260806T173519Z_moonshotai_Kimi-K2.6_thinking_SMOKE_attempt01`
is canonical slot `K26T-01`. It completed all 30 scripted turns with complete
token accounting and positive `thinking_tokens` on all 30 turns. A preceding
standalone streaming probe directly observed `reasoning_content` deltas before
visible content. The smoke's median raw first-chunk latency was 718 ms, versus
1,715 ms for content/tool TTFAT, validating the content-aware timing path.

Eligibility requires runtime `completed` and `valid`, all 30 scripted response
rows, and all 30 usage rows. `end_session` timing or absence remains judgeable
model performance and never controls eligibility. Every attempt is retained.
Collection is serial with a 30-second cooldown before every post-smoke attempt
because the shared BaseTen endpoint previously returned rolling 429s.

Read-only preflight:

```bash
.venv/bin/python ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/collect.py
```

Start or resume:

```bash
.venv/bin/python ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/collect.py --execute
```

Only `BASETEN_API_KEY` is read from `../gb-benchmarks/.env` as a fallback.
