# Cost-per-conversation-minute methodology

This artifact applies the official list prices frozen in
`pricing-2026-08-06.json` to one completed freshness conversation per model
configuration in `usage-results.json`.

Reproduce it from the repository root:

```bash
uv run python ops/model-freshness-2026-08-06/estimate_costs.py
uv run pytest -q ops/model-freshness-2026-08-06/test_estimate_costs.py
```

## Calculation

Estimated conversation minutes are actual user plus assistant words divided by
150 words/minute. This is intentionally not the accelerated benchmark process
time. Recovery calls, when present, remain in the billed token totals.

For token-priced APIs:

```text
cost = uncached_input × input_rate
     + cache_read × cache_read_rate
     + cache_write × five_minute_cache_write_rate
     + output × output_rate
```

Rates are converted from dollars per million tokens. Anthropic reports
uncached prompt, cache-read, and cache-write tokens separately, so its prompt
counter is not reduced. OpenAI, Google, hosted BaseTen, OpenRouter, and the
OpenAI-compatible Groq response report prompt totals that include cache hits;
the estimator subtracts cache-read tokens before applying the uncached rate.
Completion totals already include billed reasoning tokens, so the diagnostic
`thinking_tokens` value is not added again.

The aggregate prompt total across 30 turns does not trigger a long-context
price tier. Such tiers are based on each request. Every individual request in
this medium-context sample is below OpenAI's 272K-input threshold, so the
standard GPT-5.4 and GPT-5.5 rates apply.

Dedicated BaseTen deployments have no honest model-specific token list price.
Their rows assume one active conversation has exclusive use of one H100 at
$0.10833 per active minute (about $6.50/hour). This excludes cold-start and
idle scale-down overhead. More concurrency could lower cost per conversation,
but would be a different latency/throughput experiment.

## Interpretation and exclusions

- These are usage-shape estimates from one completed conversation, not invoices
  or statistical cost benchmarks.
- Google explicit-cache storage duration charges are excluded because the logs
  do not contain cache residence time.
- Provider volume discounts, batch rates, priority tiers, taxes, failed runs,
  retries outside the accepted run, and deployment idle time are excluded.
- Sonnet 5 uses its introductory price through 2026-08-31; reruns after that
  date require a new pricing file and regenerated output.
- The pricing inventory contains no OpenAI Pro model.

Official source URLs and per-model notes are stored beside every rate in the
pricing JSON. Generated results are `cost-results-2026-08-06.json` and
`cost-results-2026-08-06.md`.
