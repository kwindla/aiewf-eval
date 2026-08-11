# BaseTen Kimi K2.6 AIEWF medium-context campaign

This bundle collects exactly 30 complete AIEWF medium-context conversations
against BaseTen's shared Model API model `moonshotai/Kimi-K2.6`. The frozen arm
uses `reasoning_effort=none`, no filler, temperature 0.6, and `max_tokens=8192`.

The current-route freshness probe at
`runs/aiwf_medium_context/20260806T091743_moonshotai_Kimi-K2.6_fba32308`
exactly matches that signature. It has all 30 scheduled turns, 30 valid model
responses, a successful recovery `end_session`, a valid completed runtime, and
complete token accounting, so it is canonical slot `K26-01` rather than being
discarded and rerun.

Unlike the reusable first-valid-response campaign template, this campaign has
a deliberately stricter denominator: only a runtime-valid conversation with
all 30 scheduled turns, 30 valid model responses, and complete usage rows is
canonical. A missing `end_session` remains a judgeable model error and does not
make an otherwise full conversation ineligible. Zero-response, short, invalid,
timed-out, and otherwise incomplete attempts are recorded in `attempts.tsv` and
replaced. This implements the requested 30 *complete* conversations without
survivor-biasing the tool-call outcome; judging is a later, separate step.

The collector runs one BaseTen conversation at a time. The first live batch
briefly tested provider concurrency 2, but both attempts received BaseTen 429s
late in their conversations. No later slot was launched at that concurrency;
the two attempts are retained in `attempts.tsv`, and collection continues
serially. The collector writes a pending record before each child starts,
freezes source hashes, never prints the API key, and is resumable. BaseTen Model
API is serverless, so there is no custom deployment to tear down.

The shared endpoint also showed a rolling provider-rate limit after several
back-to-back full conversations: a serial request received a 429 after three
consecutive ~35-second complete conversations. The collector therefore enforces
an arm-blind 30-second cooldown before every full-conversation attempt. This
changes only campaign pacing; the model request signature is unchanged.

Read-only preflight and bundle tests:

```bash
.venv/bin/python \
  ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/collect.py

.venv/bin/pytest -q \
  ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/test_bundle.py
```

Start or resume live collection:

```bash
.venv/bin/python \
  ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/collect.py \
  --execute
```

Credentials come from `BASETEN_API_KEY`, with only that key read from
`../gb-benchmarks/.env` as a fallback.
