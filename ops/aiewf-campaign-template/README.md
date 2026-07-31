# Portable AIEWF campaign collector

This directory is a reusable, configuration-driven collector for full AIEWF
conversations. It is additive: it does not import, rewrite, or resume any frozen
campaign bundle. In particular, the completed Gemma 4 26B campaign under
`ops/baseten-gemma4-26b-a4b-vllm/` remains immutable campaign provenance.

The included JSON and TSV files are a **non-live Gemma 4 26B example**. The
endpoint is deliberately a placeholder and `serving.verified` is `false`, so an
accidental `--execute` is rejected before credentials, directories, or the
endpoint are touched.

## Configuration

Copy `configuration.example.json` and `schedule.example.tsv` for a new campaign.
All deployment- and filesystem-specific values come from JSON:

- endpoint URL and the environment variable that receives it;
- credential destination, source environment variable, and optional dotenv
  fallback source/key (the file is parsed; it is never sourced);
- requested model, accepted response model IDs, service, and pipeline;
- campaign artifact directory and benchmark run output root;
- common request environment, arm-specific environment, and variables/prefixes
  that must be removed;
- additional benchmark/service files whose bytes must be frozen in the source
  integrity manifest;
- serving gate, target count, timeout, attempt cap, and provenance markers.

Paths may be absolute or repository-root-relative. `run_one.py` calls the
existing pipeline internals directly so `paths.run_output_root` is honored; it
does not use the CLI helper that hard-codes `runs/<benchmark>/`.

Keep the final component of `run_output_root` equal to the benchmark name when
you want the existing `multi-turn-eval judge RUN_DIR` command to infer the
benchmark from the run directory's parent.

## Freeze and preflight

Before any live request:

1. Copy and edit the example JSON.
2. Create the complete fixed schedule. It must contain slots `1..N` exactly
   once and may interleave any arms defined in JSON.
3. Complete direct and Pipecat smokes against the exact serving configuration.
4. Record the disposition in the copied JSON and set `serving.verified=true`.
5. Run the default read-only preflight:

```bash
.venv/bin/python ops/aiewf-campaign-template/collect.py \
  --config path/to/configuration.json
```

Preflight validates configuration, schedule, existing manifests, canonical
transcripts, provenance, and any previously frozen source hashes. When campaign
artifacts do not exist, preflight does **not** create them. It also does not read
the credential source or open the endpoint.

The first `--execute` writes `source-sha256.txt` over the JSON, schedule, portable
runner, relevant benchmark/runtime sources, and recorder. Later invocations
refuse to continue if those sources change.

## Execute and resume

```bash
.venv/bin/python ops/aiewf-campaign-template/collect.py \
  --config path/to/configuration.json \
  --execute
```

Collection is strictly sequential: a slot is not started until the preceding
slot has a canonical outcome. A nonblocking filesystem lock rejects a second
collector. The lock descriptor is inherited by the one live child, so if the
parent collector dies, a still-running request continues to hold the lock and a
restart cannot overlap it.

Before launching a request, the collector atomically writes
`pending-attempt.json` with its exact run and console-log paths. On restart, that
attempt is classified and durably appended before any new request is launched.
This closes the usual crash window between a completed conversation and a
manifest append.

The durable artifacts are:

- `attempts.tsv`: every completed or recovered attempt;
- `canonical.tsv`: the contiguous prefix of eligible scheduled slots;
- `pending-attempt.json`: at most one launched but not yet finalized attempt;
- `campaign.log`: append-only lifecycle events;
- `logs/`: subprocess console logs;
- `source-sha256.txt`: frozen configuration and implementation identity;
- `.collection.lock`: concurrency guard.

Rerun the same command to resume. If a slot exhausts its durable attempt cap,
review its arm-blind infrastructure evidence and deliberately raise the cap:

```bash
.venv/bin/python ops/aiewf-campaign-template/collect.py \
  --config path/to/configuration.json \
  --execute \
  --max-attempts-per-slot 5
```

## Eligibility and replacement

The template implements one explicit fixed-denominator policy:
`first_valid_response`.

- A response is valid when a scheduled row contains non-empty assistant text or
  at least one tool call, from an accepted model identity and with valid frozen
  provenance.
- Once an attempt has a valid response, a model-caused early `end_session`, idle
  timeout, malformed later response, or other short conversation remains
  canonical as `fixed_denominator_short`. Its missing future turns are failures.
- An attempt with no valid response, a wrong model identity, an invalid
  transcript shape, or missing/wrong provenance is recorded as ineligible and
  replaced without changing the schedule.
- Recovery transcript rows do not count as scheduled turns. Scheduled rows must
  be unique and form a contiguous prefix from turn zero.

The collector never judges or publishes results. Judging, aggregate generation,
and README/report edits remain separate reviewed steps.
