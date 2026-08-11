# Post-collection judging workflow

The judging driver is:

`/tmp/judge_qwen36_aiewf_medium_n30.py`

It is intentionally read-only unless `--execute` is present. It refuses to
judge until `campaign.log` records all of the following:

- `RUN_COLLECTION_DONE total=60 high=30 none=30`
- `DEPLOYMENT_SCALED_TO_ZERO replicas=0`
- `CAMPAIGN_DONE`

It verifies all 60 rows of `canonical.tsv` against `frozen-order.tsv`, requires
30 canonical runs per arm, validates every transcript and its recorded counts,
and takes the run directory only from canonical column `run_dir`. Short
model-caused conversations remain canonical and are judged on their observed
scripted turns; the later aggregate must count unobserved scheduled turns as
failures in the fixed 900-turn denominator per arm.

The original frozen order, campaign configuration, audited judge source, and
CLI source are also pinned by SHA-256. Any change stops the driver for explicit
review rather than silently changing provenance.

The repository CLI's `--judge-model` option is currently not wired through to
the implementation. The driver therefore does not pass that misleading option.
Instead, it parses and pins the actual implementation constants:

- judge model: `claude-opus-4-5`
- judge version: `claude-agent-sdk-v4-turn-taking`

It fails closed if either changes, if a non-Claude or `pro` judge appears, or if
the CLI plumbing changes. No OpenAI model, pro or otherwise, is invoked.

After collection and BaseTen teardown have completed, first run the read-only
preflight:

```bash
cd /home/khkramer/src/aiewf-eval
.venv/bin/python /tmp/judge_qwen36_aiewf_medium_n30.py
```

Then start resumable judging:

```bash
cd /home/khkramer/src/aiewf-eval
.venv/bin/python /tmp/judge_qwen36_aiewf_medium_n30.py \
  --execute \
  --workers 2 \
  --max-attempts 3
```

Two workers are the conservative default for Claude usage/rate limits. The
driver caps concurrency at four, records every attempt, uses exponential retry
delays, checks that judging did not mutate transcripts, and skips structurally
valid existing results on restart.

State is written under `judging/`:

- `canonical-inputs.tsv`: frozen exact inputs and transcript SHA-256 values
- `judge-source-sha256.txt`: pinned CLI and judge source hashes
- `judge-attempts.tsv`: durable retry ledger
- `logs/`: one combined stdout/stderr log per completed attempt
- `invalid-output-snapshots/`: copies of pre-existing invalid outputs
- `COMPLETE.json`: written only after all 60 judgments validate

If a Claude usage window is exhausted, let the process finish its bounded
retries or stop it normally, then rerun the same command after the window
resets. If three attempts for a run were consumed only by a transient service
limit, resume with a deliberately higher total cap such as
`--max-attempts 5`; the original attempt ledger and logs remain intact.
