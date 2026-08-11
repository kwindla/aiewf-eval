# BaseTen Gemma 4 N=120 extension audit

Final audit: **PASS**.

| Check | Result |
|---|---:|
| Frozen scheduled slots | 120 |
| Collection attempts | 120 |
| Canonical conversations | 120 |
| Full 30-turn conversations | 120 |
| Campaign-log exits | 120 with `rc=0 timeout=false` |
| Valid judgments | 120/120, all attempt 1 |
| Judged scripted turns | 3,600 |
| Collection wall time | 5,330 seconds (1:28:50) |
| Mean / median conversation time | 44.34 / 41.0 seconds |
| Final deployment state | `SCALED_TO_ZERO`, 0 active replicas, `min_replica=0` |

The inherited collector serializes a successful subprocess exit code of zero as
`unknown` in `attempts.tsv` because its fallback expression treats integer zero
as falsy. The unchanged original N=30 campaign has the same representation.
`campaign.log` durably records all 120 exits as `rc=0 timeout=false`, and every
attempt passed transcript, model-identity, provenance, and full-turn validation.
The frozen ledger was not rewritten to conceal this known representation issue.

No collection log contains the forbidden `MTE_FILLER_DOTS active` marker. The
collector's final read-only preflight revalidated the 120-row canonical prefix
and all frozen source hashes. The unchanged original N=30 collector and judge
also passed their read-only 30/30 preflights before pooling.

Judging completed with `claude-opus-4-5` and
`claude-agent-sdk-v4-turn-taking`; the frozen judge input digest is
`d06f421073219770b4a1c4ee7dcfe90e3b813a20fa19c3a03bbb1e63c9b1a9e4`.
