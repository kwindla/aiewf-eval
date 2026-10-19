# October 2026 model evaluations: publication and evidence storage

Git contains the implementation, regression tests, reviewed campaign methods and
configurations, final reports, and compact summaries. Each `summary.json` wraps
the published aggregate metrics under `data`, with the original artifact path
and SHA-256. Per-conversation inputs, grades, transcripts, SDK events, recordings,
copied runtime trees, retries, and session state stay local or in an archive.
Preserving evidence does not require committing it.

The current quality and latency tables use the
[recovery-context summary](judge-recovery-context-20261019/summary.json), plus the
[Nova 2.5 Sonic summary](nova-2-5-sonic-20261019/user-context-v2/summary.json).
Reports from earlier stages retain their historical scores and limitations.

## Reports

- [Opus 5.5 cleanup review and verification](REVIEW.md)
- [Original Opus 5.5 refresh](judge-refresh-opus55-20261017/REPORT.md)
- [Recovery-context validation and rejudging](judge-recovery-context-20261019/REPORT.md)
- [New text and voice models](recent-models-20261018/REPORT.md)
- [Responses adapter repair and reruns](responses-repair-20261018/REPORT.md)
- [Nemotron Ultra retest](baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018/REPORT.md)
- [Grok Voice high reasoning](grok-voice-high-20261018/REPORT.md)
- [Nova 2.5 Sonic](nova-2-5-sonic-20261019/user-context-v2/REPORT.md)
- [Nova content-filter investigation](nova-2-5-sonic-20261019/filter-diagnostics/REPORT.md)

## Preserved local evidence

The [archive manifest](manifest.json) identifies a verified 36,791,580-byte archive
of the pre-cleanup branch files. The archive ID is `aiewf-eval-archive/2026-10-benchmark-refresh`; the maintainer's
local copy is under `~/src/aiewf-eval-archive/2026-10-benchmark-refresh/`.
It contains 9,212 files and a member-by-member `MANIFEST.json` with SHA-256 hashes.
Every member was read back and verified after creation. The committed manifest
records the archive checksum and source identities for the publication copies.
A verified `branch-history-d9a3e20.bundle` beside the archive preserves all original
branch commits relative to the main commit identified in the manifest. Its checksum
is recorded there too. Branch commit IDs mentioned by the reports and manifest
refer to that local bundle or backup ref after the PR branch is rewritten; they
are historical identities, not guaranteed public GitHub links. There is no off-host
copy of either the evidence archive or history bundle. Original raw `runs/` outputs and recordings are not
included in this archive; they remain in the maintainer's ignored run tree.

Links labeled **local archive** refer to evidence in that archive, not to files
available in a fresh checkout. The archive's file paths are relative to the
original repository root. Extract it into a separate directory to inspect the
original campaign files without replacing the current implementation. Do not
extract the whole archive over a current checkout. To restore audits, use:

```bash
uv run python -m scripts.generate_campaign_publication --archive /path/to/evidence.tar.gz --restore-evidence
```

This verifies the archive and each restored member, restores only excluded `ops/`
paths, and refuses to replace tracked files or differing existing evidence. It
does not restore raw `runs/` outputs; those must be available separately. Inspect
the backup branch in a separate worktree: switching this working tree to the old
backup branch and back can delete restored ignored evidence.

## Tests and reproduction

`uv run pytest -q` runs the normal tests, including checks that the README and
standalone leaderboard match the committed summaries. It needs no campaign
archive, provider credentials, or raw recordings.

`uv run pytest -q --local-evidence` additionally runs the full historical
evidence audits. These require the excluded campaign artifacts restored at their
original `ops/` paths and the original referenced `runs/` data. Missing evidence
is an error when this option is explicitly selected. The campaign-specific
`.gitignore` files keep restored working evidence out of future commits while
allowing the reviewed methods and configurations.

The preserved campaign scripts reproduce the original campaign formats and
expect that local evidence. Public report copies have relocated links; their
metric values are unchanged and checked against the verified archive by the local
evidence tests. To regenerate publication copies and their checksums, run
`uv run python -m scripts.generate_campaign_publication --archive /path/to/evidence.tar.gz`.
The campaign `publish.py` entry points reproduce historical formats; use this
publication generator afterward to restore the current public links. This cleanup did not generate conversations or
request new grades.
