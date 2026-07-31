# Filler-study data: manifests + analysis tools

Config→run-dir manifests (TSV: config-label \t run-dir) for every cell of the
filler-token study (docs/filler-token-latent-scratchpad-study.md) and the
Nemotron-Super budget sweep (docs/nemotron-super-filler-thinking-plan.md),
rescued from the session scratchpad so results stay reproducible.

- `analyze_filler.py <manifest> <cfg1,cfg2,...>` — pass%/TTFAT/thinking table
- `conversation_cluster_test.py <manifest> <cfgA> <cfgB> [--turns]` — primary
  conversation-cluster permutation test (10k permutations) with an optional
  descriptive per-turn table
- `paired_config_test.py <manifest> <cfgA> <cfgB> [--turns]` — legacy
  turn-stratified analysis retained for audit history; do not use its p-values as
  primary because turns within a conversation are not independent

The report builder is `scripts/build_filler_report.py`; it writes the durable,
self-contained `docs/filler-token-latent-scratchpad-study.html`. The mini-medium
strict-completion follow-up is preserved in `mini_medium_manifest.tsv` and
`mini_medium_attempts.txt`; `mini_medium_all_attempts.tsv` maps every success and
premature exit to its run directory. No credentials or endpoint bearer tokens are
archived here.

`gpt54mini_manifest.tsv` and `gpt54mini_attempts.txt` preserve the earlier
effort-none screen's original loose classifier and are legacy provenance only: their
`SUCCESS` labels include turn-28 exits. The authoritative strict counts were rebuilt
from all transcripts with completion defined as `end_session` exactly at scripted
turn 29 and are documented in
`docs/gpt-5.4-mini-abort-investigation-2026-07-20.md`.
