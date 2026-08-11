# Claude session recovery and restart notes — 2026-07-20

Scope: Claude's main transcript and both forked-agent transcripts from 08:00 PT onward. The first relevant message was at 08:44 PT. State was rechecked after transcript recovery; no local benchmark process is still running.

## Read this first: external lifecycle and secrets

- **BaseTen:** model `w7p8mg0w` has five deployments. The newest, `w7mjyxy`, successfully became active and is now `SCALED_TO_ZERO` (`min_replica=0`, 15-minute scale-down). The older `3yvxlee` is still reported as `DEPLOYING` with zero active replicas. The production pointer still targets failed deployment `woz0v2k`. Inspect and stop the stale deployment before doing more work; do not cold-wake `w7mjyxy` yet.
- **Modal:** `nemotron-super-b200-budget`, `nemotron-super-b200-bisect`, and `nemotron-super-b200-nvfp4-budget` are deployed with zero tasks. Temporary app `nemotron-weights-http` is deployed and currently reports three tasks.
- **Credential cleanup:** the temporary checkpoint-bridge bearer token is plaintext in `~/src/modal-super/serve_weights_http.py`, `~/src/modal-super/baseten-super/data/start.sh`, the Claude transcript, and the session scratchpad. Treat it as compromised. Do not commit either file. Once BaseTen's stale download is stopped or accounted for, stop the bridge with:

  ```bash
  ~/.pyenv/versions/3.12.10/bin/modal app stop nemotron-weights-http
  ```

  Then rotate/remove the token and replace this bridge with durable private storage or a valid NGC credential. BaseTen's current `/tmp/model` design downloads 230 GiB on every cold start, so scale-to-zero discards the checkpoint and makes each wake slow and expensive.
- A BaseTen secret named `ngc_api_key` was created from `~/.ngc/config`, but that key returned 403 for the EA checkpoint. It is not a working weight source.

## Work completed

### Filler-token study and report

- The earlier Claude artifact at <https://claude.ai/code/artifact/a60a4405-1140-47ab-b584-f7718a4885b3>
  is superseded by the repository-owned report below, which received a second
  `gpt-5.6-sol` adversarial review and reproducibility reconciliation.
- Final report source and output have now been preserved in the repository:

  ```text
  scripts/build_filler_report.py
  docs/filler-token-latent-scratchpad-study.html
  ```

- The final report covers 18 models and uses conversation-cluster permutation tests after Sol identified pseudoreplication risk. Important added result: `gpt-5.5` went 96.3% to 100.0% with 96 dots (`n=8/8`, cluster-p=0.0232, median TTFAT 882→868 ms, zero aborts).
- It also adds the ceiling-dilution framing: errors eliminated were 62% for GPT-5.4 none, 100% for GPT-5.5, 100% for GPT-5.6-sol, and 89% for GPT-5.4 low.
- Manifests for GPT-5.5, the GPT-5.4 stack/ablation, and the effort-none mini investigation were copied into `docs/filler-study-data/`.
- `src/multi_turn_eval/pipelines/base.py` was updated in two places so GPT-5.5 routes through the Responses service and receives `reasoning.effort`. It was compile-checked and GPT-5.5 received a successful two-turn smoke.
- `docs/filler-token-latent-scratchpad-study.md` is now reconciled to the
  conversation-cluster results and hedged post-review claims. The primary test tool
  is `docs/filler-study-data/conversation_cluster_test.py`; the older turn-stratified
  tool is retained only for audit history.
- OpenAI `*-pro` models are excluded from conversational-latency work in the written
  policy, shared model-policy guard, CLI, and BasePipeline factory. The earlier
  10:25 `gpt-5.5-pro` attempt failed locally in `AsyncResponses.stream()` argument
  validation before any HTTP request and wrote a zero-line transcript. During recovery,
  a two-turn process was launched inadvertently and terminated immediately after the
  scope correction; its log remained empty and it created no run directory. No pro API
  request or response is evidenced by either attempt.

### GPT-5.4-mini investigation

- `docs/gpt-5.4-mini-abort-investigation-2026-07-20.md` is now rev 3. It retains the Sol-reviewed strict classifier—a real completion requires `end_session` at scripted turn 29; turn 28 is premature—and adds the effort-medium result and selection caveats.
- At `effort=none`: true completions were 0/17 no-filler, 1/17 dots, and 3/14 dashes. The defensible new finding is exit-site relocation: dots moved failures to turn 8 (8/17 vs 0/17 no-filler, exploratory Fisher p≈0.003), rather than reliably changing overall abort probability.
- The requested `effort=medium`, no-filler-vs-dash experiment completed after Claude stopped:

  | config | true completions | survivor accuracy | median TTFAT |
  |---|---:|---:|---:|
  | no filler | 4/20 | 86.7% (`n=4`) | 1117 ms |
  | 96 dashes | 8/10 | 93.3% (`n=8`) | 1152 ms |

  The fixed completion table has Fisher arithmetic p=0.0041, but it is not calibrated
  for the adaptive stopping/allocation rule. Conditional survivor uplift is +6.7
  points with selected-subset cluster-p=0.0054, but it has no unselected-population
  interpretation. Exit sites: no-filler t13×15 and t28×1; dashes t13×1 and t15×1.

### Modal Nemotron-Super work

- vLLM 0.25.1 was reverified as current. New scripts in `~/src/modal-super/`:
  - `serve_b200_budget_v3.py`: BF16 021826 + APC/prefix caching + MTP.
  - `serve_b200_bisect.py`: MTP only, no APC.
  - `serve_b200_nvfp4_budget.py`: NVFP4 030326 + APC + MTP; startup timeout raised to 35 minutes and FlashInfer autotuning disabled.
- v3 booted and passed a four-shape wedge probe plus a four-turn harness smoke. Identical-long-prompt APC timing was 868 ms cold, then 366/361 ms cached.
- Quality gate failed badly: v3 APC+MTP scored **81.3%** (`n=6`, median TTFAT 1098 ms) versus the previous v2 reference **98.3% at 1532 ms**.
- MTP-only recovered most quality: **95.0%** (`n=6`, median TTFAT 1322 ms). This strongly implicates experimental Mamba APC, but an APC-only cell and per-turn transcript comparison are still required. Do not promote v3/APC.
- The agent's provisional “TPS” probe counted stream chunks, not tokenizer-confirmed tokens; do not report those values as exact token/s.
- NVFP4 initially failed during long compilation/autotuning, but it finally booted after the no-autotune change and returned HTTP 200 from `/v1/models`. It then scaled down. No budget, harness, throughput, or accuracy test was run.

### BaseTen port

- Truss 0.18.22 was installed in the session scratchpad. Port source is `~/src/modal-super/baseten-super/`.
- B200 and B200:2 were rejected because that instance type is gated for this BaseTen account. Configuring H100:2 resulted in an H100:4 allocation; the server uses TP4.
- Deployment history:
  - `woz0v2k`: failed because `wget` was missing; still the production pointer.
  - `qjjnl4l`: failed because the local NGC key cannot access the EA checkpoint.
  - `wl5nx87`: superseded after the bridge environment variables were not exported.
  - `3yvxlee`: superseded but still reports `DEPLOYING`; investigate/stop.
  - `w7mjyxy`: booted successfully and later scaled to zero; never smoked.
- No BaseTen `/v1/models`, budget-enforcement, harness, latency, or judged comparison test was completed.

## Restart order

1. **Secure and preserve remaining external-track state.** Stop the stale BaseTen deployment/weight transfer, then stop the temporary Modal bridge and rotate its bearer token. Copy the remaining Modal/BaseTen experiment artifacts into an appropriate repo; never copy `weights_token.txt`:

   ```text
   superV3_manifest.tsv
   modal_track_status.md
   baseten_track_status.md
   ```

2. **Mini screen documented; fixed-size replication remains.** The medium table,
   strict classifier, survivor-selection caveat, manifest, and attempt log are durable.
   It appears only in the report's failure-mode discussion, not the master accuracy
   screen. Any inferential follow-up must use a fixed sample size or model the actual
   stopping rule.

3. **Report sources repaired.** The builder/output are repo-owned and the Markdown
   uses the cluster-robust/post-Sol numbers. Republish only if an external artifact
   update is desired; local source is authoritative.

4. **Keep GPT-5.5 coverage scoped to the base model.** The base `gpt-5.5`
   no-filler-vs-dots96 screen is complete. Never run OpenAI `*-pro` variants for
   conversational-latency evaluation, including smoke tests: their latency profile is
   outside scope. Add targeted routing coverage for base `gpt-5.5` only.

5. **Resolve the Modal quality regression before more optimization.** Run BF16 021826 APC-only at b128 (`n=6`) and compare per-turn failures across v2, MTP-only, and APC+MTP. Production recommendation should remain v2 or MTP-only until APC is cleared.

6. **Test NVFP4 without the known APC confound.** Use MTP-only/no-APC for the first quality gate. Run `/v1/models`, budget 32/128/512/unlimited, four-turn harness, a real tokenizer/usage-based throughput probe, then b128 `n=6`. Because NVFP4 is snapshot 030326 while current BF16 is 021826, deploy/test the volume's matching BF16 030326 checkpoint before attributing any difference to quantization.

7. **Redesign BaseTen weights before waking it.** Preferred paths: valid EA NGC credentials in BaseTen, or a durable private S3/Hugging Face/BaseTen weights store. Do not repeatedly boot H100:4 to download 230 GiB into `/tmp`. Once durable, deploy a clean candidate, make it the intended environment/production deployment, smoke auth and `/v1/models`, verify budgets, run the four-turn harness, and only then run the planned `n=4` Modal-vs-BaseTen comparison. The harness reads `VLLM_API_KEY` in `base.py`; verify BaseTen's required auth scheme on the smoke.

## Useful status checks

```bash
# Modal app/task state
~/.pyenv/versions/3.12.10/bin/modal app list --json

# BaseTen deployments (load BASETEN_API_KEY without printing it)
curl -s https://api.baseten.co/v1/models/w7p8mg0w/deployments \
  -H "Authorization: Bearer $BASETEN_API_KEY"

# Local work left by the agents
git -C ~/src/modal-super status --short
git -C ~/src/aiewf-eval status --short
```

No commits were made. Both repositories were already dirty; preserve unrelated user changes and commit only intentionally selected files.
