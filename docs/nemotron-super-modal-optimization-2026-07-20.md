# Nemotron-Super Modal optimization follow-up — 2026-07-20

Status: today's bounded measurement is complete. The BF16 030326
snapshot-control cell remains unmeasured because its first cold start crossed
the deployed web-server startup limit and restarted; restart instructions are
below. All latency is endpoint-observed from the client and therefore includes
Modal/network overhead.

## Experimental controls

- Model service: vLLM 0.25.1, 2×B200 tensor parallelism, FP8 KV cache,
  `mamba-ssm-cache-dtype=float16`, chunked prefill, native
  `thinking_token_budget`, Super-v3 reasoning parser, Qwen3-Coder tool parser,
  and the same Nano chat template.
- Quality cell: `aiwf_medium_context`, budget 128, six attempts per candidate,
  judged inline. A completion is valid only when the transcript contains an
  actual `end_session` call on scripted turn 29. No replacement retries were
  made. Twelve new full conversations ran; six of the 18-conversation ceiling
  remain for the stopped BF16 control.
- Accuracy is the fraction of 30 turns where all four judge dimensions pass.
  TTFAT is the harness `ttfb_ms`, which excludes streamed reasoning in this
  service configuration.
- Throughput is not SSE chunk frequency. Completion-token counts come from the
  endpoint's OpenAI-compatible `usage`; decode time runs from the first
  non-empty streamed delta through the final usage event.
- Prefill is reported as endpoint-observed TTFT with one generated token. It
  includes network and scheduling overhead. Repeated identical prompts measure
  fixed-order repeat/warmup behavior, not a hardware-only prefill rate or an
  identified APC effect.

## Apps and endpoints

| Cell | App | Endpoint | Snapshot | APC | MTP |
|---|---|---|---|---:|---:|
| v2 reference | `nemotron-super-b200-budget` | `https://daily--nemotron-super-b200-budget-serve.modal.run` | BF16 021826 | no | no |
| APC+MTP v3 | same production-named app (historical deployment) | same | BF16 021826 | yes | yes |
| MTP-only bisect | `nemotron-super-b200-bisect` | `https://daily--nemotron-super-b200-bisect-serve.modal.run` | BF16 021826 | no | yes |
| APC-only bisect | `nemotron-super-bf16-021826-apc-only` | `https://daily--nemotron-super-bf16-021826-apc-only-serve.modal.run` | BF16 021826 | yes | no |
| Quantized candidate | `nemotron-super-nvfp4-030326-mtp-only` | `https://daily--nemotron-super-nvfp4-030326-mtp-only-serve.modal.run` | NVFP4 030326 | no | yes |
| Snapshot control | `nemotron-super-bf16-030326-mtp-only` | `https://daily--nemotron-super-bf16-030326-mtp-only-serve.modal.run` | BF16 030326 | no | yes |

## 021826 lever bisect

| Configuration | Judged accuracy | Strict completion | Required scripted tool calls | Median of run-median TTFAT |
|---|---:|---:|---:|---:|
| v2: no APC, no MTP | 98.3% (177/180) | 6/6 | 34/36 | 1521 ms |
| MTP only | 95.0% (171/180) | 6/6 | 32/36 | 1317 ms |
| APC only | 96.7% (174/180) | 6/6 | 34/36 | 1326 ms |
| APC+MTP | 81.3% judge-available (122/150; 5 judged) | 5/6 | 10/36 | 1094 ms across all 6 attempts |

The previously reported 81.3% was not actually a six-judge aggregate: six
conversations ran, but one judge response failed JSON parsing twice. The 81.3%
is 122/150 across the five judge-available transcripts. Those five include the
incomplete attempt and omit a strictly completed attempt whose judge failed,
so this is not survivor-only accuracy. All six transcripts were available for
deterministic tool-call and strict-completion analysis.

The regression is a tool-execution collapse specific to the combined APC+MTP
cell. Required actions occur at turns 11, 12, 15, 17, 24, and 29. APC+MTP
emitted only 10/36 expected tool names, often claiming success without a call
or printing fake `<function>` markup as text. In comparison, v2 emitted 34/36,
MTP-only 32/36, and APC-only 34/36. An exact conversation-cluster permutation
test on each run's count of the six expected tools gives APC+MTP versus
MTP-only p=0.00649 and APC+MTP versus v2 p=0.00216; MTP-only versus v2 gives
p=0.879 under the same exhaustive two-sided mean-difference test. The
descriptive factorial interaction is -3.67 tools per conversation: APC had a
0.00 observed effect without MTP and a -3.67 effect with MTP. Thus APC alone
did not reproduce the collapse in this sample, and the result is consistent
with a configuration interaction. It does not prove an APC×MTP causal effect:
the cells were sequential, non-randomized batches from separate deployments.
These unadjusted p-values are screening summaries, and their label-
exchangeability assumption is not guaranteed by that run order.

The run-level TTFAT medians (ms) were v2
`1522/1520/1461/1775/2984/1492`, MTP-only
`1380/1315/1319/1295/1677/1310`, APC-only
`1344/1339/1315/1319/1333/1319`, and APC+MTP
`1155/1098/1138/1089/1090/1075`. Summarizing those six conversation-level
values avoids overweighting attempts that added recovery interactions.

### APC-only performance probes

- Usage-grounded decode: 178.9 completion tok/s at budget 32 and 178.9 at
  budget 128.
- Budget smoke reasoning character counts: 32→175, 128→616, 512→2283,
  unlimited→4706. All returned successfully; the 1200-token smoke limit caused
  `finish_reason=length`, so these are enforcement/survivability checks rather
  than answer-quality samples.
- Repeated 12,638-token prefix TTFT: 1106, 597, 570 ms.
- Repeated 25,238-token prefix TTFT: 928, 674, 775 ms. This prompt extends the
  same repeated text used by the preceding 12,638-token probe, so its first
  observation was already partially cached and is not an independent cold
  baseline. The 12,638-token APC delta is also not an identified APC effect:
  the no-APC NVFP4 cell showed a 373-ms first-to-warm-median delta under the
  same order, and the remaining cross-cell difference is confounded by
  snapshot and quantization. A clean APC estimate needs matched checkpoints
  and randomized/interleaved order or independent cold instances.
- Four-turn budget-64 harness smoke completed all four turns.

## 030326 snapshot/quantization comparison

The NVFP4 cell completed, but the BF16 snapshot control did not become ready
inside its initial 25-minute web-server startup limit. No BF16 probe, smoke, or
full conversation was run, so this is an unpaired NVFP4 result rather than a
quantization estimate.

| Configuration | Judged survivor accuracy | Strict completion | Required scripted tool calls | Median of run-median TTFAT |
|---|---:|---:|---:|---:|
| NVFP4 030326, MTP only | 92.7% (139/150; 5 judged) | 5/6 | 30/36 | 1296 ms across all 6 attempts |
| BF16 030326, MTP only | not run | 0 attempts | not run | not run |

The NVFP4 per-survivor scores were 28, 26, 29, 27, and 29 out of 30.
Attempt 4 stopped at turn 11 after an idle timeout and therefore was neither
replaced nor judged. Attempt 5 needed an extra recovery interaction after turn
15. Those two operational failures matter independently of the 92.7%
survivor-only accuracy.

The six attempt-level median TTFAT values were
`1307/1359/1285/1279/1261/1471` ms. Their median is 1296 ms; among the five
strict survivors it is 1307 ms. At n=6, the one abort is the entire 5/6 versus
6/6 difference, so neither completion fraction is a precise reliability or
equivalence estimate.

### NVFP4 030326 performance probes

- `/v1/models` returned the intended NVFP4 030326 checkpoint and its native
  262,144-token context length.
- Usage-grounded decode: 318.2 completion tok/s at budget 32 and 304.7 at
  budget 128, both with exactly 768 completion tokens. These are absolute
  endpoint observations, not an NVFP4 uplift: the BF16 030326 control never
  became ready.
- Budget smoke reasoning character counts: 32→175, 128→639, 512→2299,
  unlimited→2655. As in the APC-only smoke, all four hit the deliberately
  short 1200-token generation cap, so this validates endpoint/budget survival
  but not answer quality.
- Repeated no-APC TTFT was 435/427/433 ms at 164 prompt tokens,
  1254/882/880 ms at 12,638 prompt tokens, and 1406/1372/1382 ms at
  25,238 prompt tokens. The large warm/repeat effect at 12,638 tokens despite
  APC being disabled shows that repeated-prefix deltas are not by themselves
  a clean measurement of APC; generic engine warmup and other caches also
  contribute.
- The four-turn budget-64 harness smoke completed all four turns.

Against the BF16 021826 MTP-only survivor scores, the NVFP4 survivor mean is
0.7 turns lower per conversation; an exact two-sided permutation test at the
conversation level gives p=0.461. Against APC-only BF16 021826 it is 1.2 turns
lower, p=0.141. These tests are underpowered and omit the NVFP4 incomplete
attempt, so they do not establish equivalence. Completion must remain a
separate endpoint: NVFP4 completed 5/6 versus 6/6 for both historical BF16
cells.

## Deployment lifecycle and safety

- APC-only app ID `ap-3fmuDcDavtJ5x7uXPtwdqa` was stopped after measurement
  and verified at zero tasks.
- The first NVFP4 deployment, `ap-pNd4zFwdmTW9nlIqm38Z71`, briefly scaled to
  two containers because repeated cold-start health polls accumulated while
  no explicit container cap was set. It was stopped immediately. Every new
  app definition was then patched with `max_containers=1` before further work.
- Capped NVFP4 app ID `ap-yTNCvJdCgvCEuBTf3Hb6Xk` was stopped after
  measurement and verified at zero tasks.
- The first BF16 control deployment, `ap-zH5qE7FDrAcrzLl4h184zK`, was stopped
  at zero tasks before measurement when a definition diff found that it had
  not inherited NVFP4's disabled FlashInfer-autotune setting. Replacement app
  `ap-MfszO5hHDeuPpDyNbKK8Hd` includes that setting. The two local definitions
  then differed only in checkpoint, app/cache names, and comments, but the
  path names do not establish identical pre-quantization weights, tokenizer,
  or generation metadata; immutable checkpoint revisions or hashes still need
  verification before any quantization attribution.
- Replacement BF16 app `ap-MfszO5hHDeuPpDyNbKK8Hd` reached weight-load,
  compile, Mamba warmup, KV profiling, and final CUDA graph capture. The
  important timestamps were: vLLM start 16:26:18 PT, weights loaded 16:30:06,
  first compile complete 16:36:55, 611.79-second Mamba profiling/warmup
  complete 16:47:07, and graph capture starting 16:50:04. At 16:51:23 the
  25-minute web-server startup guard restarted the container while capture was
  still running. The replacement worker began loading the 230.25-GiB
  checkpoint again, so the app was stopped rather than allowed another costly
  cycle. It drained to zero tasks. All new apps were then verified stopped at
  zero tasks. The raw startup log and final Modal app-list JSON are archived
  with the result artifacts.

## How to restart the stopped BF16 control

Six of the original 18-conversation allowance remain. The untracked local
definition `~/src/modal-super/serve_b200_super_bf16_030326_mtp_only.py` has now
been corrected to `timeout=40*60` and `startup_timeout=35*60`, matching the
NVFP4 deployment envelope while retaining `max_containers=1`. It also matches
the NVFP4 inference flags, including MTP one token, APC off, and FlashInfer
autotune disabled. APC-off is now explicit as `--no-enable-prefix-caching` in
both future MTP-only definitions. Preserve the exact local revision and verify
checkpoint/tokenizer/config hashes before deploying the matched control.

Restart in this order. The commands deliberately use absolute paths and a trap
so cleanup runs on success, failure, or interruption:

```bash
set -euo pipefail
MODAL=/home/khkramer/.pyenv/versions/3.12.10/bin/modal
PY=/home/khkramer/.pyenv/versions/3.12.10/bin/python
APP_NAME=nemotron-super-bf16-030326-mtp-only
ENDPOINT=https://daily--nemotron-super-bf16-030326-mtp-only-serve.modal.run
RESULT_DIR=/home/khkramer/src/aiewf-eval/docs/nemotron-super-modal-optimization-data/bf16-030326-mtp-only
mkdir -p "$RESULT_DIR"
test ! -s "$RESULT_DIR/manifest.tsv"

cd /home/khkramer/src/modal-super
cp serve_b200_super_bf16_030326_mtp_only.py "$RESULT_DIR/deployed-serve.py"
sha256sum serve_b200_super_bf16_030326_mtp_only.py \
  chat_template_nano.jinja super_v3_reasoning_parser.py \
  >"$RESULT_DIR/deployed-sources.sha256"
"$MODAL" deploy serve_b200_super_bf16_030326_mtp_only.py
APP_ID=$("$MODAL" app list --json | jq -r \
  '[.[] | select(.Description=="nemotron-super-bf16-030326-mtp-only") |
    select(.State=="deployed")][0]."App ID"')
test -n "$APP_ID" && test "$APP_ID" != null

CLEANED=0
cleanup() {
  [ "$CLEANED" = 1 ] && return 0
  CLEANED=1
  "$MODAL" app stop "$APP_ID" || true
  for _ in $(seq 1 30); do
    state_tasks=$("$MODAL" app list --json | jq -r --arg id "$APP_ID" \
      '.[] | select(."App ID"==$id) | [.State,.Tasks] | @tsv') \
      || state_tasks=missing
    [ "$state_tasks" = $'stopped\t0' ] && return 0
    sleep 2
  done
  echo "Modal app did not drain: $APP_ID $state_tasks" >&2
  return 1
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

curl --fail-with-body -sSL --max-time 2700 \
  "$ENDPOINT/v1/models" | tee "$RESULT_DIR/models.json"
test "$("$MODAL" app list --json | jq -r --arg id "$APP_ID" \
  '.[] | select(."App ID"==$id) | .Tasks')" = 1

"$PY" /home/khkramer/src/modal-super/probe_modal_super_endpoint.py \
  "$ENDPOINT" --output "$RESULT_DIR/probes.json"

cd /home/khkramer/src/aiewf-eval
env VLLM_BASE_URL="$ENDPOINT/v1" MTE_VLLM_THINKING=1 \
  MTE_VLLM_NATIVE_BUDGET=1 MTE_VLLM_THINKING_BUDGET=64 \
  MTE_VLLM_MAX_TOKENS=8192 uv run multi-turn-eval run aiwf_medium_context \
  --model nemotron-3-super-120b --service vllm-openai \
  --only-turns 0,1,2,3 >"$RESULT_DIR/four-turn-smoke.log" 2>&1

/home/khkramer/src/modal-super/run_modal_quality_cell.sh \
  bf16_030326_mtp_only "$ENDPOINT" "$RESULT_DIR"
```

Do not replace incomplete attempts. The revised runner records runner return
code, strict completion, judge status, and aggregate outcome in separate
manifest columns. After the trap verifies zero tasks, compare whole-
conversation distributions, not pooled turns, and call this a matched control
rather than a paired experiment because the conversations have no paired seed.

The compile volume may contain reusable artifacts from today's near-complete
startup, but the restart should still budget for a full cold path; cache reuse
must not be assumed.

## Recommendation

Do not promote APC+MTP: its combined cell has a large, conversation-clustered
tool-execution regression even though either lever alone survived this small
cell. Do not promote APC-only yet either: vLLM explicitly labels Mamba cache
mode `all` with APC experimental, the cell is only n=6, and the no-APC NVFP4
probe showed that generic warmup/cache effects can mimic part of an APC repeat
gain.

The conservative production choice remains BF16 021826 v2 (no APC, no MTP),
which has the best observed quality and 6/6 completion. MTP-only BF16 021826
is at most a candidate for a larger staged soak: it reduced the median of
conversation-median TTFAT from 1521 to 1317 ms, but showed an observed
3.3-point accuracy decrement (171/180 versus 177/180) with wide uncertainty at
n=6. The one-turn-per-conversation mean score difference has exact
conversation-level permutation p=0.130, which is inconclusive rather than
evidence of equivalence. Promotion should require prespecified completion and
required-tool gates, randomized short run blocks, and independent deployment/
batch replication.

NVFP4 030326 produced an absolute endpoint observation of about 305--318
completion tok/s, but 5/6 completion and 92.7% survivor accuracy are
insufficient for promotion. Finish the BF16 030326 matched control above before
attributing either quality or performance differences to NVFP4, then expand
completion testing if the comparison still looks promising.

## Artifacts

- Raw new-cell manifests, logs, and probe JSON:
  `docs/nemotron-super-modal-optimization-data/`
- Independent gpt-5.6-sol xhigh audit:
  `docs/nemotron-super-modal-optimization-sol-review-2026-07-20.md`.
- App definitions and reusable probes: `~/src/modal-super/serve_b200_super_*_only.py`,
  `~/src/modal-super/probe_modal_super_endpoint.py`, and
  `~/src/modal-super/run_modal_quality_cell.sh`.
- Historical v2 manifest:
  `docs/filler-study-data/super_modal_v2_manifest.tsv`.
- Historical APC+MTP and MTP-only manifest was recovered from the Claude
  session scratchpad as `superV3_manifest.tsv`; its run directories remain in
  `runs/aiwf_medium_context/`.
