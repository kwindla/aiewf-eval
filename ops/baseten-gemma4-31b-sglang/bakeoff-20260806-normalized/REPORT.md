# Gemma 4 31B serving bakeoff

The BaseTen SGLang v0.5.16 NEXTN/MTP deployment was selected for the N=30
AIEWF medium-context campaign. All nine bakeoff conversations completed all 30
scripted turns with valid runtimes and zero thinking tokens.

| Serving stack | Conversations | Scripted turns | TTFAT P50 | TTFAT P95 | TTFAT max |
|---|---:|---:|---:|---:|---:|
| SGLang NEXTN/MTP | 3/3 | 90 | 422ms | 537ms | 3181ms |
| vLLM APC, no MTP | 3/3 | 90 | 434ms | 648ms | 2374ms |
| SGLang APC, no MTP | 3/3 | 90 | 496ms | 728ms | 4559ms |

The two SGLang arms used the same SGLang image, target checkpoint, two H100s,
BF16, tensor parallelism, concurrency, context limit, prefix caching, and
sampling. Their prompt cache-read shares were effectively identical: 98.42%
for NEXTN/MTP and 98.48% for no-MTP. MTP therefore wins on the stable latency
statistics without a cache imbalance. Its three conversations took 107.4
seconds total (35.8 seconds mean), versus 148.1 seconds (49.4 seconds mean) for
the SGLang control. The two newly executed vLLM controls averaged 38.8 seconds;
the first vLLM conversation was a preexisting validation run and did not have a
comparable runner wall-clock measurement.

## Tool-stream compatibility finding

The first unnormalized SGLang pair stalled at scripted turn 11 even though the
server returned HTTP 200 and generated the correct `submit_session_suggestion`
call. Direct raw-SSE probes showed that SGLang's Gemma 4 parser emitted index
`2`, the tool's position in the request schema. The conforming vLLM control
emitted index `0`, the first call's position in the response. Pipecat coalesces
streamed function fragments by response-local index, so the malformed SGLang
stream produced no executable call.

Inspection of SGLang v0.5.16 and current `main` at
`18e6c61c21ad39725522c008190d2b540dd6228d` confirmed the parser source uses
the request-schema index in both streaming and non-streaming Gemma 4 paths.
The benchmark now has an explicit, default-off compatibility option,
`MTE_VLLM_NORMALIZE_TOOL_CALL_INDICES=1`, which maps first-seen raw indices to
zero-based response ordinals. Targeted offline tests cover a nonzero first
index, continued argument fragments, multiple calls, and default-off behavior.
With the option enabled, all six SGLang bakeoff conversations completed.

Raw evidence from the failed protocol is retained in the sibling
`bakeoff-20260806/` directory. This directory contains the clean post-fix
attempt ledger and logs used for selection.
