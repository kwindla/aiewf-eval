# vLLM conversation-cache: server-side hang repro

> **STATUS 2026-05-14: Fixed upstream in commit `61911a6` "Fix mamba-align
> scheduler hang on bad-block-residual prefill"** (+ follow-up `4b0ceab`
> "change conversation cache guard"). Both are vLLM-side patch changes;
> the Python service is unchanged. Servers running a vLLM build older
> than `61911a6` still reproduce the hang. This repro doc is left intact
> as the reference for the bug and as the test the fix needs to pass.

Captured 2026-05-13 against two Nemotron Omni vLLM servers (NVFP4 + BF16)
running with the client-authoritative conversation-cache patches from
`nemotron-nano-omni@khk/cache-refactor-5090` (server-side commits `fb423d7`
"step 6: vLLM serving + engine" and `11a8fe9` "step 6 followup", repackaged
in `6100582`).

## Symptom (one line)

When a `conversation_id`-bearing request includes `conversation_require_cache=true`,
the server occasionally **accepts the POST, never returns any bytes, and
never emits `[DONE]` or an error**. The client cancels at its 45 s idle
timeout. No 4xx/5xx response, no SSE frames, no `ConversationCacheMissError`.

This matches the failure class explicitly called out in
`nemotron-nano-omni/docs/conversation-cache-control-flow.md`:

> "This is why `publish-before-[DONE]` matters for streaming responses: a
> client that immediately sends the tool-result followup after the final
> SSE chunk should still find the conversation lease ready."

We believe what we're seeing is exactly that race (or a near sibling) but
in production traffic, not just on tool-result followups.

## Impact

Measured frequencies in 10×30-turn benchmark runs per server (cache-on,
the new client-authoritative contract):

| Server | Build | Cache off completion | Cache on completion | Cache on idle-timeouts |
|---|---|---|---|---|
| `192.168.7.228:8010` | BF16 | 10/10 | 5/10 | **5/10** |
| `127.0.0.1:8000` | NVFP4 | 7/10 | 6/10 | **3/10** |

Cache-on never returns any client-side error before the 45 s idle
timeout. Cache-off (which the patches don't change) runs cleanly on both
servers — so the model itself isn't the issue.

### Update 2026-05-14: bug still present after CUDA graph capture

The BF16 server was rebuilt with CUDA graph capture enabled and brought up
on `http://192.168.7.228:8000/v1`. A single cache-on 30-turn benchmark
reproduced Mode A on the very first run: the model emitted two
consecutive tool-call-only responses (`submit_dietary_request` then
`request_tech_support`, both with `output_text='\n'`) and the next user
audio request hung indefinitely. Same fingerprint as the prior repros —
6-message suffix `assistant[tool_calls],tool[result],user[audio=1],
assistant[tool_calls],tool[result],user[audio=1]`, `require_cache=true`,
no response / no error / no [DONE]. Fresh conversation_id
`mte-b9b036a0da314374a80ee4ef0d1b8120` for filtering server logs.

Artifacts: `traces/mode-a-bf16-cuda-graphs-2026-05-14/`. The CUDA graph
capture also produced a notable side-effect on cache-on cold turn 0
(18,382 ms vs 3,029 ms for cache-off on the same server) — almost
certainly first-request graph compilation. After that one-time cost,
warm cache-on TTFB on this server is ~687 ms vs cache-off ~4,070 ms
(~6× faster — the cache attach is working). But the hang at turn 16 kills
the run before the speedup pays off.

A second cache-on run ~13 minutes later confirmed:
- CUDA graph warmup is amortized: pair-2 cache-on cold turn 0 = 3144 ms
  (matches cache-off cold ~3184 ms). The 18 s was a one-time
  per-server-start cost.
- Mode A hang reproduced again at the same point in the benchmark, same
  exact request shape. Deterministic in this conversation, fresh
  conversation_id (`mte-...`-prefixed, different from pair-1).
- Warm-turn TTFB on cache-on settled at ~566 ms (vs cache-off ~4043 ms,
  ~7.2× faster), so once the server is warm the cache attach is even
  faster than pair-1's measurement.

### Root cause + fix (upstream commit `61911a6`)

The fix landed in upstream `khk/cache-refactor-5090@61911a6` on 2026-05-13:

> Fix mamba-align scheduler hang on bad-block-residual prefill
>
> When the conversation-checkpoint cap pulled `num_new_tokens` below
> `block_size` while `num_computed_tokens` was still short of the last
> mamba block boundary, `_mamba_block_aligned_split` rounded the chunk
> to zero (`num_new_tokens // block_size * block_size == 0`). The
> schedule loop broke out of the waiting branch every tick without
> scheduling the request; the attached lease stayed parked indefinitely
> while the engine reported `Running:0/Waiting:1`. Clients hit their 45s
> idle timeout. **Repro rate ~1 in 30 audio-in turns on NVFP4; ~1 in 6
> on BF16.**

That commit-message-quoted repro rate (1/30 NVFP4, 1/6 BF16) matches our
empirical 10×30 batches exactly (3/10 NVFP4 = 1/100 per turn; 5/10 BF16
= 1/60 per turn — within batch-size sampling). The follow-up `4b0ceab`
"change conversation cache guard" adjusts the lease guard around the
new code path.

Both fixes are entirely server-side (vLLM patch + new diagnostic
scripts). The Python service file `nemotron_voice/.../nemotron_omni.py`
is byte-identical between upstream `6100582` (which this repro doc was
filed against) and HEAD, so the client/vendored side needs no changes.
Servers running a vLLM build older than `61911a6` still hang; redeploy
with the new patch and the repros in `traces/` should all complete.

## Two distinct fingerprints

### Mode A: hang after two tool-call-only assistant responses

Reproduced on both servers (3/10 NVFP4, 3/10 BF16). The conversation
gets into a state where the model emits two consecutive assistant
responses whose `output_text` is just `'\n'` and whose only payload is
`tool_calls`. The third user turn arrives and packs *both* tool exchanges
into the suffix:

```
role_summary = assistant[tool_calls], tool[result],
               user[audio=1],
               assistant[tool_calls], tool[result],
               user[audio=1]
msgs sent: 6
conversation_require_cache: true
conversation_committed_message_count: ~56
```

Server accepts, hangs, never responds.

In this benchmark the trigger is the `vote_for_session` exchange around
scripted turns 22-25 (the model has to guess `session_id` from a speaker
name; recovery turns force a second tool call).

**Artifacts**: `traces/mode-a-nvfp4-on08/` and `traces/mode-a-bf16-net-on08/`.
Each contains the prior 3 successful request/response pairs and the
single hung request. `run.log.tail.txt` shows the client-side frame
sequence with timestamps.

### Mode B: hang after two vanilla turns (no tool calls)

Reproduced only on the BF16 server (2/10). Just two normal back-and-forth
turns with audio user input. Each request was a 2-message suffix:
`role_summary = assistant, user[audio=1]`. The third request — *same
shape that worked twice* — hangs.

```
role_summary = assistant, user[audio=1]     # turn 1 (full context)  -> OK
role_summary = assistant, user[audio=1]     # turn 2 (suffix)        -> OK
role_summary = assistant, user[audio=1]     # turn 3 (suffix)        -> HANG
msgs sent: 2
conversation_require_cache: true
```

No tool calls anywhere in the conversation. Pure suffix-only audio Q&A.
This is the strongest signal that the issue isn't tool-call-specific.

**Artifacts**: `traces/mode-b-bf16-net-on07/`. Contains turns 1 and 2
(both successful) and turn 3 (the hung request). `run.log.txt` is the
full DEBUG log from the entire ~60-second run.

## Client behavior (so we can rule it out)

Client: the audio-in pipeline in `aiewf-eval`, using the upstream
`NemotronOmniAudioLLMService` and `NemotronAssistantAggregator` vendored
verbatim at `aiewf-eval/src/multi_turn_eval/vendor/nemotron_omni.py`
(byte-identical to `nemotron-nano-omni/.../nemotron_omni.py` at upstream
commit `6100582`). Pipecat 1.1.0 from PyPI.

For each request the client:

1. Builds `_canonical_messages` from the Pipecat context via the new
   `NemotronAssistantAggregator` — which writes the *exact* assistant
   message (text + tool_calls combined into one message) into the
   context, so the local view matches what vLLM commits.
2. Computes `committed_count = len(_committable_messages_after_success(full_messages))`
   and sends it in the payload as
   `conversation_committed_message_count`.
3. Sends a suffix from index `committed_count` onward, with
   `conversation_require_cache: true`, against the same `conversation_id`
   used for the prior request.
4. Begins reading the SSE stream and **never sees a single byte** for
   the hung requests.

The client-side request shape per turn is in the
`*.client-request.json` traces. No malformed JSON, no
`NOT_GIVEN` sentinels, no unexpected fields. `_cache_shape_fingerprint`
(under `request_body._cache_shape_fingerprint`) is stable across the
session — we are not triggering cache rotation.

Time between the server's final `[DONE]` byte for response N and the
client POSTing request N+1 is on the order of tens of ms (Pipecat frame
propagation + audio-message construction). This is the "immediately
sends the tool-result followup" timing the upstream doc calls out, but
it also happens on plain audio Q&A turns where there's no tool-result
followup at all (Mode B).

## What's NOT the cause

- Not a context-length limit. Max `prompt_tokens` per run stays in the
  15-18 K range; `max_model_len` is 32 K.
- Not an `audio_url` parsing issue. The exact same audio data URL shape
  was processed successfully in many earlier turns of the same
  conversation.
- Not a malformed suffix. The Mode B captures show the *same* 2-message
  suffix shape that worked twice in a row.
- Not a `ConversationCacheMissError` — those return 409 in milliseconds
  and the client retries cleanly.

## What we'd love to see server-side

1. **vLLM frontend log lines around the hang**. Specifically: did the
   server log "lease acquired" for the hung request? Did it log
   "rendered prompt" / "prompt-checkpoint candidate"? Or does the
   request just disappear after the POST?

2. **The frontend ledger state for the affected conversation_id at the
   moment of the hang**. Was the previous response's checkpoint
   actually published, or is the lease still tied to the prior in-flight
   request that already streamed [DONE] to the client?

3. **Engine-side queue / scheduler state**. If the engine isn't even
   aware of the request, it's a frontend deadlock. If the engine is
   sitting on the request but never producing tokens, it's a scheduling
   or attach-skipped issue without a visible log.

The three conversation IDs in the captures (`mte-67d9220c...`,
`mte-d6f755bd...`, `mte-d7ca0fb5...`) are stable identifiers we can use
to filter server logs if you have them.

## Reproducing from scratch

The client side is straightforward:

```bash
# In aiewf-eval (commit 339417c or later — vendored NemotronOmniAudioLLMService)
NEMOTRON_OMNI_TRACE_DIR=$(mktemp -d) \
MTE_NEMOTRON_AUDIO_IN_BASE_URL=http://127.0.0.1:8000/v1 \
MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=1 \
uv run multi-turn-eval run aiwf_medium_context \
  --model nemotron_3_nano_omni \
  --service nemotron-audio-in \
  --pipeline audio-in \
  --verbose
```

To target Mode B specifically (the cleaner repro), run the full 30-turn
benchmark a few times back-to-back. It hangs at the third request in
roughly 1 in 5 runs on BF16.

To target Mode A, run with `--only-turns 22,23,24,25` and hope the model
guesses a wrong `session_id` on the first attempt (about 60-70% of runs
do, which triggers the recovery turn and the back-to-back tool-call
pattern).

All trace files referenced in this doc were produced by the upstream
`NemotronOmniAudioLLMService._write_trace_file`. Audio data URLs are
redacted to `<data-audio-base64 sha256=… chars=…>` — set
`NEMOTRON_OMNI_TRACE_REDACT_AUDIO=0` to capture the raw audio bytes if
you need byte-exact request replay.

## Pipeline order (client topology, for reference)

```
LLMUserAggregator
  → NemotronOmniAudioLLMService          # vendored upstream service
    → ToolCallRecorder                   # benchmark-only; records to transcript
      → NemotronAssistantAggregator      # vendored upstream aggregator
        → NextTurn                       # benchmark-only; advances turn loop
```

`NemotronAssistantAggregator` writes the model's exact emitted assistant
message (one row, `content + tool_calls` combined) into the shared
`LLMContext`. `ToolCallRecorder` does not alter the context; it only
mirrors `FunctionCall*Frame` frames to a per-turn ledger for scoring.

## Open questions

- Is there a per-conversation lease/lock at the frontend that can deadlock
  if the engine commit and the client's next POST race? If yes, the fix
  is probably to acquire+release atomically against a single shared
  lease record per conversation, with explicit "publish complete"
  ordering before [DONE].

- Does Mode B (no tool calls anywhere) imply the race window is just
  "SSE [DONE] is sent before frontend ledger publish completes," and
  the next POST simply blocks waiting for a publish that never happens
  for some reason? Or is there a different mechanism in play?

- Why is the BF16 server more susceptible than NVFP4 (5/10 vs 3/10)?
  The patches should be identical across builds. Maybe scheduler
  throughput differences expose the race more on BF16.

Happy to capture more traces, set
`NEMOTRON_OMNI_TRACE_REDACT_AUDIO=0` to get byte-exact request bodies,
or instrument the client further if it helps. Ping us with what you
need.
