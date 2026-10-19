# Nova 2.5 Sonic content-filter diagnostics

User authorized prompt/version/length experiments after the standard prompt was
rejected twice, including through native Bedrock. These are excluded diagnostic
requests, not scored benchmark conversations. Preserve the canonical prompt,
tasks, audio, prior evidence, and publication unchanged.

Use the same us-east-1 model ID, Tiffany voice, HIGH endpointing, Pipecat 1.12.0
native session helpers, sampling defaults, interactive `Hello!`, and continuous
silent input as the successful control. At most two simultaneous sessions, each
bounded by 45 seconds. A response counts as accepted only on actual output audio;
filter errors, other operational errors, and timeouts are distinct outcomes.

First compare short, policy-only, knowledge-base halves, a shorter knowledge-base
prefix, full prompt with and without tools, and benign padding at matching length.
Adapt subsequent tests to those results: exact excerpts/bisection, length-matched
controls, repeats, and equivalent full-content message layouts as useful. Freeze
each phase's exact prompt bytes, settings, request list, and diagnostic driver
hash before invoking it. Retain every response/error and request ID. Never infer
which safety category fired from the generic error alone, and never count a
successful shortened prompt as the standard benchmark.

[Evidence storage and local audit instructions](../../README.md).
