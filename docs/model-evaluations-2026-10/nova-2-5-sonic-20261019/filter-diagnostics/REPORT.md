# Nova 2.5 Sonic: content-filter investigation

**The evidence points to false positives on conference-reference text in SYSTEM
content, rather than a general prompt-length limit.** Two isolated triggers are
the word **Replicate** and the session title **Voice Cloning and Synthesis: Ethics
and Technology**. The same complete knowledge base is accepted as USER context,
and the model can answer questions about both passages and facts near its end.

These are observed behavioral differences. AWS returns a generic filter error,
without a category or matched span. Interpreting the triggers as an overbroad
voice-replication restriction is an inference, not a confirmed internal cause.

We ran **50 excluded diagnostic requests**: 31 generated audio, 18 received a
content-filter rejection, and one received a distinct protocol error. Each
phase's request bytes and driver hashes were frozen before its calls. All used
`amazon.nova-2-5-sonic`, us-east-1, Tiffany, HIGH endpointing, unchanged sampling
defaults, and continuous silent input. At most two sessions ran concurrently.

| Diagnostic | Characters before interactive input | Outcome |
|---|---:|---|
| Short assistant control | 63 | Audio generated |
| Standard policy alone | 4,310 | Audio generated |
| Policy plus first 8,000 KB characters | 12,310 | Audio generated |
| Full standard SYSTEM prompt | 41,161 | Blocked, both repetitions |
| Full SYSTEM prompt without tools | 41,161 | Blocked |
| Benign padding matching full prompt length | 41,161 | Audio generated |
| First / second KB halves plus short policy | 18,489 / 18,490 | Both blocked |
| Short policy plus exact voice-cloning session title | 114 | Blocked, both repetitions |
| Short policy plus `Replicate` | 73 | Blocked |
| Short policy plus complete Replicate sponsor line | 123 | Blocked |
| Sponsor description with brand name removed | 109 | Audio generated |
| Title changed to `Voice Synthesis: Ethics and Technology` | 102 | Audio generated |
| Full SYSTEM prompt minus voice-cloning session entry and Replicate line | 40,763 | Audio generated, both repetitions |
| Full SYSTEM prompt with KB wrapped in reference XML tags | 41,205 | Blocked |
| Unchanged policy in SYSTEM; entire unchanged KB in USER history | 41,161 | Audio generated in both greeting repetitions and all three readback checks |

KB bisection localized the first failure to the voice-cloning session entry and
the second to the sponsor listing, rather than the intervening community meetups.
The defensive-security session, individual meetup days, finance/healthcare meetup
titles, and “Indie Hackers Building with AI” passed their isolated tests. Removing
only the voice-cloning entry and meetups did not resolve the full prompt; removing
the voice-cloning entry and Replicate line did. Shorter prefixes alone would have
confounded length with omission of those passages.

The USER-context layout kept the exact 4,310-character policy and exact
36,851-character KB, in order, with the same tools. The greeting test reported
12,072 input text tokens, consistent with the full reference being processed.
To check more than prompt acceptance, three separate native sessions used the
full reference and collected the complete spoken response:

| Question | Spoken response |
|---|---|
| What is the exact title of session 920702? | “The exact title of session nine two zero seven zero two is Voice Cloning and Synthesis: Ethics and Technology.” |
| Which sponsor runs models in the cloud with one line of code? | “Replicate is the Expo Hall sponsor that lets you run models in the cloud with one line of code.” |
| What is the venue street address? | “The conference venue is located at seven hundred eighty Mission Street, San Francisco, California nine four one zero three.” |

Sending the policy and reference as multiple SYSTEM blocks is not a valid
alternative: AWS returned `Duplicate SYSTEM content. SYSTEM content can only be
provided once per prompt.` This is the one operational error, not a filter hit.

The most promising next integration test keeps behavioral rules in SYSTEM and
loads the full unmodified KB as a noninteractive USER history message before the
scripted conversation. This preserves every reference fact and avoids deleting
the triggering entries. It still changes message placement and must be recorded
as an adapter configuration change. These short probes do not establish 30-turn
tool behavior, context continuity, waveform validity, or benchmark quality; an
excluded full-task smoke is required before any canonical collection.

No benchmark prompt source, KB, audio, existing leaderboard, or previous evidence
was changed. There are still zero canonical Nova conversations and zero judge
requests. The original preflight report remains an accurate historical record;
this is an additive diagnosis of its blocker.

[Outcome counts and result hashes](summary.json) · [Verification (local archive)](../../README.md) ·
[Frozen requests (local archive)](../../README.md) · [Raw results (local archive)](../../README.md) · [Protocol](PROTOCOL.md)

AWS's [Nova 2.5 Sonic service card](https://docs.aws.amazon.com/ai/responsible-ai/nova-2-5-sonic/overview.html)
describes input/output filtering and says that its safety filters cannot be
configured or disabled. It does not document these specific lexical triggers.

[Evidence storage and local audit instructions](../../README.md).
