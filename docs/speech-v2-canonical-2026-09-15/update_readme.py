"""Publish the completed speech-only aggregate, preserving text-model results."""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def main():
    revision = HERE / 'rejudge-policy-compliance-1'
    assert (revision / 'complete.json').exists()
    aggregate = json.loads((revision / 'aggregate.json').read_text())
    assert len(aggregate['models']) == 5
    assert all(m['attempts'] == 25 for m in aggregate['models'])
    preserved = json.loads((HERE / 'preserved-text-inputs.json').read_text())
    readme = ROOT / 'README.md'
    before = readme.read_text()
    prefix, rest = before.split('Speech-to-speech models:', 1)
    assert hashlib.sha256(prefix.encode()).hexdigest() == preserved['README_text_section']
    for name, expected in preserved.items():
        if name != 'README_text_section':
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    _, tail = rest.split('### Sample Recordings', 1)
    section = '''Speech-to-speech models:

**Canonical speech-v2 results, September 15, 2026.** Each model received 25 attempted
30-turn conversations with the same voice-specific prompt and Opus 5 rubric.
The prompt requires reuse of known information, tool execution after authorization,
and contextual explanations for event requests the assistant cannot fulfill.
The shared speech judge was clarified to require policy compliance as well as
progress on the request, with factual grounding assessed independently against the
full knowledge base. Every saved conversation was rejudged with the same revised
rubric; no model prompts or recordings changed. See the
[before/after audit](docs/speech-v2-canonical-2026-09-15/rejudge-policy-compliance-1/AUDIT.md).
Text-model prompts and scores above are unchanged.

'''
    section += (revision / 'table.md').read_text()
    audit = json.loads((revision / 'audit.json').read_text())
    conflicts = audit['explicit_reason_score_conflicts']
    if conflicts:
        section += f'\nJudge limitation: the audit flags {len(conflicts)} explicit score/explanation mismatches across the 131 rejudged conversations. Scores are the unmodified judge outputs; see the audit for affected runs and remaining ambiguity.\n'
    section += '''
Pass Rate requires correct tool use, instruction following, and knowledge grounding
on the same turn. Every row uses 750 planned turns; missing turns after an incomplete
conversation count as failures. **Complete** reports conversations reaching all 30
turns. Answer quality on recorded turns and full failure details are available in the
[campaign report](docs/speech-v2-canonical-2026-09-15/rejudge-policy-compliance-1/REPORT.md).

'''
    for model in aggregate['models']:
        if model['complete'] < 25:
            reasons = Counter(r['failure_reason'] for r in model['reliability'] if r['failure_reason'])
            summary = ', '.join(f'{n} `{reason}`' for reason, n in sorted(reasons.items()))
            section += f"* **{model['arm']['model']}:** {25-model['complete']}/25 incomplete conversations ({summary}).\n\n"
    api_counts = {m['arm']['model']: sum(bool(r['evidence']['api_error']) for r in m['reliability']) for m in aggregate['models']}
    quota_counts = {m['arm']['model']: sum(bool(r['evidence']['rate_limit']) for r in m['reliability']) for m in aggregate['models']}
    if not any(api_counts.values()) and not any(quota_counts.values()):
        section += 'No explicit API or rate-limit errors were logged in this corrected cohort. No-audio timeouts are reported separately from explicit API errors.\n\n'
    else:
        for name in api_counts:
            if api_counts[name] or quota_counts[name]:
                section += f'API reliability: **{name}** had API-error evidence in {api_counts[name]}/25 attempts and rate-limit evidence in {quota_counts[name]}/25 attempts.\n\n'
    section += '''Gemini 3.1 reliability history: the [August investigation](docs/async-gemini-live-comparison-2026-08-04/root-cause-probes/README.md)
reported 18/20 no-response-free conversations and a WebSocket 1006 reconnect in an
additional attempt. These historical observations are separate from the current counts.

Voice-to-voice latency is measured from the end of user speech to the start of actual
model speech in the stereo recording. Latency is conditional on measurable responses;
observation counts and percentiles are in the report. For voice-agent use, our target
is under 1,500ms. These numbers include turn-detection delay and initial silence, so
they differ from the service's TTFB. Gemini 3.8 uses explicit audio activity boundaries;
the other models use server VAD. "Turn Ok" reports the separate turn-taking assessment.

The [frozen protocol](docs/speech-v2-canonical-2026-09-15/PROTOCOL.md) documents settings,
the excluded harness-debugging batch, and reproducibility artifacts. The
[previous speech table and Gemini thinking sweep](docs/speech-v2-canonical-2026-09-15/historical-speech-results.md)
used the older prompt/rubric and are archived rather than mixed into this comparison.
The original `gpt-realtime` was omitted from this refresh.

### Sample Recordings'''
    tail = tail.replace('Listen to full 30-turn benchmark conversations from each speech-to-speech model:', 'Historical recordings using the original prompt (not the speech-v2 cohort above):', 1)
    readme.write_text(prefix + section + tail)
    assert hashlib.sha256(readme.read_text().split('Speech-to-speech models:', 1)[0].encode()).hexdigest() == preserved['README_text_section']
    print('Updated README speech results; original text section verified unchanged.')


if __name__ == '__main__':
    main()
