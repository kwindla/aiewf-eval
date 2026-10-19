"""Publish the frozen refresh only after every case has an explicit disposition."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LINK = 'ops/' + HERE.name
DIMS = ('tool_use_correct', 'instruction_following', 'kb_grounding')


def ms(value):
    return '—' if value is None else f'{value:.0f}ms'


def pct(value):
    return '—' if value is None else f'{value:.1f}%'


def rank(m, latency):
    return (float('inf') if m['strict_percent'] is None else -round(m['strict_percent'], 1), round(latency), m['label'])


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |',
                      '|' + '|'.join('---' if i == 0 else '---:' for i in range(len(header))) + '|']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def text_table(data):
    rows = []
    # Sort by displayed precision so the README and its chart agree on ties.
    models = sorted(data['text_models'], key=lambda m: rank(m, m['latency']['positive']['p50_ms']))
    for m in models:
        lat = m['latency']['positive']
        rows.append([m['label'], pct(m['strict_percent']), pct(None if m['strict_percent'] is None else 100-m['strict_percent'])]
                    + [pct(m['dimension_error_percent'][dim]) for dim in DIMS]
                    + [ms(lat[k]) for k in ('p50_ms', 'p95_ms', 'max_ms')]
                    + [m['provider'], m['complete']])
    return table(['Model', 'Pass Rate', 'Any Error', 'Tool Error', 'Instruction Error', 'KB Error',
                  'TTFAT P50', 'TTFAT P95', 'TTFAT Max', 'Provider', 'N'], rows)


def text_notes():
    return f'''Every new grade uses **Opus 5.5 at medium effort**, with the same frozen text rubric. **— means full-cohort quality is unavailable because the provider rejected at least one selected input**; those rows remain visible with recalculated latency. Rejected judge requests are neither model failures nor passing grades, and their conversations are not dropped to manufacture a score. **N is complete conversations**, with 30 scripted turns each. We selected the most recent matching complete conversations, up to 30 per configuration. Fourteen rows have fewer than 30; the [cohort audit](docs/text-cohort-sizes-2026-10-17.md) explains the shortfalls. Selection does not require correct tools or a passing grade.

Accuracy is **conditional on completion**. Incomplete conversations and their observed prefixes are excluded, so these results differ from earlier attempt-based, fixed-900-turn studies. They measure quality among completed conversations and do not establish completion reliability. Any Error is the complement of Pass Rate; component errors overlap. The [refresh report]({LINK}/REPORT.md) includes exact counts, conversation-cluster confidence intervals, selected source dates, and configuration mappings.

Judge inputs exclude delimited `<think>` reasoning, including analysis before a closing `</think>` whose opening tag was supplied by the chat template. Final answers and actual tool calls are evaluated; raw captures are preserved. The [cleanup audit]({LINK}/reasoning-cleanup/manifest.json) records every changed input. Historical latency measurements are unchanged; leaked reasoning in content may have started their timer before the final answer.

TTFAT is the recorded service time to the first user-visible answer token or tool-call output. Each conversation's first scripted response is excluded; later scripted and recovery responses are included. P95 uses linear interpolation. All latency cells were recalculated from the selected captures. Local GPU measurements use localhost and omit provider network time; capture dates, SDK versions, hardware, caching, and sampling settings differ across rows. OpenAI cohorts retain recorded service-tier/SDK variation; older unlogged tiers are not inferred as defaults. The ~700ms line is a voice-use guideline, not a statistical threshold or a measured end-to-end speech latency.

The four Nemotron 3 rows now use explicitly mapped, content-aware captures: Ultra 128/96 use native budgets on **Modal**; Super/Nano 512 use **NVFP4 on AWS B200**. These replace the older Baseten-labeled legacy summaries. Changes combine cohort/serving remapping and rejudging; they cannot be attributed solely to the judge. See the [source inventory]({LINK}/SOURCE-AUDIT.md), [configuration inventory]({LINK}/configurations.json), and per-row manifests.

Reasoning, filler, precision, and serving configurations are kept separate. Gemini minimal/low labels retain their recorded thinking settings; Gemini 2.5 explicitly uses budget zero. Kimi on/off retain their different vendor-mode samplers. Muse uses its recorded GGUF low-strength configuration; Lightning and local Qwen use NVFP4. The manifests preserve available request settings and endpoint origins; missing settings are not inferred as defaults.

The [previous publication]({LINK}/historical/README.md) preserves the earlier scores, denominators, serving details, and study links (its relative links are rooted at this repository). The [October latency audit](docs/ttfat-reanalysis-2026-10/REPORT.md) and original campaign reports remain historical evidence. The requested Groq, Grok 4.6 text, Fable, Sonnet 4.6, Haiku 4.5, and Sonnet 5 removals remain in the [row archive](docs/text-rows-archived-2026-10-17.md) and [Groq archive](docs/groq-gpt-oss-120b-archived-2026-10-17.md).'''


def voice_section(data):
    models = sorted([m for m in data['voice_models'] if not m['additional_experiment']],
                    key=lambda m: rank(m, m['non_tool']['median_ms']))
    rows = []
    for m in models:
        d = m['accuracy_denominator']
        rows.append([m['label'], m['complete'], f"{m['failed_conversations']}/{m['eligible_attempts']} ({m['failure_rate_percent']:.1f}%)",
                     pct(m['strict_percent'])] + [(f"{m['scores'][dim]}/{d}" if m["quality_complete"] else "—") for dim in DIMS]
                    + [f"{m['scores']['turn_taking']}/{d}" if m['turn_taking_scored'] else '—']
                    + [ms(m['non_tool']['median_ms']), ms(m['non_tool']['max_ms']), ms(m['tool']['mean_ms']), ms(m['silence']['mean_ms'])])
    main = table(['Model', 'Selected', 'Failed conversations', 'Pass Rate', 'Tool Use', 'Instruction', 'KB Ground', 'Turn Ok',
                  'Non-Tool V2V Med', 'Non-Tool V2V Max', 'Tool V2V Mean', 'Silence Pad Mean'], rows)
    additional = []
    for m in data['voice_models']:
        if m['additional_experiment']:
            additional.append([m['label'], m['complete'], f"{m['failed_conversations']}/{m['eligible_attempts']} ({m['failure_rate_percent']:.1f}%)",
                               pct(m['strict_percent']), ms(m['first_speech_positive']['mean_ms']),
                               ms(m['first_speech_positive']['median_ms']), f"{m['first_speech_positive']['n']}/{m['accuracy_denominator']}"])
    extra = table(['Model / configuration', 'Selected', 'Failed conversations', 'Pass Rate', 'First-speech V2V mean', 'Median', 'Positive latency coverage'], additional)
    return f'''Speech-to-speech models:

**New quality grades use Opus 5.5 at medium effort**, using the shared frozen speech rubric. Each row has **25 complete, harness-valid conversations (750 turns)** from the latest validated capture profile. Standard rows use the speech-v2 prompt; the prompt-optimized GPT-Live row is marked with an asterisk. Quality is scored on recorded spoken responses, with verified audio turn-taking evidence carried forward separately.

Accuracy is conditional on completion. The original attempt ledger retains **245 attempts: 225 scored, six model/service failures, and 14 harness exclusions**. Failure rate divides incomplete model/service conversations by harness-valid attempts; top-ups do not erase failures. Original captures, prompts, alignment decisions, and September reports are preserved. See the [refresh report]({LINK}/REPORT.md) and [original sampling and capture protocol](docs/speech-results-2026-09/PROTOCOL.md).

{main}

*GPT-Live prompt-optimized uses a conversational frontend and a detailed knowledge-base backend, Terra medium, and a five-second quiet/playback gate. It uses the same questions and shared speech judge. Turn-taking is unscored (—) for this reconstructed-transcript profile; waveform latency is measured separately.*

### Additional speech experiments

The interaction-aware Gemini medium profile uses the standard prompt and provider-IDLE/closing-playback gates. The earlier-harness medium profile remains historical.

{extra}

**Failed-conversation details:** turn numbers are one-based, out of 30. Gemini regular has one no-audio timeout on turn 1; Gemini 3.1 has one on turn 25 after 24 completed turns; Gemini medium has two on turn 1; GPT-Live optimized has two 90-second no-response timeouts on turn 11 after ten completed turns. These are not labeled explicit API errors.

**Harness and measurement notes:**

- Thirteen Gemini regular captures had duplicate-tool acknowledgments suppressed by a harness defect. Twenty-one unaffected earlier completions were retained, with four validated completions added after the fix. Its failure denominator is 26 eligible attempts; the 13 harness exclusions are separate.
- One Gemini low startup-greeting capture is excluded as a harness defect and replaced. The older broken-gate GPT-Live cohort and earlier medium harness are diagnostics.
- GPT-Live standard retains the verified waveform landmark alignment. One brief GPT-Live optimized answer is recorded but unmeasurable by the canonical VAD; its latency remains missing.
- Recalculated V2V uses actual first recorded speech, including acknowledgments, and all 30 scripted turns. It measures first speech rather than the substantive answer. Positive measurements, signed/nonpositive diagnostics, missing boundaries, and tool/non-tool groups are preserved. Missing values never become zero; silence padding uses the original Silero measurements.
- Earlier Gemini captures retain their documented silent-closing audio limitations; the interaction-aware medium profile waits for provider IDLE and closing playback. Original capture limitations remain in the [September report](docs/speech-results-2026-09/REPORT.md).
- Turn-taking remains unscored for optimized GPT-Live and interaction-aware Gemini medium. It is neither a pass nor an error and is excluded from strict three-dimension accuracy for every row. Full metric coverage and confidence intervals are in the refresh report.

'''


def report(data):
    text = sorted(data['text_models'], key=lambda m: m['label'])
    voice = sorted(data['voice_models'], key=lambda m: m['label'])
    coverage_status = (
        'Full-cohort quality is available for every selected configuration.'
        if data['all_quality_complete'] else
        'Full-cohort quality is unavailable for affected rows; their sources and recalculated latency remain published.'
    )
    quality_rows = []
    for m in text + voice:
        ci = m['strict_ci95']
        interval = '—' if ci is None else f'{ci[0]:.1f}–{ci[1]:.1f}%'
        quality_rows.append([m['label'], m['complete'], m['graded_conversations'], m['provider_blocked_conversations'], f"{m['scores']['strict']}/{m['graded_turns']} observed",
                             pct(m['strict_percent']), interval, f"${m['successful_judge_cost_usd']:.2f}"])
    text_cost = sum(m['successful_judge_cost_usd'] for m in text)
    voice_cost = sum(m['successful_judge_cost_usd'] for m in voice)
    superseded_cost = data.get('cost_accounting', {}).get('superseded_grade_cost_usd', 0)
    text_cost += data.get('cost_accounting', {}).get('superseded_text_cost_usd', 0)
    cost = text_cost + voice_cost
    missing = sum(m['unknown_cost_jobs'] for m in text + voice)
    ledger = [json.loads(line) for line in (HERE/'attempts.jsonl').read_text().splitlines() if line.strip()]
    errors = [e for e in ledger if e.get('event') == 'error']
    retry_notes = []
    for path in sorted((HERE/'retries').glob('*/retry.json')):
        retry = json.loads(path.read_text())
        outcomes = retry.get('outcomes', [])
        graded = sum(item['status'] == 'graded' for item in outcomes)
        rejected = sum(item['status'] == 'provider_safeguard_rejection' for item in outcomes)
        retry_notes.append(
            f"User-requested retry finished {retry.get('finished_at', 'pending')}: "
            f"{len(retry['jobs'])} previously rejected inputs retried, {graded} graded and "
            f"{rejected} rejected again. Sources, rubric, model, effort, and existing per-input "
            f"clarifications were unchanged. Prior rejection records and completion/verification "
            f"snapshots are preserved beside the [retry record]({path.relative_to(HERE)})."
        )
    retry_note = '\n\n'.join(retry_notes)
    cleanup_path = HERE/'reasoning-cleanup/manifest.json'
    if cleanup_path.exists():
        cleanup = json.loads(cleanup_path.read_text())
        outcomes = cleanup.get('outcomes', [])
        graded = sum(item['status'] == 'graded' for item in outcomes)
        rejected = sum(item['status'] == 'provider_safeguard_rejection' for item in outcomes)
        retry_note += (
            f"\n\nUser-requested reasoning cleanup finished {cleanup.get('finished_at', 'pending')}: "
            f"audited {cleanup['conversations_audited']} inputs across {cleanup['cohorts_audited']} configurations. "
            f"Removed delimited analysis from {cleanup['changed_scripted_turns']} scripted turns in "
            f"{cleanup['changed_conversations']} conversations; {graded} replacement/new grades and {rejected} "
            f"provider rejections. All other grades were reused. The same cleanup rule applies before judging "
            f"regardless of whether an earlier request succeeded. Empty final text with a tool call remains "
            f"a tool-only response. Raw sources, selected membership, rubric, model, effort, actual tool calls, "
            f"and latency observations are unchanged. The obsolete per-input clarification was omitted. "
            f"Prior grades and rejected dispositions are archived under [reasoning-cleanup/](reasoning-cleanup/); "
            f"the [manifest](reasoning-cleanup/manifest.json) and result metadata record per-turn source/answer "
            f"hashes and removed character counts. Cost includes ${superseded_cost:.2f} for superseded grades. "
            f"Score differences on rejudged conversations can include judge variability as well as input cleanup."
        )
    coverage = []
    for m in voice:
        coverage.append([m['label'], m['first_speech_positive']['n'], m['nonpositive_n'], m['missing_latency_n'],
                         m['tool']['n'], m['non_tool']['n'], m['silence']['n'],
                         m['scores']['turn_taking'] if m['turn_taking_scored'] else 'unscored'])
    dates = []
    for m in text + voice:
        c=json.loads((HERE/m['manifest']).read_text())
        selected_dates = sorted(i.get('captured_at') or i['capture_id'][:15] for i in c['inputs'])
        dates.append([m['label'], m.get('provider') or 'speech profile', m['complete'], selected_dates[0], selected_dates[-1], f"[manifest]({m['manifest']})"])
    return f'''# README rejudging and remapping

Processing finished {data['as_of']}. Scope: **43 text configurations (1,091 conversations)** and **nine voice configurations (225 conversations)**, totaling **1,316 selected conversations / 39,480 scripted turns**. **{data['counts']['graded_conversations']} conversations received valid grades; {data['counts']['provider_blocked_conversations']} remain unscored because of provider safeguard rejections.** {coverage_status} No new model conversations were generated.

Every quality request used `claude-opus-5-5`, medium effort, adaptive thinking, no tools, and no project settings. The judge received neither generating-model identities nor prior scores. Text finished before voice judging started. The exact rubric hashes and selection/latency policies are in [aggregate.json](aggregate.json) and [PROTOCOL.md](PROTOCOL.md). Per-conversation outputs are in [results/](results/); all attempts are recorded in [attempts.jsonl](attempts.jsonl).

SDK-reported **successful-request cost: ${cost:.2f}** (${text_cost:.2f} text / ${voice_cost:.2f} voice); {missing} current successful requests lack cost metadata. The ledger contains {len(errors)} operational error events. This total is not an invented estimate for any failed request without usage metadata. Credit/balance failures receive a shared wait of at least 90 seconds before retrying. Only the first structurally valid grade per frozen input and preprocessing version is accepted; no score-based retries or edits. Initially, one Nano input received four provider `reasoning_extraction` rejections and remained rejected after an additive task clarification. That note requested grading of literal recorded output with brief score justifications; it did not remove the analysis-like source text, change criteria, or request hidden reasoning. The exact exception is in [task-clarifications.json](task-clarifications.json), with source/rubric/model/effort unchanged and the attempted clarification hash in the attempt ledger. The original unscored rejections and later retry outcomes remain in [retries/20261018/](retries/20261018/). Provider rejections never count as quality passes or failures.

{retry_note}

## Selection and comparability

The [source audit](SOURCE-AUDIT.md) and [configuration inventory](configurations.json) cover copies, archives, diagnostic captures, and configuration variants. Each text row uses its most recent matching complete conversations, up to 30. A complete transcript has every scripted index 0–29 exactly once with an assistant response or tool call. Wrong or absent required tools remain quality issues. Recovery responses are preserved in source snapshots and included in text latency. The existing judge formatter omits recovery records from both its prompt and scored denominators; later scripted responses are evaluated from the scripted transcript and realignment evidence. This limitation is shared across this refresh.

Fourteen text rows have a shortfall, detailed in the [cohort-size audit](../../docs/text-cohort-sizes-2026-10-17.md). We did not pad sample sizes by mixing filler, reasoning, precision, or serving configurations. Voice uses all 25 validated completions of each final published capture profile; newer valid matching captures were not found.

This is a **conditional-on-completion analysis**. Incomplete conversations and observed prefixes are excluded; missing future turns are not errors in this analysis. Earlier fixed-denominator, attempt-based scores answer a different question. Completion selection can favor configurations with high failure rates, especially Lightning off; quality here must not be interpreted as reliability. Voice separately retains its full 245-attempt ledger and all six failures/14 harness defects. Text manifests enumerate incomplete and differently configured candidates; historical attempt-based campaigns remain available for reliability context.

The four Nemotron 3 rows replace old unmapped Baseten summaries. Ultra uses coherent native-budget Modal captures; Super/Nano use coherent native-budget NVFP4 AWS B200 captures. Super uses the newest 30 of a recovered 40-completion study. Nano has seven at its replacement configuration; nine older legacy-parameter captures are not pooled. Ultra 96 has 19 matching native-budget captures, despite many more captures at other settings. These changes affect serving/cohort and timing as well as judging. Provider labels and NVFP4 labels were corrected. Old versus new differences are **not causal judge comparisons**. OpenAI row definitions match model, reasoning, and filler rather than creating separate rows for every historical service tier; recorded priority/fast labels and unlogged older tiers remain explicit in manifests. Their latency is a historical cohort measurement, not a standardized tier experiment. No cross-model significance claims are made: capture dates, hardware, SDKs, caching, samplers, and sometimes prompts differ.

## Quality and uncertainty

Strict pass requires tool use, instruction following, and KB grounding all to pass. Component errors overlap. Turn-taking is reported separately and never enters strict quality, even where available. Intervals use 20,000 whole-conversation bootstrap resamples with deterministic per-cohort seeds. They account for within-conversation turn correlation, but not judge uncertainty, configuration differences, or collection drift; a 100% sample yields a degenerate empirical interval and does not establish population perfection.

{table(['Configuration', 'Selected N', 'Graded N', 'Provider blocked', 'Known strict passes / graded turns', 'Full-cohort pass rate', 'Cluster 95% CI', 'Current grade cost'], quality_rows)}

Provider rejection is missing judge data, not observed model correctness. No point estimate or bootstrap interval is published for an incomplete quality cohort. Known successful grades remain in results and aggregate counts; bounds in aggregate.json show the possible full-cohort accuracy range without imputing unknown turns. Rows with unavailable quality sort after fully graded rows. The fixed selected membership is not changed based on provider acceptance.

## Latency

All numbers were recalculated from the selected frozen evidence. [latency-observations.json](latency-observations.json) contains the individual values, turn indices, recovery flags, and voice tool/silence observations. Text excludes the first scripted response, includes recovery, and publishes positive finite recorded service TTFAT; P95 uses linear interpolation. Every selected text source was checked byte-for-byte against its snapshot. Native thinking paths measure user-visible content/tool output rather than reasoning deltas where exposed by the original service. This does not standardize SDK versions or provider instrumentation retroactively. Local rows omit hosted network time.

Voice recomputation exactly reproduces every historical sample count, mean, median, maximum, missing count, and nonpositive count for all nine profiles. Original audio JSON, original quality reconstruction, and cleaned spoken input are hashed and checked. All scripted turns enter the voice measurement scope. Actual first speech includes acknowledgments and is not necessarily the substantive answer. Silence uses nonnegative original Silero padding; RMS is not substituted. Unavailable values remain null.

{table(['Voice configuration', 'Positive V2V', 'Nonpositive', 'Missing', 'Tool', 'Non-tool', 'Silence', 'Turn-taking passes / 750'], coverage)}

## Selected source coverage

Dates below are recorded source timestamps (or capture-ID timestamps when that field is unavailable). Every selected path and SHA, available generation configuration, candidate exclusion, and archive reference is preserved in the linked manifest.

{table(['Configuration', 'Provider', 'N', 'Earliest', 'Latest', 'Evidence'], dates)}

## Historical preservation and reproduction

The [previous README](historical/README.md) and [previous standalone leaderboard](historical/leaderboard-medium-context.md) preserve publication context; their relative links are rooted at the repository. The September speech package, source archives, raw captures, old judges, and earlier campaign reports were not rewritten. New scores are separate artifacts. Raw SDK responses and cleaned inputs are local-only under `runs/{HERE.name}`; committed results and manifests preserve hashes and structured judgments.

Run from the repository root with the existing environment:

```bash
.venv/bin/python {LINK}/campaign.py status
.venv/bin/python {LINK}/finalize.py
```

Reproduction of metrics requires the local source captures and frozen snapshots. Resuming judging validates matching input, rubric, model, effort, and all 30 score records before reusing a result. Publication requires every selected input to have a validated grade or an explicit provider-blocked disposition. The aggregate records processing_complete separately from complete/all_quality_complete; the latter remain false while any selected grade is unavailable. Original archives are read without mutation.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    data=json.loads((HERE/'aggregate.json').read_text())
    assert data['processing_complete'] is True
    assert len(data['text_models']) == 43 and len(data['voice_models']) == 9
    assert sum(m['complete'] for m in data['text_models']+data['voice_models']) == 1316
    text = 'Text mode models:\n\n' + f'The standalone copy is [leaderboard-medium-context.md](leaderboard-medium-context.md).\n\n![Text-model pass rates and recalculated TTFAT P50/P95](docs/text-model-voice-readiness.svg)\n\nRows are ordered by available pass rate descending, then TTFAT P50 ascending. Rows with unavailable quality appear last; N counts selected complete conversations.\n\n' + text_table(data) + '\n\n' + text_notes() + '\n\n'
    readme=(ROOT/'README.md').read_text()
    start=readme.index('Text mode models:\n')
    stop=readme.index('### Sample Recordings', start)
    readme=readme[:start]+text+voice_section(data)+readme[stop:]
    start=readme.index('### GPT-Live validation notes')
    stop=readme.index('## Quick Start',start)
    readme=readme[:start]+'''### GPT-Live validation notes

Both GPT-Live configurations use `gpt-live-1` with delegated `gpt-5.6-terra` at medium reasoning. The standard profile shares the speech-v2 frontend prompt and knowledge base; the optimized profile splits conversational frontend instructions from backend procedures. Their scores remain separate. Both final cohorts use validated completion gates without a runtime text classifier. Earlier broken-gate and custom-frontend probes remain diagnostics, as documented in the [historical report](docs/speech-results-2026-09/REPORT.md#historical-diagnostics) and [capture protocol](docs/speech-results-2026-09/PROTOCOL.md).

'''+readme[stop:]
    (ROOT/'README.md').write_text(readme)
    (ROOT/'leaderboard-medium-context.md').write_text(('# aiwf_medium_context text leaderboard\n\n'+text).rstrip()+'\n')
    (HERE/'REPORT.md').write_text(report(data))
    # Later approved fresh cohorts supersede only the current publication;
    # this refresh's frozen aggregate and report retain their original scope.
    ultra=ROOT/'ops/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018/PUBLICATION.json'
    if ultra.exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/publish_ultra_baseten.py')],cwd=ROOT,check=True)
    recent=ROOT/'ops/recent-models-20261018/PUBLICATION.json'
    if recent.exists():
        subprocess.run([sys.executable,str(recent.parent/'publish.py')],cwd=ROOT,check=True)
    high=ROOT/'ops/grok-voice-high-20261018/PUBLICATION.json'
    if high.exists():
        subprocess.run([sys.executable,str(high.parent/'publish.py')],cwd=ROOT,check=True)

    repair=ROOT/'ops/responses-repair-20261018/PUBLICATION.json'
    if repair.exists():
        subprocess.run([sys.executable,str(repair.parent/'publish.py')],cwd=ROOT,check=True)


if __name__ == '__main__':
    main()
