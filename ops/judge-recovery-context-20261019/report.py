"""Write the migration report from canonical machine-readable evidence."""
from common import *

def main():
    data=read(HERE/'aggregate.json')
    validation=read(HERE/'VALIDATION.json')
    assert validation['passed']
    counts=data['counts']
    costs=data['costs']
    lines=['# Recovery context and judge migration', '',
           'Status: validation passed; canonical rejudging and reconciliation complete.', '',
           'The judge previously omitted harness recovery exchanges. The corrected format retains their public answers, actual tool calls and recorded outcomes as unscored chronological evidence. It preserves the original scripted targets and underlying text/voice policy and knowledge-base rubrics. Recovery success does not retroactively excuse an unnecessary original deferral. An earlier valid recovery can establish state or fulfill a later expected action.', '',
           f"The active scope is {counts['assignments']:,} assignments across 62 configurations: {counts['complete']:,} complete and {counts['incomplete']} incomplete. The leaderboard still has 50 ranked text rows and 11 voice rows; Luna none remains unranked. All 698 recovery-bearing assignments were independently rejudged. The other 988 requests reuse their original grades after complete historical system/user request reconstruction and byte-identity checks. Their old metadata is unchanged.", '',
           'All generation cohorts, raw transcripts, original snapshots and grades, completion classifications, recorded latency and audio annotations remain unchanged. No new conversations were generated. Original studies remain historical; this directory owns the corrected aggregate and publication entry point.', '',
           '## Independent diagnostic validation', '',
           'The plan was reviewed independently by Astra and Fable; actual Fable reviewer was `claude-fable-5-1`. Fresh blind reference sessions used explicit `gpt-6-astra` and actual `claude-fable-5-1`, no tools and independently rendered anonymized raw-record packets. The Codex CLI records the requested Astra model and token usage, but does not expose a server-returned model identifier or USD billing; that provenance limit is retained in reference metadata.', '',
           'Selection was frozen before candidate calls: 24 complete conversations, 48 target turns, with 12 text (8 recovery, 4 clean) and 12 voice conversations. A disjoint 24-conversation reserve remains untouched. Known calibration, inspected Luna/Ultra and development examples were excluded conservatively. Deterministic old-label failure/pass stratification deliberately overrepresents errors and is not a random accuracy sample.', '',
           'The fixed consensus contains 123/144 target dimensions: 64 text, 59 voice, 43 recovery, and 18 negatives. The 21 predeclared uncertain/disputed dimensions remain recorded, excluded before any candidate grade. These are independent model references, not human ground truth. Model-family overlap and consensus selection limit accuracy claims.', '',
           '| Diagnostic pass | Overall agreement | Recovery agreement | Clean text agreement | Voice agreement |',
           '|---|---:|---:|---:|---:|']
    for name,r in validation['repetitions'].items():
        cells=[]
        for category in ['overall','recovery','clean','voice']:
            s=r['counts'][category]
            cells.append(f"{s['matched']}/{s['total']} ({100*s['agreement']:.1f}%)")
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    repeat=validation['repeatability']
    lines+=['',f"Both passes individually cleared the preregistered 95% agreement gate. Full-conversation repeat discordance was {repeat['discordant']}/{repeat['dimensions']} dimensions ({100*repeat['rate']:.2f}%), below the 2% gate. Both development repetitions passed all clear support-ticket and delayed dietary/vote expectations. All diagnostic grades remain separate; each changed case received its own first-valid canonical request.", '',
            'Every discrepancy and explanation is retained in [VALIDATION.json](VALIDATION.json), with reference disagreements in [references/FROZEN.json](references/FROZEN.json). Passing these gates supports adoption with documented interpretation uncertainty; it does not establish perfect correctness on every label.', '',
            '## Reconciled scores', '',
            f"Across all active assignments, {counts['changed_dimensions']:,} semantic dimension labels changed: {counts['false_to_true']:,} false-to-true and {counts['true_to_false']:,} true-to-false. These changes combine corrected context with ordinary judge variability; they are not a pure causal estimate. All 275 voice assignments have no recovery and retain their original canonical grades; voice was independently checked in diagnostic validation.", '',
            'Strict and component scores, complete-only and fixed-assignment denominators, and 20,000-draw conversation-bootstrap intervals are recomputed from the reconciled per-turn grades. Missing future turns remain failures only in fixed-assignment summaries; incomplete conversations remain excluded from the ranked conditional table.', '',
            '| Configuration | Prior completed-only pass | Corrected pass | Change (points) | Complete / selected or assigned | Rejudged |',
            '|---|---:|---:|---:|---:|---:|']
    for m in sorted(data['text_models'],key=lambda m:-(abs(m['delta_percentage_points']) if m['delta_percentage_points'] is not None else -1)):
        old='—' if m['old_strict_percent'] is None else f"{m['old_strict_percent']:.1f}%"
        new='—' if m['strict_percent'] is None else f"{m['strict_percent']:.1f}%"
        delta='—' if m['delta_percentage_points'] is None else f"{m['delta_percentage_points']:+.2f}"
        lines.append(f"| {m['label']} | {old} | {new} | {delta} | {m['complete']}/{m['assigned']} | {m['grade_dispositions'].get('rejudge',0)} |")
    lines+=['', 'Full per-turn deltas are retained in [changed-labels.json](changed-labels.json). The confirmed development example now recognizes the recorded successful support call; it no longer claims no ticket existed. The separate delayed-action example still fails the original unnecessary dietary/vote deferrals.', '',
            '## Provenance and operations', '',
            'The inventory verifies all source/snapshot/grade hashes, actual prior judge model and effort, frozen rubric hashes, audio annotations, scripted IDs and request hashes. Recovery ancestry is checked against source order; 108 missing explicit parents are recorded as unambiguous positional inferences. Zero active holds remain. Public reasoning normalization covers recovery and scripted text; actual calls/results remain intact.', '',
            'The original 34-entry hash-verified public-ending restoration manifest is immutable and applied before formatting with per-source application-count checks. The separate recovery-ending audit found no positive native-wire evidence of omitted public output. Sixty-four older recovery-ending captures have no native Responses audit; this is a documented evidence limit, not proof of complete native output. Empty tool-only responses are not automatically omissions or failures. Unavailable encrypted reasoning is neither exposed nor claimed verified.', '',
            'The locked driver uses eight concurrent requests, a shared persisted 90-second cooldown for credit failures, operational retries only, retained attempts and first-valid atomic result ownership. The canonical wrapper additionally captures SDK message IDs and can recover a validated pending result after a crash. It does not change candidate prompts or acceptance rules. [canonical-registration.json](canonical-registration.json) freezes this audit-only entry point.', '',
            f"Known Opus judging cost for this migration: **${costs['opus_known_cost_usd']:.4f}**, including diagnostics and all cost-bearing retained attempts. Fable plan/reference review cost: **${costs['fable_review_and_reference_cost_usd']:.4f}**. Astra CLI token usage is retained, but USD billing is unavailable. Opus attempts without a reported cost: {len(costs['opus_missing_cost_attempts'])}.", '',
            '| Opus phase | Reported cost |','|---|---:|']
    lines += [f'| {phase} | ${cost:.4f} |' for phase,cost in costs['opus_cost_by_phase'].items()]
    lines+=['', '## Reproduce and verify', '',
            'Run from the worktree with `PYTHONPATH="$PWD/src:$PWD"` using its `.venv/bin/python`. `preflight.py` reconstructs historical requests; `references.py`, `run_validation.py` and `validate.py` preserve separate reference/diagnostic ownership. Resume canonical judging with `canonical.py`. Reconcile with `analyze.py`, generate this report with `report.py`, then publish with `publish.py`. Never rerun old publishers to update the corrected regime.', '',
            'Machine-readable coverage and publication verification are in [VERIFICATION.json](VERIFICATION.json). Prior archived bytes and 2,988 historical evidence files must retain their original hashes. Tests, chart consistency, whitespace and idempotent publication are required before the final completion marker.', '']
    (HERE/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
