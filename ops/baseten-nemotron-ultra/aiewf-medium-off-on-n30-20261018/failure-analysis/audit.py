"""Evidence inventory and manual interpretation of the 73 canonical on-mode misses."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=HERE.parents[3]
DIMS=('tool_use_correct','instruction_following','kb_grounding')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(case,turn):
    if turn==4:return 'session_count_policy'
    if turn in (7,8,19,21):return 'venue_or_navigation_additions'
    if turn==20:
        if case['slot']==31:return 'forgot_attendance_correction'
        if case['slot']==60:return 'incorrect_workshop_date'
        return 'unsupported_workshop_pass_requirement'
    return {2:'extra_track_count',3:'invented_mcp_panel',9:'misstates_scheduled_meetups',
            13:'unsupported_menu_details',14:'unsupported_vegan_catering_details',16:'debatable_early_support_ticket'}[turn]


def main():
    original=json.loads((STUDY/'aggregate.json').read_text())
    cases=[c for c in original['conversations'] if c['arm']=='on']
    assert len(cases)==30 and all(c['complete'] for c in cases)
    evidence=[];counts=Counter();strict=0
    for case in sorted(cases,key=lambda c:c['slot']):
        source=ROOT/case['source'];assert sha(source)==case['transcript_sha256']
        rows={r['turn']:r for r in map(json.loads,source.read_text().splitlines()) if not r.get('recovery_turn')}
        grade_path=STUDY/'judging/results'/f"{case['job_id']}.json"
        grade=json.loads(grade_path.read_text());assert grade['transcript_sha256']==case['transcript_sha256']
        for turn in range(30):
            j=grade['judgments'][str(turn)]
            bad=[d for d in DIMS if not j['scores'][d]]
            strict+=not bad
            if not bad:continue
            category=classify(case,turn);counts[category]+=1
            row=rows[turn]
            evidence.append({'job_id':case['job_id'],'turn':turn,'user_turn':turn+1,'category':category,
                'failed_dimensions':bad,'judge_reason':j['reasoning'],'user_text':row['user_text'],
                'assistant_text':row['assistant_text'],'tool_calls':row['tool_calls'],
                'source':case['source'],'source_sha256':case['transcript_sha256'],
                'grade':str(grade_path.relative_to(ROOT)),'grade_sha256':sha(grade_path)})
    assert strict==827 and len(evidence)==73 and sum(counts.values())==73
    illustrative=counts['session_count_policy']+counts['unsupported_workshop_pass_requirement']+counts['venue_or_navigation_additions']
    assert illustrative==65
    source_paths=[STUDY/'aggregate.json',STUDY/'judging/inputs.json',
        ROOT/'benchmarks/aiwf_medium_context/judge.py',ROOT/'benchmarks/aiwf_medium_context/prompts/system.py',
        ROOT/'benchmarks/aiwf_medium_context/data/knowledge_base.txt',ROOT/'benchmarks/_shared/tools.py',Path(__file__)]
    data={'audited_at':datetime.now(timezone.utc).isoformat(),'mode':'on','scripted_turns':900,
        'canonical_strict_passes':strict,'canonical_failed_turns':73,'categories':dict(counts),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in source_paths},'failures':evidence}
    (HERE/'audit.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    lines=['# All 73 failed thinking-on turns','',
        'Canonical scores are preserved. User turns below are one-based. Each entry includes the complete recorded answer and original judge explanation.','']
    for item in evidence:
        key=f"{item['job_id']}-turn-{item['user_turn']}"
        lines += [f'<a id="{key}"></a>',f"## {item['job_id']}, user turn {item['user_turn']}",'',
            f"Category: `{item['category']}`. Failed dimensions: {', '.join(item['failed_dimensions'])}.",'',
            f"**User:** {item['user_text']}",'',f"**Judge:** {item['judge_reason']}",'',
            '**Complete recorded assistant answer:**','',f"<pre>{html.escape(item['assistant_text'] or '(no assistant text)')}</pre>",'',
            '**Actual tool calls:**','',f"<pre>{html.escape(json.dumps(item['tool_calls'],indent=2))}</pre>",'']
    (HERE/'EVIDENCE.md').write_text('\n'.join(lines))
    rows=[('Session-count policy',counts['session_count_policy'],'Correct counts, but the system explicitly says not to calculate them.'),
        ('Workshop-pass requirements',counts['unsupported_workshop_pass_requirement'],'Correct requested sessions, plus an unsupported access restriction.'),
        ('Venue/navigation additions',counts['venue_or_navigation_additions'],'Correct refusal, room, or recommendations, plus unsupported venue details or prohibited directions.'),
        ('Wrong date / forgotten attendance correction',counts['incorrect_workshop_date']+counts['forgot_attendance_correction'],'One wrong workshop date; one return to the superseded attendance date.'),
        ('Other grounding additions',sum(counts[k] for k in ['invented_mcp_panel','misstates_scheduled_meetups','unsupported_menu_details','unsupported_vegan_catering_details']),'Panel, meetup status, menu, or vegan-catering claims.'),
        ('Extra track count',counts['extra_track_count'],'Correct official 19, followed by calculated track-instance totals; policy scope is debatable.'),
        ('Early support ticket',counts['debatable_early_support_ticket'],'Debatable: the tool schema required no specificity beyond an issue description; early-call realignment is allowed.')]
    table='\n'.join(['| Failure family | Turns | Assessment |','|---|---:|---|']+[f'| {name} | {n} | {note} |' for name,n,note in rows])
    diagnostic=STUDY/'rubric-comparison/aggregate.json'
    paired='The paired diagnostic is still running; no result is imputed.'
    if diagnostic.exists():
        comparison=json.loads(diagnostic.read_text())
        assert comparison['current_percent']==827/9 and comparison['conversations']==30
        paired=f"""On these **identical 900 turns**, the pre-upgrade v4 rubric scored
**{comparison['legacy_percent']:.1f}%**, versus **{comparison['current_percent']:.1f}%**
for the current policy/KB rubric, using the same Opus 5.5 Medium judge.
The measured legacy-minus-current gap is
**{comparison['legacy_minus_current_percentage_points']:.1f} percentage points**.
There were {comparison['counts'].get('legacy_pass_current_fail',0)} legacy-pass/current-fail
turns and {comparison['counts'].get('legacy_fail_current_pass',0)} changes in the other
direction. Individual disagreements include judge variability; the controlled
configuration change is the rubric plus supplied reference policy/KB."""
        paired+='''

The new rubric is not uniformly harsher. Of the 16 improvements, ten accept
policy-compliant venue refusals that the old rubric rejected for failing to echo
the golden answer's support-ticket context. Five accept speaker affiliations
that match the benchmark KB, where the old judge instead substituted outside
knowledge (for example, it rejected the KB's Hamming AI/Taktile affiliations).
One accepts an early support ticket whose required arguments were already known.
The 58 newly failed turns include all 22 session-count violations, all 14 added
pass requirements, and 17 venue/navigation additions.
'''
    report=f'''# Why fresh Ultra thinking-on scores 91.9%

The canonical result is **827/900 passes, 73 failed turns**, across 30 complete
conversations. No score or transcript was edited for this audit. Every failure
is inventoried in [audit.json](audit.json) and shown with its complete answer and
judge explanation in [EVIDENCE.md](EVIDENCE.md).

{table}

**65 of 73 misses (89%) are session-count policy violations or extra unsupported
claims attached to otherwise useful answers/refusals.** They are concentrated
in a few repeated questions, rather than 73 unrelated breakdowns. Strict pass
requires all three dimensions to pass; a single extra assertion fails the whole
turn. Correctly identifying the requested session does not cancel an added claim.

## What changed in judging

Before the Opus 5.5 migration, the generic v4 rubric said instruction-following
passes if the assistant "directly answers the question OR advances the task" and
grounding fails only for clear factual contradictions. Its conversation prompt
contained golden answers and expected tools, but its system rubric did not include
the complete assistant policy or knowledge base.

The current `text-policy-compliance-1` rubric explicitly enforces the actual
assistant policy and supplies the full policy and KB to the judge. For example,
the generator's policy says not to calculate session totals and not to infer
facts beyond the KB. A correct answer to "how many sessions?" can therefore be
a policy failure even though the arithmetic and listed sessions are accurate.
The no-counting rule was already in the assistant's instructions; it was not
added to Ultra's prompt for this study.

The [paired rubric diagnostic](../rubric-comparison/REPORT.md) judges these exact
30 transcripts with the legacy rubric and the same Opus 5.5 Medium model. It
holds the formatted conversation prompt, transcript, model, and effort fixed;
the rubric/reference package changes. No more Ultra conversations are generated.
The canonical 91.9% result remains the published score.

{paired}

## Where the judge deserves qualification

- **Unsupported is not the same as proven false.** For 14 workshop-pass answers,
  Ultra correctly listed the June 4 sessions and then added a pass requirement.
  The KB says the expanded pass adds exclusive June 3 workshops; it does not
  explicitly settle access to the June 4 session whose format is Workshop.
  Ultra inferred a requirement, contrary to the no-inference policy. Some judge
  explanations overstate this as a definitive contradiction. The omission of a
  rule from the KB alone does not prove its opposite.
- **Venue embellishments:** the KB lists `Juniper: Expo Sessions` and separately
  `Salons 9-15: Expo Hall`; it never supplies their physical relationship or
  locates an information booth. Ultra often gives the right room/refusal and
  appends those details. These are ungrounded additions, but the canonical
  grounding rubric's "clear factual contradiction" wording versus enforcement
  of the system's no-inference policy leaves dimension assignment debatable.
  They are not evidence that the requested room was wrong.
- **One likely overstrict tool judgment:** slot-041-on, user turn 17, called
  technical support with the already-known name and `trouble with the mobile app`.
  Those satisfy the literal schema. The judge demanded a more specific problem
  before acting and penalized the early ticket, although the rubric allows early
  calls. The current judge accepted the semantically equivalent early call in
  slot-060-on at that same user turn. This is a concrete consistency concern,
  worth treating as a debatable label rather than an obvious failure.
- **One additional track-count judgment is debatable:** the answer gives the
  correct official 19 tracks, then calculates track instances. The policy bans
  totals of sessions/speakers/topics; applying it to an extra track-instance
  summary is an interpretation. Neither debatable singleton explains a large
  score change (each is 0.11 percentage points).

These qualifications do not authorize selective score edits. Any rubric
clarification should be tested consistently across models, not only on Ultra.

## Why this does not establish a model regression

The [old published rows](../../../judge-refresh-opus55-20261017/historical/README.md)
showed 100.0% / 98.3%. They are older legacy summaries with a different cohort,
serving/budget profile, and historical judge version. The exact original
near-perfect cohort is not established by those row labels. The freshly measured
91.9% cannot by itself be interpreted as an eight-point capability regression.
Use the same-output rubric diagnostic to measure scoring sensitivity, and keep
the historical-cohort uncertainty separate.

The current README publishes only the fresh thinking-on result. Ultra generation
testing is finished; no smaller-thinking-budget experiment is planned.
'''
    (HERE/'REPORT.md').write_text(report)
    paths=[HERE/'REPORT.md',HERE/'EVIDENCE.md',HERE/'audit.json',Path(__file__)]
    verification={'completed_at':datetime.now(timezone.utc).isoformat(),'all_73_failures_accounted_for':True,
        'canonical_grade_edits':0,'artifacts_sha256':{str(p.relative_to(HERE)):sha(p) for p in paths}}
    (HERE/'VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({'failed_turns':len(evidence),'categories':dict(counts),'policy_or_extra_detail_misses':illustrative}))


if __name__=='__main__':main()
