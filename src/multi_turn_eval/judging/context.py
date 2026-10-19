"""Chronological evidence for scripted judging targets and unscored recoveries."""

from datetime import datetime
import json

CONTEXT_REVISION = 'scripted-targets-with-recovery-1'

SCOPE_ADDENDUM = """\n\n# Input scope for recovery-context format revision 1
In this input format, 'every turn' and 'exactly one entry for each turn' mean
only the explicitly declared scripted scoring target IDs. Recovery R... records
and context-only scripted turns are chronological evidence, never scoring targets.
The early/late-call rules do not create a judgment or retroactive credit for an
unscored recovery. Apply the governing policy to the original scripted response
using information available at that point, and use recorded recovery actions and
outcomes when evaluating subsequent scripted claims and expected actions.
"""

RECOVERY_INSTRUCTIONS = """# Scoring targets and unscored conversation context
Only the scripted target IDs listed below receive judgments. Recovery exchanges
and context-only scripted turns are evidence, never additional scoring targets.
Records appear in their original chronological order. Recovery labels R... are
record identities, not scripted turn numbers. Scripted user messages are fixed
benchmark messages and need not react naturally to intervening recovery answers.

Judge a scripted response using information available at that point. An
unnecessary deferral of an authorized required action remains a failure at that
scripted turn even if its following recovery succeeds. Do not retroactively give
that turn credit, and do not credit an unrelated later scripted turn as though
the recovery call occurred there. A necessary clarification for an unavailable
required argument remains valid under the governing policy; recovery presence
alone is not a failure. Golden examples do not override the governing policy.

A valid action in an earlier recovery can satisfy a later expected action under
the normal early-call realignment rule. Later claims must reflect actual prior
actions and recorded results, including recovery outcomes. Duplicate-skipped is
not a new execution; consider the earlier action's evidence. Distinguish success,
failure, and an unrecorded outcome. Do not infer success or failure from absent
results, or infer execution from the assistant's prose alone. A correct actual
call can be evaluated as a call even where its result was not recorded.
"""


def validate_context(records, expected_count):
    """Validate without mutating records, retaining qualified source positions."""
    contexts = []
    last_scripted = None
    scripted = set()
    prior_time = None
    has_recovery = any(record.get('recovery_turn') for record in records)
    for index, record in enumerate(records):
        turn = record.get('turn')
        if type(turn) is not int or turn < 0:
            raise ValueError(f'Invalid transcript turn at record {index}')
        if has_recovery and record.get('ts'):
            try:
                stamp = datetime.fromisoformat(record['ts'].replace('Z', '+00:00'))
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f'Invalid chronology timestamp at record {index}') from error
            if prior_time is not None:
                try:
                    if stamp < prior_time:
                        raise ValueError(f'Contradictory transcript chronology at record {index}')
                except TypeError as error:
                    raise ValueError(f'Inconsistent timestamp timezone at record {index}') from error
            prior_time = stamp
        if record.get('recovery_turn'):
            if last_scripted is None:
                raise ValueError(f'Orphaned recovery at record {index}')
            parent = record.get('recovery_for_turn', last_scripted)
            if type(parent) is not int or parent != last_scripted:
                raise ValueError(f'Contradictory recovery parent at record {index}')
            contexts.append({'record_index': index, 'kind': 'recovery', 'turn': turn,
                             'parent': parent, 'parent_inferred': 'recovery_for_turn' not in record})
        else:
            if turn >= expected_count or turn in scripted:
                raise ValueError(f'Duplicate or out-of-range scripted target {turn}')
            if last_scripted is not None and turn <= last_scripted:
                raise ValueError(f'Out-of-order scripted target {turn}')
            scripted.add(turn)
            last_scripted = turn
            contexts.append({'record_index': index, 'kind': 'scripted', 'turn': turn})
    return contexts


def outcome_text(record):
    """Expose recorded outcome evidence without guessing from missing fields."""
    results = []
    for item in record.get('tool_results', []):
        response = item.get('response')
        evidence = response if response is not None else {}
        results.append({'name': item.get('name'), 'recorded_outcome': evidence or 'unrecorded'})
    return json.dumps(results) if results else 'unrecorded (no result record)'


def recovery_block(record, context):
    return [
        f"## Recovery R{context['record_index']} after scripted turn {context['parent']} (UNSCORED)",
        f"**User**: {record['user_text']}",
        f"**Assistant**: {record['assistant_text']}",
        f"**Actual Functions**: {json.dumps(record.get('tool_calls', []))}",
        f"**Recorded Tool Outcomes**: {outcome_text(record)}",
        '', '---', '',
    ]
