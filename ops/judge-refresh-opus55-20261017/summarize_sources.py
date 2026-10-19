"""Export directory/configuration inventory without altering frozen cohorts."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import campaign
import select_text

HERE, ROOT = campaign.HERE, campaign.ROOT
RAW = ROOT/'runs'/HERE.name


def write_table(path, header, rows):
    with path.open('w') as handle:
        writer=csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)


def main():
    audit=json.loads((RAW/'all-local-captures.json').read_text())
    broad=json.loads((RAW/'broad-discovery-check.json').read_text())
    extras=broad['extra_nonstandard_complete_transcripts']
    rules=json.loads((HERE/'text-rules.json').read_text())
    decisions=[]
    for item in extras:
        item['conflicting_capture_id']=False
        applicable=[r for r in rules if r['family']==item['family']]
        reasons={r['key']:select_text.mismatch(r,item) for r in applicable}
        assert all(reasons.values()), (item['sources'],reasons)
        decisions.append({'source':item['sources'][0], 'sha256':item['transcript_sha256'],
                          'family':item['family'], 'captured_at':item['first_record_timestamp'],
                          'exclusions':reasons,'configuration':item['configuration']})
    campaign.dump(HERE/'supplemental-source-audit.json', {
        'note':'Supplementary timing/DFlash experiments and copies outside benchmark-named folders; every source fails all matching retained configuration rules. No frozen membership changes.',
        'transcript_like_paths_examined':broad['searched_paths'],
        'new_standard_transcripts_outside_campaign_or_tests':0,
        'orphan_complete_judged_transcripts':broad['orphan_complete_judged_transcripts'],
        'parse_errors':broad['errors'], 'decisions':decisions,
        'other_conversation_file_directories':broad['other_conversation_file_directories'],
    })
    groups={}
    captures=[]
    sources=[]
    dirs=defaultdict(lambda:{'sources':0,'hashes':set(),'complete_hashes':set(),'families':set()})
    # Supplementary timing-copy counts are kept separate from primary dedup counts.
    for scope, items in [('primary',audit['captures']), ('supplemental',extras)]:
        for item in items:
            config={k:v for k,v in item['configuration'].items() if k!='log_prefix_sha256'}
            encoded=json.dumps({'family':item['family'],'model':item['model'],'config':config},sort_keys=True)
            cid=hashlib.sha256(encoded.encode()).hexdigest()[:16]
            if cid not in groups:
                groups[cid]={'id':cid,'family':item['family'],'model':item['model'],'configuration':config,
                             'primary_transcripts':0,'primary_complete':0,'supplemental_files':0,'selected':0,
                             'earliest':None,'latest':None,'directories':set()}
            group=groups[cid]
            group['primary_transcripts' if scope=='primary' else 'supplemental_files']+=1
            group['primary_complete']+=int(scope=='primary' and item['complete'] and item['scripted_users_match'])
            date=item['first_record_timestamp']
            if date:
                group['earliest']=min(group['earliest'] or date,date)
                group['latest']=max(group['latest'] or date,date)
            captures.append([scope,item['transcript_sha256'],item['family'],item['capture_id'],int(item['complete']),
                             int(item['scripted_users_match']),cid,date or '',item['configuration'].get('log_prefix_sha256','')])
            for source in item['sources']:
                directory=str(Path(source).parent.parent)
                group['directories'].add(directory)
                stats=dirs[directory]
                stats['sources']+=1; stats['hashes'].add(item['transcript_sha256']);stats['families'].add(item['family'])
                if item['complete'] and item['scripted_users_match']:stats['complete_hashes'].add(item['transcript_sha256'])
                source_scope = 'restored' if f'/runs/{HERE.name}/restored/' in source else scope
                sources.append([source_scope,item['transcript_sha256'],source])
    selected=Counter(i['transcript_sha256'] for c in campaign.cohorts('text') for i in c['inputs'])
    by_hash={r[1]:r[6] for r in captures if r[0]=='primary'}
    for sha,count in selected.items(): groups[by_hash[sha]]['selected']+=count
    for group in groups.values():group['directories']=sorted(group['directories'])
    campaign.dump(HERE/'configurations.json',{'primary_summary':audit['summary'], 'discovered_capture_files':audit['discovered_capture_files'],
        'unique_primary_transcripts':audit['unique_transcripts'],'copies':audit['duplicates'], 'restored_source_copies':sum(row[0]=='restored' for row in sources), 'groups':sorted(groups.values(),key=lambda r:(r['family'],r['id']))})
    write_table(HERE/'capture-inventory.tsv',['scope','transcript_sha256','family','capture_id','complete','scripted_users_match','configuration_id','captured_at','startup_log_sha256'], captures)
    write_table(HERE/'source-paths.tsv',['scope','transcript_sha256','source'], sources)
    write_table(HERE/'directory-inventory.tsv',['directory','source_files','distinct_hashes','complete_hashes','families'],
                [[d,v['sources'],len(v['hashes']),len(v['complete_hashes']),','.join(sorted(v['families']))] for d,v in sorted(dirs.items())])
    rows=[]
    for c in sorted(campaign.cohorts('text'),key=lambda c:c['label']):
        short=c['shortfall']; dates=sorted(i.get('captured_at') or '' for i in c['inputs'])
        rows.append('| '+ ' | '.join([c.get('display_label',c['label']),str(c['actual_n']),str(c['candidate_audit']['eligible_complete']),str(short),dates[0][:10],dates[-1][:10],f"[manifest](../ops/{HERE.name}/cohorts/{c['key']}.json)"])+' |')
    note='''# Complete-conversation cohort sizes, October 17–18, 2026

The exhaustive local search supersedes the earlier summary-only size assessment. It includes unjudged transcripts, nested projects, other checkouts, and compressed Nemotron sweeps. Historical study counts are not the selected sample size: the new analysis uses the latest complete conversations at a matching configuration, up to 30 text / 25 voice.

There are **43 text configurations, 1,091 selected complete conversations, and 14 text shortfalls**. All nine voice profiles have 25, for 225 voice conversations. No new conversations were generated. Incomplete traces and their observed prefixes are excluded from accuracy, so this is conditional-on-completion quality. Incorrect tools are graded rather than excluded.

The primary inventory has **8,060 source files, 7,370 distinct transcript byte hashes, and 690 copies across 35 text model families**. A supplementary filename-independent check found 203 complete Qwen timing/DFlash files or copies; all fail the selected serving/prompt profile. It found no additional standard transcripts outside new campaign/test files, no orphan complete judged captures, and no parse errors. These supplementary files are not counted as independent repetitions.

Most importantly, the old Super archive contains **40 complete native-budget 512 conversations**, despite an old summary reporting only 38 judged conversations. Its newest 30 are remapped to an explicit AWS B200 NVFP4 row. Nano has seven complete native-budget 512 NVFP4 conversations and nine older complete legacy-parameter 512 captures on a different stack; they are not pooled. Ultra 128 has 33 matching native-budget Modal completions, of which 30 are selected; Ultra 96 has 19. Across all serving/sampling variants the corresponding budget counts are much larger (Ultra 128: 77, Ultra 96: 56, Super 512: 81, Nano 512: 16), but these are not coherent cohorts.

The same distinction explains the OpenAI shortfalls: alternate effort levels, 96-dot filler, incomplete trials, and copied files do not count toward a matching 30. Counts below are selected complete conversations; available is the matching complete pool before taking the newest 30. See each manifest for exclusions, source timestamps, recorded request settings, and input hashes.

| Configuration | Selected N | Available | Shortfall | Earliest selected | Latest selected | Evidence |
|---|---:|---:|---:|---|---|---|
'''
    (ROOT/'docs/text-cohort-sizes-2026-10-17.md').write_text(note+'\n'.join(rows)+f'\n\nThe [source audit](../ops/{HERE.name}/SOURCE-AUDIT.md), [directory inventory](../ops/{HERE.name}/directory-inventory.tsv), and [configuration inventory](../ops/{HERE.name}/configurations.json) preserve the broader search. Different efforts, filler, quantizations, and serving endpoints are not merged merely to reach 30. Voice capture diagnostics and harness defects remain in the original attempt ledger.\n')
    print(json.dumps({'configurations':len(groups),'directories':len(dirs),'primary_sources':audit['discovered_capture_files'],'supplemental_sources':len(extras),'selected':sum(selected.values())}))


if __name__=='__main__':main()
