#!/usr/bin/env python3
"""Recompute scores and export measured results to JSON, CSV, and LaTeX."""
import argparse
import csv
import json
from pathlib import Path
from memo.benchmarks.scoring import score_answer, summarize
from memo.benchmarks.data import load_questions

def validate_result(data):
    if data.get('status')!='completed': raise ValueError('Only completed runs can be summarized')
    records=data['results'];ids=[r['id'] for r in records]
    if len(ids)!=len(set(ids)) or len(records)!=data['expected_questions']:
        raise ValueError('Duplicate or missing result IDs')
    for r in records:
        if not isinstance(r.get('prediction'),str) or not r['prediction'].strip():
            raise ValueError(f"Empty prediction: {r['id']}")
        r['choice'],r['correct']=score_answer(r,r['prediction'],data['config']['scoring'])
    return summarize(records,data['config']['benchmark'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',nargs='+')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--allow-subset',action='store_true')
    p.add_argument('--annotations',help='Optionally verify exact IDs against the original annotations')
    p.add_argument('--video-root')
    args=p.parse_args()
    if bool(args.annotations)!=bool(args.video_root): p.error('Supply both --annotations and --video-root')
    rows=[]; summaries=[]
    for name in args.results:
        data=json.loads(Path(name).read_text())
        summary=validate_result(data)
        if not args.allow_subset and (data.get('subset_run') or not summary['all_tasks_present']):
            raise ValueError('Subset or incomplete task coverage; use --allow-subset for a diagnostic report')
        if args.annotations:
            groups=load_questions(args.annotations,args.video_root,data['config']['benchmark'])
            expected={q['id'] for qs in groups.values() for q in qs}
            if expected!={q['id'] for q in data['results']}:
                raise ValueError('Result IDs do not match the full annotation set')
        row=dict(run=Path(name).stem,benchmark=data['config']['benchmark'],model=data['config']['model'],
                 questions=summary['answered'],**{k:v['accuracy_pct'] for k,v in summary['per_task'].items()},
                 average_pct=summary['paper_average_pct'])
        rows.append(row); summaries.append(dict(result=name,summary=summary))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    for name in ('summary.json','summary.csv','summary.tex'):
        if (out/name).exists(): raise FileExistsError(out/name)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    escape=lambda x:str(x).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    lines=[r'\begin{tabular}{'+'l'*len(columns)+'}',r'\hline',' & '.join(map(escape,columns))+r' \\',r'\hline']
    for row in rows:
        lines.append(' & '.join('--' if row.get(k) is None else escape(f'{row[k]:.2f}' if isinstance(row[k],float) else row[k]) for k in columns)+r' \\')
    lines += [r'\hline',r'\end{tabular}']
    (out/'summary.tex').write_text('\n'.join(lines)+'\n')
    print(out)

if __name__=='__main__': main()
