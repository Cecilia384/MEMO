#!/usr/bin/env python3
"""Summarize instrumented timing runs; keep scope and warmup counts explicit."""
import argparse
import json
from pathlib import Path
from statistics import mean

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--warmup-questions',type=int,default=1)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    data=json.loads(Path(args.result).read_text())
    if data.get('status')!='completed' or not data['config']['profile']:
        raise ValueError('A completed profiling run is required')
    if not 0<=args.warmup_questions<len(data['results']):
        raise ValueError('Warmup count must leave at least one measured question')
    rows=data['results'][args.warmup_questions:]
    def stats(values):
        values=[v for v in values if v is not None]
        return {'count':len(values),'mean_ms':mean(values) if values else None}
    report={'measured_questions':len(rows),'excluded_initial_questions':args.warmup_questions,
            'peak_allocated_gib':data['peak_allocated_gib'],
            'query':{key:stats([r['metrics'].get(key) for r in rows]) for key in ('retrieval_ms','preprocessing_ms','generation_ms','response_ms','ttft_ms','tpot_ms')},
            'module_all_frames':{k:stats(v) for k,v in data['module_profile_ms'].items()},
            'ingest_per_chunk':stats([v for video in data['videos'] for v in video['ingest_ms']]),
            'note':'Profiling serializes perception modules and synchronizes token timestamps. No repeated-generation benchmark; peak memory covers the full run.'}
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n'); print(out)

if __name__=='__main__': main()
