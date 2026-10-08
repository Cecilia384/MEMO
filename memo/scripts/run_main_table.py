#!/usr/bin/env python3
"""Preview or sequentially run the four backbones for one real-time benchmark."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
MODEL_DIRS={'qwen3':'Qwen3-VL-8B-Instruct','qwen25':'Qwen2.5-VL-7B-Instruct',
            'llava_7b':'llava-onevision-qwen2-7b-ov-hf','llava_05b':'llava-onevision-qwen2-0.5b-ov-hf'}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',choices=['streamingbench','ovobench'],required=True)
    p.add_argument('--models',choices=list(MODEL_DIRS),nargs='+',default=list(MODEL_DIRS))
    p.add_argument('--include-baselines',action='store_true')
    p.add_argument('--annotations',required=True)
    p.add_argument('--video-root',required=True)
    p.add_argument('--weights-root',default='weights')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if len(args.models)!=len(set(args.models)): p.error('Duplicate model names')
    out=Path(args.output_dir).resolve();weights=Path(args.weights_root).resolve()
    commands=[]
    for model in args.models:
        for method in (['main','baselines'] if args.include_baselines else ['main']):
            config=ROOT/'configs'/method/f'{args.benchmark}_{model}.json'
            name=f'{args.benchmark}_{model}_{method}'
            cmd=[sys.executable,'-m','memo.reproduce','--config',str(config),'--output',str(out/(name+'.json')),
                 '--annotations',str(Path(args.annotations).resolve()),'--video-root',str(Path(args.video_root).resolve()),
                 '--model-path',str(weights/MODEL_DIRS[model]),'--clip-path',str(weights/'clip-vit-large-patch14'),
                 '--gdino-path',str(weights/'grounding-dino-base'),'--sam2-ckpt',str(weights/'sam2.1_hiera_large.pt')]
            commands.append((name,cmd))
    for _,cmd in commands: print(shlex.join(cmd),flush=True)
    if not args.execute: return
    if out.exists(): raise FileExistsError('Choose a new output directory')
    out.mkdir(parents=True)
    for name,cmd in commands:
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'Completed {len(commands)} runs in {out}')

if __name__=='__main__': main()
