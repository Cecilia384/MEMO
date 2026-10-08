#!/usr/bin/env python3
"""Run an explicit MEMO experiment configuration."""
import json
import logging
import random
from pathlib import Path
from configuration import load_config, parse_args
from benchmarks.data import load_questions
from benchmarks.scoring import score_answer, summarize
from reproducibility import manifest
from eval.output import save_output

def main():
    args=parse_args()
    config=load_config(args.config)
    groups=load_questions(args.annotations,args.video_root,config['benchmark'],args.tasks,
                          args.max_videos,args.max_questions_per_video)
    count=sum(map(len,groups.values()))
    print(f'Validated {len(groups)} videos and {count} questions.',flush=True)
    if args.validate_only:
        print(json.dumps(config,indent=2)); return
    if not args.model_path: raise ValueError('--model-path is required for inference')
    output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError(f'Choose a new output path: {output}')
    payload=dict(status='initializing',config=config,arguments=vars(args),expected_questions=count,
        subset_run=bool(args.tasks or args.max_videos or args.max_questions_per_video),
        reproducibility=manifest(args,config,groups),results=[],videos=[])
    save_output(output,payload)
    try:
        import numpy as np
        import torch
        from eval.shared_components import StageBackend, RetrievalCLIPEncoder, ChunkFrameStore, build_model_adapter
        from eval.reproduction_processor import Processor
        if not torch.cuda.is_available(): raise RuntimeError('A CUDA GPU is required')
        if torch.cuda.device_count()!=1:
            raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES for reproducible device placement')
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
        random.seed(config['seed']); np.random.seed(config['seed'])
        torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
        payload['reproducibility']['gpu']=dict(name=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
            total_bytes=torch.cuda.get_device_properties(0).total_memory)
        adapter=build_model_adapter(config['model'],args.model_path,'cuda:0',qwen3_mode='image')
        backend=StageBackend(use_gpu=True)
        perception=store=retriever=text_encoder=frame_store=None
        if config['variant']!='baseline':
            clip=backend.CLIPEncoder(model_name=args.clip_path,device='cuda:0')
            detector=backend.GroundingDINODetector(weights_path=args.gdino_path,device='cuda:0',
                box_threshold=config['box_threshold'],text_threshold=config['text_threshold'])
            tracker=backend.SAM2Tracker(checkpoint=args.sam2_ckpt,model_cfg=args.sam2_cfg,device='cuda:0')
            perception=backend.PerceptionLayer(clip,detector,tracker,text_prompt=config['detection_prompt'])
            store,_,_=backend.build_chunk_store(feat_dim=clip.dim,device='cuda:0',cpu_max_size=config['cpu_max_size'])
            g,l=config['retrieval_weights']
            retriever=backend.ChunkRetriever(store=store,device='cuda:0',lambda_global=g,lambda_local=l,
                                             strategy='ours',random_seed=config['seed'])
            text_encoder=RetrievalCLIPEncoder(args.clip_path,device='cuda:0')
            frame_store=ChunkFrameStore(max_total_frames=config['max_store_frames'])
        processor=Processor(config,adapter,perception,store,retriever,text_encoder,frame_store,backend)
        payload['status']='running'
        torch.cuda.reset_peak_memory_stats()
        def record(q,raw,evidence,metrics):
            choice,correct=score_answer(q,raw,config['scoring'])
            payload['results'].append(dict(q,prediction=raw,choice=choice,correct=correct,evidence=evidence,metrics=metrics))
            payload['summary']=summarize(payload['results'],config['benchmark'])
            save_output(output,payload)
        for video,questions in groups.items():
            logging.info('Processing %s',video)
            payload['videos'].append(dict(video=video,**processor.process(video,questions,record)))
        if len(payload['results'])!=count: raise RuntimeError('Some questions were not answered')
        payload['peak_allocated_gib']=torch.cuda.max_memory_allocated()/(1024**3)
        payload['module_profile_ms']=processor.module_times
        payload['status']='completed'
    except BaseException as exc:
        payload['status']='failed';payload['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save_output(output,payload)
    print(json.dumps(payload['summary'],indent=2))
    print(f'Results saved to {output}')

if __name__=='__main__': main()
