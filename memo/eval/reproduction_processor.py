"""Causal video processing with independent historical/current frame budgets."""
from collections import deque
import time
import numpy as np
import torch
from PIL import Image
from transformers import LogitsProcessor
from memo.benchmarks.data import prompt
from memo.eval.frame_selection import dual_budget, uniform_sample
from memo.stage1.frame_extractor import FrameExtractor

class TokenClock(LogitsProcessor):
    def __init__(self): self.times=[]
    def __call__(self, input_ids, scores):
        torch.cuda.synchronize()
        self.times.append(time.perf_counter())
        return scores

class Processor:
    def __init__(self, config, adapter, perception, store, retriever, text_encoder, frame_store, backend):
        self.cfg=config
        self.adapter=adapter
        self.perception=perception
        self.store=store
        self.retriever=retriever
        self.text_encoder=text_encoder
        self.frame_store=frame_store
        self.backend=backend
        self.module_times={}
        if config['profile'] and perception and hasattr(perception,'detector'):
            # Profiling deliberately serializes module calls; use separate accuracy runs.
            perception._stream_clip=None
            perception._stream_dino=None
            for obj, name, label in [(perception.clip,'encode_frame_full','clip'),
                                      (perception.detector,'detect','grounding_dino'),
                                      (perception.tracker,'update','sam2')]:
                original=getattr(obj,name)
                def timed(*args, _fn=original, _label=label, **kwargs):
                    self.sync(); start=time.perf_counter()
                    result=_fn(*args,**kwargs)
                    self.sync()
                    self.module_times.setdefault(_label,[]).append((time.perf_counter()-start)*1000)
                    return result
                setattr(obj,name,timed)

    def sync(self):
        if self.cfg['profile']: torch.cuda.synchronize()

    def process(self, video_path, questions, record):
        from memo.stage2.stage2_similarity_gpu import SimilarityConfig
        from memo.stage2.stage3_segmentation_gpu import SegmentationConfig, StreamingSceneSegmenter
        c=self.cfg
        extractor=FrameExtractor(fps=c['target_fps'],backend='decord',anchor_end=False)
        video=extractor.extract(video_path)
        duration=extractor.get_video_info(video_path).duration
        if not video.frames: raise ValueError(f'No decoded frames: {video_path}')
        if any(q['timestamp'] is not None and q['timestamp']>duration+1e-6 for q in questions):
            raise ValueError(f'Question timestamp exceeds video duration: {video_path}')
        selected=range(len(video.frames))
        if self.store:
            self.store.clear(); self.frame_store.clear(); self.perception.reset_tracker()
            self.retriever._rng.seed(c['seed'])
        segmenter=None
        if c['variant']!='baseline':
            a,b,g=c['similarity_weights']
            sim=SimilarityConfig(alpha=a,beta=b,gamma=g,device='cuda:0',
                ema_momentum=c['ema_momentum'],max_hidden_frames=c['max_hidden_frames'],
                gamma_app=c['gamma_app'],gamma_dis=c['gamma_dis'],max_displacement=c['max_displacement'],
                spatial_alpha=c['spatial_alpha'],unmatched_spatial=c['unmatched_spatial'])
            segmenter=StreamingSceneSegmenter(self.perception,SegmentationConfig(similarity=sim,device='cuda:0',
                adaptive_window=c['adaptive_window'],initial_threshold=c['initial_threshold'],threshold_k=c['threshold_k'],
                min_chunk_frames=c['min_chunk_frames'],cut_confirmation_frames=c['cut_confirmation_frames']))
        pending={}
        # Baseline uniform sampling needs the entire observed prefix; chronology needs only a window.
        observed=[] if c['variant']=='baseline' else deque(maxlen=c['history_frames']+c['current_frames'])
        qpos=0
        stats={'sampled_frames':len(video.frames),'processed_frames':0,'chunks':0,'push_ms':[],'ingest_ms':[]}
        def ingest(chunk):
            if chunk is None: return
            self.sync(); start=time.perf_counter()
            if c['variant']=='memo':
                self.backend.ingest_chunk(chunk,similarity_calc=segmenter.similarity_calc,store=self.store,
                    feat_dim=self.perception.clip.dim)
                self.frame_store.add(chunk.chunk_id,[image for i,image in pending.items() if chunk.start_frame<=i<=chunk.end_frame])
            for i in list(pending):
                if i<=chunk.end_frame: del pending[i]
            chunk.release_feature_tensors()
            # Completed chunks have been materialized in the store; do not retain raw perception frames.
            segmenter._completed_chunks.clear()
            self.sync()
            if c['profile']: stats['ingest_ms'].append((time.perf_counter()-start)*1000)
            stats['chunks']+=1
        def answer_until(t, inclusive=True):
            nonlocal qpos
            while qpos<len(questions):
                q=questions[qpos]
                qt=float('inf') if q['timestamp'] is None else q['timestamp']
                if qt>t or (qt==t and not inclusive): break
                raw, evidence, metrics=self.answer(q,list(pending.values()),list(observed))
                record(q,raw,evidence,metrics)
                qpos+=1
        for index in selected:
            frame=video.frames[index]
            answer_until(frame.timestamp,inclusive=False)
            if qpos==len(questions): break
            pil=Image.fromarray(frame.image.astype(np.uint8))
            observed.append(pil)
            if segmenter is not None:
                native=video.native_indices[index]
                pending[native]=pil
                self.sync(); start=time.perf_counter()
                chunk=segmenter.push_frame(frame.image,native)
                self.sync()
                if c['profile']: stats['push_ms'].append((time.perf_counter()-start)*1000)
                ingest(chunk)
            stats['processed_frames']+=1
            answer_until(frame.timestamp)
            if qpos==len(questions): break
        # Finite times after the last sampled frame still see the active chunk.
        answer_until(float('inf'),inclusive=False)
        if qpos<len(questions):
            if segmenter: ingest(segmenter.flush())
            answer_until(float('inf'))
        return stats

    def answer(self,q,current,observed):
        c=self.cfg
        self.sync(); start=time.perf_counter()
        hits=[]; history=[]
        if c['variant']=='baseline':
            if c['baseline_sampling']=='all': frames=observed
            elif c['baseline_sampling']=='recent': frames=observed[-c['baseline_frames']:]
            else: frames=uniform_sample(observed,c['baseline_frames'])
        else:
            if self.store.gpu_index.size:
                vec=self.text_encoder.encode_text(q['question'])
                hits=self.retriever.query_vec(vec,top_k=c['top_k'],recall_tokens=False)
                for hit in hits: history.extend(self.frame_store.get(hit.chunk_id))
            frames=dual_budget(history,current,c['history_frames'],c['current_frames'])
        if not frames: raise ValueError(f"{q['id']}: No causal evidence frames available")
        self.sync(); retrieval_end=time.perf_counter()
        inputs=self.adapter.build_inputs(frames,prompt(q),'cuda:0')
        self.sync(); prepared=time.perf_counter()
        clock=TokenClock()
        kwargs={'logits_processor':[clock]} if c['profile'] else {}
        with torch.inference_mode():
            generated=self.adapter.model.generate(**inputs,max_new_tokens=c['max_new_tokens'],do_sample=False,use_cache=True,**kwargs)
        self.sync(); generated_at=time.perf_counter()
        raw=self.adapter.decode_output(generated,inputs.input_ids)
        end=time.perf_counter()
        evidence=dict(total_frames=len(frames),retrieved_chunks=[h.chunk_id for h in hits],
            history_available=len(history),current_available=len(current))
        metrics={}
        if c['profile']:
            metrics=dict(retrieval_ms=(retrieval_end-start)*1000,preprocessing_ms=(prepared-retrieval_end)*1000,
                generation_ms=(generated_at-prepared)*1000,response_ms=(end-start)*1000,
                generated_tokens=int(generated.shape[-1]-inputs.input_ids.shape[-1]),
                ttft_ms=(clock.times[0]-prepared)*1000 if clock.times else None,
                tpot_ms=(clock.times[-1]-clock.times[0])*1000/(len(clock.times)-1) if len(clock.times)>1 else None)
        return raw,evidence,metrics
