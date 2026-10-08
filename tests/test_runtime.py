"""CPU runtime regression checks; skipped by lightweight CI without PyTorch."""
import importlib.util
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

@skipUnless(all(importlib.util.find_spec(name) is not None for name in ('torch','transformers','sam2')),'Install inference dependencies for runtime checks')
class RuntimeTests(TestCase):
    def test_between_frame_questions_never_see_the_next_frame(self):
        import numpy as np
        import torch
        from transformers import BatchEncoding
        from configuration import DEFAULTS
        from eval.reproduction_processor import Processor
        seen=[]
        class Adapter:
            model=SimpleNamespace(generate=lambda **kwargs:torch.tensor([[0,1]]))
            def build_inputs(self,frames,text,device):
                seen.append([int(np.array(f)[0,0,0]) for f in frames])
                return BatchEncoding({'input_ids':torch.tensor([[0]])})
            def decode_output(self,*args): return 'A'
        video=SimpleNamespace(frames=[SimpleNamespace(timestamp=t,image=np.full((4,4,3),color,dtype=np.uint8)) for t,color in [(0,10),(1,20),(2,30)]],native_indices=[0,1,2])
        extractor=SimpleNamespace(extract=lambda _:video,get_video_info=lambda _:SimpleNamespace(duration=3))
        config=dict(DEFAULTS,variant='baseline',baseline_sampling='all')
        processor=Processor(config,Adapter(),None,None,None,None,None,None)
        questions=[dict(id=str(i),timestamp=t,question='Color?',options=['A','B'],answer='A',benchmark='custom') for i,t in enumerate([.5,1,None])]
        records=[]
        with patch('eval.reproduction_processor.FrameExtractor',return_value=extractor):
            processor.process('fake.mp4',questions,lambda *args:records.append(args))
        self.assertEqual(seen,[[10],[10,20],[10,20,30]])

        self.assertEqual(len(records),3)

    def test_memory_preserves_transition_and_does_not_duplicate_final_frames(self):
        import numpy as np
        import torch
        from transformers import BatchEncoding
        from configuration import DEFAULTS
        from eval.reproduction_processor import Processor
        from eval.shared_components import ChunkFrameStore
        seen=[]
        class Adapter:
            model=SimpleNamespace(generate=lambda **kwargs:torch.tensor([[0,1]]))
            def build_inputs(self,frames,text,device):
                seen.append([int(np.array(f)[0,0,0]) for f in frames])
                return BatchEncoding({'input_ids':torch.tensor([[0]])})
            def decode_output(self,*args): return 'A'
        class Segmenter:
            def __init__(self,*args,**kwargs):
                self.similarity_calc=None;self._completed_chunks=[]
            def chunk(self,number,start,end):
                return SimpleNamespace(chunk_id=number,start_frame=start,end_frame=end,release_feature_tensors=lambda:None)
            def push_frame(self,frame,index):
                return self.chunk(0,0,0) if index==1 else None
            def flush(self): return self.chunk(1,1,2)
        index=SimpleNamespace(size=0)
        hits=[]
        def clear(): index.size=0;hits.clear()
        store=SimpleNamespace(gpu_index=index,clear=clear)
        def ingest(chunk,**kwargs):
            hits.append(SimpleNamespace(chunk_id=chunk.chunk_id));index.size+=1
        retriever=SimpleNamespace(_rng=SimpleNamespace(seed=lambda _:None),query_vec=lambda *a,**k:list(hits))
        perception=SimpleNamespace(clip=SimpleNamespace(dim=4),reset_tracker=lambda:None)
        text=SimpleNamespace(encode_text=lambda _:torch.zeros(4))
        video=SimpleNamespace(frames=[SimpleNamespace(timestamp=t,image=np.full((4,4,3),color,dtype=np.uint8)) for t,color in [(0,10),(1,20),(2,30)]],native_indices=[0,1,2])
        extractor=SimpleNamespace(extract=lambda _:video,get_video_info=lambda _:SimpleNamespace(duration=3))
        processor=Processor(dict(DEFAULTS),Adapter(),perception,store,retriever,text,ChunkFrameStore(),SimpleNamespace(ingest_chunk=ingest))
        qs=[dict(id=str(i),timestamp=t,question='Color?',options=['A','B'],answer='A',benchmark='custom') for i,t in enumerate([.5,1.5,None])]
        with patch('eval.reproduction_processor.FrameExtractor',return_value=extractor), patch('stage2.stage3_segmentation_gpu.StreamingSceneSegmenter',Segmenter):
            processor.process('fake.mp4',qs,lambda *args:None)
        self.assertEqual(seen,[[10],[10,20],[10,20,30]])


@skipUnless(all(importlib.util.find_spec(name) is not None for name in ('torch','cv2','decord')),
            'Install video decoding dependencies for the sample check')
class VideoSamplingTests(TestCase):
    def test_decord_sample_indices_match_reference_video(self):
        from pathlib import Path
        from stage1.frame_extractor import FrameExtractor
        video_path = Path(__file__).resolve().parents[1] / 'examples/synthetic/sample_0_real.mp4'
        result = FrameExtractor(fps=1.0, backend='decord').extract(str(video_path))
        self.assertEqual(result.native_indices, [0, 9, 18, 27, 36, 45, 54, 63])
        self.assertEqual(len(result.frames), 8)

