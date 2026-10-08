#!/usr/bin/env python3
"""Generate an original synthetic video and all three annotation formats."""
import argparse
import csv
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='examples/synthetic')
    args=p.parse_args()
    import cv2
    import numpy as np
    root=Path(args.output_dir)
    root.mkdir(parents=True,exist_ok=True)
    video=root/'sample_0_real.mp4'
    if video.exists(): raise FileExistsError(video)
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),8,(320,240))
    if not writer.isOpened(): raise RuntimeError('MP4 writer could not be opened')
    try:
        for i in range(64):
            frame=np.full((240,320,3),255,dtype=np.uint8)
            color=(0,0,255) if i<32 else (255,0,0)
            cv2.rectangle(frame,(60+i%24,60),(200+i%24,200),color,-1)
            writer.write(frame)
    finally: writer.release()
    capture=cv2.VideoCapture(str(video));ok,_=capture.read();capture.release()
    if not ok: raise RuntimeError('Generated video is not decodable')
    custom=[dict(id='early',video=video.name,timestamp=0.5,question='What color is the square?',options=['Red','Blue'],answer='A'),
            dict(id='late',video=video.name,timestamp=4.5,question='What color is the square now?',options=['Red','Blue'],answer='B'),
            dict(id='full',video=video.name,timestamp=None,question='Describe how the square changes color.')]
    (root/'custom.json').write_text(json.dumps(custom,indent=2)+'\n')
    ovo=[dict(id=q['id'],task='ATR',video=video.name,realtime=q['timestamp'],question=q['question'],options=q['options'],gt=i)
         for i,q in enumerate(custom[:2])]
    (root/'ovobench.json').write_text(json.dumps(ovo,indent=2)+'\n')
    with (root/'streamingbench.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['question_id','task_type','question','time_stamp','answer','options'])
        writer.writeheader()
        for i,q in enumerate(custom[:2]):
            writer.writerow(dict(question_id=f'example_sample_0_q{i}',task_type='Attribute Perception',question=q['question'],time_stamp=q['timestamp'],answer=q['answer'],options=repr(['A. Red','B. Blue'])))
    (root/'expected.json').write_text(json.dumps(dict(custom_questions=3,streamingbench_questions=2,ovobench_questions=2,
        purpose='Execution and schema checks only; not a benchmark score or an exact model-output fixture.'),indent=2)+'\n')
    print(root)

if __name__=='__main__': main()
