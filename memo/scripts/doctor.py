#!/usr/bin/env python3
"""Check inference dependencies and GPU visibility without loading model weights."""
import importlib
import sys

def main():
    failures=[]
    if sys.version_info[:2] != (3,10):
        print('NOTE: the documented reference environment uses Python 3.10.')
    for name in ('torch','torchvision','transformers','accelerate','qwen_vl_utils','numpy','PIL','cv2','decord','av','sam2'):
        try:
            module=importlib.import_module(name)
            print(f'OK {name}: {getattr(module,"__version__","importable")}')
        except Exception as exc:
            failures.append(name);print(f'FAIL {name}: {exc}')
    if 'torch' not in failures:
        import torch
        print(f'CUDA available: {torch.cuda.is_available()}, visible devices: {torch.cuda.device_count()}')
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1: failures.append('single CUDA device')
    return bool(failures)

if __name__=='__main__': sys.exit(main())
