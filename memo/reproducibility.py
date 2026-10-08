"""Collect provenance without storing credentials or the full environment."""
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import platform
from pathlib import Path
import sys

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()

def manifest(args, config, groups):
    root=Path(__file__).resolve().parents[1]
    packages={}
    for name in ('torch','torchvision','transformers','accelerate','qwen-vl-utils','numpy','Pillow','opencv-python','decord','av','hydra-core','iopath','SAM-2'):
        try: packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError: packages[name]=None
    sources={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*.py'))
             if not any(part in ('results','weights','data','.venv','__pycache__') for part in p.relative_to(root).parts)}
    models={}
    for key in ('model_path','clip_path','gdino_path'):
        value=getattr(args,key,None)
        models[key]={'location':value,'metadata_sha256':{}}
        if value and Path(value).is_dir():
            models[key]['metadata_sha256']={p.name:sha256(p) for p in sorted(Path(value).glob('*.json'))}
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
        platform=platform.platform(), packages=packages, source_sha256=sources,
        annotations_sha256=sha256(args.annotations), config_file_sha256=sha256(args.config),
        effective_config=config, models=models,
        selected_questions=[{'video':v,'ids':[q['id'] for q in qs]} for v,qs in groups.items()],
        videos=[{'path':v,'bytes':Path(v).stat().st_size} for v in groups],
        note='Model weight bytes and video bytes are not hashed automatically. Archive exact revisions separately.')
