"""Strict JSON experiment configuration (standard library only)."""
import argparse
import json
import math
from pathlib import Path

DEFAULTS = dict(benchmark='custom', model='qwen3', variant='memo', target_fps=1.0,
    top_k=3, history_frames=8, current_frames=8, max_new_tokens=64, seed=42,
    scoring='strict', cpu_max_size=512, max_store_frames=1000000,
    baseline_frames=16, baseline_sampling='uniform', detection_prompt='person . car . animal . object .',
    profile=False,
    similarity_weights=[0.35, 0.45, 0.20], retrieval_weights=[0.6, 0.4],
    adaptive_window=30, initial_threshold=0.45, threshold_k=1.3,
    min_chunk_frames=4, cut_confirmation_frames=1, ema_momentum=0.3,
    max_hidden_frames=5, box_threshold=0.35, text_threshold=0.25,
    gamma_app=0.4, gamma_dis=0.2, max_displacement=0.5,
    spatial_alpha=0.6, unmatched_spatial=0.0)
CHOICES = dict(benchmark=['custom','streamingbench','ovobench'],
    model=['qwen3','qwen25','llava_7b','llava_05b'],
    variant=['memo','baseline'],
    scoring=['strict','legacy'], baseline_sampling=['uniform','recent','all'])
POSITIVE = ('target_fps','top_k','history_frames','current_frames','max_new_tokens',
    'cpu_max_size','max_store_frames','baseline_frames','adaptive_window',
    'min_chunk_frames','cut_confirmation_frames','max_hidden_frames','max_displacement')

def load_config(path):
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError('Configuration must be an object with known keys only')
    config = {**DEFAULTS, **raw}
    for key, default in DEFAULTS.items():
        value = config[key]
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, int):
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(default, float):
            valid = isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value)
        else:
            valid = isinstance(value, type(default))
        if not valid:
            raise ValueError(f'Invalid type or nonfinite value for {key}')
    for key, options in CHOICES.items():
        if config[key] not in options:
            raise ValueError(f'{key} must be one of {options}')
    for key in POSITIVE:
        if config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    for key in ('initial_threshold','ema_momentum','box_threshold','text_threshold','spatial_alpha','unmatched_spatial'):
        if not 0 <= config[key] <= 1:
            raise ValueError(f'{key} must be in [0,1]')
    for key in ('threshold_k','gamma_app','gamma_dis'):
        if config[key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    for key, length in [('similarity_weights',3),('retrieval_weights',2)]:
        value = config[key]
        if len(value) != length or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in value) or abs(sum(value)-1)>1e-6:
            raise ValueError(f'{key} must contain {length} nonnegative weights summing to 1')
    return config

def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Run MEMO manuscript protocols or a custom dataset.')
    p.add_argument('--config', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--video-root', required=True)
    p.add_argument('--output', default='results/predictions.json')
    p.add_argument('--model-path')
    p.add_argument('--clip-path', default='weights/clip-vit-large-patch14')
    p.add_argument('--gdino-path', default='weights/grounding-dino-base')
    p.add_argument('--sam2-ckpt', default='weights/sam2.1_hiera_large.pt')
    p.add_argument('--sam2-cfg', default='configs/sam2.1/sam2.1_hiera_l.yaml')
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--max-videos',type=int,default=0)
    p.add_argument('--max-questions-per-video',type=int,default=0)
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args(argv)
    if args.max_videos<0 or args.max_questions_per_video<0:
        p.error('Subset limits must be nonnegative')
    return args
