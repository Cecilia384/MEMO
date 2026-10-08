"""Frame-range and temporal-sampling helpers for the fixed decord reader."""
import math
from typing import List, Tuple

import torch


def calculate_video_frame_range(
    total_frames: int,
    video_fps: float,
    video_start: float = None,
    video_end: float = None,
) -> Tuple[int, int, int]:
    """
    根据秒级起止时间换算出帧范围，返回 start_frame, end_frame, clipped_total_frames。
    这个函数的作用是先把 benchmark 层给出的“问题时刻之前的视频片段”换算成具体的原始帧区间。
    """
    if video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {video_fps}")
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")

    max_duration = total_frames / video_fps

    if video_start is None:
        start_frame = 0
    else:
        video_start = max(0.0, min(float(video_start), max_duration))
        start_frame = math.ceil(video_start * video_fps)

    if video_end is None:
        end_frame = total_frames - 1
    else:
        video_end = max(0.0, min(float(video_end), max_duration))
        end_frame = min(math.floor(video_end * video_fps), total_frames - 1)

    if start_frame > end_frame:
        raise ValueError(
            f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}, "
            f"video_start={video_start}, video_end={video_end}, total_frames={total_frames}, fps={video_fps}"
        )

    return start_frame, end_frame, end_frame - start_frame + 1


def build_sample_indices(
    total_frames: int,
    video_fps: float,
    target_fps: float,
    max_frames: int = None,
    anchor_end: bool = False,
) -> List[int]:
    """
    在给定裁剪后的视频片段里，按固定 fps 生成采样索引，返回的是“片段内部索引”，不是全视频绝对索引。
    如果 anchor_end=True，则优先让采样更贴近片段尾部，这对只看提问前最近窗口的场景很有帮助。
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")

    step = max(video_fps / target_fps, 1.0)
    nframes = int(math.floor(total_frames / step))
    nframes = max(1, nframes)

    if max_frames is not None:
        nframes = min(nframes, int(max_frames))

    if anchor_end:
        idx_desc = torch.round(torch.arange(0, nframes) * step).long()
        idx = (total_frames - 1) - idx_desc
        idx = torch.clamp(idx, min=0, max=total_frames - 1)
        idx = torch.flip(idx, dims=[0]).tolist()
    else:
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

    return idx


