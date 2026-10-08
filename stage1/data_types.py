# pipeline/data_types.py

from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np
import torch


@dataclass
class Frame:
    """
    单个抽样后的视频帧，image 统一使用 RGB 格式的 numpy.ndarray，
    形状为 (H, W, C)，dtype 为 uint8，这样便于后续做可视化、OpenCV/PIL 转换和特征提取。
    """
    index: int
    timestamp: float
    image: np.ndarray  # RGB, (H, W, C), uint8

    def __repr__(self):
        return f"Frame(index={self.index}, timestamp={self.timestamp:.2f}, shape={self.image.shape})"


@dataclass
class VideoInfo:
    """
    视频元信息，主要由读取器阶段产生，用于后续日志、抽帧、裁剪和调试。
    """
    video_path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration: float


@dataclass
class VideoFrames:
    """
    原始抽帧结果，frames 是按时间顺序排列的 Frame 列表；timestamps / indices 冗余保存，
    是为了后续过滤、统计和对接不同模型时更方便，不需要每次都再从 Frame 对象里提取。
    """
    video_path: str
    original_fps: float
    sample_fps: float
    start_sec: float
    end_sec: float
    frames: List[Frame]
    timestamps: List[float]
    native_indices: List[int]
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self):
        return len(self.frames)
