# MEMO/stage1/frame_extractor.py

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from stage1.data_types import Frame, VideoFrames, VideoInfo
from stage1.sampling import build_sample_indices, calculate_video_frame_range

logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    使用 decord 提取 [start_sec, end_sec] 的视频内容。先计算采样索引，再只解码选中的帧。
    """

    def __init__(
        self,
        fps: float = 1.0,
        backend: str = "decord",
        max_frames: Optional[int] = None,
        anchor_end: bool = False,
    ):
        self.fps = float(fps)
        self.backend = backend
        self.max_frames = max_frames
        self.anchor_end = anchor_end

    def get_video_info(self, video_path: str) -> VideoInfo:
        """
        用 cv2 读基础元信息：fps、宽高、总帧数、时长；只读 metadata，不抽帧。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        return VideoInfo(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration=duration,
        )
    
    def extract(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: float = None,
    ) -> VideoFrames:
        """
        主入口：读取 [start_sec, end_sec] 范围内的视频，按 self.fps 抽样后返回 VideoFrames。

        视频无法解码时抛出 CorruptedVideoError，由实验入口记录失败状态。
        """
        video_path = str(Path(video_path))
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        backend = self.backend
        if backend != "decord":
            raise ValueError(f"Unsupported backend: {backend}")

        logger.info(
            f"Extracting frames from {video_path}, backend={backend}, target_fps={self.fps}, "
            f"start_sec={start_sec}, end_sec={end_sec}, max_frames={self.max_frames}, anchor_end={self.anchor_end}"
        )

        # ---------------------------------------------------------------------
        # 快路径 1：decord（先算 idx，再只解码 idx 对应帧）
        # ---------------------------------------------------------------------
        if backend == "decord":
            try:
                import decord
                from decord import VideoReader, cpu
                decord.bridge.set_bridge("torch")
            except ImportError as e:
                raise ImportError("decord is not installed, please install it first.") from e

            cap_fallback = None
            try:
                try:
                    vr = VideoReader(video_path, ctx=cpu(0), num_threads=0)
                    native_fps = float(vr.get_avg_fps())
                    total_frames = len(vr)
                except Exception as e:
                    msg = (
                        f"[skip current video] decord 打开视频失败 | "
                        f"video={video_path} | reason={repr(e)}"
                    )
                    logger.error(msg)
                    raise CorruptedVideoError(msg) from e

                start_frame, end_frame, _ = calculate_video_frame_range(
                    total_frames=total_frames,
                    video_fps=native_fps,
                    video_start=start_sec,
                    video_end=end_sec,
                )
                seg_total = end_frame - start_frame + 1

                seg_idx = build_sample_indices(
                    total_frames=seg_total,
                    video_fps=native_fps,
                    target_fps=self.fps,
                    max_frames=self.max_frames,
                    anchor_end=self.anchor_end,
                )
                idx_abs = [start_frame + i for i in seg_idx]

                seg_duration = seg_total / max(native_fps, 1e-6)
                sample_fps = len(idx_abs) / max(seg_duration, 1e-6)

                chunk_size = 64
                frames: List[Frame] = []
                timestamps: List[float] = []
                native_indices: List[int] = []

                for chunk_start in range(0, len(idx_abs), chunk_size):
                    chunk_idx = idx_abs[chunk_start: chunk_start + chunk_size]

                    try:
                        video_t_chunk = vr.get_batch(chunk_idx)

                    except Exception as decord_err:
                        logger.warning(
                            f"[decord] get_batch failed, fallback to cv2 | "
                            f"video={video_path} | "
                            f"chunk_range=[{chunk_idx[0]}-{chunk_idx[-1]}] | "
                            f"reason={repr(decord_err)}"
                        )

                        if cap_fallback is None:
                            cap_fallback = cv2.VideoCapture(video_path)
                            if not cap_fallback.isOpened():
                                msg = (
                                    f"[skip current video] OpenCV 无法打开视频 | "
                                    f"video={video_path} | "
                                    f"decord_error={repr(decord_err)}"
                                )
                                logger.error(msg)
                                raise CorruptedVideoError(msg) from decord_err

                        frames_cv2 = []
                        for abs_idx in chunk_idx:
                            cap_fallback.set(cv2.CAP_PROP_POS_FRAMES, abs_idx)
                            ret, bgr = cap_fallback.read()

                            if not ret or bgr is None:
                                msg = (
                                    f"[skip current video] 视频损坏，无法解码帧 | "
                                    f"video={video_path} | failed_frame={abs_idx} | "
                                    f"decord_error={repr(decord_err)}"
                                )
                                logger.error(msg)
                                raise CorruptedVideoError(msg) from decord_err

                            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            frames_cv2.append(torch.from_numpy(rgb))

                        video_t_chunk = torch.stack(frames_cv2)

                    for i, frame_tensor in enumerate(video_t_chunk):
                        img_np = frame_tensor.cpu().numpy()

                        if img_np.dtype != np.uint8:
                            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

                        global_idx = chunk_start + i
                        ts = float(idx_abs[global_idx]) / float(native_fps)

                        frames.append(Frame(index=global_idx, timestamp=ts, image=img_np))
                        timestamps.append(ts)
                        native_indices.append(int(idx_abs[global_idx]))

                return VideoFrames(
                    video_path=video_path,
                    original_fps=float(native_fps),
                    sample_fps=float(sample_fps),
                    start_sec=float(start_sec),
                    end_sec=float(
                        end_sec if end_sec is not None else (timestamps[-1] if timestamps else start_sec)
                    ),
                    frames=frames,
                    timestamps=timestamps,
                    native_indices=native_indices,
                    meta={
                        "backend": "decord(sampled)",
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "seg_total_frames": int(seg_total),
                        "sampled_frames": len(frames),
                        "idx_abs_head": native_indices[:10],
                    },
                )

            finally:
                if cap_fallback is not None:
                    cap_fallback.release()

class CorruptedVideoError(RuntimeError):
    """The video cannot be decoded by decord or the OpenCV fallback."""
    pass
