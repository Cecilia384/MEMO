# MEMO/stage2/stage3_segmentation.py
"""
第三阶段：动态场景切分与聚类 (Scene Segmentation)  [GPU 加速版]

维护"活跃场景缓存" (Active Scene Buffer) 与自适应阈值 τ，
通过流式方式完成语义片段 (Semantic Chunk) 的切分决策。

主要改动（numpy → torch GPU）：
  - AdaptiveThreshold：滑动窗口历史分数存储为 GPU tensor；
    mean/std 使用 torch 运算，避免每次 np.array() 转换开销。
  - SegmentationConfig 新增 device 字段，透传给 SimilarityConfig。
  - 其余切分逻辑（帧计数、防抖判断）仍为纯 Python 标量，无需搬到 GPU。
"""

from __future__ import annotations
import numpy as np
import torch
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from memo.stage2.stage1_perception_gpu import FramePerception, PerceptionLayer
from memo.stage2.stage2_similarity_gpu import SimilarityCalculator, SimilarityResult, SimilarityConfig


def sync_time() -> float:
    """获取当前时间，如果是GPU运算则先同步，确保计时准确"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SemanticChunk:
    """一个完整的语义片段"""
    chunk_id: int
    start_frame: int
    end_frame: int
    frames: list[FramePerception]
    similarity_scores: list[SimilarityResult]   # 片段内相似度序列
    entity_buffer: dict = field(default_factory=dict)  # object_buffer 快照 {track_id: ObjectBufferState}

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def release_masks(self):
        """
        释放帧内 TrackedObject 的 GPU mask tensor，防止长视频 OOM。
        mask 在 stage2 相似度计算完成后不再需要，但 global_feature / patch_tokens /
        local_feature 仍供 stage3 ingestion 使用，不释放。
        """
        for f in self.frames:
            for obj in f.tracked_objects:
                obj.mask = None

    def release_feature_tensors(self):
        """在 chunk ingest 完成后释放帧级特征，避免 _completed_chunks 长期持有大 tensor。"""
        for f in self.frames:
            f.global_feature = None
            f.patch_tokens = None
            for obj in f.tracked_objects:
                obj.local_feature = None
                obj.mask = None

    def __repr__(self) -> str:
        return (f"SemanticChunk(id={self.chunk_id}, "
                f"frames=[{self.start_frame}, {self.end_frame}], "
                f"len={self.duration_frames})")


@dataclass
class SegmentationConfig:
    # 相似度计算配置
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    # 自适应阈值：滑动窗口大小
    adaptive_window: int = 30 #25 10
    # 初始固定阈值（窗口不足时使用）
    initial_threshold: float = 0.45 # 默认 0.45 0.55
    
    # 连续低相似度帧数阈值（防抖：避免单帧噪声误切）
    cut_confirmation_frames: int = 1

    # 阈值低点敏感度倍数（τ = mean - k * std）
    threshold_k: float = 1.3 # 默认 1.3，降低使切分更灵敏0.8


    # 最小语义片段帧数（防止碎片化切分）
    min_chunk_frames: int = 4   # 默认 8
    # # 自适应阈值：滑动窗口大小
    # adaptive_window: int = 10 #25

    # # 初始固定阈值（窗口不足时使用）
    # initial_threshold: float = 0.55 # 默认 0.45

    # # 连续低相似度帧数阈值（防抖：避免单帧噪声误切）
    # cut_confirmation_frames: int = 1

    # # 阈值低点敏感度倍数（τ = mean - k * std）
    # threshold_k: float = 0.8 # 默认 1.3，降低使切分更灵敏


    # 最小语义片段帧数（防止碎片化切分）
    min_chunk_frames: int = 4   # 默认 8

    # 运行设备（透传给 SimilarityConfig）
    device: str = "cuda"

    def __post_init__(self):
        # 保证 similarity.device 与顶层 device 一致
        self.similarity.device = self.device


# ---------------------------------------------------------------------------
# 自适应阈值管理器（GPU 加速版）
# ---------------------------------------------------------------------------

import math

class AdaptiveThreshold:
    # ... (保持 init 签名不变) ...
    def __init__(self, window_size: int = 30, initial_tau: float = 0.45, k: float = 1.2, device: str = "cuda"):
        self.window_size = window_size
        self.initial_tau = initial_tau
        self.k = k
        # 去掉 GPU tensor，改用原生 Python 列表或 Ring Buffer
        self._buf = [0.0] * window_size
        self._ptr = 0
        self._count = 0
        
        # O(1) 维护变量
        self._sum = 0.0
        self._sq_sum = 0.0

    def push(self, score: float):
        # 确保 score 是标量 (如果在上游已经是 float，直接使用)
        old_val = self._buf[self._ptr]
        
        self._buf[self._ptr] = score
        
        if self._count < self.window_size:
            self._count += 1
            self._sum += score
            self._sq_sum += score * score
        else:
            # 窗口已满，减去旧值，加上新值
            self._sum = self._sum - old_val + score
            self._sq_sum = self._sq_sum - (old_val * old_val) + (score * score)
            
        self._ptr = (self._ptr + 1) % self.window_size

    @property
    def tau(self) -> float:
        min_fill = max(5, self.window_size // 3)
        if self._count < min_fill:
            return self.initial_tau

        mean = self._sum / self._count
        # 计算方差，注意浮点精度问题导致的负数
        variance = max(0.0, (self._sq_sum / self._count) - (mean * mean))
        
        # 如果 unbiased=False，则总体标准差直接开根号即可
        std = math.sqrt(variance)
        
        tau = mean - self.k * std
        return max(0.1, min(0.9, tau)) # 等价于 torch.clamp
    
    def reset_history(self):
        """场景切换后选择不重置（保持跨场景平滑基准）"""
        pass



# ---------------------------------------------------------------------------
# 第三阶段：流式分割器
# ---------------------------------------------------------------------------

class StreamingSceneSegmenter:
    """
    流式视频语义场景切分器（完整三阶段 Pipeline）。

    使用方式（逐帧推送）:
        segmenter = StreamingSceneSegmenter(...)
        for frame in video_stream:
            chunk = segmenter.push_frame(frame, frame_idx)
            if chunk:
                process(chunk)          # 收到完整片段
        final_chunk = segmenter.flush() # 处理最后一个片段
    """

    def __init__(self,
                 perception: PerceptionLayer,
                 config: Optional[SegmentationConfig] = None):
        self.perception = perception
        self.cfg = config or SegmentationConfig()

        self.similarity_calc = SimilarityCalculator(self.cfg.similarity)
        self.adaptive_tau    = AdaptiveThreshold(
            window_size  = self.cfg.adaptive_window,
            initial_tau  = self.cfg.initial_threshold,
            k            = self.cfg.threshold_k,
            device       = self.cfg.device,
        )

        # 活跃场景缓存
        self._active_frames: list[FramePerception] = []
        self._active_scores: list[SimilarityResult] = []
        self._cut_candidate_count: int = 0   # 连续低于阈值的帧数（防抖）

        self._chunk_counter: int = 0
        self._completed_chunks: list[SemanticChunk] = []
        
        # 新增：记录“当前帧原始相似度”和“该帧对应的 tau”
        self.latest_raw_similarity: Optional[SimilarityResult] = None
        self.latest_raw_tau: Optional[float] = None

        
        self.stats = {
            "perception_time": 0.0,
            "similarity_time": 0.0,
            "decision_time": 0.0,
            "processed_frames": 0
        }
    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def push_frame(self, frame_rgb: np.ndarray,
               frame_idx: int) -> Optional[SemanticChunk]:

        self.stats["processed_frames"] += 1

        # 每帧开始先清空
        self.latest_raw_similarity = None
        self.latest_raw_tau = None

        perception = self.perception.process_frame(frame_rgb, frame_idx)
        sim_result = self.similarity_calc.update(perception)

        if len(self._active_frames) >= 2:
            old_frame = self._active_frames[-2]
            for obj in old_frame.tracked_objects:
                obj.mask = None

        self._active_frames.append(perception)
        completed_chunk = None

        if sim_result is not None:
            # 先保存“原始值”
            self.latest_raw_similarity = sim_result

            self._active_scores.append(sim_result)
            self.adaptive_tau.push(sim_result.s_total)

            # 保存“该帧真正用于判断 cut 的 tau”
            self.latest_raw_tau = self.adaptive_tau.tau

            completed_chunk = self._decide_cut(sim_result)

        return completed_chunk
    
    def flush(self) -> Optional[SemanticChunk]:
        """
        处理视频结束，强制输出最后一个语义片段（即使未触发切分）。
        """
        if not self._active_frames:
            return None
        return self._seal_chunk()

    def get_all_chunks(self) -> list[SemanticChunk]:
        return list(self._completed_chunks)

    # ------------------------------------------------------------------
    # 切分决策逻辑（纯 Python 标量，无需 GPU）
    # ------------------------------------------------------------------

    def _decide_cut(self, sim: SimilarityResult) -> Optional[SemanticChunk]:
        tau    = self.adaptive_tau.tau         # 已是 Python float
        is_low = sim.s_total < tau

        if is_low:
            self._cut_candidate_count += 1
        else:
            self._cut_candidate_count = 0

        trigger   = (self._cut_candidate_count >= self.cfg.cut_confirmation_frames)
        # 计算去除过渡帧后的实际 chunk 长度
        n_carry   = self._cut_candidate_count if trigger else 0
        too_short = (len(self._active_frames) - n_carry < self.cfg.min_chunk_frames)

        if trigger and not too_short:
            self._log_cut(sim, tau)

            # ── 边界修正：最后 _cut_candidate_count 帧属于新场景 ──
            # 这些帧的相似度低是因为它们与旧场景不同，应归入下一个 chunk
            carry_over_frames = self._active_frames[-n_carry:]
            self._active_frames = self._active_frames[:-n_carry]
            self._active_scores = self._active_scores[:-n_carry]

            chunk = self._seal_chunk()
            self._reset_after_cut()

            # 将过渡帧作为新场景的起始帧重新注入
            for fp in carry_over_frames:
                self._active_frames.append(fp)
                sim_result = self.similarity_calc.update(fp)
                if sim_result is not None:
                    self._active_scores.append(sim_result)
                    self.adaptive_tau.push(sim_result.s_total)

            return chunk

        return None

    def _seal_chunk(self) -> Optional[SemanticChunk]:
        """
        封装当前活跃缓存为一个 SemanticChunk。
        自动释放已完成 chunk 内的 mask tensor 以节省 GPU 显存。
        """
        if not self._active_frames:
            return None

        start_idx = self._active_frames[0].frame_idx
        end_idx   = self._active_frames[-1].frame_idx

        chunk = SemanticChunk(
            chunk_id          = self._chunk_counter,
            start_frame       = start_idx,
            end_frame         = end_idx,
            frames            = list(self._active_frames),
            similarity_scores = list(self._active_scores),
            entity_buffer     = dict(self.similarity_calc.object_buffer),
        )
        # 释放 chunk 内所有帧的 mask tensor（stage3 不需要 mask）
        chunk.release_masks()

        self._chunk_counter += 1
        self._completed_chunks.append(chunk)
        return chunk

    def _reset_after_cut(self):
        """场景切换后清空活跃缓存，重置各子模块"""
        self._active_frames.clear()
        self._active_scores.clear()
        self._cut_candidate_count = 0

        self.perception.reset_tracker()
        self.similarity_calc.reset()
        self.adaptive_tau.reset_history()

    def _log_cut(self, sim: SimilarityResult, tau: float):
        print(
            f"[CUT] frame={sim.frame_idx:5d} | "
            f"S_total={sim.s_total:.3f} < τ={tau:.3f} | "
            f"S_global={sim.s_global:.3f} S_spatial={sim.s_spatial:.3f} "
            f"S_local={sim.s_local:.3f} P_penalty={sim.p_penalty:.3f} | "
            f"appeared={sim.n_appeared} disappeared={sim.n_disappeared}"
        )


# ---------------------------------------------------------------------------
# 便捷：批量处理帧列表
# ---------------------------------------------------------------------------

def segment_video_frames(
    frames: list[np.ndarray],
    perception: PerceptionLayer,
    config: Optional[SegmentationConfig] = None,
) -> list[SemanticChunk]:
    """
    对帧列表做完整的流式场景切分，返回所有 SemanticChunk。
    """
    segmenter = StreamingSceneSegmenter(perception, config)
    chunks: list[SemanticChunk] = []

    for idx, frame in enumerate(frames):
        chunk = segmenter.push_frame(frame, idx)
        if chunk:
            chunks.append(chunk)

    last = segmenter.flush()
    if last:
        chunks.append(last)

    return chunks


# ---------------------------------------------------------------------------
# 便捷：从视频文件读取（依赖 OpenCV）
# ---------------------------------------------------------------------------

def segment_video_file(
    video_path: str,
    perception: PerceptionLayer,
    config: Optional[SegmentationConfig] = None,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
) -> list[SemanticChunk]:
    try:
        import cv2
    except ImportError:
        raise ImportError("需要安装 opencv-python: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    segmenter = StreamingSceneSegmenter(perception, config)
    chunks: list[SemanticChunk] = []
    frame_idx       = 0
    processed_count = 0

    # 全局耗时统计
    global_stats = {
        "video_io_time": 0.0,
        "color_cvt_time": 0.0,
        "total_start": time.perf_counter()
    }

    print(f"[Segmenter] 开始处理: {video_path}")
    try:
        while True:
            # 测试 OpenCV I/O 读取速度
            t_io_start = time.perf_counter()
            ret, bgr = cap.read()
            global_stats["video_io_time"] += (time.perf_counter() - t_io_start)
            
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            if frame_idx % frame_skip == 0:
                # 测试颜色转换速度
                t_cvt_start = time.perf_counter()
                rgb   = bgr[:, :, ::-1]  # BGR -> RGB
                global_stats["color_cvt_time"] += (time.perf_counter() - t_cvt_start)
                
                chunk = segmenter.push_frame(rgb, frame_idx)
                if chunk:
                    chunks.append(chunk)
                    print(f"  ✓ {chunk}")
                processed_count += 1

            frame_idx += 1

    finally:
        cap.release()

    last = segmenter.flush()
    if last:
        chunks.append(last)
        print(f"  ✓ {last} [final]")
        
    total_time = time.perf_counter() - global_stats["total_start"]

    print(f"\n[Segmenter] 完成。共读取 {frame_idx} 帧，"
          f"实际处理 {processed_count} 帧，切分出 {len(chunks)} 个语义片段。")
    
    # ---------------- 耗时分析报告打印 ----------------
    if processed_count > 0:
        p_time = segmenter.stats["perception_time"]
        s_time = segmenter.stats["similarity_time"]
        d_time = segmenter.stats["decision_time"]
        io_time = global_stats["video_io_time"]
        cvt_time = global_stats["color_cvt_time"]
        
        print("\n" + "="*40)
        print(" ⏱ 性能瓶颈分析报告 (Performance Profile)")
        print("="*40)
        print(f"总计耗时: {total_time:.3f} s")
        print(f"每帧平均处理耗时: {(total_time / processed_count) * 1000:.2f} ms / 帧")
        print("-" * 40)
        print(f"1. 视频解码与读取 (Video I/O) : {io_time:.3f} s  ({(io_time/total_time)*100:.1f}%)")
        print(f"2. BGR 转 RGB (Color Cvt)   : {cvt_time:.3f} s  ({(cvt_time/total_time)*100:.1f}%)")
        print(f"3. 特征感知层 (Perception)    : {p_time:.3f} s  ({(p_time/total_time)*100:.1f}%)")
        print(f"4. 相似度计算 (Similarity)    : {s_time:.3f} s  ({(s_time/total_time)*100:.1f}%)")
        print(f"5. 切分策略运算 (Decision)    : {d_time:.3f} s  ({(d_time/total_time)*100:.1f}%)")
        print("="*40 + "\n")

    return chunks
