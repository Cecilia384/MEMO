# MEMO/stage2/stage2_similarity.py
"""
第二阶段：多维度时空连续性度量 (Similarity Calculation)  [GPU 加速版]

S_total = α·S_spatial + β·S_local + γ·S_global

主要改动：
  - 缓存命中后的 FramePerception 默认留在 CPU。
  - 仅在 update() 内把当前帧 global_feature / local_feature / 当前帧 mask
    和上一帧缓存 mask 搬到 GPU 参与计算。
  - EMA/object_buffer 继续驻留 GPU，避免重复搬运历史特征。
  - 其余相似度计算逻辑（cosine、mask IoU、EMA 更新）保持不变。
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from memo.stage2.stage1_perception_gpu import FramePerception


def _to_device_tensor(t: Optional[torch.Tensor], device: str, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if t is None:
        return None
    kwargs = {"device": device, "non_blocking": True}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return t if (t.device.type == device and (dtype is None or t.dtype == dtype)) else t.to(**kwargs)


@dataclass
class SimilarityConfig:
    # 融合权重
    alpha: float = 0.35   # S_spatial 权重
    beta: float = 0.45    # S_local 权重
    gamma: float = 0.20   # S_global 权重

    # EMA 动量（局部特征缓存更新）
    ema_momentum: float = 0.3   # α in v̄_i^(t) = α·v_i^(t) + (1-α)·v̄_i^(t-1)

    # 惩罚项系数（非对称：新出现 > 消失）
    gamma_app: float = 0.4   # 新物体涌现惩罚系数
    gamma_dis: float = 0.2   # 物体消失惩罚系数

    # 抗遮挡：隐藏帧数容忍上限
    max_hidden_frames: int = 5

    # 空间相似度：最大位移归一化基准
    spatial_alpha: float = 0.6
    unmatched_spatial: float = 0.0
    max_displacement: float = 0.5

    # 运行设备
    device: str = "cuda"


# ---------------------------------------------------------------------------
# 场景缓存：每个被跟踪物体的历史状态
# ---------------------------------------------------------------------------

@dataclass
class ObjectBufferState:
    """场景缓存中单个物体的状态（特征保存为 GPU tensor）"""
    track_id: int
    ema_feature: torch.Tensor    # v̄_i：EMA 累积特征，shape (D,)，在 GPU 上
    weight: float                # w_i：归一化像素面积 / 置信度
    hidden_count: int = 0        # 连续未观测帧数（抗遮挡计数器 c_m）
    is_hidden: bool = False      # 是否处于隐藏状态


@dataclass
class SimilarityResult:
    """单帧相似度计算的完整输出"""
    s_total: float
    s_spatial: float
    s_local: float
    s_global: float
    p_penalty: float
    frame_idx: int
    # 调试信息
    n_appeared: int = 0
    n_disappeared: int = 0
    n_matched: int = 0


# ---------------------------------------------------------------------------
# 第二阶段主类
# ---------------------------------------------------------------------------

class SimilarityCalculator:
    """
    多维度时空连续性度量器（GPU 加速版）。

    核心计算全部在 GPU 上完成：
      - 全局余弦相似度：torch cosine_similarity
      - 空间 IoU + 位移：torch bool mask 运算 + 向量减法
      - 局部 EMA 缓存：torch tensor，in-place 乘加
      - 惩罚项：torch 标量加权求和

    前提：FramePerception / TrackedObject 中的特征字段均已是 GPU tensor
    （由 stage1 CLIPEncoder / SAM2Tracker 直接产出），无需在此处转换。

    内部维护：
      - prev_perception: 上一帧感知结果
      - object_buffer: Dict[track_id -> ObjectBufferState]（ema_feature 在 GPU）
      - _prev_global_t / _prev_obj_map / _prev_mask_map: 上一帧预缓存的 GPU tensors
    """

    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.cfg = config or SimilarityConfig()
        self.device = self.cfg.device

        self.prev_perception: Optional[FramePerception] = None
        self.object_buffer: dict[int, ObjectBufferState] = {}

        # 上一帧在 GPU 上的缓存
        self._prev_global_t: Optional[torch.Tensor] = None   # (D,)
        self._prev_obj_map: dict[int, torch.Tensor] = {}     # tid -> local feat (D,)
        self._prev_mask_map: dict[int, torch.Tensor] = {}    # tid -> bool mask (H,W)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def update(self, curr: FramePerception) -> Optional[SimilarityResult]:
        """
        输入当前帧感知结果，返回与上一帧的相似度。
        第一帧返回 None（无法比较）。

        改动：仅在 update() 内把当前帧相似度计算所需张量搬到 GPU，
              避免缓存命中后整帧 perception 长驻 GPU。
        """
        if self.prev_perception is None:
            self._init_buffer(curr)
            self.prev_perception = curr
            return None

        # ── 当前帧：仅把本帧计算所需张量搬到 GPU ─────────────────────────
        curr_global_t: torch.Tensor = _to_device_tensor(curr.global_feature, self.device, torch.float32)
        curr_obj_map: dict[int, torch.Tensor] = {
            o.track_id: _to_device_tensor(o.local_feature, self.device, torch.float32)
            for o in curr.tracked_objects
        }
        curr_mask_map: dict[int, torch.Tensor] = {
            o.track_id: _to_device_tensor(o.mask, self.device, torch.bool)
            for o in curr.tracked_objects
            if o.mask is not None
        }

        # ── 三个维度 ─────────────────────────────────────────────────────
        s_global  = self._calc_global(curr_global_t)
        s_spatial = self._calc_spatial(curr, curr_mask_map)
        s_local, p_penalty, debug = self._calc_local(curr, curr_obj_map)

        s_total = (self.cfg.alpha * s_spatial
                   + self.cfg.beta  * s_local
                   + self.cfg.gamma * s_global)

        # ── 更新 EMA 缓存 ────────────────────────────────────────────────
        self._update_buffer(curr, curr_obj_map)

        # ── 保存当前帧 GPU 缓存供下一帧使用 ─────────────────────────────
        self._prev_global_t = curr_global_t
        self._prev_obj_map  = curr_obj_map
        self._prev_mask_map = curr_mask_map
        self.prev_perception = curr

        return SimilarityResult(
            s_total   = max(0.0, min(1.0, float(s_total))),
            s_spatial = max(0.0, min(1.0, float(s_spatial))),
            s_local   = float(s_local),
            s_global  = max(0.0, min(1.0, s_global)),
            p_penalty = float(p_penalty),
            frame_idx = curr.frame_idx,
            n_appeared    = debug["n_appeared"],
            n_disappeared = debug["n_disappeared"],
            n_matched     = debug["n_matched"],
        )

    def reset(self):
        """场景切换后清空所有状态"""
        self.prev_perception = None
        self.object_buffer.clear()
        self._prev_global_t = None
        self._prev_obj_map.clear()
        self._prev_mask_map.clear()

    # ------------------------------------------------------------------
    # S_global：全局余弦相似度（GPU）
    # ------------------------------------------------------------------

    def _calc_global(self, curr_global_t: torch.Tensor) -> float:
        """
        v_prev, v_curr 均已归一化（stage1 encode_frame_full 中 F.normalize），
        所以点积即余弦相似度，映射到 [0,1]。
        """
        cos_sim = torch.dot(self._prev_global_t, curr_global_t)  # scalar tensor
        return ((cos_sim + 1.0) / 2.0).item()

    # ------------------------------------------------------------------
    # S_spatial：SAM2 Mask IoU + 中心点位移（GPU）
    # ------------------------------------------------------------------
    def _calc_spatial(self,
                      curr: FramePerception,
                      curr_mask_map: dict[int, torch.Tensor]) -> float:
        prev_objs = self.prev_perception.tracked_objects
        curr_objs = curr.tracked_objects

        if not prev_objs or not curr_objs:
            return 0.5

        prev_ids = {o.track_id for o in prev_objs}
        curr_ids = {o.track_id for o in curr_objs}
        common   = prev_ids & curr_ids

        if not common:
            return self.cfg.unmatched_spatial

        prev_obj_by_id = {o.track_id: o for o in prev_objs}
        curr_obj_by_id = {o.track_id: o for o in curr_objs}

        common_list = list(common)

        # ── 1. 批量计算 Mask IoU（GPU，大尺寸 H×W tensor 值得留在 GPU）──
        prev_masks = []
        curr_masks = []
        for tid in common_list:
            p_mask = self._prev_mask_map.get(tid)
            c_mask = curr_mask_map.get(tid)
            if p_mask is not None and c_mask is not None and p_mask.shape == c_mask.shape:
                prev_masks.append(p_mask)
                curr_masks.append(c_mask)

        if prev_masks:
            p_masks_t = torch.stack(prev_masks)
            c_masks_t = torch.stack(curr_masks)

            inter = (p_masks_t & c_masks_t).sum(dim=(1, 2)).float()
            union = (p_masks_t | c_masks_t).sum(dim=(1, 2)).float()
            iou_t = inter / (union + 1e-8)
            s_iou = iou_t.mean().item()
        else:
            s_iou = 0.0

        # ── 2. 中心点位移（纯 Python，少量标量不值得创建 GPU tensor）──
        import math
        max_disp = self.cfg.max_displacement
        disp_sum = 0.0
        n = len(common_list)
        for tid in common_list:
            pc = prev_obj_by_id[tid].bbox.center
            cc = curr_obj_by_id[tid].bbox.center
            d = math.hypot(cc[0] - pc[0], cc[1] - pc[1])
            disp_sum += max(0.0, 1.0 - d / max_disp)
        s_disp = disp_sum / n

        return self.cfg.spatial_alpha * s_iou + (1 - self.cfg.spatial_alpha) * s_disp

    # ------------------------------------------------------------------
    # S_local：带 EMA 缓存 + P_penalty 的局部语义连续性（GPU）
    # ------------------------------------------------------------------
    
    def _calc_local(self,
                    curr: FramePerception,
                    curr_obj_map: dict[int, torch.Tensor]) \
            -> tuple[float, float, dict]:
        cfg = self.cfg
        O_curr = set(curr_obj_map.keys())
        O_buf  = set(self.object_buffer.keys())

        matched_ids = O_curr & O_buf
        matched_list = list(matched_ids)

        # ── 1. 匹配集合：批量余弦相似度 + 加权求和（GPU，向量化值得）────────
        if matched_list:
            curr_obj_by_id = {o.track_id: o for o in curr.tracked_objects}

            # 构建批量 tensor：(M, D)
            obs_vecs = torch.stack([curr_obj_map[tid] for tid in matched_list])
            buf_vecs = torch.stack([self.object_buffer[tid].ema_feature for tid in matched_list])
            weights  = torch.tensor([curr_obj_by_id[tid].pixel_area_ratio for tid in matched_list],
                                    dtype=torch.float32, device=self.device)

            # 逐行余弦相似度（映射到 [0,1]）
            cos_sims = (obs_vecs * buf_vecs).sum(dim=1)
            sims     = (cos_sims + 1.0) / 2.0

            weight_sum = weights.sum()
            if weight_sum > 0:
                s_local_raw = ((sims * weights).sum() / weight_sum).item()
            else:
                s_local_raw = 0.5
        else:
            s_local_raw = 0.5

        # ── 2. 惩罚项 P_penalty（纯 Python，少量标量不值得 GPU tensor）────
        curr_obj_by_id = {o.track_id: o for o in curr.tracked_objects}

        A = O_curr - O_buf   # 新出现
        total_weight_curr = sum(curr_obj_by_id[tid].pixel_area_ratio for tid in O_curr) + 1e-8
        appeared_weight   = sum(curr_obj_by_id[tid].pixel_area_ratio for tid in A)

        p_app = cfg.gamma_app * (appeared_weight / total_weight_curr)

        # 隐藏物体衰减惩罚
        p_dis = 0.0
        max_hf = max(cfg.max_hidden_frames, 1)
        for _, s in self.object_buffer.items():
            if s.is_hidden and s.hidden_count > 0:
                ratio = min(s.hidden_count / max_hf, 1.0)
                p_dis += s.weight * (ratio ** 2)
        p_dis *= cfg.gamma_dis

        p_penalty = p_app + p_dis
        s_local   = s_local_raw - p_penalty

        debug = {
            "n_appeared":    len(A),
            "n_disappeared": sum(1 for s in self.object_buffer.values() if s.is_hidden and s.hidden_count > 0),
            "n_matched":     len(matched_ids),
        }
        return s_local, p_penalty, debug

    # ------------------------------------------------------------------
    # 缓存管理（EMA 更新在 GPU tensor 上 in-place 完成）
    # ------------------------------------------------------------------

    def _init_buffer(self, perception: FramePerception):
        """
        初始化缓存（第一帧）。
        改动：直接使用 stage1 输出的 GPU tensor，无需 _to_gpu / _mask_to_gpu 转换。
        """
        self.object_buffer.clear()
        self._prev_global_t = _to_device_tensor(perception.global_feature, self.device, torch.float32)
        self._prev_obj_map  = {}
        self._prev_mask_map = {}

        for obj in perception.tracked_objects:
            feat_t = _to_device_tensor(obj.local_feature, self.device, torch.float32)
            self.object_buffer[obj.track_id] = ObjectBufferState(
                track_id    = obj.track_id,
                ema_feature = feat_t.clone(),
                weight      = obj.pixel_area_ratio,
            )
            self._prev_obj_map[obj.track_id]  = feat_t
            if obj.mask is not None:
                self._prev_mask_map[obj.track_id] = _to_device_tensor(obj.mask, self.device, torch.bool)

    def _update_buffer(self,
                       curr: FramePerception,
                       curr_obj_map: dict[int, torch.Tensor]):
        """
        EMA 更新：
          v̄_i^(t) = α·v_i^(t) + (1-α)·v̄_i^(t-1)
        新出现目标直接加入缓存，消失目标递增隐藏计数并超阈值删除。

        优化：对已存在目标做 batch 化 EMA + normalize，减少 Python 循环中
        逐个调用 F.normalize 的开销。
        """
        cfg = self.cfg
        O_curr = set(curr_obj_map.keys())
        curr_obj_by_id = {o.track_id: o for o in curr.tracked_objects}

        # 分类：已存在（需 EMA 更新） vs 新增
        existing_tids = []
        new_tids = []
        for tid in curr_obj_map:
            if tid in self.object_buffer:
                existing_tids.append(tid)
            else:
                new_tids.append(tid)

        # ── batch 化 EMA 更新 ──────────────────────────────────────────
        if existing_tids:
            alpha = cfg.ema_momentum
            obs_batch = torch.stack([curr_obj_map[tid] for tid in existing_tids])       # (M, D)
            ema_batch = torch.stack([self.object_buffer[tid].ema_feature for tid in existing_tids])  # (M, D)

            # 批量 EMA + 归一化
            updated = alpha * obs_batch + (1 - alpha) * ema_batch
            updated = F.normalize(updated, dim=-1)

            for i, tid in enumerate(existing_tids):
                buf = self.object_buffer[tid]
                buf.ema_feature  = updated[i]
                buf.weight       = curr_obj_by_id[tid].pixel_area_ratio
                buf.hidden_count = 0
                buf.is_hidden    = False

        # ── 新增目标 ──────────────────────────────────────────────────
        for tid in new_tids:
            self.object_buffer[tid] = ObjectBufferState(
                track_id    = tid,
                ema_feature = curr_obj_map[tid].clone(),
                weight      = curr_obj_by_id[tid].pixel_area_ratio,
            )

        # ── 消失目标 ──────────────────────────────────────────────────
        for tid in list(self.object_buffer.keys()):
            if tid not in O_curr:
                buf = self.object_buffer[tid]
                buf.hidden_count += 1
                buf.is_hidden     = True
                if buf.hidden_count > cfg.max_hidden_frames:
                    del self.object_buffer[tid]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    计算两个 bool mask 的 IoU（numpy 版，供外部直接调用兼容）。
    SimilarityCalculator 内部已改用 GPU tensor 版本。
    """
    if mask_a.shape != mask_b.shape:
        return 0.0
    intersection = (mask_a & mask_b).sum()
    union        = (mask_a | mask_b).sum()
    return float(intersection) / (float(union) + 1e-8)


def _decay_function(hidden_count: int, max_hidden: int) -> float:
    """
    抗遮挡衰减函数 f(c_m)：惩罚权重随丢失帧数非线性递增。
    使用平方曲线：f(c) = (c / max_hidden)^2
    """
    ratio = min(hidden_count / max(max_hidden, 1), 1.0)
    return ratio ** 2