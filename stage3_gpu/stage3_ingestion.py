# stage3_ingestion.py
"""
Stage2 → Stage3 桥接模块：语义片段入库

重构说明（numpy → torch.Tensor）：
  - _to_numpy() 移除，改为 _to_tensor(feat, device) 统一转换为 GPU Tensor
  - _aggregate_global_feature()：特征堆叠、加权平均、归一化全部在 GPU 完成
  - _extract_entity_features()：直接返回 GPU torch.Tensor (K, D)，无 numpy 中转
  - ingest_chunk()：vlm_tokens 拼接改为 torch.cat，全程 GPU；
    特征维度不匹配时使用 torch.zeros + 切片修正（替代 numpy zero-padding）

核心职责（不变）：
  1. 从 SemanticChunk 帧序列聚合全局摘要向量 I_global（→ GPU 索引）
     策略：按帧间相似度差异加权平均，变化大的帧权重更高
  2. 从 SimilarityCalculator.object_buffer 提取实体 EMA 特征 E_entity（→ GPU 索引）
  3. 拼接各帧的 VLM Patch Token（→ GPU Token Pool）
  4. 调用 ChunkStore.add_chunk() 原子写入
"""

from __future__ import annotations

import warnings
from typing import Optional, List, Union
import numpy as np
import torch
import torch.nn.functional as F

# stage2 类型（调用方负责 sys.path 正确）
from stage2.stage3_segmentation_gpu import SemanticChunk
from stage2.stage2_similarity_gpu import SimilarityCalculator

from .stage3_storage import ChunkStore


# ---------------------------------------------------------------------------
# 内部工具：将任意特征统一转为 GPU Tensor float32
# ---------------------------------------------------------------------------

def _to_tensor(
    feat: Union[torch.Tensor, "np.ndarray"],
    device: str,
) -> torch.Tensor:
    """
    将 GPU/CPU torch.Tensor 或 np.ndarray 统一转换为指定 device 的 float32 Tensor。
    torch.Tensor 直接 .to()，np.ndarray 经 torch.from_numpy() 转换（零拷贝）。
    """
    if isinstance(feat, torch.Tensor):
        return feat.to(device=device, dtype=torch.float32)
    # np.ndarray 兜底
    import numpy as np
    return torch.from_numpy(np.asarray(feat, dtype="float32")).to(device)


# ---------------------------------------------------------------------------
# 聚合策略：I_global（GPU 全程）
# ---------------------------------------------------------------------------

def _aggregate_global_feature(
    global_features: List[Union[torch.Tensor, "np.ndarray"]],
    similarity_scores: Optional[List[float]],
    device: str,
) -> torch.Tensor:
    """
    将帧序列的全局特征聚合为单一摘要向量 I_global (D,) GPU Tensor。

    策略：差异度加权平均
      - similarity_scores 可用时：帧间相似度越低（变化越大）→ 权重越高
      - 无 scores 时退化为均值池化

    返回 L2 归一化向量 (D,) GPU Tensor float32。
    """
    # 统一转 GPU Tensor，再堆叠
    feats = torch.stack(
        [_to_tensor(f, device) for f in global_features], dim=0
    )  # (T, D)

    T = feats.shape[0]

    if similarity_scores and len(similarity_scores) >= T - 1:
        # 权重：1.0 - sim，变化越大权重越高；第一帧基准权重 1.0
        raw_w = torch.ones(T, dtype=torch.float32, device=device)
        for i, sim in enumerate(similarity_scores[: T - 1]):
            raw_w[i + 1] = max(0.1, 1.0 - float(sim))
        weights = raw_w / raw_w.sum()                        # (T,) 归一化
    else:
        weights = torch.full((T,), 1.0 / T, dtype=torch.float32, device=device)

    # 加权求和：weights (T,) → (T, 1) 广播
    I_global = (feats * weights.unsqueeze(1)).sum(dim=0)    # (D,)
    return F.normalize(I_global, dim=0)                     # L2 归一化 (D,)


# ---------------------------------------------------------------------------
# 聚合策略：E_entity（GPU 全程）
# ---------------------------------------------------------------------------

def _extract_entity_features(
    entity_buffer: dict,
    max_hidden_frames: int,
    device: str,
) -> torch.Tensor:
    """
    从 SemanticChunk.entity_buffer 快照中提取活跃实体 EMA 特征。

    ema_feature 已为 GPU torch.Tensor (D,)，F.normalize 后直接 stack。
    返回 (K, D) GPU Tensor；无活跃实体时返回 (0, 0) 空 Tensor。
    """
    if not entity_buffer:
        return torch.zeros((0, 0), dtype=torch.float32, device=device)

    vecs: List[torch.Tensor] = []
    for state in entity_buffer.values():
        if state.hidden_count <= max_hidden_frames // 2:
            # ema_feature 已在 _update_buffer() 中归一化，直接使用
            feat = state.ema_feature.to(device=device, dtype=torch.float32)
            vecs.append(feat)

    if not vecs:
        return torch.zeros((0, 0), dtype=torch.float32, device=device)

    return torch.stack(vecs, dim=0)   # (K, D) GPU


# ---------------------------------------------------------------------------
# 主入库函数
# ---------------------------------------------------------------------------

def ingest_chunk(
    chunk,                          # SemanticChunk
    similarity_calc=None,           # (已废弃，保留兼容旧签名)
    store=None,                     # ChunkStore
    feat_dim: int = 768,
    skip_entity: bool = False,      # [NEW] True → global_only 消融：跳过实体提取
) -> None:
    """
    将一个 SemanticChunk 写入 ChunkStore。全程 GPU Tensor 运算。
 
    参数：
        chunk           : stage2 输出的语义片段
        similarity_calc : (已废弃) 保留仅为兼容旧调用签名，不再使用
        store           : 目标存储器
        feat_dim        : 特征维度（用于构建空 Tensor）
        skip_entity     : True 时直接存空 E_entity，跳过实体特征提取；
                          用于 global_only 消融以节省计算开销。
    """
    frames = chunk.frames
    if not frames:
        return
 
    device = store.gpu_index.device
 
    # ── 1. 收集帧级全局特征（GPU Tensor） ─────────────────────────────
    global_features = [f.global_feature for f in frames]
    sim_scores      = [s.s_total for s in chunk.similarity_scores]
 
    # ── 2. 聚合 I_global（GPU 加权平均 + L2 归一化） ───────────────────
    I_global = _aggregate_global_feature(global_features, sim_scores, device)
 
    # ── 3. 提取 E_entity ───────────────────────────────────────────────
    if skip_entity:
        # global_only 消融：检索阶段只用 I_global，实体信息不需要存储
        E_entity = torch.zeros((0, feat_dim), dtype=torch.float32, device=device)
    else:
        # 正常路径：从 chunk 快照中提取活跃实体 EMA 特征
        max_hidden_frames = 5
        if similarity_calc is not None and hasattr(similarity_calc, "cfg"):
            max_hidden_frames = getattr(similarity_calc.cfg, "max_hidden_frames", 5)
        E_entity = _extract_entity_features(chunk.entity_buffer, max_hidden_frames, device)
 
        # 修正空矩阵维度
        if E_entity.shape[0] == 0:
            E_entity = torch.zeros((0, feat_dim), dtype=torch.float32, device=device)
        elif E_entity.shape[1] != feat_dim:
            K = E_entity.shape[0]
            d = min(E_entity.shape[1], feat_dim)
            padded = torch.zeros((K, feat_dim), dtype=torch.float32, device=device)
            padded[:, :d] = E_entity[:, :d]
            E_entity = padded
 
    # ── 4. 构建 VLM Token 矩阵（帧级 Patch Token 拼接，GPU） ──────────
    if frames[0].patch_tokens is not None:
        cpu_tokens = torch.cat(
            [
                (
                    f.patch_tokens
                    if not isinstance(f.patch_tokens, torch.Tensor)
                    else f.patch_tokens.detach().cpu()
                )
                for f in frames
            ],
            dim=0,
        )
        vlm_tokens = cpu_tokens.to(device=device, dtype=torch.float32)
    else:
        import warnings
        warnings.warn(
            "[stage3_ingestion] FramePerception.patch_tokens 为 None，"
            "退化为 global_feature 近似 VLM Token（丢失帧内空间结构）。"
            "请将 PerceptionLayer 升级为使用 CLIPEncoder.encode_frame_full()。",
            RuntimeWarning,
            stacklevel=2,
        )
        vlm_tokens = torch.stack(
            [_to_tensor(f, device) for f in global_features], dim=0
        )
 
    # ── 5. 原子写入 ────────────────────────────────────────────────────
    store.add_chunk(
        chunk_id=chunk.chunk_id,
        start_frame=chunk.start_frame,
        end_frame=chunk.end_frame,
        I_global=I_global,
        E_entity=E_entity,
        vlm_tokens=vlm_tokens,
    )