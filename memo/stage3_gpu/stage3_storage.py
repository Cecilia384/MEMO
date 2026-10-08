# stage3_storage.py
"""
第三阶段存储模块：异构特征的分层存储

将高维稠密特征（CPU）与轻量级检索索引（GPU）完全解耦：
  - CPUTokenPool   : 高分辨率 VLM Token 存储池，[N, D] 以 CPU torch.Tensor 存储
  - GPUChunkIndex  : 轻量级多模态索引，{I_global, E_entity}，驻留 GPU

重构说明（numpy → torch.Tensor）：
  - CPUTokenPool：内部存储由 np.ndarray 改为 CPU torch.Tensor，
    LRU 改用 collections.OrderedDict，get/put 复杂度从 O(n) 降至 O(1)；
    Token 数据仍保留在 CPU 内存，架构语义不变
  - ChunkMeta：I_global / E_entity 由 np.ndarray 改为 GPU torch.Tensor
  - GPUChunkIndex：_build_global_matrix 直接 torch.stack，无 numpy 中转；
    search() 中实体分支通过预先堆叠的 padded 张量实现并行批量打分

使用方：
    store, pool, index = build_chunk_store(feat_dim=768, device="cuda")

    # 写入一个 SemanticChunk
    store.add_chunk(chunk_id, start_frame, end_frame,
                    I_global, E_entity, vlm_tokens)

    # 检索时仅需 index，命中后通过 pool 召回原始 token
    results = index.search(query_global, query_entity, top_k=5)
    tokens  = pool.get(chunk_id)          # → torch.Tensor [N, D] on CPU
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ChunkMeta:
    """
    存储在 GPU 索引中的轻量级元信息。
    I_global : 片段全局语义摘要向量  (D,)  GPU torch.Tensor float32
    E_entity : 局部实体记忆集合      (K, D) GPU torch.Tensor float32，K 可为 0
    """
    chunk_id:    int
    start_frame: int
    end_frame:   int
    duration:    int                        # end_frame - start_frame + 1
    I_global:    torch.Tensor               # (D,)  GPU
    E_entity:    torch.Tensor               # (K, D) GPU，K=0 时为空张量


# ---------------------------------------------------------------------------
# CPU 高分辨率 Token 存储池
# ---------------------------------------------------------------------------

class CPUTokenPool:
    """
    CPU 内存中的高分辨率 VLM Token 存储池。

    所有 token 以 CPU torch.Tensor (float32) 存储，架构上与 GPU 索引解耦。
    LRU 淘汰基于 collections.OrderedDict，get/put 均为 O(1)。

    写入:  pool.put(chunk_id, tokens)       # tokens: torch.Tensor [N, D]（任意 device）
    读取:  pool.get(chunk_id)               # → torch.Tensor [N, D] on CPU | None
    删除:  pool.evict(chunk_id)
    清空:  pool.clear()
    """

    def __init__(self, max_size: int = 2048):
        self._max_size = max_size
        # OrderedDict 充当 LRU 缓存：末尾 = 最近访问，O(1) move_to_end
        self._store: OrderedDict[int, torch.Tensor] = OrderedDict()

    # ── 写 ──────────────────────────────────────────────────────────────

    def put(self, chunk_id: int, tokens: torch.Tensor) -> None:
        """
        存入一个 chunk 的高分辨率 token。
        tokens 形状：(N, D)；无论来自 GPU 还是 CPU，统一转为 CPU float32 存储。
        """
        # 确保落在 CPU，不占用 GPU 显存
        t = tokens.detach().cpu().float()
        if chunk_id in self._store:
            # 已存在：更新值并移至末尾（O(1)）
            self._store.move_to_end(chunk_id)
            self._store[chunk_id] = t
            return
        if len(self._store) >= self._max_size:
            # LRU 淘汰：弹出最久未访问的首元素（O(1)）
            self._store.popitem(last=False)
        self._store[chunk_id] = t

    # ── 读 ──────────────────────────────────────────────────────────────

    def get(self, chunk_id: int) -> Optional[torch.Tensor]:
        """返回 chunk 对应的 [N, D] CPU Tensor，未命中返回 None"""
        if chunk_id not in self._store:
            return None
        # LRU 更新：O(1)
        self._store.move_to_end(chunk_id)
        return self._store[chunk_id]

    # ── 管理 ─────────────────────────────────────────────────────────────

    def evict(self, chunk_id: int) -> bool:
        if chunk_id in self._store:
            del self._store[chunk_id]
            return True
        return False

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"CPUTokenPool(size={self.size}/{self._max_size})"


# ---------------------------------------------------------------------------
# GPU 轻量级多模态索引
# ---------------------------------------------------------------------------

class GPUChunkIndex:
    """
    GPU 端轻量级多模态索引。
    每个 chunk 存储：
      - I_global : 帧级全局特征聚合摘要     (D,)  GPU Tensor
      - E_entity : 核心实体 EMA 特征集合    (K, D) GPU Tensor

    所有 chunk 的 I_global 堆叠为矩阵 (M, D)，支持 GPU 并行打分。
    实体分支通过 padded 张量 (M, K_max, D) 实现全并行，消除 for 循环。
    """

    def __init__(self, feat_dim: int = 768, device: str = "cuda"):
        self.feat_dim = feat_dim
        self.device   = device
        self._metas: List[ChunkMeta] = []

        # Dirty-flag 缓存：add() 后置脏，search() 前重建
        self._global_matrix:  Optional[torch.Tensor] = None   # (M, D)
        self._entity_matrix:  Optional[torch.Tensor] = None   # (M, K_max, D)
        self._entity_mask:    Optional[torch.Tensor] = None   # (M, K_max) bool
        self._dirty: bool = False

    # ── 写 ──────────────────────────────────────────────────────────────

    def add(self, meta: ChunkMeta) -> None:
        """插入一个 ChunkMeta（I_global + E_entity 均为 GPU Tensor）"""
        self._metas.append(meta)
        self._dirty = True

    # ── 构建 GPU 矩阵 ────────────────────────────────────────────────────

    def _build_matrices(self) -> None:
        """
        将所有 ChunkMeta 的 I_global / E_entity 批量堆叠为 GPU 矩阵。

        - _global_matrix : (M, D)           直接 torch.stack
        - _entity_matrix : (M, K_max, D)    zero-padded，消除实体分支 for 循环
        - _entity_mask   : (M, K_max) bool  标记有效实体位置
        """
        if not self._metas:
            self._global_matrix = self._entity_matrix = self._entity_mask = None
            self._dirty = False
            return

        # ── 全局矩阵 ─────────────────────────────────────────────────
        self._global_matrix = torch.stack(
            [m.I_global for m in self._metas], dim=0
        )  # (M, D)

        # ── 实体矩阵（padded） ───────────────────────────────────────
        M = len(self._metas)
        K_max = max(
            (m.E_entity.shape[0] for m in self._metas if m.E_entity.ndim == 2),
            default=0,
        )
        if K_max > 0:
            entity_matrix = torch.zeros(M, K_max, self.feat_dim,
                                        dtype=torch.float32, device=self.device)
            entity_mask   = torch.zeros(M, K_max,
                                        dtype=torch.bool, device=self.device)
            for i, meta in enumerate(self._metas):
                k = meta.E_entity.shape[0]
                if k > 0:
                    entity_matrix[i, :k] = meta.E_entity
                    entity_mask[i, :k]   = True
            self._entity_matrix = entity_matrix   # (M, K_max, D)
            self._entity_mask   = entity_mask     # (M, K_max)
        else:
            self._entity_matrix = None
            self._entity_mask   = None

        self._dirty = False

    # ── 并行检索 ─────────────────────────────────────────────────────────

    def search(
        self,
        query_global: torch.Tensor,              # (D,) GPU
        query_entity: Optional[torch.Tensor],    # (D,) GPU | None
        top_k: int = 5,
        lambda_global: float = 0.6,
        lambda_local:  float = 0.4,
    ) -> List[Dict[str, Any]]:
        """
        GPU 全并行多粒度相似度打分，返回 Top-K 结果。
        实体分支由 padded 矩阵批量计算，消除逐 chunk for 循环。

        返回列表元素:
            {
              "chunk_id":   int,
              "score":      float,
              "sim_global": float,
              "sim_local":  float,
              "meta":       ChunkMeta,
            }
        """
        if not self._metas:
            return []
        if self._dirty:
            self._build_matrices()

        # ── 宏观场景匹配度 Sim_global ───────────────────────────────────
        # query_global: (D,)  →  (1, D)；global_matrix: (M, D)
        q_g = F.normalize(query_global.unsqueeze(0), dim=-1)   # (1, D)
        G   = F.normalize(self._global_matrix, dim=-1)         # (M, D)
        sim_global = (G @ q_g.T).squeeze(-1)                   # (M,)

        # ── 局部实体响应度 Sim_local（全并行）──────────────────────────
        sim_local = torch.zeros(len(self._metas), device=self.device)

        if query_entity is not None and self._entity_matrix is not None:
            # q_e : (D,) → (1, 1, D)
            q_e = F.normalize(query_entity, dim=-1).unsqueeze(0).unsqueeze(0)
            # E   : (M, K_max, D) → 归一化
            E_norm = F.normalize(self._entity_matrix, dim=-1)           # (M, K_max, D)
            # 点积: (M, K_max, D) x (1, 1, D) → 沿 D 求和 → (M, K_max)
            sims = (E_norm * q_e).sum(dim=-1)                           # (M, K_max)
            # 无效 padding 位置置为 -inf，再取行最大
            sims = sims.masked_fill(~self._entity_mask, float("-inf"))  # (M, K_max)
            # 若某行全为无效（K=0），max 结果为 -inf，clamp 至 0
            sim_local = sims.max(dim=-1).values.clamp(min=0.0)          # (M,)

        # ── 融合打分 ───────────────────────────────────────────────────
        scores = lambda_global * sim_global + lambda_local * sim_local  # (M,)
        k      = min(top_k, len(self._metas))
        top_vals, top_idxs = torch.topk(scores, k)

        results = []
        for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
            meta = self._metas[idx]
            results.append({
                "chunk_id":   meta.chunk_id,
                "score":      float(val),
                "sim_global": float(sim_global[idx].item()),
                "sim_local":  float(sim_local[idx].item()),
                "meta":       meta,
            })
        return results

    # ── 管理 ─────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._metas.clear()
        self._global_matrix = self._entity_matrix = self._entity_mask = None
        self._dirty = False

    def get_all_meta(self) -> List[ChunkMeta]:
        return list(self._metas)

    @property
    def size(self) -> int:
        return len(self._metas)

    def __repr__(self) -> str:
        return (f"GPUChunkIndex(chunks={self.size}, "
                f"feat_dim={self.feat_dim}, device={self.device})")


# ---------------------------------------------------------------------------
# ChunkStore：统一写入门面（解耦存储与索引）
# ---------------------------------------------------------------------------

class ChunkStore:
    """
    统一的 chunk 写入接口，内部将数据分发到：
      - CPUTokenPool  （高分辨率 token，CPU Tensor）
      - GPUChunkIndex （轻量级索引，GPU Tensor）

    检索时：GPUChunkIndex.search() → Top-K；
            CPUTokenPool.get(chunk_id) → 原始 token CPU Tensor。
    """

    def __init__(
        self,
        cpu_pool:  CPUTokenPool,
        gpu_index: GPUChunkIndex,
    ):
        self.cpu_pool  = cpu_pool
        self.gpu_index = gpu_index

    def add_chunk(
        self,
        chunk_id:    int,
        start_frame: int,
        end_frame:   int,
        # GPU 索引所需（均为 GPU torch.Tensor float32）
        I_global:   torch.Tensor,          # (D,)
        E_entity:   torch.Tensor,          # (K, D)，K 可为 0
        # CPU Pool 所需
        vlm_tokens: torch.Tensor,          # (N, D)
    ) -> ChunkMeta:
        """
        原子写入：同时更新 GPU 索引与 CPU Token 池。
        返回写入的 ChunkMeta。
        """
        device = self.gpu_index.device

        I_global_gpu = I_global.to(device=device, dtype=torch.float32)
        E_entity_gpu = E_entity.to(device=device, dtype=torch.float32)

        # 保证空实体张量形状正确
        if E_entity_gpu.ndim == 1 or E_entity_gpu.shape[0] == 0:
            E_entity_gpu = torch.zeros(
                (0, I_global_gpu.shape[0]), dtype=torch.float32, device=device
            )

        meta = ChunkMeta(
            chunk_id=chunk_id,
            start_frame=start_frame,
            end_frame=end_frame,
            duration=end_frame - start_frame + 1,
            I_global=I_global_gpu,
            E_entity=E_entity_gpu,
        )
        self.gpu_index.add(meta)
        self.cpu_pool.put(chunk_id, vlm_tokens)
        return meta

    def recall_tokens(self, chunk_id: int) -> Optional[torch.Tensor]:
        """根据 chunk_id 从 CPU 池中召回高分辨率 token Tensor"""
        return self.cpu_pool.get(chunk_id)

    def clear(self) -> None:
        self.cpu_pool.clear()
        self.gpu_index.clear()

    def __repr__(self) -> str:
        return f"ChunkStore(gpu_index={self.gpu_index}, cpu_pool={self.cpu_pool})"


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_chunk_store(
    feat_dim:     int = 768,
    device:       str = "cuda",
    cpu_max_size: int = 2048,
) -> Tuple[ChunkStore, CPUTokenPool, GPUChunkIndex]:
    """
    一次性构建整套存储结构，返回 (store, pool, index)。
    pool 和 index 仍可独立访问。
    """
    pool  = CPUTokenPool(max_size=cpu_max_size)
    index = GPUChunkIndex(feat_dim=feat_dim, device=device)
    store = ChunkStore(pool, index)
    return store, pool, index