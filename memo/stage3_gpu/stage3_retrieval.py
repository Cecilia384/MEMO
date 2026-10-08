# stage3_retrieval.py
"""
第三阶段检索模块：多粒度级联相似度检索

重构说明（numpy → torch.Tensor）：
  - RetrievalResult.vlm_tokens 由 np.ndarray 改为 torch.Tensor (GPU)
  - retrieve_frames_for_query 中的 token 聚合由 np.concatenate / np.linspace
    改为 torch.cat + 整数索引，全程 GPU 运算
  - CLIPTextEncoder.encode() 输出保持 GPU Tensor，无变化
  - ChunkStore / GPUTokenPool 接口已全部返回 GPU Tensor，此处无需额外转换

Pipeline:
  Query (text / vec)
      │
      ▼ CLIP Text Encoder（可选，若传入向量则跳过）
      │
      ▼ GPU 并行初筛  GPUChunkIndex.search()
        → Top-K (chunk_id, score, meta)
      │
      ▼ GPU Token 召回  GPUTokenPool.get(chunk_id)
        → [N, D] VLM Tokens（GPU Tensor）
      │
      ▼ RetrievalResult（完整结构化输出）
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from memo.stage3_gpu.stage3_storage import ChunkStore, ChunkMeta, GPUChunkIndex, CPUTokenPool


# ---------------------------------------------------------------------------
# 检索结果
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """单次检索的完整输出"""
    rank:       int                         # 排名（0-based）
    chunk_id:   int
    score:      float
    sim_global: float
    sim_local:  float
    meta:       ChunkMeta
    vlm_tokens: Optional[torch.Tensor]     # (N, D) CPU Tensor；未命中则 None


# ---------------------------------------------------------------------------
# CLIP 文本编码器（轻量封装，检索时按需使用）
# ---------------------------------------------------------------------------

class CLIPTextEncoder:
    """
    CLIP Text Encoder 封装。
    encode() 返回 GPU torch.Tensor (D,)，与 GPUChunkIndex.search() 接口一致。
    """

    def __init__(
        self,
        model_path: str = "openai/clip-vit-large-patch14",
        device:     str = "cuda",
    ):
        from transformers import CLIPModel, CLIPProcessor
        self.device    = device
        self.model     = CLIPModel.from_pretrained(model_path).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_path)
        print(f"[CLIPTextEncoder] Loaded '{model_path}'. device={device}")

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """
        输入文本，返回归一化特征向量 (D,) GPU Tensor。
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out    = self.model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        feat = self.model.text_projection(out.pooler_output)  # (1, D)
        return F.normalize(feat, dim=-1).squeeze(0)           # (D,) GPU


# ---------------------------------------------------------------------------
# 核心检索器
# ---------------------------------------------------------------------------

class ChunkRetriever:
    """
    多粒度级联检索器。

    支持两种查询模式：
      1. 文本查询  retriever.query_text("a red car on highway", top_k=5)
      2. 向量查询  retriever.query_vec(vec_tensor, top_k=5)

    内部执行：
      ① GPU 并行初筛（宏观 + 局部实体相似度加权）
      ② GPU Token 召回（GPUTokenPool，O(1) LRU）
    """

    def __init__(
        self,
        store:        ChunkStore,
        text_encoder: Optional[CLIPTextEncoder] = None,
        device:       str = "cuda",
        lambda_global: float = 0.6,
        lambda_local:  float = 0.4,
        strategy:     str = "ours",
        random_seed:  int = 42,
    ):
        self.store         = store
        self.text_encoder  = text_encoder
        self.device        = device
        self.lambda_global = lambda_global
        self.lambda_local  = lambda_local
        self.strategy      = strategy
        self.random_seed   = random_seed
        self._rng          = random.Random(random_seed)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def query_text(
        self,
        text:          str,
        top_k:         int  = 5,
        recall_tokens: bool = True,
    ) -> List[RetrievalResult]:
        """
        文本查询。text_encoder 必须已初始化。
        recall_tokens: 是否从 GPU 池召回 [N, D] token（False 时略快）。
        """
        if self.text_encoder is None:
            raise ValueError("text_encoder 未初始化，请在构造时传入 CLIPTextEncoder。")
        query_vec = self.text_encoder.encode(text)   # (D,) GPU
        return self._search(query_vec, query_vec, top_k, recall_tokens)

    def query_vec(
        self,
        vec:           torch.Tensor,    # (D,) GPU Tensor，已归一化
        top_k:         int  = 5,
        recall_tokens: bool = True,
    ) -> List[RetrievalResult]:
        """向量查询（全局向量同时作为实体查询参考）。"""
        return self._search(vec, vec, top_k, recall_tokens)

    def query_vec_dual(
        self,
        global_vec:    torch.Tensor,             # (D,) 宏观查询向量
        entity_vec:    Optional[torch.Tensor],   # (D,) 实体查询向量，None 则禁用局部打分
        top_k:         int  = 5,
        recall_tokens: bool = True,
    ) -> List[RetrievalResult]:
        """双向量查询：分别指定宏观查询和实体查询向量。"""
        return self._search(global_vec, entity_vec, top_k, recall_tokens)

    # ------------------------------------------------------------------
    # 内部流程
    # ------------------------------------------------------------------

    def _search_semantic(
        self,
        query_global: torch.Tensor,
        query_entity: Optional[torch.Tensor],
        top_k: int,
    ) -> List[dict]:
        return self.store.gpu_index.search(
            query_global=query_global,
            query_entity=query_entity,
            top_k=top_k,
            lambda_global=self.lambda_global,
            lambda_local=self.lambda_local,
        )

    def _search_nearest_k(self, top_k: int) -> List[dict]:
        metas = sorted(self.store.gpu_index.get_all_meta(), key=lambda meta: meta.chunk_id, reverse=True)
        selected = metas[: max(top_k, 0)]
        return [
            {
                "chunk_id": meta.chunk_id,
                "score": float(len(selected) - rank),
                "sim_global": 0.0,
                "sim_local": 0.0,
                "meta": meta,
            }
            for rank, meta in enumerate(selected)
        ]

    def _search_random_k(self, top_k: int) -> List[dict]:
        metas = sorted(self.store.gpu_index.get_all_meta(), key=lambda meta: meta.chunk_id)
        if not metas or top_k <= 0:
            return []
        if len(metas) <= top_k:
            selected = list(metas)
        else:
            selected = self._rng.sample(metas, top_k)
        return [
            {
                "chunk_id": meta.chunk_id,
                "score": 0.0,
                "sim_global": 0.0,
                "sim_local": 0.0,
                "meta": meta,
            }
            for meta in selected
        ]

    def _materialize_results(self, raw_results: List[dict], recall_tokens: bool) -> List[RetrievalResult]:
        final: List[RetrievalResult] = []
        for rank, r in enumerate(raw_results):
            tokens = self.store.recall_tokens(r["chunk_id"]) if recall_tokens else None
            final.append(RetrievalResult(
                rank=rank,
                chunk_id=r["chunk_id"],
                score=r["score"],
                sim_global=r["sim_global"],
                sim_local=r["sim_local"],
                meta=r["meta"],
                vlm_tokens=tokens,
            ))
        final.sort(key=lambda res: res.meta.start_frame)
        return final

    def _search(
        self,
        query_global:  torch.Tensor,
        query_entity:  Optional[torch.Tensor],
        top_k:         int,
        recall_tokens: bool,
    ) -> List[RetrievalResult]:
        """
        ① 按 strategy 选择检索方式
        ② GPU token 召回
        ③ 按时间顺序（start_frame）对召回结果排序，保证输入 VLM 时序一致
        """
        if self.strategy in {"ours", "global_only", "local_only"}:
            raw_results = self._search_semantic(query_global, query_entity, top_k)
        elif self.strategy == "nearest_k":
            raw_results = self._search_nearest_k(top_k)
        elif self.strategy == "random_k":
            raw_results = self._search_random_k(top_k)
        else:
            raise ValueError(f"Unsupported retrieval strategy: {self.strategy}")
        return self._materialize_results(raw_results, recall_tokens)

    # ------------------------------------------------------------------
    # 便捷：批量文本查询（返回合并后去重帧列表）
    # ------------------------------------------------------------------

    def retrieve_frames_for_query(
        self,
        text:       str,
        top_k:      int = 3,
        max_frames: int = 16,
    ) -> Tuple[List[RetrievalResult], torch.Tensor]:
        """
        检索 Top-K 片段并聚合其 VLM Token（行方向拼接），
        同时做均匀子采样以限制帧数。全程 GPU Tensor 运算。

        返回: (results, aggregated_tokens)
            aggregated_tokens 形状：(min(N_total, max_frames), D) GPU Tensor
        """
        results = self.query_text(text, top_k=top_k, recall_tokens=True)

        token_chunks = [
            r.vlm_tokens for r in results if r.vlm_tokens is not None
        ]

        if not token_chunks:
            empty = torch.zeros(
                (0, self.store.gpu_index.feat_dim),
                dtype=torch.float32,
                device="cpu",        # 与 CPUTokenPool 一致
            )
            return results, empty

        all_tokens = torch.cat(token_chunks, dim=0)   # (N_total, D) CPU Tensor

        if all_tokens.shape[0] > max_frames:
            # 均匀采样索引在 CPU 计算即可
            idxs = torch.linspace(
                0, all_tokens.shape[0] - 1, max_frames
            ).long()
            all_tokens = all_tokens[idxs]             # (max_frames, D) CPU

        return results, all_tokens