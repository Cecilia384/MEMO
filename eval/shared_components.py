"""
MEMO/eval/shared_components.py
--------------------
三个数据集测试脚本（OVO-Bench / RVS / StreamingBench）共用的组件：

  _TeeStream             — 日志双写（终端 + 文件）
  StageBackend           — Stage2 / Stage3 模块动态加载（CPU / GPU）
  RetrievalCLIPEncoder   — 检索侧 CLIP 文本编码器
  ChunkFrameStore        — CPU 帧缓存（chunk_id → PIL帧列表，FIFO 淘汰）
  BaseLLMAdapter         — 多模态基座模型适配器抽象基类
  LlavaOVAdapter         — LLaVA-OV 适配器
  Qwen25VLAdapter        — Qwen2.5-VL 适配器
  Qwen3VLAdapter         — Qwen3-VL 适配器（支持 image / video 两种输入模式）
  build_model_adapter    — 按名称构建并加载适配器的工厂函数
"""

import abc
import importlib
import logging
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel, CLIPProcessor
from transformers.generation.logits_process import LogitsProcessor

logger = logging.getLogger(__name__)

 
# ─────────────────────────────────────────────────────────────
# 日志双写流
# ─────────────────────────────────────────────────────────────

class _TeeStream:
    """将 print / logging 同时输出到终端与日志文件。"""

    def __init__(self, filename: str):
        self._term = sys.__stdout__
        self._file = open(filename, "a", encoding="utf-8", buffering=1)

    def write(self, msg: str):
        self._term.write(msg)
        self._file.write(msg)
        self._file.flush()

    def flush(self):
        self._term.flush()
        self._file.flush()

    def isatty(self):
        return self._term.isatty()

    def fileno(self):
        return self._term.fileno()


# ─────────────────────────────────────────────────────────────
# Stage2 / Stage3 动态后端
# ─────────────────────────────────────────────────────────────

class StageBackend:
    """
    封装 Stage2 / Stage3 的模块导入。
    默认使用 GPU 版本（--use_cpu_backend 时切换到 CPU 版本）。
    """

    def __init__(self, use_gpu: bool = True):
        if not use_gpu:
            raise ValueError("This minimal package contains only the GPU backend.")
        self.use_gpu = use_gpu
        suffix = "_gpu" if use_gpu else ""

        perception_mod   = self._import(f"stage2.stage1_perception{suffix}")
        similarity_mod   = self._import(f"stage2.stage2_similarity{suffix}")
        segmentation_mod = self._import(f"stage2.stage3_segmentation{suffix}")

        self.CLIPEncoder             = perception_mod.CLIPEncoder
        self.GroundingDINODetector   = perception_mod.GroundingDINODetector
        self.SAM2Tracker             = perception_mod.SAM2Tracker
        self.PerceptionLayer         = perception_mod.PerceptionLayer

        self.SimilarityCalculator    = similarity_mod.SimilarityCalculator
        self.SimilarityConfig        = similarity_mod.SimilarityConfig

        self.StreamingSceneSegmenter = segmentation_mod.StreamingSceneSegmenter
        self.SegmentationConfig      = segmentation_mod.SegmentationConfig

        s3_pkg = "stage3_gpu" if use_gpu else "stage3"
        self.build_chunk_store = self._import(f"{s3_pkg}.stage3_storage").build_chunk_store
        self.ChunkRetriever    = self._import(f"{s3_pkg}.stage3_retrieval").ChunkRetriever
        self.ingest_chunk      = self._import(f"{s3_pkg}.stage3_ingestion").ingest_chunk

        tag = "GPU 后端" if use_gpu else "CPU 后端"
        logger.info(f"StageBackend 初始化完成：{tag}")

    @staticmethod
    def _import(module_name: str):
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"无法导入 '{module_name}': {e}") from e


# ─────────────────────────────────────────────────────────────
# 检索侧 CLIP 编码器
# ─────────────────────────────────────────────────────────────

class RetrievalCLIPEncoder:
    def __init__(self, model_path: str, device: str = "cuda"):
        logger.info("加载检索 CLIP 模型...")
        self.device    = device
        self.model     = CLIPModel.from_pretrained(model_path).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_path)
        logger.info("检索 CLIP 加载完成")

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out    = self.model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        feat = self.model.text_projection(out.pooler_output)
        return F.normalize(feat, dim=-1).squeeze(0)


# ─────────────────────────────────────────────────────────────
# CPU 帧缓存
# ─────────────────────────────────────────────────────────────

class ChunkFrameStore:
    """chunk_id → PIL帧列表；超限 FIFO 淘汰。"""

    def __init__(self, max_frames_per_chunk: int = 8, max_total_frames: int = 2048):
        # self.max_frames_per_chunk = max_frames_per_chunk
        self.max_total_frames     = max_total_frames
        self._store: Dict[int, List[Image.Image]] = {}
        self._order: List[int] = []
        self._total: int = 0

    # def _subsample_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
    #     if len(frames) <= self.max_frames_per_chunk:
    #         return list(frames)
    #     idxs = np.linspace(0, len(frames) - 1, self.max_frames_per_chunk, dtype=int)
    #     return [frames[i] for i in idxs]

    def add(self, chunk_id: int, frames: List[Image.Image]) -> None:
        if not frames:
            return
        # frames = self._subsample_frames(frames)
        if chunk_id in self._store:
            self._total -= len(self._store[chunk_id])
            self._store[chunk_id] = frames
            self._total += len(frames)
        else:
            self._store[chunk_id] = frames
            self._order.append(chunk_id)
            self._total += len(frames)
        while self._total > self.max_total_frames and len(self._order) > 1:
            old_id  = self._order.pop(0)
            evicted = self._store.pop(old_id, [])
            self._total -= len(evicted)

    def get(self, chunk_id: int) -> List[Image.Image]:
        return self._store.get(chunk_id, [])

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()
        self._total = 0

    def __repr__(self) -> str:
        return (
            f"ChunkFrameStore(chunks={len(self._store)}, "
            f"total_frames={self._total}/{self.max_total_frames})"
        )


# ─────────────────────────────────────────────────────────────
# 多模态基座模型适配器
# ─────────────────────────────────────────────────────────────

class _TokenTimestampProcessor(LogitsProcessor):
    def __init__(self, timestamps: List[float]):
        self.timestamps = timestamps

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.timestamps.append(time.perf_counter())
        return scores


class BaseLLMAdapter(abc.ABC):
    def __init__(self):
        self.model     = None
        self.processor = None

    @abc.abstractmethod
    def load(self, model_path: str, device: str) -> None: ...

    @abc.abstractmethod
    def build_inputs(
        self, pil_frames: List[Image.Image], text_prompt: str, device: str
    ) -> Any: ...

    def decode_output(self, gen_ids: torch.Tensor, input_ids: torch.Tensor) -> str:
        trimmed = [out[len(inp):] for inp, out in zip(input_ids, gen_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def measure_generate(self, inputs: Any, max_new_tokens: int, device: str) -> Dict[str, Any]:
        if device.startswith("cuda"):
            torch.cuda.synchronize(device=device)
            baseline_allocated = torch.cuda.memory_allocated(device=device)
            baseline_reserved = torch.cuda.memory_reserved(device=device)
            torch.cuda.reset_peak_memory_stats(device=device)
        else:
            baseline_allocated = 0
            baseline_reserved = 0

        token_timestamps: List[float] = []
        timestamp_processor = _TokenTimestampProcessor(token_timestamps)
        t_gen_start = time.perf_counter()
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,  # 显式置空，交由贪心解码接管
                top_p=None,        # 显式置空
                top_k=None,        # 显式置空
                use_cache=True,
                logits_processor=[timestamp_processor],
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize(device=device)
            peak_allocated = torch.cuda.max_memory_allocated(device=device)
            peak_reserved = torch.cuda.max_memory_reserved(device=device)
        else:
            peak_allocated = 0
            peak_reserved = 0
        t_gen_end = time.perf_counter()

        output_text = self.decode_output(gen_ids, inputs.input_ids)
        input_len = int(inputs.input_ids.shape[-1])
        output_len = int(gen_ids.shape[-1])
        generated_tokens = max(output_len - input_len, 0)

        ttft_ms = None
        tpot_ms = None
        if token_timestamps:
            ttft_ms = (token_timestamps[0] - t_gen_start) * 1000.0
            if len(token_timestamps) >= 2:
                diffs = [
                    (token_timestamps[i] - token_timestamps[i - 1]) * 1000.0
                    for i in range(1, len(token_timestamps))
                ]
                tpot_ms = sum(diffs) / len(diffs)

        return {
            "gen_ids": gen_ids,
            "output_text": output_text,
            "generated_tokens": generated_tokens,
            "generation_ms": (t_gen_end - t_gen_start) * 1000.0,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "baseline_allocated_gb": baseline_allocated / (1024 ** 3),
            "baseline_reserved_gb": baseline_reserved / (1024 ** 3),
            "peak_allocated_gb": peak_allocated / (1024 ** 3),
            "peak_reserved_gb": peak_reserved / (1024 ** 3),
            "delta_peak_allocated_gb": max(peak_allocated - baseline_allocated, 0) / (1024 ** 3),
        }

class LlavaOVAdapter(BaseLLMAdapter):
    def load(self, model_path: str, device: str) -> None:
        from transformers import LlavaOnevisionForConditionalGeneration

        model_tag = "0.5B" if "0.5b" in model_path.lower() else "7B"
        logger.info(f"加载 LLaVA-OV 模型... 规格={model_tag}")
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",

        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        logger.info(f"LLaVA-OV 加载完成: 规格={model_tag}")

    def build_inputs(
        self, pil_frames: List[Image.Image], text_prompt: str, device: str
    ) -> Any:
        # pil_frames = [f.resize((336, 336)) for f in pil_frames]
        messages   = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text_prompt}]}]
        prompt     = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[prompt], videos=[pil_frames], return_tensors="pt").to(
            device, dtype=self.model.dtype
        )


class Qwen25VLAdapter(BaseLLMAdapter):
    def load(self, model_path: str, device: str) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration
        logger.info("加载 Qwen2.5-VL 模型...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            device_map="auto",

        )

        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )
        logger.info("Qwen2.5-VL 加载完成")

    def build_inputs(
        self, pil_frames: List[Image.Image], text_prompt: str, device: str
    ) -> Any:
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "video", "video": pil_frames, "fps": 1.0},
            {"type": "text",  "text": text_prompt},
        ]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        return self.processor(
            text=[prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device, dtype=self.model.dtype)


class Qwen3VLAdapter(BaseLLMAdapter):
    """
    Qwen3 推荐用逐帧 image 模式（更稳）；
    video 模式因 temporal_patch_size 压缩可能导致 rope index iterator 耗尽。
    """

    def __init__(self, mode: str = "image"):
        super().__init__()
        assert mode in ("image", "video")
        self.mode = mode

    def load(self, model_path: str, device: str) -> None:
        from transformers import Qwen3VLForConditionalGeneration
        logger.info(f"加载 Qwen3-VL 模型... (mode={self.mode})")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",

        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )
        logger.info("Qwen3-VL 加载完成")

    def build_inputs(
        self, pil_frames: List[Image.Image], text_prompt: str, device: str
    ) -> Any:
        from qwen_vl_utils import process_vision_info

        if self.mode == "video":
            messages = [{"role": "user", "content": [
                {"type": "video", "video": pil_frames, "fps": 1.0},
                {"type": "text",  "text": text_prompt},
            ]}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            return self.processor(
                text=[prompt], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(device, dtype=getattr(self.model, "dtype", torch.float16))

        # mode == "image"（推荐）：每帧作为独立 image
        image_content = [{"type": "image", "image": frame} for frame in pil_frames]
        messages = [{"role": "user", "content": image_content + [{"type": "text", "text": text_prompt}]}]
        prompt   = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _, _ = process_vision_info(
            messages, image_patch_size=16,
            return_video_kwargs=True, return_video_metadata=True,
        )
        return self.processor(
            text=[prompt], images=image_inputs, videos=None,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

def build_model_adapter(
    model_name: str,
    model_path: str,
    device: str,
    qwen3_mode: str = "image",
) -> BaseLLMAdapter:
    if model_name in ("llava_7b", "llava_05b"):   # ← 拆开，替换原来的 "llava"
        adapter = LlavaOVAdapter()
    elif model_name == "qwen25":
        adapter = Qwen25VLAdapter()
    elif model_name == "qwen3":
        adapter = Qwen3VLAdapter(mode=qwen3_mode)
    else:
        raise ValueError(f"未知 model={model_name}")
    adapter.load(model_path, device)
    return adapter