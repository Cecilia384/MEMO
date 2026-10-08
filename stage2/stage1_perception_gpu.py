# MEMO/stage2/stage1_perception.py
"""
第一阶段：多模态流式感知与特征提取 (Perception Layer)
并行提取三个维度的核心场景特征：
  1. 全局语义特征 (CLIP)
  2. 开放词汇目标检测与时序分割 (Grounding-DINO + SAM2)
  3. 局部对象语义特征 (SAM2 mask + CLIP)

主要改动（numpy → torch GPU 直出）：
  - CLIPEncoder.encode_frame_full() / encode_frame() / encode_masked_region()
    不再 .cpu().numpy()，直接返回归一化 GPU tensor。
  - SAM2Tracker.update() 返回的 mask 改为 GPU bool tensor（torch.Tensor）。
  - TrackedObject.mask          : np.ndarray  →  torch.Tensor  (H,W) bool  [GPU]
  - TrackedObject.local_feature : np.ndarray  →  torch.Tensor  (D,)  f32   [GPU]
  - FramePerception.global_feature : np.ndarray  →  torch.Tensor  (D,)  f32   [GPU]
  - FramePerception.patch_tokens   : np.ndarray  →  torch.Tensor  (N,D) f32   [GPU]
    (仍可为 None，语义不变)
  下游 stage2 可直接使用这些 tensor，无需再调用 _to_gpu / _mask_to_gpu。
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn.functional as F
import os
import time
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

def sync_time() -> float:
    """获取当前时间，如果是GPU运算则先同步，确保计时准确"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

@dataclass
class BoundingBox:
    """归一化坐标 [0, 1] 的边界框 (x1, y1, x2, y2)"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class TrackedObject:
    """单个被跟踪目标的完整状态"""
    track_id: int                         # 全局唯一跟踪 ID
    bbox: BoundingBox                     # 当前帧边界框
    mask: torch.Tensor                    # 像素级 mask (H, W) bool  [GPU tensor]
    local_feature: torch.Tensor           # CLIP 局部语义特征向量 v_i^(t) (D,) [GPU tensor]
    pixel_area_ratio: float               # 占总像素比例（用作权重 w_i）
    label: str = "object"                # Grounding-DINO 检测类别标签


@dataclass
class FramePerception:
    """单帧完整感知输出"""
    frame_idx: int
    global_feature: torch.Tensor           # V_global: CLIP 帧级池化全局特征 (D,) [GPU tensor]
    tracked_objects: list[TrackedObject]   # O_t 集合
    patch_tokens: Optional[torch.Tensor] = None  # VLM Patch Token 序列 (N_patches, D) [GPU tensor]
                                                  # 为 None 时表示旧版感知层未提取 patch tokens。
    raw_frame: Optional[np.ndarray] = None  # 原始 BGR/RGB 帧（调试用，保留 numpy）


# ---------------------------------------------------------------------------
# CLIPEncoder —— 直接输出 GPU tensor
# ---------------------------------------------------------------------------

class CLIPEncoder:
    """
    CLIP 视觉编码器（ViT-L/14）。
    使用 HuggingFace transformers 加载。

    改动：所有 encode_* 方法直接返回归一化 GPU tensor，
    不再执行 .cpu().numpy() 转换。
    """

    def __init__(self,
                 model_name: str = "openai/clip-vit-large-patch14",
                 device: str = "cuda"):

        self.device = device
        self.model_name = model_name
        _local = os.path.isabs(model_name) or os.path.isdir(model_name)
        _clip_kwargs = {'local_files_only': True} if _local else {}
        self.model = CLIPModel.from_pretrained(model_name, **_clip_kwargs).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name, **_clip_kwargs)
        self.model.eval()
        self.dim = self.model.config.projection_dim  # 通常为 768

        # 缓存 CLIP 图像预处理参数，避免每帧重复创建 tensor
        if hasattr(self.model.config, 'vision_config'):
            self.img_size = self.model.config.vision_config.image_size
            self.patch_size = self.model.config.vision_config.patch_size
        else:
            self.img_size = 224
            self.patch_size = 14
        self.grid_size = self.img_size // self.patch_size  # 16 for ViT-L/14 @224px

        self._norm_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=device
        ).view(1, 3, 1, 1)
        self._norm_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=device
        ).view(1, 3, 1, 1)

        print(f"[CLIPEncoder] Loaded '{model_name}'. device={device}, dim={self.dim}")

    @torch.no_grad()
    def encode_frame(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """
        输入: frame_rgb (H, W, 3) uint8
        输出: 归一化池化特征向量 (D,) float32  [GPU tensor]
        （仅用于不需要 patch tokens 的场合；推荐使用 encode_frame_full）
        """
        global_feat, _ = self.encode_frame_full(frame_rgb)
        return global_feat

    def _preprocess_gpu(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """
        纯 GPU 图像预处理：numpy (H,W,3) uint8 → (1,3,img_size,img_size) 归一化 GPU tensor。
        跳过 PIL.Image.fromarray + CLIPProcessor，消除 CPU→PIL→tensor 的中间转换开销。

        严格复现 CLIPImageProcessor 的流程：
          1. 短边缩放到 img_size（保持宽高比）
          2. 中心裁剪到 (img_size, img_size)
          3. 归一化到 [0,1] 后减均值除标准差
        """
        # numpy HWC uint8 → GPU float32 CHW
        t = torch.from_numpy(frame_rgb).to(device=self.device, dtype=torch.float32)
        t = t.permute(2, 0, 1).unsqueeze(0)                          # (1,3,H,W)

        # Step 1: 短边缩放到 img_size（保持宽高比，与 CLIPProcessor 一致）
        _, _, H, W = t.shape
        if H <= W:
            new_H = self.img_size
            new_W = int(round(W * self.img_size / H))
        else:
            new_W = self.img_size
            new_H = int(round(H * self.img_size / W))
        t = F.interpolate(t, size=(new_H, new_W),
                          mode='bicubic', align_corners=False,
                          antialias=True)

        # Step 2: 中心裁剪到 (img_size, img_size)
        top  = (new_H - self.img_size) // 2
        left = (new_W - self.img_size) // 2
        t = t[:, :, top : top + self.img_size, left : left + self.img_size]

        # Step 3: 归一化
        t = t.clamp(0, 255) / 255.0                                  # → [0, 1]
        return (t - self._norm_mean) / self._norm_std                 # CLIP 标准归一化

    @torch.no_grad()
    def encode_frame_full(
        self, frame_rgb: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        单次前向传播，同时返回池化全局特征与逐 Patch Token 特征。

        输入: frame_rgb (H, W, 3) uint8
        返回:
            global_feature : 归一化池化特征  (D,)           float32  [GPU tensor]
                             对应 CLIP 的 pooler_output → visual_projection
            patch_tokens   : 归一化 Patch Token (N_patches, D) float32  [GPU tensor]
                             N_patches = (img_size/patch_size)^2 + 1 (含 CLS)
                             例：ViT-L/14 @224px → N_patches = 257, D = 768

        两个输出共享同一次 vision_model 前向，无额外计算开销。
        特征保留在 GPU 上，不再拷贝至 CPU / 转为 numpy。
        使用纯 GPU 预处理路径，跳过 PIL / CLIPProcessor 的 CPU 开销。
        """
        pixel_values = self._preprocess_gpu(frame_rgb)                # (1,3,224,224) GPU

        # 单次 vision_model 前向：同时拿到 pooler_output 和 last_hidden_state
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
        )
        # pooler_output: (1, vision_hidden_dim)  — post_layernorm 后的 CLS token
        # last_hidden_state: (1, N_patches+1, vision_hidden_dim)
        pooled_hidden = vision_outputs.pooler_output          # (1, V_dim)
        all_hidden    = vision_outputs.last_hidden_state[0]   # (N_patches+1, V_dim)

        # 投影到 CLIP 对齐空间 (V_dim → D)
        global_proj = self.model.visual_projection(pooled_hidden)   # (1, D)
        patch_proj  = self.model.visual_projection(all_hidden)       # (N_patches+1, D)

        # 归一化后直接返回 GPU tensor，省去 .cpu().numpy() 转换
        global_feature = F.normalize(global_proj, dim=-1).squeeze(0)  # (D,)
        patch_tokens   = F.normalize(patch_proj,  dim=-1)             # (N_patches+1, D)

        return global_feature, patch_tokens

    @torch.no_grad()
    def local_features_from_patches(
        self,
        patch_tokens: torch.Tensor,
        masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        利用已有的 patch tokens，通过 mask spatial pooling 提取局部特征，
        无需额外 CLIP forward pass。

        原理：patch_tokens 中 [1:] 为空间 patch（grid×grid），
        将 mask 下采样到 patch grid 分辨率后，对落在 mask 内的 patch
        做 mean pooling 即可得到该目标的局部语义特征。

        输入:
            patch_tokens : (N_patches+1, D) float32 [GPU tensor]，含 CLS token
            masks        : list of (H, W) bool [GPU tensor]
        输出:
            (N, D) float32 [GPU tensor]，归一化局部特征
        """
        if not masks:
            return torch.empty((0, self.dim), device=self.device)

        grid = self.grid_size  # 16 for ViT-L/14 @224px
        D = patch_tokens.shape[1]

        # 去掉 CLS token，只保留空间 patch tokens → (grid*grid, D)
        spatial_flat = patch_tokens[1:].contiguous()  # (grid*grid, D)

        # 将所有 mask 堆叠并下采样到 patch grid 分辨率
        masks_tensor = torch.stack(masks).float().unsqueeze(1)     # (N, 1, H, W)
        masks_ds = F.interpolate(
            masks_tensor, size=(grid, grid),
            mode='bilinear', align_corners=False,
        ).squeeze(1)                                                # (N, grid, grid)
        mask_flat = (masks_ds > 0.3).view(len(masks), grid * grid, 1).float()  # (N, G, 1)

        # 向量化批量 masked mean pooling
        # spatial_flat: (G, D) → broadcast (1, G, D) * (N, G, 1) → (N, G, D)
        weighted = spatial_flat.unsqueeze(0) * mask_flat            # (N, G, D)
        sums   = weighted.sum(dim=1)                                # (N, D)
        counts = mask_flat.sum(dim=1)                               # (N, 1)

        # 处理 mask 为空的目标（fallback 到 CLS token）
        empty = (counts.squeeze(1) == 0)                            # (N,)
        counts = counts.clamp(min=1.0)
        pooled = sums / counts                                      # (N, D)
        if empty.any():
            pooled[empty] = patch_tokens[0]                         # CLS fallback

        return F.normalize(pooled, dim=-1)

    @torch.no_grad()
    def encode_masked_regions_batch_gpu(self,
                                             frame_rgb: np.ndarray,
                                             masks: list[torch.Tensor]) -> torch.Tensor:
        """
        纯 GPU 批量编码多个被 mask 遮挡的区域（无 PIL / Processor 开销）。
        输入: frame_rgb (H, W, 3) uint8, masks list of (H, W) bool [GPU tensors]
        输出: 归一化特征向量矩阵 (N, D) float32 [GPU tensor]
        """
        if not masks:
            return torch.empty((0, self.dim), device=self.device)

        frame_tensor = torch.from_numpy(frame_rgb).to(device=self.device, dtype=torch.float32)
        
        masks_tensor = torch.stack(masks)

        masked_frames = frame_tensor.unsqueeze(0) * masks_tensor.unsqueeze(-1)

        masked_frames = masked_frames.permute(0, 3, 1, 2)

        masked_frames = F.interpolate(
            masked_frames,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )

        masked_frames = masked_frames / 255.0  # 转到 [0, 1] 区间

        pixel_values = (masked_frames - self._norm_mean) / self._norm_std

        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            # return_dict=True,
        )
        
        pooled_hidden = vision_outputs.pooler_output
        global_proj = self.model.visual_projection(pooled_hidden)

        # 返回 (N, D) 的 GPU Tensor
        return F.normalize(global_proj, dim=-1)

    @torch.no_grad()
    def encode_masked_region(self,
                             frame_rgb: np.ndarray,
                             mask: torch.Tensor) -> torch.Tensor:
        """
        [已废弃] 单目标 masked region 编码。
        内部转发到 encode_masked_regions_batch_gpu，避免浪费 patch_tokens 计算。
        新代码请直接使用 encode_masked_regions_batch_gpu 或 local_features_from_patches。
        """
        import warnings
        warnings.warn(
            "encode_masked_region() 已废弃，请改用 encode_masked_regions_batch_gpu() "
            "或 local_features_from_patches()。",
            DeprecationWarning, stacklevel=2,
        )
        return self.encode_masked_regions_batch_gpu(frame_rgb, [mask]).squeeze(0)


# ---------------------------------------------------------------------------
# GroundingDINODetector —— 不变（输入/输出与 stage1 内部逻辑相关）
# ---------------------------------------------------------------------------

class GroundingDINODetector:
    """
    Grounding-DINO 开放词汇检测器。

    安装（二选一）：
      方式 A：pip install groundingdino-py
      方式 B：git clone https://github.com/IDEA-Research/GroundingDINO
              cd GroundingDINO && pip install -e .
              # 下载权重：
              # wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

    config_path  示例：
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path 示例：
        "weights/groundingdino_swint_ogc.pth"
    """

    def __init__(self,
             weights_path: str,
             device: str = "cuda",
             box_threshold: float = 0.35,
             text_threshold: float = 0.25):


        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.processor = GroundingDinoProcessor.from_pretrained(
            weights_path, local_files_only=True)
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            weights_path, local_files_only=True).to(device)
        self.weights_path = weights_path
        self.model.eval()
        print(f"[GroundingDINO] Loaded from '{weights_path}'. "
            f"box_th={box_threshold}, text_th={text_threshold}")

    @torch.no_grad()
    def detect(self,
           frame_rgb: np.ndarray,
           text_prompt: str = (
               "person  . "
               "car . truck . bus . bicycle . motorcycle . "
               "chair . table . sofa . bed . desk . door . window . "
               "bottle . cup . bowl . food . "
               "phone . laptop . screen . monitor . keyboard . "
               "text . sign . whiteboard . board . book ."
               "ball . bat . racket . "
               "dog . cat . animal . "
           )) -> list[tuple[BoundingBox, str]]:
    # def detect(self,
    #            frame_rgb: np.ndarray,
    #            text_prompt: str = "object . animal .") \
    #         -> list[tuple[BoundingBox, str]]:
        """
        返回 [(BoundingBox, label), ...]，坐标已归一化到 [0, 1]。

        GroundingDINO 的 predict() 直接返回归一化 cxcywh，
        这里转换为 x1y1x2y2 格式。
        """
        img_tensor = Image.fromarray(frame_rgb)

        inputs = self.processor(images=img_tensor, text=text_prompt, return_tensors="pt").to(device=self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[img_tensor.size[::-1]]
        )

        boxes, logits, phrases = results[0]["boxes"].cpu().numpy(), results[0]["scores"].cpu().numpy(), results[0]["text_labels"]

        return boxes, logits, phrases


# ---------------------------------------------------------------------------
# SAM2Tracker —— mask 直接输出 GPU bool tensor
# ---------------------------------------------------------------------------

class SAM2Tracker:
    """
    SAM2 视频目标分割与跨帧跟踪器。

    安装：
      git clone https://github.com/facebookresearch/sam2
      cd sam2 && pip install -e .
      # 下载模型权重（以 large 为例）：
      # wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

    checkpoint 示例：
        "weights/sam2.1_hiera_large.pt"
    model_cfg  示例（相对于 sam2 包根目录的配置文件）：
        "configs/sam2.1/sam2.1_hiera_l.yaml"

    改动：update() 返回的 mask 由 numpy bool array 改为 GPU bool tensor，
    与 CLIPEncoder.encode_masked_region() 的新签名对齐。
    """

    def __init__(self,
                 checkpoint: str,
                 model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 device: str = "cuda"):


        self.device = device
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

        self._next_id: int = 0
        self._active_tracks: dict[int, BoundingBox] = {}  # track_id -> last bbox
        print(f"[SAM2Tracker] Loaded checkpoint='{checkpoint}'. device={device}")

    def reset(self):
        """清空跟踪状态（场景切换后调用）"""
        self._next_id = 0
        self._active_tracks.clear()

    @torch.no_grad()
    def update(self,
               frame_rgb: np.ndarray,
               detections: list[tuple[BoundingBox, str]]) \
            -> list[tuple[int, BoundingBox, torch.Tensor, str]]:
        """
        给定新帧与检测框，为每个检测框生成 SAM2 精细 mask，
        并通过 IoU 匹配完成简单跨帧跟踪。

        返回值: [(track_id, bbox, mask (H,W) bool [GPU tensor], label), ...]
        """
        if not len(detections[0]):
            return []
        if isinstance(frame_rgb, np.ndarray):
            frame_rgb = frame_rgb.copy()
        H, W = frame_rgb.shape[:2]

        # ── SAM2 批量 mask 预测（仅在有检测时才运行 image encoder）──────────
        self.predictor.set_image(frame_rgb)

        boxes_px = detections[0]

        masks_logits, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_px,                 # (N, 4)
            multimask_output=False,       # 每个 box 输出 1 个 mask
        )

        if masks_logits.ndim == 4:
            masks_logits = masks_logits.squeeze(1)

        # 转为 GPU bool tensor
        if isinstance(masks_logits, np.ndarray):
            masks_bool = torch.from_numpy(masks_logits > 0.0).to(
                device=self.device, dtype=torch.bool)   # (N, H, W)
        else:
            masks_bool = (masks_logits > 0.0).to(
                device=self.device, dtype=torch.bool)   # (N, H, W)

        # ── 跨帧 IoU 匹配（矩阵化计算）─────────────────────────────────────
        results: list[tuple[int, BoundingBox, torch.Tensor, str]] = []
        matched_ids: set[int] = set()

        # 构建当前帧检测框列表
        det_boxes = [BoundingBox(b[0], b[1], b[2], b[3], s)
                     for b, s in zip(detections[0], detections[1])]

        if self._active_tracks:
            # 矩阵化 IoU 计算：(N_det, N_track)
            track_ids = list(self._active_tracks.keys())
            track_bboxes = [self._active_tracks[tid] for tid in track_ids]
            iou_matrix = _bbox_iou_matrix(det_boxes, track_bboxes)

            for i, (det_box, label, mask_t) in enumerate(
                    zip(det_boxes, detections[2], masks_bool)):
                best_j = -1
                best_iou = 0.3  # IoU 阈值
                for j, tid in enumerate(track_ids):
                    if tid in matched_ids:
                        continue
                    if iou_matrix[i][j] > best_iou:
                        best_iou = iou_matrix[i][j]
                        best_j = j

                if best_j >= 0:
                    track_id = track_ids[best_j]
                    matched_ids.add(track_id)
                else:
                    track_id = self._next_id
                    self._next_id += 1

                self._active_tracks[track_id] = det_box
                results.append((track_id, det_box, mask_t, label))
        else:
            # 无历史轨迹，全部为新目标
            for det_box, label, mask_t in zip(det_boxes, detections[2], masks_bool):
                track_id = self._next_id
                self._next_id += 1
                self._active_tracks[track_id] = det_box
                results.append((track_id, det_box, mask_t, label))

        # 移除长时间未匹配的轨迹
        result_ids = {r[0] for r in results}
        stale = [tid for tid in self._active_tracks
                 if tid not in matched_ids and tid not in result_ids]
        for tid in stale:
            del self._active_tracks[tid]

        return results


# ---------------------------------------------------------------------------
# 第一阶段主类
# ---------------------------------------------------------------------------

class PerceptionLayer:
    """
    多模态流式感知层。
    对每一帧并行执行：
      - CLIP 全局特征提取
      - Grounding-DINO 目标检测
      - SAM2 跨帧跟踪 + mask 生成
      - CLIP 局部目标特征提取

    改动：process_frame() 内部直接使用 GPU tensor，
    FramePerception / TrackedObject 中所有特征字段均为 GPU tensor。
    """

    def __init__(self,
                 clip_encoder: CLIPEncoder,
                 detector: GroundingDINODetector,
                 tracker: SAM2Tracker,
                #  text_prompt: str = "object . person . car . animal .",
                 text_prompt: str = (
                    "person . hand . "
                    "car . truck . bus . bicycle . motorcycle . "
                    "chair . table . sofa . bed . desk . door . window . "
                    "bottle . cup . bowl . food . book ."
                    "phone . laptop . screen . monitor . keyboard . "
                    "text . sign . whiteboard . board . "
                    "ball . bat . racket . "
                    "dog . cat . "
                    # "fire . smoke . water . "
                ),
                 use_patch_pooling: bool = True,
                 dino_detect_interval: int = 1):
        self.clip = clip_encoder
        self.detector = detector
        self.tracker = tracker
        self.text_prompt = text_prompt
        self.use_patch_pooling = use_patch_pooling

        # DINO 间隔检测：每 dino_detect_interval 帧运行一次完整检测，
        # 中间帧复用上次检测结果让 SAM2 跟踪。设为 1 则每帧都检测（兼容旧行为）。
        self.dino_detect_interval = max(1, dino_detect_interval)
        self._frame_since_last_detect: int = 0
        self._last_detections = None

        # CUDA Streams：CLIP 与 DINO 并行执行
        self._stream_clip = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._stream_dino = torch.cuda.Stream() if torch.cuda.is_available() else None

        print(f"[PerceptionLayer] local feature mode: "
              f"{'patch_pooling (0 extra CLIP forward)' if use_patch_pooling else 'masked_region_forward (N extra CLIP forwards)'}")
        if dino_detect_interval > 1:
            print(f"[PerceptionLayer] DINO interval detection: every {dino_detect_interval} frames")

        self.stats = {
            "clip_global_time": 0.0,
            "dino_detect_time": 0.0,
            "sam2_track_time": 0.0,
            "clip_local_time": 0.0,
            "processed_frames": 0
        }

    def process_frame(self,
                      frame_rgb: np.ndarray,
                      frame_idx: int) -> FramePerception:
        """
        处理单帧。
        优化：CLIP 全局特征与 DINO 检测通过 CUDA Stream 并行执行；
        DINO 支持间隔检测（dino_detect_interval > 1 时中间帧复用上次检测结果）。
        """
        self.stats["processed_frames"] += 1
        H, W = frame_rgb.shape[:2]

        # 判断本帧是否需要运行 DINO 检测
        run_dino = (self._frame_since_last_detect >= self.dino_detect_interval
                    or self._last_detections is None)

        # ── CLIP + DINO 并行 ─────────────────────────────────────────────
        if self._stream_clip is not None and run_dino:
            # 两个独立模型用不同 stream 并行
            with torch.cuda.stream(self._stream_clip):
                global_feature, patch_tokens = self.clip.encode_frame_full(frame_rgb)
            with torch.cuda.stream(self._stream_dino):
                detections = self.detector.detect(frame_rgb, self.text_prompt)
            # 等待两个 stream 都完成
            torch.cuda.synchronize()
        else:
            # 串行回退（CPU 模式或复用上次检测时无需并行）
            global_feature, patch_tokens = self.clip.encode_frame_full(frame_rgb)
            if run_dino:
                detections = self.detector.detect(frame_rgb, self.text_prompt)

        if run_dino:
            self._last_detections = detections
            self._frame_since_last_detect = 0
        else:
            detections = self._last_detections
        self._frame_since_last_detect += 1

        # ── SAM2 跨帧跟踪 + Mask 生成 ───────────────────────────────────
        tracked_raw = self.tracker.update(frame_rgb, detections)

        # ── 局部语义特征 ────────────────────────────────────────────────
        tracked_objects: list[TrackedObject] = []
        total_pixels = H * W

        if tracked_raw:
            masks = [item[2] for item in tracked_raw]

            masks_tensor = torch.stack(masks) # (N, H, W)
            # pixel_ratios 保持为 GPU tensor，延迟到需要 float 时才取值
            pixel_ratios_t = masks_tensor.sum(dim=(1, 2)).float() / total_pixels

            if self.use_patch_pooling:
                local_feats_batch = self.clip.local_features_from_patches(patch_tokens, masks)
            else:
                local_feats_batch = self.clip.encode_masked_regions_batch_gpu(frame_rgb, masks)

            # 一次性取回所有 pixel_ratio（单次同步）
            pixel_ratios = pixel_ratios_t.tolist()

            for i, (track_id, bbox, mask_t, label) in enumerate(tracked_raw):
                tracked_objects.append(TrackedObject(
                    track_id=track_id,
                    bbox=bbox,
                    mask=mask_t,
                    local_feature=local_feats_batch[i],
                    pixel_area_ratio=pixel_ratios[i],
                    label=label,
                ))

        return FramePerception(
            frame_idx=frame_idx,
            global_feature=global_feature,
            patch_tokens=patch_tokens,
            tracked_objects=tracked_objects,
        )

    def reset_tracker(self):
        """场景切换后重置跟踪器状态与 DINO 检测缓存"""
        self.tracker.reset()
        self._frame_since_last_detect = 0
        self._last_detections = None

    def print_timing_stats(self):
        """打印 Perception 层性能瓶颈分析报告"""
        frames = self.stats["processed_frames"]
        if frames == 0:
            print("[PerceptionLayer] 暂无处理数据。")
            return
            
        total_time = (self.stats["clip_global_time"] + 
                      self.stats["dino_detect_time"] + 
                      self.stats["sam2_track_time"] + 
                      self.stats["clip_local_time"])
        
        print("\n" + "="*55)
        print(" ⏱ Perception Layer 内部模块耗时分析报告")
        print("="*55)
        print(f"处理帧数: {frames} 帧")
        print(f"感知层总耗时: {total_time:.3f} s")
        print(f"每帧平均耗时: {(total_time / frames) * 1000:.2f} ms/帧")
        print("-" * 55)
        
        def print_stat(name, key):
            t = self.stats[key]
            pct = (t / total_time) * 100 if total_time > 0 else 0
            avg_ms = (t / frames) * 1000
            print(f"{name:.<25}: {t:>6.3f} s  ({pct:>5.1f}%) | 平均 {avg_ms:>6.2f} ms/帧")

        print_stat("1. CLIP 全局特征提取", "clip_global_time")
        print_stat("2. Grounding-DINO 检测", "dino_detect_time")
        print_stat("3. SAM2 跟踪与 Mask", "sam2_track_time")
        print_stat("4. CLIP 局部特征提取", "clip_local_time")
        print("="*55 + "\n")
# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    """两个 BoundingBox 之间的 IoU（保留供外部兼容调用）"""
    ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / (union + 1e-8)


def _bbox_iou_matrix(boxes_a: list[BoundingBox],
                     boxes_b: list[BoundingBox]) -> list[list[float]]:
    """
    矩阵化 IoU 计算：返回 (N, M) 的 IoU 矩阵。
    boxes_a: N 个检测框, boxes_b: M 个跟踪框。
    纯 Python 向量化，避免 N*M 次函数调用开销。
    """
    N, M = len(boxes_a), len(boxes_b)
    if N == 0 or M == 0:
        return [[0.0] * M for _ in range(N)]

    # 预提取坐标，减少属性访问开销
    a_coords = [(a.x1, a.y1, a.x2, a.y2, a.area) for a in boxes_a]
    b_coords = [(b.x1, b.y1, b.x2, b.y2, b.area) for b in boxes_b]

    iou_mat: list[list[float]] = []
    for ax1, ay1, ax2, ay2, a_area in a_coords:
        row = []
        for bx1, by1, bx2, by2, b_area in b_coords:
            ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = a_area + b_area - inter
            row.append(inter / (union + 1e-8))
        iou_mat.append(row)
    return iou_mat


def _bbox_to_mask(bbox: BoundingBox, H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    x1 = int(bbox.x1 * W); y1 = int(bbox.y1 * H)
    x2 = int(bbox.x2 * W); y2 = int(bbox.y2 * H)
    mask[y1:y2, x1:x2] = True
    return mask