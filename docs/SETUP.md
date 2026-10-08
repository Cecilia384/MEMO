# Setup and Model Weights

Run commands from the repository root. Inference requires Linux, Python 3.10, and one visible NVIDIA CUDA GPU. The [paper](https://arxiv.org/pdf/2609.38900) reports experiments on NVIDIA A6000 GPUs and single-A6000 efficiency measurements. A fresh installation has not been certified.

```bash
conda create -n memo python=3.10 -y
conda activate memo
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python -m memo.scripts.doctor
```

The CUDA 12.8 wheels need a compatible driver. `SAM2_BUILD_CUDA=0` skips SAM2's optional CUDA extension; small-hole or small-region postprocessing may be skipped. FlashAttention and a separate GroundingDINO installation are not required.

Download the perception models and one backbone. Weights are not distributed here.

```bash
mkdir -p weights
hf download openai/clip-vit-large-patch14 \
  --local-dir weights/clip-vit-large-patch14
hf download IDEA-Research/grounding-dino-base \
  --local-dir weights/grounding-dino-base
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir weights/Qwen3-VL-8B-Instruct
curl -L --fail \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o weights/sam2.1_hiera_large.pt
```

CLIP, Grounding-DINO, and backbone directories need their configuration and processor/tokenizer files as well as weights. SAM2 uses `configs/sam2.1/sam2.1_hiera_l.yaml` from the installed SAM2 package.

| Config model | Backbone |
| --- | --- |
| `qwen3` | Qwen3-VL-8B-Instruct, per-frame image mode |
| `qwen25` | Qwen2.5-VL-7B-Instruct |
| `llava_7b` | LLaVA-OneVision Qwen2 7B, Hugging Face version |
| `llava_05b` | LLaVA-OneVision Qwen2 0.5B, Hugging Face version |

Use `--model-path` for the matching backbone. The exact manuscript model revisions were not provided; pin revisions for comparisons. The runner requires exactly one visible GPU. CUDA numbering can differ from `nvidia-smi`; `CUDA_VISIBLE_DEVICES=GPU-<UUID>` selects a card unambiguously.
