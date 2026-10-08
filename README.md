# MEMO

**Multi-Level Entity-Aware Memory for Streaming Video Understanding**

MEMO is a training-free video memory framework. It combines global visual features, entity features, and spatial continuity to partition a video into semantic chunks. At question time, it retrieves relevant historical chunks and combines their visual evidence with the current chunk for multimodal reasoning.

This repository is a **reproduction release candidate** organized around the supplied MEMO manuscript. It includes native annotation readers for the real-time subsets of StreamingBench and OVO-Bench, explicit experiment configurations, vanilla-model baselines, profiling, result aggregation, and a self-contained synthetic example.

**Status:** the code and small-example execution can be checked independently. The manuscript's full benchmark scores have not been reproduced with this release. Known differences between the manuscript and the supplied development code are recorded in [Protocol and Provenance](docs/PROTOCOL.md). The project is licensed under MIT; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

## Contents

- [Method](#method)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Quick Start](#quick-start)
- [Benchmark Evaluation](#benchmark-evaluation)
- [Performance Profiling](#performance-profiling)
- [Outputs and Reproducibility](#outputs-and-reproducibility)
- [Validation and Limitations](#validation-and-limitations)
- [Citation and License](#citation-and-license)

## Method

```text
Video sampled at 1 FPS
        |
        v
CLIP + Grounding-DINO + SAM2
        |
        v
Global, entity, and spatial similarity -> Adaptive temporal chunks
        |
        v
GPU retrieval indices + CPU visual evidence
        |
Question -> CLIP text features -> Retrieve top-3 historical chunks
        |
        v
Up to 8 historical frames + up to 8 current-chunk frames
        |
        v
Multimodal model -> Answer -> Per-task and overall evaluation
```

Questions use only frames at or before their timestamps. Questions with a `null` timestamp are answered after the video. Videos are processed independently, with memory reset between videos. The implementation preloads sampled frames before temporal processing; it does not provide a live camera or network-stream interface.

## Repository Layout

```text
MEMO_github/
├── README.md
├── LICENSE                       # MIT License
├── NOTICE.md                     # Source and dependency licensing notice
├── CITATION.cff
├── requirements.txt              # Reference dependency pins
├── reproduce.py                  # Main benchmark and reproduction entry point
├── configuration.py              # Strict JSON configuration validation
├── reproducibility.py            # Environment, source, and input provenance
├── configs/
│   ├── main/                     # 2 benchmarks x 4 MEMO backbones
│   ├── baselines/                # Explicit baseline sampling protocols
│   ├── profiling/                # Instrumented Qwen2.5 run
│   ├── smoke.json
│   └── paper_targets.json        # Manuscript values, NOT measured results
├── benchmarks/                   # Native data readers, prompts, and scoring
├── stage1/                       # Decord decoding and temporal sampling
├── stage2/                       # Perception and temporal segmentation
├── stage3_gpu/                   # Memory construction and retrieval
├── eval/                         # Model adapters, temporal processing, output
├── scripts/                      # Environment check, suites, reports, example
├── examples/synthetic/           # Original generated video and annotations
├── tests/                        # Lightweight and runtime regression checks
├── docs/                         # Dataset, protocol, and validation details
└── .github/workflows/checks.yml   # CPU-only checks; no model downloads
```

Run commands from the repository root. No parent checkout, editable local dependency, private cache, or existing result file is required.

`reproduce.py` is the supported entry point for both benchmarks and custom datasets. `stage1/sampling.py` holds the frame-index calculation, `stage1/frame_extractor.py` decodes those frames, and `eval/output.py` writes results atomically.

## Installation

Inference requires Linux, Python 3.10, and an NVIDIA CUDA GPU. The existing local reference environment uses an RTX A6000. The minimum GPU memory requirement has not been established. Install Git and curl before using the commands below.

```bash
conda create -n memo python=3.10 -y
conda activate memo
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python scripts/doctor.py
```

These pins describe the reference environment, not a fresh-install certification. CUDA 12.8 wheels require a compatible driver. `SAM2_BUILD_CUDA=0` skips the optional SAM2 CUDA extension; small-hole or small-region postprocessing may be skipped. FlashAttention and a separate GroundingDINO installation are not required.

The reproduction runner requires exactly one visible GPU so automatic model placement cannot silently use additional devices. CUDA device numbers can differ from `nvidia-smi` numbering; check the selected GPU and its free memory before a run. On shared machines, `CUDA_VISIBLE_DEVICES=GPU-<UUID>` selects the intended card unambiguously. The Qwen3-VL-8B two-video real-sample diagnostic exceeded a 24 GB RTX 4090 in one run, so use an adequately sized card for that configuration.

## Model Weights

Prepare the perception models and one backbone. Weights are not included.

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

CLIP, Grounding-DINO, and backbone directories must contain complete configuration and processor/tokenizer files as well as weights. Grounding-DINO is loaded locally and must be downloaded in advance. SAM2 uses `configs/sam2.1/sam2.1_hiera_l.yaml` from the installed SAM2 package.

| Configuration model | Backbone |
| --- | --- |
| `qwen3` | Qwen3-VL-8B-Instruct, per-frame image mode |
| `qwen25` | Qwen2.5-VL-7B-Instruct |
| `llava_7b` | LLaVA-OneVision Qwen2 7B, Hugging Face version |
| `llava_05b` | LLaVA-OneVision Qwen2 0.5B, Hugging Face version |

Use `--model-path` to select the matching weights. The precise revisions used for the manuscript tables were not provided. Record and reuse an explicit revision when downloading models for comparisons.

## Quick Start

The repository includes an eight-second synthetic video with a square that changes from red to blue. It is intended to validate execution, not benchmark accuracy. No benchmark data download is needed.

First validate annotations without loading any models:

```bash
python reproduce.py \
  --config configs/smoke.json \
  --annotations examples/synthetic/custom.json \
  --video-root examples/synthetic \
  --validate-only
```

Run the complete pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce.py \
  --config configs/smoke.json \
  --annotations examples/synthetic/custom.json \
  --video-root examples/synthetic \
  --model-path weights/Qwen3-VL-8B-Instruct \
  --output results/smoke.json

python scripts/summarize.py results/smoke.json \
  --output-dir results/smoke_report --allow-subset
```

Expected execution contract: `status` is `completed`, there are three nonempty answers, and each question uses at most 16 frames. Exact answer strings and accuracy are not pass/fail criteria. To regenerate the sample in a new directory:

```bash
python scripts/make_example.py --output-dir examples/generated
```

Use [Custom Datasets](docs/CUSTOM_DATASET.md) for your own JSON annotations.

## Benchmark Evaluation

The paper's main table covers **StreamingBench real-time** and **OVO-Bench real-time**, not OVO backward or forward tasks. Obtain datasets separately and follow [Dataset Preparation](docs/DATASETS.md). Preserve original resolutions and do not pre-resize benchmark videos.

StreamingBench:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce.py \
  --config configs/main/streamingbench_qwen3.json \
  --annotations data/streamingbench/Real_Time_Visual_Understanding.csv \
  --video-root data/streamingbench/videos \
  --model-path weights/Qwen3-VL-8B-Instruct \
  --output results/streamingbench_qwen3.json

python scripts/summarize.py results/streamingbench_qwen3.json \
  --annotations data/streamingbench/Real_Time_Visual_Understanding.csv \
  --video-root data/streamingbench/videos \
  --output-dir results/streamingbench_qwen3_report
```

OVO-Bench:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce.py \
  --config configs/main/ovobench_qwen3.json \
  --annotations data/ovobench/ovo_bench_new.json \
  --video-root data/ovobench \
  --model-path weights/Qwen3-VL-8B-Instruct \
  --output results/ovobench_qwen3.json

python scripts/summarize.py results/ovobench_qwen3.json \
  --annotations data/ovobench/ovo_bench_new.json \
  --video-root data/ovobench \
  --output-dir results/ovobench_qwen3_report
```

For the other backbones, change the configuration filename and `--model-path` together. All eight MEMO configurations are in `configs/main/`. Dataset and model paths are relative to the current working directory unless absolute paths are supplied.

For a diagnostic run, add `--max-videos 1 --max-questions-per-video 2`. Diagnostic results are marked as subsets; use `--allow-subset` when summarizing them. Never compare their aggregate scores with full-table targets.

To preview commands for all four MEMO backbones on one benchmark:

```bash
python scripts/run_main_table.py \
  --benchmark streamingbench \
  --annotations data/streamingbench/Real_Time_Visual_Understanding.csv \
  --video-root data/streamingbench/videos \
  --weights-root weights --output-dir results/main_table
```

Append `--include-baselines` to include vanilla-model comparisons. Append `--execute` and expose one GPU to run sequentially. Each experiment gets a separate log and result file. Model directory names are listed in the script; alternatively, use individual commands with `--model-path`.

Baseline configurations are in `configs/baselines/`. LLaVA baselines use 32 uniformly sampled frames from the observed prefix. Qwen baselines use all observed 1-FPS frames. These are explicit interpretations of the manuscript frame column; the exact original baseline truncation policy was not supplied.

## Performance Profiling

Run `reproduce.py` with `configs/profiling/streamingbench_qwen25.json`, the StreamingBench annotations, and Qwen2.5 weights. After the run:

```bash
python scripts/profile_report.py results/profile_qwen25.json \
  --warmup-questions 1 --output results/profile_report.json
```

Profiling serializes perception modules and synchronizes token timestamps. The report separates retrieval, preprocessing, generation, complete response latency, TTFT, TPOT, and peak allocated memory. This is an explicit release timing protocol, not a reproduction of the historical table's exact measurements. Proprietary and third-party competitor implementations are not included.

## Outputs and Reproducibility

Each output contains the resolved configuration, dataset selection, source and annotation hashes, dependency versions, model metadata hashes, GPU information, raw answers, selected-frame counts, per-task counts, and scores. Model weight bytes and video bytes are not automatically hashed; retain their exact versions separately.

Output files are saved atomically after each question. Existing files are not overwritten and interrupted runs are not resumed. A run must finish with `status: completed` and exactly the expected question IDs before its results can be used for a full report.

`scripts/summarize.py` exports JSON, CSV, and LaTeX. StreamingBench uses micro accuracy over questions; OVO real-time uses macro accuracy over its six tasks. Reports include task counts so coverage is visible. Supplying the original annotations to the report command additionally checks exact ID coverage.

Main configurations use the local development scorers' permissive `legacy` policies for comparison. Set `scoring` to `strict` in a copied configuration for conservative option-letter parsing. These policies can produce different scores; see [Protocol and Provenance](docs/PROTOCOL.md).

Manuscript targets are stored separately in `configs/paper_targets.json`. They are never loaded as predictions or used to fill missing measurements.

## Validation and Limitations

```bash
python -m unittest discover -s tests -v
python reproduce.py --help
```

The GitHub workflow checks annotations, configuration validation, frame allocation, scoring, and syntax without installing model dependencies. A temporal regression test additionally runs when inference dependencies are installed. See [Validation](docs/VALIDATION.md) for the checks actually completed for this release.

Important implementation limits:

- Sampled frames are preloaded on CPU; memory use is not bounded for arbitrarily long videos.
- A large historical image-cache threshold reduces eviction during benchmark runs but is not an unlimited-history guarantee. GPU indices can outlive evicted image evidence.
- The current model adapters re-encode selected images for each question. Cross-question reuse of backbone-specific visual embeddings described in the manuscript is not implemented.
- Baseline Qwen configurations use all observed sampled frames and can exceed model context or device memory on long videos. The manuscript does not specify the original baseline truncation policy.
- The supplied development scripts and manuscript disagree on some frame budgets and evidence handling. This release follows explicit documented configurations; numerical equivalence to the original tables requires the original experiment artifacts.

## Citation and License

Please cite the [MEMO paper](https://arxiv.org/abs/2609.38900) (arXiv DOI: `10.48550/arXiv.2609.38900`) when using this implementation. Author metadata and the paper citation are included in [CITATION.cff](CITATION.cff). The paper is accepted by ACM Multimedia 2026; add the final proceedings metadata when confirmed.

The project code and documentation are available under the [MIT License](LICENSE). Confirm any third-party attribution obligations before public distribution. Model weights and benchmark data are not distributed here; their own terms apply. See [NOTICE.md](NOTICE.md).
