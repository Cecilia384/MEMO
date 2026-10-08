<div align="center">

# MEMO

**Multi-Level Entity-Aware Memory for Streaming Video Understanding**

**ACM Multimedia 2026** · [Paper](https://arxiv.org/abs/2609.38900) · [Citation](#citation-and-license) · [MIT License](LICENSE)

</div>

MEMO is a training-free framework for streaming video understanding. It builds multi-level, entity-aware structured memory and recalls relevant visual evidence for each query.

<p align="center">
  <a href="assets/teaser.png"><img src="assets/teaser.png" alt="Comparison of streaming video memory paradigms" width="80%"></a>
</p>
<p align="center"><em>Lightweight structured indices guide on-demand access to high-resolution visual evidence.</em></p>

## 🧠 Overview

The pipeline has four stages:

- **Multi-Level Entity-Aware Perception:** Measures semantic continuity using global semantics, local entity features, and spatial structure.
- **Online Temporal Chunking:** Uses an adaptive similarity threshold to form semantically coherent chunks.
- **Structured Memory Construction:** Stores lightweight global and entity-level retrieval indices on GPU and high-resolution visual evidence on CPU.
- **Query-Specific Evidence Retrieval:** Recalls visual evidence from relevant chunks and combines it with frames from the current active segment for MLLM reasoning.

<p align="center">
  <a href="assets/pipeline.png"><img src="assets/pipeline.png" alt="MEMO pipeline: perception, chunking, memory, and retrieval" width="100%"></a>
</p>
<p align="center"><em>MEMO's four-stage streaming pipeline.</em></p>

## 📊 Main Results

Online results from [Table 1 of the paper](https://arxiv.org/pdf/2609.38900) are average accuracy (%) over OVO-Bench's six real-time tasks and StreamingBench's real-time subset.

**Online methods with training**

| Method | Frames | OVO-Bench | StreamingBench |
| --- | ---: | ---: | ---: |
| VideoLLM-online-8B | 2 fps | 20.8 | 36.0 |
| Dispider-7B | 1 fps | 54.6 | 67.6 |
| Flash-VStream-7B | 1 fps | 29.9 | 23.2 |
| ViSpeak | 1 fps | 66.3 | 74.4 |
| TimeChat-Online-7B | 1 fps | 61.9 | 75.3 |
| StreamForest-7B | 1 fps | 61.2 | 77.3 |

**Training-free online methods**

| Backbone / method | Frames | OVO-Bench | StreamingBench |
| --- | ---: | ---: | ---: |
| LLaVA-OneVision-0.5B | 32 | 49.7 | 59.6 |
| ↳ + ReKV | 0.5 fps | 43.8 | 57.4 |
| ↳ **+ MEMO** | 1 fps | 50.4 | 59.5 |
| LLaVA-OneVision-7B | 32 | 63.1 | 71.1 |
| ↳ + ReKV | 0.5 fps | 57.3 | 69.1 |
| ↳ + LiveVLM | 0.5 fps | — | 72.9 |
| ↳ + StreamKV | 0.5 fps | — | 68.8 |
| ↳ **+ MEMO** | 1 fps | 66.6 | 73.2 |
| Qwen2.5-VL-7B | 1 fps | 59.9 | 73.3 |
| ↳ + FluxMem | 1 fps | 67.2 | 76.4 |
| ↳ **+ MEMO** | 1 fps | 66.9 | 78.5 |
| Qwen3-VL-8B | 1 fps | 70.1 | 73.2 |
| ↳ **+ MEMO** | 1 fps | **76.0** | **83.7** |

These are reported paper results; full benchmarks have not been rerun with this release. See the [online table LaTeX](docs/ONLINE_RESULTS.tex), [protocol](docs/PROTOCOL.md), and [validation](docs/VALIDATION.md).

## 🚀 Getting Started

Inference requires Linux, Python 3.10, one NVIDIA CUDA GPU, and model weights. For the tested configurations, prepare at least 24 GB of GPU memory, or 48 GB for Qwen3-VL-8B. See [Setup and Model Weights](docs/SETUP.md) for per-model guidance.

Validate the synthetic example without loading models:

```bash
python -m memo.reproduce \
  --config configs/smoke.json \
  --annotations examples/synthetic/custom.json \
  --video-root examples/synthetic \
  --validate-only
```

Run it with Qwen3-VL-8B:

```bash
CUDA_VISIBLE_DEVICES=0 python -m memo.reproduce \
  --config configs/smoke.json \
  --annotations examples/synthetic/custom.json \
  --video-root examples/synthetic \
  --model-path weights/Qwen3-VL-8B-Instruct \
  --output results/smoke.json
```

Benchmark configs are in `configs/main/` (four backbones × two benchmarks). See [Dataset Preparation](docs/DATASETS.md) and [Benchmark Evaluation](docs/EVALUATION.md); custom JSON is covered in [Custom Datasets](docs/CUSTOM_DATASET.md).

## 📁 Repository Guide

| Path | Purpose |
| --- | --- |
| `memo/` | Inference, benchmarks, utilities, and tests |
| `configs/` | Experiment configurations |
| `examples/`, `assets/` | Sample data and figures |
| `docs/` | Setup, evaluation, and validation guides |

## Citation and License

If you use MEMO, please cite the [paper](https://arxiv.org/abs/2609.38900):

```bibtex
@misc{li2026memomultilevelentityawarememory,
  title={MEMO: Multi-Level Entity-Aware Memory for Streaming Video Understanding},
  author={Yinying Li and Yuqian Fu and Yulin Dai and Jingyu Gong and Tianwen Qian and Xiaoling Wang},
  year={2026},
  eprint={2609.38900},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  doi={10.48550/arXiv.2609.38900},
  url={https://arxiv.org/abs/2609.38900}
}
```

The code is released under the [MIT License](LICENSE). Model weights and benchmark data have their own terms; see [NOTICE.md](NOTICE.md).
