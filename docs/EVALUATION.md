# Benchmark Evaluation and Outputs

The main experiments use StreamingBench real-time and OVO-Bench real-time. Prepare the native annotations and videos as described in [Dataset Preparation](DATASETS.md). Preserve original video resolution.

For the separately supplied EgoProbe evaluation package, see [EgoProbe evaluation](EGOPROBE.md).

## Run one configuration

StreamingBench example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m memo.reproduce \
  --config configs/main/streamingbench_qwen3.json \
  --annotations data/streamingbench/Real_Time_Visual_Understanding.csv \
  --video-root data/streamingbench/videos \
  --model-path weights/Qwen3-VL-8B-Instruct \
  --output results/streamingbench_qwen3.json

python -m memo.scripts.summarize results/streamingbench_qwen3.json \
  --annotations data/streamingbench/Real_Time_Visual_Understanding.csv \
  --video-root data/streamingbench/videos \
  --output-dir results/streamingbench_qwen3_report
```

OVO-Bench example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m memo.reproduce \
  --config configs/main/ovobench_qwen3.json \
  --annotations data/ovobench/ovo_bench_new.json \
  --video-root data/ovobench \
  --model-path weights/Qwen3-VL-8B-Instruct \
  --output results/ovobench_qwen3.json

python -m memo.scripts.summarize results/ovobench_qwen3.json \
  --annotations data/ovobench/ovo_bench_new.json \
  --video-root data/ovobench \
  --output-dir results/ovobench_qwen3_report
```

For another backbone, change the configuration and `--model-path` together. The eight MEMO settings are under `configs/main/`; vanilla baselines are under `configs/baselines/`. The baseline frame policies are documented in [Protocol and Provenance](PROTOCOL.md).

To preview all four models for one benchmark, run:

```bash
python -m memo.scripts.run_main_table \
  --benchmark streamingbench \
  --annotations data/streamingbench/Real_Time_Visual_Understanding.csv \
  --video-root data/streamingbench/videos \
  --weights-root weights --output-dir results/main_table
```

Add `--include-baselines` for the vanilla comparisons and `--execute` to run the commands sequentially on one visible GPU. Each experiment gets its own log and result file.

For a quick diagnostic, add `--max-videos 1 --max-questions-per-video 2`. Such outputs are marked as subsets and require `--allow-subset` for summary generation; their scores are not full benchmark results.

## Profiling

Use `configs/profiling/streamingbench_qwen25.json` with StreamingBench annotations and Qwen2.5 weights. Then run:

```bash
python -m memo.scripts.profile_report results/profile_qwen25.json \
  --warmup-questions 1 --output results/profile_report.json
```

Profiling serializes perception modules and synchronizes token timestamps. It reports retrieval, preprocessing, generation, response latency, TTFT, TPOT, and peak allocated memory. The release timing protocol differs from the historical manuscript setup.

## Result integrity

Results include configuration, selected question IDs, source and annotation hashes, dependency versions, model metadata hashes, GPU information, answers, frame counts, and scores. Model weight and video bytes are not hashed automatically; preserve their exact versions separately.

The runner saves results atomically after each question, never overwrites an existing output, and does not resume an interrupted run. For a complete report, `status` must be `completed` with exactly the expected question IDs. Pass the original annotations to `python -m memo.scripts.summarize` to verify ID coverage.

The report exports JSON, CSV, and LaTeX. StreamingBench uses micro accuracy across questions; OVO real-time uses the mean of six task accuracies. The main configurations use the supplied development scorers' permissive `legacy` rules; [Protocol and Provenance](PROTOCOL.md) describes them and the optional `strict` mode. Published targets in `configs/paper_targets.json` are never loaded as predictions.
