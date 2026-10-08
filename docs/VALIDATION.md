# Validation Record

Date: September 30, 2026.

## October 8, 2026 modular cleanup check

The legacy `run_eval.py` entry point and its processor classes, the disabled prefilter and perception cache, and unused video-reader backends were removed. The fixed decord path now uses `stage1/sampling.py`, while result persistence lives in `eval/output.py`. All 19 distributed run/example configurations retain the same values apart from the two disabled prefilter fields.

The `streamingkfs` environment passed all 14 automated tests, including a new video sampling check. Decoding the bundled synthetic video and a 448-frame StreamingBench sample produced the same frame counts, first ten native indices, and SHA256 of concatenated RGB frame bytes before and after cleanup. All four backbones completed two-question MEMO runs on each real-time benchmark: eight completed runs with two nonempty answers each. `scripts/summarize.py --allow-subset` generated both reports and verified exact question IDs against the subset annotations. All eight runs preserved their prior question IDs and per-video sampled/processed frame and chunk counts. Seven runs preserved both answer choices; LLaVA 0.5B's second OVO-Bench choice varied, consistent with its earlier sample-level variability. Logs and results are under the ignored local directory `results/verification_cleanup_20261008/`.

These diagnostics establish that the cleaned code executes on the checked samples. Full benchmark scores and fresh installation remain unverified.

## October 8, 2026 recheck

The existing `streamingkfs` environment passed all 13 automated tests and the dependency doctor with one visible CUDA device. Four backbones (`qwen3`, `qwen25`, `llava_7b`, `llava_05b`) each completed a new MEMO run on both real-time benchmarks: eight completed runs, two nonempty answers per run, and successful independent aggregation with `scripts/summarize.py --allow-subset`. A second aggregation supplied each subset's original annotations and video root to verify exact question-ID coverage. Results and logs are under the ignored local directory `results/verification_20261008/`.

The StreamingBench subset contains two PR questions and the OVO-Bench subset contains two ATR questions. These runs check executable paths only; they do not establish full benchmark accuracy or the paper's ten-task/six-task averages. An initial Qwen3 StreamingBench attempt ran out of memory on a 24 GB RTX 4090; both Qwen3 checks completed on a 48 GB RTX A6000. The other six checks completed on the RTX 4090. LLaVA 0.5B's answer to one OVO sample differed from the September 30 A6000 result, so two-question accuracy should not be treated as stable evidence. The local dataset inputs, model weights, and outputs are not part of the public release.

## Completed Checks

- 13 automated checks passed in the existing Python 3.10 inference environment.
- Lightweight checks run without model weights. Two temporal runtime tests require the inference dependencies and are skipped when those dependencies are unavailable.
- All 18 shipped experiment configurations passed strict configuration validation: eight MEMO main configurations, eight baseline configurations, one smoke configuration, and one profiling configuration.
- Both native benchmark annotation readers passed format checks using the synthetic fixtures.
- The release ZIP was extracted into a separate temporary directory. File checksums, lightweight tests, and annotation validation passed with an empty PYTHONPATH and no parent checkout.
- The dependency doctor passed in the existing reference environment with exactly one visible CUDA device.
- Qwen3-VL-8B-Instruct with CLIP, Grounding-DINO, and SAM2 completed the custom synthetic-video run: three nonempty answers, a completed result, and the expected frame-budget contract.
- The same model combination completed an instrumented StreamingBench-format synthetic run: two nonempty answers. Timing output and JSON/CSV/LaTeX score reports were generated successfully. Full-task coverage was correctly reported as incomplete, with no paper average.
- Runtime regression checks verify that between-frame questions do not see the next frame, transition frames survive chunk ingestion, and final frames are not duplicated between memory and the active chunk.

The GPU runs used one visible NVIDIA RTX A6000, the existing local dependency environment, and local model weights. Selected observed predictions and dependency versions are recorded in `../examples/validation_observed.json`, with private paths omitted. The profiling check configuration is `../examples/validation_profile_config.json`. Both videos/annotation fixtures are synthetic and distributed in this repository.

These checks do not validate full benchmark accuracy, other model backbones, baseline accuracy, a fresh dependency installation, or the manuscript's latency values. No full benchmark run was launched as part of release preparation. Exact output wording can vary across hardware and software versions.

## Commands

```bash
python -m unittest discover -s tests -v
python reproduce.py --help
python reproduce.py --config configs/smoke.json \
  --annotations examples/synthetic/custom.json \
  --video-root examples/synthetic --validate-only
```

The complete GPU smoke command is in the main README. To exercise instrumented processing with the supplied fixture, use `examples/validation_profile_config.json` and `examples/synthetic/streamingbench.csv`, keeping `--max-videos 1 --max-questions-per-video 2` to mark it as a subset run.

Full result files contain local paths and are intentionally not distributed as public benchmark evidence. The sanitized observed-output file records only the synthetic validation scope.
