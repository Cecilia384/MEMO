# EgoProbe evaluation

EgoProbe's frozen evaluation package is supplied separately. Extract it beside `memo/`, then apply [`memo/egoprobe.patch`](../memo/egoprobe.patch) from the extracted directory:

```bash
cd MEMO_EgoProbe_full_evaluation_20261008
patch -p1 -i ../memo/egoprobe.patch
```

The patch connects the evaluator to this repository's `memo/` package, uses the Qwen3-VL-8B MEMO frame budgets (8 historical + 8 current), clears completed chunk tensors, and excludes hidden model cache files from run identity hashing. Set `model.upstream_path` to the absolute path of this repository's `memo/` directory in both evaluation configs. Set paths to Qwen3-VL-8B-Instruct, CLIP ViT-L/14, Grounding-DINO base, and SAM2.1 Hiera Large weights. Use one visible NVIDIA GPU per run.

The package limits its in-memory visual cache to 2,048 frames by default. This is a deployment setting for long EgoLife streams; include its value when reporting results. `model.memo.max_store_frames` can be increased when host RAM permits.

For QaEgo4D, set `media_roots.qaego4d_video` to the directory containing the 688 MP4 files. For EgoLife, set `media_roots.egolife_video` to the directory containing `A1_JAKE/DAY1` through `DAY7`, then build its media index:

```bash
python -m evaluation.egolife_index \
  --video-root <EGOLIFE_VIDEO_ROOT> \
  --overlap-resolutions data/egolife/overlap_resolutions.json \
  --output data/egolife/media_index.json
```

Run `evaluation.run_eval` with each dataset's config, then `evaluation.score_results --run-dir <OUTPUT_DIR>`. The package's `DEPLOYMENT.md` specifies the full commands and resume procedure. The generated media index, checkpoints, predictions, and supplied benchmark data remain outside Git.
