# Dataset Preparation

Use the same dataset release and split as the experiment being reproduced. This repository does not distribute benchmark annotations or videos and does not silently download them.

## StreamingBench Real-Time

Pass the native `Real_Time_Visual_Understanding.csv` file. Required columns:

| Column | Format |
| --- | --- |
| `question_id` | Unique question ID containing `sample_<id>` when using the standard naming convention |
| `task_type` | Task code or full task name |
| `question` | Question text |
| `time_stamp` | Seconds, `MM:SS`, or `HH:MM:SS` |
| `answer` | Option letter |
| `options` | Python/JSON-style list of strings, such as `['A. Red', 'B. Blue']` |

Supported task order: OP, CR, CS, ATP, EU, TR, PR, SU, ACP, CT. Full names are defined in `benchmarks/data.py`, including the dataset spelling `Clips Summarize`.

The standard video mapping is `sample_<id>_real.mp4` under `--video-root`. An optional `video` CSV column overrides this mapping for repackaged data. Option strings are passed to the prompt unchanged, so retain their original labels. The loader uses `ast.literal_eval`, never executable `eval`.

The table's average is micro accuracy across all questions. A complete report requires every task and all selected annotation IDs. Full benchmark size and dataset revision must be recorded from the actual annotation file; they are not guessed from the manuscript.

## OVO-Bench Real-Time

Pass the native JSON array (the local development scripts call it `ovo_bench_new.json`). Each selected record requires:

```json
{
  "id": "example_1",
  "task": "ATR",
  "video": "relative/path/to/video.mp4",
  "realtime": 4.5,
  "question": "What color is the object?",
  "options": ["Red", "Blue"],
  "gt": 1
}
```

`gt` is a zero-based answer index. Video paths are resolved against `--video-root`. The runner selects only OCR, ACR, ATR, STU, FPD, and OJR. Backward and forward tasks are excluded because they do not appear in the supplied main table. Options are lettered by the OVO prompt builder.

The table average is the unweighted mean of the six task accuracies. It is not accuracy over all OVO tasks and is not necessarily micro accuracy over the six-task subset.

## Validation and Subsets

Append `--validate-only` to an evaluation command to inspect the selected counts without loading models. Missing videos, duplicate IDs, invalid answer indices, unknown selected task names, and malformed timestamps fail visibly. Decoding and comparison with the actual video duration happen during inference.

Use `--tasks ATP` or `--max-videos 1 --max-questions-per-video 2` only for diagnostics. Subset selections are recorded in the result and cannot be silently summarized as full runs.

The included `examples/synthetic/streamingbench.csv` and `ovobench.json` are original synthetic format examples, not samples taken from those benchmarks.
