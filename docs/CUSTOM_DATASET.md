# Custom Datasets

Use a JSON array with one question per record:

```json
[
  {
    "id": "video1_q1",
    "video": "video1.mp4",
    "timestamp": 12.5,
    "question": "What is the person holding?",
    "options": ["A cup", "A book", "A phone"],
    "answer": "A"
  },
  {
    "id": "video1_q2",
    "video": "video1.mp4",
    "timestamp": null,
    "question": "Describe the main activity."
  }
]
```

`id`, `video`, and `question` must be nonempty strings; IDs must be unique. Paths are relative to `--video-root` or absolute. Timestamps are finite, nonnegative seconds no later than the video duration. Omit the timestamp or use `null` for a question after the full video.

`options` must contain 2–26 nonempty strings, or be omitted/empty for an open-ended question. A multiple-choice reference can be an option letter or zero-based integer index. Open-ended references must be strings. Missing or null references are allowed but excluded from scoring.

Copy `configs/smoke.json` and change the relevant settings for your experiment. Use `benchmark: "custom"` and pass the file with `--config` to `reproduce.py`. Open-ended predictions require a separate dataset-specific evaluator.

Use `reproduce.py` for custom datasets and the benchmark configurations. The runner records provenance and applies independent historical and current frame budgets.
