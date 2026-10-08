# Protocol and Provenance

## What This Release Reproduces

This release makes the manuscript's MEMO settings executable and records enough information to audit a new run. It is a reconstruction from the supplied working-tree implementation and manuscript, not a recovery of an immutable original experiment snapshot. No full-table result is claimed to have been regenerated.

The runner shares perception, similarity, segmentation, ingestion, retrieval, and model-adapter code with the supplied implementation. Dataset adapters, independent frame budgets, configuration validation, result provenance, and the temporal evaluation loop were organized for this release. Original working-tree source hashes are listed in `SOURCE_MANIFEST.json`; no clean upstream commit was available for the exported minimal implementation.

## Main Experiment Settings

| Setting | Release value | Source or interpretation |
| --- | --- | --- |
| Sampling | 1 FPS, no manual resize, no prefilter | Manuscript |
| Retrieval | Top 3 chunks, presented chronologically | Manuscript and supplied retriever |
| Historical/current frames | At most 8 / 8, independently sampled | Manuscript; total at most 16 |
| Chunking spatial/local/global weights | 0.35 / 0.45 / 0.20 | Manuscript |
| Retrieval global/local weights | 0.6 / 0.4 | Manuscript |
| Detection box/text thresholds | 0.35 / 0.25 | Manuscript |
| Association IoU | 0.3, fixed in tracker implementation | Manuscript and source |
| Entity EMA | 0.3 | Manuscript |
| Spatial IoU/displacement balance | 0.6 / 0.4 | Manuscript |
| Maximum normalized displacement | 0.5 | Manuscript |
| Hidden-frame budget | 5 | Manuscript |
| Threshold window / multiplier | 30 / 1.3 | Manuscript |
| Adaptive threshold clamp | 0.1–0.9, fixed in implementation | Manuscript |
| Initial threshold | 0.45 | Supplied source; not specified in the implementation paragraph |
| Boundary confirmation / minimum chunk length | 1 / 4 frames | Manuscript |
| New/disappeared-object penalties | 0.4 / 0.2 | Supplied source; numerical values absent from the manuscript |
| Random seed | 42 | Explicit release setting |
| Generation | Greedy, maximum 64 new tokens | Explicit release setting; original scripts disagree on token limits |
| Detection category prompt | `person . car . animal . object .` | Existing StreamingBench evaluator; original OVO default differs |
| Storage | GPU indices, CPU images; image-cache threshold 1,000,000 frames | Release setting to reduce eviction; not an unlimited-history guarantee |
| Model/data revisions | Must be provided and recorded by the experimenter | Exact original revisions were not supplied |

JSON files contain complete configurations. Unknown keys, invalid types, nonfinite values, and invalid weights are rejected. Dataset/model paths are command-line arguments rather than private paths embedded in configurations.

## Confirmed Differences and Open Questions

| Topic | Manuscript | Supplied development code | Release behavior |
| --- | --- | --- | --- |
| Frame budget | Independent 8+8 caps | Minimal and current StreamingBench paths cap the combined evidence at 8 | Independent `history_frames` and `current_frames`, both 8 |
| Spatial score with no matched objects | Zero | Similarity code returns 0.3 in this case | Configurable `unmatched_spatial`, set to 0.0 |
| Query timing | No future observations | The minimal export already contains temporal fixes beyond older benchmark loops | Answer between-sample questions before processing the next frame; store completed chunks before answering at the current frame timestamp |
| Final evidence | Completed chunks plus current memory | Historical scripts have varied tail-frame handling | Final chunk is ingested once; it is not duplicated as current evidence |
| Visual representation reuse | Reuse previously encoded backbone-specific evidence | Included adapters build image inputs and invoke the model vision encoder for every question; stored CLIP patches are not backbone embeddings | No cross-question backbone feature-cache claim |
| Entity visual features | Describes masked object inputs | Default perception uses CLIP patch pooling under object masks | Retains supplied patch-pooling implementation |
| Efficiency | Historical A6000 table and figure | Warmup, output lengths, caching, concurrency, and exact evaluation subset are not fixed | Separate synchronized profiling configuration; do not equate measurements with historical targets |

To claim exact table reproduction, resolve these differences against the final original run commands, code revision, model revisions, annotation hashes, prompts, frame selection, and per-question predictions. Do not tune settings against the test set merely to force agreement with the reported averages.

## Table and Figure Coverage

| Manuscript item | Included support | Remaining qualification |
| --- | --- | --- |
| `tab:benchmark-results`, MEMO rows | Eight main configurations; both native data readers; per-task CSV/LaTeX export | Full evaluation not rerun; original revisions unresolved |
| `tab:benchmark-results`, vanilla rows | Four baseline backbones on both datasets | Prefix/truncation policies are explicit interpretations of the frame column |
| `tab:latency_breakdown` | Per-module, ingestion, retrieval, generation, TTFT/TPOT instrumentation | Synchronized release protocol differs from original timing context |
| Efficiency comparison figure | MEMO timing and memory collection | Competitor implementations and original environments are not included |

Reported target values are transcribed into `configs/paper_targets.json`. Missing figure values are deliberately not inferred. Targets are never mixed with measured output.

## Scoring

`strict` scoring accepts a single option letter in a small set of formats and rejects prose or ambiguous answers. Unparseable predictions with a reference answer count as incorrect.

`legacy` reproduces the simple policies found in the supplied local scoring paths: StreamingBench compares an uppercased stripped response with the reference letter or accepts it as a prefix; OVO checks whether the reference letter occurs anywhere in the raw response. OVO's rule is case-sensitive and permissive. These are local compatibility policies, not a claim that this package is the official benchmark evaluator.

StreamingBench reports micro accuracy across questions. OVO reports the unweighted mean of the six real-time task accuracies. Reports additionally expose micro accuracy, available-task macro accuracy, and per-task denominators. A paper-average field is withheld if any expected task is missing. For full-run reporting, supply annotations to the report command so that exact result IDs are checked, not just aggregate counts.

## Timing

Profiling is opt-in. Accuracy configurations do not enable synchronization instrumentation. In profiling mode, CLIP, Grounding-DINO, and SAM2 are timed separately with synchronized boundaries and serialized calls. Module means include all calls. Ingestion is reported per completed chunk, not automatically relabeled as per-frame update cost.

Query timing separates retrieval/evidence selection, input preprocessing, generation, and the complete response through decoded text. TTFT starts at generation invocation, and TPOT uses subsequent synchronized logits-processor timestamps. These timestamps are instrumentation points, not network delivery times. Single-token answers have no TPOT. The report can exclude initial questions as warmup; it does not repeat every question for multiple formal runs. Peak allocated memory covers the full inference run, including any warmup questions, but excludes loading transients before the peak counter is reset.

## Publication Readiness

Before a public release: confirm reused-code attribution, add final ACM Multimedia proceedings metadata when available, confirm final experiment protocols, and validate installation in a fresh environment. The project uses the MIT License, and the arXiv DOI is recorded in `CITATION.cff`. This is a local release candidate; no remote repository has been created or published.
