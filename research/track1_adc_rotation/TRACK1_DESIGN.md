# Track 1 — ADC & Rotation: design, paths, and scope

**Branch:** `track1-adc-rotation` (off upstream `main` @ `6ce7035`).

## 0. Key finding that shapes this track

**The four configurations require no production code change.** ADC and random rotation already
ship as `QFrameBitEncoder` parameters:

- `enable_adc` (default `false`) — `QFrameBitEncoder.java:36-37`
- `random_rotation` (default `false`) — `QFrameBitEncoder.java:38-39`

So the four configs are four **index mappings** over existing code, and "implementing Track 1" =
building a benchmark that drives the real quantizer + scorer, not writing new scoring logic
(satisfying the "reuse existing implementations" requirement).

| # | config | `enable_adc` | `random_rotation` |
|---|---|---|---|
| 1 | baseline | false | false |
| 2 | ADC only | true | false |
| 3 | rotation only | false | true |
| 4 | ADC + rotation | true | true |

## 1. The "clearly document" checklist (code-verified)

**Which indexing path is modified.**
None (code). Index-time quantization runs in
`codec/KNN990Codec/NativeEngines990KnnVectorsWriter.java` → `QuantizationService.train()` →
`OneBitScalarQuantizer.train()` (per segment, 25K reservoir sample). When `random_rotation=true`,
`QuantizerHelper` builds a `d×d` orthonormal matrix (`RandomGaussianRotation`) and applies it to
each vector **before** the per-dimension-mean threshold + bit packing (`OneBitScalarQuantizer.quantize`
→ `BitPacker`). ADC does **not** change indexing. So config selects behavior along the *existing*
QFrame binary indexing path.

**Which search path is modified.**
None (code); ADC selects a different *load + score* path that already exists:
- **No ADC:** binary index; query is binarized; symmetric **Hamming** distance (Faiss binary index).
- **ADC:** `faiss_wrapper.cpp:LoadIndexWithStreamADC` rewrites the binary index into a float
  `IndexHNSW`/`FaissIndexBQ` at load; the query stays FP32 and is transformed by
  `OneBitScalarQuantizer.transformWithADC`; scoring uses the asymmetric LUT
  (`faiss_index_bq.h`) / `KNNScoringUtil.l2SquaredADC` | `innerProductADC`. ADC is **1-bit only**.
- Rotation does not change the search *path*; it only means the query is rotated by the stored
  matrix before binarization (no ADC) or before the ADC transform.
- The rescore path (`NativeEngineKnnVectorQuery` + `RescoreContext`) is **unchanged** by either flag.

**What metadata is stored.**
Per-segment quantization state (`*.osknnqstate`, `KNN990QuantizationStateWriter`):
- always: per-dimension **mean thresholds**; **below/above-threshold means** (used by ADC).
- when rotation on: the **`d×d` rotation matrix** (dense `float[][]`) — **O(d²) extra state per
  segment** (e.g. ~9.4 MB/segment at d=1536). ADC adds **no** extra stored bytes (the below/above
  means already exist); it only changes query handling + load path.

**Whether index compatibility changes.**
Yes, per field, set at index creation:
- **Rotation** changes the *stored codes* (vectors are rotated before binarization), so a
  rotated index is **not** interchangeable with a non-rotated one; toggling requires reindex.
- **ADC** does not change the stored 1-bit codes, but changes the load/score path; it is a
  read-time behavior selected by the mapping. Only new segments adopt a changed mapping.

**Whether the query remains FP32.**
- **ADC configs (2,4): yes** — query kept full precision, transformed (`transformWithADC`), scored
  asymmetrically against 1-bit docs.
- **Non-ADC configs (1,3): no** — query is binarized to 1-bit for symmetric Hamming.

**Whether document vectors remain binary or quantized.**
**Binary (1-bit) in all four configs.** Rotation rotates *before* binarizing (output still 1-bit);
ADC does not change doc storage. Docs are never kept FP32 in the index (FP32 lives only in the
Lucene `.vec` flat file used for optional rescore).

**Whether exact FP32 rescoring is enabled.**
Orthogonal / independent of both flags — controlled by `RescoreContext` (`on_disk` default
oversample, dim-keyed). All four configs can run with or without rescore. **The benchmark sweeps
oversample (no-rescore, 1×, 2×, 3×, 5×) precisely to separate the pre-rescore effect of
ADC/rotation from what survives FP32 reranking.**

## 2. What this branch delivers (tiers)

- **T1 — algorithmic (this commit, measured):**
  `src/test/java/org/opensearch/knn/research/Track1AdcRotationBenchmarkTests.java` drives the
  **real** `OneBitScalarQuantizer` / `RandomGaussianRotation` / `transformWithADC` /
  `KNNScoringUtil.{l2SquaredADC,innerProductADC}` (symmetric = Hamming over packed bits) on
  controlled datasets. Measures **recall@k, ranking agreement (Spearman), quantized-score bias &
  calibrated error variance**, and **rescore depth to reach 0.95 recall** — across
  config × datatype × dim × metric × k × oversample. No native build required.
- **T2 — systems (staged, NOT in this commit):** p50/p95/p99 latency, index size, memory,
  bytes-read during rescore, candidates visited/reranked, cold/warm cache — require the JNI native
  index + a running node + opensearch-benchmark. Commands in §4.

**Scope honesty:** T1 measures *quantization quality* (recall/ordering/error), which is where the
ADC/rotation go/no-go science lives. It does **not** measure latency, index size, or the
graph-traversal interaction with filters — those are T2 and are not claimed here.

## 3. Success / failure definition (fixed BEFORE running)

**Success** (any one, at matched setting):
- ≥ 3 recall@10 points at equal latency/oversample; **or**
- same recall with ≥ 20% lower p99 (T2); **or**
- same recall with ≥ 30% fewer reranked candidates (i.e. lower depth-to-0.95).

**Failure** (any one):
- < 1 recall point improvement with no latency/depth gain; **or**
- > 10% p99 regression (T2); **or**
- material index-size/memory increase without recall benefit (rotation's O(d²) state is the risk).

Every claimed gain is reported with its exact {config, datatype, dim, metric, k, oversample}.

## 4. Reproduce

**T1 (this commit):**
```bash
./gradlew test --tests "org.opensearch.knn.research.Track1AdcRotationBenchmarkTests" \
    -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
# writes research/track1_adc_rotation/results/track1_results.csv
```

**T2 (staged) — the four mappings** (identical except the two flags), e.g.:
```json
{ "type": "knn_vector", "dimension": 1536,
  "method": { "name": "hnsw", "engine": "faiss", "space_type": "l2",
    "parameters": { "encoder": { "name": "binary",
      "parameters": { "bits": 1, "enable_adc": <bool>, "random_rotation": <bool> } } } } }
```
Then index a dataset and run k-NN queries with/without `rescore`, under opensearch-benchmark,
capturing latency percentiles, index size, and bytes-read. (Full T2 runner is future work.)

## 5. Files changed on this branch
- `src/test/java/org/opensearch/knn/research/Track1AdcRotationBenchmarkTests.java` (new, T1 harness)
- `research/track1_adc_rotation/TRACK1_DESIGN.md` (this file)
- `research/track1_adc_rotation/results/track1_results.csv` (generated)
- `research/track1_adc_rotation/FINDINGS.md` (success/failure matrix + recommendation, after the run)

No production Java/JNI files are modified — the four configs are existing parameters.
