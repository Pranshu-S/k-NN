# Lucene #16029 Follow-up Experiments

**Exact Lucene PR #16092 (Hadamard preconditioning) + #16030 (data-blind SQ) in front of Lucene's real 1-bit BBQ scoring.**

Harness: [`Track2Lucene16092Tests.java`](../../src/test/java/org/opensearch/knn/research/lucene16092/Track2Lucene16092Tests.java) ·
verbatim rotation: [`HadamardRotation.java`](../../src/test/java/org/opensearch/knn/research/lucene16092/HadamardRotation.java) ·
raw CSVs in [`results/`](results/) (`step15_*`, `step16_*`).

---

## 1. Question

1. Does Lucene #16092's **exact** preconditioning reproduce/improve the earlier Fashion-MNIST result (from Track‑2 step 14, which used a home-grown sign-only rotation)?
2. Does #16092's **permutation** (which my old rotation lacked) improve over the old rotation?
3. Does the gain persist with the **HNSW graph topology held fixed**?
4. Does it help an **already-isotropic** control (iid Gaussian 784-D)?
5. How much does **#16030 data-blind** (no centering) quantization hurt candidate quality without preconditioning, and how much does #16092 recover?

Primary metric throughout: **candidate recall@10 BEFORE any fp32 rerank** — i.e. how often the true top‑10 (by exact fp32 L2) appear among the candidates a single BBQ-scored HNSW search collects. This isolates the quality of the quantized representation + scorer for graph traversal, which is the thing preconditioning is supposed to fix.

## 2. Implementations tested (with provenance)

All BBQ scoring is Lucene's **real** production 1-bit path, not an approximation:
`OptimizedScalarQuantizer.scalarQuantize(…, bits=1, centroid)` → `packAsBinary` for the doc; `scalarQuantize(…, bits=4)` → `transposeHalfByte` for the query; `VectorUtil.int4BitDotProduct`; then the exact Lucene104 corrective-term Euclidean formula. This fast path is **proven bit-identical** to the real `Lucene104ScalarQuantizedVectorScorer` (see §3, parity = 0).

| label | what it is | provenance |
|---|---|---|
| `fp32` | exact float L2 (reference / ground truth) | — |
| `bbq_1bit` | Lucene real 1-bit BBQ, centered (centroid = data mean) | Lucene 10.5.0 `OptimizedScalarQuantizer` + `Lucene104ScalarQuantizedVectorScorer` |
| `oldrht_bbq` | my Track‑2 step‑14 rotation (random **sign flips + block-FWHT, NO permutation**) → BBQ | this repo (kept for comparison) |
| `l16092_bbq` | **exact Lucene #16092** rotation (sign flips → Fisher‑Yates permutation → block-diagonal FWHT) → BBQ | verbatim `HadamardRotation.java`, PR #16092 |
| data-blind | BBQ with **centroid = 0** (#16030 `enableCentering=false`) | Lucene #16030 |

**Lucene PR SHAs (fetched 2026-08-18):**
- #16092 "Add hadamard rotation to vector fields" — head **`132f8b17715db53a54608a0bbc8cb75b16be7f87`** (branch `rotation`, updated 2026-08-13).
  - rotation source copied verbatim (sha256 `1044cf4edd1c21909f8c2745b8c39c308b5fe0c1ba6a853ba5aef0fd842a6e7c`); only the package line + one javadoc cross-ref differ from upstream, no executable code changed.
- #16030 "Implement data-blind scalar quantization" — head **`b9c9cc6d516b9c8e3a12026f2d77cc4658d4ce56`** (branch `sq-data-blind`, updated 2026-08-14).
  - Verified semantics: in `Lucene105ScalarQuantizedVectorsWriter`, `enableCentering=false` sets `clusterCenter = new float[dim]` (a **zero vector**); `true` uses the data mean. The Lucene104 and Lucene105 euclidean scorer formulas are byte-identical, so reproducing data-blind as *centroid = 0* through the real scorer is faithful. Dropping the raw float sidecar is a **storage-format** change (see §10) orthogonal to candidate recall.

The exact #16092 transform (from the verbatim source):
```
R(x) = FWHT_blockdiag( Permute( SignFlip(x) ) ) , normalized 1/sqrt(blockSize) per block
blocks = binary decomposition of d   e.g. 784 = 512 + 256 + 16
sign flips  : Random(seed).nextBoolean() per coord
permutation : Fisher-Yates with the SAME Random stream
```
My old rotation is identical **except it omits the permutation step**. So `oldrht_bbq` vs `l16092_bbq` is a clean A/B on the permutation.

## 3. Methodology

- **Datasets:** Fashion-MNIST‑784 (real, ann-benchmarks; 20 000 index vectors, 200 held-out queries) and an **iid Gaussian 784‑D** control (15 000 index, 200 queries, fixed seed). Euclidean.
- **HNSW:** `HnswGraphBuilder` M=16, beamWidth=100, fixed build seed 71 — Lucene's real builder/searcher. Identical for every representation.
- **Candidate sweep ("candidate count" = `TopKnnCollector` k, NOT ef_search):** 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000.
- **candidate recall@10:** fraction of the exact-fp32 top‑10 found among the collected candidates, averaged over the 200 eval queries, **before rerank**.
- **flat recall@10:** same overlap but with an exhaustive (no-graph) scan by the rep's estimated distance — isolates representation quality from graph quality.
- **Distance error:** MAE / RMSE / p95|err| of the estimated vs exact L2 over sampled query–doc pairs, in the rep's own (rotated) space.
- **Rotation seeds:** the two rotated variants are each run over 3 seeds {1234567, 2345678, 3456789}; `fp32`/`bbq` are rotation-independent (1 run).
- **Build vs search vs rerank kept separate (Part 8):** native-graph rows build the graph *with* the rep's own scorer; common-graph rows build ONE fp32 graph and only swap the search scorer; **no fp32 rerank is ever applied** — the reported number is candidate quality.
- **Correctness anchors (both pass):**
  - **BBQ parity** — the fast scoring formula equals the real `Lucene104ScalarQuantizedVectorScorer` (driven via an in-memory `QuantizedByteVectorValues`) with **max relative error 0.000e+00 over 3000 random comparisons** across dims {784,256,128,64,100} and zero + non-zero centroids.
  - **Rotation invariants** — determinism per (dim,seed); L2/dot/distance preservation; `784 = 512+256+16`; the permutation genuinely spreads energy across blocks; invertible; no input mutation. (Mirrors the PR's own `TestHadamardRotation`.)

## 4. Fashion-MNIST results (native graph — build + search both quantized)

Candidate recall@10 before rerank; rotated rows are mean over 3 seeds (k@ targets had **zero seed variance**).

| scorer | flat@10 | k@0.90 | k@0.95 | k@0.97 | k@0.99 | MAE | RMSE | bytes/vec | bits/dim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 1.000 | 10 | 10 | 20 | 20 | 0 | 0 | 3136 | 32.0 |
| bbq_1bit (centered) | 0.515 | 75 | **100** | 150 | 200 | 122.1 | 154.4 | 114 | 1.163 |
| oldrht → bbq | ~0.76 | 20 | **30** | 30 | 50 | 46.9 | 62.6 | 114 | 1.163 |
| **#16092 → bbq** | ~0.77 | 20 | **30** | 30 | 50 | **40.4** | **53.4** | 114 | 1.163 |

- **#16092 reproduces the rotation win: k@0.95 100 → 30 (3.3×)**, k@0.90 75 → 20, k@0.99 200 → 50 — at identical storage (114 B/vec; rotation metadata is one global seed).
- **#16092 ≈ old-RHT on candidate count** (both 30) but its permutation gives a **consistently lower distance error** (MAE 40.4 vs 46.9, RMSE 53.4 vs 62.6, across all 3 seeds) — the permutation improves the estimator but not enough to move the discrete k threshold on this dataset.

## 5. Common FP32 graph (topology fixed, only the search scorer changes)

Isolates the **traversal-scoring** effect from graph-construction. One fp32 graph, searched by each scorer.

| scorer (search) | k@0.90 | k@0.95 | k@0.97 |
|---|---:|---:|---:|
| fp32 | 10 | 10 | 20 |
| bbq_1bit | 50 | **75** | 100 |
| oldrht → bbq | 20 | **30** | 30 |
| #16092 → bbq | 20 | **30** | 30 |

- On a fixed fp32 topology, preconditioning cuts k@0.95 **75 → 30 (2.5×)**. Both rotations equal here.
- **What this proves:** the scoring/traversal benefit persists even when graph topology is held fixed — it is not merely that rotation builds a better graph. **What it does NOT prove:** anything about whether rotation improves graph *construction* (that is the native-vs-common gap: BBQ's own graph needs k@0.95 = 100 vs 75 on the fp32 graph → rotation also fixes a construction penalty, but this table alone does not isolate that causally).

## 6. iid Gaussian control (already isotropic → rotation should be a no-op)

| scorer | native k@0.95 | common-graph k@0.95 | MAE |
|---|---:|---:|---:|
| fp32 | 500 | 500 | 0 |
| bbq_1bit | 750 | 750 | 0.396 |
| oldrht → bbq | 750 | 750 | ~0.41 |
| #16092 → bbq | 750 | 750 | ~0.40 |

- **Rotation changes nothing** on isotropic data (k@0.95 = 750 for all, native and common; MAE within noise). This is the control working: an orthogonal transform cannot help an already rotation-invariant distribution, so the Fashion-MNIST gain is *exploiting structure*, not adding information. (Minor: one #16092 seed missed 0.97 within k≤1000 on the common graph — noise; k@0.95 unaffected.)

## 7. Centered vs data-blind × rotation matrix (Fashion-MNIST)

Candidate recall@10 before rerank. Rotated rows are mean over 3 seeds. Data-blind = #16030 `enableCentering=false` (centroid = 0).

**Native graph:**

| mode | flat@10 | k@0.90 | k@0.95 | k@0.97 | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| A — centered BBQ | 0.515 | 75 | 100 | 150 | 122 | 154 |
| B — data-blind BBQ | 0.580 | 50 | **75** | 100 | 57 | 84 |
| C — #16092 + centered | 0.767 | 20 | **30** | 30 | 40 | 53 |
| D — #16092 + data-blind | 0.650 | 30 | 50 | 75 | 129 | 166 |

**Common fp32 graph** (topology fixed): A k@0.95=75, B k@0.95=75, C k@0.95=**30**, D k@0.95=50.

**iid‑784** (essential k@0.95 comparison): A=B=C=D=**750** — centering and rotation are both no-ops on isotropic data (iid Gaussian mean ≈ 0, so "centered" ≈ "data-blind").

Reading the matrix:

- **SURPRISE (negative for the "centering helps" prior):** *without* rotation, **data-blind BEATS centered** on Fashion-MNIST — k@0.95 75 vs 100, MAE 57 vs 122, flat 0.58 vs 0.52. On this dataset #16030 data-blind does not cost candidate quality; it improves it. (On iid they tie, as expected.)
- **With #16092 rotation, centered becomes the better choice** — C (30) clearly beats D (50). Rotation helps centered (100→30, 3.3×) more than it helps data-blind (75→50, 1.5×).
- **Does #16092 make data-blind competitive with centered?** *No.* With the rotation, centered wins (k@0.95 30 vs 50) and data-blind's distance error stays high (MAE 129 vs 40). So preconditioning does **not** close the data-blind↔centered gap here — it *opens* one in favor of centered, even though data-blind was ahead before rotation. The best overall configuration is **#16092 + centered (k@0.95 = 30)**.

## 8. What the results support

**Strongly supported by these experiments**
- The exact Lucene #16092 transform **reproduces** the Fashion-MNIST candidate-recall improvement of Track‑2 step 14: k@0.95 for 1-bit BBQ drops 100 → 30 (native) / 75 → 30 (fixed fp32 graph).
- The benefit **persists with graph topology held fixed** (§5) — it is a real traversal-scoring effect, not only a graph-construction artifact.
- On an **isotropic control the transform is a no-op** (§6), the signature of a structure-exploiting effect rather than a scoring bug or lucky seed. Zero seed variance on the k@ targets.
- The fast BBQ formula used for the sweeps is **exactly** Lucene's real scorer (parity 0).

**Suggestive but not proven**
- #16092's permutation gives a **modest, consistent distance-error reduction** over the sign-only rotation (MAE 40 vs 47) but did **not** translate into a lower candidate count on these two datasets. Whether the permutation matters for recall likely depends on the data (it should matter more when structure is concentrated in a few original coordinates that a single block would otherwise trap).

**Not measured / out of scope**
- **Latency / QPS.** No wall-clock claim: the sweeps use the proven-equal fast formula, not the production SIMD scorer, and rotation cost (~O(d log d)/vector) was not equal-recall profiled. The claim is candidate-count, not time.
- **On-disk storage** beyond the encoded code+corrective bytes (see §10).
- Larger N, higher dims (768/1536 embeddings), cosine/MIP.

## 9. Surprises / failures / negative results

- **Permutation did not reduce candidate count** on either dataset (k identical to sign-only), despite lowering MAE/RMSE. A clean negative for "does the extra step in #16092 help *recall*."
- **Data-blind BEAT centered without rotation** on Fashion-MNIST (k@0.95 75 vs 100) — the opposite of the "centering helps" prior. But once #16092 rotation is applied, centered overtakes data-blind (30 vs 50). So the centered-vs-data-blind ordering *flips* depending on whether preconditioning is present — a genuine interaction, not a monotone effect.
- **iid control unchanged** — reported rather than summarized as "no regression": the numbers are genuinely flat (750/750), including the direction of the tiny MAE differences.
- **Prior step‑14 storage number was overstated** (see §10 / correctness note) — flagged because step 14's result is posted on #16029.
- Native vs common-graph **disagree in an informative way**: BBQ's self-built graph (k@0.95=100) is worse than BBQ on the fp32 graph (75), i.e. plain BBQ also builds a degraded topology; rotation removes both penalties.

## 10. Storage (actual, not conceptual)

Measured from the real Lucene encoding for 784‑D, `SINGLE_BIT_QUERY_NIBBLE`:
- packed 1‑bit code = `getDocPackedLength(784)` = **98 B** (not 104).
- corrective terms = 4 × int (lowerInterval, upperInterval, additionalCorrection as float-bits; quantizedComponentSum) = **16 B**.
- **total = 114 B/vec** → effective **1.163 bits/dim**. (fp32 = 3136 B.)
- Rotation adds **no per-vector bytes** — one global seed + block config for the whole index.
- **#16030 data-blind** stores the *same* 114 B/vec of quantized code+corrective; its advertised ~4× index-size win comes from **not writing the raw float vectors** (the rerank sidecar), which is orthogonal to the candidate-recall measured here. We do not claim a disk number.

## 11. Correctness note — does anything invalidate the step‑14 post on #16029?

**Recall/ef conclusions: unchanged and now independently confirmed.** The step‑14 BBQ scoring is bit-identical to the real Lucene scorer (parity 0), so its candidate-recall / ef numbers stand. **Storage numbers: corrected.** Step 14 reported 104 B code + 16 B = 120 B/vec (1.224 bits/dim); the real Lucene 784‑D encoding is **98 + 16 = 114 B/vec (1.163 bits/dim)**. Step 14 packed to `discretize(dim,64)=832` (→104 B) instead of the codec's `getDiscreteDimensions=784` (→98 B); the extra 48 bits are zeros that contribute nothing to `int4BitDotProduct`, so scores/ranking are unaffected — only the byte count was overstated. Both compared variants shared the same number, so every *relative* claim (rotation adds no storage; 3.3× fewer candidates) is intact.

## 12. Recommended next experiment

The rotation reproduces and is confirmed real, but a **large low-bit scoring gap remains**: even after #16092, 1-bit BBQ needs k=30 for 0.95 candidate recall vs k=10 for fp32, and flat@10 is ~0.77 vs 1.00. The permutation helps the estimator but not the candidate count. That points at the **estimator**, not the rotation, as the next lever — i.e. a corrected / RaBitQ-style distance estimator (Track‑2 step 13 already showed a RaBitQ-inspired 1-bit reaching ~fp32 candidate quality). Worth a controlled `#16092-rotation + corrected-estimator` run on real 768/1536‑D embeddings, plus an equal-recall latency profile to convert the candidate-count win into wall-clock.
