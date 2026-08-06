# Track 2, Step 7 — Rotated multi-bit representation, consistent graph geometry: results & verdict

**Branch:** `track2-adaptive-rescore`. Real Lucene HNSW graph per representation (21 graphs = 7 reps ×
3 segments), each built AND navigated on its own geometry (fixing the step-6 mismatch). Harness:
[`Track2RotatedMultibitNativeTests.java`](../../src/test/java/org/opensearch/knn/research/Track2RotatedMultibitNativeTests.java).
Raw: [`results/track2_rotmb_{flat,configs,selected}.csv`](results/).
Warm latency only (sandbox); cold/p99 → real hardware.

## Representation (documented honestly)

**OSQ-style rotated scalar quantization — NOT literal RaBitQ.** Optional dense random orthogonal
rotation R (Gaussian + Gram-Schmidt, fixed seed, L2-preserving, identical for base+query); per-dim
uniform scalar quantization to b∈{1,2,4} bits over the (rotated) data's per-dim [min,max]; the code
dequantizes to a reconstruction `x̂`. **One geometry = L2 over `x̂`, used for both graph build
(symmetric `L2(x̂_a,x̂_b)`) and query navigation (asymmetric `L2(rotated_fp32_q, x̂_x)`)** — build and
navigate are now consistent. Ground truth = exact fp32 top-k over original vectors (rotation-invariant).

## Central question — answered YES (via bits + consistency, NOT rotation)

**Native candidate-generation ceiling vs `ef_search`:**

SIFT-128:
| rep | ef20 | ef50 | ef100 | ef200 | ef500 |
|---|---|---|---|---|---|
| 1bit | 0.568 | 0.785 | 0.892 | 0.960 | 0.994 |
| **2bit** | 0.847 | **0.974** | 0.995 | 0.999 | 1.000 |
| **4bit** | **0.970** | 0.998 | 0.999 | 1.000 | 1.000 |
| fp32 | 0.976 | 0.998 | 0.999 | 1.000 | 1.000 |

isotropic-768 / clustered-128 @ef500: **1bit 0.732 / 0.540 → 4bit 0.962 / 0.965 (≈ fp32 0.962 / 0.972).**

**To meet the 0.95 SLA on SIFT: 1-bit needs ef≈200, 2-bit ef≈50, 4-bit ef≈20 — a 4–10× reduction in
graph-search effort.** And the previously-infeasible segments (isotropic, clustered) **become feasible
at 4-bit**, where 1-bit topped out at 0.54–0.73. So yes: more bits, used consistently, raise native
candidate generation enough to hit high-recall SLAs at far lower ef and unlock hard segments.

## Hypotheses

- **H2 (more bits raise graph ceilings): STRONGLY CONFIRMED.** 4-bit ≈ fp32 native ceiling
  everywhere; 2-bit close behind on SIFT. This is the dominant lever.
- **H4 (better geometry → lower ef): CONFIRMED (via bits).** 4-bit reaches the SLA at 4–10× lower ef
  than 1-bit; the graph-work saving is large and real.
- **H3 (geometry consistency matters): CONFIRMED — the key structural fix.** Consistent 2-bit geometry
  hits SIFT ceiling **0.995 @ef100**, vs step-6's *mismatched* corrected-scorer-on-1-bit-graph
  **0.904 @ef500**. And flat quality now TRANSFERS to native (4-bit flat top-k 0.924 → native ceiling
  0.999) — unlike step 6, precisely because build and navigate share one geometry.
- **H5 (better candidates → shallower rerank): CONFIRMED directionally.** With 4-bit's near-fp32
  ordering, the calibrated rerank depth is tiny (≤ 20).
- **H6 (per-segment representation selection): mechanism works** — the harness picks per segment; the
  ceiling data supports 1-bit for easy (SIFT at high ef) and 4-bit for hard (isotropic/clustered).
- **H1 (rotation helps structured data): NOT CONFIRMED — often harmful with this quantizer.** Rotation
  helps 1-bit on SIFT (0.892→0.952 @ef100) and clustered (0.167→0.485), but **catastrophically hurts
  isotropic** (1bit_rot flat 0.009, native 0.033 vs 1bit 0.273) and **hurts 2/4-bit generally**
  (SIFT 2bit 0.995 → 2bit_rot 0.964). A generic orthogonal rotation + generic per-dim scalar quantizer
  is **not** the RaBitQ co-design; rotation alone is not a reliable win and can destroy the geometry.
  → **the lever is BITS + consistency, not rotation.**

## Storage trade-off

Code bytes/vec = `dim·b/8`: SIFT 1-bit 16 B → 4-bit **64 B** (4×, but **8× less than fp32's 512 B**);
isotropic 1-bit 96 B → 4-bit 384 B. Estimated code storage (excl. graph + fp32 rerank):

| bits | 1M × 128-D | 100M × 768-D | 1B × 1536-D |
|---|---|---|---|
| 1-bit | 16 MB | 9.6 GB | 192 GB |
| **4-bit** | **64 MB** | **38 GB** | **768 GB** |
| fp32 | 512 MB | 307 GB | 6 TB |

4-bit costs 4× the 1-bit codes but buys feasibility + 4–10× lower ef (less graph work/latency) and
stays 8× under fp32 — a good trade where the constraint is recall/latency, not code storage.

## Verdict

```
STRONG-GO on multi-bit + consistent geometry; NO-GO on (this) rotation.
Confirmed:
  - More bits, used consistently for build AND navigate, raise the native HNSW candidate ceiling to
    near-fp32 (4-bit) and reach the 0.95 SLA at 4-10x lower ef_search than 1-bit ADC.
  - Consistent geometry fixes step-6's mismatch: 2-bit 0.995@ef100 vs corrected-on-1bit 0.904@ef500;
    flat quality now transfers to the graph.
  - Previously-infeasible high-dim/clustered segments become feasible at 4-bit (~fp32 ceiling).
Not confirmed / negative:
  - Generic random rotation + generic scalar quantizer is NOT a reliable win and is CATASTROPHIC on
    isotropic data. Rotation is not the lever here; bits + geometry consistency are.
  - Under conservative 50-query calibration with margin 0.005, isotropic 4-bit (ceiling 0.962) is
    borderline-uncertified -> needs more cal queries or a smaller margin (honest limit, flagged).
Decision: adopt a consistent multi-bit (2-4 bit) quantized graph; drop the mismatched-corrected-scorer
approach; treat rotation as optional/data-gated, not default. This confirms, natively, the direction
all six prior steps pointed to.
```

## Remaining gap to a full RaBitQ/OSQ

My negative rotation result is a property of *generic* rotation + *generic* scalar quant, not of
RaBitQ. **True RaBitQ co-designs rotation with a normalized 1-bit estimator + provable error bound**,
which is precisely what makes 1-bit work well *with* rotation — potentially matching my 2–4-bit ceilings
at 1-bit storage. Implementing that specific estimator (not a generic scalar quantizer) is the honest
next step to reclaim the rotation upside; the geometry-consistency and multi-bit findings here already
stand on their own.

## Production design note (future OpenSearch/Lucene prototype — not implemented)

Add a consistent multi-bit quantized codec behind one `QuantizedVectorCodec` interface whose
`symmetric_distance` (build) and `query_distance` (navigate) share the same reconstruction geometry;
build the HNSW graph with that codec (not a Hamming graph navigated by a different scorer). Per segment
at flush/merge, calibrate `(bits, ef_search, rerank_depth)` for the SLA (≥50 sample queries; emit
`INCREASE_BIT_WIDTH` / `SLA_UNACHIEVABLE` explicitly), store it in segment metadata, read at query time,
with user override + conservative fallback. Keep fp32 for exact rerank. Rotation stays behind a flag,
off by default until a RaBitQ-grade estimator replaces the generic scalar quantizer.

## Reproduce
```bash
./gradlew :test --tests "org.opensearch.knn.research.Track2RotatedMultibitNativeTests" \
    -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
```
