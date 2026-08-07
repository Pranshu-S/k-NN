# Track 2, Step 9 — Optimized packed scoring kernels + fair equal-recall re-benchmark

**Branch:** `track2-adaptive-rescore`. Harness:
[`Track2OptimizedKernelTests.java`](../../src/test/java/org/opensearch/knn/research/Track2OptimizedKernelTests.java).
Raw: [`results/track2_kernel_{micro,native_agg,threads,selected,pareto}_arm64-jit_seed71.csv`](results/).
This step closes the honest gap flagged in step 8: the 1/2/4-bit numbers there used a **scalar
unpack-and-accumulate** scorer, so warm latency understated multi-bit and overstated fp32. Here we
implement the **exact optimized kernel** for this representation and re-run the equal-recall
comparison fairly.

## Environment (item 23)

Apple **M5 Pro**, 15 cores (no SMT), 24 GB, **arm64 / NEON** (no AVX2/AVX-512 on host), L1d 64 KB,
L2 8 MB, no exposed L3/SLC. JDK 21 Temurin, HotSpot C2. N=6000 vectors/segment, K=10, HNSW M=16
beam=100 seed=71, NCAL=50 (calibration) + NEVAL=200 (held-out eval, disjoint). Latency is
**warm** (JIT-hot, resident) and **not CPU-pinned / not page-cache-evicted** — cycles are estimates
at a nominal 4.4 GHz (labeled `_est`). Each warm percentile is the best of 4 repetitions to suppress
scheduler jitter; threaded QPS uses 2400 queries recycled across a shared cursor.

## 1. What "optimized kernel" means here (and why not popcount)

Representation is **unchanged** (per constraints): per-dim uniform scalar quant to `2^b` levels,
reconstructed to `x̂`, scored with **L2**. `asymL2(q,code) = Σ_d (q_d − recon(d, lvl_d))²`.

**Optimized kernel = query-side ADC lookup table.** Once per query, precompute
`lut[d·L + l] = (q_d − recon(d, l))²` for every dim `d`, level `l` (`L=2^b`). Per-candidate score
becomes `Σ_d lut[d·L + lvl_d]` — the hot loop is **unpack-level + table-load + add**, with **no**
per-dim float reconstruction (`mn + l/(L−1)·rng`) or multiply. Summed ascending-`d` ⇒ **bit-exact**
to the scalar reference. Byte-blocked unpack: 1-bit = 8 dims/byte, 2-bit = 4, 4-bit = 2, plus a
per-dim tail loop for non-block-multiple dims. Symmetric (build) scoring is optimized with a per-dim
per-level **reconstruction table** (`recon[d·L+l]`), also bit-exact.

**Why not XOR+popcount for 1-bit:** writing the L2 score as `C_q + Σ_{set bits} δ_d` needs
`δ_d = (q_d−hi_d)² − (q_d−lo_d)²`, which **varies per dim** (per-dim min/max) — so a single
`Long.bitCount` cannot collapse it. Popcount only collapses under a **constant-δ** (normalized/rotated
RaBitQ) code, a *different* representation (the step-7 rotation branch). The task explicitly forbids
substituting Hamming when not mathematically equivalent, so 1-bit uses the exact LUT. The plugin's own
popcount (`KNNScoringUtil.hamming`, `Long.bitCount` → arm64 `CNT`) is the right primitive **only** for
that other representation.

**Dispatch / ISA (items 2, 8):** kernels are labeled `scalar` / `lut` and ISA `arm64-jit`; the scalar
path is retained as reference + portable fallback (never removed). No AVX2/AVX-512 exists on this host
and the plugin's production SIMD lives in C++ JNI/FAISS (out of scope). A C++ AVX2/NEON port is the
production step — captured in the design note, **not** implemented; we do **not** report scalar-fallback
latency as the production result, nor claim SIMD we didn't run.

## 2. Correctness (item 11) — PASS

**4032 asymmetric + 2880 symmetric bit-exact checks** (`optimized == scalar`, tolerance 0), across
dims {128, 384, 768, 1536, 130, 100, 7, 1} (incl. odd + non-block-multiple + tail), 4 seeds, random
vectors **and** adversarial all-zero + all-max-symbol codes. Because scores are bit-identical, HNSW
candidate ordering, candidate ceiling, and recall are **provably unchanged** — the optimization is
pure latency, not a quality change.

## 3. Scoring microbench — optimized kernel is real, but does not beat fp32 per-score (items 9, 12)

`ns/score` (best of hot/large × seq/random shown; full grid in CSV). Accumulator: fp32 sum, dim ≤ 1536,
per-dim contribution bounded by `rng²` — 32-bit float accumulation is safe (no integer LUT / no overflow).

| dim | 1bit scalar→lut | 2bit scalar→lut | 4bit scalar→lut | fp32 | bytes/score (1/2/4/fp32) |
|---|---|---|---|---|---|
| 128 (hot, seq) | 575 → **333** (1.7×) | 184 → **77** (2.4×) | 184 → **80** (2.3×) | **68** | 16 / 32 / 64 / 512 |
| 128 (random, LLC-pressure) | 273 → **171** | 267 → **173** | 272 → **178** | **158** | — |
| 768 (hot, seq) | 1102 → **609** (1.8×) | 1105 → **605** (1.8×) | 1106 → **609** (1.8×) | **547** | 96 / 192 / 384 / 3072 |

- **The LUT kernel is a genuine ~1.5–2.4× speedup over scalar quantized** — the honest gap from step 8
  is closed.
- **But fp32 per-score is still ≤ optimized-quantized per-score** (128-D: fp32 68 ns vs 2bit_lut 77 ns
  hot; 158 vs 173 under LLC-pressure). Reason: fp32 L2 **autovectorizes** (NEON 4-wide FMA); the LUT is
  a **scalar gather** that does not vectorize in Java. So even optimized, packed compute doesn't win the
  per-score race on this ISA in a JVM.
- **Query-prep is amortized (item 6):** LUT build is 1–10 µs (grows with dim·2^b: 4-bit/768-D ≈ 8–10 µs);
  over the hundreds–thousands of nodes an HNSW query visits it is < 1% of query time. It is reported
  separately and never hidden inside score throughput.
- **Large-working-set (48 MB ≫ L2) + random access (item 16):** quantized degrades less than fp32
  relatively (fp32 128-D 57→158 ns seq→random), but fp32's absolute per-score stays lowest — at this
  scale 512 B/vec still streams fine on M5's bandwidth. The bytes/score advantage (8–32×) does **not**
  convert to a compute win here.

## 4. Native equal-recall on SIFT-128 — the product answer (items 13, 22)

All configs re-calibrated conservatively (held-out recall ≥ 0.95); optimized LUT navigation:

| rep | ef | rerank depth | held-out recall | mean nodes | bytes read/query | p50 | p95 | p99 | QPS(1T) | idx B/vec | build ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1-bit** | 500 | 300 | 0.984 | 2047 | 32,754 | 394.6 | 447.5 | 485.5 | 2,545 | 16 | 2784 |
| **2-bit** | 50 | 50 | 0.976 | 583 | **18,655** | 86.7 | 102.0 | 117.2 | 12,023 | 32 | 2833 |
| **4-bit** | 50 | 20 | 0.998 | 537 | 34,359 | **77.2** | 96.3 | 110.0 | **13,157** | 64 | 2691 |
| **fp32** | 20 | 10 | 0.972 | 297 | 152,078 | **31.6** | 39.8 | 43.8 | **31,902** | 512 | 665 |

- **Optimized 2/4-bit beat 1-bit decisively at equal recall: ~5× lower p50/p95/p99 and ~5× higher QPS.**
  The "1-bit as default" idea is refuted — with a real kernel, 2-bit and 4-bit dominate 1-bit on every
  latency axis at ≤ 2× storage. (Note 1-bit here re-calibrated to ef=500/depth=300: conservative
  certification over the held-out split needed wider ef than the step-8 anchor, which makes 1-bit look
  even worse — an honest consequence of not tuning on the eval set.)
- **fp32 still wins warm single-query latency and single-thread QPS** (31.6 µs, 31.9K QPS) — because
  (a) optimized packed compute can't beat autovectorized fp32 (§3), and (b) at N=6000 the **entire fp32
  index is ~3 MB < 8 MB L2**, so its 7–8× larger bytes-read never bottlenecks. This is the **decisive
  scale caveat**: the memory-traffic advantage that should favor multi-bit needs a working set ≫ cache,
  which this segment size cannot create.
- **2-bit is the bytes-read champion** (18.7 KB/query, lowest of all reps) and the storage-efficient
  default; 4-bit is fastest quantized (best candidate quality → shallow rerank) at 2× 2-bit's storage.

## 5. Threaded throughput (item 18) — no bandwidth crossover at this scale

QPS @ {1,2,4,8} threads / scaling efficiency (SIFT):

| rep | 1T | 2T | 4T | 8T | eff@8T |
|---|---|---|---|---|---|
| 1-bit | 2,502 | 4,344 | 6,406 | 8,480 | 0.42 |
| 2-bit | 12,065 | 23,305 | 38,960 | 57,570 | 0.60 |
| 4-bit | 13,470 | 25,940 | 45,316 | 58,268 | 0.55 |
| **fp32** | 31,778 | 58,463 | 98,649 | **132,355** | 0.52 |

fp32 scales **as well or better** than quantized to 8 threads and stays the QPS leader — its higher
bytes-read does **not** saturate memory bandwidth here (3 MB index, M5's large bandwidth + SLC). So the
hypothesized "fp32 saturates first under concurrency" **did not occur at N=6000** — honestly, the
sandbox scale is too small to force the crossover. All reps keep p99 bounded under load (no collapse).

## 6. Hard high-dim segments (item 14) — where multi-bit actually wins

| segment | rep | feasible? | held-out recall | p50 | p99 | idx B/vec | bytes read/query |
|---|---|---|---|---|---|---|---|
| isotropic-768 | 1-bit / 2-bit | **INFEASIBLE** (ceiling < 0.95 @ ef≤500) | — | — | — | 96 / 192 | — |
| isotropic-768 | **4-bit** | ✅ | 0.979 | 2601 | 2677 | 384 | **1.44 MB** |
| isotropic-768 | fp32 | ✅ | 0.990 | 2589 | 2824 | 3072 | **11.6 MB** |
| clustered-768 | 1-bit / 2-bit | **INFEASIBLE** | — | — | — | 96 / 192 | — |
| clustered-768 | **4-bit** | ✅ | 0.985 | 1055 | 1429 | 384 | **0.52 MB** |
| clustered-768 | fp32 | **INFEASIBLE** (ceiling < 0.95 @ ef≤500) | — | — | — | 3072 | — |

- **1-bit and 2-bit remain infeasible** on both hard segments — 4-bit is the **only** quantized option
  that reaches the SLA, confirming step 7/8 natively with the fair kernel.
- **isotropic-768: 4-bit ≈ fp32 latency (2601 vs 2589 µs p50) but reads 8× fewer bytes/query
  (1.44 MB vs 11.6 MB) at 8× less storage.** Here the memory-traffic gap is large in absolute terms;
  under real DRAM/page-cache pressure (not reproducible in-sandbox) 4-bit should pull ahead. Under the
  p99 and memory-aware objectives the calibrator **selects 4-bit over fp32** on isotropic.
- **clustered-768: fp32 HNSW ceiling stayed < 0.95 while 4-bit reached 0.985** — a synthetic-pathology
  result (32 well-separated clusters make the fp32 graph under-explore; mild quantization noise aids
  traversal). Reported as observed and **flagged for real-hardware verification** — not over-claimed as
  a general "quantization beats fp32 recall" effect.

## 7. Build-time (item 19) — optimized symmetric scoring barely helps; build stays costly

Optimized recon-table symmetric scorer vs scalar during construction (SIFT, ms):
1-bit 2891→2784, 2-bit 2881→2833, 4-bit 2699→2691 — **marginal** (~2–4%). Graph construction is
dominated by neighbor-selection bookkeeping, not reconstruction arithmetic, so the recon-table doesn't
move it much. **Quantized build stays ~4× slower than fp32** (665 ms) — build cost remains a real
penalty (a genuine C++ SIMD symmetric kernel would help more than this Java recon-table did).

## 8. Per-segment selection & Pareto (items 20, 21)

Calibrator (target 0.95, margin 0.005, 50 cal queries) picks per **objective**:

| segment | min-mean-latency | min-p99-latency | memory-aware (λ_store=0.02, λ_bw=0.001) |
|---|---|---|---|
| sift_real | **fp32** | **fp32** | **2-bit** |
| isotropic-768 | fp32 (barely: 2588 vs 2601) | **4-bit** | **4-bit** |
| clustered-768 | **4-bit** (fp32 infeasible) | **4-bit** | **4-bit** |

**The chosen representation is objective- and segment-dependent** — exactly the honest story:
fp32 wins pure warm latency at cache-resident scale; 2-bit wins storage/bytes on easy segments; 4-bit
wins tail latency, memory-aware cost, and is mandatory for feasibility on hard/high-dim segments. All
four reps are **Pareto-optimal on SIFT** (fp32=latency, 1-bit=storage, 2-bit=bytes-read, 4-bit=recall)
— no single universal winner, and we don't present one weighting as canonical.

## Verdict

```
PARTIAL-GO. Optimized kernels turn the multi-bit graph-quality win into a real latency win OVER 1-BIT,
and make 4-bit practical on hard segments, but do NOT overturn fp32 at cache-resident single-node scale.

STRONG evidence (kernel-fair, this run):
  - Optimized LUT is bit-exact and ~1.5-2.4x faster than scalar quantized; recall/ordering unchanged.
  - 2-bit AND 4-bit beat 1-bit at equal recall by ~5x in p50, p95, p99 AND single-thread QPS.
    -> 1-bit is not the default; 2-bit is the storage/bytes-efficient default, 4-bit the fast/feasible
       escalation. (matches the established design direction)
  - 4-bit is the ONLY feasible quantized option on isotropic-768 / clustered-768; there it matches fp32
    latency at 8x less storage and 8x less bytes-read, and is auto-selected under p99 + memory-aware
    objectives.

Why NOT strong-go:
  - fp32 still wins warm single-query latency (31.6us) and 1T/8T QPS on SIFT. Two honest reasons:
    (a) autovectorized fp32 FMA beats a scalar-gather LUT on arm64/JVM (a C++ SIMD/byte-shuffle kernel
        is the remaining lever, not implemented here); and
    (b) at N=6000 the fp32 index (3MB) fits L2, so its 7-8x bytes-read never bottlenecks -> the
        memory-traffic advantage of multi-bit has nothing to bite on. The bandwidth/concurrency
        crossover the design predicts requires a working set >> cache (100M x 768 fp32 = ~300GB),
        which the sandbox cannot create. UNPROVEN here, not refuted.
  - Build cost stays ~4x fp32; the Java recon-table barely helped (real gain needs a C++ symmetric SIMD
    kernel).
  - clustered-768 "fp32 infeasible" is a synthetic pathology -> verify on real hardware.

Decision: keep the established direction. Adopt optimized packed kernels (LUT now; C++ SIMD byte-shuffle
next) so 2-bit/4-bit decisively beat 1-bit. Default 2-bit on feasible segments; escalate to 4-bit where
2-bit's ceiling is short or memory dominates; keep fp32 for exact rerank AND as the latency choice on
small cache-resident compute-hot segments. The multi-bit-vs-fp32 memory win is real in bytes (proven)
but needs a large-index, cache-pressured, real-hardware run to show up in latency (deferred).
```

## Decision-gate mapping (item 25)

Lands in **Partial go**, precisely: *"Multi-bit beats 1-bit but not fp32 in warm single-query latency"*
+ *"Four-bit is required for feasibility"* + *"Build cost remains high."* Not strong-go because the
*"cache-pressure advantage over fp32"* gate is **unproven at this scale** (not failed — untestable in
sandbox). Not no-go because multi-bit convincingly **beats 1-bit** and throughput **does** improve over
1-bit.

## Minimal OpenSearch/Lucene kernel-integration design note (item 19.19 — NOT implemented)

- **Packed code format:** per-segment `bits ∈ {1,2,4}` + per-dim `(min,max)`; codes packed 8/4/2
  dims/byte, documented little-endian shift order, final byte's padding bits forced 0 and **excluded**
  from scoring (safe for arbitrary dim; tail loop handles non-block-multiple dims).
- **Distance-computer dispatch:** one `PackedQuantizedDistance` interface exposing `symmetric(a,b)`
  (build) and `asymmetricLUT(queryLUT, code)` (query). The **same** scorer drives graph construction,
  upper-level navigation, level-0 traversal, and candidate ordering — no scorer mismatch (the step-6
  bug). Query builds the ADC LUT once; navigation reuses it.
- **CPU feature selection:** runtime dispatch via the plugin's existing native feature detection —
  x86 **AVX2 byte-shuffle / gather LUT** (and AVX-512 VPOPCNTDQ **only** for a future constant-δ
  RaBitQ 1-bit), **arm64 NEON** table lookup (`TBL`) — with the **scalar LUT as guaranteed fallback**.
  No illegal instructions on unsupported hardware; identical results within float tolerance.
- **Segment bit-width metadata:** store `(bits, ef_search, rerank_depth, status)` chosen at flush/merge
  by the conservative calibrator (≥50 sample queries; escalate 1→2→4-bit pilots on a sample; emit
  `CALIBRATED_{2,4}BIT` / `SLA_UNACHIEVABLE` / `USE_FP32`). Objective is deployment-configurable
  (mean-latency / p99 / memory-aware).
- **Scalar fallback:** always present (reference + portability); selected when no SIMD path matches.
- **Exact fp32 rerank:** retain fp32 originals; rerank the calibrated depth; return exact top-k. fp32
  may also be the *primary* representation on tiny/compute-hot cache-resident segments where the
  calibrator finds it cheapest.
- **Backward compatibility:** new codec behind a format version tag; existing fp32/1-bit indexes read
  unchanged; bit-width is per-segment so mixed-width shards interoperate.

## Reproduce
```bash
./gradlew :test --tests "org.opensearch.knn.research.Track2OptimizedKernelTests" \
    -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
# testKernelCorrectness  -> bit-exact assertions
# testOptimizedBenchmark -> results/track2_kernel_{micro,native_agg,threads,selected,pareto}_arm64-jit_seed71.csv
```
Hardware note: warm/indicative only. The decisive **large-index, cache-pressured, real-hardware**
run (fp32 working set ≫ LLC; pinned cores; dropped page cache; a C++ SIMD byte-shuffle kernel) is the
next step to test the multi-bit-vs-fp32 memory-traffic crossover — not doable in this sandbox.
```
