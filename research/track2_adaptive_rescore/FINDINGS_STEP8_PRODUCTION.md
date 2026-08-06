# Track 2, Step 8 — Production-shaped multi-bit HNSW: end-to-end verdict

**Branch:** `track2-adaptive-rescore`. Real Lucene HNSW graphs over **real packed 1/2/4-bit codes**
(8/4/2 dims per byte) + fp32 reference, consistent geometry (L2 over per-dim reconstruction, build
AND navigate). Harness:
[`Track2ProductionMultibitTests.java`](../../src/test/java/org/opensearch/knn/research/Track2ProductionMultibitTests.java).
Raw: [`results/track2_prod_{micro,configs,selected}.csv`](results/). **Warm latency only** — sandbox
can't pin CPUs / evict cache / drive threaded load, so cold-cache, throughput-scaling, and
p99-under-load are deferred to real hardware. Rotation OFF (step-7 result). Scoring uses a **scalar
unpack-and-accumulate fallback** — no SIMD/LUT (see caveat).

## The memory-bandwidth crux (item 20) — REFUTED; multi-bit wins at equal recall

The central worry: does 4-bit's larger code (4× bytes/node) erase its lower-`ef` benefit? Equal-recall
(0.95 SLA) on SIFT-128 — the cheapest config each bit width needs, with real bytes-read = nodes×codeB:

| representation | ef to hit 0.95 | nodes visited | code B/vec | **bytes read/query** | warm p50 µs |
|---|---|---|---|---|---|
| **1-bit** | 200 | 1281 | 16 | **20,493** | 366 |
| **2-bit** | 50 | 555 | 32 | **17,746** | 148 |
| **4-bit** | 20 | 283 | 64 | **18,107** | 68 |
| fp32 | 20 | 279 | 512 | 142,923 | 35 |

**At equal recall, 2-bit and 4-bit read the SAME-or-FEWER bytes than 1-bit** (17.7K / 18.1K vs 20.5K)
— the ~4–10× fewer nodes visited more than offsets the 2–4× larger code. **The larger code does NOT
increase memory traffic; the crux worry is refuted.** And **2-bit strictly dominates 1-bit** at equal
recall (−13% bytes, −60% warm p50, only 2× storage). fp32 is warm-fastest but reads **7× more bytes**
and needs **8× more storage** — it loses badly on the cache/disk-relevant metric.

## Feasibility on hard segments (native confirmation of step 7)

isotropic-768 / clustered-768: **only 4-bit meets the 0.95 SLA** (1-bit/2-bit ceilings stay < 0.95 at
ef≤500). Selected 4-bit ef=500 → nodes 3741 / 1341, bytes 1.44 MB / 515 KB, warm p50 4674 / 1794 µs —
expensive, but the *only* feasible quantized option. Multi-bit unlocks previously-infeasible segments.

## Scoring microbenchmark — and the honest kernel gap

ns/score (scalar unpack), representative dims:

| dim | 1-bit | 2-bit | 4-bit | fp32 |
|---|---|---|---|---|
| 128 | 323 | 208 | 210 | **137** |
| 768 | 1153 | 1129 | 1117 | **547** |

**Packed scoring is SLOWER per-score than fp32** here, because I implemented only the **scalar
unpack-and-accumulate fallback** — every dim is shift-decoded to a float and squared, with no SIMD or
LUT. The real quantized speed advantage requires the optimized kernels the task calls for (item 4/21):
**1-bit = XOR+popcount Hamming over d/8 bytes; 2/4-bit = byte-shuffle / precomputed-query LUT (ADC)**.
Those are significant native work and are NOT implemented here. **Consequence:** the warm-latency
numbers understate multi-bit's per-score potential and overstate fp32's (fp32 skips unpack); the
node-count and bytes-read reductions are what carry the multi-bit win, and those are kernel-independent.

## Storage & indexing

Code bytes/vec = `dim·b/8`: SIFT 1/2/4-bit = 16/32/64 B (fp32 512); 768-D = 96/192/384 B (fp32 3072).
4-bit is 4× the 1-bit codes, **8× under fp32**. Totals (codes only): **4-bit 1M×128 = 64 MB, 100M×768
= 38 GB, 1B×1536 = 768 GB** (vs fp32 6 TB). Build time (scalar-scored, inflated): packed ~4–6× slower
than fp32 build (isotropic 1-bit 45 s vs fp32 7.5 s; 4-bit 27.7 s) — again an artifact of scalar
scoring during construction; optimized kernels would cut this sharply.

## Per-segment selection

The calibrator selected **4-bit on every segment** under the warm-p50 objective (4-bit's near-fp32
ceiling lets it use the lowest ef + shallow rerank). H6 ("easy→1-bit, hard→4-bit") only partly
materialized: under a **bytes/storage objective, 2-bit wins SIFT** (17.7 KB & 32 B/vec vs 4-bit's
18.1 KB & 64 B) — so the "right" per-segment bit width is objective-dependent, and 2-bit is the
storage-efficient sweet spot on easy data while 4-bit is mandatory on hard data. Statuses emitted:
`CALIBRATED_2BIT`/`CALIBRATED_4BIT`/`SLA_UNACHIEVABLE`.

## Verdict

```
PARTIAL-GO, leaning strong — multi-bit is the right native direction; latency needs optimized kernels.
Confirmed (architecture-independent, honest):
  - Memory-bandwidth worry REFUTED: at equal 0.95 recall, 2-bit/4-bit read <= 1-bit's bytes (fewer
    nodes offset larger codes) and 7x fewer than fp32.
  - 2-bit strictly dominates 1-bit at equal recall (fewer bytes, lower warm p50, 2x storage).
  - 4-bit is the ONLY feasible quantized option on hard/high-dim segments where 1-bit/2-bit can't
    meet the SLA -> multi-bit unlocks them natively.
  - Storage stays 8x under fp32 at 4-bit.
Caveats (why not full strong-go):
  - Scoring is scalar unpack (no SIMD/LUT) -> per-score is slower than fp32; a FAIR latency verdict
    needs the optimized kernels (Hamming popcount, byte-shuffle/LUT ADC). Node/bytes wins hold
    regardless; absolute warm latency does not.
  - Build cost is inflated by scalar scoring.
  - Cold-cache / threaded throughput / p99-under-load deferred to real hardware -- where multi-bit's
    7x bytes-read advantage over fp32 should matter MOST.
  - Per-segment bit-width choice is objective-dependent (2-bit for storage/bytes, 4-bit for latency/
    feasibility); not a single universal pick.
Decision: adopt consistent 2-bit/4-bit quantized graphs; 2-bit as the storage-efficient default on
feasible segments, 4-bit where 2-bit's ceiling is short; implement optimized packed scoring kernels
before claiming absolute latency wins; keep fp32 for rerank + reference. Rotation stays off.
```

## Minimal future OpenSearch/Lucene design note (not implemented)

- **Index format/versioning:** new quantized vector format tag carrying `bits ∈ {1,2,4}` + per-dim
  min/max (or per-segment quantization params); packed code layout (8/4/2 dims/byte, documented
  endianness/padding, padding bits forced 0 and excluded from scoring).
- **Per-segment bit-width metadata:** store `(bits, ef_search, candidate_count, rerank_depth,
  status)` chosen at flush/merge (conservative calibration, ~50 sample queries; escalate 1→2→4-bit
  pilots on a sample rather than building all full graphs; emit `SLA_UNACHIEVABLE`/`USE_FP32`).
- **Distance-computer selection:** pick the packed scorer for the segment's `bits` (with a scalar
  fallback + an optimized SIMD/LUT path per arch); the SAME scorer drives graph build and query.
- **Query-time parameter lookup:** read the segment's `(bits, ef, cand, depth)`; user override
  allowed; conservative fallback (global default / fp32) for tiny or infeasible segments.
- **Exact rerank:** keep fp32 originals; rerank the calibrated depth; return exact top-k.

## Reproduce
```bash
./gradlew :test --tests "org.opensearch.knn.research.Track2ProductionMultibitTests" \
    -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
```
Hardware note: warm/indicative latency only; run cold-cache + threaded throughput + p99-under-load on
a controlled machine, and implement SIMD/LUT packed scoring, before any absolute-latency claim.
