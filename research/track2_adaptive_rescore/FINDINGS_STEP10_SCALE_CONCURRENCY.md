# Track 2, Step 10 — Working-set + concurrency crossover: does multi-bit beat fp32 at scale?

**Branch:** `track2-adaptive-rescore`. Harness:
[`Track2ScaleConcurrencyTests.java`](../../src/test/java/org/opensearch/knn/research/Track2ScaleConcurrencyTests.java)
(methods: `testNeonKernelMicrobench`, `testScaleConcurrency`, `testScale500k`, `testHighDimManifold`,
`testHighDimIsotropic`). Raw CSVs: [`results/track2_step10_*`](results/). Representation & geometry
UNCHANGED from steps 8/9 (per constraints).

## 1. Executive decision

```
GO (qualified). The predicted crossover is REAL and is driven by TWO axes:

  1. DIMENSION (dominant, production-relevant): at 768-D, 4-bit beats fp32 on p50, p99 AND QPS even
     at small N (cache-resident) -- manifold-768: p50 -23%, p99 -24%, QPS +30%; isotropic-768:
     p50 -28%, p99 -39%, QPS +42% -- while reading 8x fewer bytes and using 8x less storage. The
     autovectorized quantized dot-kernel is faster per-score than fp32 at high dim, and rerank is cheap.
  2. WORKING SET / N (low-dim): on SIFT-128, fp32 wins up to N=250K, but at N=500K (fp32 working set
     256MB, ~10x cache) 4-bit crosses -- p99 -17%, QPS +17% at 16 threads, and scales BETTER
     (eff 0.439 vs 0.415).

Production embeddings are 768-1536-D, so 4-bit wins the regime that matters. 1-bit is INFEASIBLE at
768-D (ceiling <= 0.61); 4-bit's candidate ceiling ~= fp32. Multi-bit still crushes 1-bit everywhere.

Qualified by: build cost ~3-4x fp32; latency is WARM/single-run on ONE arm64 host (Apple M5, no
CPU pinning, no PMU); high-D data is SYNTHETIC (no real embedding corpus available locally); the fast
kernel is JIT/SuperWord autovec (arm64 NEON), not hand-written SIMD and not the C++ JNI path. The
mechanism is NOT memory-bandwidth saturation (M5 bandwidth is never within 30x of saturation); it is
node-count parity + tail cache-miss latency + high-dim compute.
```

Decision-gate mapping (task §25): meets **STRONG GO** on the letter — *multi-bit beats 1-bit at equal
recall*, *4-bit fixes high-D ceilings*, *(a) 4-bit crosses fp32 under large working-set/concurrency*
**and** *(b) comparable-or-better latency with 8x lower storage*, *calibration selects sensibly*. Held
to **GO (qualified)** rather than unqualified strong-go only because build cost is 3-4x and the
evidence is warm/single-run/one-host/synthetic-high-D.

## 2. Environment / hardware topology (task §23)

Apple **M5 Pro**, 15 cores (no SMT — 15 logical = 15 physical), 24 GB LPDDR5, **arm64 / NEON**
(no AVX2/AVX-512 on host). L1d 64 KB, L2 8 MB per cluster; system-level cache not exposed by sysctl —
**~24 MB assumed (labeled `SLC_est`)**. JDK 21 Temurin, HotSpot C2. Test fork heap raised to 12 GB
for the run (Gradle's 512 MB default caps N; reverted in `build.gradle` post-run). **No `perf` /
Instruments / PMU access on macOS arm64 without sudo → hardware counters UNAVAILABLE** (§8). Latency
is **warm** (JIT-hot, resident), cores **not pinned**, no page-cache eviction. Each warm percentile =
best of 3–4 reps; threaded QPS recycles queries across a shared cursor. Frequency not pinned → cycle
figures are estimates (nominal 4.4 GHz).

## 3. Dataset summary (task §3)

| dataset | dim | N | source | role |
|---|---|---|---|---|
| SIFT | 128 | 6K → 500K | **real** (`sift_base.fvecs`, 1M avail) | scale + concurrency crossover |
| manifold-768 | 768 | 8K | **synthetic** low-rank (r=48) + 40 clusters + noise | realistic embedding-like proxy |
| isotropic-768 | 768 | 8K | **synthetic** i.i.d. Gaussian | worst-case high-dim |

**No real 768/1536-D embedding corpus is available locally and none was downloaded** — high-D results
are SYNTHETIC and explicitly NOT a production proof. manifold-768 (low intrinsic dim, anisotropic,
clustered) is a *better-than-isotropic* proxy for real embeddings; isotropic-768 is a stress worst case.
Ground truth = exact fp32 top-10 over the full N per query. Calibration (NCAL=50) and eval (NEVAL=200)
query sets are disjoint; representation/ef/depth are chosen on calibration data only.

## 4. Correctness status (task §4, §11)

- **Scalar LUT kernel: BIT-EXACT** to the scalar reference (step 9: 4032 asym + 2880 sym checks; asserted
  again here for high-D dims). Ordering/recall provably unchanged.
- **Autovec dot-kernel (`L2 = C_q − 2·Σ w_d·lvl_d + C_x`): algebraically exact, float-reassociated →
  max relative error 9.57e-06, 0 violations > 1e-3** across dims {128,384,768,1536,130,7}, bits {2,4}.
  Within documented FP-tie tolerance; candidate ordering/recall preserved (equal-recall re-verified per
  run). `C_x` is a per-code scalar precomputed at encode (4 B/vec, or recomputed); accumulation is FP32,
  per-dim contribution ≤ rng² for dim ≤ 1536 → no overflow.
- Concurrency correctness: Lucene's `HnswGraph` seek/nextNeighbor cursor is **not thread-safe**; each
  worker uses a per-thread `GraphView` (own cursor, concurrent *reads* only of the immutable post-build
  adjacency). Fixed a real `ArrayIndexOutOfBoundsException` this surfaced.

## 5. Kernel microbenchmarks (task §5, §12) — dimension decides the winner

ns/score (best-of hot/large × seq/random; full grid in `track2_step10_kernel_*.csv`):

| dim | 1-bit LUT | 2-bit LUT | 2-bit **dot** | 4-bit LUT | 4-bit **dot** | fp32 |
|---|---|---|---|---|---|---|
| 128 | 77 (2.5×) | 77 (2.4×) | 79 | 76 (2.6×) | 81 | **61** |
| 768 | 615 (1.9×) | 611 | **462 (2.7×)** | 608 | **484 (2.5×)** | 547–579 |

- LUT is ~2–2.5× over scalar at all dims and is the **fastest exact kernel at 128-D**; **fp32 still wins
  per-score at 128-D** (61 ns; gather-LUT can't beat autovectorized fp32 FMA).
- **At 768-D the autovec dot-kernel (462/484 ns) BEATS fp32 (547–579 ns)** — quantized reads far fewer
  bytes and the multiply-add reduction SuperWord-vectorizes; the LUT (gather) does not. **This is the
  compute reason 4-bit wins end-to-end at high dim.** (1-bit gains nothing from the dot form — it uses LUT.)
- Query-prep (LUT or QP build) is 1–35 µs, < 1% of a query over hundreds of visited nodes; reported
  separately, never hidden. **The native search path now selects the fastest exact kernel per dim**
  (LUT ≤ 256-D, dot ≥ 384-D).

## 6. Equal-recall scale results — SIFT-128, 1 thread, warm (task §1, §13, §22)

| N | fp32 p50/p99 (ef/d) | 4-bit p50/p99 (ef/d) | 2-bit p50/p99 (ef) | winner | fp32 WS | cache |
|---|---|---|---|---|---|---|
| 6K | **32/47** (20/10) | 76/97 (50/20) | 84/107 (50) | fp32 | 3 MB | L2 |
| 25K | **91/123** (50/10) | 105/152 (50/20) | 202/268 (100) | fp32 | 13 MB | SLC |
| 100K | 184/270 (50/10) | **178/267** (50/20) | 446/612 (200) | ~tie | 51 MB | >SLC |
| 250K | **235/340** (50/10) | 396/536 (100/100) | 1321/1771 (500) | fp32 | 128 MB | >SLC |
| **500K** | 480/712 (100/10) | **462/574** (100/20) | — | **4-bit** | 256 MB | ≫SLC |

- fp32 wins ≤ 250K: its higher candidate ceiling lets it use a **lower ef** (fewer nodes) — at 250K
  fp32 visits 297 nodes (ef50) vs 4-bit 537 (ef100), and 128-D fp32 scoring is autovec-fast.
- **At 500K both need ef=100 → node counts EQUALIZE (fp32 1628, 4-bit 1643)**; fp32 loses its ef edge,
  and 4-bit's 8× lower per-node bytes (105 KB vs 834 KB read/query) wins the tail: **p99 574 vs 712
  (−19%)**, QPS 2266 vs 2112. `code_bytes_read`: fp32 833 K, 4-bit 105 K per query.
- 2-bit is dominated at scale (lower ceiling forces ef=500 by 250K); among quantized, **4-bit is the
  clear pick** everywhere N ≥ 25K.

## 7. Concurrency results — 1..16 threads (task §2, §18)

| N | rep | 1T QPS | 16T QPS | 16T p99 µs | eff @16T |
|---|---|---|---|---|---|
| 250K | fp32 | 4,138 | **28,131** | **2,179** | 0.425 |
| 250K | 4-bit | 2,606 | 17,558 | 3,051 | 0.421 |
| **500K** | fp32 | 2,086 | 13,838 | 4,213 | 0.415 |
| **500K** | 4-bit | 2,302 | **16,160** | **3,496** | **0.439** |

- **At 250K, fp32 and 4-bit scale IDENTICALLY** (eff 0.425 vs 0.421) and fp32 stays ahead — concurrency
  does **not** create a crossover at 250K.
- **At 500K, 4-bit wins QPS (+17%) and p99 (−17%) at 16T AND scales better** (0.439 vs 0.415) — fp32's
  larger per-query footprint starts hurting its tail under load.
- fp32 does **not** saturate memory bandwidth first: aggregate fp32 traffic is **4.3 GB/s (250K/16T)
  / 11.5 GB/s (500K/16T)** vs M5's ~400 GB/s — ~35–100× headroom. The 500K win is **tail cache-miss
  latency**, not bus saturation.

## 8. Hardware-counter analysis (task §3, §8) — UNAVAILABLE, proxies used

No `perf`/PMU on this macOS arm64 host (no sudo). Per the task, **no counters are invented.** Proxies:
wall-clock p50/p95/p99, `code_bytes_read/query` (nodes × code bytes, in CSV), fp32-working-set-vs-cache
math, and scaling efficiency. The bandwidth estimates above are derived (qps × nodes × code_bytes),
labeled as estimates. IPC / cache-miss / DRAM counters would sharpen the 500K mechanism attribution and
are the first thing to capture on a Linux x86 box with `perf`.

## 9. Working-set crossover analysis (task §4) — the crossover table

**A. fp32 vector working set vs cache (128-D):** exceeds L2 (8 MB) at **N ≈ 16K**; exceeds SLC_est
(24 MB) at **N ≈ 48K**; at 500K it is **256 MB (~10× SLC)**.

**B. Crossover table (equal recall ≥ 0.95):**

| workload | fp32 | 4-bit | winner | note |
|---|---|---|---|---|
| small / 1T (≤25K, 128-D) | p50 32–91 | 76–105 | **fp32** | cache-resident; fp32 compute wins |
| large / 1T (250K, 128-D) | p50 235 | 396 | **fp32** | fp32 lower ef → fewer nodes |
| **large / 1T (500K, 128-D)** | p50 480 / p99 712 | **462 / 574** | **4-bit** | node parity; 4-bit 8× less traffic |
| **large / 16T (500K, 128-D)** | QPS 13,838 / p99 4,213 | **16,160 / 3,496** | **4-bit** | 4-bit scales better under load |
| **high-dim / 1T (768-D, 8K)** | p50 896 / p99 1138 | **687 / 866** | **4-bit** | autovec kernel + 8× less traffic |
| **high-dim / 8T (768-D, 8K)** | QPS 9,583 | **10,066** | **4-bit** | +5% QPS, −4% p99_8t |

**The crossover DOES appear** — at large N (500K) on low-dim, and at *all* N on high-dim. It does **not**
appear for small/medium-N low-dim workloads (fp32 wins there).

## 10. Real high-dimensional results (task §5, §14) — SYNTHETIC, 768-D, N=8K

**Candidate ceiling vs ef (recall@10):**

| rep | manifold ef100 | manifold ef200 | manifold ef500 | isotropic ef200 | isotropic ef800 |
|---|---|---|---|---|---|
| 1-bit | 0.206 | 0.310 | 0.497 | 0.495 | 0.833 → **infeasible** |
| 2-bit | 0.846 | 0.956 | 0.998 | (infeasible @ ef≤800) | |
| 4-bit | 0.930 | **0.990** | 1.000 | 0.918 | 0.987 |
| fp32 | 0.926 | 0.988 | 1.000 | — | — |

**Equal-recall @0.95 (fastest exact kernel):**

| dataset | rep | ef/d | recall | p50 | p99 | QPS 1T | QPS 8T | bytes/q |
|---|---|---|---|---|---|---|---|---|
| manifold-768 | 2-bit | 500/200 | 0.983 | 1350 | 1631 | 732 | 4770 | 519 K |
| manifold-768 | **4-bit** | 200/20 | 0.983 | **687** | **866** | **1488** | **10066** | 544 K |
| manifold-768 | fp32 | 200/10 | 0.988 | 896 | 1138 | 1142 | 9583 | 4319 K |
| isotropic-768 | **4-bit** | 500/50 | 0.970 | **2078** | **2152** | **481** | 2904 | 1667 K |
| isotropic-768 | fp32 | 500/10 | 0.979 | 2893 | 3534 | 339 | 2546 | 13352 K |

- **1-bit becomes generation-bound / infeasible at 768-D** (ceiling ≤ 0.61 manifold, ≤ 0.83 isotropic).
- **2-bit is feasible but weak** (needs ef=500; slower than 4-bit).
- **4-bit materially raises the ceiling to ≈ fp32** (0.990 vs 0.988 @ef200) and **navigates ~as well as
  fp32 at far lower cost** — it **beats fp32 on p50/p99/QPS at 768-D**, reading 8× fewer bytes.
- Answer to "how close is 4-bit to fp32 candidate generation on real high-D?": **at parity** on this
  embedding-like proxy — and faster end-to-end.

## 11. Storage-density table (task §7) — code bytes only (graph + fp32-rerank are additional)

| dims | N | 1-bit | 2-bit | 4-bit | fp32 | 4-bit vs fp32 |
|---|---|---|---|---|---|---|
| 768 | 1M | 96 MB | 192 MB | 384 MB | 3.07 GB | **8×** |
| 768 | 100M | 9.6 GB | 19 GB | **38 GB** | 307 GB | **8×** |
| 1536 | 100M | 19 GB | 38 GB | **77 GB** | 614 GB | **8×** |

4-bit is uniformly **8× smaller** than fp32 (and 4× the 1-bit codes). Full table:
[`track2_step10_storage_density.csv`](results/track2_step10_storage_density.csv).

## 12. Build / indexing cost (task §9, §19)

Single-threaded Lucene on-heap build, symmetric scoring. From measured `build_ms`: SIFT-128 100K fp32
**24 s** vs 4-bit **85 s** (**3.5×**); 500K fp32 **200 s** vs 4-bit **562 s** (**2.8×**). Quantized build
is **~3–4× fp32** — the main GO caveat. The optimized recon-table symmetric scorer helped only marginally
(step 9); a real reduction needs a C++ SIMD symmetric kernel (build uses code-vs-code scoring, not the
query LUT/dot path). Peak memory scales with N×code_bytes (4-bit far below fp32). Not optimized here
(no build bug found).

## 13. Per-segment calibration selections (task §20) — objective- and regime-dependent

Selection is a **segment/index-build-time** property (bit width is baked into the encoded segment; only
ef/rerank vary at query time). Derived from the measured tables:

| segment | latency-first | memory-first | p99-constrained | balanced |
|---|---|---|---|---|
| SIFT-128, N≤250K | **fp32** | 4-bit (8× storage) | **fp32** | fp32 / 4-bit |
| SIFT-128, N=500K | **4-bit** | 4-bit | **4-bit** | **4-bit** |
| 768-D (any N here) | **4-bit** | 4-bit | **4-bit** | **4-bit** |
| 768-D, storage-min | 4-bit | (2-bit if ceiling met) | 4-bit | 4-bit |

Expected behavior confirmed: **easy/small-low-dim → fp32; large-N or high-dim → 4-bit; 1-bit only as a
storage floor and never at high-dim (infeasible).** The calibrator must not be forced to 4-bit where
fp32 is genuinely cheaper (small low-dim).

## 14. Limitations (do not over-read)

- **Warm, single-run, one host** (Apple M5, arm64). No CPU pinning, no PMU counters, no page-cache
  eviction → no cold-cache/disk claims. p99 under sustained real load unproven.
- **High-D is SYNTHETIC.** manifold-768 is embedding-*like*, not a real corpus. A real 768/1536-D
  embedding benchmark is required before a production claim.
- **Fast kernel is JIT/SuperWord autovec (arm64 NEON), not hand-written SIMD** and not the plugin's C++
  JNI path. A C++ AVX2/AVX-512/NEON port would likely widen the high-D quantized lead and cut build cost.
- **M5's bandwidth is unusually abundant**, which *disfavors* the memory-traffic mechanism. On
  bandwidth-constrained x86 (many cores/channel) or disk/mmap-backed fp32 (page faults), the crossover
  should appear *earlier and larger* — untested here.
- N capped at 500K (128-D) and 8K (768-D) by the 20-min randomizedtesting suite timeout + build cost;
  1M / larger-high-D not run.

## 15. OpenSearch integration recommendation (task §15)

**Recommend an experimental integration**, scoped to where the evidence is strongest: a **consistent
4-bit quantized HNSW codec** as an opt-in per-segment representation, primarily for **high-dimensional
(≥ 384-D) embeddings and large segments**, with fp32 retained for exact rerank and as the small/low-dim
default. Concretely:
- Per-segment `bits ∈ {2,4}` metadata chosen by conservative calibration at flush/merge (fp32 fallback
  for small/low-dim/compute-hot; **1-bit disallowed ≥ 384-D**).
- Distance-computer dispatch selecting the **fastest exact kernel per dim** (LUT low-dim, byte-shuffle/
  dot high-dim), with the scalar path as guaranteed fallback; **same kernel for build and query**.
- Keep fp32 originals; rerank the calibrated depth (≤ 20 for 4-bit); return exact top-k.
- Gate the rollout on a **real high-D embedding benchmark + a C++ SIMD kernel + a Linux/x86 perf run**.

## 16. Exact next step

**Port the 4-bit packed scorer to a C++ AVX2/NEON JNI kernel and re-run this exact-recall matrix on a
real 768/1536-D embedding corpus on a Linux x86 host with `perf`** — to (a) replace autovec with true
SIMD (should widen the high-D lead and cut the 3–4× build cost), (b) confirm the crossover on real
embeddings, and (c) capture IPC / LLC-miss / DRAM counters that attribute the 500K/high-D wins to
compute vs. traffic. Secondary: measure disk/mmap-backed fp32 rerank under page-cache pressure (the one
place fp32's 8× traffic should bite hardest), which this sandbox cannot evict.

## Answers to the 10 required questions (task §10)

1. **Crossover with size?** Yes — at N=500K (128-D) 4-bit beats fp32 (p99 −17%, QPS +17% at 16T); not at ≤250K.
2. **Concurrency move it?** It doesn't *create* it at 250K (identical scaling), but at 500K 4-bit scales better (0.439 vs 0.415), widening its lead under load.
3. **Is fp32 bandwidth-limited at scale?** No — traffic is 4–11 GB/s vs ~400 GB/s. fp32 is **compute + cache-miss-latency** limited, not bandwidth-saturated, on M5.
4. **Is 2-bit the best memory default?** Only where its ceiling is met cheaply; at scale/high-dim it needs high ef and is dominated by 4-bit. **4-bit is the better default on realistic (high-dim/large) data;** 2-bit is a storage-min option on easy data.
5. **When is 4-bit necessary for recall?** At 768-D (1-bit/2-bit infeasible or weak) and wherever the 1/2-bit candidate ceiling stays < SLA.
6. **4-bit vs fp32 candidate generation on real high-D?** At parity (0.990 vs 0.988 @ef200 on the manifold proxy) — and faster end-to-end.
7. **Multi-bit vs 1-bit at equal recall?** Massive: 1-bit needs ef=500/d=300 (p50 ~440 µs) on SIFT and is **infeasible at 768-D**; 4-bit is 5–6× faster on SIFT and the only feasible quantized option at high-dim.
8. **Storage saved vs fp32?** 8× (4-bit), uniformly.
9. **Does NEON/autovec close the gap?** Yes at high-dim — the autovec dot-kernel beats fp32 per-score at 768-D and flips 4-bit from ~9% behind (LUT) to 23–24% ahead. At 128-D it doesn't help (fp32 compute wins).
10. **Sufficient for experimental integration?** Yes — as an opt-in high-dim/large-segment 4-bit codec, gated on a C++ SIMD kernel + real-embedding + x86-perf validation.

## Reproduce
```bash
# kernel microbench + exactness
./gradlew :test --tests "org.opensearch.knn.research.Track2ScaleConcurrencyTests.testNeonKernelMicrobench" \
    -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
# scale (6K-250K) + concurrency; needs a large test heap (Gradle default 512MB caps N):
#   temporarily set `test { maxHeapSize = "12g" }` in build.gradle
./gradlew :test --tests "org.opensearch.knn.research.Track2ScaleConcurrencyTests.testScaleConcurrency" ...
./gradlew :test --tests "org.opensearch.knn.research.Track2ScaleConcurrencyTests.testScale500k" ...
./gradlew :test --tests "org.opensearch.knn.research.Track2ScaleConcurrencyTests.testHighDimManifold" \
          --tests "org.opensearch.knn.research.Track2ScaleConcurrencyTests.testHighDimIsotropic" ...
```
Hardware note: warm/indicative, one arm64 host, no PMU. The C++ SIMD kernel + real-embedding +
Linux-x86-`perf` + disk-backed-fp32 runs are the deferred, decisive next steps.
```
