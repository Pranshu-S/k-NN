# Collective Benchmark — Methodology

Evaluates **standard / ACORN-1 / ACORN-γ / RACORN-1 / RACORN-1+ / exact** on
filtered Faiss HNSW, revalidating the standalone findings and testing the
RACORN-1 paper's claims (arXiv:2607.00768) in a common harness.

## 1. Honesty & scope statement (read first)

The decisive quantities for these traversal policies — **recall** and
**visit-count / distance-computations** — live at the native Faiss layer. This
harness measures them through the **exact same** `faiss::IndexHNSWFlat` +
`IDSelector` + traversal code that the OpenSearch native path invokes
(`jni/src/acorn_hnsw.cpp` is a port of the validated `research/acorn/src`
implementations). So recall and visit-count transfer directly to OpenSearch.

What this harness does **not** measure live (a real single-node cluster + JNI
build could not be run in this environment): end-to-end REST latency, JVM/GC/heap,
Lucene filter-bitmap construction cost, and coordinating-node/shard reduction.
Those are analyzed from the code path and, for segments, via corpus partitioning.
**No live-cluster numbers are fabricated.**

**Latency caveat.** The research RACORN/ACORN implementations use `std::priority_queue`
and a full two-hop neighbour scan per node — correct but with higher constant
overhead than an optimized engine (the paper uses USearch). Following the paper,
we therefore treat **visit-count (distance computations) as the primary,
implementation-independent speedup signal**, and report wall-clock latency with
this caveat. Visit-count ratios here are faithful; absolute µs are pessimistic for
ACORN/RACORN relative to a production implementation.

## 2. Strategies

| id | strategy | graph | notes |
|---|---|---|---|
| H | HNSW In-filtering (distance-first) | std M=16 | paper's recall-preserving HNSW baseline |
| Hl | HNSW In-filtering, larger M | std M=32 | denser-graph baseline |
| Fx | faiss-native filtered HNSW (`STANDARD`) | std M=16 | **today's OpenSearch path** |
| A1 | ACORN-1 (faithful, RACORN BR=0) | std M=16 | filter-first, no bridges |
| Ag | ACORN-γ graph + ACORN traversal | γ=8 graph | graph densification (build-time) |
| R1 | RACORN-1 (ASF, bridge_ratio=1.0) | std M=16 | filter-failing bridges |
| R1q | RACORN-1 (ASF, bridge_ratio=0.25) | std M=16 | neg-correlation frontier |
| R1p | RACORN-1+ (RACORN-1 + Adaptive Exact Fallback) | std M=16 | exact switch at extreme-low sel |
| Ex | exact filtered search | — | oracle / crossover |

All ANN strategies share the **same** vectors, filters, queries, index settings,
warm-up, and query order per (correlation, cand, seed). ACORN-1/RACORN/HNSW-infilter
all run over the **same** standard graph (search-time only).

## 3. Data & correlation conditions (paper §7.5 mapped to OpenSearch)

Clustered Gaussian mixture (`nc≈n/2000` centres, σ=0.06) — the analogue of the
paper's K-means clusters and of real product/tenant/language clustering. Queries
drawn near a fixed cluster (cluster 0). Three query–filter correlation conditions:

- **no** — random filter (query-independent). *Random ACL / hash-tenant analogue.*
- **pos** — valid = nearest docs to cluster 0 (near query). *Category/availability
  filter aligned with the query's semantic region.*
- **neg** — valid = nearest docs to the farthest cluster (far from query).
  *Adversarial: eligible content lies in a different semantic region than the query
  (cross-domain tenant, region-eligibility mismatch).* **This is the regime where
  ACORN-1 collapses and RACORN is designed to help.**

`correlation_mix(ρ)` (continuous 0→1) is also available in the standalone harness;
the three discrete conditions above are the headline cases.

## 4. Selectivity / candidate counts

Absolute candidate counts `{100, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000}`
(= 0.1%…50% at 100K) — reported as absolute `cand` because the exact-search
crossover is governed by absolute eligible count, not %.

## 5. Configs

k=10; ef_search ∈ {50,100,200,400}; d ∈ {128,768}; metric ∈ {L2, IP(cosine,
normalized)}; graph M ∈ {16, 32}; ef_construction=100; ACORN-γ γ=8, M_β=M. n ∈
{100K, 1M}. **Multiple graph seeds** (≥2) with median + standard deviation reported.

## 6. Ground truth & recall

Per query, exact top-k over **eligible (filter-passing) documents only**:
`recall@k = |returned ∩ truth| / min(k, |eligible|)`. Handles `|eligible| < k`.
Never compared against unfiltered ground truth. Truth computed once per
(correlation, cand, seed) and reused across all strategies and ef.

## 7. Metrics recorded (per strategy × config, seed-aggregated)

recall (median, sd); p50/p95(sd)/p99/mean latency; distance computations (ndis);
bridges used; AEF switches. Plus, from build: graph bytes, degree, build time (in
the ACORN-γ overhead table). 3-query warm-up separated from measurement; single
thread for latency; concurrency measured separately (standalone harness).

## 8. Reproduce

```bash
bash research/acorn/scripts/build.sh bench_collective racorn_smoke tests
./research/acorn/build/tests               # correctness (incl. ACORN/γ)
./research/acorn/build/racorn_smoke        # RACORN vs ACORN-1 on a negative-correlation filter
./research/acorn/build/bench_collective 100000 128 l2 60 2 <out.csv> tag   # 100K L2
./research/acorn/build/bench_collective 1000000 128 l2 40 2 <out.csv> tag  # 1M scale
python3 research/acorn/opensearch-benchmark/scripts/analyze_collective.py <out.csv>
```

Environment: Apple arm64, clang 21, `-O3 -DNDEBUG` (release), macOS Accelerate
BLAS, deterministic single-thread build (no OpenMP) for stable measurement. Faiss
1.11.0 (submodule pin `5616caad`); k-NN branch `claude/acorn-gamma-faiss-hnsw-729026`.
