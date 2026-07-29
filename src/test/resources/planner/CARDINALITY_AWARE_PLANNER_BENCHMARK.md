# Cardinality-Aware Filtered k-NN Planner — Experiment Notes (Prototype)

This planner is a prototype. The threshold `max(1000, k * 10)` is a **starting hypothesis**, not a
validated production default. This note describes how to measure whether exact search actually beats
filtered ANN for selective filters, and at what cardinality the crossover sits.

## What to measure

For each run, capture:

| Metric | Source |
|---|---|
| total documents (shard-local) | `IndexReader.numDocs()` |
| filtered cardinality | `IndexSearcher.count(filterQuery)` (the planner's estimator) |
| k | query |
| planner / cardinality-estimation time | wrap `BoundedExactSearchDecider.plan` |
| exact execution time | latency when the planner forces `EXACT` |
| ANN execution time | latency with the planner disabled (approximate path) |
| total query latency | search request took time |
| vectors scored | exact = filtered cardinality; ANN ≈ ef_search × segments (graph visits) |
| recall @ k | exact results are the oracle; measure ANN recall against them |

## Cardinalities to sweep

```
0, 10, 100, 1,000, 10,000
```

Hold total docs fixed (e.g. 1,000,000 per shard) and vary the filter's selectivity to hit each target
cardinality. Repeat for `k ∈ {10, 100}` so the `k × 10` arm of the threshold is exercised
(`k = 100 → threshold 1,000`; `k = 200 → threshold 2,000`).

## How to run each arm

The planner is engaged automatically for Lucene-engine FLOAT, non-nested, top-k, filtered,
rescore-disabled queries. To compare arms without changing production settings:

- **Exact arm:** issue the filtered query whose cardinality is `≤ threshold` — the planner selects
  exact automatically. Confirm via the debug log line
  `Cardinality-aware k-NN plan ... strategy:EXACT`.
- **ANN arm (baseline):** issue the same query with a filter whose cardinality is just above the
  threshold, or run against a build without this change, to get the approximate path.
- **Match-none arm:** a filter matching zero docs should return immediately (`strategy:MATCH_NONE`).

Enable the debug log to observe decisions:

```
PUT /_cluster/settings
{ "transient": { "logger.org.opensearch.knn.index.query.KNNQueryFactory": "DEBUG" } }
```

## Suggested harness

Extend the existing OpenSearch k-NN benchmarking tooling (`benchmarks/`) or the big5/`osb` vector
workloads. A minimal standalone reproduction can be built on top of
`ExactFilteredKNNVectorQueryTests` (already multi-segment) by timing `searcher.search(...)` for the
exact query versus `OSKnnFloatVectorQuery` at increasing filtered cardinalities.

## Measured results (this repo, in-process) — iteration 2

Run (produces both the 10k and 100k tables; read them from the JUnit XML `<system-out>`):

```
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home
./gradlew :test --tests "org.opensearch.knn.index.query.CardinalityAwarePlannerBenchmarkTests" \
  -x spotlessJavaCheck -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
```

Common config: `dimension=128`, `k=10`, `efSearch=100`, EUCLIDEAN unit-norm random vectors, single
shard, `threshold = max(1000, k*10) = 1000`. **Query cache disabled** and **strategy order rotated per
iteration** (fixes the iteration-1 methodology artifacts). Index build excluded from timings; exact
matched the brute-force oracle in every case.

### 100,000 docs (warmup 3, measured 12) — the informative corpus

| Cardinality | Selectivity | ANN med/p95 µs | Exact med/p95 µs | OldPlanner med/p95 µs | OptPlanner med/p95 µs | Strategy | Recall |
| ----------: | ----------: | -------------- | ---------------- | --------------------- | --------------------- | -------- | -----: |
| 0 | 0.00% | 6.2 / 7.0 | 24.2 / 30.0 | 10.1 / 23.0 | 22.3 / 29.5 | MATCH_NONE | n/a |
| 10 | 0.01% | 376.5 / 618.9 | 155.5 / 294.0 | 245.6 / 307.6 | **146.0 / 274.1** | EXACT | 1.000 |
| 100 | 0.10% | 412.7 / 581.1 | 146.9 / 211.4 | 227.2 / 251.9 | **151.7 / 328.1** | EXACT | 1.000 |
| 1,000 | 1.00% | 740.5 / 1066.5 | 231.7 / 2266.2 | 291.4 / 512.8 | **236.1 / 352.5** | EXACT | 1.000 |
| 5,000 | 5.00% | 1951.4 / 2427.4 | **644.4 / 735.5** | 2029.0 / 3017.9 | 1767.6 / 2491.0 | APPROXIMATE | 1.000 |
| 10,000 | 10.00% | **732.2 / 995.1** | 1063.4 / 1225.1 | 893.9 / 1132.4 | 930.6 / 1196.8 | APPROXIMATE | **0.700** |
| 25,000 | 25.00% | **1483.5 / 2066.2** | 2433.0 / 2664.2 | 1676.9 / 1885.5 | 1610.8 / 1971.8 | APPROXIMATE | 0.800 |
| 50,000 | 50.00% | **1192.7 / 1753.4** | 4604.0 / 4826.2 | 1441.5 / 1879.7 | 1316.7 / 1731.5 | APPROXIMATE | 0.700 |

### 10,000 docs (warmup 5, measured 20)

| Cardinality | Selectivity | ANN med/p95 µs | Exact med/p95 µs | OldPlanner med/p95 µs | OptPlanner med/p95 µs | Strategy | Recall |
| ----------: | ----------: | -------------- | ---------------- | --------------------- | --------------------- | -------- | -----: |
| 0 | 0.00% | 21.0 / 25.0 | 87.6 / 125.4 | 29.8 / 50.8 | 89.1 / 140.7 | MATCH_NONE | n/a |
| 10 | 0.10% | 486.4 / 572.6 | 499.7 / 920.5 | 806.5 / 1004.6 | **486.6 / 584.9** | EXACT | 1.000 |
| 100 | 1.00% | 450.5 / 472.8 | 416.6 / 540.3 | 672.8 / 924.9 | **414.2 / 597.5** | EXACT | 1.000 |
| 1,000 | 10.00% | 1191.3 / 1342.5 | 563.3 / 813.0 | 808.9 / 1055.9 | **600.5 / 805.3** | EXACT | 1.000 |
| 5,000 | 50.00% | 1594.1 / 1840.6 | **1357.1 / 1537.5** | 1942.5 / 2173.5 | 2045.3 / 2526.5 | APPROXIMATE | 1.000 |
| 10,000 | 100.00% | **810.5 / 1038.2** | 1649.5 / 1878.8 | 868.8 / 1212.2 | 988.5 / 1134.4 | APPROXIMATE | 1.000 |

### Bounded-count overhead vs old full count (100k)

| Cardinality | Old overhead µs (full count) | Opt overhead µs (bounded) | Full advances | Bounded advances | Advances saved |
| ----------: | ---------------------------: | ------------------------: | ------------: | ---------------: | -------------: |
| 10 | 76.6 | 60.7 | 10 | 10 | 0 |
| 100 | 71.8 | 61.3 | 100 | 100 | 0 |
| 1,000 | 71.9 | 72.8 | 1,000 | 1,000 | 0 |
| 5,000 | 72.4 | 115.5 | 5,000 | 1,001 | 3,999 |
| 10,000 | 69.9 | 87.0 | 10,000 | 1,001 | 8,999 |
| 25,000 | 69.6 | **4.9** | 25,000 | 1,001 | 23,999 |
| 50,000 | 59.4 | **4.7** | 50,000 | 1,001 | 48,999 |

### Reading of iteration-2

1. **The iteration-1 5,000-anomaly is resolved.** With the cache disabled and order rotated, the
   optimised planner selecting APPROXIMATE is now *slower* than raw ANN (bounded overhead + the same
   ANN), as it must be — e.g. 10k/card-5000: ANN 1594 µs, OptPlanner 2045 µs. Standalone ANN and
   planner-selected ANN are the same code; the earlier inversion was purely query-cache warming under a
   fixed ANN→exact→planner order.
2. **A crossover now appears.** At 100k, exact is faster through 5,000 (644 µs vs ANN 1951 µs, ~3×) but
   ANN wins at 10,000 (732 µs vs exact 1063 µs). Crossover ∈ (5,000, 10,000] ≈ 5–10 % selectivity.
   Exact latency scales linearly with cardinality (50k → 4.6 ms); ANN is roughly flat-to-non-monotonic.
3. **ANN reliability degrades at scale** — recall falls to **0.70–0.80** at 10k–50k cardinality on 100k
   docs, exactly the regime where the planner (correctly) prefers ANN for latency. So beyond the
   crossover there is a real latency-vs-recall trade-off; exact stays at recall 1.0 by construction.
4. **The bounded/reuse optimisation works.** For large filters the bounded counter stops at 1,001
   advances, saving up to ~49,000 advances and cutting isolated overhead ~15× (59 µs → 4.7 µs at 50k).
   End-to-end, the optimised planner beats the old full-count planner for the exact band — e.g.
   10k/card-10: **806 µs → 487 µs**; 100k/card-10: **246 µs → 146 µs** — and even edges out raw
   standalone exact there because it reuses the collected doc ids instead of re-evaluating the filter.

### Threshold evaluation

- The measured crossover (~5,000–10,000 for 128-dim/k10/ef100) is **well above** the prototype
  `threshold = 1000`. At card 5,000 the planner routes to ANN, which is ~3× slower **and** loses recall,
  when exact would be both faster and exact. **On this workload `1000` is too low**; a value nearer
  ~5,000 would fit better.
- **But do not change the default from one workload.** The crossover depends on dimension, `ef_search`,
  segment count and filter cost; ANN latency here is even non-monotonic in cardinality. The bounded
  counter makes a higher threshold cheap to evaluate (it still stops at `threshold + 1`), so the main
  cost of raising it is the exact scan itself.
- Recommended next step: sweep `ef_search ∈ {50,100,200}`, `dimension ∈ {128,768}`, and corpus up to
  1M to map the crossover surface before proposing any concrete default. This run pins the crossover
  only for one configuration.

## Measured results — iteration 3 (multi-query, distribution sweep)

`CardinalityAwarePlannerBenchmarkTests` now runs **50 query vectors** (30 for 768-dim) per scenario,
reports the full recall@k distribution, and tests **two filter distributions**: `random` (scattered
doc ids) and `correlated` (a contiguous block of vector-space clusters). Env: JDK 21, macOS aarch64,
15 cores, 512 MB test heap, query cache disabled, order rotated, seed 42. Exact results are the recall
oracle. Configs: 100k × {dim128 (k10 ef50/100/200; k100 ef100), dim768 (k10 ef100)}.

### Headline: distribution flips the decision at equal cardinality

Same corpus/dim/k/ef, **card = 5,000**, 128-dim k10 ef100:

| Distribution | ANN p50 | Exact p50 | Faster path | ANN mean recall |
| ------------ | ------: | --------: | ----------- | --------------: |
| random       | 8175 µs | 4193 µs   | **exact** (~2×) | 1.000 |
| correlated   |  742 µs | 1801 µs   | **ANN** (~2.4×) | 0.996 |

The optimal path is **opposite** for the same cardinality. `random` filters scatter accepted docs, so
filtered-HNSW thrashes (ANN latency spikes to 8–16 ms at 5–10 % selectivity, then *recovers* at 25–50 %
— non-monotonic); `correlated` filters keep accepted docs in one graph region, so ANN stays flat-fast
(~700 µs) but loses a little recall. Exact latency is monotonic in cardinality regardless.

### Crossover ranges (cardinality where ANN overtakes exact)

| Config | random crossover | correlated crossover |
| ------ | ---------------- | -------------------- |
| 100k dim128 k10 ef100 | ~10,000–25,000 (ANN spikes 5k–10k, so exact wins that band too) | ~2,000–2,500 |
| 100k dim128 k10 ef200 | ~5,000–7,500 | ~2,000 |
| 100k dim128 k100 ef100 | ~10,000–25,000 | ~1,000–2,500 |
| 100k dim768 k10 ef100 | ~10,000–25,000 (exact 2× cheaper at 5k–10k) | ~1,000–2,500 |

Higher dimension pushes the correlated crossover down and inflates exact cost (768-dim exact at 50k =
94 ms). The crossover is driven by **distribution + dimension + ef**, not cardinality alone; because
corpus was fixed at 100k, selectivity and absolute cardinality co-vary and could not be separated here.

### ANN recall@k distribution

- `random`: mean recall ≈ 1.000 up to 10k; dips to 0.87–0.99 at 25k–50k (worst at dim768: mean 0.87,
  min 0.00 at 50k).
- `correlated`: recall degrades **earlier and deeper** — dim128 k10 ef50 mean 0.958 from card 1,000
  (only 58–62 % of queries perfect); k100 mean 0.977 with just 6 % perfect; dim768 mean 0.94 (min 0.80)
  from 2,500. Raising ef helps (ef200 correlated ≈ 0.998). This is the exact-search *reliability* case:
  in the low-cardinality correlated band the planner picks exact, which is both fast and recall-1.0.

### Decision-model comparison (regret vs. an oracle picking the faster measured path, 88 scenarios)

Constants fitted on dim128/k10/ef100 (calibration); evaluated on all scenarios:

| Model | Faster-path selection | Mean regret µs | p95 regret µs | ANN chosen w/ recall loss | Exact chosen w/ ANN ≥2× faster |
| ----- | --------------------: | -------------: | ------------: | ------------------------: | -----------------------------: |
| A: max(1000,10·k) (current) | 71.1 % | 1396 | 8832 | 21 | 0 |
| B: fixed 2500 | 73.3 % | 1419 | 8832 | 18 | 1 |
| B: fixed 5000 | 73.3 % | 1313 | 7626 | 15 | 5 |
| B: fixed 7500 | 73.3 % | 1040 | 7626 | 12 | 9 |
| B: fixed 10000 | 73.3 % | **726** | **2961** | 9 | 14 |
| C: 100·ef | 73.3 % | 819 | 6454 | 11 | 12 |
| D: 5000·128/dim | 71.1 % | 1321 | 8832 | 17 | 4 |
| E: cost card·dim | 66.7 % | 1409 | 8832 | 21 | 0 |

**No model exceeds ~73 % faster-path selection.** The ceiling exists because the optimal path flips with
distribution at equal (cardinality, dim, k, ef) — information not available to any cardinality-only rule
(nor to the cost model E, which assumes a flat ANN baseline the data violates by 20×). Raising the
threshold (B:10000, C:100·ef) roughly halves mean/p95 regret and cuts ANN-with-recall-loss cases, but
trades that for doing exact when ANN was ≥2× faster (the fast correlated band) 12–14×. So the choice is
a **latency-vs-recall preference**, not a single optimum.

### 1,000,000-doc corpus (dim128 k10 ef100, 30 queries, FSDirectory)

The 1M run completed in ~455 s (no OOM; vectors stay off-heap in `FSDirectory` under the 512 MB test
heap). It reinforces the distribution effect and the reliability case:

| Distribution | Cardinality | ANN p50 | Exact p50 | Faster | ANN mean recall |
| ------------ | ----------: | ------: | --------: | ------ | --------------: |
| random | 5,000 | 6579 µs | 3404 µs | exact | 1.000 |
| random | 50,000 | 8356 µs | 7973 µs | ~tie | 0.913 |
| correlated | 5,000 | 728 µs | 439 µs | exact | **0.873** |
| correlated | 10,000 | 467 µs | 700 µs | ANN | **0.783** |
| correlated | 50,000 | 754 µs | 3054 µs | ANN | **0.773** (3 % perfect) |

At 1M, `random` crossover moves up to ~25,000–50,000 cardinality (ANN is relatively more expensive at
scale), while `correlated` crossover is ~7,500–10,000 but with **severe ANN recall loss** (mean 0.77,
min 0.50). The absolute-cardinality crossover band (tens of thousands) is more stable across corpora
than selectivity (which shifts from ~10–25 % at 100k to ~2.5–5 % at 1M) — i.e. **absolute cardinality
is the more stable of the two predictors, but distribution still dominates both.**

### Interpretation

- The prototype rule A (threshold 1000) is the most latency-conservative (never wastes a slow exact) but
  leaves the most recall on the table (21 ANN-with-recall-loss cases). A modestly higher fixed threshold
  (~5,000–7,500) or `≈100·ef` lowers regret in these workloads, but none generalizes cleanly.
- The genuinely predictive signal is **graph locality of the filter**, which correlates with recall and
  ANN latency but is not observable from cardinality. The bounded-count pass already collects the matched
  doc ids; their spread could serve as a cheap locality proxy for a future adaptive rule — but that is
  out of scope here.

## Sparse exact-filter representation (iteration 3, `SparseFilterRepresentationBenchmarkTests`)

The exact path previously built a `FixedBitSet(maxDoc)` per matched leaf. Comparison of representations
(memory via `ramBytesUsed()`; the target sparse case is few matches in a large segment):

| Representation | maxDoc | Matches | Memory (bytes) | Build µs | Iterate µs |
| -------------- | -----: | ------: | -------------: | -------: | ---------: |
| FixedBitSet    | 10,000,000 | 10   | 1,250,040 | 8.21 | 122.92 |
| SparseFixedBitSet | 10,000,000 | 10 | 29,616 | 0.71 | 6.29 |
| **IntArrayDISI** | 10,000,000 | 10 | **56** | **0.08** | **0.04** |
| FixedBitSet    | 10,000,000 | 5,000 | 1,250,040 | 30.83 | 182.04 |
| SparseFixedBitSet | 10,000,000 | 5,000 | 112,776 | 52.71 | 56.29 |
| **IntArrayDISI** | 10,000,000 | 5,000 | **20,016** | **0.25** | **5.67** |

`IntArrayDocIdSetIterator` (adopted) is `O(matches)` memory — **~22,000× smaller** than `FixedBitSet`
for 10 matches in a 10M-doc leaf (56 B vs 1.25 MB) — and faster to build and iterate. The only regime
where `int[]` costs more bytes than `FixedBitSet` is a *dense* filter in a *small* segment (e.g. 5,000
matches in a 100k-doc leaf: 20 KB vs 12.5 KB), which is outside the sparse target and still iterates
faster. End-to-end exact latency on a real 100k×128 index was equal-or-slightly-better with `int[]`
(e.g. card 5,000: 246 µs vs 291 µs for `FixedBitSet`). `ExactSearcher` consumes only a
`DocIdSetIterator`, so no bit set is required and no exact-search logic was duplicated.

## Filter-locality feature study — iteration 4 (`FilterLocalityBenchmarkTests`)

Question: can a cheap locality signal from the bounded filter matches beat the ~73 % faster-path ceiling
of cardinality-only rules — and does it survive **adversarial** distributions that decouple document-id
locality from vector-space locality? Env: JDK 21, macOS aarch64, 512 MB heap, 50 queries/scenario,
100k/dim128/k10/ef100, query cache disabled.

Four distributions cross (doc-id locality) × (vector locality), with fitting on the two "natural"
distributions and evaluation on two held-out adversarial ones:

| Distribution | Set | doc-id normSpan | vector-centroid dispersion | crossover (exact→ANN) |
| ------------ | --- | --------------: | -------------------------: | --------------------- |
| rand_rand (scatter id, scatter vec) | calibration | ~0.95 | ~0.98 | ~50,000 (exact wins to 25k) |
| contig_corr (contig id, corr vec) | calibration | ~0.05 | ~0.73 | ~5,000 |
| **scatter_corr** (scatter id, **corr vec**) | **validation** | ~0.95 | ~0.73→0.98 | ~7,500 |
| **contig_rand** (contig id, **rand vec**) | **validation** | ~0.06 | ~0.99 | ~7,500 |

The adversarial rows confirm the interpretation warning: `scatter_corr` has **high** doc-id scatter
(normSpan ~0.95) but **correlated** vectors; `contig_rand` has **low** doc-id scatter (~0.06) but
**random** vectors. Document-id scatter is fully decoupled from vector locality — it is an
insertion-order artifact.

### Latency-oracle model comparison (calibration vs held-out validation)

| Model | CAL faster-path | VAL faster-path | VAL mean regret µs | VAL p95 regret µs |
| ----- | --------------: | --------------: | -----------------: | ----------------: |
| A: max(1000,10k) (current) | 62.5 % | 75.0 % | 86 | 710 |
| B: fixed 5000 | 75.0 % | 100.0 % | 0 | 0 |
| B: fixed 10000 | 75.0 % | 75.0 % | 138 | 1508 |
| C: 100·ef | 75.0 % | 75.0 % | 138 | 1508 |
| F: doc-id scatter | **93.8 %** | **68.8 %** | 543 | 6246 |
| G: segment leaves | 75.0 % | 62.5 % | 562 | 6246 |
| H: vector dispersion | **93.8 %** | **56.3 %** | 575 | 6246 |
| I: staged (H in ambiguous band) | 93.8 % | 56.3 % | 575 | 6246 |

**The locality models overfit and collapse on held-out data.** F (doc-id) and H (vector dispersion)
both score 93.8 % on calibration but drop to **68.8 % / 56.3 %** on the adversarial validation set —
*below* the cardinality-only baseline (A: 75 %) — with ~6× higher mean regret (543–575 µs vs 86 µs).
The doc-id collapse proves the artifact directly; the vector-dispersion collapse shows that raw sample
dispersion does not capture query-relative graph navigability either (`contig_rand` has the highest
dispersion ~0.99 yet ANN is *fast* there). `B: fixed 5000` scoring 100 % on validation is a small-set
coincidence (both adversarial distributions happen to cross near 5–7.5k) — it scores only 75 % on
calibration, so no single fixed threshold is robust across all four.

### Feature-extraction overhead

| Card | Scatter feature µs (A+B) | Vector feature µs (C) | % of exact p50 |
| ---: | -----------------------: | --------------------: | -------------: |
| 1,000 | 410 | 725 | 215 % |
| 5,000 | 219 | 329 | 40 % |
| 10,000 | 203 | 59 | 12 % |
| 50,000 | 1256 | 37 | 22 % |

(card 500 excluded — cold-start outlier ~6.7 ms.) In the exact-favorable band (card ≤ 5,000) the feature
work is 40–215 % of the exact query it would inform — material cost for no held-out benefit.

### Recall-constrained oracle (Phase 7)

| Recall target | Exact selections | ANN selections | Mean latency µs | Rule-A violations |
| ------------: | ---------------: | -------------: | --------------: | ----------------: |
| 0.90 | 4 | 28 | 1705 | 4 |
| 0.95 | 4 | 28 | 1705 | 4 |
| 0.99 | 11 | 21 | 1779 | 10 |

Enforcing recall ≥ 0.99 forces 11/32 scenarios onto exact (mean latency +4 %). The current latency-only
rule A would violate a 0.99 recall target in **10/32** scenarios (choosing ANN where ANN recall < 0.99 —
mostly `contig_rand` at cardinality ≥ 7,500 where ANN recall falls to 0.73–0.87). Latency-optimal and
recall-safe are genuinely different policies.

### Phase 8 decision: **no production change**

No signal met the bar (held-out faster-path ≥ 85 %, or ≥ 40 % regret reduction). Doc-id scatter and
segment concentration are insertion-order artifacts; vector-sample dispersion does not generalize and
costs bounded vector reads. **Graph-locality / ANN difficulty is not cheaply inferable from the
pre-search state available to the planner.** The production planner (threshold `max(1000, 10·k)`,
bounded counting, sparse-iterator exact reuse) is left unchanged; all iteration-4 code is test-only.

## Online adaptive ANN fallback — iteration 5 feasibility (`OnlineAnnProbeBenchmarkTests`)

Question: can an *online* probe (run ANN with a small visit budget, observe progress, abort to exact if
poor) beat the pre-search approaches?

### Lucene already implements the "abort slow ANN → exact" fallback

Decompiling `AbstractKnnVectorQuery.getLeafResults` (Lucene 10.5.0), per leaf:

```
acceptDocs = AcceptDocs.fromIteratorSupplier(filterScorer, liveDocs, maxDoc)
if (acceptDocs.cost() <= k)                                  // tiny filtered set
    return exactSearch(ctx, acceptDocs.iterator(), timeout)
results = approximateSearch(ctx, acceptDocs, visitLimit, collectorManager)
if (results.totalHits.relation == EQUAL_TO && results.scoreDocs.length >= k)
    return results                                            // ANN completed and found k
return exactSearch(ctx, acceptDocs.iterator(), timeout)       // ANN early-terminated / short -> EXACT
```

So Lucene already: (a) does exact for `cost <= k`; (b) runs ANN; (c) **falls back to exact when ANN
hits its visit limit (early-terminated) or returns < k, reusing the same filter `AcceptDocs`** and
guaranteeing exact results. `OSKnnFloatVectorQuery` overrides only `mergeLeafResults`, so OpenSearch
inherits this fallback unchanged. What it does **not** catch is ANN that *completes* (`relation ==
EQUAL_TO`) with poor global recall — the `contig_rand` case.

### API visibility (Lucene 10.5.0)

| API / signal | Available | Visibility | Reusable by OpenSearch | Notes |
| ------------ | --------- | ---------- | ---------------------- | ----- |
| `KnnCollector.visitedCount()` / `earlyTerminated()` / `visitLimit()` | yes | public interface | yes | graph work + early-termination observable |
| `TopKnnCollector(k, visitLimit)` | yes | public ctor | yes | lets you cap the probe budget |
| `LeafReader.searchNearestVectors(field, target, KnnCollector, AcceptDocs)` | yes | public | yes | run HNSW with a custom collector |
| `AcceptDocs.fromLiveDocs(Bits, maxDoc)` | yes | public static | yes | build accept set from filter bits |
| `AbstractKnnVectorQuery.exactSearch` / `approximateSearch` / `getLeafResults` | partial | protected / private | no (private) | fallback logic is `private`; not overridable |
| Resumable HNSW traversal state (continuation) | **no** | — | no | one-shot search; no continuation API |
| HnswGraph topology / per-node neighbours | no | codec-internal | no | requires casting to codec reader |
| Rejected-candidate count | no | — | no | not exposed by the collector |

### Probe-budget results (100k/dim128/k10/ef100, 30 queries, 4 distributions × 4 cardinalities)

| Budget | Probe p50 µs | % of full ANN visits | Selection acc | Mean regret µs | p95 regret µs | Recall viol. | Slower-than-both |
| -----: | -----------: | -------------------: | ------------: | -------------: | ------------: | -----------: | ---------------: |
| 50 | 408 | 10.0 % | 50.0 % | 715 | 2328 | 0 | 11/16 |
| 100 | 481 | 18.6 % | 50.0 % | 787 | 2400 | 0 | 11/16 |
| 250 | 715 | 46.1 % | 50.0 % | 1021 | 2612 | 0 | 11/16 |
| 500 | 1035 | 67.4 % | 50.0 % | 1293 | 3114 | 0 | 11/16 |
| 1000 | 1574 | 84.8 % | 75.0 % | 1630 | 5310 | 2 | 9/16 |

- **`earlyTerminated` saturates at 1.00** for all scenarios at budgets ≤ 500, and `collected < k` even
  for healthy correlated filters — so the probe rule degenerates to "always EXACT": **50 % selection
  accuracy, below the 73 % cardinality baseline.**
- It only becomes discriminative (75 %) at budget 1000, which consumes **84.8 % of the full ANN's
  visits** — probe + fallback is then near-duplicate work, and **net slower than both raw paths in
  9–11 of 16 scenarios**.
- **The probe cannot predict recall loss.** The two real recall-loss cases (`contig_rand` recall
  0.80 / 0.72) show probe signals identical to the 14 healthy cases.
- **Continuation is impossible** — no resumable traversal state — so a probe is pure duplicate work.

### Phase-8 verdict: Outcome 5 (not economically useful), reinforced by Outcome 3

An additional online probe is **not economically useful**: at cheap budgets the visit signals don't
discriminate; at a discriminative budget the probe ≈ full ANN cost (duplicate work, net-slower-than-both
in the majority of cases); and it cannot detect the recall-loss case at all. Meanwhile **Lucene's
built-in post-ANN exact fallback already covers the "slow/incomplete ANN → exact" case** (Outcome 3),
reusing the filter and guaranteeing exactness, and OpenSearch inherits it. Recommendation: retain only
bounded exact eligibility (MATCH_NONE + small-filter exact) on top of Lucene's native ANN + fallback; do
not pursue online probing. **No production change; all iteration-5 code is test-only.**

## Hard cap on exact candidates — iteration 7 (`BoundedExactHardCapBenchmarkTests`)

`candidateLimit = max(1000, 10·k)` lets a large `k` trigger huge exact scans (`k = 10,000 → 100,000`
candidates). The exact path does **no `QueryTimeout` check mid-scan**, so such a scan is uninterruptible.
Worst-case exact latency vs candidate cardinality (100k corpus, random/worst-case filter; dim 768 is the
clean worst — dim 128's low-cardinality cells are JIT-cold outliers):

| Cardinality | dim384 p95 | dim768 p95 | dim768 MAX |
| ----------: | ---------: | ---------: | ---------: |
| 5,000 | 6.6 ms | 9.9 ms | 10.1 ms |
| 10,000 | 12.1 ms | 19.0 ms | 19.0 ms |
| 25,000 | 25.8 ms | 45.0 ms | 46.0 ms |
| 50,000 | 41.4 ms | 80.5 ms | 81.4 ms |
| 100,000 | 63.8 ms | **153.4 ms** | **171.6 ms** |

Candidate caps (worst permitted p95/MAX across dims):

| Hard cap | worst permitted card | worst p95 | worst MAX |
| -------: | -------------------: | --------: | --------: |
| none (current) | 100,000 | 153 ms | 172 ms |
| 2,500 | 2,500 | 6.6 ms | 9.4 ms |
| 5,000 | 5,000 | 13.8 ms* | 13.9 ms* |
| **10,000 (selected)** | 10,000 | ~19 ms | ~19 ms |
| 25,000 | 25,000 | 45.0 ms | 46.0 ms |

(*the 5,000-row worst p95 of 13.8 ms is a dim-128 JIT-cold artifact; the clean dim-768 value is 9.9 ms.)

**Selected `HARD_MAX_EXACT_CANDIDATES = 10,000.**` It is a no-op for every `k ≤ 1000` (`max(1000, 10k) ≤
10000`), so it changes nothing for the common case and only bites the pathological large-`k` regime,
cutting the worst-case exact p95 from ~153 ms to ~19 ms (~8×) with ≤ 40 KB candidate memory. Formula
(overflow-safe): `candidateLimit = min(10000, max(1000, multiplyExact(10, k)))`.

## What a result would need to show before changing the threshold

- Exact latency < ANN latency across the selective band, **and**
- Exact recall = 1.0 (by construction) while ANN recall drops for very selective filters (the known
  "filtered ANN degrades when the graph is mostly filtered out" failure mode), **and**
- Cardinality-estimation overhead (`count`) is small relative to the query it saves.

Do **not** claim `1,000` or `10 × k` is optimal without these numbers. The crossover is expected to
depend on dimension, `ef_search`, segment count, and filter cost.
