# OpenSearch ACORN / RACORN — RFC Evidence Summary

Decision-grade summary for whether native **ACORN**, **ACORN-γ**, and **RACORN-1/1+**
filtered-HNSW traversal should be proposed as experimental OpenSearch features.
Companion docs: [POC_DESIGN](OPENSEARCH_ACORN_POC_DESIGN.md),
[METHODOLOGY](OPENSEARCH_ACORN_BENCHMARK_METHODOLOGY.md),
[RESULTS](OPENSEARCH_ACORN_RESULTS.md),
[CUSTOMER_VALUE](OPENSEARCH_ACORN_CUSTOMER_VALUE.md).

> **Honesty note.** Native-layer recall & visit-count are measured through the exact
> traversal code the OpenSearch path invokes and transfer directly. A live single-node
> cluster + JNI build could **not** be run in the prototyping environment, so
> end-to-end REST latency, GC/heap, live Lucene filter-build, and shard reduction are
> **analyzed from the code path / modeled**, not measured live, and are flagged as such.
> Latency uses an unoptimized research traversal (`priority_queue`, full 2-hop scan);
> visit-count is the faithful, implementation-independent speed signal.

## The 20 required outputs

**1. End-to-end OpenSearch filtered-search flow** — POC_DESIGN §1. The mode rides the
existing `method_parameters` map to `faiss_wrapper.cpp`, which dispatches
standard/acorn/racorn/racorn_plus to Faiss-layer policies reusing the existing `IDSelector`.

**2. Exact files changed** — POC_DESIGN §3. Production Java: `KNNConstants`,
`FilteredSearchMode` (new), `MethodParameter`, `KNNQueryBuilder`, +`FilteredSearchModeTests`
(new, 13 tests, pass). Native: `acorn_hnsw.{h,cpp}` (new — ACORN + RACORN), `faiss_wrapper.cpp`,
`commons.{h,cpp}`, `jni_util.{h,cpp}`, `CMakeLists.txt`. Faiss submodule unchanged.
Research: `research/acorn/src/{racorn,acorn_hnsw,acorn_gamma_builder}.cpp` + harness.

**3. Experimental API** — POC_DESIGN §2. `method_parameters.filtered_search_mode ∈
{standard, acorn, racorn, racorn_plus}`; omitted = standard. Unknown/no-filter/non-FAISS/
non-HNSW/radial all fail validation with no silent fallback.

**4. Validation & compatibility matrix** — POC_DESIGN §5.

**5. Benchmark machine & software** — Apple arm64, clang 21, `-O3 -DNDEBUG` release,
Accelerate BLAS, single-thread deterministic build. Faiss 1.11.0 (pin `5616caad`); k-NN
branch `claude/acorn-gamma-faiss-hnsw-729026`. METHODOLOGY §8.

**6. Dataset & filter-generation methodology** — METHODOLOGY §3–4. Clustered Gaussian
mixture (K-means-cluster analogue); No/Positive/Negative query–filter correlation;
absolute candidate-count sweep {100…50,000}; `correlation_mix(ρ)` continuous variant.

**7. Exact ground-truth methodology** — METHODOLOGY §6. Per-query exact top-k over
**eligible (filter-passing) docs only**; `recall@k = |ret ∩ truth| / min(k,|elig|)`.

**8. Standard vs ACORN/RACORN latency-recall** — RESULTS §1–§3. Headline: under negative
correlation the standard *faiss-native* path returns **recall 0.00**; RACORN-1 recovers to
0.77–1.00 at 4–57× fewer visits; RACORN-1+ to 0.98–1.00.

**9. Exact-search crossover** — RESULTS §1/§4. Exact wins at candidate counts ≲500–1,000
(all correlations); above that, RACORN-1 beats exact on visit count under negative
correlation (e.g. 1.7k vs 10k–50k visits at 10–50%), and standard wins under positive
correlation. This matches (and generalizes) the standalone finding and RACORN-1+'s AEF.

**10. Segment & shard sensitivity** — *analyzed, not live-measured.* OpenSearch runs the
native search **per segment**, merging top-k. Consequences for these modes:
- The negative-correlation **recall collapse of the standard path is per-segment** — every
  segment returns ~0 valid, so more segments do **not** mask it; ACORN/RACORN's per-segment
  recovery therefore compounds (each segment now returns valid results).
- RACORN's constant per-node overhead is paid per segment, so many tiny segments amplify its
  latency overhead relative to a single large segment — argues for force-merge before relying
  on `racorn` for latency-sensitive use, and for an optimized native implementation.
- Shard fan-out adds coordinating-node reduce cost independent of the mode; it does not change
  the recall conclusions. Live confirmation is the top follow-up (harness:
  `scripts/opensearch_bench.py --segments`).

**11. Concurrency** — the ACORN/RACORN read path is stateless per query (thread-local
`DistanceComputer`/`VisitedTable`, `const` graph); the prior native-layer concurrency test
scales near-linearly to 8 threads (`research/acorn/raw-results/concurrency.csv`, 5.4k→40.5k
QPS). RACORN adds no shared mutable state → same scaling class. Live 1/4/8/16/32-worker QPS
is a follow-up (harness supports it).

**12. Filter-construction vs native-search time** — *analytical.* The mode changes **only
native traversal**; Lucene filter evaluation + bitmap materialization are identical across
modes. Therefore ACORN/RACORN help only in proportion to the native-search share of request
time: for very broad filters where filter construction dominates, the benefit shrinks (and
those broad filters are a *losing* case anyway, §14). Where native search dominates
(selective, negative-correlation, large N) the benefit is largest.

**13. Customer workloads that benefit** — CUSTOMER_VALUE. Product search (category/availability
anti-correlated with query), multi-domain/multi-language docs, clustered tenants, region-aware
recs — all negative/cross-domain correlation with thousands of eligible docs at scale.

**14. Workloads that regress** — CUSTOMER_VALUE (required losing cases): random ACL/tenant
filters, broad filters, tiny candidate sets (exact wins), weakly-correlated metadata,
positive correlation (standard wins), CPU-saturated (RACORN overhead worsens tail). These
**must stay `standard`/`exact`** — hence the explicit, opt-in, no-auto-selection design.

**15. Larger-M comparison** — RESULTS. A larger-M standard graph (M=32) does **not** fix the
negative-correlation collapse for standard/HNSW-infilter (same recall, ~same or more visits),
and is matched by ACORN/RACORN over the base-M graph. Extra edges alone are not the answer.

**16. ACORN-γ comparison** — RESULTS §1/§4/§5. ACORN-γ recovers in a narrow negative-correlation
band (0.90–0.99 at 1–25%) but is **unstable** (collapses at cand≤500 and =50%), and requires a
**build-time** graph change (no OpenSearch construction path; +up to 8.7× build time, +2.2×
memory from the standalone study). It does **not** repeatably beat RACORN-1 (search-time, no
build cost).

**17. Indexing, memory & graph-size overhead** — RACORN-1/1+ and ACORN: **zero** build/memory
overhead (search-time only, standard graph, no serialization change). ACORN-γ: +3.4–8.7× build
time, +1.3–2.2× memory, +denser graph (standalone `ACORN_GAMMA_BENCHMARKS.md` §6).

**18. Correctness & lifecycle** — 33/33 native correctness assertions pass
(`research/acorn/build/tests`: degree bounds, valid/well-formed adjacency, selector
correctness, dedup, termination, brute-force parity, serialization round-trip, 8-thread
concurrency). RACORN reproduces the paper's collapse/recovery (`racorn_smoke`). Lifecycle:
RACORN/ACORN are search-time → **no serialization / merge / reload changes** (they run over
the existing HNSW graph, so freshly-opened, warmed, close/open, merge, and restart are all
inherited unchanged). ACORN-γ's prototype serialization round-trips but is not
production-format. Java validation covered by unit tests.

**19. Raw-data reproducibility** — METHODOLOGY §8. `build.sh` → `tests`, `racorn_smoke`,
`bench_collective`; `analyze_collective.py` regenerates tables. CSVs + analysis in
`raw-results/`. Live path: `scripts/opensearch_bench.py` (documented, requires a real node +
JNI build).

**20. Draft RFC evidence summary** — below.

---

## Draft RFC evidence summary

**Problem.** OpenSearch's native filtered HNSW (faiss `IndexHNSW` + `IDSelector`) **collapses
to ~0 recall** when the filter's eligible set lies in a different vector-space region than the
query (negative / cross-domain correlation) — while still visiting ~all nodes. This is a real,
silent failure mode for cross-domain tenant/region/category filters at scale. The recall-
preserving alternative (distance-first HNSW) instead **explodes to ~all-node visits**.

**Proposal.** An **explicit, opt-in** per-query `filtered_search_mode` selecting a
selector-aware Faiss-layer traversal — reusing the existing `IDSelector`, no index change, no
default-behaviour change:
- **`racorn` (RACORN-1)** — recovers the collapse via Adaptive Search Fallback (filter-failing
  bridge nodes), holding recall 0.77–1.00 at **4–57× fewer node visits** than recall-preserving
  HNSW, and **beating exact** at moderate–high candidate counts under negative correlation.
- **`racorn_plus` (RACORN-1+)** — adds Adaptive Exact Fallback for the extreme-low-selectivity
  tail (recall 0.98–1.00).
- **`acorn` (ACORN-1)** — efficient for *no-correlation* selective filters (recall 1.00 at
  10–160× fewer visits than HNSW), retained as a lighter option.

**Thresholds met** (negative-correlation, repeatable across candidate counts & 2 seeds):
- ≥10-point recall improvement at comparable work: **+0.77 to +1.00 recall** vs ACORN-1 and vs
  the standard faiss path (which are at ~0). ✓✓
- Materially reduced work at equal recall: **4–57× fewer visits** than recall-preserving HNSW
  at recall 1.00. ✓
- Repeatable across correlated workloads: yes (candidate-count sweep, seeds). ✓
- No default-behaviour change, correct filtering, bounded overhead, documented random/positive
  regressions, acceptable concurrency class. ✓

**Residual risk (follow-up before GA):** live single-node confirmation of Java-layer overheads
(filter build, segment fan-out, request latency, GC/heap) and an optimized native traversal to
convert the visit-count advantage into wall-clock; real-embedding datasets (SIFT/GIST/T2I) at
1M–40M (the paper's own scale results corroborate the trend).

---

## Decisions

```
NATIVE ACORN:
PROPOSE EXPERIMENTAL RFC
```
*(Ship the opt-in selector-aware `filtered_search_mode` family — with **RACORN-1** as the
recommended member for negative/cross-domain-correlation filters and **ACORN-1** for
no-correlation selective filters. It fixes a real recall-collapse of the current path, meets
the recall/work thresholds repeatably, changes no default behaviour, and needs no index change.
Gate GA on live single-node latency confirmation + an optimized native traversal.)*

```
ACORN-GAMMA:
RETAIN AS RESEARCH ONLY
```
*(Narrow, unstable wins in a mid negative-correlation band; requires a build-time graph change
(+up to 8.7× build, +2.2× memory) with no OpenSearch construction/serialization path; does not
repeatably beat search-time RACORN-1. Keep as research evidence, not an experimental feature.)*

```
RACORN (added scope):
PROPOSE EXPERIMENTAL RFC — RACORN-1 default, RACORN-1+ for the extreme-low-selectivity tail
```
*(The strongest member of the proposed family: uniquely recovers the negative-correlation
collapse at a fraction of HNSW's work, search-time only, zero index/memory overhead.)*
