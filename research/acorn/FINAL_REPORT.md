# ACORN-γ for native Faiss HNSW — Final Research Report

Authoritative summary for the RFC decision. Detail in
[`ACORN_GAMMA_DESIGN.md`](ACORN_GAMMA_DESIGN.md) (architecture + algorithms) and
[`ACORN_GAMMA_BENCHMARKS.md`](ACORN_GAMMA_BENCHMARKS.md) (methodology + full
results). Prototype code: [`src/`](src/); raw data: [`raw-results/`](raw-results/).

The 18 required items:

### 1. Current native Faiss filtered-search architecture
Faiss **1.11.0** (submodule pin `5616caad`), patched by `init-faiss.cmake`
(`0001-0010`; none ACORN-related). Path: `KNNWeight.searchLeaf` → filter BitSet →
exact/ANN gate (`filterIdsCount≤k` or distance-cost model) →
`FilterIdsSelector` (BITMAP/BATCH) → `JNIService` → `FaissService.queryIndexWithFilter`
→ `QueryIndex_WithFilter` (`faiss_wrapper.cpp:779`) → `IDSelectorJlongBitmap`/
`IDSelectorBatch` → `SearchParametersHNSW.sel` → `IndexHNSW::search` →
`search_from_candidates` (`HNSW.cpp:592`). **Selector gates only the result heap;
the beam pushes every neighbour** — traversal is filter-agnostic. Exact fallback
is in Java (`ExactSearcher`), not C++. HNSW ordinal == Lucene docId. Full trace:
[`raw-results/phase1-trace-notes.md`](raw-results/phase1-trace-notes.md).

### 2. ACORN traversal design
`FilteredHnswSearchMode{STANDARD,ACORN}`. ACORN = predicate-subgraph BFS over
`faiss::HNSW`: enqueue only predicate-passing neighbours; **bridge through
rejected neighbours** (two-hop) at list position ≥ `M_β` (γ>1) or always (γ=1) to
reach passing two-hop nodes; per-node expansion capped at `2M` found; stock ef /
relative-distance stop; filtered greedy descent with a "nearest fails predicate →
accept any passing" escape. Predicate = **`IDSelector::is_member`**. STANDARD mode
= stock `faiss::HNSW::search`. Design doc §2; code
[`src/acorn_hnsw.cpp`](src/acorn_hnsw.cpp).

### 3. ACORN-γ graph-construction design
Predicate-agnostic densification: level-0 degree `M_β+1.5M`, upper `M·γ`;
collect `2Mγ` candidates; relaxed greedy stop for γ>1; **two-hop compression
prune** (keep nearest `M_β`, drop already-two-hop-reachable); `efConstruction=Mγ`;
deterministic single-thread; records edges considered/retained/pruned. Design doc
§3; code [`src/acorn_gamma_builder.cpp`](src/acorn_gamma_builder.cpp).

### 4. Deviations from the paper
(1) `IDSelector` predicate instead of dense `char*` map; (2) graph stored as plain
`faiss::HNSW` (one type for both); (3) upper-level `add_link` truncates instead of
the reference's overflow-prone unshrunk write; (4) single-thread deterministic
build; (5) bridge nodes not marked visited (faithful); (6) no OpenMP/BLAS in the
test binary. Design doc §4.

### 5. Exact files changed
**No production Java / `jni/src` changes.** All under
[`research/acorn/`](.): `src/{acorn.h,acorn_hnsw.cpp,acorn_gamma_builder.cpp,
smoke.cpp,tests.cpp,bench.cpp,probe.cpp,shim/omp.h}`, `scripts/{build.sh,
run_all.sh,analyze.py,faiss_sources.txt}`, the two design/benchmark docs, and
`raw-results/`. Faiss submodule left pristine (no new patch); production would add
Faiss patch `0011-Add-ACORN-hnsw-search-and-acorn-gamma-build`.

### 6. Test-only configuration mechanism
Native structs only: `AcornGammaBuildParameters{enabled,gamma,M,max_degree,
M_beta,efConstruction,seed}` + `FilteredHnswSearchMode`, selected directly in the
C++ harness. No OpenSearch mapping added. Design doc §6.

### 7. Correctness results
`research/acorn/build/tests`: **33/33 assertions pass** — degree bounds,
valid/well-formed adjacency, build determinism, selector correctness, 0/<k/
all-match edge cases, duplicate suppression, cycles/termination, serialization
round-trip + bad-magic rejection, 8-thread concurrency parity, **brute-force
parity** at 3 selectivities, ACORN-1/ACORN-γ mode separation.

### 8. Full benchmark methodology
Real Faiss `IndexHNSWFlat`; clustered Gaussian data; 7 filter relationships;
selectivities 0.5→0.001 with absolute candidate counts; corpus 100K + 1M; dims
128 + 768; L2 + IP(cosine); k∈{10,100}; ef∈{50,100,250,500}; recall@k vs exact
oracle; per-query percentiles + 11 counters. Baselines A–F. Benchmarks doc §2.

### 9. Search latency & recall tables
Benchmarks doc §3 (traversal Δrecall), §4 (decision framework, latency+recall),
§7 (γ-recall). Headline: ACORN traversal **+0.3…+0.7 recall** on clustered/
correlated filters, **−0.6…−0.8 on random**; exact fastest+perfect at low cand.

### 10. Index-build & memory-overhead tables
Benchmarks doc §6: ACORN-γ γ=32 = **8.7× build time, 2.2× memory** vs standard;
larger-M (M=48) builds in the *same* 10 s at comparable memory.

### 11. ACORN-1 vs ACORN-γ
Benchmarks doc §5. ACORN-γ beats ACORN-1 only by being denser; the same gain is
had from ACORN traversal over a larger *standard* graph. ACORN-1 over base-M wins
**0** workloads outright.

### 12. ACORN-γ vs larger-M
Benchmarks doc §5/§4. Larger-M+ACORN wins **8–10** workloads to ACORN-γ's **3**,
builds 3–10× faster, behaves predictably. ACORN-γ construction not repeatably
justified.

### 13. Best & worst workloads
**Best for ACORN-γ:** cluster/correlated at 0.1–1% selectivity (beats even exact
2–3×, e.g. cluster 0.5%: 52 µs vs 140 µs). **Worst:** random filters (ACORN
traversal −0.6…−0.8 recall), and any selectivity where mis-tuned γ → recall 0.
**Best overall:** exact at low candidate counts (23/40 workloads).

### 14. Recommended γ
**None generalizes** (Benchmarks §7): recall is all-or-nothing and non-monotonic
in γ (cluster 1%: γ=8→0.98, γ=16→1.0, γ=32→0.0). If forced, γ ≈ 1/selectivity for
the *specific* clustered workload — but robustness cannot be guaranteed.

### 15. Faiss upstreaming feasibility
Feasible (reference ACORN is itself a Faiss fork; this is a clean HNSW-layer
module). The defensible upstream unit is the **traversal** (an `IDSelector`-aware
predicate-subgraph search mode), not the unstable γ graph. Benchmarks §12.

### 16. Production gaps remaining
Prototype serialization (not Faiss-compatible); no Java mapping / SQ / PQ / byte /
binary / nested / IVF / GPU; γ auto-selection unsolved (likely unsolvable
robustly); no real-embedding validation; scale ≤1M vs paper's 1M–100M.
Benchmarks §13.

### 17. Proposed OpenSearch user-facing mapping
**Not proposed** — results don't justify it (item 11.11 of Benchmarks). The
low-selectivity regime is already served by OpenSearch's exact fallback. Had it
been justified: a test-only `method.parameters.filter_search={mode,gamma}` +
query-time `ef_search`, explicitly *not* an auto-planner.

### 18. Final recommendation
The **ACORN traversal** meaningfully improves filtered native Faiss HNSW for
*metadata-correlated* (clustered/correlated) filters — the common OpenSearch case
— recovering 0.3–0.7 recall where stock filtered HNSW routes away from the valid
set. But the benefit is **conditional** (hurts random filters), the dedicated
**ACORN-γ graph does not repeatably beat a larger-M standard graph under ACORN
traversal**, costs up to ~9× to build, and is **operationally fragile**
(all-or-nothing, non-monotonic, un-generalizable γ). The regime where ACORN-γ is
uniquely best is narrow, and the broader low-selectivity regime is already owned
by OpenSearch's exact fallback. A full ACORN-γ mapping feature is **not**
justified on this evidence; the portable win is the ACORN *traversal* as an
optional filtered-search mode (ACORN-1), worth further prototyping on
real-embedding workloads at ≥10M scale. The 1M scale run reinforces this
(ACORN-γ unique wins shrink 3→1; exact + larger-M+ACORN absorb the rest), and the
result is consistent across 100K/1M × d128/d768 × L2/IP.

```
ACORN-1 ONLY
```
