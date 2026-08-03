# ACORN-γ for native Faiss HNSW — Prototype Design

Research prototype for OpenSearch k-NN. Determines whether an ACORN-γ graph and
ACORN-aware traversal materially improve the latency/recall trade-off of
**filtered** native Faiss HNSW search, versus (A) stock Faiss `IDSelector`
search, (B) ACORN-1 traversal over a standard graph, (D/E) larger-`M` HNSW, and
(F) exact filtered search.

Status: **research prototype**. Test-only build & configuration; no production
mapping, no backward-compatible serialization, native Faiss only, CPU HNSW,
float `IndexHNSWFlat`, top-k, L2 + inner-product, single-vector data. Deferred
(per scope): SQ/PQ, byte/binary, nested/multi-vector, radius search, IVF,
GPU/CAGRA, automatic planner selection, production mapping.

---

## 1. Current native Faiss filtered-search architecture

Bundled Faiss: **1.11.0** (`jni/external/faiss`, submodule pin
`5616caad…`), patched by `jni/cmake/init-faiss.cmake` with
`jni/patches/faiss/0001-0010`. Those 10 patches cover multi-vector, IVFPQ
precompute sharing, range-search params, binary vectors, level-0 multi-vector
search, binary nested search, SQ struct tweaks, a write-index skip, a CAGRA
radial fix, and a selective throw. **None implement ACORN / two-hop /
predicate-subgraph traversal** → confirmed no pre-existing ACORN support in the
bundled or upstream-pinned Faiss.

### Filtered call chain (native Faiss HNSW)

```
KNNWeight.searchLeaf                         src/main/java/org/opensearch/knn/index/query/KNNWeight.java:311
  → getFilteredDocsBitSet (filter BitSet)                                    :380
  → gate isFilteredExactSearchPreferred      (exact if filterIdsCount<=k, or  :623
      cost model MAX_DISTANCE_COMPUTATIONS >= card*dim, or index setting)
  → DefaultKNNWeight.doANNSearch             .../query/DefaultKNNWeight.java:49
  → FilterIdsSelector.getFilterIdSelector    .../query/FilterIdsSelector.java:78
        BITMAP(0): raw Lucene FixedBitSet long[]   | BATCH(1): long[] of docids
  → JNIService.queryIndex                     .../jni/JNIService.java:287
  → FaissService.queryIndexWithFilter         .../jni/FaissService.java:311
  → JNI Java_..._queryIndexWithFilter         jni/src/org_opensearch_knn_jni_FaissService.cpp:431
  → QueryIndex_WithFilter                      jni/src/faiss_wrapper.cpp:779
        IDSelectorJlongBitmap (custom, :44) | faiss::IDSelectorBatch
        SearchParametersHNSW{ efSearch, sel=idSelector, grp=parentGrouper }   :817-830
  → IndexIDMap::search → IndexHNSW::search → HNSW::search                     :843
        selector checked in faiss::search_from_candidates   faiss/impl/HNSW.cpp:592
```

**Key architectural fact that motivates ACORN.** In stock Faiss
`search_from_candidates`, the selector gates only the **result heap**
(`add_to_heap`, `HNSW.cpp:671`), but **every** neighbour — pass or fail — is
still pushed to the beam (`candidates.push(idx,dis)`, `HNSW.cpp:680`). So the
traversal is *filter-agnostic*: it explores the graph by geometry and filters
the output. On a **selective** filter the result threshold never tightens
(few/no results), so the ef-bounded stop condition
(`count_below(d0) >= efSearch`) triggers late — search over-explores — yet still
converges to the query's geometric neighbourhood, which may contain **no**
predicate-passing vectors. Result: high cost *and* low recall when the valid set
is concentrated away from the query. This is precisely the regime ACORN targets.

- IDs: vectors are added in doc order via `IndexIDMap::add_with_ids`
  (`faiss_index_service.cpp:140`), so **HNSW internal ordinal == Lucene docId**;
  the selector's ids are meaningful directly on internal ordinals.
- `omp_set_num_threads(1)` at `faiss_wrapper.cpp:804`: Faiss runs single-threaded
  per query; parallelism is per-segment above Faiss.
- Exact filtered search is **not** in C++; it is Java
  (`ExactSearcher.searchLeaf`, `.../exactsearch/ExactSearcher.java:97`), scoring
  filtered candidates through a Lucene `VectorScorer`.

### HNSW internals the prototype builds on (`faiss/impl/HNSW.{h,cpp}`)

- Flat CSR-like link store: `neighbors[]` + `offsets[]` +
  `cum_nneighbor_per_level[]`; level-0 degree `2M`, upper `M`
  (`set_default_probas`). `neighbor_range(node,level,&b,&e)` slices a node's list.
- Build: `add_with_locks` → `add_links_starting_from` → `search_neighbors_to_add`
  (greedy, efConstruction) → `shrink_neighbor_list` (RNG diversification heuristic)
  → `add_link` (reverse edges, re-prune).
- Search: upper-level `greedy_update_nearest` then level-0
  `search_from_candidates` (bounded `MinimaxHeap`).
- `SearchParametersHNSW{ efSearch, check_relative_distance, bounded_queue }`
  inherits `sel` (IDSelector) and `grp`.

---

## 2. ACORN traversal design (Phase 2) — `research/acorn/src/acorn_hnsw.cpp`

Selector-aware ACORN predicate-subgraph traversal, ported from the ACORN
reference (`TAG-Research/ACORN`, a Faiss fork:
`hybrid_search_from_candidates` / `hybrid_greedy_update_nearest`) to Faiss
1.11.0, with the filter predicate expressed as **`faiss::IDSelector::is_member(id)`**
instead of the reference's dense `char* filter_map`. Implemented in the Faiss
HNSW layer (a standalone module over `faiss::HNSW`), **not** in
`faiss_wrapper.cpp`.

```cpp
enum class FilteredHnswSearchMode { STANDARD, ACORN };
```

`STANDARD` calls stock `faiss::HNSW::search` (baselines A, D); `ACORN` runs the
traversal below (baselines B, C, E). The traversal operates over **either** a
standard `faiss::HNSW` graph or an ACORN-γ graph — the only per-index knobs it
reads are `gamma` and `M_beta`.

### Traversal (level-0 bounded BFS, `acorn_search`)

For each beam node `v0` popped from the `MinimaxHeap` (capacity
`max(efSearch,k)`), iterate its neighbour list `[begin,end)`; for neighbour `v1`
at list position `p = j-begin`:

1. **Predicate test** `pass(v1) = sel->is_member(v1)`. If it passes, count it
   toward `num_found`.
2. **Direct enqueue.** If `v1` passes and is unvisited: mark visited, compute
   `d = qdis(v1)`, push to the result max-heap and to the beam. A **failing**
   `v1` is *not* enqueued and *not* distance-computed (unlike stock Faiss, which
   routes through it).
3. **Two-hop bridge.** If `p >= M_beta` (γ>1) **or** `gamma==1` (ACORN-1), expand
   *through* `v1` — pass **or** fail — into `v1`'s neighbours `v2`: skip `v2`
   that fail the predicate; for passing, unvisited `v2` compute `d2`, push to
   results + beam. This is how a rejected node acts as a **bridge** to reach
   predicate-passing two-hop neighbours, emulating the denser predicate subgraph.
4. **Expansion bound.** Once `num_found >= 2*M` passing neighbours are seen for
   `v0`, stop expanding that node (prevents blow-up).

Stopping: stock relative-distance rule `candidates.count_below(d0) >= efSearch`
(default), or `nstep > efSearch` when `check_relative_distance` is off (records a
visit-limit termination). Duplicate suppression via `VisitedTable` for passing
nodes. Distances for inner-product are negated (`NegativeDistanceComputer`) so
the HNSW `CMax` heap semantics hold, then reverted at output; results are sorted
best-first.

### Filtered upper-level descent (`acorn_greedy_descent`)

Greedy from the global `entry_point` down through levels `max_level..1`, moving
to the nearest **passing** neighbour. γ>1 skips failing neighbours; γ==1 bridges
through them. Escape clause: if the current `nearest` itself fails the predicate,
**any** passing neighbour is accepted so the descent leaves a non-passing region
and lands in the predicate subgraph before the level-0 BFS.

**ACORN-1 vs ACORN-γ at search time.** ACORN-1 (`gamma==1`) expands through
*every* neighbour (max reach, higher cost). ACORN-γ (`gamma>1`) expands only
neighbours past list position `M_beta`, relying on the pre-densified graph;
`M_beta` is the single knob dividing "filter directly" (nearest, precise) from
"two-hop expand" (tail, reconstruct density).

### Per-query statistics collected

`dist_computations, nodes_visited, first_hop_expansions, second_hop_expansions,
selector_checks, accepted, rejected, hops, visit_limit_terminations,
results_returned` — plus wall-clock for percentiles/QPS in the harness.

---

## 3. ACORN-γ graph construction (Phase 3) — `research/acorn/src/acorn_gamma_builder.cpp`

Predicate-agnostic denser graph, faithful to the ACORN reference
(`set_default_probas`, `search_neighbors_to_add`, `shrink_neighbor_list`,
`add_link`). Deterministic single-thread build (fixed seeds).

```cpp
struct AcornGammaBuildParameters { bool enabled; int gamma; int M;
                                   int max_degree; int M_beta; int efConstruction; int seed; };
```

Construction rules (all predicate-agnostic — no filter values consulted):

| aspect | level 0 | levels > 0 |
|---|---|---|
| stored degree cap | `M_beta + floor(1.5·M)` | `M·γ` |
| candidates collected | `2·M·γ` | `M·γ` |
| greedy stop | relaxed for γ>1 (keep expanding to `2Mγ`) | same |
| pruning | **two-hop compression** (below) | keep nearest `M·γ` |

`efConstruction = M·γ` (auto). Constraint `M_beta <= 2·M·γ` (throws otherwise).

**Two-hop compression prune (level 0, `acorn_shrink`).** Enumerate candidates
nearest→farthest. Keep the first `M_beta` unconditionally. Beyond that, **prune a
candidate iff it is already two-hop reachable** from an already-kept node
(tracked in a `neigh_of_neigh` set that folds in each kept node's level-0
neighbours); otherwise keep it. This yields a sparse-but-navigable level-0 list
and is where the ACORN "compression" lives. Build records
`candidate_edges_considered / edges_retained / edges_pruned`.

`gamma=1` reproduces standard-HNSW density behaviour as the closest control (used
as the ACORN-1 graph is a *standard* Faiss graph — see §5). A **larger-`M`**
standard graph is benchmarked separately so ACORN-specific value is distinguished
from simple densification.

### Deviations from the paper / reference (§4 requirement)

1. **Predicate = `IDSelector`, not `char* filter_map`.** Intentional, to match
   the OpenSearch efficient-filter path and reuse `is_member`. Semantically
   identical (a per-id boolean); costs one virtual call per test.
2. **Graph stored as plain `faiss::HNSW`** (not a bespoke `ACORN` struct with a
   `NeighNode`/metadata array). ACORN-γ and standard graphs share one type, so
   the traversal runs over either and serialization is uniform. `gamma`/`M_beta`
   travel alongside the graph in `AcornIndex`.
3. **Upper-level `add_link` overflow guard.** The reference writes the full
   candidate set back at upper levels without shrinking (can overflow when a list
   is exactly full); the prototype truncates to the `M·γ` nearest. Rarely
   triggered (upper capacity is large); prevents a latent buffer overrun.
4. **Single-threaded deterministic build** (no OpenMP), for reproducibility and
   controlled measurement. Faiss's own level assignment (`rng(12345)`) and
   per-level shuffle (`rng2(789)`) are preserved so ordering matches Faiss.
5. **Bridge nodes are not marked visited** (faithful to the reference): a failing
   two-hop bridge may be re-expanded from different beam nodes. Documented cost;
   the inner `v2` loop still skips visited passing nodes.
6. **No OpenMP / BLAS in the prototype binary.** A no-op `omp.h` shim + macOS
   Accelerate; build path only. The production JNI build is unaffected.

---

## 5. Files created / changed

**All prototype code is isolated under `research/acorn/` — no production Java or
`jni/src` changes.** The bundled Faiss submodule is left pristine (no new patch);
the prototype compiles a minimal Faiss subset and adds ACORN as separate
translation units in the `faiss`/`acorn` namespaces. For a production build this
would become a Faiss patch `0011-Add-ACORN-hnsw-search-and-acorn-gamma-build`.

```
research/acorn/
  ACORN_GAMMA_DESIGN.md          this document
  ACORN_GAMMA_BENCHMARKS.md      methodology + results + analysis + verdict
  src/
    acorn.h                      public API: modes, params, stats, build/search/serialize
    acorn_hnsw.cpp               Phase 2: STANDARD wrapper + ACORN traversal + exact oracle
    acorn_gamma_builder.cpp      Phase 3: standard + ACORN-γ build + prototype serialization
    shim/omp.h                   no-op OpenMP shim (test build only)
    smoke.cpp                    end-to-end smoke test
    tests.cpp                    correctness suite (33 assertions)
    bench.cpp                    benchmark harness (baselines A–F)
    probe.cpp                    minimal build probe
  scripts/
    build.sh                     clang++ build (minimal Faiss subset + Accelerate)
    faiss_sources.txt            Faiss subset manifest
    run_all.sh                   benchmark driver
  raw-results/                   CSV outputs + trace notes
  build/                         objects + binaries (gitignored)
```

## 6. Test-only configuration mechanism

No public OpenSearch mapping is added. The prototype is driven entirely by the
native structs `AcornGammaBuildParameters{enabled,gamma,M,max_degree,M_beta,
efConstruction,seed}` and `FilteredHnswSearchMode{STANDARD,ACORN}`, selected
directly in the C++ harness (`bench.cpp`, `tests.cpp`). This satisfies the
"test-only index description / native parameter" allowance without touching
production API surface. A production surfacing (if justified) is proposed in the
benchmarks doc.

## 7. Prototype serialization format

`save_acorn_index` / `load_acorn_index` write a compact, **prototype-only** file:
magic `0xAC0F0001`, header `{d, metric, is_acorn_gamma, gamma, M, M_beta}`, the
raw vectors, then the `faiss::HNSW` arrays (`assign_probas`,
`cum_nneighbor_per_level`, `levels`, `offsets`, `neighbors`, `entry_point`,
`max_level`). Loading a file without the marker **throws loudly**. This is
explicitly *not* Faiss- or production-compatible; production would extend Faiss
HNSW serialization behind a version flag.

## 8. Build & reproduce

```bash
bash research/acorn/scripts/build.sh smoke tests bench   # clang++ + Accelerate, no cmake
./research/acorn/build/tests                             # 33/33 correctness assertions
bash research/acorn/scripts/run_all.sh                   # full benchmark matrix → raw-results/*.csv
```
No cmake, no BLAS install, no libomp: a minimal subset of the pinned Faiss 1.11.0
source is compiled directly with a no-op OpenMP shim and macOS Accelerate, giving
a deterministic single-thread build ideal for controlled latency measurement.
