# Phase 1 — Existing native Faiss filtered-search trace (notes)

Faiss bundled version: **1.11.0** (submodule pinned at `5616caad4cea4e8326cdef90aa177bd618af8531`,
`jni/external/faiss`). Patched by k-NN via `jni/cmake/init-faiss.cmake` applying
`jni/patches/faiss/0001..0010` (multi-vector, IVFPQ precomp sharing, range-search params,
binary vector, multi-vector level_0 search, binary nested, SQ struct tweaks, write skip,
CAGRA radial fix, selective throw). **None** of the 10 patches implement ACORN / two-hop /
predicate-subgraph traversal → confirmed no pre-existing ACORN support.

## Call chain (filtered Faiss HNSW)
1. `KNNWeight.searchLeaf` (`src/main/java/org/opensearch/knn/index/query/KNNWeight.java:311`)
   builds filter `BitSet` (`getFilteredDocsBitSet:380`), cardinality gate at :320.
2. Strategy gate `isFilteredExactSearchPreferred` (:623):
   - `filterIdsCount <= k` → exact (:656)
   - index setting `ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD` (:666)
   - cost model `MAX_DISTANCE_COMPUTATIONS >= filterCardinality * dim` (:651)
3. `DefaultKNNWeight.doANNSearch` (`DefaultKNNWeight.java:49`) →
   `FilterIdsSelector.getFilterIdSelector` (`FilterIdsSelector.java:78`) picks BITMAP(0)/BATCH(1):
   - dense `FixedBitSet` → BITMAP, passes raw `long[]` word array
   - sparse → BATCH `long[]` of docids (when `card*8 <= maxId/8`)
4. `JNIService.queryIndex` (:287) → `FaissService.queryIndexWithFilter` (:311) →
   JNI `Java_..._queryIndexWithFilter` (`org_opensearch_knn_jni_FaissService.cpp:431`) →
   `QueryIndex_WithFilter` (`faiss_wrapper.cpp:779`).
5. Selector built (`faiss_wrapper.cpp:809-815`): BITMAP→`IDSelectorJlongBitmap` (custom, :44,
   Lucene FixedBitSet bit layout), BATCH→stock `faiss::IDSelectorBatch`.
6. `SearchParametersHNSW hnswParams; hnswParams.efSearch=...; hnswParams.sel=idSelector.get();`
   (`:817-830`), optional `hnswParams.grp` for nested parent grouping.
7. `indexReader->search(1, q, k, dis, ids, searchParameters)` where `indexReader` is
   `faiss::IndexIDMap*` wrapping `IndexHNSW` (`:843`). `omp_set_num_threads(1)` at :804.

## Key facts for the prototype
- **Where the selector is checked during search**: inside `faiss::search_from_candidates`
  (`faiss/impl/HNSW.cpp:592`) — `add_to_heap` lambda at :671 gates *results* with
  `sel->is_member(idx)` but **still pushes every neighbor to the candidate queue** (:680).
  i.e. current Faiss traversal is filter-agnostic; the filter only prunes the result set.
  This is exactly the ACORN-1 opportunity: on sparse filters the ef-bounded beam fills with
  rejected routing nodes and finds too few valid nodes → recall collapses.
- **id identity**: vectors added in doc order via `IndexIDMap::add_with_ids`
  (`faiss_index_service.cpp:140`), so HNSW internal ordinal == Lucene docId; the selector's
  ids are docIds and are meaningful directly on internal ordinals. (IndexIDMap wraps the
  selector with an IDSelectorTranslated internally.)
- **HNSW construction**: `HNSW::add_with_locks` → `add_links_starting_from` →
  `search_neighbors_to_add` (greedy, efConstruction) → `shrink_neighbor_list` (RNG heuristic,
  keeps M / 2M) → `add_link` (reverse edges, re-prune). (`faiss/impl/HNSW.cpp`)
- **Graph storage**: flat `neighbors[]` + `offsets[]` + `cum_nneighbor_per_level[]`; level-0
  degree = 2M, upper = M (`set_default_probas:74`). `SearchParametersHNSW` has efSearch,
  check_relative_distance, bounded_queue, and inherits `sel`, `grp`.
- **Serialization**: `write_HNSW`/`read_HNSW` in `faiss/impl/index_read.cpp`/`index_write.cpp`.
- **Exact filtered search**: NOT in C++. Java `ExactSearcher.searchLeaf`
  (`.../exactsearch/ExactSearcher.java:97`) scores filtered candidates via Lucene VectorScorer.
