/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * EXPERIMENTAL (POC): selector-aware **ACORN-1** filtered traversal for native
 * Faiss HNSW, integrated into the OpenSearch k-NN native layer.
 *
 * SCOPE (read before extending):
 *   - This implements ACORN-1 ONLY: a two-hop, filter-aware, bounded best-first
 *     traversal over a *standard* Faiss HNSW graph (base-level degree ~2*M). It
 *     needs NO build-time graph change.
 *   - It does NOT implement general ACORN-γ. ACORN-γ requires a physically denser
 *     graph built at index time (γ× neighbours + M_beta pruning); that graph does
 *     not exist on a standard OpenSearch index, so a γ>1 request is rejected here
 *     (see ACORN_ERR_UNSUPPORTED_GAMMA) rather than silently mis-traversed.
 *
 * This is the "Faiss-layer" traversal policy behind the OpenSearch
 * `filtered_search_mode: acorn` query parameter. `faiss_wrapper` dispatches here
 * only when mode == acorn; standard traversal is left entirely unchanged.
 *
 * ID SPACE: HNSW traversal runs on internal storage ordinals, but the OpenSearch
 * IDSelector is defined over EXTERNAL ids (Lucene doc ids). Every selector call
 * therefore translates internal→external via IndexIDMap::id_map, exactly as
 * faiss::IDSelectorTranslated does on the stock path. See translate_and_test().
 */
#ifndef OPENSEARCH_KNN_ACORN_HNSW_H
#define OPENSEARCH_KNN_ACORN_HNSW_H

#include <cstdint>

#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/impl/IDSelector.h"

namespace knn_jni {
namespace acorn {

// Status codes returned by search() (negative = error; >=0 = number of results).
// The JNI caller maps a negative return to a thrown error; this keeps search() a
// pure status-return function (no exceptions from the traversal itself).
enum : int {
    ACORN_ERR_ARGS = -1,               // null pointer / k<=0 / efSearch<=0
    ACORN_ERR_UNSUPPORTED_GAMMA = -2,  // gamma != 1 (ACORN-γ not supported here)
    ACORN_ERR_INDEX = -3,              // wrapped index is not IndexHNSW / storage null / id_map inconsistent
};

// Benchmark instrumentation. Populated only when collect_stats=true (negligible
// overhead otherwise). Field names state exactly what is counted so a metric is
// never mistaken for something it is not.
struct AcornSearchStats {
    uint64_t segment_vector_count = 0;        // index ntotal
    uint64_t selector_checks = 0;             // is_member() calls (post ID translation)
    uint64_t selector_matches = 0;            // is_member() == true
    uint64_t selector_rejections = 0;         // is_member() == false
    uint64_t distance_computations = 0;       // DistanceComputer evaluations
    uint64_t candidates_expanded = 0;         // frontier nodes popped and expanded
    uint64_t bridge_nodes_expanded = 0;       // DISTINCT nodes whose 2-hop adjacency was walked as a bridge
    uint64_t bridge_reexpansions_avoided = 0; // times a bridge was skipped because already expanded (fix: was a bug)
    uint64_t graph_neighbors_examined = 0;    // 1-hop + 2-hop neighbour slots inspected
    uint64_t eligible_discovered = 0;         // UNIQUE filter-passing nodes admitted to results/frontier
    uint64_t frontier_insertions = 0;         // candidates.push() calls
    uint64_t results_returned = 0;            // final result count
    void reset() { *this = AcornSearchStats{}; }
};

// Thread-local stats for the most recent ACORN search on this thread.
AcornSearchStats& last_stats();

// Whether to collect per-query stats. Off by default; enabled by env
// KNN_ACORN_BENCH_STATS=1. Read once.
bool benchmark_stats_enabled();

/**
 * Run ACORN-1 filtered top-k over an IndexIDMap-wrapped IndexHNSW.
 *
 * @param id_map        the IndexIDMap the OpenSearch loader hands to search
 * @param query         query vector (dim = index->d)
 * @param k             top-k (> 0)
 * @param efSearch      beam width (> 0); normalized to max(efSearch, k) internally
 * @param sel           filter predicate over EXTERNAL ids (must be non-null)
 * @param gamma         MUST be 1 (ACORN-1). Any other value returns
 *                      ACORN_ERR_UNSUPPORTED_GAMMA (ACORN-γ is not supported on a
 *                      standard HNSW graph).
 * @param distances     out: k distances, best-first (space-type oriented)
 * @param labels        out: k external ids (id_map translated), -1 padded
 * @param collect_stats fill last_stats() when true
 * @return number of results written (0..k), or a negative ACORN_ERR_* code.
 */
int search(
        const faiss::IndexIDMap* id_map,
        const float* query,
        int k,
        int efSearch,
        const faiss::IDSelector* sel,
        int gamma,
        float* distances,
        faiss::idx_t* labels,
        bool collect_stats);

}  // namespace acorn
}  // namespace knn_jni

#endif  // OPENSEARCH_KNN_ACORN_HNSW_H
