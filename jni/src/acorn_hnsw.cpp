/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * EXPERIMENTAL (POC) ACORN-1 filtered traversal for native Faiss HNSW.
 *
 * ACORN-1-style bounded best-first traversal over a STANDARD Faiss HNSW graph:
 * only filter-passing nodes become results; filter-rejected nodes act as two-hop
 * routing bridges (each expanded at most once). See acorn_hnsw.h for scope/ID-space
 * contract. Algorithm cross-checked against research/acorn/ (real Faiss 1.11.0).
 */
#include "acorn_hnsw.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "faiss/MetricType.h"
#include "faiss/impl/AuxIndexStructures.h"  // VisitedTable
#include "faiss/impl/DistanceComputer.h"    // DistanceComputer, NegativeDistanceComputer
#include "faiss/impl/HNSW.h"
#include "faiss/utils/Heap.h"               // maxheap_*

namespace knn_jni {
namespace acorn {

using faiss::DistanceComputer;
using faiss::HNSW;
using faiss::IDSelector;
using faiss::VisitedTable;
using storage_idx_t = HNSW::storage_idx_t;

AcornSearchStats& last_stats() {
    static thread_local AcornSearchStats stats;
    return stats;
}

bool benchmark_stats_enabled() {
    static const bool enabled = []() {
        const char* e = std::getenv("KNN_ACORN_BENCH_STATS");
        return e != nullptr && std::string(e) == "1";
    }();
    return enabled;
}

namespace {

// For similarity metrics (inner product / cosine) wrap the distance computer so
// that "smaller is better" holds for the HNSW heaps, exactly as faiss IndexHNSW
// does internally. Distances are reverted once at finalization.
DistanceComputer* make_dc(const faiss::Index* storage) {
    if (faiss::is_similarity_metric(storage->metric_type)) {
        return new faiss::NegativeDistanceComputer(storage->get_distance_computer());
    }
    return storage->get_distance_computer();
}

// Centralized filter: translate an INTERNAL HNSW ordinal to its EXTERNAL id via
// IndexIDMap::id_map, then test the OpenSearch selector on the external id — the
// same contract as faiss::IDSelectorTranslated. Bounds-checked and stat-counted so
// a selector is never accidentally called on an internal ordinal, and every real
// selector invocation is reflected in statistics.
struct Filter {
    const IDSelector* sel;
    const std::vector<int64_t>* id_map;  // external id by internal ordinal
    AcornSearchStats* st;
    inline bool operator()(storage_idx_t internal) const {
        if (internal < 0 || static_cast<size_t>(internal) >= id_map->size()) {
            return false;  // defensive: out-of-range ordinal never passes
        }
        const faiss::idx_t external = (*id_map)[internal];
        if (st) st->selector_checks++;
        const bool ok = sel->is_member(external);
        if (st) { if (ok) st->selector_matches++; else st->selector_rejections++; }
        return ok;
    }
};

// Plain (unfiltered) greedy descent on an upper level — faiss's greedy_update_nearest.
// The upper levels are sparse (O(log n) nodes); their only job is to reach the query
// neighbourhood cheaply. ACORN's filter-aware two-hop bridging is applied at level 0
// (where the predicate subgraph matters), not here — this avoids paying 2-hop selector
// + distance cost on every upper-level hop for no recall benefit. A filter-failing
// landing node is handled at level 0 (it seeds the frontier as a bridge).
void greedy_descent(
        const HNSW& hnsw, DistanceComputer& qdis, int level,
        storage_idx_t& nearest, float& d_nearest, AcornSearchStats* st) {
    for (;;) {
        const storage_idx_t prev = nearest;
        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        for (size_t i = begin; i < end; i++) {
            const storage_idx_t v = hnsw.neighbors[i];
            if (v < 0) break;
            const float d = qdis(v);
            if (st) st->distance_computations++;
            if (d < d_nearest) { d_nearest = d; nearest = v; }
        }
        if (nearest == prev) return;
    }
}

}  // namespace

int search(
        const faiss::IndexIDMap* id_map,
        const float* query,
        int k,
        int efSearch,
        const IDSelector* sel,
        int gamma,
        float* distances,
        faiss::idx_t* labels,
        bool collect_stats) {
    // ---- fix #10: validation / defensive checks (status returns, no exceptions) ----
    if (id_map == nullptr || query == nullptr || sel == nullptr ||
        distances == nullptr || labels == nullptr) {
        return ACORN_ERR_ARGS;
    }
    if (k <= 0 || efSearch <= 0) return ACORN_ERR_ARGS;
    if (gamma != 1) return ACORN_ERR_UNSUPPORTED_GAMMA;  // ACORN-1 only (fix #1)

    const faiss::IndexHNSW* index_hnsw = dynamic_cast<const faiss::IndexHNSW*>(id_map->index);
    if (index_hnsw == nullptr) return ACORN_ERR_INDEX;   // unsupported wrapped index type
    const faiss::Index* storage = index_hnsw->storage;
    if (storage == nullptr) return ACORN_ERR_INDEX;

    const faiss::idx_t ntotal = index_hnsw->ntotal;
    const HNSW& hnsw = index_hnsw->hnsw;

    // empty index → k empty slots, 0 results (not an error)
    if (ntotal == 0 || hnsw.entry_point == -1) {
        for (int i = 0; i < k; i++) { labels[i] = -1; distances[i] = 0; }
        return 0;
    }
    // id_map must describe every internal ordinal; otherwise translation is unsafe.
    if (static_cast<faiss::idx_t>(id_map->id_map.size()) != ntotal) return ACORN_ERR_INDEX;

    AcornSearchStats* st = nullptr;
    if (collect_stats) { last_stats().reset(); st = &last_stats(); }
    if (st) st->segment_vector_count = ntotal;

    // base connectivity M (fix #11: direct expression, level-0 cap ≈ 2*M)
    const int M = std::max(1, hnsw.nb_neighbors(0) / 2);
    const bool sim = faiss::is_similarity_metric(index_hnsw->metric_type);

    std::unique_ptr<DistanceComputer> qdis_holder(make_dc(storage));
    DistanceComputer& qdis = *qdis_holder;
    qdis.set_query(query);
    const Filter pass{sel, &id_map->id_map, st};

    // ---- upper-level plain greedy descent: cheaply reach the query neighbourhood.
    // (ACORN's filter-aware two-hop bridging is applied at level 0 only.) ----
    storage_idx_t nearest = hnsw.entry_point;
    float d_nearest = qdis(nearest);
    if (st) st->distance_computations++;
    for (int level = hnsw.max_level; level >= 1; level--) {
        greedy_descent(hnsw, qdis, level, nearest, d_nearest, st);
    }
    const bool nearest_passes = pass(nearest);  // one selector check on the landing node

    // ---- level-0 ACORN-1 bounded best-first traversal, SIMD-batched distances ----
    const int ef = std::max(efSearch, k);  // fix #6: normalized ef used consistently
    HNSW::MinimaxHeap candidates(ef);

    std::vector<float> D(k);
    std::vector<faiss::idx_t> I(k);
    int nres = 0;

    // Separate per-query state (fix #3, #12: VisitedTables, not unordered_set).
    VisitedTable admitted(ntotal);  // passing node in results/frontier (no dup admission)
    VisitedTable bridged(ntotal);   // node's 2-hop adjacency already walked (no repeat)

    // Batched scorer: passing nodes are buffered and scored 4-at-a-time via
    // DistanceComputer::distances_batch_4 (SIMD on the flat L2 path), then pushed to
    // the result heap + frontier. This closes the per-distance gap with faiss's own
    // inner loop — the decisive optimization for wall-clock parity.
    storage_idx_t bq[4]; int bn = 0;
    auto flush = [&]() {
        if (bn == 0) return;
        float bd[4] = {0, 0, 0, 0};
        if (bn == 4) qdis.distances_batch_4(bq[0], bq[1], bq[2], bq[3], bd[0], bd[1], bd[2], bd[3]);
        else for (int t = 0; t < bn; t++) bd[t] = qdis(bq[t]);
        if (st) st->distance_computations += bn;
        for (int t = 0; t < bn; t++) {
            if (nres < k) faiss::maxheap_push(++nres, D.data(), I.data(), bd[t], bq[t]);
            else if (bd[t] < D[0]) faiss::maxheap_replace_top(nres, D.data(), I.data(), bd[t], bq[t]);
            candidates.push(bq[t], bd[t]);
            if (st) st->frontier_insertions++;
        }
        bn = 0;
    };
    auto admit = [&](storage_idx_t v) {  // unique passing node → buffer for batched scoring
        admitted.set(v);
        if (st) st->eligible_discovered++;
        bq[bn++] = v;
        if (bn == 4) flush();
    };

    // fix #8: explicit seed. The landing node seeds the frontier as a bridge even if it
    // fails the filter; it enters results only if it passes (distance already known).
    if (nearest_passes) {
        faiss::maxheap_push(++nres, D.data(), I.data(), d_nearest, nearest);
        admitted.set(nearest);
        if (st) st->eligible_discovered++;
    }
    candidates.push(nearest, d_nearest);
    if (st) st->frontier_insertions++;

    const int expand_cap = 2 * M;  // fix #4/#5: reference ACORN expansion threshold

    while (candidates.size() > 0) {
        float d0 = 0;
        const storage_idx_t v0 = candidates.pop_min(&d0);
        if (candidates.count_below(d0) >= ef) break;  // fix #6: relative-distance stop on ef
        if (st) st->candidates_expanded++;

        size_t begin, end;
        hnsw.neighbor_range(v0, 0, &begin, &end);
        int found = 0;  // fix #4: counts only UNIQUE newly-admitted passing nodes

        for (size_t j = begin; j < end && found < expand_cap; j++) {
            const storage_idx_t v1 = hnsw.neighbors[j];
            if (v1 < 0) break;
            if (st) st->graph_neighbors_examined++;
            const bool v1pass = pass(v1);

            // 1-hop: admit a passing node once (scored in the next batch flush)
            if (v1pass && !admitted.get(v1)) { admit(v1); if (++found >= expand_cap) break; }

            // 2-hop bridge THROUGH v1 (pass or fail) — at most once per node (fix #3)
            if (bridged.get(v1)) { if (st) st->bridge_reexpansions_avoided++; continue; }
            bridged.set(v1);
            if (st) st->bridge_nodes_expanded++;
            size_t b2, e2;
            hnsw.neighbor_range(v1, 0, &b2, &e2);
            for (size_t j2 = b2; j2 < e2 && found < expand_cap; j2++) {
                const storage_idx_t v2 = hnsw.neighbors[j2];
                if (v2 < 0) break;
                if (st) st->graph_neighbors_examined++;
                if (!pass(v2) || admitted.get(v2)) continue;  // unique admissions only
                admit(v2);
                ++found;
            }
        }
        flush();  // score any passing nodes still buffered for this v0
    }
    flush();  // safety: never drop a buffered node

    // ---- finalize (fix #9): sort best-first, revert similarity negation once,
    // translate internal ordinal → external id exactly once, pad with -1. ----
    std::vector<std::pair<float, faiss::idx_t>> tmp;
    tmp.reserve(nres);
    for (int i = 0; i < nres; i++) tmp.emplace_back(sim ? -D[i] : D[i], I[i]);
    std::sort(tmp.begin(), tmp.end(), [sim](const std::pair<float, faiss::idx_t>& a,
                                            const std::pair<float, faiss::idx_t>& b) {
        return sim ? a.first > b.first : a.first < b.first;
    });
    for (size_t i = 0; i < tmp.size(); i++) {
        distances[i] = tmp[i].first;
        const storage_idx_t internal = tmp[i].second;
        labels[i] = (internal >= 0 && static_cast<size_t>(internal) < id_map->id_map.size())
                        ? id_map->id_map[internal]
                        : -1;
    }
    for (int i = static_cast<int>(tmp.size()); i < k; i++) { labels[i] = -1; distances[i] = 0; }
    if (st) st->results_returned = nres;
    return nres;
}

}  // namespace acorn
}  // namespace knn_jni
