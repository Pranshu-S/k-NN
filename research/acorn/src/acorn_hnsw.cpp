/*
 * Phase 2 — selector-aware ACORN traversal over faiss::HNSW.
 *
 * Ported from the ACORN reference (TAG-Research/ACORN, hybrid_search_from_
 * candidates / hybrid_greedy_update_nearest) to faiss 1.11.0, using
 * faiss::IDSelector::is_member(id) as the filter predicate. STANDARD mode calls
 * the stock faiss::HNSW::search so the two modes share identical storage.
 */
#include "acorn.h"

#include <algorithm>
#include <cmath>

#include <faiss/impl/AuxIndexStructures.h> // VisitedTable
#include <faiss/impl/DistanceComputer.h>   // DistanceComputer, NegativeDistanceComputer
#include <faiss/impl/ResultHandler.h>      // HeapBlockResultHandler
#include <faiss/utils/Heap.h>              // maxheap_*

using faiss::DistanceComputer;
using faiss::HNSW;
using faiss::IDSelector;
using faiss::VisitedTable;
using storage_idx_t = HNSW::storage_idx_t;

namespace acorn {

// Build a DistanceComputer over the storage; negate for similarity metrics so
// the HNSW CMax (min-)heap semantics hold, exactly as faiss IndexHNSW does.
static DistanceComputer* make_dc(const faiss::IndexFlat& storage) {
    if (faiss::is_similarity_metric(storage.metric_type)) {
        return new faiss::NegativeDistanceComputer(storage.get_distance_computer());
    }
    return storage.get_distance_computer();
}

int graph_max_degree(const HNSW& hnsw, int level) {
    int mx = 0;
    for (size_t i = 0; i < hnsw.levels.size(); i++) {
        if (hnsw.levels[i] <= level) continue;
        size_t b, e;
        hnsw.neighbor_range(i, level, &b, &e);
        int deg = 0;
        for (size_t j = b; j < e; j++)
            if (hnsw.neighbors[j] >= 0) deg++;
        mx = std::max(mx, deg);
    }
    return mx;
}

double graph_avg_degree(const HNSW& hnsw, int level) {
    uint64_t tot = 0, cnt = 0;
    for (size_t i = 0; i < hnsw.levels.size(); i++) {
        if (hnsw.levels[i] <= level) continue;
        size_t b, e;
        hnsw.neighbor_range(i, level, &b, &e);
        for (size_t j = b; j < e; j++)
            if (hnsw.neighbors[j] >= 0) tot++;
        cnt++;
    }
    return cnt ? double(tot) / cnt : 0.0;
}

/* ----------------------------- STANDARD mode ----------------------------- */

static int standard_search(
        const AcornIndex& index,
        const float* query,
        int k,
        int efSearch,
        const IDSelector* sel,
        faiss::idx_t* out_ids,
        float* out_dis,
        SearchStats* stats) {
    const HNSW& hnsw = index.hnsw;
    if (hnsw.entry_point == -1) return 0;

    std::unique_ptr<DistanceComputer> dc(make_dc(*index.storage));
    dc->set_query(query);
    VisitedTable vt(index.ntotal());

    using RH = faiss::HeapBlockResultHandler<HNSW::C>;
    RH bres(1, out_dis, out_ids, k);
    RH::SingleResultHandler res(bres);
    res.begin(0);

    faiss::SearchParametersHNSW params;
    params.efSearch = efSearch;
    params.bounded_queue = true;          // MUST be set: the unbounded path ignores the selector
    // faiss default is true (matches OpenSearch); KNN_REL_CHECK=0 forces the
    // exhaustive beam (termination becomes nstep>efSearch) — diagnostic only.
    const char* rc = getenv("KNN_REL_CHECK");
    params.check_relative_distance = !(rc && rc[0]=='0');
    params.sel = const_cast<faiss::IDSelector*>(sel); // faiss stores a non-const ptr; is_member is const

    faiss::HNSWStats s = hnsw.search(*dc, res, vt, &params);
    res.end();

    // Faiss leaves the heap in-place; reorder best-first and count results.
    int nres = 0;
    for (int i = 0; i < k; i++)
        if (out_ids[i] >= 0) nres++;
    // similarity metric: revert negated distances
    if (faiss::is_similarity_metric(index.metric))
        for (int i = 0; i < k; i++) out_dis[i] = -out_dis[i];
    // sort best-first
    std::vector<std::pair<float, faiss::idx_t>> tmp;
    for (int i = 0; i < k; i++)
        if (out_ids[i] >= 0) tmp.emplace_back(out_dis[i], out_ids[i]);
    bool sim = faiss::is_similarity_metric(index.metric);
    std::sort(tmp.begin(), tmp.end(), [&](auto& a, auto& b) {
        return sim ? a.first > b.first : a.first < b.first;
    });
    for (size_t i = 0; i < tmp.size(); i++) { out_dis[i] = tmp[i].first; out_ids[i] = tmp[i].second; }
    for (size_t i = tmp.size(); i < (size_t)k; i++) { out_ids[i] = -1; out_dis[i] = 0; }

    if (stats) {
        stats->dist_computations += s.ndis;
        stats->nodes_visited += s.ndis; // faiss counts a distance per touched node
        stats->first_hop_expansions += s.ndis;
        stats->hops += s.nhops;
        stats->results_returned += nres;
        // faiss checks the selector once per candidate distance eval on the filtered path
        if (sel) stats->selector_checks += s.ndis;
    }
    return nres;
}

/* ------------------------------- ACORN mode ------------------------------ */

namespace {
struct Pred {
    const IDSelector* sel;
    SearchStats* st;
    inline bool operator()(faiss::idx_t id) {
        if (!sel) return true;
        if (st) st->selector_checks++;
        bool ok = sel->is_member(id);
        if (st) { if (ok) st->accepted++; else st->rejected++; }
        return ok;
    }
};
} // namespace

// Raw predicate test without stats bookkeeping (used for the "current nearest
// fails predicate" escape clause so we don't double-count selector checks).
static inline bool raw_pass(const IDSelector* sel, storage_idx_t id) {
    return !sel || sel->is_member(id);
}

// Filtered greedy descent on an upper level (ACORN hybrid_greedy_update_nearest).
static void acorn_greedy_descent(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        Pred& pass,
        int gamma,
        int M,
        int level,
        storage_idx_t& nearest,
        float& d_nearest,
        SearchStats* st) {
    for (;;) {
        int num_found = 0;
        storage_idx_t prev = nearest;
        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        for (size_t i = begin; i < end; i++) {
            storage_idx_t v = hnsw.neighbors[i];
            if (v < 0) break;
            if (st) st->first_hop_expansions++;
            bool vp = pass(v);
            if (vp) {
                num_found++;
            } else if (gamma > 1) {
                continue;
            }
            if (vp) {
                float dis = qdis(v);
                if (st) st->dist_computations++;
                if (dis < d_nearest || !raw_pass(pass.sel, nearest)) {
                    nearest = v;
                    d_nearest = dis;
                }
                if (num_found >= M) break;
            }
            if (gamma == 1) { // ACORN-1: expand through every neighbour
                size_t b2, e2;
                hnsw.neighbor_range(v, level, &b2, &e2);
                for (size_t j = b2; j < e2; j++) {
                    storage_idx_t v2 = hnsw.neighbors[j];
                    if (v2 < 0) break;
                    if (st) st->second_hop_expansions++;
                    if (pass(v2)) {
                        num_found++;
                        float d2 = qdis(v2);
                        if (st) st->dist_computations++;
                        if (d2 < d_nearest || !raw_pass(pass.sel, nearest)) { nearest = v2; d_nearest = d2; }
                        if (num_found >= M) break;
                    }
                }
            }
        }
        if (nearest == prev) return;
    }
}

int acorn_search(
        const AcornIndex& index,
        const float* query,
        int k,
        int efSearch,
        const IDSelector* sel,
        faiss::idx_t* out_ids,
        float* out_dis,
        SearchStats* stats) {
    const HNSW& hnsw = index.hnsw;
    if (hnsw.entry_point == -1) return 0;
    const int gamma = index.gamma;
    const int M = index.M;
    const int M_beta = index.M_beta;

    std::unique_ptr<DistanceComputer> qdis_holder(make_dc(*index.storage));
    DistanceComputer& qdis = *qdis_holder;
    qdis.set_query(query);
    VisitedTable vt(index.ntotal());

    Pred pass{sel, stats};

    // --- upper-level filtered greedy descent from the global entry point ---
    storage_idx_t nearest = hnsw.entry_point;
    float d_nearest = qdis(nearest);
    if (stats) stats->dist_computations++;
    for (int level = hnsw.max_level; level >= 1; level--)
        acorn_greedy_descent(hnsw, qdis, pass, gamma, M, level, nearest, d_nearest, stats);

    // --- level-0 bounded BFS (ACORN hybrid_search_from_candidates) ---
    int ef = std::max(efSearch, k);
    HNSW::MinimaxHeap candidates(ef);
    candidates.push(nearest, d_nearest);

    // local max-heap of results (smaller internal distance = better)
    std::vector<float> D(k);
    std::vector<faiss::idx_t> I(k);
    int nres = 0;

    // seed results from initial candidate list
    for (int i = 0; i < candidates.size(); i++) {
        faiss::idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        if (pass(v1)) {
            if (nres < k) faiss::maxheap_push(++nres, D.data(), I.data(), d, v1);
            else if (d < D[0]) faiss::maxheap_replace_top(nres, D.data(), I.data(), d, v1);
        }
        vt.set(v1);
        if (stats) stats->nodes_visited++;
    }

    bool do_dis_check = true; // faiss default check_relative_distance
    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);
        if (do_dis_check) {
            if (candidates.count_below(d0) >= efSearch) break;
        }
        size_t begin, end;
        hnsw.neighbor_range(v0, 0, &begin, &end);

        int num_found = 0;
        bool keep_expanding = true;

        for (size_t j = begin; j < end; j++) {
            storage_idx_t v1 = hnsw.neighbors[j];
            if (v1 < 0) break;
            if (stats) stats->first_hop_expansions++;

            bool v1pass = pass(v1);
            if (v1pass) num_found++;

            if (!vt.get(v1)) {
                if (v1pass) {
                    vt.set(v1);
                    if (stats) stats->nodes_visited++;
                    float d = qdis(v1);
                    if (stats) stats->dist_computations++;
                    if (nres < k) faiss::maxheap_push(++nres, D.data(), I.data(), d, v1);
                    else if (d < D[0]) faiss::maxheap_replace_top(nres, D.data(), I.data(), d, v1);
                    candidates.push(v1, d);
                    if (num_found >= M * 2) { keep_expanding = false; break; }
                }
                // Two-hop expansion: bridge THROUGH v1 (pass or fail) into its
                // predicate-passing neighbours. For γ>1 only past position M_beta;
                // for γ==1 (ACORN-1) always.
                if (((int)(j - begin) >= M_beta && keep_expanding) || gamma == 1) {
                    size_t b2, e2;
                    hnsw.neighbor_range(v1, 0, &b2, &e2);
                    for (size_t j2 = b2; j2 < e2; j2++) {
                        storage_idx_t v2 = hnsw.neighbors[j2];
                        if (v2 < 0) break;
                        if (stats) stats->second_hop_expansions++;
                        if (!pass(v2)) continue;
                        num_found++;
                        if (vt.get(v2)) continue;
                        vt.set(v2);
                        if (stats) stats->nodes_visited++;
                        float d2 = qdis(v2);
                        if (stats) stats->dist_computations++;
                        if (nres < k) faiss::maxheap_push(++nres, D.data(), I.data(), d2, v2);
                        else if (d2 < D[0]) faiss::maxheap_replace_top(nres, D.data(), I.data(), d2, v2);
                        candidates.push(v2, d2);
                        if (num_found >= M * 2) { keep_expanding = false; break; }
                    }
                }
            }
        }
        nstep++;
        if (!do_dis_check && nstep > efSearch) { if (stats) stats->visit_limit_terminations++; break; }
    }

    // finalize: maxheap -> sorted best-first; revert IP negation
    std::vector<std::pair<float, faiss::idx_t>> tmp;
    for (int i = 0; i < nres; i++) tmp.emplace_back(D[i], I[i]);
    bool sim = faiss::is_similarity_metric(index.metric);
    for (auto& p : tmp) if (sim) p.first = -p.first;
    std::sort(tmp.begin(), tmp.end(), [&](auto& a, auto& b) {
        return sim ? a.first > b.first : a.first < b.first;
    });
    for (size_t i = 0; i < tmp.size(); i++) { out_dis[i] = tmp[i].first; out_ids[i] = tmp[i].second; }
    for (int i = tmp.size(); i < k; i++) { out_ids[i] = -1; out_dis[i] = 0; }
    if (stats) stats->results_returned += nres;
    return nres;
}

/* ------------------------------- dispatch -------------------------------- */

int filtered_search(
        const AcornIndex& index,
        const float* query,
        int k,
        int efSearch,
        FilteredHnswSearchMode mode,
        const IDSelector* sel,
        faiss::idx_t* out_ids,
        float* out_dis,
        SearchStats* stats) {
    if (mode == FilteredHnswSearchMode::STANDARD)
        return standard_search(index, query, k, efSearch, sel, out_ids, out_dis, stats);
    return acorn_search(index, query, k, efSearch, sel, out_ids, out_dis, stats);
}

/* --------------------------- exact oracle (F) ---------------------------- */

int exact_filtered_search(
        const AcornIndex& index,
        const float* query,
        int k,
        const IDSelector* sel,
        faiss::idx_t* out_ids,
        float* out_dis,
        SearchStats* stats) {
    std::unique_ptr<DistanceComputer> dc(make_dc(*index.storage));
    dc->set_query(query);
    bool sim = faiss::is_similarity_metric(index.metric);
    std::vector<std::pair<float, faiss::idx_t>> all;
    faiss::idx_t n = index.ntotal();
    for (faiss::idx_t i = 0; i < n; i++) {
        if (sel) { if (stats) stats->selector_checks++; if (!sel->is_member(i)) continue; }
        float d = (*dc)(i); // negated for sim
        if (stats) stats->dist_computations++;
        all.emplace_back(sim ? -d : d, i);
    }
    std::sort(all.begin(), all.end(), [&](auto& a, auto& b) {
        return sim ? a.first > b.first : a.first < b.first;
    });
    int nres = std::min((int)all.size(), k);
    for (int i = 0; i < nres; i++) { out_dis[i] = all[i].first; out_ids[i] = all[i].second; }
    for (int i = nres; i < k; i++) { out_ids[i] = -1; out_dis[i] = 0; }
    if (stats) stats->results_returned += nres;
    return nres;
}

} // namespace acorn
