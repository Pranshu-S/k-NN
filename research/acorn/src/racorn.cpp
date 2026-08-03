/*
 * RACORN-1 / RACORN-1+ traversal (arXiv:2607.00768) over a standard faiss::HNSW.
 *
 * Faithful port of ACORN-SEARCH-LAYER (Alg. 1) + RACORN1-EXPAND (Alg. 2):
 *   - filter-first beam: only predicate-passing nodes enter the result heap W;
 *   - Adaptive Search Fallback (ASF): when passing 1-hop/2-hop candidates are
 *     insufficient (|C2| < target = n*bridge_ratio), admit filter-FAILING 2-hop
 *     nodes as transient BRIDGES into the candidate queue (never W), and mark
 *     the failing nodes visited to prevent re-evaluation;
 *   - stride sampling for spatial diversity at bridge-selection and C2-cap sites;
 *   - RACORN-1+: Adaptive Exact Fallback (AEF) — if the running predicate-pass
 *     ratio among visited nodes drops below a threshold, switch to exact search.
 *
 * Predicate = faiss::IDSelector::is_member (matches the OpenSearch filter path).
 */
#include "acorn.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>

using faiss::DistanceComputer;
using faiss::HNSW;
using faiss::IDSelector;
using faiss::VisitedTable;
using storage_idx_t = HNSW::storage_idx_t;

namespace acorn {

static DistanceComputer* racorn_make_dc(const faiss::IndexFlat& storage) {
    if (faiss::is_similarity_metric(storage.metric_type))
        return new faiss::NegativeDistanceComputer(storage.get_distance_computer());
    return storage.get_distance_computer();
}

// Stride sampling: keep T items spread across P (radial diversity) rather than a
// biased prefix. If !stride, this is plain prefix truncation.
static void stride_sample(const std::vector<storage_idx_t>& P, int T, bool stride,
                          std::vector<storage_idx_t>& out) {
    out.clear();
    if (T <= 0) return;
    if ((int)P.size() <= T) { out = P; return; }
    if (!stride) { out.assign(P.begin(), P.begin() + T); return; }
    size_t sigma = P.size() / (size_t)T;
    if (sigma < 1) sigma = 1;
    for (size_t i = 0; i < P.size() && (int)out.size() < T; i += sigma) out.push_back(P[i]);
}

namespace {
struct Node { float d; storage_idx_t id; };
struct MinByDist { bool operator()(const Node& a, const Node& b) const { return a.d > b.d; } }; // min-heap
struct MaxByDist { bool operator()(const Node& a, const Node& b) const { return a.d < b.d; } }; // max-heap
} // namespace

// Unfiltered greedy descent on an upper level (standard HNSW), to seed the beam.
static void greedy_descent_unfiltered(const HNSW& hnsw, DistanceComputer& qdis,
                                      int level, storage_idx_t& nearest, float& d_nearest,
                                      SearchStats* st) {
    for (;;) {
        storage_idx_t prev = nearest;
        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            storage_idx_t v = hnsw.neighbors[j];
            if (v < 0) break;
            float d = qdis(v);
            if (st) st->dist_computations++;
            if (d < d_nearest) { d_nearest = d; nearest = v; }
        }
        if (nearest == prev) return;
    }
}

int racorn_search(
        const AcornIndex& index,
        const float* query,
        int k,
        int efSearch,
        const IDSelector* sel,
        const RacornParams& params,
        faiss::idx_t* out_ids,
        float* out_dis,
        SearchStats* stats) {
    const HNSW& hnsw = index.hnsw;
    const bool sim = faiss::is_similarity_metric(index.metric);
    auto finalize = [&](std::vector<std::pair<float, storage_idx_t>>& res) -> int {
        std::sort(res.begin(), res.end(), [&](const std::pair<float,storage_idx_t>& a,
                                              const std::pair<float,storage_idx_t>& b) {
            return sim ? a.first > b.first : a.first < b.first; // best-first
        });
        int n = std::min((int)res.size(), k);
        for (int i = 0; i < n; i++) { out_dis[i] = res[i].first; out_ids[i] = res[i].second; }
        for (int i = n; i < k; i++) { out_ids[i] = -1; out_dis[i] = 0; }
        if (stats) stats->results_returned += n;
        return n;
    };

    if (hnsw.entry_point == -1) { for (int i=0;i<k;i++){out_ids[i]=-1;out_dis[i]=0;} return 0; }

    std::unique_ptr<DistanceComputer> qdis_holder(racorn_make_dc(*index.storage));
    DistanceComputer& qdis = *qdis_holder;
    qdis.set_query(query);

    // predicate helper with stats
    auto pass = [&](storage_idx_t id) {
        if (stats) stats->selector_checks++;
        bool ok = sel->is_member(id);
        if (stats) { if (ok) stats->accepted++; else stats->rejected++; }
        return ok;
    };

    const int ef = std::max(efSearch, k);
    const int M = hnsw.nb_neighbors(0);          // base-level degree (== 2*M_user)
    const double bridge_ratio = params.bridge_ratio;

    // --- upper-level greedy descent (unfiltered) ---
    storage_idx_t nearest = hnsw.entry_point;
    float d_nearest = qdis(nearest);
    if (stats) stats->dist_computations++;
    for (int level = hnsw.max_level; level >= 1; level--)
        greedy_descent_unfiltered(hnsw, qdis, level, nearest, d_nearest, stats);

    // --- bottom-layer RACORN beam ---
    VisitedTable vt(index.ntotal());
    std::priority_queue<Node, std::vector<Node>, MinByDist> C; // candidate min-heap
    std::priority_queue<Node, std::vector<Node>, MaxByDist> W; // result max-heap (cap ef)

    vt.set(nearest);
    C.push({d_nearest, nearest});
    bool ep_pass = pass(nearest);
    if (ep_pass) W.push({d_nearest, nearest});

    // AEF (RACORN-1+) running pass-ratio tracking over distance-evaluated nodes.
    uint64_t eval_total = 0, eval_pass = 0;
    const int aef_min_probe = params.aef_min_probe > 0 ? params.aef_min_probe : std::max(2 * ef, 128);

    // scratch
    std::vector<storage_idx_t> C1, C2, Bpool, fails1, sampled;
    std::unordered_set<storage_idx_t> seen;

    while (!C.empty()) {
        Node cur = C.top();
        if ((int)W.size() >= ef && !W.empty() && cur.d > W.top().d) break;
        C.pop();
        storage_idx_t u = cur.id;
        if (stats) stats->hops++;

        // ---- RACORN1-EXPAND(u) ----
        C1.clear(); C2.clear(); Bpool.clear(); fails1.clear(); seen.clear();
        int n_unvisited_1hop = 0;
        size_t ub, ue; hnsw.neighbor_range(u, 0, &ub, &ue);
        for (size_t j = ub; j < ue; j++) {
            storage_idx_t v = hnsw.neighbors[j];
            if (v < 0) break;
            if (stats) stats->first_hop_expansions++;
            if (vt.get(v)) continue;
            n_unvisited_1hop++;
            seen.insert(v);
            if (pass(v)) C1.push_back(v); else fails1.push_back(v);
        }
        // 2-hop scan over all 1-hop neighbours
        for (size_t j = ub; j < ue; j++) {
            storage_idx_t v = hnsw.neighbors[j];
            if (v < 0) break;
            size_t vb, ve; hnsw.neighbor_range(v, 0, &vb, &ve);
            for (size_t j2 = vb; j2 < ve; j2++) {
                storage_idx_t w = hnsw.neighbors[j2];
                if (w < 0) break;
                if (stats) stats->second_hop_expansions++;
                if (vt.get(w) || w == u) continue;
                if (seen.count(w)) continue; // dedup vs C1/fails1 and prior 2-hop
                seen.insert(w);
                if (pass(w)) C2.push_back(w); else Bpool.push_back(w);
            }
        }

        double target = n_unvisited_1hop * bridge_ratio;
        std::vector<storage_idx_t> B;
        if ((double)C2.size() < target) { // Adaptive Search Fallback
            // register failing nodes visited (prevents re-evaluation on re-entry)
            for (storage_idx_t v : fails1) vt.set(v);
            for (storage_idx_t w : Bpool) vt.set(w);
            if ((int)W.size() < ef) { // bridge-skip gate
                int needed = (int)std::ceil(target) - (int)C2.size();
                if (needed > 0) {
                    if ((int)Bpool.size() > needed) { stride_sample(Bpool, needed, params.stride_sampling, sampled); B = sampled; }
                    else B = Bpool;
                }
            }
        }
        // C2 cap with stride sampling
        if ((int)(C1.size() + C2.size()) > M) {
            stride_sample(C2, M - (int)C1.size(), params.stride_sampling, sampled);
            C2 = sampled;
        }

        // ---- evaluate returned nodes: C1 ∪ C2 (passes) ∪ B (bridges, fail) ----
        auto consider = [&](storage_idx_t e, bool is_bridge) {
            vt.set(e);
            float de = qdis(e);
            if (stats) stats->dist_computations++;
            eval_total++;
            bool p = !is_bridge; // C1/C2 already known to pass; bridges are fails
            if (p) eval_pass++;
            if ((int)W.size() < ef || (!W.empty() && de < W.top().d)) {
                C.push({de, e});
                if (p) { // F(e) guard: bridges never enter the result heap W
                    W.push({de, e});
                    if ((int)W.size() > ef) W.pop();
                }
            }
        };
        for (storage_idx_t e : C1) consider(e, false);
        for (storage_idx_t e : C2) consider(e, false);
        for (storage_idx_t e : B)  { consider(e, true); if (stats) stats->bridges_used++; }

        // ---- RACORN-1+ Adaptive Exact Fallback ----
        if (params.enable_aef && eval_total >= (uint64_t)aef_min_probe) {
            double pass_ratio = (double)eval_pass / (double)eval_total;
            if (pass_ratio < params.aef_threshold) {
                if (stats) stats->aef_switches++;
                std::vector<std::pair<float, storage_idx_t>> res;
                faiss::idx_t n = index.ntotal();
                for (faiss::idx_t i = 0; i < n; i++) {
                    if (stats) stats->selector_checks++;
                    if (!sel->is_member(i)) { if (stats) stats->rejected++; continue; }
                    if (stats) stats->accepted++;
                    float d = qdis(i);
                    if (stats) stats->dist_computations++;
                    res.emplace_back(sim ? -d : d, i);
                }
                return finalize(res);
            }
        }
    }

    // drain W -> best-first
    std::vector<std::pair<float, storage_idx_t>> res;
    while (!W.empty()) { res.emplace_back(sim ? -W.top().d : W.top().d, W.top().id); W.pop(); }
    return finalize(res);
}

// Faithful distance-first HNSW In-filtering (paper §2.1): traversal is
// predicate-agnostic (every neighbour is a candidate); the predicate gates only
// the result heap; search continues until ef PASSING results are held.
int hnsw_infilter_search(
        const AcornIndex& index,
        const float* query,
        int k,
        int efSearch,
        const IDSelector* sel,
        faiss::idx_t* out_ids,
        float* out_dis,
        SearchStats* stats) {
    const HNSW& hnsw = index.hnsw;
    const bool sim = faiss::is_similarity_metric(index.metric);
    if (hnsw.entry_point == -1) { for (int i=0;i<k;i++){out_ids[i]=-1;out_dis[i]=0;} return 0; }

    std::unique_ptr<DistanceComputer> qdis_holder(racorn_make_dc(*index.storage));
    DistanceComputer& qdis = *qdis_holder;
    qdis.set_query(query);
    auto pass = [&](storage_idx_t id){ if(stats) stats->selector_checks++; bool ok=sel->is_member(id); if(stats){ if(ok) stats->accepted++; else stats->rejected++;} return ok; };

    const int ef = std::max(efSearch, k);
    storage_idx_t nearest = hnsw.entry_point;
    float d_nearest = qdis(nearest);
    if (stats) stats->dist_computations++;
    for (int level = hnsw.max_level; level >= 1; level--)
        greedy_descent_unfiltered(hnsw, qdis, level, nearest, d_nearest, stats);

    VisitedTable vt(index.ntotal());
    std::priority_queue<Node, std::vector<Node>, MinByDist> C;
    std::priority_queue<Node, std::vector<Node>, MaxByDist> W;
    vt.set(nearest);
    C.push({d_nearest, nearest});
    if (pass(nearest)) W.push({d_nearest, nearest});

    while (!C.empty()) {
        Node cur = C.top();
        if ((int)W.size() >= ef && !W.empty() && cur.d > W.top().d) break;
        C.pop();
        if (stats) stats->hops++;
        size_t b,e; hnsw.neighbor_range(cur.id, 0, &b, &e);
        for (size_t j=b;j<e;j++) {
            storage_idx_t v = hnsw.neighbors[j];
            if (v < 0) break;
            if (stats) stats->first_hop_expansions++;
            if (vt.get(v)) continue;
            vt.set(v);
            if (stats) stats->nodes_visited++;
            float dv = qdis(v);
            if (stats) stats->dist_computations++;
            if ((int)W.size() < ef || (!W.empty() && dv < W.top().d)) {
                C.push({dv, v});
                if (pass(v)) { W.push({dv, v}); if ((int)W.size() > ef) W.pop(); }
            }
        }
    }
    std::vector<std::pair<float, storage_idx_t>> res;
    while (!W.empty()) { res.emplace_back(sim ? -W.top().d : W.top().d, W.top().id); W.pop(); }
    std::sort(res.begin(), res.end(), [&](const std::pair<float,storage_idx_t>& a, const std::pair<float,storage_idx_t>& b){ return sim ? a.first > b.first : a.first < b.first; });
    int nres = std::min((int)res.size(), k);
    for (int i=0;i<nres;i++){ out_dis[i]=res[i].first; out_ids[i]=res[i].second; }
    for (int i=nres;i<k;i++){ out_ids[i]=-1; out_dis[i]=0; }
    if (stats) stats->results_returned += nres;
    return nres;
}

} // namespace acorn
