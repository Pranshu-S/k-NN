/*
 * Phase 3 — ACORN-γ graph construction over faiss::HNSW (faiss 1.11.0).
 *
 * build_standard_hnsw: faithful faiss HNSW build (uses faiss::HNSW::add_with_
 *   locks), single-threaded/deterministic. Baselines A/B/D/E.
 * build_acorn_gamma:   denser level-0 (M_beta + 1.5M), two-hop compression
 *   prune (ACORN::shrink_neighbor_list), predicate-agnostic. Baseline C.
 *
 * Deviations from the ACORN reference are documented in ACORN_GAMMA_DESIGN.md
 * and flagged inline with [DEVIATION].
 */
#include "acorn.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <unordered_set>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/random.h>

using faiss::DistanceComputer;
using faiss::HNSW;
using faiss::VisitedTable;
using storage_idx_t = HNSW::storage_idx_t;
using NodeDistFarther = HNSW::NodeDistFarther;
using NodeDistCloser = HNSW::NodeDistCloser;

namespace acorn {

static DistanceComputer* make_build_dc(const faiss::IndexFlat& storage) {
    if (faiss::is_similarity_metric(storage.metric_type))
        return new faiss::NegativeDistanceComputer(storage.get_distance_computer());
    return storage.get_distance_computer();
}

static faiss::IndexFlat* make_storage(int d, faiss::MetricType metric) {
    return new faiss::IndexFlat(d, metric);
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

/* ============================ STANDARD build ============================= */

void build_standard_hnsw(
        AcornIndex& index,
        int d,
        faiss::MetricType metric,
        int M,
        int efConstruction,
        faiss::idx_t n,
        const float* x,
        int seed) {
    index.d = d;
    index.metric = metric;
    index.is_acorn_gamma = false;
    index.gamma = 1;
    index.M = M;
    index.M_beta = 2 * M; // used only if ACORN traversal runs over this graph
    index.storage.reset(make_storage(d, metric));
    index.storage->add(n, x);

    HNSW& hnsw = index.hnsw;
    hnsw = HNSW(M);
    hnsw.efConstruction = efConstruction;
    hnsw.rng = faiss::RandomGenerator(seed);

    double t0 = now_ms();
    int max_level = hnsw.prepare_level_tab(n, false);

    // bucket points by level (faiss order), shuffle within level, add high->low
    std::vector<int> hist;
    std::vector<int> order(n);
    for (faiss::idx_t i = 0; i < n; i++) {
        int lvl = hnsw.levels[i] - 1;
        while (lvl >= (int)hist.size()) hist.push_back(0);
        hist[lvl]++;
    }
    std::vector<int> offs(hist.size() + 1, 0);
    for (size_t i = 0; i + 1 < hist.size(); i++) offs[i + 1] = offs[i] + hist[i];
    { std::vector<int> cur = offs; for (faiss::idx_t i = 0; i < n; i++) { int lvl = hnsw.levels[i] - 1; order[cur[lvl]++] = i; } }

    std::vector<omp_lock_t> locks(n);
    for (faiss::idx_t i = 0; i < n; i++) omp_init_lock(&locks[i]);
    VisitedTable vt(n);
    std::unique_ptr<DistanceComputer> dc(make_build_dc(*index.storage));
    faiss::RandomGenerator rng2(789);

    int i1 = n;
    for (int lvl = (int)hist.size() - 1; lvl >= 0; lvl--) {
        int i0 = i1 - hist[lvl];
        for (int j = i0; j < i1; j++) std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);
        for (int i = i0; i < i1; i++) {
            storage_idx_t pt = order[i];
            dc->set_query(x + pt * d);
            hnsw.add_with_locks(*dc, lvl, pt, locks, vt, /*keep_max_size_level0=*/false);
        }
        i1 = i0;
    }
    for (faiss::idx_t i = 0; i < n; i++) omp_destroy_lock(&locks[i]);

    index.build_stats = BuildStats{};
    index.build_stats.build_ms = now_ms() - t0;
    index.build_stats.levels = max_level + 1;
    index.build_stats.avg_degree_l0 = graph_avg_degree(hnsw, 0);
    index.build_stats.max_degree_l0 = graph_max_degree(hnsw, 0);
    index.build_stats.graph_bytes =
            hnsw.neighbors.size() * sizeof(storage_idx_t) + hnsw.offsets.size() * sizeof(size_t);
    // edges retained = non-empty level-0 slots
    uint64_t retained = 0;
    for (storage_idx_t v : hnsw.neighbors) if (v >= 0) retained++;
    index.build_stats.edges_retained = retained;
}

/* ============================= ACORN-γ build ============================= */
// set_default_probas variant: level-0 cap = M_beta + 1.5M, upper = M*gamma.
static void acorn_set_probas(HNSW& hnsw, int M, float levelMult, int M_beta, int gamma) {
    hnsw.assign_probas.clear();
    hnsw.cum_nneighbor_per_level.clear();
    int nn = 0;
    hnsw.cum_nneighbor_per_level.push_back(0);
    for (int level = 0;; level++) {
        float proba = std::exp(-level / levelMult) * (1 - std::exp(-1.0f / levelMult));
        if (proba < 1e-9) break;
        hnsw.assign_probas.push_back(proba);
        nn += (level == 0) ? (int)(M_beta + 1.5 * M) : M * gamma;
        hnsw.cum_nneighbor_per_level.push_back(nn);
    }
}

namespace {
// ACORN two-hop compression prune (level 0 only). Enumerates candidates
// nearest->farthest; keeps the first M_beta unconditionally; beyond that,
// prunes a candidate already two-hop reachable from kept nodes.
void acorn_shrink(
        const HNSW& hnsw,
        std::priority_queue<NodeDistFarther>& input, // top = nearest
        std::vector<NodeDistFarther>& output,
        int max_size,
        int M_beta,
        uint64_t& considered,
        uint64_t& pruned) {
    std::unordered_set<storage_idx_t> non; // two-hop reachable set
    int node_num = 0;
    while (!input.empty()) {
        node_num++;
        NodeDistFarther v1 = input.top();
        input.pop();
        considered++;
        bool good = true;
        if (node_num > M_beta && non.count(v1.id) > 0) { good = false; pruned++; }
        if (good) {
            output.push_back(v1);
            if ((int)output.size() >= max_size) return;
            non.insert(v1.id);
            if (node_num > M_beta) {
                size_t b, e;
                hnsw.neighbor_range(v1.id, 0, &b, &e);
                for (size_t j = b; j < e; j++) {
                    if (hnsw.neighbors[j] < 0) break;
                    non.insert(hnsw.neighbors[j]);
                }
            }
            if ((int)non.size() >= max_size) break;
        }
    }
}

// ACORN candidate collection (denser: 2*M*gamma at level 0). Mirrors ACORN
// search_neighbors_to_add: relaxed greedy stop for gamma>1, per-node visit cap.
void acorn_search_neighbors_to_add(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry,
        int level,
        int M_base,
        int gamma,
        VisitedTable& vt) {
    std::priority_queue<NodeDistFarther> candidates;
    candidates.emplace(d_entry, entry_point);
    results.emplace(d_entry, entry_point);
    vt.set(entry_point);

    int want = (level == 0) ? 2 * M_base * gamma : hnsw.nb_neighbors(level);
    while (!candidates.empty()) {
        NodeDistFarther cur = candidates.top();
        if ((cur.d > results.top().d && gamma == 1) || (int)results.size() >= want) break;
        int node = cur.id;
        candidates.pop();
        size_t begin, end;
        hnsw.neighbor_range(node, level, &begin, &end);
        int iters = 0;
        for (size_t i = begin; i < end; i++) {
            storage_idx_t nid = hnsw.neighbors[i];
            if (nid < 0) break;
            if (vt.get(nid)) continue;
            vt.set(nid);
            if (++iters > M_base) break;
            float dis = qdis(nid);
            if ((int)results.size() < hnsw.efConstruction || results.top().d > dis) {
                results.emplace(dis, nid);
                candidates.emplace(dis, nid);
                if ((int)results.size() > hnsw.efConstruction) results.pop();
            }
            if (++iters > M_base) break;
        }
    }
    vt.advance();
}

// find a free slot / evict for a link at (src,level); level-0 uses ACORN prune,
// upper levels keep nearest (bounded truncation). [DEVIATION]: upper-level full
// case truncates by distance instead of the reference's unshrunk write (which
// can overflow); triggers rarely since upper capacity is M*gamma.
void acorn_add_link(
        HNSW& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level,
        int M_beta,
        uint64_t& considered,
        uint64_t& pruned) {
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);
    if (hnsw.neighbors[end - 1] == -1) {
        size_t i = end;
        while (i > begin && hnsw.neighbors[i - 1] == -1) i--;
        hnsw.neighbors[i] = dest;
        return;
    }
    std::priority_queue<NodeDistCloser> res;
    res.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++)
        res.emplace(qdis.symmetric_dis(src, hnsw.neighbors[i]), hnsw.neighbors[i]);

    int cap = (int)(end - begin);
    std::vector<NodeDistFarther> keep;
    if (level == 0) {
        std::priority_queue<NodeDistFarther> far; // top = nearest
        while (!res.empty()) { far.emplace(res.top().d, res.top().id); res.pop(); }
        acorn_shrink(hnsw, far, keep, cap, M_beta, considered, pruned);
    } else {
        // keep the `cap` nearest
        std::vector<NodeDistCloser> tmp;
        while (!res.empty()) { tmp.push_back(res.top()); res.pop(); }
        std::sort(tmp.begin(), tmp.end(), [](const NodeDistCloser& a, const NodeDistCloser& b){ return a.d < b.d; });
        for (int i = 0; i < cap && i < (int)tmp.size(); i++) keep.emplace_back(tmp[i].d, tmp[i].id);
    }
    size_t i = begin;
    for (auto& nd : keep) if (i < end) hnsw.neighbors[i++] = nd.id;
    while (i < end) hnsw.neighbors[i++] = -1;
}

void acorn_add_links_starting_from(
        HNSW& hnsw,
        DistanceComputer& qdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        int M_base,
        int M_beta,
        int gamma,
        VisitedTable& vt,
        uint64_t& considered,
        uint64_t& retained,
        uint64_t& pruned) {
    std::priority_queue<NodeDistCloser> link_targets;
    acorn_search_neighbors_to_add(hnsw, qdis, link_targets, nearest, d_nearest, level, M_base, gamma, vt);

    int max_size = hnsw.nb_neighbors(level);
    // prune the collected candidate set down to max_size for pt_id's own list
    if (level == 0) {
        std::priority_queue<NodeDistFarther> far;
        while (!link_targets.empty()) { far.emplace(link_targets.top().d, link_targets.top().id); link_targets.pop(); }
        std::vector<NodeDistFarther> keep;
        acorn_shrink(hnsw, far, keep, max_size, M_beta, considered, pruned);
        for (auto& nd : keep) link_targets.emplace(nd.d, nd.id);
    } else {
        // keep nearest max_size
        std::vector<NodeDistCloser> tmp;
        while (!link_targets.empty()) { tmp.push_back(link_targets.top()); link_targets.pop(); }
        std::sort(tmp.begin(), tmp.end(), [](const NodeDistCloser& a, const NodeDistCloser& b){ return a.d < b.d; });
        for (int i = 0; i < max_size && i < (int)tmp.size(); i++) link_targets.emplace(tmp[i].d, tmp[i].id);
    }

    std::vector<storage_idx_t> added;
    while (!link_targets.empty()) {
        storage_idx_t other = link_targets.top().id;
        acorn_add_link(hnsw, qdis, pt_id, other, level, M_beta, considered, pruned);
        added.push_back(other);
        retained++;
        link_targets.pop();
    }
    for (storage_idx_t other : added)
        acorn_add_link(hnsw, qdis, other, pt_id, level, M_beta, considered, pruned);
}

void acorn_construction_greedy(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int M_base,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    for (;;) {
        storage_idx_t prev = nearest;
        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        int iters = 0;
        for (size_t i = begin; i < end; i++) {
            storage_idx_t v = hnsw.neighbors[i];
            if (v < 0) break;
            if (++iters > M_base) break;
            float dis = qdis(v);
            if (dis < d_nearest) { nearest = v; d_nearest = dis; }
        }
        if (nearest == prev) return;
    }
}
} // namespace

void build_acorn_gamma(
        AcornIndex& index,
        int d,
        faiss::MetricType metric,
        const AcornGammaBuildParameters& params,
        faiss::idx_t n,
        const float* x) {
    int M = params.M;
    int gamma = std::max(1, params.gamma);
    int M_beta = params.M_beta > 0 ? params.M_beta : 2 * M;
    int efc = params.efConstruction > 0 ? params.efConstruction : M * gamma;
    if (M_beta > 2 * M * gamma) throw std::runtime_error("M_beta must be <= 2*M*gamma");

    index.d = d;
    index.metric = metric;
    index.is_acorn_gamma = true;
    index.gamma = gamma;
    index.M = M;
    index.M_beta = M_beta;
    index.storage.reset(make_storage(d, metric));
    index.storage->add(n, x);

    HNSW& hnsw = index.hnsw;
    hnsw = HNSW(M);
    acorn_set_probas(hnsw, M, 1.0 / std::log(M), M_beta, gamma);
    hnsw.efConstruction = efc;
    hnsw.rng = faiss::RandomGenerator(params.seed);

    double t0 = now_ms();
    hnsw.prepare_level_tab(n, false);
    int max_level = 0;
    for (faiss::idx_t i = 0; i < n; i++) max_level = std::max(max_level, hnsw.levels[i] - 1);

    // bucket + shuffle within level (deterministic, single thread)
    std::vector<int> hist;
    for (faiss::idx_t i = 0; i < n; i++) {
        int lvl = hnsw.levels[i] - 1;
        while (lvl >= (int)hist.size()) hist.push_back(0);
        hist[lvl]++;
    }
    std::vector<int> offs(hist.size() + 1, 0);
    for (size_t i = 0; i + 1 < hist.size(); i++) offs[i + 1] = offs[i] + hist[i];
    std::vector<int> order(n);
    { std::vector<int> cur = offs; for (faiss::idx_t i = 0; i < n; i++) { int lvl = hnsw.levels[i] - 1; order[cur[lvl]++] = i; } }

    VisitedTable vt(n);
    std::unique_ptr<DistanceComputer> dc(make_build_dc(*index.storage));
    faiss::RandomGenerator rng2(789);
    uint64_t considered = 0, retained = 0, pruned = 0;

    int i1 = n;
    for (int lvl = (int)hist.size() - 1; lvl >= 0; lvl--) {
        int i0 = i1 - hist[lvl];
        for (int j = i0; j < i1; j++) std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);
        for (int i = i0; i < i1; i++) {
            storage_idx_t pt = order[i];
            dc->set_query(x + pt * d);

            // first point becomes entry point
            if (hnsw.entry_point == -1) { hnsw.max_level = lvl; hnsw.entry_point = pt; continue; }

            storage_idx_t nearest = hnsw.entry_point;
            float d_nearest = (*dc)(nearest);
            int start = hnsw.max_level;
            for (int L = start; L > lvl; L--)
                acorn_construction_greedy(hnsw, *dc, M, L, nearest, d_nearest);
            for (int L = std::min(lvl, hnsw.max_level); L >= 0; L--)
                acorn_add_links_starting_from(hnsw, *dc, pt, nearest, d_nearest, L,
                        M, M_beta, gamma, vt, considered, retained, pruned);
            if (lvl > hnsw.max_level) { hnsw.max_level = lvl; hnsw.entry_point = pt; }
        }
        i1 = i0;
    }

    index.build_stats = BuildStats{};
    index.build_stats.build_ms = now_ms() - t0;
    index.build_stats.levels = max_level + 1;
    index.build_stats.candidate_edges_considered = considered;
    index.build_stats.edges_retained = retained;
    index.build_stats.edges_pruned = pruned;
    index.build_stats.avg_degree_l0 = graph_avg_degree(hnsw, 0);
    index.build_stats.max_degree_l0 = graph_max_degree(hnsw, 0);
    index.build_stats.graph_bytes =
            hnsw.neighbors.size() * sizeof(storage_idx_t) + hnsw.offsets.size() * sizeof(size_t);
}

/* ========================= prototype serialization ====================== */
static const uint32_t ACORN_PROTO_MAGIC = 0xAC0F0001u; // "ACORN proto v1"

void save_acorn_index(const AcornIndex& index, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("save_acorn_index: cannot open " + path);
    auto w = [&](const void* p, size_t n) { std::fwrite(p, 1, n, f); };
    w(&ACORN_PROTO_MAGIC, 4);
    int32_t hdr[6] = {index.d, (int32_t)index.metric, index.is_acorn_gamma, index.gamma, index.M, index.M_beta};
    w(hdr, sizeof(hdr));
    int64_t n = index.ntotal();
    w(&n, 8);
    // vectors
    std::vector<float> vecs(n * index.d);
    index.storage->reconstruct_n(0, n, vecs.data());
    w(vecs.data(), vecs.size() * sizeof(float));
    // hnsw structure
    const HNSW& h = index.hnsw;
    auto wv = [&](auto& vec) { int64_t sz = vec.size(); w(&sz, 8); if (sz) w(vec.data(), sz * sizeof(vec[0])); };
    wv(h.assign_probas); wv(h.cum_nneighbor_per_level); wv(h.levels); wv(h.offsets);
    { int64_t sz = h.neighbors.size(); w(&sz, 8); if (sz) w(h.neighbors.data(), sz * sizeof(storage_idx_t)); }
    int32_t tail[2] = {h.entry_point, h.max_level};
    w(tail, sizeof(tail));
    std::fclose(f);
}

void load_acorn_index(AcornIndex& index, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("load_acorn_index: cannot open " + path);
    auto r = [&](void* p, size_t n) { if (std::fread(p, 1, n, f) != n) throw std::runtime_error("load: short read"); };
    uint32_t magic; r(&magic, 4);
    if (magic != ACORN_PROTO_MAGIC)
        throw std::runtime_error("load_acorn_index: bad magic (prototype-only format, not production-compatible)");
    int32_t hdr[6]; r(hdr, sizeof(hdr));
    index.d = hdr[0]; index.metric = (faiss::MetricType)hdr[1]; index.is_acorn_gamma = hdr[2];
    index.gamma = hdr[3]; index.M = hdr[4]; index.M_beta = hdr[5];
    int64_t n; r(&n, 8);
    std::vector<float> vecs(n * index.d);
    r(vecs.data(), vecs.size() * sizeof(float));
    index.storage.reset(make_storage(index.d, index.metric));
    index.storage->add(n, vecs.data());
    HNSW& h = index.hnsw;
    h = HNSW(index.M);
    auto rv = [&](auto& vec) { int64_t sz; r(&sz, 8); vec.resize(sz); if (sz) r(vec.data(), sz * sizeof(vec[0])); };
    rv(h.assign_probas); rv(h.cum_nneighbor_per_level); rv(h.levels); rv(h.offsets);
    { int64_t sz; r(&sz, 8); h.neighbors.resize(sz); if (sz) r(h.neighbors.data(), sz * sizeof(storage_idx_t)); }
    int32_t tail[2]; r(tail, sizeof(tail));
    h.entry_point = tail[0]; h.max_level = tail[1];
    std::fclose(f);
}

} // namespace acorn
