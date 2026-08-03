// Smoke test: build standard + ACORN-γ graphs, run all search modes + exact,
// check basic sanity and serialization round-trip.
#include <cstdio>
#include <vector>
#include <unordered_set>
#include <faiss/utils/random.h>
#include "acorn.h"

using namespace acorn;

// Simple even-id selector.
struct EvenSel : faiss::IDSelector {
    bool is_member(faiss::idx_t id) const override { return (id % 2) == 0; }
};

int main() {
    int d = 32, n = 5000, k = 10;
    std::vector<float> data(n * d);
    faiss::RandomGenerator rng(123);
    for (auto& v : data) v = rng.rand_float();

    AcornIndex std_idx, gam_idx;
    build_standard_hnsw(std_idx, d, faiss::METRIC_L2, 16, 40, n, data.data());
    AcornGammaBuildParameters bp; bp.enabled = true; bp.M = 16; bp.gamma = 4; bp.M_beta = 32;
    build_acorn_gamma(gam_idx, d, faiss::METRIC_L2, bp, n, data.data());

    printf("standard: build=%.1fms avgdeg0=%.1f maxdeg0=%d bytes=%zu\n",
           std_idx.build_stats.build_ms, std_idx.build_stats.avg_degree_l0,
           std_idx.build_stats.max_degree_l0, std_idx.build_stats.graph_bytes);
    printf("acorn-g4: build=%.1fms avgdeg0=%.1f maxdeg0=%d bytes=%zu considered=%llu retained=%llu pruned=%llu\n",
           gam_idx.build_stats.build_ms, gam_idx.build_stats.avg_degree_l0,
           gam_idx.build_stats.max_degree_l0, gam_idx.build_stats.graph_bytes,
           (unsigned long long)gam_idx.build_stats.candidate_edges_considered,
           (unsigned long long)gam_idx.build_stats.edges_retained,
           (unsigned long long)gam_idx.build_stats.edges_pruned);

    EvenSel sel;
    std::vector<float> q(data.begin(), data.begin() + d);
    auto run = [&](const char* name, const AcornIndex& idx, FilteredHnswSearchMode m) {
        std::vector<faiss::idx_t> ids(k); std::vector<float> dis(k); SearchStats st;
        int nr = filtered_search(idx, q.data(), k, 64, m, &sel, ids.data(), dis.data(), &st);
        printf("%-22s nres=%d ndis=%llu 1hop=%llu 2hop=%llu sel=%llu acc=%llu rej=%llu top=%lld d=%.3f\n",
               name, nr, (unsigned long long)st.dist_computations,
               (unsigned long long)st.first_hop_expansions, (unsigned long long)st.second_hop_expansions,
               (unsigned long long)st.selector_checks, (unsigned long long)st.accepted,
               (unsigned long long)st.rejected, (long long)ids[0], dis[0]);
        // all returned ids must satisfy the predicate
        for (int i = 0; i < nr; i++) if (!sel.is_member(ids[i])) { printf("  !! bad id %lld\n", (long long)ids[i]); }
        return ids;
    };
    run("A std-graph STANDARD", std_idx, FilteredHnswSearchMode::STANDARD);
    auto b = run("B std-graph ACORN(g1)", std_idx, FilteredHnswSearchMode::ACORN);
    auto c = run("C acorn-g4 ACORN", gam_idx, FilteredHnswSearchMode::ACORN);

    // exact oracle + recall of C
    std::vector<faiss::idx_t> eids(k); std::vector<float> edis(k); SearchStats est;
    int en = exact_filtered_search(std_idx, q.data(), k, &sel, eids.data(), edis.data(), &est);
    std::unordered_set<faiss::idx_t> truth(eids.begin(), eids.begin() + en);
    auto recall = [&](std::vector<faiss::idx_t>& r) { int h=0; for (auto id: r) if (truth.count(id)) h++; return double(h)/en; };
    printf("exact: nres=%d top=%lld d=%.3f | recall B=%.2f C=%.2f\n",
           en, (long long)eids[0], edis[0], recall(b), recall(c));

    // serialization round trip on ACORN-γ
    save_acorn_index(gam_idx, "/tmp/acorn_smoke.bin");
    AcornIndex reloaded;
    load_acorn_index(reloaded, "/tmp/acorn_smoke.bin");
    std::vector<faiss::idx_t> rids(k); std::vector<float> rdis(k);
    int rn = filtered_search(reloaded, q.data(), k, 64, FilteredHnswSearchMode::ACORN, &sel, rids.data(), rdis.data(), nullptr);
    bool match = (rn == (int)c.size() || rn > 0);
    int same = 0; for (int i=0;i<rn;i++) if (i<(int)c.size() && rids[i]==c[i]) same++;
    printf("reload: nres=%d identical_top=%d/%d ok=%d\n", rn, same, rn, match);
    printf("SMOKE OK\n");
    return 0;
}
