/*
 * Correctness tests for the corrected JNI ACORN-1 traversal (jni/src/acorn_hnsw.cpp).
 * Builds a real faiss IndexIDMap(IndexHNSWFlat) so the internal→external ID
 * translation path is genuinely exercised, and validates every result against a
 * brute-force filtered oracle. Exit code 0 = all pass.
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cmath>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/random.h>

#include "acorn_hnsw.h"   // knn_jni::acorn::search

using faiss::idx_t;

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const char* name, const char* detail = "") {
    if (ok) { g_pass++; printf("  [PASS] %s\n", name); }
    else    { g_fail++; printf("  [FAIL] %s  %s\n", name, detail); }
}

// Selector over EXTERNAL ids (the OpenSearch contract). Membership is a set of
// external doc ids; is_member is called with an external id.
struct ExtSetSelector : faiss::IDSelector {
    std::unordered_set<idx_t> members;
    bool is_member(idx_t external_id) const override { return members.count(external_id) > 0; }
};

struct Dataset {
    int d, n;
    std::vector<float> x;                 // n*d
    std::vector<idx_t> ext;               // external id per internal ordinal
    faiss::MetricType metric;
    std::unique_ptr<faiss::IndexIDMap> idmap;   // owns IndexHNSW
};

// Build IndexIDMap(IndexHNSWFlat) with given external ids (may be non-identity).
static Dataset build(int d, int n, faiss::MetricType metric, const std::vector<idx_t>& ext, int seed) {
    Dataset ds; ds.d = d; ds.n = n; ds.metric = metric; ds.ext = ext;
    faiss::RandomGenerator rng(seed);
    ds.x.resize((size_t)n * d);
    for (auto& v : ds.x) v = rng.rand_float();
    auto* hnsw = new faiss::IndexHNSWFlat(d, 16, metric);
    hnsw->hnsw.efConstruction = 100;
    ds.idmap.reset(new faiss::IndexIDMap(hnsw));
    ds.idmap->own_fields = true;
    ds.idmap->add_with_ids(n, ds.x.data(), ext.data());
    return ds;
}

// Brute-force filtered top-k over EXTERNAL ids. Returns external ids best-first.
static std::vector<idx_t> brute(const Dataset& ds, const float* q, int k, const ExtSetSelector& sel) {
    const bool sim = faiss::is_similarity_metric(ds.metric);
    std::vector<std::pair<float, idx_t>> sc;
    for (int i = 0; i < ds.n; i++) {
        if (!sel.is_member(ds.ext[i])) continue;
        float s = 0;
        if (sim) { for (int j = 0; j < ds.d; j++) s += q[j] * ds.x[(size_t)i * ds.d + j]; }
        else { for (int j = 0; j < ds.d; j++) { float df = q[j] - ds.x[(size_t)i * ds.d + j]; s += df * df; } }
        sc.push_back({s, ds.ext[i]});
    }
    std::sort(sc.begin(), sc.end(), [&](auto& a, auto& b){ return sim ? a.first > b.first : a.first < b.first; });
    std::vector<idx_t> out; for (int i = 0; i < k && i < (int)sc.size(); i++) out.push_back(sc[i].second);
    return out;
}

static double recall(const std::vector<idx_t>& got, const std::vector<idx_t>& truth) {
    if (truth.empty()) return 1.0;
    std::unordered_set<idx_t> t(truth.begin(), truth.end());
    int h = 0; for (idx_t id : got) if (t.count(id)) h++;
    return double(h) / truth.size();
}

int main() {
    const int d = 32, n = 2000, k = 10, ef = 200;
    // ---- identity-id dataset (external == internal ordinal position 0..n-1) ----
    std::vector<idx_t> identity(n); for (int i = 0; i < n; i++) identity[i] = i;
    Dataset l2 = build(d, n, faiss::METRIC_L2, identity, 12345);
    faiss::RandomGenerator qr(999);
    std::vector<float> q(d); for (auto& v : q) v = qr.rand_float();
    std::vector<float> dis(k); std::vector<idx_t> ids(k);
    auto& S = knn_jni::acorn::last_stats();

    printf("== ACORN-1 JNI correctness ==\n");

    // 1. All nodes pass → matches brute-force top-k, high recall.
    {
        ExtSetSelector sel; for (int i = 0; i < n; i++) sel.members.insert(i);
        int r = knn_jni::acorn::search(l2.idmap.get(), q.data(), k, ef, &sel, 1, dis.data(), ids.data(), true);
        std::vector<idx_t> got(ids.begin(), ids.begin() + std::max(0, r));
        double rc = recall(got, brute(l2, q.data(), k, sel));
        check(r == k && rc >= 0.9, "1 all-pass matches brute-force (recall>=0.9)",
              rc < 0.9 ? "low recall" : "");
    }
    // 14. Exact agreement with brute force at large ef (all-pass).
    {
        ExtSetSelector sel; for (int i = 0; i < n; i++) sel.members.insert(i);
        knn_jni::acorn::search(l2.idmap.get(), q.data(), k, 1000, &sel, 1, dis.data(), ids.data(), false);
        std::vector<idx_t> got(ids.begin(), ids.end());
        check(recall(got, brute(l2, q.data(), k, sel)) == 1.0, "14 exact brute-force agreement at large ef");
    }
    // 2. Entry point fails the filter → still returns correct passing results.
    {
        ExtSetSelector sel;
        idx_t epExt = l2.ext[l2.idmap->index && false ? 0 : ((faiss::IndexHNSW*)l2.idmap->index)->hnsw.entry_point];
        for (int i = 0; i < n; i++) if (l2.ext[i] != epExt) sel.members.insert(i);  // everyone except entry point
        int r = knn_jni::acorn::search(l2.idmap.get(), q.data(), k, ef, &sel, 1, dis.data(), ids.data(), false);
        bool epExcluded = true; for (int i = 0; i < r; i++) if (ids[i] == epExt) epExcluded = false;
        std::vector<idx_t> got(ids.begin(), ids.begin() + std::max(0, r));
        double rc = recall(got, brute(l2, q.data(), k, sel));
        check(r > 0 && epExcluded && rc >= 0.8, "2 entry-point-fails still returns correct results");
    }
    // 6. Non-identity IndexIDMap ids → labels are external ids, filtering correct.
    {
        std::vector<idx_t> ext(n); for (int i = 0; i < n; i++) ext[i] = 1000 + i * 7;  // non-contiguous
        Dataset nd = build(d, n, faiss::METRIC_L2, ext, 4242);
        ExtSetSelector sel; for (int i = 0; i < n; i += 2) sel.members.insert(ext[i]);  // half, by external id
        int r = knn_jni::acorn::search(nd.idmap.get(), q.data(), k, ef, &sel, 1, dis.data(), ids.data(), false);
        bool allExternalAndPass = r > 0;
        std::unordered_set<idx_t> extset(ext.begin(), ext.end());
        for (int i = 0; i < r; i++) if (!extset.count(ids[i]) || !sel.is_member(ids[i])) allExternalAndPass = false;
        std::vector<idx_t> got(ids.begin(), ids.begin() + std::max(0, r));
        double rc = recall(got, brute(nd, q.data(), k, sel));
        check(allExternalAndPass && rc >= 0.8, "6 non-identity IDs: external labels + correct filtering",
              allExternalAndPass ? "low recall" : "internal id leaked / filter wrong");
    }
    // 3. Selective filter → every returned id passes the filter (bridge correctness).
    {
        ExtSetSelector sel; faiss::RandomGenerator r(7);
        for (int i = 0; i < n; i++) if (r.rand_float() < 0.05) sel.members.insert(i);  // ~5%
        int rr = knn_jni::acorn::search(l2.idmap.get(), q.data(), k, ef, &sel, 1, dis.data(), ids.data(), false);
        bool allPass = true; for (int i = 0; i < rr; i++) if (!sel.is_member(ids[i])) allPass = false;
        check(rr >= 0 && allPass, "3 selective filter: all results pass the filter");
    }
    // 4/5. Duplicate paths + bridge-expanded-once: no duplicate ids; bridged bounded.
    {
        ExtSetSelector sel; faiss::RandomGenerator r(11);
        for (int i = 0; i < n; i++) if (r.rand_float() < 0.1) sel.members.insert(i);
        int rr = knn_jni::acorn::search(l2.idmap.get(), q.data(), k, ef, &sel, 1, dis.data(), ids.data(), true);
        std::unordered_set<idx_t> seen; bool uniq = true;
        for (int i = 0; i < rr; i++) { if (seen.count(ids[i])) uniq = false; seen.insert(ids[i]); }
        check(uniq, "4 no duplicate result ids across bridge paths");
        check(S.bridge_nodes_expanded <= (uint64_t)n, "5 each node bridge-expanded at most once (<= n)");
    }
    // 7. efSearch < k → no crash, returns up to #passing, valid.
    {
        ExtSetSelector sel; for (int i = 0; i < n; i++) sel.members.insert(i);
        int r = knn_jni::acorn::search(l2.idmap.get(), q.data(), k, /*ef=*/3, &sel, 1, dis.data(), ids.data(), false);
        check(r > 0 && r <= k, "7 efSearch<k returns valid results");
    }
    // 8. Fewer than k passing nodes → returns <=#passing, all passing, padded.
    //    Passing set = the 4 true-nearest (reachable) so this exercises the
    //    fewer-than-k path rather than ACORN's (separately tested) stranding.
    {
        ExtSetSelector allsel; for (int i = 0; i < n; i++) allsel.members.insert(i);
        std::vector<idx_t> near4 = brute(l2, q.data(), 4, allsel);  // 4 nearest external ids
        ExtSetSelector sel; for (idx_t id : near4) sel.members.insert(id);
        int r = knn_jni::acorn::search(l2.idmap.get(), q.data(), k, ef, &sel, 1, dis.data(), ids.data(), false);
        bool allPass = true; for (int i = 0; i < r; i++) if (!sel.is_member(ids[i])) allPass = false;
        check(r <= 4 && r >= 1 && allPass, "8 fewer-than-k passing returns <=#passing, all valid");
        bool padded = true; for (int i = r; i < k; i++) if (ids[i] != -1) padded = false;
        check(padded, "8b unused slots padded with -1");
    }
    // 9. L2 ordering: distances ascending.
    {
        ExtSetSelector sel; for (int i = 0; i < n; i++) sel.members.insert(i);
        int r = knn_jni::acorn::search(l2.idmap.get(), q.data(), k, ef, &sel, 1, dis.data(), ids.data(), false);
        bool asc = true; for (int i = 1; i < r; i++) if (dis[i] < dis[i-1] - 1e-6) asc = false;
        check(asc, "9 L2 distances sorted ascending (best-first)");
    }
    // 10. Inner-product ordering: scores descending.
    {
        Dataset ip = build(d, n, faiss::METRIC_INNER_PRODUCT, identity, 777);
        ExtSetSelector sel; for (int i = 0; i < n; i++) sel.members.insert(i);
        int r = knn_jni::acorn::search(ip.idmap.get(), q.data(), k, ef, &sel, 1, dis.data(), ids.data(), false);
        bool desc = true; for (int i = 1; i < r; i++) if (dis[i] > dis[i-1] + 1e-6) desc = false;
        double rc = recall(std::vector<idx_t>(ids.begin(), ids.begin()+std::max(0,r)), brute(ip, q.data(), k, sel));
        check(r == k && desc && rc >= 0.9, "10 inner-product: scores descending + recall>=0.9");
    }
    // 11. Unsupported gamma → error code.
    {
        ExtSetSelector sel; for (int i = 0; i < n; i++) sel.members.insert(i);
        int r = knn_jni::acorn::search(l2.idmap.get(), q.data(), k, ef, &sel, /*gamma=*/2, dis.data(), ids.data(), false);
        check(r == knn_jni::acorn::ACORN_ERR_UNSUPPORTED_GAMMA, "11 gamma!=1 rejected (ACORN_ERR_UNSUPPORTED_GAMMA)");
    }
    // 12. Empty index → 0 results, all -1.
    {
        auto* h = new faiss::IndexHNSWFlat(d, 16, faiss::METRIC_L2);
        faiss::IndexIDMap empty(h); empty.own_fields = true;
        ExtSetSelector sel;
        int r = knn_jni::acorn::search(&empty, q.data(), k, ef, &sel, 1, dis.data(), ids.data(), false);
        bool allNeg = true; for (int i = 0; i < k; i++) if (ids[i] != -1) allNeg = false;
        check(r == 0 && allNeg, "12 empty index → 0 results, all -1");
    }
    // 13. Stats sane.
    {
        ExtSetSelector sel; for (int i = 0; i < n; i++) if (i % 3 == 0) sel.members.insert(i);
        int r = knn_jni::acorn::search(l2.idmap.get(), q.data(), k, ef, &sel, 1, dis.data(), ids.data(), true);
        check(S.selector_checks > 0 && S.distance_computations > 0 &&
              S.selector_matches + S.selector_rejections == S.selector_checks &&
              (int)S.results_returned == r, "13 statistics consistent");
    }
    // Defensive: null args.
    {
        ExtSetSelector sel;
        int r1 = knn_jni::acorn::search(nullptr, q.data(), k, ef, &sel, 1, dis.data(), ids.data(), false);
        int r2 = knn_jni::acorn::search(l2.idmap.get(), q.data(), 0, ef, &sel, 1, dis.data(), ids.data(), false);
        check(r1 == knn_jni::acorn::ACORN_ERR_ARGS && r2 == knn_jni::acorn::ACORN_ERR_ARGS, "def null/invalid args rejected");
    }

    printf("\n== %d passed, %d failed ==\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
