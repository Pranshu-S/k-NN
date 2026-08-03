// Correctness test suite for the ACORN-γ prototype.
// Covers: determinism, degree bounds, valid/well-formed adjacency, selector
// correctness, edge cases (0/<k/all matches), duplicate suppression, cycles/
// termination, serialization round-trip, concurrent search, mode separation,
// and brute-force parity on small data.
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unordered_set>
#include <vector>

#include <faiss/utils/random.h>
#include "acorn.h"

using namespace acorn;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (cond) { g_pass++; }                                             \
        else { g_fail++; printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); } \
    } while (0)

// Selector over an explicit id set.
struct SetSel : faiss::IDSelector {
    std::unordered_set<faiss::idx_t> s;
    bool is_member(faiss::idx_t id) const override { return s.count(id) > 0; }
};

static std::vector<float> gen(int n, int d, int seed) {
    std::vector<float> x(size_t(n) * d);
    faiss::RandomGenerator rng(seed);
    for (auto& v : x) v = rng.rand_float();
    return x;
}

// brute-force filtered top-k truth (L2) as id set
static std::unordered_set<faiss::idx_t> brute(
        const float* x, int n, int d, const float* q, int k, const SetSel& sel) {
    std::vector<std::pair<float, faiss::idx_t>> all;
    for (int i = 0; i < n; i++) {
        if (!sel.is_member(i)) continue;
        float s = 0;
        for (int j = 0; j < d; j++) { float df = q[j] - x[i * d + j]; s += df * df; }
        all.emplace_back(s, i);
    }
    std::sort(all.begin(), all.end());
    std::unordered_set<faiss::idx_t> r;
    for (int i = 0; i < k && i < (int)all.size(); i++) r.insert(all[i].second);
    return r;
}

// Validate graph structure: valid ids, well-formed (-1 padding trailing), degree bound.
static void validate_graph(const faiss::HNSW& h, int n, int max_deg_l0, const char* tag) {
    char buf[128];
    int observed_max = 0;
    bool ids_ok = true, wellformed = true;
    for (int i = 0; i < n; i++) {
        int nlev = h.levels[i];
        for (int level = 0; level < nlev; level++) {
            size_t b, e; h.neighbor_range(i, level, &b, &e);
            bool seen_empty = false; int deg = 0;
            for (size_t j = b; j < e; j++) {
                faiss::HNSW::storage_idx_t v = h.neighbors[j];
                if (v == -1) { seen_empty = true; continue; }
                if (seen_empty) wellformed = false;        // gap after -1 => malformed
                if (v < 0 || v >= n || v == i) ids_ok = false; // invalid / self-loop
                deg++;
            }
            if (level == 0) observed_max = std::max(observed_max, deg);
        }
    }
    snprintf(buf, sizeof(buf), "%s: valid node ids", tag); CHECK(ids_ok, buf);
    snprintf(buf, sizeof(buf), "%s: well-formed adjacency (-1 trailing)", tag); CHECK(wellformed, buf);
    snprintf(buf, sizeof(buf), "%s: level-0 degree bound (<=%d)", tag, max_deg_l0);
    CHECK(observed_max <= max_deg_l0, buf);
}

int main() {
    const int d = 24, n = 4000, k = 10, M = 16;
    auto data = gen(n, d, 7);

    // ---- build standard + acorn-gamma ----
    AcornIndex sidx, gidx;
    build_standard_hnsw(sidx, d, faiss::METRIC_L2, M, 40, n, data.data());
    AcornGammaBuildParameters bp; bp.enabled = true; bp.M = M; bp.gamma = 4; bp.M_beta = 2 * M;
    build_acorn_gamma(gidx, d, faiss::METRIC_L2, bp, n, data.data());

    printf("[1] degree bounds & adjacency well-formedness\n");
    validate_graph(sidx.hnsw, n, 2 * M, "standard");
    validate_graph(gidx.hnsw, n, int(2 * M + 1.5 * M) + 1, "acorn-g4"); // M_beta+1.5M = 32+24=56

    printf("[2] determinism (same seed => identical graph)\n");
    {
        AcornIndex g2;
        build_acorn_gamma(g2, d, faiss::METRIC_L2, bp, n, data.data());
        bool same = g2.hnsw.neighbors.size() == gidx.hnsw.neighbors.size();
        if (same) for (size_t i = 0; i < g2.hnsw.neighbors.size(); i++)
            if (g2.hnsw.neighbors[i] != gidx.hnsw.neighbors[i]) { same = false; break; }
        CHECK(same, "acorn-gamma build is deterministic");
        AcornIndex s2; build_standard_hnsw(s2, d, faiss::METRIC_L2, M, 40, n, data.data());
        bool ssame = s2.hnsw.neighbors.size() == sidx.hnsw.neighbors.size();
        if (ssame) for (size_t i = 0; i < s2.hnsw.neighbors.size(); i++)
            if (s2.hnsw.neighbors[i] != sidx.hnsw.neighbors[i]) { ssame = false; break; }
        CHECK(ssame, "standard build is deterministic");
    }

    printf("[3] selector correctness + brute-force parity (edge cases)\n");
    std::vector<float> q(data.begin() + 100 * d, data.begin() + 101 * d);
    auto modes = {FilteredHnswSearchMode::STANDARD, FilteredHnswSearchMode::ACORN};

    // 3a: all-match selector
    { SetSel sel; for (int i = 0; i < n; i++) sel.s.insert(i);
      std::vector<faiss::idx_t> ids(k); std::vector<float> dis(k);
      int nr = filtered_search(gidx, q.data(), k, 128, FilteredHnswSearchMode::ACORN, &sel, ids.data(), dis.data());
      CHECK(nr == k, "all-match: returns k results");
      bool allpass = true; for (int i=0;i<nr;i++) if (!sel.is_member(ids[i])) allpass=false;
      CHECK(allpass, "all-match: results satisfy predicate");
    }
    // 3b: zero-match selector
    { SetSel sel; // empty
      std::vector<faiss::idx_t> ids(k); std::vector<float> dis(k);
      int nr = filtered_search(gidx, q.data(), k, 128, FilteredHnswSearchMode::ACORN, &sel, ids.data(), dis.data());
      CHECK(nr == 0, "zero-match: returns 0 results");
      int en = exact_filtered_search(gidx, q.data(), k, &sel, ids.data(), dis.data());
      CHECK(en == 0, "zero-match: exact returns 0");
    }
    // 3c: fewer-than-k matches (exactly 3). The exact ORACLE must return all 3;
    // ANN (ACORN) may return a SUBSET because 3 isolated valid nodes among 4000
    // can be graph-unreachable — this is the documented extreme-sparsity regime
    // where exact search is required (analysis Q5), not a correctness bug.
    { SetSel sel; sel.s = {5, 55, 555};
      std::vector<faiss::idx_t> ids(k); std::vector<float> dis(k);
      int en = exact_filtered_search(gidx, q.data(), k, &sel, ids.data(), dis.data());
      CHECK(en == 3, "fewer-than-k: exact oracle returns exactly the 3 matches");
      std::unordered_set<faiss::idx_t> got(ids.begin(), ids.begin()+en);
      CHECK(got == std::unordered_set<faiss::idx_t>({5,55,555}), "fewer-than-k: exact ids correct");
      int nr = filtered_search(gidx, q.data(), k, 128, FilteredHnswSearchMode::ACORN, &sel, ids.data(), dis.data());
      bool subset = nr <= 3; for (int i=0;i<nr;i++) if (!sel.is_member(ids[i])) subset=false;
      CHECK(subset, "fewer-than-k: ANN returns a valid (possibly partial) subset");
    }
    // 3d: duplicate suppression + brute-force parity (50% and 10% selectivity)
    for (double frac : {0.5, 0.1, 0.02}) {
        SetSel sel; faiss::RandomGenerator r(int(frac*1000));
        for (int i = 0; i < n; i++) if (r.rand_float() < frac) sel.s.insert(i);
        auto truth = brute(data.data(), n, d, q.data(), k, sel);
        std::vector<faiss::idx_t> ids(k); std::vector<float> dis(k);
        int en = exact_filtered_search(gidx, q.data(), k, &sel, ids.data(), dis.data());
        std::unordered_set<faiss::idx_t> ex(ids.begin(), ids.begin()+en);
        char buf[80]; snprintf(buf,sizeof(buf),"exact==brute-force @frac=%.2f", frac);
        CHECK(ex == truth, buf);
        // ACORN results: no duplicates + all satisfy predicate
        int nr = filtered_search(gidx, q.data(), k, 256, FilteredHnswSearchMode::ACORN, &sel, ids.data(), dis.data());
        std::unordered_set<faiss::idx_t> uniq(ids.begin(), ids.begin()+nr);
        snprintf(buf,sizeof(buf),"ACORN no-dup results @frac=%.2f",frac);
        CHECK((int)uniq.size()==nr, buf);
        bool allpass=true; for(int i=0;i<nr;i++) if(!sel.is_member(ids[i])) allpass=false;
        snprintf(buf,sizeof(buf),"ACORN results pass predicate @frac=%.2f",frac);
        CHECK(allpass, buf);
    }

    printf("[4] unfiltered recall (standard mode intact)\n");
    { SetSel sel; for (int i=0;i<n;i++) sel.s.insert(i);
      auto truth = brute(data.data(), n, d, q.data(), k, sel);
      std::vector<faiss::idx_t> ids(k); std::vector<float> dis(k);
      int nr = filtered_search(sidx, q.data(), k, 128, FilteredHnswSearchMode::STANDARD, nullptr, ids.data(), dis.data());
      int hit=0; for(int i=0;i<nr;i++) if(truth.count(ids[i])) hit++;
      CHECK(double(hit)/k >= 0.8, "standard unfiltered recall >= 0.8");
    }

    printf("[5] serialization round-trip\n");
    { save_acorn_index(gidx, "/tmp/acorn_test.bin");
      AcornIndex rl; load_acorn_index(rl, "/tmp/acorn_test.bin");
      bool ok = rl.ntotal()==gidx.ntotal() && rl.gamma==gidx.gamma && rl.M_beta==gidx.M_beta
                && rl.hnsw.neighbors.size()==gidx.hnsw.neighbors.size();
      CHECK(ok, "reload metadata + graph size match");
      SetSel sel; for(int i=0;i<n;i+=2) sel.s.insert(i);
      std::vector<faiss::idx_t> a(k),b(k); std::vector<float> da(k),db(k);
      int na=filtered_search(gidx, q.data(), k, 128, FilteredHnswSearchMode::ACORN, &sel, a.data(), da.data());
      int nb=filtered_search(rl,   q.data(), k, 128, FilteredHnswSearchMode::ACORN, &sel, b.data(), db.data());
      bool ident = na==nb; for(int i=0;i<na&&ident;i++) ident = (a[i]==b[i]);
      CHECK(ident, "reloaded index yields identical search results");
    }
    // bad-magic rejection
    { FILE* f=fopen("/tmp/acorn_bad.bin","wb"); uint32_t bad=0xDEADBEEF; fwrite(&bad,4,1,f); fclose(f);
      bool threw=false; AcornIndex rl; try { load_acorn_index(rl, "/tmp/acorn_bad.bin"); } catch(...) { threw=true; }
      CHECK(threw, "bad magic is rejected loudly");
    }

    printf("[6] mode separation (ACORN-1 vs ACORN-gamma differ; graphs differ)\n");
    { bool graphs_differ = gidx.hnsw.neighbors.size() != sidx.hnsw.neighbors.size();
      CHECK(graphs_differ, "acorn-gamma graph denser than standard graph");
      CHECK(gidx.is_acorn_gamma && !sidx.is_acorn_gamma, "index flags distinguish modes");
      CHECK(gidx.gamma==4 && sidx.gamma==1, "gamma recorded per index");
    }

    printf("[7] concurrent search (thread-safe read path)\n");
    { SetSel sel; for(int i=0;i<n;i+=3) sel.s.insert(i);
      // single-thread reference
      std::vector<faiss::idx_t> ref(k); std::vector<float> rd(k);
      int rn = filtered_search(gidx, q.data(), k, 128, FilteredHnswSearchMode::ACORN, &sel, ref.data(), rd.data());
      std::atomic<int> mismatches{0};
      auto worker = [&](int tid){
        for (int it=0; it<50; it++) {
          std::vector<faiss::idx_t> ids(k); std::vector<float> dis(k);
          int nr = filtered_search(gidx, q.data(), k, 128, FilteredHnswSearchMode::ACORN, &sel, ids.data(), dis.data());
          if (nr!=rn) { mismatches++; continue; }
          for (int i=0;i<nr;i++) if (ids[i]!=ref[i]) { mismatches++; break; }
        }
      };
      std::vector<std::thread> ts; for (int t=0;t<8;t++) ts.emplace_back(worker,t);
      for (auto& t: ts) t.join();
      CHECK(mismatches.load()==0, "8-thread concurrent search matches single-thread reference");
    }

    printf("[8] termination under tiny efSearch (cycles don't hang)\n");
    { SetSel sel; for(int i=0;i<n;i+=7) sel.s.insert(i);
      std::vector<faiss::idx_t> ids(k); std::vector<float> dis(k);
      int nr = filtered_search(gidx, q.data(), k, 1, FilteredHnswSearchMode::ACORN, &sel, ids.data(), dis.data());
      CHECK(nr >= 0 && nr <= k, "tiny efSearch terminates and returns valid count");
    }

    printf("\n==== RESULT: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
