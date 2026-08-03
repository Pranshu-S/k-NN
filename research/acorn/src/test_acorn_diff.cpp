/*
 * Differential + before/after: corrected JNI ACORN-1 (knn_jni::acorn::search)
 * vs the research reference (acorn::filtered_search, ACORN mode) on the SAME
 * HNSW graph and vectors. Reports recall-vs-brute-force for both, distance
 * computations, and the corrected impl's bridge_reexpansions_avoided (the repeated
 * bridge expansions the reference performs and the fix eliminates).
 *
 * Correlation is per-query-local (pos = nearest to the query, neg = farthest,
 * rand = random) matching the main benchmark methodology.
 */
#include <cstdio>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <numeric>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/random.h>

#include "acorn_hnsw.h"   // corrected JNI ACORN-1
#include "acorn.h"        // research reference (namespace acorn)

using faiss::idx_t;

struct SetSel : faiss::IDSelector {
    std::unordered_set<idx_t> m;
    bool is_member(idx_t id) const override { return m.count(id) > 0; }
};

int main() {
    const int d = 64, n = 5000, k = 10, ef = 200, NQ = 25;
    faiss::RandomGenerator rng(2024);
    std::vector<float> x((size_t)n * d); for (auto& v : x) v = rng.rand_float();
    std::vector<float> q((size_t)NQ * d); for (auto& v : q) v = rng.rand_float();

    // ---- research reference graph, then SHARE it with a faiss IndexIDMap ----
    acorn::AcornIndex ai;
    acorn::build_standard_hnsw(ai, d, faiss::METRIC_L2, 16, 100, n, x.data());
    ai.gamma = 1; ai.M_beta = 32;

    auto* hf = new faiss::IndexHNSWFlat(d, 16, faiss::METRIC_L2);
    faiss::IndexIDMap idmap(hf); idmap.own_fields = true;  // wrap while empty (faiss requires it)
    hf->storage->add(n, x.data());   // vectors in ordinal order 0..n-1
    hf->hnsw = ai.hnsw;              // copy the SAME graph
    hf->ntotal = n;
    idmap.id_map.resize(n); std::iota(idmap.id_map.begin(), idmap.id_map.end(), 0); // identity
    idmap.ntotal = n;

    // per-query distance order (for pos/neg correlation)
    std::vector<std::vector<int>> ord(NQ);
    for (int i = 0; i < NQ; i++) {
        std::vector<std::pair<float,int>> ds(n); const float* qi = &q[(size_t)i*d];
        for (int b = 0; b < n; b++) { float s=0; for (int j=0;j<d;j++){float df=qi[j]-x[(size_t)b*d+j]; s+=df*df;} ds[b]={s,b}; }
        std::sort(ds.begin(), ds.end());
        ord[i].resize(n); for (int b=0;b<n;b++) ord[i][b]=ds[b].second;
    }
    auto mkmask = [&](const char* corr, int cand, int qi){
        SetSel s;
        if (std::string(corr)=="rand"){ faiss::RandomGenerator r(1000+qi); int p=0; while(p<cand){int id=r.rand_int(n); s.m.insert(id); p=s.m.size();} }
        else if (std::string(corr)=="pos"){ for(int j=0;j<cand;j++) s.m.insert(ord[qi][j]); }
        else { for(int j=0;j<cand;j++) s.m.insert(ord[qi][n-1-j]); }
        return s;
    };
    auto brute = [&](const float* qi, const SetSel& s){
        std::vector<std::pair<float,idx_t>> sc;
        for (int i=0;i<n;i++){ if(!s.m.count(i)) continue; float d2=0; for(int j=0;j<d;j++){float df=qi[j]-x[(size_t)i*d+j]; d2+=df*df;} sc.push_back({d2,i}); }
        std::sort(sc.begin(),sc.end()); std::unordered_set<idx_t> t; for(int i=0;i<k&&i<(int)sc.size();i++) t.insert(sc[i].second); return t;
    };

    printf("%-6s %6s | %-22s | %-22s | %s\n", "corr","sel","reference (before)","corrected JNI (after)","fix");
    printf("%-6s %6s | %8s %12s | %8s %12s | %s\n","","","recall","ndis","recall","ndis","bridge_reexp_avoided");
    const char* corrs[3] = {"pos","rand","neg"};
    int cands[4] = { n/1000, n/100, n/10, n/2 };  // 0.1%, 1%, 10%, 50%
    for (const char* corr : corrs) {
        for (int cand : cands) {
            double recRef=0, recJni=0, ndisRef=0, ndisJni=0, avoided=0; int den=0;
            std::vector<idx_t> ii(k); std::vector<float> dd(k);
            for (int i=0;i<NQ;i++){
                SetSel sel = mkmask(corr, cand, i);
                auto truth = brute(&q[(size_t)i*d], sel);
                // reference (before)
                acorn::SearchStats sR;
                int rR = acorn::filtered_search(ai, &q[(size_t)i*d], k, ef, acorn::FilteredHnswSearchMode::ACORN, &sel, ii.data(), dd.data(), &sR);
                int hR=0; for(int j=0;j<rR;j++) if(truth.count(ii[j])) hR++;
                // corrected jni (after)
                int rJ = knn_jni::acorn::search(&idmap, &q[(size_t)i*d], k, ef, &sel, 1, dd.data(), ii.data(), true);
                auto& sJ = knn_jni::acorn::last_stats();
                int hJ=0; for(int j=0;j<rJ;j++) if(truth.count(ii[j])) hJ++;
                if(!truth.empty()){ recRef+=double(hR)/truth.size(); recJni+=double(hJ)/truth.size(); den++; }
                ndisRef+=sR.dist_computations; ndisJni+=sJ.distance_computations; avoided+=sJ.bridge_reexpansions_avoided;
            }
            printf("%-6s %5.0f%% | %8.2f %12.0f | %8.2f %12.0f | %.0f\n",
                   corr, 100.0*cand/n, den?recRef/den:0, ndisRef/NQ, den?recJni/den:0, ndisJni/NQ, avoided/NQ);
        }
    }
    return 0;
}
