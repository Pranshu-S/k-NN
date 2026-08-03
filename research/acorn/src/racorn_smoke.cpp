// Sanity: RACORN-1 / RACORN-1+ vs ACORN-1 / standard / exact on a NEGATIVE-
// correlation filter (valid cluster far from query) — the regime where ACORN-1
// collapses and RACORN should recover recall. Small scale.
#include <cstdio>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <faiss/utils/random.h>
#include "acorn.h"
using namespace acorn;
using clk = std::chrono::high_resolution_clock;
struct Sel: faiss::IDSelector { const std::vector<char>* m; bool is_member(faiss::idx_t id) const override { return (*m)[id]; } };

int main(){
    int n=20000, d=64, k=10, nc=20, nq=200;
    faiss::RandomGenerator rng(7);
    std::vector<float> centers((size_t)nc*d); for(auto&v:centers)v=rng.rand_float();
    std::vector<float> x((size_t)n*d); std::vector<int> cl(n); float sig=0.06f;
    for(int i=0;i<n;i++){int c=rng.rand_int(nc); cl[i]=c; for(int j=0;j<d;j++) x[(size_t)i*d+j]=centers[(size_t)c*d+j]+sig*(rng.rand_float()*2-1);}
    // queries from a fixed cluster 0; NEGATIVE filter = valid nodes in cluster 5 (far)
    std::vector<float> q((size_t)nq*d); for(int i=0;i<nq;i++){for(int j=0;j<d;j++) q[(size_t)i*d+j]=centers[0*d+j]+sig*(rng.rand_float()*2-1);}
    std::vector<char> mask(n,0); int cand=0; for(int i=0;i<n;i++) if(cl[i]==5){mask[i]=1;cand++;}
    printf("negative-correlation filter: cand=%d (%.2f%%), query cluster 0, valid cluster 5\n", cand, 100.0*cand/n);

    AcornIndex A; build_standard_hnsw(A,d,faiss::METRIC_L2,16,100,n,x.data());
    AcornGammaBuildParameters bp; bp.enabled=true; bp.M=16; bp.gamma=8; bp.M_beta=16; AcornIndex G; build_acorn_gamma(G,d,faiss::METRIC_L2,bp,n,x.data());
    Sel sel; sel.m=&mask;

    // ground truth per query (exact over valid)
    std::vector<std::unordered_set<faiss::idx_t>> truth(nq);
    { std::vector<faiss::idx_t> ids(k); std::vector<float> dd(k);
      for(int i=0;i<nq;i++){ int nr=exact_filtered_search(A,&q[(size_t)i*d],k,&sel,ids.data(),dd.data()); for(int j=0;j<nr;j++) truth[i].insert(ids[j]); } }

    auto eval=[&](const char* name, auto fn){
        std::vector<double> lat; double rec=0; int den=0; SearchStats agg;
        std::vector<faiss::idx_t> ids(k); std::vector<float> dd(k);
        for(int w=0;w<3;w++) fn(&q[0],ids.data(),dd.data(),nullptr);
        for(int i=0;i<nq;i++){ SearchStats st; auto t0=clk::now(); int nr=fn(&q[(size_t)i*d],ids.data(),dd.data(),&st); auto t1=clk::now();
            lat.push_back(std::chrono::duration<double,std::micro>(t1-t0).count()); agg.add(st);
            auto&tr=truth[i]; if(!tr.empty()){int h=0;for(int j=0;j<nr;j++)if(tr.count(ids[j]))h++; rec+=double(h)/tr.size(); den++;} }
        std::sort(lat.begin(),lat.end());
        printf("%-16s recall=%.3f  mean=%6.1fus p95=%6.1fus  ndis/q=%llu bridges/q=%llu aef=%llu\n",
            name, den?rec/den:0, std::accumulate(lat.begin(),lat.end(),0.0)/lat.size(), lat[(size_t)(0.95*(lat.size()-1))],
            (unsigned long long)(agg.dist_computations/nq),(unsigned long long)(agg.bridges_used/nq),(unsigned long long)agg.aef_switches);
    };
    int ef=200;
    eval("A faiss-std",  [&](const float*qq,faiss::idx_t*I,float*D,SearchStats*s){return filtered_search(A,qq,k,ef,FilteredHnswSearchMode::STANDARD,&sel,I,D,s);});
    eval("A' HNSW-infilt",[&](const float*qq,faiss::idx_t*I,float*D,SearchStats*s){return hnsw_infilter_search(A,qq,k,ef,&sel,I,D,s);});
    eval("B ACORN-1(pap)",[&](const float*qq,faiss::idx_t*I,float*D,SearchStats*s){RacornParams p; p.bridge_ratio=0.0; return racorn_search(A,qq,k,ef,&sel,p,I,D,s);});
    eval("B2 acorn(mine)",[&](const float*qq,faiss::idx_t*I,float*D,SearchStats*s){return filtered_search(A,qq,k,ef,FilteredHnswSearchMode::ACORN,&sel,I,D,s);});
    eval("E ACORN-gamma",[&](const float*qq,faiss::idx_t*I,float*D,SearchStats*s){return filtered_search(G,qq,k,ef,FilteredHnswSearchMode::ACORN,&sel,I,D,s);});
    eval("R RACORN-1",   [&](const float*qq,faiss::idx_t*I,float*D,SearchStats*s){RacornParams p; p.bridge_ratio=1.0; return racorn_search(A,qq,k,ef,&sel,p,I,D,s);});
    eval("R RACORN-1 br.25",[&](const float*qq,faiss::idx_t*I,float*D,SearchStats*s){RacornParams p; p.bridge_ratio=0.25; return racorn_search(A,qq,k,ef,&sel,p,I,D,s);});
    eval("R+ RACORN-1+", [&](const float*qq,faiss::idx_t*I,float*D,SearchStats*s){RacornParams p; p.bridge_ratio=1.0; p.enable_aef=true; p.aef_threshold=0.02; return racorn_search(A,qq,k,ef,&sel,p,I,D,s);});
    eval("F exact",      [&](const float*qq,faiss::idx_t*I,float*D,SearchStats*s){return exact_filtered_search(A,qq,k,&sel,I,D,s);});
    return 0;
}
