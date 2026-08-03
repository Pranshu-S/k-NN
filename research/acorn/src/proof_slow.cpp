/*
 * SIMPLIFIED PROOF: today's OpenSearch filtered k-NN does not scale for a
 * selective filter that is not aligned with the query.
 *
 * ONE fixed scenario, index size N swept up. Filter = 1% of the corpus, in a
 * vector-space region DIFFERENT from the query ("show me products in category X"
 * where the query vector isn't in X — a routine cross-domain filter). Same
 * queries, same filter fraction at every N. We measure, at matched target recall:
 *   - "OpenSearch today (HNSW filtered, recall-preserving)"  = hnsw_infilter
 *   - "Exact filtered (brute force over eligible)"           = exact
 *   - "RACORN-1+ (proposed fix)"                             = racorn + AEF
 * reporting recall, distance computations (work), and wall-clock.
 *
 * The proof: HNSW's work grows ~O(N) (it must scan a large fraction of the whole
 * index to keep recall), so a 1% filter on a 1M index makes it touch ~the entire
 * million vectors — while the answer only needs the 1% eligible set.
 */
#include <cstdio>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cmath>
#include <faiss/utils/random.h>
#include "acorn.h"
using namespace acorn;
using clk = std::chrono::high_resolution_clock;
struct Sel: faiss::IDSelector { const std::vector<char>* m; bool is_member(faiss::idx_t id) const override { return (*m)[id]; } };

struct Res { double recall, mean_us, ndis, p95_us; };
template<class Fn>
static Res run(const std::vector<float>& q, int d, int nq, int k,
               const std::vector<std::unordered_set<faiss::idx_t>>& truth, Fn fn){
    std::vector<double> lat; double rec=0; int den=0; double nd=0;
    std::vector<faiss::idx_t> ids(k); std::vector<float> dd(k);
    for(int w=0;w<2;w++) fn(&q[0],ids.data(),dd.data(),nullptr);
    for(int i=0;i<nq;i++){ SearchStats st; auto t0=clk::now(); int nr=fn(&q[(size_t)i*d],ids.data(),dd.data(),&st); auto t1=clk::now();
        lat.push_back(std::chrono::duration<double,std::micro>(t1-t0).count()); nd+=st.dist_computations;
        auto&tr=truth[i]; if(!tr.empty()){int h=0;for(int j=0;j<nr;j++)if(tr.count(ids[j]))h++; rec+=double(h)/tr.size(); den++;} }
    std::sort(lat.begin(),lat.end());
    return Res{ den?rec/den:0, std::accumulate(lat.begin(),lat.end(),0.0)/lat.size(), nd/nq, lat[(size_t)(0.95*(lat.size()-1))] };
}

int main(int argc, char** argv){
    int d=128, k=10, nq=20, ef=200; double SEL=0.01;
    std::vector<int> Ns = {100000, 250000, 500000, 1000000};
    if (argc>1){ Ns.clear(); for(int i=1;i<argc;i++) Ns.push_back(atoi(argv[i])); }
    printf("Scenario: %.0f%%-selective filter, eligible set in a DIFFERENT vector region than the query.\n", SEL*100);
    printf("d=%d k=%d ef_search=%d, %d queries.  'work' = distance computations per query.\n\n", d,k,ef,nq);
    printf("%9s | %-34s | %7s | %10s | %10s\n","N (index)","approach","recall","work/query","latency");
    printf("%9s-+-%-34s-+-%7s-+-%10s-+-%10s\n","---------","----------------------------------","-------","----------","----------");
    for(int N : Ns){
        int nc=std::max(16,N/2000); faiss::RandomGenerator rng(1);
        std::vector<float> centers((size_t)nc*d); for(auto&v:centers)v=rng.rand_float();
        std::vector<float> x((size_t)N*d); std::vector<int> cl(N); float sig=0.06f;
        for(int i=0;i<N;i++){int c=rng.rand_int(nc); cl[i]=c; for(int j=0;j<d;j++) x[(size_t)i*d+j]=centers[(size_t)c*d+j]+sig*(rng.rand_float()*2-1);}
        std::vector<float> q((size_t)nq*d); for(int i=0;i<nq;i++){for(int j=0;j<d;j++) q[(size_t)i*d+j]=centers[0*d+j]+sig*(rng.rand_float()*2-1);}
        // far cluster from cluster 0
        int fc=1; float best=-1; for(int c=0;c<nc;c++){float s=0;for(int j=0;j<d;j++){float df=centers[j]-centers[(size_t)c*d+j];s+=df*df;} if(s>best){best=s;fc=c;}}
        int want=(int)(SEL*N);
        std::vector<std::pair<float,int>> dist(N);
        for(int i=0;i<N;i++){float s=0;for(int j=0;j<d;j++){float df=centers[(size_t)fc*d+j]-x[(size_t)i*d+j];s+=df*df;} dist[i]={s,i};}
        std::partial_sort(dist.begin(),dist.begin()+want,dist.end());
        std::vector<char> mask(N,0); for(int i=0;i<want;i++) mask[dist[i].second]=1;
        Sel sel; sel.m=&mask;
        AcornIndex A; build_standard_hnsw(A,d,faiss::METRIC_L2,16,100,N,x.data());
        std::vector<std::unordered_set<faiss::idx_t>> truth(nq);
        { std::vector<faiss::idx_t> ids(k); std::vector<float> dd(k);
          for(int i=0;i<nq;i++){int nr=exact_filtered_search(A,&q[(size_t)i*d],k,&sel,ids.data(),dd.data()); for(int j=0;j<nr;j++)truth[i].insert(ids[j]);} }

        auto today = run(q,d,nq,k,truth,[&](const float*qq,faiss::idx_t*I,float*D,SearchStats*s){return hnsw_infilter_search(A,qq,k,ef,&sel,I,D,s);});
        auto exact = run(q,d,nq,k,truth,[&](const float*qq,faiss::idx_t*I,float*D,SearchStats*s){return exact_filtered_search(A,qq,k,&sel,I,D,s);});
        RacornParams rp; rp.bridge_ratio=1.0; rp.enable_aef=true; rp.aef_threshold=2000.0/N;
        auto fix = run(q,d,nq,k,truth,[&](const float*qq,faiss::idx_t*I,float*D,SearchStats*s){return racorn_search(A,qq,k,ef,&sel,rp,I,D,s);});

        auto row=[&](const char* name, const Res& r){
            printf("%9d | %-34s | %6.2f  | %9.0f  | %8.2fms\n", N, name, r.recall, r.ndis, r.mean_us/1000.0); };
        printf("%9d | eligible (1%% of index)             |         | %9d  |\n", N, want);
        row("OpenSearch today (filtered HNSW)", today);
        row("Exact over eligible", exact);
        row("RACORN-1+ (proposed fix)", fix);
        printf("%9s-+-%-34s-+-%7s-+-%10s-+-%10s\n","---------","----------------------------------","-------","----------","----------");
        fflush(stdout);
    }
    return 0;
}
