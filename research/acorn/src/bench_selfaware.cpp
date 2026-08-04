/*
 * Self-aware ANN: inline early-abort vs post-hoc fallback (OpenSearch today) vs plain.
 * Standard in-filtering base (the ES-parity path). Three fallback modes:
 *   NONE    : plain filtered ANN (collapses under negative correlation).
 *   POSTHOC : run the FULL ANN; if < k results -> exact. (OpenSearch's current behaviour.)
 *   INLINE  : monitor a running loss signal during the walk; once past a min-probe gate,
 *             if pass-ratio collapses OR the top-k stalls, ABORT mid-walk -> exact.
 * Reports recall / latency / ANN nodes examined before the fallback fired / exact%.
 * Real SIFT, per-query-local correlation, bitset selector.
 * Usage: bench_selfaware [n] [passThr] [stallLimit]   (default 100000 0.02 64)
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
using idx_t=faiss::idx_t; using sidx=faiss::HNSW::storage_idx_t;
enum Mode { NONE, POSTHOC, INLINE };
struct SetSel : faiss::IDSelector { std::vector<char> bit; bool is_member(idx_t id) const override { return id>=0&&(size_t)id<bit.size()&&bit[id]; } };
static int D=128; static const float* XG=nullptr;
static bool load_fvecs(const char*p,int want,std::vector<float>&o,int&d){FILE*f=fopen(p,"rb");if(!f)return false;int dd=0;if(fread(&dd,4,1,f)!=1){fclose(f);return false;}fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);long rec=4+4L*dd,tot=b/rec;int n=want>0?std::min<long>(want,tot):tot;o.resize((size_t)n*dd);for(int i=0;i<n;i++){int t;if(fread(&t,4,1,f)!=1||t!=dd){fclose(f);return false;}if(fread(&o[(size_t)i*dd],4,dd,f)!=(size_t)dd){fclose(f);return false;}}fclose(f);d=dd;return true;}
static inline float fl2(const float*q,sidx v){const float*x=&XG[(size_t)v*D];float s=0;for(int j=0;j<D;j++){float df=q[j]-x[j];s+=df*df;}return s;}
static int exact_full(const float*q,const SetSel&sel,faiss::IndexIDMap*map,int N,int k,std::vector<float>&od,std::vector<idx_t>&oi,uint64_t&exdis){
    auto ext=[&](sidx v){return map->id_map[v];};std::vector<float> D2(k);std::vector<idx_t> I2(k);int nres=0;
    for(int i=0;i<N;i++){if(!sel.is_member(ext(i)))continue;float d=fl2(q,i);exdis++;if(nres<k)faiss::maxheap_push(++nres,D2.data(),I2.data(),d,i);else if(d<D2[0])faiss::maxheap_replace_top(nres,D2.data(),I2.data(),d,i);}
    std::vector<std::pair<float,idx_t>> t;for(int i=0;i<nres;i++)t.push_back({D2[i],I2[i]});std::sort(t.begin(),t.end());
    for(size_t i=0;i<t.size();i++){od[i]=t[i].first;oi[i]=ext(t[i].second);}for(int i=(int)t.size();i<k;i++)oi[i]=-1;return nres;
}
static int search(faiss::IndexIDMap*map,const float*q,int k,int ef,const SetSel&sel,Mode mode,int minProbe,double passThr,int stallLimit,
                  std::vector<float>&od,std::vector<idx_t>&oi,uint64_t*annExam,bool*usedExact){
    auto*ih=dynamic_cast<faiss::IndexHNSW*>(map->index);const faiss::HNSW&h=ih->hnsw;const int N=ih->ntotal;
    auto ext=[&](sidx v){return map->id_map[v];};if(usedExact)*usedExact=false;
    uint64_t examined=0,passed=0; int stall=0;
    sidx nearest=h.entry_point;float dn=fl2(q,nearest);
    for(int lvl=h.max_level;lvl>=1;lvl--){for(;;){auto pv=nearest;size_t b,e;h.neighbor_range(nearest,lvl,&b,&e);for(size_t i=b;i<e;i++){auto v=h.neighbors[i];if(v<0)break;float d=fl2(q,v);if(d<dn){dn=d;nearest=v;}}if(nearest==pv)break;}}
    const int efn=std::max(ef,k);faiss::HNSW::MinimaxHeap cand(efn);
    std::vector<float> Dh(k);std::vector<idx_t> Ih(k);int nres=0;faiss::VisitedTable vis(N);
    auto add=[&](sidx v){vis.set(v);float d=fl2(q,v);bool imp=false;if(sel.is_member(ext(v))){if(nres<k){faiss::maxheap_push(++nres,Dh.data(),Ih.data(),d,v);imp=true;}else if(d<Dh[0]){faiss::maxheap_replace_top(nres,Dh.data(),Ih.data(),d,v);imp=true;}}cand.push(v,d);return imp;};
    add(nearest);
    while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;
        // INLINE self-abort: route to exact ONLY when the filter is fighting the walk
        // (pass-ratio collapsed). A stall means the ANN converged with good results ->
        // that is handled by the normal relative-distance stop, NOT an exact fallback.
        (void)stall; (void)stallLimit;
        if(mode==INLINE && examined>(uint64_t)minProbe && (double)passed/examined<passThr){
            if(usedExact)*usedExact=true; if(annExam)*annExam=examined; uint64_t ex=0; int r=exact_full(q,sel,map,N,k,od,oi,ex); return r;
        }
        size_t b,e;h.neighbor_range(v0,0,&b,&e); bool improvedThisStep=false;
        for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;if(!vis.get(v1)){examined++;if(sel.is_member(ext(v1)))passed++;if(add(v1))improvedThisStep=true;}}
        if(nres>=k){ if(improvedThisStep)stall=0; else stall++; }
    }
    if(annExam)*annExam=examined;
    // POSTHOC fallback: full ANN done; too few results -> exact
    if(mode==POSTHOC && nres<k){ if(usedExact)*usedExact=true; uint64_t ex=0; return exact_full(q,sel,map,N,k,od,oi,ex); }
    std::vector<std::pair<float,idx_t>> t;for(int i=0;i<nres;i++)t.push_back({Dh[i],Ih[i]});std::sort(t.begin(),t.end());
    for(size_t i=0;i<t.size();i++){od[i]=t[i].first;oi[i]=ext(t[i].second);}for(int i=(int)t.size();i<k;i++)oi[i]=-1;return nres;
}
int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):100000; double passThr=argc>2?atof(argv[2]):0.02; int stallLimit=argc>3?atoi(argv[3]):64; const int K=10,NQ=50;
    std::vector<float> X;int d=128;if(!load_fvecs("research/acorn/data/sift/sift_base.fvecs",N,X,d)){fprintf(stderr,"need sift\n");return 1;}N=(int)(X.size()/d);D=d;XG=X.data();
    std::vector<float> Q;int qd=0;load_fvecs("research/acorn/data/sift/sift_query.fvecs",NQ,Q,qd);
    std::vector<idx_t> ids(N);for(int i=0;i<N;i++)ids[i]=i;
    auto*hh=new faiss::IndexHNSWFlat(d,16,faiss::METRIC_L2);hh->hnsw.efConstruction=100;auto*map=new faiss::IndexIDMap(hh);map->own_fields=true;map->add_with_ids(N,X.data(),ids.data());
    fprintf(stderr,"[bench_selfaware] n=%d passThr=%.3f stall=%d built\n",N,passThr,stallLimit);
    std::vector<std::vector<int>> ord(NQ);
    for(int i=0;i<NQ;i++){std::vector<std::pair<float,int>> ds(N);const float*q=&Q[(size_t)i*d];for(int b=0;b<N;b++)ds[b]={fl2(q,b),b};std::sort(ds.begin(),ds.end());ord[i].resize(N);for(int b=0;b<N;b++)ord[i][b]=ds[b].second;}
    auto mask=[&](const char*c,int cand,int qi){SetSel s;s.bit.assign(N,0);if(std::string(c)=="no"){faiss::RandomGenerator r(1000+qi);int p=0;while(p<cand){int id=r.rand_int(N);if(!s.bit[id]){s.bit[id]=1;p++;}}}else if(std::string(c)=="pos"){for(int j=0;j<cand;j++)s.bit[ord[qi][j]]=1;}else{for(int j=0;j<cand;j++)s.bit[ord[qi][N-1-j]]=1;}return s;};
    auto brute=[&](const float*q,const SetSel&s){std::vector<std::pair<float,idx_t>> sc;for(int i=0;i<N;i++){if(!s.bit[i])continue;sc.push_back({fl2(q,i),i});}std::sort(sc.begin(),sc.end());std::unordered_set<idx_t> t;for(int i=0;i<K&&i<(int)sc.size();i++)t.insert(sc[i].second);return t;};
    using clk=std::chrono::high_resolution_clock;int EFS[3]={100,250,500};double sels[5]={0.001,0.01,0.05,0.10,0.25};
    struct Cfg{const char*n;Mode m;}; Cfg cfgs[]={{"plain",NONE},{"post-hoc(OS today)",POSTHOC},{"inline-abort",INLINE}};
    for(const char* corr:{"neg","no"}){
        printf("\n########## correlation=%s : standard base (rec / latency-us / ANN-examined / exact%%) ##########\n",corr);
        printf("%6s",""); for(auto&c:cfgs)printf(" | %-26s",c.n); printf("\n");
        for(double se:sels){int cand=std::max(K,(int)(se*N));
            std::vector<SetSel> M(NQ);std::vector<std::unordered_set<idx_t>> tr(NQ);for(int i=0;i<NQ;i++){M[i]=mask(corr,cand,i);tr[i]=brute(&Q[(size_t)i*d],M[i]);}
            printf("%5.1f%%",100.0*cand/N);
            for(auto&c:cfgs){ double bR=-1,bL=0,bA=0,bE=0;
                for(int ef:EFS){double r=0,l=0,a=0;int ex=0,den=0;int minProbe=3*std::max(ef,K);std::vector<float> dd(K);std::vector<idx_t> ii(K);
                    for(int i=0;i<NQ;i++){uint64_t ann=0;bool used=false;auto t0=clk::now();int rr=search(map,&Q[(size_t)i*d],K,ef,M[i],c.m,minProbe,passThr,stallLimit,dd,ii,&ann,&used);auto t1=clk::now();
                        l+=std::chrono::duration<double,std::micro>(t1-t0).count();a+=ann;if(used)ex++;int h=0;for(int j=0;j<rr;j++)if(tr[i].count(ii[j]))h++;if(!tr[i].empty()){r+=double(h)/tr[i].size();den++;}}
                    double R=den?r/den:0;if(R>bR+1e-9){bR=R;bL=l/NQ;bA=a/NQ;bE=100.0*ex/NQ;} }
                printf(" | %.2f/%5.0f/%6.0f/%3.0f",bR,bL,bA,bE);
            }
            printf("\n");
        }
    }
    fflush(stdout); std::_Exit(0);
}
