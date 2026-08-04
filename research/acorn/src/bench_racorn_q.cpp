/*
 * RACORN (ASF+AEF over STD / ACORN-1 / ACORN-gamma) WITH the ES-style quantize+rescore
 * recipe: cheap binary first pass + full-precision rescore.
 *   - Quantization: 1-bit, mean-relative sign (BBQ-style; SIFT is non-negative so
 *     bit_d = x_d > mean_d). 128-D -> 128 bits (2x uint64). First-pass distance = Hamming.
 *   - Rescore: the walk keeps firstPassK = k*oversample candidates by Hamming; then re-rank
 *     them with full-precision L2 -> top-k.
 *   - AEF (exact fallback) always uses full precision.
 * Compares, per base+ASF+AEF: full-precision vs quantized+rescore (recall / latency).
 * Usage: bench_racorn_q [n] [gamma] [oversample]   (default 100000 12 10)
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
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
enum Base { STD, ACORN1, GAMMA };
struct SetSel : faiss::IDSelector { std::vector<char> bit; bool is_member(idx_t id) const override { return id>=0&&(size_t)id<bit.size()&&bit[id]; } };
static bool load_fvecs(const char*p,int want,std::vector<float>&o,int&d){FILE*f=fopen(p,"rb");if(!f)return false;int dd=0;if(fread(&dd,4,1,f)!=1){fclose(f);return false;}fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);long rec=4+4L*dd,tot=b/rec;int n=want>0?std::min<long>(want,tot):tot;o.resize((size_t)n*dd);for(int i=0;i<n;i++){int t;if(fread(&t,4,1,f)!=1||t!=dd){fclose(f);return false;}if(fread(&o[(size_t)i*dd],4,dd,f)!=(size_t)dd){fclose(f);return false;}}fclose(f);d=dd;return true;}

static int D=128, NW=2;                    // NW = 64-bit words per code (128-D -> 2)
static std::vector<uint64_t> CODES;        // N * NW binary codes
static std::vector<float> MEANS;           // per-dim mean
static const float* XG=nullptr;
static inline void encode(const float* v, uint64_t* out){ for(int w=0;w<NW;w++)out[w]=0; for(int j=0;j<D;j++) if(v[j]>MEANS[j]) out[j>>6]|=(1ULL<<(j&63)); }
static inline int hamming(const uint64_t* a, const uint64_t* b){ int s=0; for(int w=0;w<NW;w++) s+=__builtin_popcountll(a[w]^b[w]); return s; }
static inline float fl2(const float* q, sidx v){ const float* x=&XG[(size_t)v*D]; float s=0; for(int j=0;j<D;j++){float df=q[j]-x[j]; s+=df*df;} return s; }

static int exact_full(const float*q,const SetSel&sel,faiss::IndexIDMap*map,int N,int k,std::vector<float>&od,std::vector<idx_t>&oi){
    auto ext=[&](sidx v){return map->id_map[v];}; std::vector<float> D2(k); std::vector<idx_t> I2(k); int nres=0;
    for(int i=0;i<N;i++){ if(!sel.is_member(ext(i)))continue; float d=fl2(q,i); if(nres<k)faiss::maxheap_push(++nres,D2.data(),I2.data(),d,i); else if(d<D2[0])faiss::maxheap_replace_top(nres,D2.data(),I2.data(),d,i); }
    std::vector<std::pair<float,idx_t>> t;for(int i=0;i<nres;i++)t.push_back({D2[i],I2[i]});std::sort(t.begin(),t.end());
    for(size_t i=0;i<t.size();i++){od[i]=t[i].first;oi[i]=ext(t[i].second);}for(int i=(int)t.size();i<k;i++)oi[i]=-1; return nres;
}

// base+ASF+AEF filtered search; quant=false -> full-precision L2; quant=true -> Hamming first
// pass keeping firstPassK by Hamming, then full-L2 rescore to k.
static int rsearch(faiss::IndexIDMap*map,const float*q,const uint64_t*qc,int k,int ef,const SetSel&sel,Base base,
                   bool asf,bool aef,double aef_thr,int Mbase,int stride,bool quant,int firstPassK,
                   std::vector<float>&od,std::vector<idx_t>&oi){
    auto*ih=dynamic_cast<faiss::IndexHNSW*>(map->index); const faiss::HNSW&h=ih->hnsw; const int N=ih->ntotal;
    auto ext=[&](sidx v){return map->id_map[v];};
    auto dist=[&](sidx v)->float{ return quant ? (float)hamming(qc,&CODES[(size_t)v*NW]) : fl2(q,v); };
    const int kcap = quant ? firstPassK : k;                 // first pass keeps more when quantized
    uint64_t examined=0,passed=0;
    sidx nearest=h.entry_point; float dn=dist(nearest);
    for(int lvl=h.max_level;lvl>=1;lvl--){for(;;){auto pv=nearest;size_t b,e;h.neighbor_range(nearest,lvl,&b,&e);for(size_t i=b;i<e;i++){auto v=h.neighbors[i];if(v<0)break;float d=dist(v);if(d<dn){dn=d;nearest=v;}}if(nearest==pv)break;}}
    const int efn=std::max(ef,kcap); faiss::HNSW::MinimaxHeap cand(efn);
    std::vector<float> Dh(kcap); std::vector<idx_t> Ih(kcap); int nres=0; faiss::VisitedTable vis(N),brd(N);
    auto admit=[&](sidx v,bool isRes){ vis.set(v); float d=dist(v); if(isRes){ if(nres<kcap)faiss::maxheap_push(++nres,Dh.data(),Ih.data(),d,v); else if(d<Dh[0])faiss::maxheap_replace_top(nres,Dh.data(),Ih.data(),d,v);} cand.push(v,d); };
    { bool p=sel.is_member(ext(nearest)); if(base==STD)admit(nearest,p); else if(p)admit(nearest,true); else cand.push(nearest,dn); }
    while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;
        size_t b,e;h.neighbor_range(v0,0,&b,&e); int found=0; std::vector<sidx> failing;
        for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;examined++;bool p=sel.is_member(ext(v1));if(p)passed++;
            if(base==STD){ if(!vis.get(v1))admit(v1,p); }
            else{ if(p){ if(!vis.get(v1)){admit(v1,true);found++;} }
                  else{ failing.push_back(v1);
                        if(base==ACORN1&&!brd.get(v1)){brd.set(v1);size_t b2,e2;h.neighbor_range(v1,0,&b2,&e2);for(size_t j2=b2;j2<e2;j2++){auto v2=h.neighbors[j2];if(v2<0)break;examined++;bool p2=sel.is_member(ext(v2));if(p2)passed++;if(p2&&!vis.get(v2)){admit(v2,true);found++;}}} } }
        }
        if(asf&&base!=STD&&found<Mbase&&!failing.empty()){int stp=std::max(1,stride);for(size_t i=0;i<failing.size();i+=stp){sidx fb=failing[i];if(!vis.get(fb))admit(fb,false);}}
        if(aef&&examined>(uint64_t)(3*efn)&&(double)passed/examined<aef_thr){ return exact_full(q,sel,map,N,k,od,oi); }
    }
    // gather first-pass results
    std::vector<std::pair<float,sidx>> cands; for(int i=0;i<nres;i++)cands.push_back({Dh[i],Ih[i]});
    if(quant){ // rescore with full precision
        std::vector<std::pair<float,sidx>> rr; for(auto&c:cands) rr.push_back({fl2(q,c.second),c.second});
        std::sort(rr.begin(),rr.end()); int m=std::min((int)rr.size(),k);
        for(int i=0;i<m;i++){od[i]=rr[i].first;oi[i]=ext(rr[i].second);} for(int i=m;i<k;i++)oi[i]=-1; return m;
    }
    std::sort(cands.begin(),cands.end()); int m=std::min((int)cands.size(),k);
    for(int i=0;i<m;i++){od[i]=cands[i].first;oi[i]=ext(cands[i].second);} for(int i=m;i<k;i++)oi[i]=-1; return m;
}

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):100000,G=argc>2?atoi(argv[2]):12,OS=argc>3?atoi(argv[3]):10; const int K=10,NQ=50,Mbase=16; double aefT=0.02;
    std::vector<float> X;int d=128; if(!load_fvecs("research/acorn/data/sift/sift_base.fvecs",N,X,d)){fprintf(stderr,"need sift\n");return 1;} N=(int)(X.size()/d); D=d; NW=(d+63)/64; XG=X.data();
    std::vector<float> Q;int qd=0; load_fvecs("research/acorn/data/sift/sift_query.fvecs",NQ,Q,qd);
    // means + codes
    MEANS.assign(d,0); for(int i=0;i<N;i++)for(int j=0;j<d;j++)MEANS[j]+=X[(size_t)i*d+j]; for(int j=0;j<d;j++)MEANS[j]/=N;
    CODES.assign((size_t)N*NW,0); for(int i=0;i<N;i++)encode(&X[(size_t)i*d],&CODES[(size_t)i*NW]);
    std::vector<uint64_t> QC((size_t)NQ*NW,0); for(int i=0;i<NQ;i++)encode(&Q[(size_t)i*d],&QC[(size_t)i*NW]);
    std::vector<idx_t> ids(N);for(int i=0;i<N;i++)ids[i]=i;
    auto build=[&](int M){auto*hh=new faiss::IndexHNSWFlat(d,M,faiss::METRIC_L2);hh->hnsw.efConstruction=100;auto*m=new faiss::IndexIDMap(hh);m->own_fields=true;m->add_with_ids(N,X.data(),ids.data());return m;};
    auto* smap=build(16); auto* gmap=build(16*G);
    fprintf(stderr,"[bench_racorn_q] n=%d gamma=%d oversample=%d firstPassK=%d built\n",N,G,OS,K*OS);
    std::vector<std::vector<int>> ord(NQ);
    for(int i=0;i<NQ;i++){std::vector<std::pair<float,int>> ds(N);const float*q=&Q[(size_t)i*d];for(int b=0;b<N;b++)ds[b]={fl2(q,b),b};std::sort(ds.begin(),ds.end());ord[i].resize(N);for(int b=0;b<N;b++)ord[i][b]=ds[b].second;}
    auto mask=[&](const char*c,int cand,int qi){SetSel s;s.bit.assign(N,0);if(std::string(c)=="no"){faiss::RandomGenerator r(1000+qi);int p=0;while(p<cand){int id=r.rand_int(N);if(!s.bit[id]){s.bit[id]=1;p++;}}}else if(std::string(c)=="pos"){for(int j=0;j<cand;j++)s.bit[ord[qi][j]]=1;}else{for(int j=0;j<cand;j++)s.bit[ord[qi][N-1-j]]=1;}return s;};
    auto brute=[&](const float*q,const SetSel&s){std::vector<std::pair<float,idx_t>> sc;for(int i=0;i<N;i++){if(!s.bit[i])continue;sc.push_back({fl2(q,i),i});}std::sort(sc.begin(),sc.end());std::unordered_set<idx_t> t;for(int i=0;i<K&&i<(int)sc.size();i++)t.insert(sc[i].second);return t;};
    using clk=std::chrono::high_resolution_clock; int EFS[4]={50,100,250,500}; double sels[4]={0.001,0.01,0.05,0.25};
    struct Cfg{const char*name;Base base;faiss::IndexIDMap*map;}; Cfg bases[]={{"STD",STD,smap},{"ACORN1",ACORN1,smap},{"GAMMA",GAMMA,gmap}};
    for(const char* corr:{"neg","no"}){
        printf("\n########## correlation=%s : base+ASF+AEF, FP vs Quant+Rescore (rec / latency-us) ##########\n",corr);
        printf("%6s","sel"); for(auto&bc:bases){printf(" | %-13s FP | %-8s Q+R",bc.name,bc.name);} printf("\n");
        for(double se:sels){int cand=std::max(K,(int)(se*N));
            std::vector<SetSel> M(NQ);std::vector<std::unordered_set<idx_t>> tr(NQ);for(int i=0;i<NQ;i++){M[i]=mask(corr,cand,i);tr[i]=brute(&Q[(size_t)i*d],M[i]);}
            printf("%5.1f%%",100.0*cand/N);
            for(auto&bc:bases){ for(int qm=0;qm<2;qm++){ bool quant=(qm==1); double bR=-1,bL=0;
                for(int ef:EFS){ double r=0,l=0;int den=0;std::vector<float> dd(K);std::vector<idx_t> ii(K);
                    for(int i=0;i<NQ;i++){auto t0=clk::now();int rr=rsearch(bc.map,&Q[(size_t)i*d],&QC[(size_t)i*NW],K,ef,M[i],bc.base,true,true,aefT,Mbase,3,quant,K*OS,dd,ii);auto t1=clk::now();
                        l+=std::chrono::duration<double,std::micro>(t1-t0).count();int h=0;for(int j=0;j<rr;j++)if(tr[i].count(ii[j]))h++;if(!tr[i].empty()){r+=double(h)/tr[i].size();den++;}}
                    double R=den?r/den:0;if(R>bR+1e-9){bR=R;bL=l/NQ;} }
                printf(" | %.2f/%5.0f",bR,bL); } }
            printf("\n");
        }
    }
    fflush(stdout); std::_Exit(0);
}
