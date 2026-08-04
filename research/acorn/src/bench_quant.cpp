/*
 * How much does quantization move the ACORN-1 crossover at 1536-D (a typical production dim)?
 * The graph, walk, and work-counts are identical regardless of representation -- ONLY the
 * cost of one distance changes. So we build the graph once (on 128-D; topology is tile-
 * invariant), materialize each vector at 1536-D in three representations, and time
 * standard vs ACORN-1 under each distance kernel:
 *   fp32  : 1536 float subtract-square-add  (full precision)
 *   int8  : 1536 int8 scalar-quantized      (~scalar quantization / 8-bit)
 *   1bit  : 1536 bits, Hamming popcount      (~BBQ / binary; cost only -- recall not modeled)
 * Reports ns/distance per kernel + std-vs-ACORN-1 latency & speedup under each.
 * Scattered filter. SIFT tiled 128->1536. Usage: bench_quant [n]  (default 60000)
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
#include <faiss/impl/HNSW.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
using idx_t=faiss::idx_t; using sidx=faiss::HNSW::storage_idx_t;
enum Kernel { FP32, INT8, BIT1 };
static int D=1536; static Kernel KERN=FP32;
static const float* XF=nullptr;   const float* QFcur=nullptr;   // fp32 vectors / query
static const int8_t* X8=nullptr;  const int8_t* Q8cur=nullptr;  // int8 codes
static const uint64_t* XB=nullptr; const uint64_t* QBcur=nullptr; static int BW=0; // bit codes, words/vec
static uint64_t g_nd=0, g_ni=0;
static bool load_fvecs(const char*p,int want,std::vector<float>&o,int&d){FILE*f=fopen(p,"rb");if(!f)return false;int dd=0;if(fread(&dd,4,1,f)!=1){fclose(f);return false;}fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);long rec=4+4L*dd,tot=b/rec;int n=want>0?std::min<long>(want,tot):tot;o.resize((size_t)n*dd);for(int i=0;i<n;i++){int t;if(fread(&t,4,1,f)!=1||t!=dd){fclose(f);return false;}if(fread(&o[(size_t)i*dd],4,dd,f)!=(size_t)dd){fclose(f);return false;}}fclose(f);d=dd;return true;}
static inline float dfp32(const float*q,const float*x){float s=0;for(int j=0;j<D;j++){float df=q[j]-x[j];s+=df*df;}return s;}
static inline float dint8(const int8_t*q,const int8_t*x){int s=0;for(int j=0;j<D;j++){int df=(int)q[j]-(int)x[j];s+=df*df;}return (float)s;}
static inline float dbit(const uint64_t*q,const uint64_t*x){int s=0;for(int w=0;w<BW;w++)s+=__builtin_popcountll(q[w]^x[w]);return (float)s;}
// query-vs-stored distance, dispatched on the active kernel (query pointers set per query)
static inline float dv(sidx v){g_nd++;
    if(KERN==FP32)return dfp32(QFcur,&XF[(size_t)v*D]);
    if(KERN==INT8)return dint8(Q8cur,&X8[(size_t)v*D]);
    return dbit(QBcur,&XB[(size_t)v*BW]);}
static void finalize(std::vector<float>&Dh,std::vector<idx_t>&Ih,int nres,faiss::IndexIDMap*map,int k,std::vector<idx_t>&oi){
    std::vector<std::pair<float,idx_t>> t;for(int i=0;i<nres;i++)t.push_back({Dh[i],Ih[i]});std::sort(t.begin(),t.end());
    for(size_t i=0;i<t.size();i++)oi[i]=map->id_map[t[i].second];for(int i=(int)t.size();i<k;i++)oi[i]=-1;}
static void descend(const faiss::HNSW&h,sidx&nr,float&dn){dn=dv(nr);for(int lvl=h.max_level;lvl>=1;lvl--){for(;;){auto pv=nr;size_t b,e;h.neighbor_range(nr,lvl,&b,&e);for(size_t i=b;i<e;i++){auto v=h.neighbors[i];if(v<0)break;float d=dv(v);if(d<dn){dn=d;nr=v;}}if(nr==pv)break;}}}
static int m_std(faiss::IndexIDMap*map,int k,int ef,const std::vector<char>&m,std::vector<idx_t>&oi){
    auto*ih=(faiss::IndexHNSW*)map->index;const faiss::HNSW&h=ih->hnsw;const int N=ih->ntotal;sidx nr=h.entry_point;float dn;descend(h,nr,dn);
    const int efn=std::max(ef,k);faiss::HNSW::MinimaxHeap cand(efn);std::vector<float> Dh(k);std::vector<idx_t> Ih(k);int nres=0;faiss::VisitedTable vis(N);
    auto add=[&](sidx v){vis.set(v);float d=dv(v);if(m[v]){if(nres<k)faiss::maxheap_push(++nres,Dh.data(),Ih.data(),d,v);else if(d<Dh[0])faiss::maxheap_replace_top(nres,Dh.data(),Ih.data(),d,v);}cand.push(v,d);};
    add(nr);while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;size_t b,e;h.neighbor_range(v0,0,&b,&e);for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;g_ni++;if(!vis.get(v1))add(v1);}}
    finalize(Dh,Ih,nres,map,k,oi);return nres;}
static int m_acorn1(faiss::IndexIDMap*map,int k,int ef,const std::vector<char>&m,std::vector<idx_t>&oi){
    auto*ih=(faiss::IndexHNSW*)map->index;const faiss::HNSW&h=ih->hnsw;const int N=ih->ntotal;
    const int efn=std::max(ef,k);faiss::HNSW::MinimaxHeap cand(efn);std::vector<float> Dh(k);std::vector<idx_t> Ih(k);int nres=0;faiss::VisitedTable vis(N),brd(N);
    auto admit=[&](sidx v){vis.set(v);float d=dv(v);if(nres<k)faiss::maxheap_push(++nres,Dh.data(),Ih.data(),d,v);else if(d<Dh[0])faiss::maxheap_replace_top(nres,Dh.data(),Ih.data(),d,v);cand.push(v,d);};
    sidx nr=h.entry_point;float dn;descend(h,nr,dn);if(m[nr])admit(nr);else cand.push(nr,dn);
    while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;size_t b,e;h.neighbor_range(v0,0,&b,&e);
        for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;g_ni++;
            if(m[v1]){if(!vis.get(v1))admit(v1);}
            else if(!brd.get(v1)){brd.set(v1);size_t b2,e2;h.neighbor_range(v1,0,&b2,&e2);for(size_t j2=b2;j2<e2;j2++){auto v2=h.neighbors[j2];if(v2<0)break;g_ni++;if(m[v2]&&!vis.get(v2))admit(v2);}}
        }}
    finalize(Dh,Ih,nres,map,k,oi);return nres;}
int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):60000;const int K=10,NQ=50,TILE=12; D=128*TILE; // 1536
    std::vector<float> X0;int d=128;if(!load_fvecs("research/acorn/data/sift/sift_base.fvecs",N,X0,d)){fprintf(stderr,"need sift\n");return 1;}N=(int)(X0.size()/d);
    std::vector<float> Q0;int qd=0;load_fvecs("research/acorn/data/sift/sift_query.fvecs",NQ,Q0,qd);
    // build graph on 128-D (tile-invariant topology), IDMap identity
    D=128;auto*hh=new faiss::IndexHNSWFlat(128,16,faiss::METRIC_L2);hh->hnsw.efConstruction=100;auto*map=new faiss::IndexIDMap(hh);map->own_fields=true;
    std::vector<idx_t> ids(N);for(int i=0;i<N;i++)ids[i]=i;map->add_with_ids(N,X0.data(),ids.data());D=128*TILE;
    fprintf(stderr,"[bench_quant] n=%d D=%d graph built\n",N,D);
    // materialize 1536-D fp32 (tiled), int8 (scale by global max-abs), 1-bit (per-dim mean threshold)
    std::vector<float> XF_(( size_t)N*D),QF_((size_t)NQ*D);
    for(int i=0;i<N;i++)for(int t=0;t<TILE;t++)std::copy(&X0[(size_t)i*128],&X0[(size_t)i*128]+128,&XF_[(size_t)i*D+(size_t)t*128]);
    for(int i=0;i<NQ;i++)for(int t=0;t<TILE;t++)std::copy(&Q0[(size_t)i*128],&Q0[(size_t)i*128]+128,&QF_[(size_t)i*D+(size_t)t*128]);
    XF=XF_.data();
    float mx=0;for(size_t i=0;i<XF_.size();i++)mx=std::max(mx,std::fabs(XF_[i])); float sc=127.0f/(mx>0?mx:1);
    std::vector<int8_t> X8_((size_t)N*D),Q8_((size_t)NQ*D);
    for(size_t i=0;i<XF_.size();i++)X8_[i]=(int8_t)std::lround(XF_[i]*sc);
    for(size_t i=0;i<QF_.size();i++)Q8_[i]=(int8_t)std::lround(QF_[i]*sc);
    X8=X8_.data();
    std::vector<float> mean(D,0);for(int i=0;i<N;i++)for(int j=0;j<D;j++)mean[j]+=XF_[(size_t)i*D+j];for(int j=0;j<D;j++)mean[j]/=N;
    BW=(D+63)/64; std::vector<uint64_t> XB_((size_t)N*BW,0),QB_((size_t)NQ*BW,0);
    for(int i=0;i<N;i++)for(int j=0;j<D;j++)if(XF_[(size_t)i*D+j]>mean[j])XB_[(size_t)i*BW+j/64]|=(1ULL<<(j&63));
    for(int i=0;i<NQ;i++)for(int j=0;j<D;j++)if(QF_[(size_t)i*D+j]>mean[j])QB_[(size_t)i*BW+j/64]|=(1ULL<<(j&63));
    XB=XB_.data();
    using clk=std::chrono::high_resolution_clock;
    // ---- microbench: ns per distance for each kernel ----
    printf("\n==== cost of ONE 1536-D distance (ns), and vs 128-D fp32 SIFT baseline ====\n");
    { volatile float sink=0; const int R=2000000;
      auto t0=clk::now();for(int i=0;i<R;i++)sink+=dfp32(&QF_[(i%NQ)*D],&XF_[(size_t)(i%N)*D]);auto t1=clk::now();double nf=std::chrono::duration<double,std::nano>(t1-t0).count()/R;
      t0=clk::now();for(int i=0;i<R;i++)sink+=dint8(&Q8_[(i%NQ)*D],&X8_[(size_t)(i%N)*D]);t1=clk::now();double ni=std::chrono::duration<double,std::nano>(t1-t0).count()/R;
      t0=clk::now();for(int i=0;i<R;i++)sink+=dbit(&QB_[(i%NQ)*BW],&XB_[(size_t)(i%N)*BW]);t1=clk::now();double nb=std::chrono::duration<double,std::nano>(t1-t0).count()/R;
      // 128-D fp32 reference
      int Dsave=D;D=128;t0=clk::now();for(int i=0;i<R;i++)sink+=dfp32(&Q0[(i%NQ)*128],&X0[(size_t)(i%N)*128]);t1=clk::now();double n128=std::chrono::duration<double,std::nano>(t1-t0).count()/R;D=Dsave;
      printf("  128-D fp32 (SIFT)  : %5.1f ns  (1.0x reference)\n",n128);
      printf("  1536-D fp32        : %5.1f ns  (%.1fx a SIFT distance)\n",nf,nf/n128);
      printf("  1536-D int8        : %5.1f ns  (%.1fx SIFT; %.1fx cheaper than fp32-1536)\n",ni,ni/n128,nf/ni);
      printf("  1536-D 1-bit(Hamm) : %5.1f ns  (%.1fx SIFT; %.1fx cheaper than fp32-1536)\n",nb,nb/n128,nf/nb);
      (void)sink; }
    // masks (scattered)
    double sels[2]={0.05,0.25}; int EF=200;
    std::vector<std::vector<std::vector<char>>> M(2, std::vector<std::vector<char>>(NQ));
    for(int s=0;s<2;s++){int cand=std::max(K,(int)(sels[s]*N));for(int i=0;i<NQ;i++){std::vector<char> b(N,0);faiss::RandomGenerator r(1000+i);int p=0;while(p<cand){int id=r.rand_int(N);if(!b[id]){b[id]=1;p++;}}M[s][i]=b;}}
    const char* kn[3]={"fp32","int8","1bit"};
    printf("\n==== standard vs ACORN-1 latency at 1536-D under each kernel (scattered, ef=%d) ====\n",EF);
    printf("%6s | %-6s | %-14s | %-14s | %s\n","sel","kernel","standard us","ACORN-1 us","ACORN-1 speedup");
    for(int s=0;s<2;s++){
        for(int kk=0;kk<3;kk++){KERN=(Kernel)kk;
            std::vector<idx_t> oi(K);
            auto run=[&](int which){double best=1e18;for(int rep=0;rep<3;rep++){auto t0=clk::now();
                for(int i=0;i<NQ;i++){ if(KERN==FP32){QFcur=&QF_[(size_t)i*D];} else if(KERN==INT8){Q8cur=&Q8_[(size_t)i*D];} else {QBcur=&QB_[(size_t)i*BW];}
                    if(which==0)m_std(map,K,EF,M[s][i],oi);else m_acorn1(map,K,EF,M[s][i],oi);}
                auto t1=clk::now();best=std::min(best,std::chrono::duration<double,std::micro>(t1-t0).count()/NQ);}return best;};
            double ls=run(0),la=run(1);
            printf("%5.0f%% | %-6s | %-14.0f | %-14.0f | %.2fx %s\n",100.0*sels[s],kn[kk],ls,la,ls/la,ls/la>=1?"(ACORN wins)":"(standard wins)");
        }
    }
    fflush(stdout); std::_Exit(0);
}
