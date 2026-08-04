/*
 * Controlled dimension-scaling proof: does ACORN-1 overtake standard in-filtering when
 * distances get expensive? Tiling each SIFT vector T times scales every L2 distance by exactly
 * T, so the HNSW graph, the walk, recall, #distances and #inspections are ALL invariant to T --
 * the ONLY thing that changes is the wall-cost of one distance. So we:
 *   - build the graph ONCE on 128-D SIFT (topology is tile-invariant),
 *   - materialize genuinely tiled contiguous vectors at each D=128*T (realistic memory traffic),
 *   - report the tile-invariant work-counts (distances / inspections) once,
 *   - measure standard vs ACORN-1 latency at each dimension and show the crossover.
 * Scattered ("no") filter, the regime where both reach comparable recall (fair apples-to-apples).
 * Usage: bench_highdim [n]   (default 100000)
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
static int D=128; static const float* XG=nullptr;             // search-time vectors (tiled)
static uint64_t g_ndist=0, g_ninsp=0;                          // per-query work counters
static bool load_fvecs(const char*p,int want,std::vector<float>&o,int&d){FILE*f=fopen(p,"rb");if(!f)return false;int dd=0;if(fread(&dd,4,1,f)!=1){fclose(f);return false;}fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);long rec=4+4L*dd,tot=b/rec;int n=want>0?std::min<long>(want,tot):tot;o.resize((size_t)n*dd);for(int i=0;i<n;i++){int t;if(fread(&t,4,1,f)!=1||t!=dd){fclose(f);return false;}if(fread(&o[(size_t)i*dd],4,dd,f)!=(size_t)dd){fclose(f);return false;}}fclose(f);d=dd;return true;}
static inline float l2(const float*a,const float*b){float s=0;for(int j=0;j<D;j++){float df=a[j]-b[j];s+=df*df;}return s;}
static inline float l2v(const float*q,sidx v){g_ndist++;return l2(q,&XG[(size_t)v*D]);}
static void finalize(std::vector<float>&Dh,std::vector<idx_t>&Ih,int nres,faiss::IndexIDMap*map,int k,std::vector<idx_t>&oi){
    std::vector<std::pair<float,idx_t>> t;for(int i=0;i<nres;i++)t.push_back({Dh[i],Ih[i]});std::sort(t.begin(),t.end());
    for(size_t i=0;i<t.size();i++)oi[i]=map->id_map[t[i].second];for(int i=(int)t.size();i<k;i++)oi[i]=-1;
}
static void descend(const faiss::HNSW&h,const float*q,sidx&nr,float&dn){dn=l2v(q,nr);for(int lvl=h.max_level;lvl>=1;lvl--){for(;;){auto pv=nr;size_t b,e;h.neighbor_range(nr,lvl,&b,&e);for(size_t i=b;i<e;i++){auto v=h.neighbors[i];if(v<0)break;float d=l2v(q,v);if(d<dn){dn=d;nr=v;}}if(nr==pv)break;}}}
// standard in-filter (fixed entry): traverse ALL neighbours, distance all, results = passing
static int m_std(faiss::IndexIDMap*map,const float*q,int k,int ef,const std::vector<char>&m,std::vector<idx_t>&oi){
    auto*ih=(faiss::IndexHNSW*)map->index;const faiss::HNSW&h=ih->hnsw;const int N=ih->ntotal;sidx nr=h.entry_point;float dn;descend(h,q,nr,dn);
    const int efn=std::max(ef,k);faiss::HNSW::MinimaxHeap cand(efn);std::vector<float> Dh(k);std::vector<idx_t> Ih(k);int nres=0;faiss::VisitedTable vis(N);
    auto add=[&](sidx v){vis.set(v);float d=l2v(q,v);if(m[v]){if(nres<k)faiss::maxheap_push(++nres,Dh.data(),Ih.data(),d,v);else if(d<Dh[0])faiss::maxheap_replace_top(nres,Dh.data(),Ih.data(),d,v);}cand.push(v,d);};
    add(nr);while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;size_t b,e;h.neighbor_range(v0,0,&b,&e);for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;g_ninsp++;if(!vis.get(v1))add(v1);}}
    finalize(Dh,Ih,nres,map,k,oi);return nres;
}
// ACORN-1: frontier = passing-only; failing neighbour used as a 2-hop bridge (peek its neighbours)
static int m_acorn1(faiss::IndexIDMap*map,const float*q,int k,int ef,const std::vector<char>&m,std::vector<idx_t>&oi){
    auto*ih=(faiss::IndexHNSW*)map->index;const faiss::HNSW&h=ih->hnsw;const int N=ih->ntotal;
    const int efn=std::max(ef,k);faiss::HNSW::MinimaxHeap cand(efn);std::vector<float> Dh(k);std::vector<idx_t> Ih(k);int nres=0;faiss::VisitedTable vis(N),brd(N);
    auto admit=[&](sidx v){vis.set(v);float d=l2v(q,v);if(nres<k)faiss::maxheap_push(++nres,Dh.data(),Ih.data(),d,v);else if(d<Dh[0])faiss::maxheap_replace_top(nres,Dh.data(),Ih.data(),d,v);cand.push(v,d);};
    sidx nr=h.entry_point;float dn;descend(h,q,nr,dn);if(m[nr])admit(nr);else cand.push(nr,dn);
    while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;size_t b,e;h.neighbor_range(v0,0,&b,&e);
        for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;g_ninsp++;
            if(m[v1]){if(!vis.get(v1))admit(v1);}
            else if(!brd.get(v1)){brd.set(v1);size_t b2,e2;h.neighbor_range(v1,0,&b2,&e2);for(size_t j2=b2;j2<e2;j2++){auto v2=h.neighbors[j2];if(v2<0)break;g_ninsp++;if(m[v2]&&!vis.get(v2))admit(v2);}}
        }}
    finalize(Dh,Ih,nres,map,k,oi);return nres;
}
// ACORN-gamma: dense graph (M=16*gamma), plain 1-hop passing-only frontier (no bridges needed)
static int m_gamma(faiss::IndexIDMap*map,const float*q,int k,int ef,const std::vector<char>&m,std::vector<idx_t>&oi){
    auto*ih=(faiss::IndexHNSW*)map->index;const faiss::HNSW&h=ih->hnsw;const int N=ih->ntotal;
    const int efn=std::max(ef,k);faiss::HNSW::MinimaxHeap cand(efn);std::vector<float> Dh(k);std::vector<idx_t> Ih(k);int nres=0;faiss::VisitedTable vis(N);
    auto admit=[&](sidx v){vis.set(v);float d=l2v(q,v);if(nres<k)faiss::maxheap_push(++nres,Dh.data(),Ih.data(),d,v);else if(d<Dh[0])faiss::maxheap_replace_top(nres,Dh.data(),Ih.data(),d,v);cand.push(v,d);};
    sidx nr=h.entry_point;float dn;descend(h,q,nr,dn);if(m[nr])admit(nr);else cand.push(nr,dn);
    while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;size_t b,e;h.neighbor_range(v0,0,&b,&e);
        for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;g_ninsp++;if(m[v1]&&!vis.get(v1))admit(v1);}}
    finalize(Dh,Ih,nres,map,k,oi);return nres;
}
static uint64_t edge_bytes(faiss::IndexIDMap*map){auto*ih=(faiss::IndexHNSW*)map->index;return (uint64_t)ih->hnsw.neighbors.size()*sizeof(int);}
int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):100000;int G=argc>2?atoi(argv[2]):12;const int K=10,NQ=50,ef=200;
    std::vector<float> X;int d=128;if(!load_fvecs("research/acorn/data/sift/sift_base.fvecs",N,X,d)){fprintf(stderr,"need sift\n");return 1;}N=(int)(X.size()/d);
    std::vector<float> Q;int qd=0;load_fvecs("research/acorn/data/sift/sift_query.fvecs",NQ,Q,qd);
    std::vector<idx_t> ids(N);for(int i=0;i<N;i++)ids[i]=i;
    // build graphs ONCE on 128-D (topology is tile-invariant); IDMap identity here
    D=128;XG=X.data();
    using clk=std::chrono::high_resolution_clock;
    auto*hh=new faiss::IndexHNSWFlat(d,16,faiss::METRIC_L2);hh->hnsw.efConstruction=100;auto*map=new faiss::IndexIDMap(hh);map->own_fields=true;
    auto b0=clk::now();map->add_with_ids(N,X.data(),ids.data());auto b1=clk::now();double bt_std=std::chrono::duration<double>(b1-b0).count();
    auto*hg=new faiss::IndexHNSWFlat(d,16*G,faiss::METRIC_L2);hg->hnsw.efConstruction=100;auto*dmap=new faiss::IndexIDMap(hg);dmap->own_fields=true;
    auto g0=clk::now();dmap->add_with_ids(N,X.data(),ids.data());auto g1=clk::now();double bt_g=std::chrono::duration<double>(g1-g0).count();
    fprintf(stderr,"[bench_highdim] n=%d graphs built (standard M=16, gamma M=%d)\n",N,16*G);
    // ---- build-time + index-size (edges) cost: the price of gamma's density -----------------
    uint64_t eb_std=edge_bytes(map),eb_g=edge_bytes(dmap);
    printf("\n==== ACORN-gamma build/size cost (gamma=%d -> M=%d) ====\n",G,16*G);
    printf("            | build time | edge storage (proxy for index size)\n");
    printf("standard M16| %8.2fs  | %6.1f MB\n",bt_std,eb_std/1e6);
    printf("gamma  M%-4d| %8.2fs  | %6.1f MB   (%.1fx build, %.1fx edges)\n",16*G,bt_g,eb_g/1e6,bt_g/bt_std,(double)eb_g/eb_std);
    // scattered ("no") ground truth + masks are computed on 128-D (tile-invariant ordering)
    double sels[3]={0.01,0.05,0.25};int Ts[4]={1,8,16,32};
    std::vector<std::vector<char>> M[3];std::vector<std::unordered_set<int>> TR[3];
    for(int si=0;si<3;si++){int cand=std::max(K,(int)(sels[si]*N));M[si].resize(NQ);TR[si].resize(NQ);
        for(int i=0;i<NQ;i++){std::vector<char> s(N,0);faiss::RandomGenerator r(1000+i);int p=0;while(p<cand){int id=r.rand_int(N);if(!s[id]){s[id]=1;p++;}}M[si][i]=s;
            const float*q=&Q[(size_t)i*d];std::vector<std::pair<float,int>> sc;for(int b=0;b<N;b++)if(s[b])sc.push_back({l2(q,&X[(size_t)b*d]),b});std::sort(sc.begin(),sc.end());std::unordered_set<int> t;for(int j=0;j<K&&j<(int)sc.size();j++)t.insert(sc[j].second);TR[si][i]=t;}}
    auto rc=[&](std::vector<idx_t>&oi,int r,std::unordered_set<int>&tr){if(tr.empty())return 1.0;int h=0;for(int j=0;j<r;j++)if(tr.count((int)oi[j]))h++;return double(h)/tr.size();};
    (void)ef;
    // Each method is compared at ITS OWN best-recall ef (gamma's edge is reaching recall at
    // lower ef -- a fixed ef hides that). Recall + work-counts are dimension-invariant (tiling
    // preserves distance order), so we pick ef* here once and reuse it at every dimension.
    int EFS[4]={40,80,160,320};
    auto pick=[&](int which,int si,int&efStar,double&R,uint64_t&DI,uint64_t&IN){
        double bR=-1;efStar=EFS[0];std::vector<idx_t> oi(K);
        for(int e:EFS){double r=0;uint64_t di=0,in=0;
            for(int i=0;i<NQ;i++){g_ndist=g_ninsp=0;int rr;
                if(which==0)rr=m_std(map,&Q[(size_t)i*d],K,e,M[si][i],oi);
                else if(which==1)rr=m_acorn1(map,&Q[(size_t)i*d],K,e,M[si][i],oi);
                else rr=m_gamma(dmap,&Q[(size_t)i*d],K,e,M[si][i],oi);
                r+=rc(oi,rr,TR[si][i]);di+=g_ndist;in+=g_ninsp;}
            r/=NQ;if(r>bR+1e-9){bR=r;efStar=e;DI=di/NQ;IN=in/NQ;}}
        R=bR;
    };
    int efS[3][3];  // [si][method]
    // ---- Part 1: per-method best-recall work-counts (dimension-invariant) ------------------
    printf("\n==== per-query work at each method's best-recall ef (dimension-invariant) ====\n");
    printf("%6s | %-26s | %-26s | %-26s\n","sel","standard rec/dist/insp/ef","ACORN-1  rec/dist/insp/ef","ACORN-g  rec/dist/insp/ef");
    for(int si=0;si<3;si++){
        double R[3];uint64_t DI[3],IN[3];int e3[3];
        for(int mth=0;mth<3;mth++){pick(mth,si,e3[mth],R[mth],DI[mth],IN[mth]);efS[si][mth]=e3[mth];}
        printf("%5.0f%% | %.2f /%5llu /%6llu /%3d | %.2f /%5llu /%6llu /%3d | %.2f /%5llu /%6llu /%3d\n",100.0*sels[si],
            R[0],(unsigned long long)DI[0],(unsigned long long)IN[0],e3[0],
            R[1],(unsigned long long)DI[1],(unsigned long long)IN[1],e3[1],
            R[2],(unsigned long long)DI[2],(unsigned long long)IN[2],e3[2]);
    }

    // ---- Part 2: latency vs dimension, each method at its best-recall ef --------------------
    printf("\n==== latency us as distances get expensive (each at best-recall ef); xN vs standard ====\n");
    for(int si=0;si<3;si++){
        printf("\n-- scattered %.0f%% selectivity (ef*: std=%d acorn1=%d gamma=%d) --\n",100.0*sels[si],efS[si][0],efS[si][1],efS[si][2]);
        printf("%6s | %-13s | %-20s | %-20s\n","dim","standard us","ACORN-1 us (x)","ACORN-g us (x)");
        for(int ti=0;ti<4;ti++){int T=Ts[ti];int Deff=128*T;
            std::vector<float> Xt((size_t)N*Deff),Qt((size_t)NQ*Deff);          // materialize tiled vectors
            for(int i=0;i<N;i++)for(int t=0;t<T;t++)std::copy(&X[(size_t)i*d],&X[(size_t)i*d]+d,&Xt[(size_t)i*Deff+(size_t)t*d]);
            for(int i=0;i<NQ;i++)for(int t=0;t<T;t++)std::copy(&Q[(size_t)i*d],&Q[(size_t)i*d]+d,&Qt[(size_t)i*Deff+(size_t)t*d]);
            D=Deff;XG=Xt.data();
            std::vector<idx_t> oi(K);
            auto run=[&](int which,int e){double best=1e18;for(int rep=0;rep<3;rep++){auto t0=clk::now();for(int i=0;i<NQ;i++){if(which==0)m_std(map,&Qt[(size_t)i*Deff],K,e,M[si][i],oi);else if(which==1)m_acorn1(map,&Qt[(size_t)i*Deff],K,e,M[si][i],oi);else m_gamma(dmap,&Qt[(size_t)i*Deff],K,e,M[si][i],oi);}auto t1=clk::now();best=std::min(best,std::chrono::duration<double,std::micro>(t1-t0).count()/NQ);}return best;};
            double ls=run(0,efS[si][0]),la=run(1,efS[si][1]),lg=run(2,efS[si][2]);
            printf("%5dD | %-13.0f | %-8.0f %.2fx%-7s | %-8.0f %.2fx%s\n",Deff,ls,la,ls/la,ls/la>=1?"(win)":"",lg,ls/lg,ls/lg>=1?"(win)":"");
        }
    }
    fflush(stdout); std::_Exit(0);
}
