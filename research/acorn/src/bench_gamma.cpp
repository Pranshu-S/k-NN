/*
 * 3-way proof: standard faiss filtered search vs corrected ACORN-1 (JNI, search-time
 * 2-hop) vs ACORN-γ (faiss's own SIMD search on the denser ACORN-γ graph). The γ graph
 * is built once via build_acorn_gamma and its HNSW is copied into a faiss IndexHNSWFlat,
 * so the γ column uses the SAME optimized faiss traversal as standard — the only
 * variable is the graph. Real SIFT, per-query-local correlation.
 * Usage: bench_gamma [n] [gamma]   (default 100000, 12)
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
#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
#include "acorn_hnsw.h"   // corrected ACORN-1
#include "acorn.h"        // build_acorn_gamma (research)

using faiss::idx_t;
struct SetSel : faiss::IDSelector { std::vector<char> bit; bool is_member(idx_t id) const override { return id>=0&&(size_t)id<bit.size()&&bit[id]; } };

// Proper ACORN-gamma search: FILTERED 1-hop best-first over the DENSE graph.
// Frontier = passing nodes only (predicate subgraph); distances computed only on
// passing nodes (SIMD-batched). No 2-hop — the dense graph makes 1-hop suffice.
static int gamma_search(faiss::IndexIDMap* map,const float*q,int k,int ef,const SetSel& sel,
                        std::vector<float>&outd,std::vector<idx_t>&outi,uint64_t*ndis_out){
    auto* ih=dynamic_cast<faiss::IndexHNSW*>(map->index); const faiss::HNSW& hnsw=ih->hnsw;
    const int N=ih->ntotal; std::unique_ptr<faiss::DistanceComputer> dc(ih->storage->get_distance_computer()); dc->set_query(q);
    auto ext=[&](faiss::HNSW::storage_idx_t v){ return map->id_map[v]; };
    uint64_t ndis=0;
    faiss::HNSW::storage_idx_t nearest=hnsw.entry_point; float dn=(*dc)(nearest); ndis++;
    for(int lvl=hnsw.max_level;lvl>=1;lvl--){ for(;;){ auto prev=nearest; size_t b,e; hnsw.neighbor_range(nearest,lvl,&b,&e);
        for(size_t i=b;i<e;i++){auto v=hnsw.neighbors[i];if(v<0)break;float d=(*dc)(v);ndis++;if(d<dn){dn=d;nearest=v;}} if(nearest==prev)break; } }
    const int efn=std::max(ef,k); faiss::HNSW::MinimaxHeap cand(efn);
    std::vector<float> D(k); std::vector<idx_t> I(k); int nres=0; faiss::VisitedTable admitted(N);
    faiss::HNSW::storage_idx_t bq[4]; int bn=0;
    auto flush=[&](){ if(!bn)return; float bd[4]={0,0,0,0}; if(bn==4)dc->distances_batch_4(bq[0],bq[1],bq[2],bq[3],bd[0],bd[1],bd[2],bd[3]); else for(int t=0;t<bn;t++)bd[t]=(*dc)(bq[t]); ndis+=bn;
        for(int t=0;t<bn;t++){ if(nres<k)faiss::maxheap_push(++nres,D.data(),I.data(),bd[t],bq[t]); else if(bd[t]<D[0])faiss::maxheap_replace_top(nres,D.data(),I.data(),bd[t],bq[t]); cand.push(bq[t],bd[t]); } bn=0; };
    if(sel.is_member(ext(nearest))){ faiss::maxheap_push(++nres,D.data(),I.data(),dn,nearest); admitted.set(nearest); }
    cand.push(nearest,dn);
    while(cand.size()>0){ float d0; auto v0=cand.pop_min(&d0); if(cand.count_below(d0)>=efn)break;
        size_t b,e; hnsw.neighbor_range(v0,0,&b,&e);
        for(size_t j=b;j<e;j++){ auto v1=hnsw.neighbors[j]; if(v1<0)break; if(sel.is_member(ext(v1))&&!admitted.get(v1)){ admitted.set(v1); bq[bn++]=v1; if(bn==4)flush(); } }
        flush();
    }
    std::vector<std::pair<float,idx_t>> tmp; for(int i=0;i<nres;i++)tmp.push_back({D[i],I[i]});
    std::sort(tmp.begin(),tmp.end());
    for(size_t i=0;i<tmp.size();i++){outd[i]=tmp[i].first;outi[i]=ext(tmp[i].second);} for(int i=(int)tmp.size();i<k;i++)outi[i]=-1;
    if(ndis_out)*ndis_out=ndis; return nres;
}

static bool load_fvecs(const char* path,int want,std::vector<float>&out,int&d){
    FILE*fp=fopen(path,"rb"); if(!fp)return false; int dd=0; if(fread(&dd,4,1,fp)!=1){fclose(fp);return false;}
    fseek(fp,0,SEEK_END); long b=ftell(fp); fseek(fp,0,SEEK_SET); long rec=4+4L*dd,tot=b/rec; int n=want>0?std::min<long>(want,tot):tot;
    out.resize((size_t)n*dd); for(int i=0;i<n;i++){int t;if(fread(&t,4,1,fp)!=1||t!=dd){fclose(fp);return false;} if(fread(&out[(size_t)i*dd],4,dd,fp)!=(size_t)dd){fclose(fp);return false;}}
    fclose(fp); d=dd; return true;
}

// wrap a research AcornIndex's HNSW graph in a faiss IndexIDMap(IndexHNSWFlat)
static faiss::IndexIDMap* wrap(const faiss::HNSW& g, const std::vector<float>& X, int N, int d){
    auto* hf=new faiss::IndexHNSWFlat(d,16,faiss::METRIC_L2);
    auto* map=new faiss::IndexIDMap(hf); map->own_fields=true;
    hf->storage->add(N,X.data()); hf->hnsw=g; hf->ntotal=N;
    map->id_map.resize(N); for(int i=0;i<N;i++) map->id_map[i]=i; map->ntotal=N;
    return map;
}

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):100000; int G=argc>2?atoi(argv[2]):12; const int K=10,NQ=50;
    std::vector<float> X; int d=128;
    if(!load_fvecs("research/acorn/data/sift/sift_base.fvecs",N,X,d)){fprintf(stderr,"need sift_base.fvecs\n");return 1;}
    N=(int)(X.size()/d);
    std::vector<float> Q; int qd=0; load_fvecs("research/acorn/data/sift/sift_query.fvecs",NQ,Q,qd);

    std::vector<idx_t> ids0(N); for(int i=0;i<N;i++) ids0[i]=i;
    // standard graph: HNSW M=16 (level-0 degree ~32)
    auto* hs=new faiss::IndexHNSWFlat(d,16,faiss::METRIC_L2); hs->hnsw.efConstruction=100;
    faiss::IndexIDMap* smap=new faiss::IndexIDMap(hs); smap->own_fields=true;
    smap->add_with_ids(N,X.data(),ids0.data());
    // ACORN-gamma graph: a genuinely denser HNSW, M = gamma*16 (level-0 degree ~2*gamma*16).
    // This is the paper's idea — pay the denser neighbourhood ONCE at build time so search
    // does 1-hop (not 2-hop) over the predicate subgraph.
    auto* hg=new faiss::IndexHNSWFlat(d,16*G,faiss::METRIC_L2); hg->hnsw.efConstruction=100;
    faiss::IndexIDMap* gmap=new faiss::IndexIDMap(hg); gmap->own_fields=true;
    gmap->add_with_ids(N,X.data(),ids0.data());
    fprintf(stderr,"[bench_gamma] n=%d gamma=%d (dense M=%d) graphs built\n",N,G,16*G);

    std::vector<std::vector<int>> ord(NQ);
    for(int i=0;i<NQ;i++){ std::vector<std::pair<float,int>> ds(N); const float*q=&Q[(size_t)i*d];
        for(int b=0;b<N;b++){float s=0;for(int j=0;j<d;j++){float df=q[j]-X[(size_t)b*d+j];s+=df*df;}ds[b]={s,b};}
        std::sort(ds.begin(),ds.end()); ord[i].resize(N); for(int b=0;b<N;b++)ord[i][b]=ds[b].second; }
    auto mask=[&](const char*c,int cand,int qi){ SetSel s; s.bit.assign(N,0);
        if(std::string(c)=="no"){faiss::RandomGenerator r(1000+qi);int p=0;while(p<cand){int id=r.rand_int(N);if(!s.bit[id]){s.bit[id]=1;p++;}}}
        else if(std::string(c)=="pos"){for(int j=0;j<cand;j++)s.bit[ord[qi][j]]=1;}
        else{for(int j=0;j<cand;j++)s.bit[ord[qi][N-1-j]]=1;} return s; };
    auto brute=[&](const float*q,const SetSel&s){ std::vector<std::pair<float,idx_t>> sc;
        for(int i=0;i<N;i++){if(!s.bit[i])continue;float d2=0;for(int j=0;j<d;j++){float df=q[j]-X[(size_t)i*d+j];d2+=df*df;}sc.push_back({d2,i});}
        std::sort(sc.begin(),sc.end()); std::unordered_set<idx_t> t; for(int i=0;i<K&&i<(int)sc.size();i++)t.insert(sc[i].second); return t; };
    using clk=std::chrono::high_resolution_clock;
    auto faiss_run=[&](faiss::IndexIDMap* m,int ef,std::vector<SetSel>&M,std::vector<std::unordered_set<idx_t>>&tr,double&rec,double&lat,double&nd){
        std::vector<float> dd(K); std::vector<idx_t> ii(K); rec=0;lat=0;nd=0;int den=0;
        for(int i=0;i<NQ;i++){faiss::SearchParametersHNSW p;p.efSearch=ef;p.sel=&M[i];faiss::hnsw_stats.reset();
            auto t0=clk::now();m->search(1,&Q[(size_t)i*d],K,dd.data(),ii.data(),&p);auto t1=clk::now();
            lat+=std::chrono::duration<double,std::micro>(t1-t0).count();nd+=faiss::hnsw_stats.ndis;
            int h=0;for(int j=0;j<K;j++)if(ii[j]>=0&&tr[i].count(ii[j]))h++; if(!tr[i].empty()){rec+=double(h)/tr[i].size();den++;}}
        rec=den?rec/den:0;lat/=NQ;nd/=NQ; };

    const char* corrs[3]={"neg","no","pos"}; int EFS[5]={25,50,100,250,500};
    double sels[6]={0.001,0.01,0.05,0.10,0.25,0.50};
    printf("\n#### n=%d  ACORN-gamma=%d  (best recall over ef; rec/ndis/latency)\n",N,G);
    for(const char* c:corrs){
        printf("\n=== correlation=%s ===\n",c);
        printf("%6s | %-22s | %-22s | %-22s\n","sel","standard","ACORN-1 (2-hop)","ACORN-gamma (dense)");
        for(double se:sels){ int cand=std::max(K,(int)(se*N));
            std::vector<SetSel> M(NQ); std::vector<std::unordered_set<idx_t>> tr(NQ);
            for(int i=0;i<NQ;i++){M[i]=mask(c,cand,i);tr[i]=brute(&Q[(size_t)i*d],M[i]);}
            double bS=-1,bSl=0,bSn=0,bA=-1,bAl=0,bAn=0,bG=-1,bGl=0,bGn=0;
            std::vector<float> dd(K);std::vector<idx_t> ii(K);
            for(int ef:EFS){
                double r,l,n; faiss_run(smap,ef,M,tr,r,l,n); if(r>bS+1e-9){bS=r;bSl=l;bSn=n;}
                // ACORN-gamma: filtered 1-hop on the dense graph (proper traversal)
                { double gr=0,gl=0,gn=0;int gden=0; std::vector<float> dd(K); std::vector<idx_t> ii(K);
                  for(int i=0;i<NQ;i++){uint64_t gd=0; auto t0=clk::now(); int rr=gamma_search(gmap,&Q[(size_t)i*d],K,ef,M[i],dd,ii,&gd); auto t1=clk::now();
                    gl+=std::chrono::duration<double,std::micro>(t1-t0).count(); gn+=gd;
                    int h=0;for(int j=0;j<rr;j++)if(tr[i].count(ii[j]))h++; if(!tr[i].empty()){gr+=double(h)/tr[i].size();gden++;}}
                  gr=gden?gr/gden:0;gl/=NQ;gn/=NQ; if(gr>bG+1e-9){bG=gr;bGl=gl;bGn=gn;} }
                double ar=0,al=0,an=0;int den=0;
                for(int i=0;i<NQ;i++){auto t0=clk::now();int rr=knn_jni::acorn::search(smap,&Q[(size_t)i*d],K,ef,&M[i],1,dd.data(),ii.data(),true);auto t1=clk::now();
                    al+=std::chrono::duration<double,std::micro>(t1-t0).count();an+=knn_jni::acorn::last_stats().distance_computations;
                    int h=0;for(int j=0;j<rr;j++)if(tr[i].count(ii[j]))h++; if(!tr[i].empty()){ar+=double(h)/tr[i].size();den++;}}
                ar=den?ar/den:0;al/=NQ;an/=NQ; if(ar>bA+1e-9){bA=ar;bAl=al;bAn=an;}
            }
            printf("%5.1f%% | %.2f / %5.0f / %5.0fus | %.2f / %5.0f / %5.0fus | %.2f / %5.0f / %5.0fus\n",
                   100.0*cand/N, bS,bSn,bSl, bA,bAn,bAl, bG,bGn,bGl);
        }
    }
    fflush(stdout); std::_Exit(0);
}
