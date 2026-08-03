/*
 * RACORN = ASF + AEF applied to three base filtered traversals:
 *   STD    : faiss-style in-filtering (traverse all, distance all, results=passing) on M=16 graph
 *   ACORN1 : passing-only frontier + 2-hop bridge on M=16 graph
 *   GAMMA  : passing-only frontier + 1-hop on the dense (M=gamma*16) graph
 *
 * ASF (Adaptive Search Fallback): when a node's expansion yields < M_base passing
 *   neighbours, admit stride-sampled FAILING neighbours as transient frontier bridges
 *   (never results) to keep the walk alive (recovers low-sel / negative-corr collapse).
 * AEF (Adaptive Exact Fallback): if the running pass-ratio (passed/examined) drops below
 *   a threshold after enough exploration, abandon the graph walk and do EXACT search over
 *   the eligible set (cheap when selectivity is tiny; recall 1.0).
 *
 * Reports, per base: base | +ASF | +ASF+AEF (recall / latency), for negative & no
 * correlation (where ASF/AEF matter). Real SIFT, per-query-local correlation, bitset sel.
 * Usage: bench_racorn [n] [gamma] [aef_thr]   (default 100000 12 0.02)
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
using faiss::idx_t; using sidx=faiss::HNSW::storage_idx_t;
enum Base { STD, ACORN1, GAMMA };
struct SetSel : faiss::IDSelector { std::vector<char> bit; bool is_member(idx_t id) const override { return id>=0&&(size_t)id<bit.size()&&bit[id]; } };
static bool load_fvecs(const char*p,int want,std::vector<float>&o,int&d){FILE*f=fopen(p,"rb");if(!f)return false;int dd=0;if(fread(&dd,4,1,f)!=1){fclose(f);return false;}fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);long rec=4+4L*dd,tot=b/rec;int n=want>0?std::min<long>(want,tot):tot;o.resize((size_t)n*dd);for(int i=0;i<n;i++){int t;if(fread(&t,4,1,f)!=1||t!=dd){fclose(f);return false;}if(fread(&o[(size_t)i*dd],4,dd,f)!=(size_t)dd){fclose(f);return false;}}fclose(f);d=dd;return true;}

static int exact_over_eligible(faiss::DistanceComputer&dc,const SetSel&sel,faiss::IndexIDMap*map,int N,int k,
                               std::vector<float>&outd,std::vector<idx_t>&outi,uint64_t&ndis){
    auto ext=[&](sidx v){return map->id_map[v];};
    std::vector<float> D(k); std::vector<idx_t> I(k); int nres=0;
    for(int i=0;i<N;i++){ if(!sel.is_member(ext(i))) continue; float d=dc(i); ndis++;
        if(nres<k)faiss::maxheap_push(++nres,D.data(),I.data(),d,i); else if(d<D[0])faiss::maxheap_replace_top(nres,D.data(),I.data(),d,i); }
    std::vector<std::pair<float,idx_t>> tmp;for(int i=0;i<nres;i++)tmp.push_back({D[i],I[i]});std::sort(tmp.begin(),tmp.end());
    for(size_t i=0;i<tmp.size();i++){outd[i]=tmp[i].first;outi[i]=ext(tmp[i].second);}for(int i=(int)tmp.size();i<k;i++)outi[i]=-1;
    return nres;
}

static int rsearch(faiss::IndexIDMap*map,const float*q,int k,int ef,const SetSel&sel,Base base,
                   bool asf,bool aef,double aef_thr,int Mbase,int stride,
                   std::vector<float>&outd,std::vector<idx_t>&outi,uint64_t*ndis_o,uint64_t*exam_o,bool*used_exact){
    auto*ih=dynamic_cast<faiss::IndexHNSW*>(map->index); const faiss::HNSW&h=ih->hnsw; const int N=ih->ntotal;
    std::unique_ptr<faiss::DistanceComputer> dc(ih->storage->get_distance_computer()); dc->set_query(q);
    auto ext=[&](sidx v){return map->id_map[v];}; uint64_t ndis=0,exam=0,passed=0; if(used_exact)*used_exact=false;
    sidx nearest=h.entry_point; float dn=(*dc)(nearest); ndis++;
    for(int lvl=h.max_level;lvl>=1;lvl--){for(;;){auto pv=nearest;size_t b,e;h.neighbor_range(nearest,lvl,&b,&e);for(size_t i=b;i<e;i++){auto v=h.neighbors[i];if(v<0)break;float d=(*dc)(v);ndis++;if(d<dn){dn=d;nearest=v;}}if(nearest==pv)break;}}
    const int efn=std::max(ef,k); faiss::HNSW::MinimaxHeap cand(efn);
    std::vector<float> D(k); std::vector<idx_t> I(k); int nres=0; faiss::VisitedTable vis(N),brd(N);
    // batch: (id,isResult). isResult=false => frontier-only bridge (STD failing / ASF bridge).
    sidx bq[4]; char br[4]; int bn=0;
    auto flush=[&](){if(!bn)return;float bd[4]={0,0,0,0};if(bn==4)dc->distances_batch_4(bq[0],bq[1],bq[2],bq[3],bd[0],bd[1],bd[2],bd[3]);else for(int t=0;t<bn;t++)bd[t]=(*dc)(bq[t]);ndis+=bn;
        for(int t=0;t<bn;t++){ if(br[t]){ if(nres<k)faiss::maxheap_push(++nres,D.data(),I.data(),bd[t],bq[t]); else if(bd[t]<D[0])faiss::maxheap_replace_top(nres,D.data(),I.data(),bd[t],bq[t]); } cand.push(bq[t],bd[t]); } bn=0;};
    auto push=[&](sidx v,bool isRes){vis.set(v);bq[bn]=v;br[bn]=isRes?1:0;bn++;if(bn==4)flush();};
    { bool p=sel.is_member(ext(nearest)); if(base==STD){push(nearest,p);} else if(p){push(nearest,true);} else {cand.push(nearest,dn);} }
    flush();  // ensure the seed reaches the frontier before the loop
    while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;
        size_t b,e;h.neighbor_range(v0,0,&b,&e); int found=0; std::vector<sidx> failing;
        for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;exam++; bool p=sel.is_member(ext(v1)); if(p)passed++;
            if(base==STD){ if(!vis.get(v1)) push(v1,p); }
            else { if(p){ if(!vis.get(v1)){push(v1,true);found++;} }
                   else { failing.push_back(v1);
                          if(base==ACORN1 && !brd.get(v1)){ brd.set(v1); size_t b2,e2;h.neighbor_range(v1,0,&b2,&e2);
                              for(size_t j2=b2;j2<e2;j2++){auto v2=h.neighbors[j2];if(v2<0)break;exam++;bool p2=sel.is_member(ext(v2));if(p2)passed++;if(p2&&!vis.get(v2)){push(v2,true);found++;}} } } }
        }
        // ASF: passing supply short -> admit stride-sampled failing bridges (frontier only)
        if(asf && base!=STD && found<Mbase && !failing.empty()){
            int st=std::max(1,stride); for(size_t i=0;i<failing.size();i+=st){ sidx fb=failing[i]; if(!vis.get(fb)) push(fb,false); }
        }
        flush();
        // AEF: pass-ratio collapse -> exact over eligible
        if(aef && exam>(uint64_t)(3*efn) && (double)passed/exam < aef_thr){
            if(used_exact)*used_exact=true; std::unique_ptr<faiss::DistanceComputer> dc2(ih->storage->get_distance_computer()); dc2->set_query(q);
            int r=exact_over_eligible(*dc2,sel,map,N,k,outd,outi,ndis); if(ndis_o)*ndis_o=ndis; if(exam_o)*exam_o=exam+N; return r;
        }
    }
    std::vector<std::pair<float,idx_t>> tmp;for(int i=0;i<nres;i++)tmp.push_back({D[i],I[i]});std::sort(tmp.begin(),tmp.end());
    for(size_t i=0;i<tmp.size();i++){outd[i]=tmp[i].first;outi[i]=ext(tmp[i].second);}for(int i=(int)tmp.size();i<k;i++)outi[i]=-1;
    if(ndis_o)*ndis_o=ndis; if(exam_o)*exam_o=exam; return nres;
}

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):100000,G=argc>2?atoi(argv[2]):12; double aefT=argc>3?atof(argv[3]):0.02; const int K=10,NQ=50,Mbase=16;
    std::vector<float> X;int d=128; if(!load_fvecs("research/acorn/data/sift/sift_base.fvecs",N,X,d)){fprintf(stderr,"need sift\n");return 1;} N=(int)(X.size()/d);
    std::vector<float> Q;int qd=0; load_fvecs("research/acorn/data/sift/sift_query.fvecs",NQ,Q,qd);
    std::vector<idx_t> ids(N);for(int i=0;i<N;i++)ids[i]=i;
    auto build=[&](int M){auto*h=new faiss::IndexHNSWFlat(d,M,faiss::METRIC_L2);h->hnsw.efConstruction=100;auto*m=new faiss::IndexIDMap(h);m->own_fields=true;m->add_with_ids(N,X.data(),ids.data());return m;};
    auto* smap=build(16); auto* gmap=build(16*G);
    fprintf(stderr,"[bench_racorn] n=%d gamma=%d aef_thr=%.3f built\n",N,G,aefT);
    std::vector<std::vector<int>> ord(NQ);
    for(int i=0;i<NQ;i++){std::vector<std::pair<float,int>> ds(N);const float*q=&Q[(size_t)i*d];for(int b=0;b<N;b++){float s=0;for(int j=0;j<d;j++){float df=q[j]-X[(size_t)b*d+j];s+=df*df;}ds[b]={s,b};}std::sort(ds.begin(),ds.end());ord[i].resize(N);for(int b=0;b<N;b++)ord[i][b]=ds[b].second;}
    auto mask=[&](const char*c,int cand,int qi){SetSel s;s.bit.assign(N,0);if(std::string(c)=="no"){faiss::RandomGenerator r(1000+qi);int p=0;while(p<cand){int id=r.rand_int(N);if(!s.bit[id]){s.bit[id]=1;p++;}}}else if(std::string(c)=="pos"){for(int j=0;j<cand;j++)s.bit[ord[qi][j]]=1;}else{for(int j=0;j<cand;j++)s.bit[ord[qi][N-1-j]]=1;}return s;};
    auto brute=[&](const float*q,const SetSel&s){std::vector<std::pair<float,idx_t>> sc;for(int i=0;i<N;i++){if(!s.bit[i])continue;float d2=0;for(int j=0;j<d;j++){float df=q[j]-X[(size_t)i*d+j];d2+=df*df;}sc.push_back({d2,i});}std::sort(sc.begin(),sc.end());std::unordered_set<idx_t> t;for(int i=0;i<K&&i<(int)sc.size();i++)t.insert(sc[i].second);return t;};
    using clk=std::chrono::high_resolution_clock;
    int EFS[4]={50,100,250,500}; double sels[5]={0.001,0.01,0.05,0.10,0.25};
    struct Cfg{const char*name;Base base;faiss::IndexIDMap*map;bool asf;bool aef;};
    Cfg cfgs[]={
        {"STD",STD,smap,false,false},{"STD+ASF+AEF",STD,smap,true,true},
        {"ACORN1",ACORN1,smap,false,false},{"ACORN1+ASF",ACORN1,smap,true,false},{"ACORN1+ASF+AEF",ACORN1,smap,true,true},
        {"GAMMA",GAMMA,gmap,false,false},{"GAMMA+ASF",GAMMA,gmap,true,false},{"GAMMA+ASF+AEF",GAMMA,gmap,true,true},
    };
    for(const char* corr:{"neg","no"}){
        printf("\n################ correlation=%s (rec / latency-us / exact%%) ################\n",corr);
        printf("%6s","sel"); for(auto&c:cfgs) printf(" | %-16s",c.name); printf("\n");
        for(double se:sels){ int cand=std::max(K,(int)(se*N));
            std::vector<SetSel> M(NQ);std::vector<std::unordered_set<idx_t>> tr(NQ);for(int i=0;i<NQ;i++){M[i]=mask(corr,cand,i);tr[i]=brute(&Q[(size_t)i*d],M[i]);}
            printf("%5.1f%%",100.0*cand/N);
            for(auto&c:cfgs){ double bR=-1,bL=0,bE=0;
                for(int ef:EFS){ double r=0,l=0;int ex=0,den=0; std::vector<float> dd(K);std::vector<idx_t> ii(K);
                    for(int i=0;i<NQ;i++){uint64_t nd=0,xm=0;bool used=false;auto t0=clk::now();int rr=rsearch(c.map,&Q[(size_t)i*d],K,ef,M[i],c.base,c.asf,c.aef,aefT,Mbase,3,dd,ii,&nd,&xm,&used);auto t1=clk::now();
                        l+=std::chrono::duration<double,std::micro>(t1-t0).count();if(used)ex++;int h=0;for(int j=0;j<rr;j++)if(tr[i].count(ii[j]))h++;if(!tr[i].empty()){r+=double(h)/tr[i].size();den++;}}
                    double R=den?r/den:0; if(R>bR+1e-9){bR=R;bL=l/NQ;bE=100.0*ex/NQ;} }
                printf(" | %.2f/%5.0f/%2.0f",bR,bL,bE);
            }
            printf("\n");
        }
    }
    fflush(stdout); std::_Exit(0);
}
