/*
 * ACORN-gamma COMPRESSION: build the dense graph (M=gamma*16), then two-hop-prune the
 * level-0 neighbour lists (keep first M_beta; drop any further neighbour already
 * reachable via a two-hop path through a retained neighbour). Search recovers pruned
 * nodes with ADAPTIVE 2-hop (only when 1-hop yields < M_base passing). Compares:
 *   standard | ACORN-gamma dense (1-hop) | ACORN-gamma compressed (1-hop + adaptive 2-hop)
 * on size, recall, latency. Real SIFT, per-query-local correlation.
 * Usage: bench_compress [n] [gamma] [m_beta]   (default 100000 12 32)
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
struct SetSel : faiss::IDSelector { std::vector<char> bit; bool is_member(idx_t id) const override { return id>=0&&(size_t)id<bit.size()&&bit[id]; } };
static bool load_fvecs(const char*p,int want,std::vector<float>&o,int&d){FILE*f=fopen(p,"rb");if(!f)return false;int dd=0;if(fread(&dd,4,1,f)!=1){fclose(f);return false;}fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);long rec=4+4L*dd,tot=b/rec;int n=want>0?std::min<long>(want,tot):tot;o.resize((size_t)n*dd);for(int i=0;i<n;i++){int t;if(fread(&t,4,1,f)!=1||t!=dd){fclose(f);return false;}if(fread(&o[(size_t)i*dd],4,dd,f)!=(size_t)dd){fclose(f);return false;}}fclose(f);d=dd;return true;}

// Two-hop compression of level-0 lists. Returns retained-edge count. Prunes IN PLACE:
// retained neighbours packed to the front of each node's slot range, remainder set -1.
static long compress_level0(faiss::HNSW& h,int M_beta){
    long retained=0, total=0;
    std::unordered_set<sidx> two_hop;
    for(sidx u=0; u<(sidx)(h.offsets.size()-1); u++){
        size_t b,e; h.neighbor_range(u,0,&b,&e);
        std::vector<sidx> orig; for(size_t j=b;j<e;j++){ if(h.neighbors[j]<0)break; orig.push_back(h.neighbors[j]); }
        total += (long)(e-b);
        two_hop.clear(); std::vector<sidx> keep;
        for(size_t p=0;p<orig.size();p++){ sidx v=orig[p];
            bool redundant = (p>=(size_t)M_beta) && two_hop.count(v)>0;
            if(!redundant){ keep.push_back(v);
                size_t vb,ve; h.neighbor_range(v,0,&vb,&ve);
                for(size_t j=vb;j<ve;j++){ if(h.neighbors[j]<0)break; two_hop.insert(h.neighbors[j]); }
            }
        }
        for(size_t j=0;j<(e-b);j++) h.neighbors[b+j] = (j<keep.size())?keep[j]:-1;
        retained += (long)keep.size();
    }
    fprintf(stderr,"[compress] level-0 edges %ld -> %ld (%.1f%% kept)\n",total,retained,100.0*retained/total);
    return retained;
}

// Filtered best-first: 1-hop always; adaptive 2-hop through retained neighbours only when
// 1-hop found < M_base passing (recovers two-hop-pruned nodes). SIMD-batched distances.
static int csearch(faiss::IndexIDMap*map,const float*q,int k,int ef,const SetSel&sel,int M_base,bool two_hop,
                   std::vector<float>&outd,std::vector<idx_t>&outi,uint64_t*ndis_o,uint64_t*exam_o){
    auto*ih=dynamic_cast<faiss::IndexHNSW*>(map->index); const faiss::HNSW&h=ih->hnsw; const int N=ih->ntotal;
    std::unique_ptr<faiss::DistanceComputer> dc(ih->storage->get_distance_computer()); dc->set_query(q);
    auto ext=[&](sidx v){return map->id_map[v];}; uint64_t ndis=0,exam=0;
    sidx nearest=h.entry_point; float dn=(*dc)(nearest); ndis++;
    for(int lvl=h.max_level;lvl>=1;lvl--){for(;;){auto pv=nearest;size_t b,e;h.neighbor_range(nearest,lvl,&b,&e);for(size_t i=b;i<e;i++){auto v=h.neighbors[i];if(v<0)break;float d=(*dc)(v);ndis++;if(d<dn){dn=d;nearest=v;}}if(nearest==pv)break;}}
    const int efn=std::max(ef,k); faiss::HNSW::MinimaxHeap cand(efn);
    std::vector<float> D(k); std::vector<idx_t> I(k); int nres=0; faiss::VisitedTable adm(N),brd(N);
    sidx bq[4]; int bn=0;
    auto flush=[&](){if(!bn)return;float bd[4]={0,0,0,0};if(bn==4)dc->distances_batch_4(bq[0],bq[1],bq[2],bq[3],bd[0],bd[1],bd[2],bd[3]);else for(int t=0;t<bn;t++)bd[t]=(*dc)(bq[t]);ndis+=bn;for(int t=0;t<bn;t++){if(nres<k)faiss::maxheap_push(++nres,D.data(),I.data(),bd[t],bq[t]);else if(bd[t]<D[0])faiss::maxheap_replace_top(nres,D.data(),I.data(),bd[t],bq[t]);cand.push(bq[t],bd[t]);}bn=0;};
    auto admit=[&](sidx v){adm.set(v);bq[bn++]=v;if(bn==4)flush();};
    if(sel.is_member(ext(nearest))){faiss::maxheap_push(++nres,D.data(),I.data(),dn,nearest);adm.set(nearest);}
    cand.push(nearest,dn);
    while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;
        size_t b,e;h.neighbor_range(v0,0,&b,&e); int found=0;
        std::vector<sidx> nbrs;
        for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;exam++;nbrs.push_back(v1);if(sel.is_member(ext(v1))&&!adm.get(v1)){admit(v1);found++;}}
        // adaptive 2-hop: only if 1-hop found too few passing (recovers pruned nodes)
        if(two_hop && found<M_base){ for(sidx v1:nbrs){ if(brd.get(v1))continue; brd.set(v1);
            size_t b2,e2;h.neighbor_range(v1,0,&b2,&e2); for(size_t j2=b2;j2<e2;j2++){auto v2=h.neighbors[j2];if(v2<0)break;exam++;if(sel.is_member(ext(v2))&&!adm.get(v2)){admit(v2);found++;}} } }
        flush();
    }
    std::vector<std::pair<float,idx_t>> tmp;for(int i=0;i<nres;i++)tmp.push_back({D[i],I[i]});std::sort(tmp.begin(),tmp.end());
    for(size_t i=0;i<tmp.size();i++){outd[i]=tmp[i].first;outi[i]=ext(tmp[i].second);}for(int i=(int)tmp.size();i<k;i++)outi[i]=-1;
    if(ndis_o)*ndis_o=ndis; if(exam_o)*exam_o=exam; return nres;
}

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):100000,G=argc>2?atoi(argv[2]):12,MB=argc>3?atoi(argv[3]):32; const int K=10,NQ=50,Mbase=16;
    std::vector<float> X;int d=128; if(!load_fvecs("research/acorn/data/sift/sift_base.fvecs",N,X,d)){fprintf(stderr,"need sift\n");return 1;} N=(int)(X.size()/d);
    std::vector<float> Q;int qd=0; load_fvecs("research/acorn/data/sift/sift_query.fvecs",NQ,Q,qd);
    std::vector<idx_t> ids(N);for(int i=0;i<N;i++)ids[i]=i;
    auto build=[&](int M){auto*h=new faiss::IndexHNSWFlat(d,M,faiss::METRIC_L2);h->hnsw.efConstruction=100;auto*m=new faiss::IndexIDMap(h);m->own_fields=true;m->add_with_ids(N,X.data(),ids.data());return m;};
    auto* smap=build(16);                 // standard
    auto* dmap=build(16*G);               // ACORN-gamma dense
    // compressed copy of the dense graph
    auto* hc=new faiss::IndexHNSWFlat(d,16*G,faiss::METRIC_L2); auto* cmap=new faiss::IndexIDMap(hc); cmap->own_fields=true;
    hc->storage->add(N,X.data()); hc->hnsw=dynamic_cast<faiss::IndexHNSW*>(dmap->index)->hnsw; hc->ntotal=N;
    cmap->id_map=ids; cmap->ntotal=N;
    long denseE=(long)dynamic_cast<faiss::IndexHNSW*>(dmap->index)->hnsw.neighbors.size();
    long keptE=compress_level0(hc->hnsw,MB);
    fprintf(stderr,"[bench_compress] n=%d gamma=%d M_beta=%d built\n",N,G,MB);

    std::vector<std::vector<int>> ord(NQ);
    for(int i=0;i<NQ;i++){std::vector<std::pair<float,int>> ds(N);const float*q=&Q[(size_t)i*d];for(int b=0;b<N;b++){float s=0;for(int j=0;j<d;j++){float df=q[j]-X[(size_t)b*d+j];s+=df*df;}ds[b]={s,b};}std::sort(ds.begin(),ds.end());ord[i].resize(N);for(int b=0;b<N;b++)ord[i][b]=ds[b].second;}
    auto mask=[&](const char*c,int cand,int qi){SetSel s;s.bit.assign(N,0);if(std::string(c)=="no"){faiss::RandomGenerator r(1000+qi);int p=0;while(p<cand){int id=r.rand_int(N);if(!s.bit[id]){s.bit[id]=1;p++;}}}else if(std::string(c)=="pos"){for(int j=0;j<cand;j++)s.bit[ord[qi][j]]=1;}else{for(int j=0;j<cand;j++)s.bit[ord[qi][N-1-j]]=1;}return s;};
    auto brute=[&](const float*q,const SetSel&s){std::vector<std::pair<float,idx_t>> sc;for(int i=0;i<N;i++){if(!s.bit[i])continue;float d2=0;for(int j=0;j<d;j++){float df=q[j]-X[(size_t)i*d+j];d2+=df*df;}sc.push_back({d2,i});}std::sort(sc.begin(),sc.end());std::unordered_set<idx_t> t;for(int i=0;i<K&&i<(int)sc.size();i++)t.insert(sc[i].second);return t;};
    using clk=std::chrono::high_resolution_clock;
    auto run=[&](faiss::IndexIDMap*m,bool two_hop,int ef,std::vector<SetSel>&M,std::vector<std::unordered_set<idx_t>>&tr,double&rec,double&lat,double&nd){
        std::vector<float> dd(K);std::vector<idx_t> ii(K);rec=0;lat=0;nd=0;int den=0;
        for(int i=0;i<NQ;i++){uint64_t g=0,x=0;auto t0=clk::now();int r=csearch(m,&Q[(size_t)i*d],K,ef,M[i],Mbase,two_hop,dd,ii,&g,&x);auto t1=clk::now();lat+=std::chrono::duration<double,std::micro>(t1-t0).count();nd+=g;int h=0;for(int j=0;j<r;j++)if(tr[i].count(ii[j]))h++;if(!tr[i].empty()){rec+=double(h)/tr[i].size();den++;}}
        rec=den?rec/den:0;lat/=NQ;nd/=NQ;};
    auto fstd=[&](int ef,std::vector<SetSel>&M,std::vector<std::unordered_set<idx_t>>&tr,double&rec,double&lat){std::vector<float> dd(K);std::vector<idx_t> ii(K);rec=0;lat=0;int den=0;for(int i=0;i<NQ;i++){faiss::SearchParametersHNSW p;p.efSearch=ef;p.sel=&M[i];auto t0=clk::now();smap->search(1,&Q[(size_t)i*d],K,dd.data(),ii.data(),&p);auto t1=clk::now();lat+=std::chrono::duration<double,std::micro>(t1-t0).count();int h=0;for(int j=0;j<K;j++)if(ii[j]>=0&&tr[i].count(ii[j]))h++;if(!tr[i].empty()){rec+=double(h)/tr[i].size();den++;}}rec=den?rec/den:0;lat/=NQ;};

    printf("\n#### n=%d gamma=%d M_beta=%d   graph edges: dense=%ld compressed=%ld (%.0f%% of dense)\n",N,G,MB,denseE,keptE,100.0*keptE/denseE);
    printf("#### total index size x (vs standard): dense~%.2f  compressed~%.2f  (vectors=%.0fMB)\n",
        (denseE*4.0+(double)N*d*4)/((double)N*(16*2)*4+(double)N*d*4), (keptE*4.0+(double)N*d*4)/((double)N*(16*2)*4+(double)N*d*4), (double)N*d*4/1e6);
    int EFS[5]={25,50,100,250,500}; double sels[5]={0.01,0.05,0.10,0.25,0.50};
    for(const char* c:{"no","pos"}){
        printf("\n=== correlation=%s ===  rec / latency\n",c);
        printf("%6s | %-14s | %-18s | %-22s\n","sel","standard","gamma-dense(1hop)","gamma-compressed");
        for(double se:sels){int cand=std::max(K,(int)(se*N));
            std::vector<SetSel> M(NQ);std::vector<std::unordered_set<idx_t>> tr(NQ);for(int i=0;i<NQ;i++){M[i]=mask(c,cand,i);tr[i]=brute(&Q[(size_t)i*d],M[i]);}
            double bS=-1,bSl=0,bD=-1,bDl=0,bDn=0,bC=-1,bCl=0,bCn=0;
            for(int ef:EFS){double r,l,n;fstd(ef,M,tr,r,l);if(r>bS+1e-9){bS=r;bSl=l;}
                run(dmap,false,ef,M,tr,r,l,n);if(r>bD+1e-9){bD=r;bDl=l;bDn=n;}
                run(cmap,true,ef,M,tr,r,l,n);if(r>bC+1e-9){bC=r;bCl=l;bCn=n;}}
            printf("%5.1f%% | %.2f / %5.0fus | %.2f / %6.0fus | %.2f / %6.0fus\n",100.0*cand/N,bS,bSl,bD,bDl,bC,bCl);
        }
    }
    fflush(stdout); std::_Exit(0);
}
