/*
 * Seeded entry points for filtered search: instead of starting the HNSW walk at the fixed
 * top node, seed the frontier with S eligible docs sampled from the filter, then run a
 * bounded best-first walk on the predicate subgraph toward the query. Idea: start INSIDE the
 * eligible region so the walk slides to the nearest eligible, sidestepping the entry->eligible
 * traversal that collapses under negative correlation.
 * Compares: fixed-entry (standard in-filtering) | seeded (S) | seeded (S) + 2-hop | exact.
 * Focus: compact far cluster (fc) + diffuse far (neg) + scattered (no). SIFT1M.
 * Usage: bench_seeded [n]   (default 100000)
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
#include <faiss/impl/HNSW.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
using idx_t=faiss::idx_t; using sidx=faiss::HNSW::storage_idx_t;
static int D=128; static const float* XG=nullptr;
static bool load_fvecs(const char*p,int want,std::vector<float>&o,int&d){FILE*f=fopen(p,"rb");if(!f)return false;int dd=0;if(fread(&dd,4,1,f)!=1){fclose(f);return false;}fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);long rec=4+4L*dd,tot=b/rec;int n=want>0?std::min<long>(want,tot):tot;o.resize((size_t)n*dd);for(int i=0;i<n;i++){int t;if(fread(&t,4,1,f)!=1||t!=dd){fclose(f);return false;}if(fread(&o[(size_t)i*dd],4,dd,f)!=(size_t)dd){fclose(f);return false;}}fclose(f);d=dd;return true;}
static inline float l2(const float*a,const float*b){float s=0;for(int j=0;j<D;j++){float df=a[j]-b[j];s+=df*df;}return s;}
static inline float l2v(const float*q,sidx v){return l2(q,&XG[(size_t)v*D]);}

// seeded filtered best-first over the standard graph. seeds = eligible docs.
static int seeded(faiss::IndexIDMap*map,const float*q,int k,int ef,const std::vector<char>&m,
                  const std::vector<int>&seeds,bool two_hop,std::vector<idx_t>&oi,uint64_t*dis){
    auto*ih=dynamic_cast<faiss::IndexHNSW*>(map->index);const faiss::HNSW&h=ih->hnsw;const int N=ih->ntotal;
    uint64_t nd=0; const int efn=std::max(ef,k); faiss::HNSW::MinimaxHeap cand(efn);
    std::vector<float> Dh(k);std::vector<idx_t> Ih(k);int nres=0;faiss::VisitedTable vis(N),brd(N);
    auto admit=[&](sidx v){vis.set(v);float d=l2v(q,v);nd++; if(nres<k)faiss::maxheap_push(++nres,Dh.data(),Ih.data(),d,v);else if(d<Dh[0])faiss::maxheap_replace_top(nres,Dh.data(),Ih.data(),d,v); cand.push(v,d);};
    for(int s:seeds) if(!vis.get(s)) admit(s);                    // seed the frontier with eligible docs
    while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;
        size_t b,e;h.neighbor_range(v0,0,&b,&e);
        for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;
            if(m[v1]){ if(!vis.get(v1))admit(v1); }
            else if(two_hop && !brd.get(v1)){ brd.set(v1); size_t b2,e2;h.neighbor_range(v1,0,&b2,&e2);
                for(size_t j2=b2;j2<e2;j2++){auto v2=h.neighbors[j2];if(v2<0)break;if(m[v2]&&!vis.get(v2))admit(v2);} }
        }
    }
    std::vector<std::pair<float,idx_t>> t;for(int i=0;i<nres;i++)t.push_back({Dh[i],Ih[i]});std::sort(t.begin(),t.end());
    for(size_t i=0;i<t.size();i++)oi[i]=map->id_map[t[i].second];for(int i=(int)t.size();i<k;i++)oi[i]=-1;if(dis)*dis=nd;return (int)t.size();
}
// fixed-entry standard in-filtering (baseline that collapses under negative correlation)
static int fixed(faiss::IndexIDMap*map,const float*q,int k,int ef,const std::vector<char>&m,std::vector<idx_t>&oi,uint64_t*dis){
    auto*ih=dynamic_cast<faiss::IndexHNSW*>(map->index);const faiss::HNSW&h=ih->hnsw;const int N=ih->ntotal;uint64_t nd=0;
    sidx nr=h.entry_point;float dn=l2v(q,nr);nd++;
    for(int lvl=h.max_level;lvl>=1;lvl--){for(;;){auto pv=nr;size_t b,e;h.neighbor_range(nr,lvl,&b,&e);for(size_t i=b;i<e;i++){auto v=h.neighbors[i];if(v<0)break;float d=l2v(q,v);nd++;if(d<dn){dn=d;nr=v;}}if(nr==pv)break;}}
    const int efn=std::max(ef,k);faiss::HNSW::MinimaxHeap cand(efn);std::vector<float> Dh(k);std::vector<idx_t> Ih(k);int nres=0;faiss::VisitedTable vis(N);
    auto add=[&](sidx v){vis.set(v);float d=l2v(q,v);nd++;if(m[v]){if(nres<k)faiss::maxheap_push(++nres,Dh.data(),Ih.data(),d,v);else if(d<Dh[0])faiss::maxheap_replace_top(nres,Dh.data(),Ih.data(),d,v);}cand.push(v,d);};
    add(nr);
    while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;size_t b,e;h.neighbor_range(v0,0,&b,&e);for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;if(!vis.get(v1))add(v1);}}
    std::vector<std::pair<float,idx_t>> t;for(int i=0;i<nres;i++)t.push_back({Dh[i],Ih[i]});std::sort(t.begin(),t.end());
    for(size_t i=0;i<t.size();i++)oi[i]=map->id_map[t[i].second];for(int i=(int)t.size();i<k;i++)oi[i]=-1;if(dis)*dis=nd;return (int)t.size();
}
int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):100000;const int K=10,NQ=50;
    std::vector<float> X;int d=128;if(!load_fvecs("research/acorn/data/sift/sift_base.fvecs",N,X,d)){fprintf(stderr,"need sift\n");return 1;}N=(int)(X.size()/d);D=d;XG=X.data();
    std::vector<float> Q;int qd=0;load_fvecs("research/acorn/data/sift/sift_query.fvecs",NQ,Q,qd);
    std::vector<idx_t> ids(N);for(int i=0;i<N;i++)ids[i]=i;
    auto*hh=new faiss::IndexHNSWFlat(d,16,faiss::METRIC_L2);hh->hnsw.efConstruction=100;auto*map=new faiss::IndexIDMap(hh);map->own_fields=true;map->add_with_ids(N,X.data(),ids.data());
    fprintf(stderr,"[bench_seeded] n=%d built\n",N);
    std::vector<std::vector<int>> ord(NQ);
    for(int i=0;i<NQ;i++){std::vector<std::pair<float,int>> ds(N);const float*q=&Q[(size_t)i*d];for(int b=0;b<N;b++)ds[b]={l2v(q,b),b};std::sort(ds.begin(),ds.end());ord[i].resize(N);for(int b=0;b<N;b++)ord[i][b]=ds[b].second;}
    auto mask=[&](const char*c,int cand,int qi){std::vector<char> s(N,0);std::string cc(c);
        if(cc=="no"){faiss::RandomGenerator r(1000+qi);int p=0;while(p<cand){int id=r.rand_int(N);if(!s[id]){s[id]=1;p++;}}}
        else if(cc=="fc"){const float* a=&XG[(size_t)ord[qi][N-1]*D];std::vector<std::pair<float,int>> ds(N);for(int b=0;b<N;b++)ds[b]={l2(a,&XG[(size_t)b*D]),b};std::partial_sort(ds.begin(),ds.begin()+cand,ds.end());for(int j=0;j<cand;j++)s[ds[j].second]=1;}
        else{for(int j=0;j<cand;j++)s[ord[qi][N-1-j]]=1;} return s;};
    auto truth=[&](const float*q,const std::vector<char>&m){std::vector<std::pair<float,int>> sc;for(int i=0;i<N;i++)if(m[i])sc.push_back({l2v(q,i),i});std::sort(sc.begin(),sc.end());std::unordered_set<int> t;for(int i=0;i<K&&i<(int)sc.size();i++)t.insert(sc[i].second);return t;};
    auto rec=[&](std::vector<idx_t>&oi,int r,std::unordered_set<int>&tr){if(tr.empty())return 1.0;int h=0;for(int j=0;j<r;j++)if(tr.count((int)oi[j]))h++;return double(h)/tr.size();};
    using clk=std::chrono::high_resolution_clock;int ef=200;int SEEDS[3]={8,32,128};double sels[3]={0.01,0.05,0.25};
    for(const char* corr:{"fc","neg","no"}){
        printf("\n########## correlation=%s (rec / latency-us) ##########\n",corr);
        printf("%6s | %-14s","sel","fixed-entry"); for(int S:SEEDS)printf(" | seed=%-3d(+2h)",S); printf("\n");
        for(double se:sels){int cand=std::max(K,(int)(se*N));
            std::vector<std::vector<char>> M(NQ);std::vector<std::unordered_set<int>> TR(NQ);std::vector<std::vector<int>> ELI(NQ);
            for(int i=0;i<NQ;i++){M[i]=mask(corr,cand,i);TR[i]=truth(&Q[(size_t)i*d],M[i]);for(int b=0;b<N;b++)if(M[i][b])ELI[i].push_back(b);}
            printf("%5.1f%%",100.0*cand/N);
            {double r=0,l=0;std::vector<idx_t> oi(K);for(int i=0;i<NQ;i++){uint64_t dd=0;auto t0=clk::now();int rr=fixed(map,&Q[(size_t)i*d],K,ef,M[i],oi,&dd);auto t1=clk::now();l+=std::chrono::duration<double,std::micro>(t1-t0).count();r+=rec(oi,rr,TR[i]);}printf(" | %.2f/%6.0f",r/NQ,l/NQ);}
            for(int S:SEEDS){double r=0,l=0;std::vector<idx_t> oi(K);
                for(int i=0;i<NQ;i++){ // sample S eligible seeds, strided for spread
                    std::vector<int> seeds; int m=(int)ELI[i].size(); int st=std::max(1,m/S); for(int j=0;j<m&&(int)seeds.size()<S;j+=st)seeds.push_back(ELI[i][j]);
                    uint64_t dd=0;auto t0=clk::now();int rr=seeded(map,&Q[(size_t)i*d],K,ef,M[i],seeds,true,oi,&dd);auto t1=clk::now();l+=std::chrono::duration<double,std::micro>(t1-t0).count();r+=rec(oi,rr,TR[i]);}
                printf(" | %.2f/%6.0f",r/NQ,l/NQ);}
            printf("\n");
        }
    }
    fflush(stdout); std::_Exit(0);
}
