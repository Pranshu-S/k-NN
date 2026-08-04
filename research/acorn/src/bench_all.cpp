/*
 * Grand comparison: standard vs every technique from this investigation, on identical
 * data/queries, across all correlation regimes x selectivity. Best recall over ef, latency us.
 * Methods: standard(fixed-entry) | ACORN-1(2hop) | ACORN-gamma(dense 1hop) |
 *          self-aware(std+inline AEF) | seeded(entry from eligible +2hop) | exact(ceiling)
 * Correlations: pos | no | fc(compact far cluster) | neg(diffuse far shell)
 * Usage: bench_all [n] [gamma]   (default 100000 12)
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
static void finalize(std::vector<float>&Dh,std::vector<idx_t>&Ih,int nres,faiss::IndexIDMap*map,int k,std::vector<idx_t>&oi){
    std::vector<std::pair<float,idx_t>> t;for(int i=0;i<nres;i++)t.push_back({Dh[i],Ih[i]});std::sort(t.begin(),t.end());
    for(size_t i=0;i<t.size();i++)oi[i]=map->id_map[t[i].second];for(int i=(int)t.size();i<k;i++)oi[i]=-1;
}
static int m_exact(faiss::IndexIDMap*map,const float*q,int k,const std::vector<char>&m,std::vector<idx_t>&oi){
    const int N=((faiss::IndexHNSW*)map->index)->ntotal;std::vector<float> Dh(k);std::vector<idx_t> Ih(k);int nres=0;
    for(int i=0;i<N;i++){if(!m[i])continue;float d=l2v(q,i);if(nres<k)faiss::maxheap_push(++nres,Dh.data(),Ih.data(),d,i);else if(d<Dh[0])faiss::maxheap_replace_top(nres,Dh.data(),Ih.data(),d,i);}
    finalize(Dh,Ih,nres,map,k,oi);return nres;
}
static void descend(const faiss::HNSW&h,const float*q,sidx&nr,float&dn){dn=l2v(q,nr);for(int lvl=h.max_level;lvl>=1;lvl--){for(;;){auto pv=nr;size_t b,e;h.neighbor_range(nr,lvl,&b,&e);for(size_t i=b;i<e;i++){auto v=h.neighbors[i];if(v<0)break;float d=l2v(q,v);if(d<dn){dn=d;nr=v;}}if(nr==pv)break;}}}
// standard in-filter (fixed entry): traverse all, distance all, results=passing
static int m_std(faiss::IndexIDMap*map,const float*q,int k,int ef,const std::vector<char>&m,std::vector<idx_t>&oi){
    auto*ih=(faiss::IndexHNSW*)map->index;const faiss::HNSW&h=ih->hnsw;const int N=ih->ntotal;sidx nr=h.entry_point;float dn;descend(h,q,nr,dn);
    const int efn=std::max(ef,k);faiss::HNSW::MinimaxHeap cand(efn);std::vector<float> Dh(k);std::vector<idx_t> Ih(k);int nres=0;faiss::VisitedTable vis(N);
    auto add=[&](sidx v){vis.set(v);float d=l2v(q,v);if(m[v]){if(nres<k)faiss::maxheap_push(++nres,Dh.data(),Ih.data(),d,v);else if(d<Dh[0])faiss::maxheap_replace_top(nres,Dh.data(),Ih.data(),d,v);}cand.push(v,d);};
    add(nr);while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;size_t b,e;h.neighbor_range(v0,0,&b,&e);for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;if(!vis.get(v1))add(v1);}}
    finalize(Dh,Ih,nres,map,k,oi);return nres;
}
// frontier=passing-only; two_hop=ACORN-1, else 1-hop (use dense map for gamma). seeds optional.
static int m_pred(faiss::IndexIDMap*map,const float*q,int k,int ef,const std::vector<char>&m,bool two_hop,const std::vector<int>*seeds,std::vector<idx_t>&oi){
    auto*ih=(faiss::IndexHNSW*)map->index;const faiss::HNSW&h=ih->hnsw;const int N=ih->ntotal;
    const int efn=std::max(ef,k);faiss::HNSW::MinimaxHeap cand(efn);std::vector<float> Dh(k);std::vector<idx_t> Ih(k);int nres=0;faiss::VisitedTable vis(N),brd(N);
    auto admit=[&](sidx v){vis.set(v);float d=l2v(q,v);if(nres<k)faiss::maxheap_push(++nres,Dh.data(),Ih.data(),d,v);else if(d<Dh[0])faiss::maxheap_replace_top(nres,Dh.data(),Ih.data(),d,v);cand.push(v,d);};
    if(seeds){for(int s:*seeds)if(!vis.get(s))admit(s);} else {sidx nr=h.entry_point;float dn;descend(h,q,nr,dn);if(m[nr])admit(nr);else cand.push(nr,dn);}
    while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;size_t b,e;h.neighbor_range(v0,0,&b,&e);
        for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;
            if(m[v1]){if(!vis.get(v1))admit(v1);}
            else if(two_hop&&!brd.get(v1)){brd.set(v1);size_t b2,e2;h.neighbor_range(v1,0,&b2,&e2);for(size_t j2=b2;j2<e2;j2++){auto v2=h.neighbors[j2];if(v2<0)break;if(m[v2]&&!vis.get(v2))admit(v2);}}
        }}
    finalize(Dh,Ih,nres,map,k,oi);return nres;
}
// self-aware: standard walk + inline pass-ratio abort -> exact
static int m_selfaware(faiss::IndexIDMap*map,const float*q,int k,int ef,const std::vector<char>&m,std::vector<idx_t>&oi){
    auto*ih=(faiss::IndexHNSW*)map->index;const faiss::HNSW&h=ih->hnsw;const int N=ih->ntotal;sidx nr=h.entry_point;float dn;descend(h,q,nr,dn);
    const int efn=std::max(ef,k);int minProbe=3*efn;double passThr=0.02;faiss::HNSW::MinimaxHeap cand(efn);std::vector<float> Dh(k);std::vector<idx_t> Ih(k);int nres=0;faiss::VisitedTable vis(N);uint64_t exm=0,pas=0;
    auto add=[&](sidx v){vis.set(v);float d=l2v(q,v);if(m[v]){if(nres<k)faiss::maxheap_push(++nres,Dh.data(),Ih.data(),d,v);else if(d<Dh[0])faiss::maxheap_replace_top(nres,Dh.data(),Ih.data(),d,v);}cand.push(v,d);};
    add(nr);
    while(cand.size()>0){float d0;auto v0=cand.pop_min(&d0);if(cand.count_below(d0)>=efn)break;
        if(exm>(uint64_t)minProbe&&(double)pas/exm<passThr)return m_exact(map,q,k,m,oi);
        size_t b,e;h.neighbor_range(v0,0,&b,&e);for(size_t j=b;j<e;j++){auto v1=h.neighbors[j];if(v1<0)break;if(!vis.get(v1)){exm++;if(m[v1])pas++;add(v1);}}}
    finalize(Dh,Ih,nres,map,k,oi);return nres;
}
int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):100000,G=argc>2?atoi(argv[2]):12;const int K=10,NQ=50;
    std::vector<float> X;int d=128;if(!load_fvecs("research/acorn/data/sift/sift_base.fvecs",N,X,d)){fprintf(stderr,"need sift\n");return 1;}N=(int)(X.size()/d);D=d;XG=X.data();
    std::vector<float> Q;int qd=0;load_fvecs("research/acorn/data/sift/sift_query.fvecs",NQ,Q,qd);
    std::vector<idx_t> ids(N);for(int i=0;i<N;i++)ids[i]=i;
    auto build=[&](int M){auto*hh=new faiss::IndexHNSWFlat(d,M,faiss::METRIC_L2);hh->hnsw.efConstruction=100;auto*mp=new faiss::IndexIDMap(hh);mp->own_fields=true;mp->add_with_ids(N,X.data(),ids.data());return mp;};
    auto* smap=build(16); auto* dmap=build(16*G);
    fprintf(stderr,"[bench_all] n=%d gamma=%d built\n",N,G);
    std::vector<std::vector<int>> ord(NQ);for(int i=0;i<NQ;i++){std::vector<std::pair<float,int>> ds(N);const float*q=&Q[(size_t)i*d];for(int b=0;b<N;b++)ds[b]={l2v(q,b),b};std::sort(ds.begin(),ds.end());ord[i].resize(N);for(int b=0;b<N;b++)ord[i][b]=ds[b].second;}
    auto mask=[&](const char*c,int cand,int qi){std::vector<char> s(N,0);std::string cc(c);
        if(cc=="no"){faiss::RandomGenerator r(1000+qi);int p=0;while(p<cand){int id=r.rand_int(N);if(!s[id]){s[id]=1;p++;}}}
        else if(cc=="pos"){for(int j=0;j<cand;j++)s[ord[qi][j]]=1;}
        else if(cc=="fc"){const float* a=&XG[(size_t)ord[qi][N-1]*D];std::vector<std::pair<float,int>> ds(N);for(int b=0;b<N;b++)ds[b]={l2(a,&XG[(size_t)b*D]),b};std::partial_sort(ds.begin(),ds.begin()+cand,ds.end());for(int j=0;j<cand;j++)s[ds[j].second]=1;}
        else{for(int j=0;j<cand;j++)s[ord[qi][N-1-j]]=1;} return s;};
    auto truth=[&](const float*q,const std::vector<char>&m){std::vector<std::pair<float,int>> sc;for(int i=0;i<N;i++)if(m[i])sc.push_back({l2v(q,i),i});std::sort(sc.begin(),sc.end());std::unordered_set<int> t;for(int i=0;i<K&&i<(int)sc.size();i++)t.insert(sc[i].second);return t;};
    auto rc=[&](std::vector<idx_t>&oi,int r,std::unordered_set<int>&tr){if(tr.empty())return 1.0;int h=0;for(int j=0;j<r;j++)if(tr.count((int)oi[j]))h++;return double(h)/tr.size();};
    using clk=std::chrono::high_resolution_clock;int EFS[4]={50,100,250,500};double sels[4]={0.001,0.01,0.05,0.25};
    const char* names[6]={"standard","ACORN-1","ACORN-g","selfaware","seeded","exact"};
    for(const char* corr:{"pos","no","fc","neg"}){
        printf("\n############ correlation=%s (best recall / latency-us) ############\n",corr);
        printf("%6s",""); for(int mth=0;mth<6;mth++)printf(" | %-13s",names[mth]); printf("\n");
        for(double se:sels){int cand=std::max(K,(int)(se*N));
            std::vector<std::vector<char>> M(NQ);std::vector<std::unordered_set<int>> TR(NQ);std::vector<std::vector<int>> ELI(NQ);
            for(int i=0;i<NQ;i++){M[i]=mask(corr,cand,i);TR[i]=truth(&Q[(size_t)i*d],M[i]);for(int b=0;b<N;b++)if(M[i][b])ELI[i].push_back(b);}
            printf("%5.1f%%",100.0*cand/N);
            for(int mth=0;mth<6;mth++){ double bR=-1,bL=0;
                for(int ef:EFS){ if(mth==5&&ef!=EFS[0])continue; double r=0,l=0;std::vector<idx_t> oi(K);
                    for(int i=0;i<NQ;i++){ std::vector<int> seeds; if(mth==4){int mm=(int)ELI[i].size();int st=std::max(1,mm/8);for(int j=0;j<mm&&(int)seeds.size()<8;j+=st)seeds.push_back(ELI[i][j]);}
                        auto t0=clk::now();int rr;
                        if(mth==0)rr=m_std(smap,&Q[(size_t)i*d],K,ef,M[i],oi);
                        else if(mth==1)rr=m_pred(smap,&Q[(size_t)i*d],K,ef,M[i],true,nullptr,oi);
                        else if(mth==2)rr=m_pred(dmap,&Q[(size_t)i*d],K,ef,M[i],false,nullptr,oi);
                        else if(mth==3)rr=m_selfaware(smap,&Q[(size_t)i*d],K,ef,M[i],oi);
                        else if(mth==4)rr=m_pred(smap,&Q[(size_t)i*d],K,ef,M[i],true,&seeds,oi);
                        else rr=m_exact(smap,&Q[(size_t)i*d],K,M[i],oi);
                        auto t1=clk::now();l+=std::chrono::duration<double,std::micro>(t1-t0).count();r+=rc(oi,rr,TR[i]);}
                    double R=r/NQ; if(R>bR+1e-9){bR=R;bL=l/NQ;} }
                printf(" | %.2f/%6.0f",bR,bL);
            }
            printf("\n");
        }
    }
    fflush(stdout); std::_Exit(0);
}
