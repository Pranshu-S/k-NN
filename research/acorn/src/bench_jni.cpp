/*
 * Benchmark the CORRECTED JNI ACORN-1 (knn_jni::acorn::search) vs stock faiss
 * (IndexIDMap::search + IDSelector = the real OpenSearch standard path) on real
 * SIFT, per-query-local correlation. Produces the standard-vs-ACORN table from
 * the shipping code. Usage: bench_jni [n]   (default 100000)
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
#include <faiss/impl/HNSW.h>        // faiss::hnsw_stats
#include <faiss/utils/random.h>
#include "acorn_hnsw.h"

using faiss::idx_t;
static int D=128;

// Bitset selector — matches OpenSearch's IDSelectorJlongBitmap (O(1) bit test),
// NOT a hash set. Selector cost must reflect the shipping system, since ACORN calls
// is_member on every 1-hop and 2-hop neighbour.
struct SetSel : faiss::IDSelector {
    std::vector<char> bit;                 // membership by (identity) external id
    void add(idx_t id){ if(id>=0){ if((size_t)id>=bit.size()) bit.resize(id+1,0); bit[id]=1; } }
    size_t size() const { size_t c=0; for(char b:bit) c+=b; return c; }
    bool is_member(idx_t id) const override { return id>=0 && (size_t)id<bit.size() && bit[id]; }
};

static bool load_fvecs(const char* path,int want,std::vector<float>&out,int&d){
    FILE*fp=fopen(path,"rb"); if(!fp)return false; int dd=0; if(fread(&dd,4,1,fp)!=1){fclose(fp);return false;}
    fseek(fp,0,SEEK_END); long b=ftell(fp); fseek(fp,0,SEEK_SET); long rec=4+4L*dd,tot=b/rec; int n=want>0?std::min<long>(want,tot):tot;
    out.resize((size_t)n*dd); for(int i=0;i<n;i++){int t;if(fread(&t,4,1,fp)!=1||t!=dd){fclose(fp);return false;} if(fread(&out[(size_t)i*dd],4,dd,fp)!=(size_t)dd){fclose(fp);return false;}}
    fclose(fp); d=dd; return true;
}

int main(int argc,char**argv){
    int N = argc>1?atoi(argv[1]):100000; int mult=argc>2?atoi(argv[2]):1; const int K=10, NQ=50;
    std::vector<float> X; int d=128;
    if(!load_fvecs("research/acorn/data/sift/sift_base.fvecs",N,X,d)){ fprintf(stderr,"need sift_base.fvecs\n"); return 1; }
    N=(int)(X.size()/d);
    std::vector<float> Q; int qd=0; load_fvecs("research/acorn/data/sift/sift_query.fvecs",NQ,Q,qd);
    // Dimension multiplier: tile each vector mult times to make distance mult× more
    // expensive while keeping the SAME nearest-neighbour structure — isolates the
    // effect of distance-computation cost (proxy for high-dimensional data like GIST).
    if(mult>1){ int nd=d*mult;
        std::vector<float> X2((size_t)N*nd); for(int i=0;i<N;i++)for(int m=0;m<mult;m++)for(int j=0;j<d;j++)X2[(size_t)i*nd+m*d+j]=X[(size_t)i*d+j]; X.swap(X2);
        std::vector<float> Q2((size_t)NQ*nd); for(int i=0;i<NQ;i++)for(int m=0;m<mult;m++)for(int j=0;j<d;j++)Q2[(size_t)i*nd+m*d+j]=Q[(size_t)i*d+j]; Q.swap(Q2); d=nd; }
    D=d;
    fprintf(stderr,"[bench_jni] n=%d d=%d (mult=%d) nq=%d\n",N,d,mult,NQ);

    auto* hf=new faiss::IndexHNSWFlat(d,16,faiss::METRIC_L2); hf->hnsw.efConstruction=100;
    faiss::IndexIDMap idmap(hf); idmap.own_fields=true;
    std::vector<idx_t> ids0(N); for(int i=0;i<N;i++) ids0[i]=i;
    idmap.add_with_ids(N,X.data(),ids0.data());
    fprintf(stderr,"[bench_jni] graph built\n");

    // per-query order
    std::vector<std::vector<int>> ord(NQ);
    for(int i=0;i<NQ;i++){ std::vector<std::pair<float,int>> ds(N); const float*q=&Q[(size_t)i*d];
        for(int b=0;b<N;b++){float s=0;for(int j=0;j<d;j++){float df=q[j]-X[(size_t)b*d+j];s+=df*df;}ds[b]={s,b};}
        std::sort(ds.begin(),ds.end()); ord[i].resize(N); for(int b=0;b<N;b++)ord[i][b]=ds[b].second; }
    auto mask=[&](const char*c,int cand,int qi){ SetSel s; s.bit.assign(N,0);
        if(std::string(c)=="no"){faiss::RandomGenerator r(1000+qi);int p=0;while(p<cand){int id=r.rand_int(N); if(!s.bit[id]){s.bit[id]=1;p++;}}}
        else if(std::string(c)=="pos"){for(int j=0;j<cand;j++)s.bit[ord[qi][j]]=1;}
        else{for(int j=0;j<cand;j++)s.bit[ord[qi][N-1-j]]=1;} return s; };
    auto brute=[&](const float*q,const SetSel&s){ std::vector<std::pair<float,idx_t>> sc;
        for(int i=0;i<N;i++){if(!s.bit[i])continue;float d2=0;for(int j=0;j<d;j++){float df=q[j]-X[(size_t)i*d+j];d2+=df*df;}sc.push_back({d2,i});}
        std::sort(sc.begin(),sc.end()); std::unordered_set<idx_t> t; for(int i=0;i<K&&i<(int)sc.size();i++)t.insert(sc[i].second); return t; };

    const char* corrs[3]={"neg","no","pos"}; int EFS[4]={50,100,250,500};
    double sels[8]={0.001,0.005,0.01,0.02,0.05,0.10,0.25,0.50};
    printf("\n#### CORRECTED JNI ACORN-1 vs stock faiss (IndexIDMap::search)  n=%d  SIFT1M\n",N);
    for(const char* c:corrs){
        printf("\n=== correlation=%s ===  cell = recall / ndis / mean-latency  [best recall over ef]\n",c);
        printf("%6s | %-24s | %-24s\n","sel","standard","acorn-1(corrected)");
        for(double se:sels){ int cand=std::max(K,(int)(se*N));
            std::vector<std::unordered_set<idx_t>> tr(NQ); std::vector<SetSel> masks(NQ);
            for(int i=0;i<NQ;i++){masks[i]=mask(c,cand,i); tr[i]=brute(&Q[(size_t)i*d],masks[i]);}
            double bStdR=-1,bStdN=0,bStdL=0,bStdH=0,bAcR=-1,bAcN=0,bAcL=0,bAcX=0;
            std::vector<float> dd(K); std::vector<idx_t> ii(K);
            using clk=std::chrono::high_resolution_clock;
            for(int ef:EFS){
                // standard: faiss IndexIDMap::search with selector (mean latency over NQ)
                double sr=0,sn=0,sl=0,shop=0; int den=0;
                for(int i=0;i<NQ;i++){ faiss::SearchParametersHNSW p; p.efSearch=ef; p.sel=&masks[i];
                    faiss::hnsw_stats.reset();
                    auto t0=clk::now(); idmap.search(1,&Q[(size_t)i*d],K,dd.data(),ii.data(),&p); auto t1=clk::now();
                    sl+=std::chrono::duration<double,std::micro>(t1-t0).count();
                    sn+=faiss::hnsw_stats.ndis; shop+=faiss::hnsw_stats.nhops;
                    int h=0; for(int j=0;j<K;j++) if(ii[j]>=0&&tr[i].count(ii[j]))h++;
                    if(!tr[i].empty()){sr+=double(h)/tr[i].size();den++;} }
                double sR=den?sr/den:0; if(sR>bStdR+1e-9){bStdR=sR;bStdN=sn/NQ;bStdL=sl/NQ;bStdH=shop/NQ;}
                // acorn corrected
                double ar=0,an=0,al=0,ax=0; den=0;
                for(int i=0;i<NQ;i++){ auto t0=clk::now(); int r=knn_jni::acorn::search(&idmap,&Q[(size_t)i*d],K,ef,&masks[i],1,dd.data(),ii.data(),true); auto t1=clk::now();
                    al+=std::chrono::duration<double,std::micro>(t1-t0).count();
                    an+=knn_jni::acorn::last_stats().distance_computations;
                    ax+=knn_jni::acorn::last_stats().graph_neighbors_examined;
                    int h=0; for(int j=0;j<r;j++) if(tr[i].count(ii[j]))h++;
                    if(!tr[i].empty()){ar+=double(h)/tr[i].size();den++;} }
                double aR=den?ar/den:0; if(aR>bAcR+1e-9){bAcR=aR;bAcN=an/NQ;bAcL=al/NQ;bAcX=ax/NQ;}
            }
            // standard neighbours inspected ≈ hops × base degree (2*M = 32)
            printf("%5.1f%% | std: rec %.2f ndis %.0f examined~%.0f  %.0fus | acorn: rec %.2f ndis %.0f examined %.0f  %.0fus\n",
                   100.0*cand/N, bStdR,bStdN,bStdH*32,bStdL, bAcR,bAcN,bAcX,bAcL);
        }
    }
    fflush(stdout); std::_Exit(0);
}
