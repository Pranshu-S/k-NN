/*
 * Phase 2 prototype: filter-sketch partition pruning vs full exact fallback.
 * Partition vectors into P IVF-style cells (centroids). Per-cell "sketch" = which docs are
 * eligible (filter INTERSECT cell). For a query, rank cells by centroid distance, SKIP cells
 * with 0 eligible docs, probe the nearest `nprobe` eligible-containing cells, and compute
 * distances only for eligible docs in those cells. Compares recall / latency / #distances vs
 * full exact (scan ALL eligible). Focus: negative correlation, where eligible docs cluster
 * far from the query and the near cells are empty of eligible.
 * Usage: bench_partition [n] [P]   (default 100000 512)
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <faiss/utils/random.h>
using idx_t=long;
static int D=128; static const float* XG=nullptr;
static bool load_fvecs(const char*p,int want,std::vector<float>&o,int&d){FILE*f=fopen(p,"rb");if(!f)return false;int dd=0;if(fread(&dd,4,1,f)!=1){fclose(f);return false;}fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);long rec=4+4L*dd,tot=b/rec;int n=want>0?std::min<long>(want,tot):tot;o.resize((size_t)n*dd);for(int i=0;i<n;i++){int t;if(fread(&t,4,1,f)!=1||t!=dd){fclose(f);return false;}if(fread(&o[(size_t)i*dd],4,dd,f)!=(size_t)dd){fclose(f);return false;}}fclose(f);d=dd;return true;}
static inline float l2(const float*a,const float*b){float s=0;for(int j=0;j<D;j++){float df=a[j]-b[j];s+=df*df;}return s;}
static inline float l2v(const float*q,int v){return l2(q,&XG[(size_t)v*D]);}

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):100000, P=argc>2?atoi(argv[2]):512; const int K=10,NQ=50;
    std::vector<float> X;int d=128;if(!load_fvecs("research/acorn/data/sift/sift_base.fvecs",N,X,d)){fprintf(stderr,"need sift\n");return 1;}N=(int)(X.size()/d);D=d;XG=X.data();
    std::vector<float> Q;int qd=0;load_fvecs("research/acorn/data/sift/sift_query.fvecs",NQ,Q,qd);
    // ---- build IVF cells: P random centroids + assign (1 Lloyd refine) ----
    faiss::RandomGenerator rng(7); std::vector<float> C((size_t)P*d);
    for(int c=0;c<P;c++){int r=rng.rand_int(N);for(int j=0;j<d;j++)C[(size_t)c*d+j]=X[(size_t)r*d+j];}
    std::vector<int> assign(N);
    for(int it=0;it<2;it++){
        for(int i=0;i<N;i++){float best=1e30f;int bc=0;for(int c=0;c<P;c++){float dd=l2(&X[(size_t)i*d],&C[(size_t)c*d]);if(dd<best){best=dd;bc=c;}}assign[i]=bc;}
        std::vector<double> acc((size_t)P*d,0);std::vector<int> cnt(P,0);
        for(int i=0;i<N;i++){int c=assign[i];cnt[c]++;for(int j=0;j<d;j++)acc[(size_t)c*d+j]+=X[(size_t)i*d+j];}
        for(int c=0;c<P;c++)if(cnt[c])for(int j=0;j<d;j++)C[(size_t)c*d+j]=acc[(size_t)c*d+j]/cnt[c];
    }
    std::vector<std::vector<int>> cell(P); for(int i=0;i<N;i++)cell[assign[i]].push_back(i);
    fprintf(stderr,"[bench_partition] n=%d P=%d built (avg cell=%d)\n",N,P,N/P);
    // per-query cell order
    std::vector<std::vector<int>> ord(NQ);
    for(int i=0;i<NQ;i++){std::vector<std::pair<float,int>> ds(N);const float*q=&Q[(size_t)i*d];for(int b=0;b<N;b++)ds[b]={l2v(q,b),b};std::sort(ds.begin(),ds.end());ord[i].resize(N);for(int b=0;b<N;b++)ord[i][b]=ds[b].second;}
    auto mask=[&](const char*c,int cand,int qi){std::vector<char> s(N,0);std::string cc(c);
        if(cc=="no"){faiss::RandomGenerator r(1000+qi);int p=0;while(p<cand){int id=r.rand_int(N);if(!s[id]){s[id]=1;p++;}}}
        else if(cc=="pos"){for(int j=0;j<cand;j++)s[ord[qi][j]]=1;}
        else if(cc=="fc"){ // compact far cluster: eligible = cand nearest to a FAR anchor (realistic "category=X far in vector space")
            const float* anchor=&XG[(size_t)ord[qi][N-1]*D]; std::vector<std::pair<float,int>> ds(N);
            for(int b=0;b<N;b++)ds[b]={l2(anchor,&XG[(size_t)b*D]),b}; std::partial_sort(ds.begin(),ds.begin()+cand,ds.end());
            for(int j=0;j<cand;j++)s[ds[j].second]=1; }
        else{for(int j=0;j<cand;j++)s[ord[qi][N-1-j]]=1;} // "neg" = diffuse farthest shell (pathological)
        return s;};
    auto truth=[&](const float*q,const std::vector<char>&m){std::vector<std::pair<float,int>> sc;for(int i=0;i<N;i++)if(m[i])sc.push_back({l2v(q,i),i});std::sort(sc.begin(),sc.end());std::unordered_set<int> t;for(int i=0;i<K&&i<(int)sc.size();i++)t.insert(sc[i].second);return t;};
    using clk=std::chrono::high_resolution_clock;
    // full exact over eligible
    auto exact=[&](const float*q,const std::vector<char>&m,uint64_t&dis){std::vector<std::pair<float,int>> sc;for(int i=0;i<N;i++)if(m[i]){sc.push_back({l2v(q,i),i});dis++;}std::partial_sort(sc.begin(),sc.begin()+std::min((int)sc.size(),K),sc.end());std::unordered_set<int> r;for(int i=0;i<K&&i<(int)sc.size();i++)r.insert(sc[i].second);return r;};
    // partition-pruned: probe nearest nprobe eligible-containing cells
    auto pruned=[&](const float*q,const std::vector<char>&m,int nprobe,uint64_t&dis){
        std::vector<std::pair<float,int>> co(P);for(int c=0;c<P;c++)co[c]={l2(q,&C[(size_t)c*d]),c};std::sort(co.begin(),co.end());
        std::vector<std::pair<float,int>> best; int probed=0;
        for(int t=0;t<P&&probed<nprobe;t++){int c=co[t].second; bool any=false;
            for(int v:cell[c]) if(m[v]){best.push_back({l2v(q,v),v});dis++;any=true;}
            if(any)probed++;                                   // count only eligible-containing cells
        }
        std::partial_sort(best.begin(),best.begin()+std::min((int)best.size(),K),best.end());
        std::unordered_set<int> r;for(int i=0;i<K&&i<(int)best.size();i++)r.insert(best[i].second);return r;
    };
    auto rec=[&](const std::unordered_set<int>&got,const std::unordered_set<int>&tr){if(tr.empty())return 1.0;int h=0;for(int id:got)if(tr.count(id))h++;return double(h)/tr.size();};
    int NPROBE[4]={1,4,16,64}; double sels[4]={0.01,0.05,0.10,0.25};
    for(const char* corr:{"fc","neg","no"}){
        printf("\n########## correlation=%s : full-exact vs partition-pruned (rec / latency-us / #dist) ##########\n",corr);
        printf("%6s | %-22s",corr,"full exact"); for(int np:NPROBE)printf(" | pruned np=%-3d",np); printf("\n");
        for(double se:sels){int cand=std::max(K,(int)(se*N));
            std::vector<std::vector<char>> M(NQ);std::vector<std::unordered_set<int>> TR(NQ);for(int i=0;i<NQ;i++){M[i]=mask(corr,cand,i);TR[i]=truth(&Q[(size_t)i*d],M[i]);}
            // full exact
            {double r=0,l=0,dc=0;int den=0;for(int i=0;i<NQ;i++){uint64_t dis=0;auto t0=clk::now();auto g=exact(&Q[(size_t)i*d],M[i],dis);auto t1=clk::now();l+=std::chrono::duration<double,std::micro>(t1-t0).count();dc+=dis;r+=rec(g,TR[i]);den++;}
             printf("%5.1f%% | %.2f/%6.0f/%7.0f",100.0*cand/N,r/den,l/NQ,dc/NQ);}
            for(int np:NPROBE){double r=0,l=0,dc=0;int den=0;for(int i=0;i<NQ;i++){uint64_t dis=0;auto t0=clk::now();auto g=pruned(&Q[(size_t)i*d],M[i],np,dis);auto t1=clk::now();l+=std::chrono::duration<double,std::micro>(t1-t0).count();dc+=dis;r+=rec(g,TR[i]);den++;}
             printf(" | %.2f/%5.0f/%6.0f",r/den,l/NQ,dc/NQ);}
            printf("\n");
        }
    }
    fflush(stdout); std::_Exit(0);
}
