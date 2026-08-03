#include <cstdio>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <string>
#include <faiss/utils/random.h>
#include "acorn.h"
using namespace acorn; using clk=std::chrono::high_resolution_clock;
struct Sel: faiss::IDSelector{ const std::vector<char>* m; bool is_member(faiss::idx_t id)const override{return (*m)[id];} };
static int N,D,K,NC,NQ; static std::vector<float> X,CTR,Q; static std::vector<int> CL; static int FC;
static std::vector<char> mkfilter(const std::string& corr,int cand){
    std::vector<char> m(N,0);
    if(corr=="no"){faiss::RandomGenerator r(7);int p=0;while(p<cand){int id=r.rand_int(N);if(!m[id]){m[id]=1;p++;}}return m;}
    int center=(corr=="pos")?0:FC; std::vector<std::pair<float,int>> ds(N);
    for(int i=0;i<N;i++){float s=0;for(int j=0;j<D;j++){float df=CTR[(size_t)center*D+j]-X[(size_t)i*D+j];s+=df*df;}ds[i]={s,i};}
    std::partial_sort(ds.begin(),ds.begin()+cand,ds.end());for(int i=0;i<cand;i++)m[ds[i].second]=1;return m;
}
int main(){
    N=100000;D=128;K=10;NC=50;NQ=30; faiss::RandomGenerator rng(1);
    CTR.resize((size_t)NC*D);for(auto&v:CTR)v=rng.rand_float();
    X.resize((size_t)N*D);CL.resize(N);float sig=0.06f;
    for(int i=0;i<N;i++){int c=rng.rand_int(NC);CL[i]=c;for(int j=0;j<D;j++)X[(size_t)i*D+j]=CTR[(size_t)c*D+j]+sig*(rng.rand_float()*2-1);}
    Q.resize((size_t)NQ*D);for(int i=0;i<NQ;i++)for(int j=0;j<D;j++)Q[(size_t)i*D+j]=CTR[0*D+j]+sig*(rng.rand_float()*2-1);
    FC=1;float best=-1;for(int c=0;c<NC;c++){float s=0;for(int j=0;j<D;j++){float df=CTR[j]-CTR[(size_t)c*D+j];s+=df*df;}if(s>best){best=s;FC=c;}}
    AcornIndex A; build_standard_hnsw(A,D,faiss::METRIC_L2,16,100,N,X.data());
    printf("n=%d d=%d  faiss-native = CORRECTED (bounded, selector honored)\n",N,D);
    const char* corrs[3]={"neg","no","pos"}; int cands[8]={100,500,1000,2000,5000,10000,25000,50000};
    for(int ci=0;ci<3;ci++){ std::string corr=corrs[ci];
      printf("\n=== %s ===  sel | faiss-native today | HNSW-infilt | RACORN-1+ | exact  (r=recall nd=dist-comps ms)\n",corr.c_str());
      for(int cc=0;cc<8;cc++){ int cand=cands[cc]; std::vector<char> mask=mkfilter(corr,cand); Sel sel; sel.m=&mask;
        std::vector<std::unordered_set<faiss::idx_t>> tr(NQ); std::vector<faiss::idx_t> ii(K); std::vector<float> dd(K);
        for(int i=0;i<NQ;i++){int nr=exact_filtered_search(A,&Q[(size_t)i*D],K,&sel,ii.data(),dd.data());for(int j=0;j<nr;j++)tr[i].insert(ii[j]);}
        double R[4]={0,0,0,0},ND[4]={0,0,0,0},MS[4]={0,0,0,0}; int den=0;
        for(int i=0;i<NQ;i++){ if(!tr[i].empty())den++;
          for(int mode=0;mode<4;mode++){ SearchStats s; auto t0=clk::now(); int nr;
            if(mode==0)nr=filtered_search(A,&Q[(size_t)i*D],K,200,FilteredHnswSearchMode::STANDARD,&sel,ii.data(),dd.data(),&s);
            else if(mode==1)nr=hnsw_infilter_search(A,&Q[(size_t)i*D],K,200,&sel,ii.data(),dd.data(),&s);
            else if(mode==2){RacornParams p;p.bridge_ratio=1.0;p.enable_aef=true;p.aef_threshold=1500.0/N;nr=racorn_search(A,&Q[(size_t)i*D],K,200,&sel,p,ii.data(),dd.data(),&s);}
            else nr=exact_filtered_search(A,&Q[(size_t)i*D],K,&sel,ii.data(),dd.data(),&s);
            auto t1=clk::now(); MS[mode]+=std::chrono::duration<double,std::micro>(t1-t0).count(); ND[mode]+=s.dist_computations;
            if(!tr[i].empty()){int h=0;for(int j=0;j<nr;j++)if(tr[i].count(ii[j]))h++;R[mode]+=double(h)/tr[i].size();}}}
        printf("%5.1f%% | r=%.2f nd=%5.0f %5.1fms | r=%.2f nd=%6.0f %5.1fms | r=%.2f nd=%5.0f %5.1fms | r=%.2f nd=%5.0f %4.1fms\n",
          100.0*cand/N, R[0]/den,ND[0]/NQ,MS[0]/NQ/1000, R[1]/den,ND[1]/NQ,MS[1]/NQ/1000, R[2]/den,ND[2]/NQ,MS[2]/NQ/1000, R[3]/den,ND[3]/NQ,MS[3]/NQ/1000);
      }}
    return 0;
}
