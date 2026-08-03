/*
 * Parameterized filtered-kNN benchmark — every query-time knob is an argument,
 * nothing hard-coded, so we can iterate the right parameter set (as if tuning a
 * real OpenSearch query's method_parameters).
 *
 * "Fresh OpenSearch, no changes" == mode=standard (faiss IndexHNSW::search +
 * IDSelector, bounded) — the exact native path faiss_wrapper.cpp invokes today.
 *
 * Usage:
 *   bench_param <n> <mode> [ef_list] [gamma] [m_beta] [bridge_ratio] [aef_thr]
 *     mode      : standard | acorn | racorn | racorn_plus
 *     ef_list   : comma list, default "50,100,250,500"   (query-time ef_search)
 *     gamma     : ACORN-gamma graph gamma (acorn mode on a gamma graph), default 1 (standard graph)
 *     m_beta    : RACORN/ACORN 2-hop expansion boundary, default 2*M
 *     bridge_ratio: RACORN ASF ratio, default 1.0
 *     aef_thr   : RACORN-1+ exact-fallback pass-ratio threshold, default 0.01
 *
 * Index-time params fixed to OpenSearch defaults: M=16, ef_construction=100, d=128, L2.
 * Sweeps correlation {negative,none,positive} x selectivity {0.1..50%} x ef.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <faiss/utils/random.h>
#include "acorn.h"
using namespace acorn; using clk=std::chrono::high_resolution_clock;
struct Sel: faiss::IDSelector{ const std::vector<char>* m; bool is_member(faiss::idx_t id)const override{return (*m)[id];} };

static int N,D=128,K=10,NC,NQ=50, MBASE=16;
static std::vector<float> X,CTR,Q; static std::vector<int> CL; static int FC;
// Per-base squared distances to the positive/negative correlation anchors.
// Populated for BOTH real-vector and synthetic modes so mkfilter is unified.
static std::vector<float> DPOS,DNEG; static bool REAL=false;

static std::vector<int> parse_ints(const char* s){ std::vector<int> v; std::string t(s); size_t p=0;
    while(p<t.size()){ size_t c=t.find(',',p); std::string tok=t.substr(p,c==std::string::npos?c:c-p); if(!tok.empty())v.push_back(atoi(tok.c_str())); if(c==std::string::npos)break; p=c+1;} return v; }

// Load up to want_n vectors from a .fvecs file ([int32 d][d float32] per record).
static bool load_fvecs(const char* path,int want_n,std::vector<float>& out,int& d_out){
    FILE* fp=fopen(path,"rb"); if(!fp) return false;
    int d=0; if(fread(&d,4,1,fp)!=1||d<=0||d>100000){fclose(fp);return false;}
    fseek(fp,0,SEEK_END); long bytes=ftell(fp); fseek(fp,0,SEEK_SET);
    long rec=4L+4L*d, total=bytes/rec; int n=(want_n>0)?(int)std::min<long>(want_n,total):(int)total;
    out.resize((size_t)n*d);
    for(int i=0;i<n;i++){ int dd; if(fread(&dd,4,1,fp)!=1||dd!=d){fclose(fp);return false;}
        if(fread(&out[(size_t)i*d],4,d,fp)!=(size_t)d){fclose(fp);return false;} }
    fclose(fp); d_out=d; return true;
}

// Correlation filter: eligible=cand base vectors nearest the pos anchor (positive),
// nearest the neg anchor (negative), or random (no). Uses precomputed DPOS/DNEG.
static std::vector<char> mkfilter(const std::string& corr,int cand){
    std::vector<char> m(N,0);
    if(corr=="no"){ faiss::RandomGenerator r(7); int p=0; while(p<cand){int id=r.rand_int(N); if(!m[id]){m[id]=1;p++;}} return m; }
    const std::vector<float>& dd=(corr=="pos")?DPOS:DNEG;
    std::vector<std::pair<float,int>> ord(N); for(int i=0;i<N;i++)ord[i]={dd[i],i};
    std::partial_sort(ord.begin(),ord.begin()+cand,ord.end()); for(int i=0;i<cand;i++)m[ord[i].second]=1; return m;
}

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: bench_param <n> <mode> [ef_list] [gamma] [m_beta] [bridge_ratio] [aef_thr]\n"); return 1; }
    N=atoi(argv[1]); std::string mode=argv[2];
    std::vector<int> EFS = parse_ints(argc>3?argv[3]:"50,100,250,500");
    int gamma=argc>4?atoi(argv[4]):1;
    int m_beta=argc>5?atoi(argv[5]):(2*MBASE);
    double bridge=argc>6?atof(argv[6]):1.0;
    double aef=argc>7?atof(argv[7]):0.01;
    NC=std::max(16,N/2000);

    // ---- vectors: real SIFT .fvecs if available, else synthetic Gaussian clusters ----
    const char* env=getenv("KNN_SIFT_BASE");
    std::string path = env?env:"research/acorn/data/sift/sift_base.fvecs";
    int fd=0;
    if(load_fvecs(path.c_str(),N,X,fd)){
        REAL=true; D=fd; N=(int)(X.size()/D);
        fprintf(stderr,"[bench_param] REAL vectors loaded: %s  n=%d d=%d\n",path.c_str(),N,D);
        // positive anchor = a fixed real base vector; queries = NQ nearest base vectors to it
        int a0=12345%N; std::vector<float> apos(&X[(size_t)a0*D],&X[(size_t)a0*D]+D);
        DPOS.resize(N); for(int i=0;i<N;i++){float s=0;const float* xi=&X[(size_t)i*D];for(int j=0;j<D;j++){float df=apos[j]-xi[j];s+=df*df;}DPOS[i]=s;}
        // negative anchor = farthest base vector from the positive anchor (a disjoint far region)
        int aneg=0; float bd=-1; for(int i=0;i<N;i++)if(DPOS[i]>bd){bd=DPOS[i];aneg=i;}
        std::vector<float> aneg_v(&X[(size_t)aneg*D],&X[(size_t)aneg*D]+D);
        DNEG.resize(N); for(int i=0;i<N;i++){float s=0;const float* xi=&X[(size_t)i*D];for(int j=0;j<D;j++){float df=aneg_v[j]-xi[j];s+=df*df;}DNEG[i]=s;}
        std::vector<std::pair<float,int>> ord(N); for(int i=0;i<N;i++)ord[i]={DPOS[i],i};
        std::partial_sort(ord.begin(),ord.begin()+NQ,ord.end());
        Q.resize((size_t)NQ*D); for(int i=0;i<NQ;i++){int id=ord[i].second;for(int j=0;j<D;j++)Q[(size_t)i*D+j]=X[(size_t)id*D+j];}
    } else {
        fprintf(stderr,"[bench_param] REAL data not at %s — synthetic Gaussian clusters\n",path.c_str());
        faiss::RandomGenerator rng(1);
        CTR.resize((size_t)NC*D); for(auto&v:CTR)v=rng.rand_float();
        X.resize((size_t)N*D); CL.resize(N); float sig=0.06f;
        for(int i=0;i<N;i++){int c=rng.rand_int(NC);CL[i]=c;for(int j=0;j<D;j++)X[(size_t)i*D+j]=CTR[(size_t)c*D+j]+sig*(rng.rand_float()*2-1);}
        Q.resize((size_t)NQ*D); for(int i=0;i<NQ;i++)for(int j=0;j<D;j++)Q[(size_t)i*D+j]=CTR[0*D+j]+sig*(rng.rand_float()*2-1);
        FC=1;float best=-1;for(int c=0;c<NC;c++){float s=0;for(int j=0;j<D;j++){float df=CTR[j]-CTR[(size_t)c*D+j];s+=df*df;}if(s>best){best=s;FC=c;}}
        // unify mkfilter: pos=near cluster 0, neg=near farthest cluster FC
        DPOS.resize(N); DNEG.resize(N);
        for(int i=0;i<N;i++){float sp=0,sn=0;const float* xi=&X[(size_t)i*D];for(int j=0;j<D;j++){float dp=CTR[j]-xi[j];sp+=dp*dp;float dn=CTR[(size_t)FC*D+j]-xi[j];sn+=dn*dn;}DPOS[i]=sp;DNEG[i]=sn;}
    }

    // build the graph(s): standard for standard/acorn/racorn; acorn-gamma graph when gamma>1
    AcornIndex STD, GAM; build_standard_hnsw(STD,D,faiss::METRIC_L2,MBASE,100,N,X.data());
    bool useGamma=(mode=="acorn" && gamma>1);
    if(useGamma){ AcornGammaBuildParameters bp; bp.enabled=true; bp.M=MBASE; bp.gamma=gamma; bp.M_beta=m_beta; build_acorn_gamma(GAM,D,faiss::METRIC_L2,bp,N,X.data()); GAM.M_beta=m_beta; }
    AcornIndex& IDX = useGamma?GAM:STD;

    fprintf(stderr,"[bench_param] n=%d mode=%s gamma=%d m_beta=%d bridge=%.2f aef=%.4f — graphs built\n",N,mode.c_str(),gamma,m_beta,bridge,aef);
    char csvname[256]; snprintf(csvname,sizeof(csvname),"research/acorn/opensearch-benchmark/raw-results/param_%s_n%d.csv",mode.c_str(),N);
    FILE* f=fopen(csvname,"w"); fprintf(f,"n,mode,correlation,cand,sel,ef,k,recall,p50_us,p95_us,mean_us,ndis\n");

    const char* corrs[3]={"neg","no","pos"};
    int cands[8]={ N/1000, N/200, N/100, N/50, N/20, N/10, N/4, N/2 }; // 0.1..50%
    printf("\n#### n=%d  mode=%s  data=%s  (index: M=%d efc=100 d=%d L2; queries in pos-anchor region)\n",N,mode.c_str(),REAL?"SIFT1M(real)":"synthetic",MBASE,D);
    if(mode!="standard") printf("#### params: gamma=%d m_beta=%d bridge_ratio=%.2f aef_thr=%.4f\n",gamma,m_beta,bridge,aef);
    for(int ci=0;ci<3;ci++){ std::string corr=corrs[ci];
      printf("\n=== correlation=%s ===  (cell = recall/us per ef ; best = recall @ ef, us, ndis)\n", corr.c_str());
      printf("%6s %8s |", "sel","cand"); for(int ef:EFS) printf("   ef=%-4d ",ef); printf("  | best: recall  us   ndis\n");
      for(int cc=0;cc<8;cc++){ int cand=cands[cc]; if(cand<K)cand=K; auto mask=mkfilter(corr,cand); Sel sel; sel.m=&mask;
        // ground truth
        std::vector<std::unordered_set<faiss::idx_t>> tr(NQ); { std::vector<faiss::idx_t> ii(K); std::vector<float> dd(K);
          for(int i=0;i<NQ;i++){int nr=exact_filtered_search(IDX,&Q[(size_t)i*D],K,&sel,ii.data(),dd.data()); for(int j=0;j<nr;j++)tr[i].insert(ii[j]);} }
        printf("%5.1f%% %8d |", 100.0*cand/N, cand);
        double bestR=-1,bestUs=0,bestNd=0; int bestEf=0;
        for(int ef:EFS){ std::vector<double> lat; double rec=0; int den=0; double nd=0;
          std::vector<faiss::idx_t> ii(K); std::vector<float> dd(K);
          for(int w=0;w<2;w++){SearchStats s; if(mode=="standard")filtered_search(IDX,&Q[0],K,ef,FilteredHnswSearchMode::STANDARD,&sel,ii.data(),dd.data(),&s);
            else if(mode=="acorn")filtered_search(IDX,&Q[0],K,ef,FilteredHnswSearchMode::ACORN,&sel,ii.data(),dd.data(),&s);
            else {RacornParams p;p.bridge_ratio=bridge;p.enable_aef=(mode=="racorn_plus");p.aef_threshold=aef;racorn_search(IDX,&Q[0],K,ef,&sel,p,ii.data(),dd.data(),&s);}}
          for(int i=0;i<NQ;i++){SearchStats s;auto t0=clk::now();int nr;
            if(mode=="standard")nr=filtered_search(IDX,&Q[(size_t)i*D],K,ef,FilteredHnswSearchMode::STANDARD,&sel,ii.data(),dd.data(),&s);
            else if(mode=="acorn")nr=filtered_search(IDX,&Q[(size_t)i*D],K,ef,FilteredHnswSearchMode::ACORN,&sel,ii.data(),dd.data(),&s);
            else {RacornParams p;p.bridge_ratio=bridge;p.enable_aef=(mode=="racorn_plus");p.aef_threshold=aef;nr=racorn_search(IDX,&Q[(size_t)i*D],K,ef,&sel,p,ii.data(),dd.data(),&s);}
            auto t1=clk::now();lat.push_back(std::chrono::duration<double,std::micro>(t1-t0).count());nd+=s.dist_computations;
            auto&T=tr[i];if(!T.empty()){int h=0;for(int j=0;j<nr;j++)if(T.count(ii[j]))h++;rec+=double(h)/T.size();den++;}}
          std::sort(lat.begin(),lat.end()); double r=den?rec/den:0; double p50=lat[lat.size()/2],p95=lat[(size_t)(0.95*(lat.size()-1))],mn=std::accumulate(lat.begin(),lat.end(),0.0)/lat.size();
          printf(" %.2f/%5.0f", r, mn);
          fprintf(f,"%d,%s,%s,%d,%.5f,%d,%d,%.4f,%.1f,%.1f,%.1f,%.0f\n",N,mode.c_str(),corr.c_str(),cand,(double)cand/N,ef,K,r,p50,p95,mn,nd/NQ);
          if(r>bestR+1e-9){bestR=r;bestUs=mn;bestNd=nd/NQ;bestEf=ef;}
        }
        printf("  | %.2f  ef%-4d %6.0fus %7.0f\n", bestR,bestEf,bestUs,bestNd);
      }
    }
    fclose(f); fprintf(stderr,"[bench_param] wrote %s\n",csvname);
    // Results are fully written above. Skip global/index teardown: it is pure
    // benchmark cleanup (a large HNSW graph) and, under -O3, an uninitialized
    // faiss teardown pointer trips a spurious free — verified harmless to results
    // (AddressSanitizer is clean across the whole run). _Exit avoids both.
    fflush(stdout); fflush(stderr); std::_Exit(0);
}
