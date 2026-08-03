/*
 * Collective ACORN / ACORN-γ / RACORN-1 / RACORN-1+ benchmark (native path).
 *
 * Strategies (all filtered top-k over the SAME data/filters/queries):
 *   H   HNSW In-filtering (distance-first, recall-preserving)      [paper HNSW]
 *   Hl  HNSW In-filtering, larger M                                 [denser graph]
 *   Fx  faiss-native filtered HNSW (STANDARD mode)                  [today's OpenSearch path]
 *   A1  ACORN-1 (faithful, RACORN BR=0)                             [collapses at low sel/neg]
 *   Ag  ACORN-γ graph + ACORN traversal                            [graph densification]
 *   R1  RACORN-1 (ASF, BR=1.0)
 *   R1q RACORN-1 (ASF, BR=0.25)                                     [neg-correlation frontier]
 *   R1p RACORN-1+ (RACORN-1 + Adaptive Exact Fallback)
 *   Ex  exact filtered search                                      [oracle / crossover]
 *
 * Correlation conditions (paper §7.5): queries drawn near cluster 0.
 *   no   random filter (query-independent)
 *   pos  valid = nearest to cluster 0 center (near query)
 *   neg  valid = nearest to the farthest cluster center (far from query)
 *
 * Usage: bench_collective <n> <d> <l2|ip> <nq> <seeds> <out.csv> [tag]
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#include <faiss/utils/random.h>
#include "acorn.h"

using namespace acorn;
using clk = std::chrono::high_resolution_clock;

struct Data { int n,d,nc,nq; std::vector<float> x,centers,queries; std::vector<int> cluster; int qcluster, farcluster; };

static Data gen(int n,int d,int nq,int seed,bool norm){
    Data ds; ds.n=n; ds.d=d; ds.nq=nq; ds.nc=std::max(16,n/2000);
    faiss::RandomGenerator rng(seed);
    ds.centers.resize((size_t)ds.nc*d); for(auto&v:ds.centers)v=rng.rand_float();
    ds.x.resize((size_t)n*d); ds.cluster.resize(n); float sigma=0.06f;
    for(int i=0;i<n;i++){int c=rng.rand_int(ds.nc); ds.cluster[i]=c; for(int j=0;j<d;j++) ds.x[(size_t)i*d+j]=ds.centers[(size_t)c*d+j]+sigma*(rng.rand_float()*2-1);}
    ds.qcluster=0;
    // farthest cluster from qcluster
    ds.farcluster=1; float best=-1;
    for(int c=0;c<ds.nc;c++){ float s=0; for(int j=0;j<d;j++){float df=ds.centers[j]-ds.centers[(size_t)c*d+j]; s+=df*df;} if(s>best){best=s;ds.farcluster=c;} }
    ds.queries.resize((size_t)nq*d);
    for(int i=0;i<nq;i++){ for(int j=0;j<d;j++) ds.queries[(size_t)i*d+j]=ds.centers[(size_t)ds.qcluster*d+j]+sigma*(rng.rand_float()*2-1); }
    if(norm){ auto nm=[&](std::vector<float>&v){ for(size_t i=0;i+d<=v.size();i+=d){ double s=0; for(int j=0;j<d;j++) s+=(double)v[i+j]*v[i+j]; float inv=s>0?1.0f/std::sqrt((float)s):0; for(int j=0;j<d;j++) v[i+j]*=inv; } }; nm(ds.x); nm(ds.queries); }
    return ds;
}

// build a filter of exactly `cand` docs under a correlation mode
static std::vector<char> make_filter(const Data& ds, const std::string& mode, int cand, int seed){
    int n=ds.n,d=ds.d; faiss::RandomGenerator rng(seed);
    std::vector<char> keep(n,0);
    if(mode=="no"){ int placed=0; while(placed<cand){int id=rng.rand_int(n); if(!keep[id]){keep[id]=1;placed++;}} return keep; }
    int center = (mode=="pos") ? ds.qcluster : ds.farcluster;
    std::vector<std::pair<float,int>> dist(n);
    for(int i=0;i<n;i++){ float s=0; for(int j=0;j<d;j++){float df=ds.centers[(size_t)center*d+j]-ds.x[(size_t)i*d+j]; s+=df*df;} dist[i]={s,i}; }
    std::partial_sort(dist.begin(), dist.begin()+std::min(n,cand), dist.end());
    for(int i=0;i<cand;i++) keep[dist[i].second]=1;
    return keep;
}

struct Sel: faiss::IDSelector { const std::vector<char>* m; bool is_member(faiss::idx_t id) const override { return (*m)[id]; } };

static double pct(std::vector<double> v,double p){ if(v.empty())return 0; std::sort(v.begin(),v.end()); return v[std::min(v.size()-1,(size_t)std::llround(p/100.0*(v.size()-1)))]; }
static double meanv(const std::vector<double>&v){ return v.empty()?0:std::accumulate(v.begin(),v.end(),0.0)/v.size(); }
static double sd(const std::vector<double>&v){ if(v.size()<2)return 0; double m=meanv(v),s=0; for(double x:v)s+=(x-m)*(x-m); return std::sqrt(s/(v.size()-1)); }

struct R { double recall,p50,p95,p99,mean; double ndis,bridges,aef; };

template<class Fn>
static R run(const Data& ds, const std::vector<char>& mask, int k,
             const std::vector<std::unordered_set<faiss::idx_t>>& truth, Fn fn){
    std::vector<double> lat; double rec=0; int den=0; SearchStats agg;
    std::vector<faiss::idx_t> ids(k); std::vector<float> dd(k);
    for(int w=0;w<3&&w<ds.nq;w++) fn(&ds.queries[(size_t)w*ds.d],ids.data(),dd.data(),nullptr);
    for(int q=0;q<ds.nq;q++){ SearchStats st; auto t0=clk::now(); int nr=fn(&ds.queries[(size_t)q*ds.d],ids.data(),dd.data(),&st); auto t1=clk::now();
        lat.push_back(std::chrono::duration<double,std::micro>(t1-t0).count()); agg.add(st);
        auto&tr=truth[q]; if(!tr.empty()){int h=0;for(int i=0;i<nr;i++)if(tr.count(ids[i]))h++; rec+=double(h)/tr.size(); den++;} }
    return R{ den?rec/den:0, pct(lat,50),pct(lat,95),pct(lat,99),meanv(lat),
              (double)agg.dist_computations/ds.nq,(double)agg.bridges_used/ds.nq,(double)agg.aef_switches };
}

int main(int argc,char**argv){
    if(argc<7){ fprintf(stderr,"usage: bench_collective <n> <d> <l2|ip> <nq> <seeds> <out.csv> [tag]\n"); return 1; }
    int n=atoi(argv[1]),d=atoi(argv[2]); std::string ms=argv[3]; int nq=atoi(argv[4]); int seeds=atoi(argv[5]);
    const char* out=argv[6]; const char* tag=argc>7?argv[7]:"col";
    faiss::MetricType metric=(ms=="ip")?faiss::METRIC_INNER_PRODUCT:faiss::METRIC_L2; bool norm=(metric==faiss::METRIC_INNER_PRODUCT);
    const int k=10, Mbase=16, Mlarge=32, gamma=8;

    // AEF threshold: switch to exact when running pass-ratio implies < ~1500 eligible.
    double aef_thr_for = 1500.0; // absolute eligible floor; converted to ratio per n below

    FILE* f=fopen(out,"w");
    fprintf(f,"tag,n,d,metric,correlation,cand,sel,strategy,k,ef,recall_med,recall_sd,p50_med,p95_med,p95_sd,p99_med,mean_us_med,ndis_med,bridges_med,aef_med,seeds\n");

    std::vector<int> cands={100,250,500,1000,2000,5000,10000,25000,50000};
    std::vector<std::string> corrs={"no","pos","neg"};
    std::vector<int> efs={50,100,200,400};

    // build per-seed graphs once
    struct G{ Data ds; AcornIndex std16, std32, gam; };
    std::vector<G> GS(seeds);
    for(int s=0;s<seeds;s++){
        GS[s].ds=gen(n,d,nq,1000+s,norm);
        build_standard_hnsw(GS[s].std16,d,metric,Mbase,100,n,GS[s].ds.x.data(),12345+s);
        build_standard_hnsw(GS[s].std32,d,metric,Mlarge,100,n,GS[s].ds.x.data(),12345+s);
        AcornGammaBuildParameters bp; bp.enabled=true; bp.M=Mbase; bp.gamma=gamma; bp.M_beta=Mbase; bp.seed=12345+s;
        build_acorn_gamma(GS[s].gam,d,metric,bp,n,GS[s].ds.x.data());
        fprintf(stderr,"[collective] seed %d graphs built\n",s);
    }

    auto emit=[&](const char* corr,int cand,const char* strat,int ef,
                  std::vector<double>&rec,std::vector<double>&p50,std::vector<double>&p95,std::vector<double>&p99,
                  std::vector<double>&mu,std::vector<double>&nd,std::vector<double>&br,std::vector<double>&ae){
        fprintf(f,"%s,%d,%d,%s,%s,%d,%.5f,%s,%d,%d,%.5f,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%.2f,%d\n",
            tag,n,d,ms.c_str(),corr,cand,(double)cand/n,strat,k,ef,
            pct(rec,50),sd(rec),pct(p50,50),pct(p95,50),sd(p95),pct(p99,50),pct(mu,50),pct(nd,50),pct(br,50),pct(ae,50),seeds);
    };

    for(auto&corr:corrs){
      for(int cand:cands){ if(cand>n) continue;
        double aef_ratio = aef_thr_for/n;
        // Build masks + ground truth ONCE per (corr,cand,seed); reuse everywhere.
        std::vector<std::vector<char>> masks(seeds);
        std::vector<std::vector<std::unordered_set<faiss::idx_t>>> truths(seeds);
        for(int s=0;s<seeds;s++){
            masks[s]=make_filter(GS[s].ds,corr,cand,777+s);
            Sel sel; sel.m=&masks[s];
            truths[s].assign(nq,{});
            std::vector<faiss::idx_t> ids(k); std::vector<float> dd(k);
            for(int q=0;q<nq;q++){int nr=exact_filtered_search(GS[s].std16,&GS[s].ds.queries[(size_t)q*d],k,&sel,ids.data(),dd.data()); for(int i=0;i<nr;i++)truths[s][q].insert(ids[i]);}
        }
        // exact
        { std::vector<double> rec,p50,p95,p99,mu,nd,br,ae;
          for(int s=0;s<seeds;s++){ Sel sel; sel.m=&masks[s];
            auto r=run(GS[s].ds,masks[s],k,truths[s],[&](const float*qq,faiss::idx_t*I,float*D,SearchStats*st){return exact_filtered_search(GS[s].std16,qq,k,&sel,I,D,st);});
            rec.push_back(r.recall);p50.push_back(r.p50);p95.push_back(r.p95);p99.push_back(r.p99);mu.push_back(r.mean);nd.push_back(r.ndis);br.push_back(r.bridges);ae.push_back(r.aef);
          }
          emit(corr.c_str(),cand,"Ex_exact",0,rec,p50,p95,p99,mu,nd,br,ae);
        }
        for(int ef:efs){
          auto sweep=[&](const char* name, auto call){
            std::vector<double> rec,p50,p95,p99,mu,nd,br,ae;
            for(int s=0;s<seeds;s++){ Sel sel; sel.m=&masks[s];
              auto r=run(GS[s].ds,masks[s],k,truths[s],[&](const float*qq,faiss::idx_t*I,float*D,SearchStats*st){return call(s,&sel,qq,ef,I,D,st);});
              rec.push_back(r.recall);p50.push_back(r.p50);p95.push_back(r.p95);p99.push_back(r.p99);mu.push_back(r.mean);nd.push_back(r.ndis);br.push_back(r.bridges);ae.push_back(r.aef);
            }
            emit(corr.c_str(),cand,name,ef,rec,p50,p95,p99,mu,nd,br,ae);
          };
          sweep("H_hnsw_infilt",   [&](int s,const Sel*sel,const float*qq,int ef,faiss::idx_t*I,float*D,SearchStats*st){return hnsw_infilter_search(GS[s].std16,qq,k,ef,sel,I,D,st);});
          sweep("Hl_hnsw_large",   [&](int s,const Sel*sel,const float*qq,int ef,faiss::idx_t*I,float*D,SearchStats*st){return hnsw_infilter_search(GS[s].std32,qq,k,ef,sel,I,D,st);});
          sweep("Fx_faiss_std",    [&](int s,const Sel*sel,const float*qq,int ef,faiss::idx_t*I,float*D,SearchStats*st){return filtered_search(GS[s].std16,qq,k,ef,FilteredHnswSearchMode::STANDARD,sel,I,D,st);});
          sweep("A1_acorn1",       [&](int s,const Sel*sel,const float*qq,int ef,faiss::idx_t*I,float*D,SearchStats*st){RacornParams p;p.bridge_ratio=0.0;return racorn_search(GS[s].std16,qq,k,ef,sel,p,I,D,st);});
          sweep("Ag_acorn_gamma",  [&](int s,const Sel*sel,const float*qq,int ef,faiss::idx_t*I,float*D,SearchStats*st){return filtered_search(GS[s].gam,qq,k,ef,FilteredHnswSearchMode::ACORN,sel,I,D,st);});
          sweep("R1_racorn1",      [&](int s,const Sel*sel,const float*qq,int ef,faiss::idx_t*I,float*D,SearchStats*st){RacornParams p;p.bridge_ratio=1.0;return racorn_search(GS[s].std16,qq,k,ef,sel,p,I,D,st);});
          sweep("R1q_racorn1_br25",[&](int s,const Sel*sel,const float*qq,int ef,faiss::idx_t*I,float*D,SearchStats*st){RacornParams p;p.bridge_ratio=0.25;return racorn_search(GS[s].std16,qq,k,ef,sel,p,I,D,st);});
          sweep("R1p_racorn1plus", [&](int s,const Sel*sel,const float*qq,int ef,faiss::idx_t*I,float*D,SearchStats*st){RacornParams p;p.bridge_ratio=1.0;p.enable_aef=true;p.aef_threshold=aef_ratio;return racorn_search(GS[s].std16,qq,k,ef,sel,p,I,D,st);});
        }
      }
      fprintf(stderr,"[collective] correlation=%s done\n",corr.c_str());
    }
    fclose(f); fprintf(stderr,"[collective] wrote %s\n",out); return 0;
}
