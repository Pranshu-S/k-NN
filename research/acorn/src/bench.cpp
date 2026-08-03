/*
 * ACORN-γ benchmark harness. Builds graphs once per (n,d,metric), then sweeps
 * filter types x selectivities x efSearch x k over baselines A-F, emitting one
 * CSV row per (baseline,gamma,filter,selectivity,ef,k). Also a concurrency mode.
 *
 * Usage:
 *   bench <n> <d> <l2|ip> <nq> <seed> <out.csv> [tag]
 *   bench concurrency <n> <d> <l2|ip> <out.csv> [tag]
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <faiss/utils/random.h>
#include "acorn.h"

using namespace acorn;
using clk = std::chrono::high_resolution_clock;

// ------------------------------- data ------------------------------------
struct Dataset {
    int n, d, nc;
    std::vector<float> x;      // n*d
    std::vector<int> cluster;  // n
    std::vector<float> centers;// nc*d
    std::vector<float> queries;// nq*d
    int nq;
};

static Dataset gen_clustered(int n, int d, int nq, int seed) {
    Dataset ds; ds.n = n; ds.d = d; ds.nq = nq; ds.nc = std::max(16, n / 2000);
    faiss::RandomGenerator rng(seed);
    ds.centers.resize(size_t(ds.nc) * d);
    for (auto& v : ds.centers) v = rng.rand_float();
    ds.x.resize(size_t(n) * d); ds.cluster.resize(n);
    float sigma = 0.08f;
    for (int i = 0; i < n; i++) {
        int c = rng.rand_int(ds.nc); ds.cluster[i] = c;
        for (int j = 0; j < d; j++)
            ds.x[size_t(i) * d + j] = ds.centers[size_t(c) * d + j] + sigma * (rng.rand_float() * 2 - 1);
    }
    ds.queries.resize(size_t(nq) * d);
    for (int i = 0; i < nq; i++) {
        int c = rng.rand_int(ds.nc);
        for (int j = 0; j < d; j++)
            ds.queries[size_t(i) * d + j] = ds.centers[size_t(c) * d + j] + sigma * (rng.rand_float() * 2 - 1);
    }
    return ds;
}

// ---------------------------- filter sets --------------------------------
// Returns a sorted valid-id vector for a (type,target selectivity).
static std::vector<faiss::idx_t> make_filter(
        const Dataset& ds, const std::string& type, double sel, int seed) {
    int n = ds.n, d = ds.d;
    faiss::RandomGenerator rng(seed);
    std::vector<char> keep(n, 0);
    int want = std::max(1, (int)std::llround(sel * n));

    if (type == "random" || type == "shuffled" || type == "sparse") {
        // shuffled: id layout is irrelevant here (ordinal==id) => same as random.
        std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
        for (int i = 0; i < n; i++) std::swap(idx[i], idx[rng.rand_int(n)]);
        for (int i = 0; i < want; i++) keep[idx[i]] = 1;
    } else if (type == "correlated") {
        // membership correlated with vector position: project on random dir, keep top.
        std::vector<float> dir(d); for (auto& v : dir) v = rng.rand_float() * 2 - 1;
        std::vector<std::pair<float,int>> proj(n);
        for (int i = 0; i < n; i++) { float p = 0; for (int j=0;j<d;j++) p += dir[j]*ds.x[size_t(i)*d+j]; proj[i] = {p, i}; }
        std::sort(proj.begin(), proj.end());
        for (int i = 0; i < want; i++) keep[proj[i].second] = 1;
    } else if (type == "cluster") {
        // single contiguous region: grow from clusters nearest a random anchor.
        std::vector<float> anchor(ds.centers.begin(), ds.centers.begin()+d);
        int ac = rng.rand_int(ds.nc);
        for (int j=0;j<d;j++) anchor[j] = ds.centers[size_t(ac)*d+j];
        std::vector<std::pair<float,int>> dist(n);
        for (int i=0;i<n;i++){ float s=0; for(int j=0;j<d;j++){float df=anchor[j]-ds.x[size_t(i)*d+j]; s+=df*df;} dist[i]={s,i}; }
        std::sort(dist.begin(), dist.end());
        for (int i=0;i<want;i++) keep[dist[i].second]=1;
    } else if (type == "multicluster") {
        // union of several cluster-local blobs
        int blobs = 5; int per = want / blobs;
        for (int b=0;b<blobs;b++){
            int ac = rng.rand_int(ds.nc);
            std::vector<std::pair<float,int>> dist(n);
            for (int i=0;i<n;i++){ float s=0; for(int j=0;j<d;j++){float df=ds.centers[size_t(ac)*d+j]-ds.x[size_t(i)*d+j]; s+=df*df;} dist[i]={s,i}; }
            std::partial_sort(dist.begin(), dist.begin()+per, dist.end());
            for (int i=0;i<per;i++) keep[dist[i].second]=1;
        }
    } else if (type == "adversarial") {
        // valid = low-density outliers (far from global centroid): poorly connected
        // in a standard HNSW graph, stressing predicate-subgraph navigability.
        std::vector<float> cen(d,0); for(int i=0;i<n;i++) for(int j=0;j<d;j++) cen[j]+=ds.x[size_t(i)*d+j];
        for (auto& v:cen) v/=n;
        std::vector<std::pair<float,int>> dist(n);
        for (int i=0;i<n;i++){ float s=0; for(int j=0;j<d;j++){float df=cen[j]-ds.x[size_t(i)*d+j]; s+=df*df;} dist[i]={-s,i}; } // farthest first
        std::sort(dist.begin(), dist.end());
        for (int i=0;i<want;i++) keep[dist[i].second]=1;
    } else { fprintf(stderr,"unknown filter type %s\n", type.c_str()); exit(1); }

    std::vector<faiss::idx_t> ids;
    for (int i=0;i<n;i++) if (keep[i]) ids.push_back(i);
    return ids;
}

struct VecSel : faiss::IDSelector {
    const std::vector<char>* mask;
    bool is_member(faiss::idx_t id) const override { return (*mask)[id]; }
};

// ------------------------------ metrics ----------------------------------
static void norm_stats(SearchStats& s, int nq) {
    if (nq <= 0) return;
    s.dist_computations/=nq; s.nodes_visited/=nq; s.first_hop_expansions/=nq;
    s.second_hop_expansions/=nq; s.selector_checks/=nq; s.accepted/=nq;
    s.rejected/=nq; s.hops/=nq; s.visit_limit_terminations/=nq; s.results_returned/=nq;
}

static double pct(std::vector<double>& v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t i = (size_t)std::llround(p/100.0*(v.size()-1));
    return v[std::min(i, v.size()-1)];
}

struct Row {
    std::string baseline, graph, mode, filter; double sel_target, sel_actual;
    long cand; int M,gamma,M_beta,ef,k; double recall,qps,p50,p95,p99,mx,mean;
    SearchStats st; double build_ms; size_t gbytes; double avgd0; int maxd0;
};

static void emit_header(FILE* f) {
    fprintf(f,"tag,n,d,metric,filter,sel_target,sel_actual,cand,baseline,graph,mode,M,gamma,M_beta,ef,k,"
              "recall,qps,p50_us,p95_us,p99_us,max_us,mean_us,ndis,nodes_visited,first_hop,second_hop,"
              "sel_checks,accepted,rejected,hops,results_returned,build_ms,graph_bytes,avg_deg0,max_deg0\n");
}
static void emit_row(FILE* f, const char* tag, int n, int d, const char* metric, const Row& r) {
    fprintf(f,"%s,%d,%d,%s,%s,%.4f,%.4f,%ld,%s,%s,%s,%d,%d,%d,%d,%d,"
             "%.4f,%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.1f,%zu,%.1f,%d\n",
        tag,n,d,metric,r.filter.c_str(),r.sel_target,r.sel_actual,r.cand,
        r.baseline.c_str(),r.graph.c_str(),r.mode.c_str(),r.M,r.gamma,r.M_beta,r.ef,r.k,
        r.recall,r.qps,r.p50,r.p95,r.p99,r.mx,r.mean,
        (unsigned long long)r.st.dist_computations,(unsigned long long)r.st.nodes_visited,
        (unsigned long long)r.st.first_hop_expansions,(unsigned long long)r.st.second_hop_expansions,
        (unsigned long long)r.st.selector_checks,(unsigned long long)r.st.accepted,
        (unsigned long long)r.st.rejected,(unsigned long long)r.st.hops,
        (unsigned long long)r.st.results_returned,r.build_ms,r.gbytes,r.avgd0,r.maxd0);
}

// Run a baseline over all queries; returns aggregate Row (recall vs precomputed truth).
static Row run_baseline(
        const AcornIndex& idx, FilteredHnswSearchMode mode, const char* bname, const char* gname,
        const Dataset& ds, const std::vector<char>& mask, int k, int ef,
        const std::vector<std::unordered_set<faiss::idx_t>>& truth, long cand) {
    VecSel sel; sel.mask = &mask;
    std::vector<double> lat; lat.reserve(ds.nq);
    SearchStats agg; double rec_sum = 0; int rec_den = 0;
    std::vector<faiss::idx_t> ids(k); std::vector<float> dis(k);
    // warmup
    for (int w = 0; w < 3 && w < ds.nq; w++)
        filtered_search(idx, &ds.queries[size_t(w)*ds.d], k, ef, mode, &sel, ids.data(), dis.data(), nullptr);
    for (int qi = 0; qi < ds.nq; qi++) {
        SearchStats st;
        auto t0 = clk::now();
        int nr = filtered_search(idx, &ds.queries[size_t(qi)*ds.d], k, ef, mode, &sel, ids.data(), dis.data(), &st);
        auto t1 = clk::now();
        lat.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        agg.add(st);
        const auto& tr = truth[qi];
        if (!tr.empty()) {
            int hit = 0; for (int i=0;i<nr;i++) if (tr.count(ids[i])) hit++;
            rec_sum += double(hit) / tr.size(); rec_den++;
        }
    }
    norm_stats(agg, ds.nq);
    Row r; r.baseline=bname; r.graph=gname;
    r.mode = (mode==FilteredHnswSearchMode::ACORN?"ACORN":"STANDARD");
    r.M=idx.M; r.gamma=idx.gamma; r.M_beta=idx.M_beta; r.ef=ef; r.k=k; r.cand=cand;
    r.recall = rec_den ? rec_sum/rec_den : 0;
    r.mean = std::accumulate(lat.begin(),lat.end(),0.0)/lat.size();
    r.p50=pct(lat,50); r.p95=pct(lat,95); r.p99=pct(lat,99); r.mx=pct(lat,100);
    r.qps = r.mean>0 ? 1e6/r.mean : 0;
    r.st = agg; r.build_ms=idx.build_stats.build_ms; r.gbytes=idx.build_stats.graph_bytes;
    r.avgd0=idx.build_stats.avg_degree_l0; r.maxd0=idx.build_stats.max_degree_l0;
    return r;
}

static Row run_exact(const AcornIndex& idx, const char* bname, const Dataset& ds,
        const std::vector<char>& mask, int k,
        const std::vector<std::unordered_set<faiss::idx_t>>& truth, long cand) {
    VecSel sel; sel.mask = &mask;
    std::vector<double> lat; SearchStats agg; double rec_sum=0; int rec_den=0;
    std::vector<faiss::idx_t> ids(k); std::vector<float> dis(k);
    for (int qi=0; qi<ds.nq; qi++) {
        SearchStats st; auto t0=clk::now();
        int nr = exact_filtered_search(idx, &ds.queries[size_t(qi)*ds.d], k, &sel, ids.data(), dis.data(), &st);
        auto t1=clk::now();
        lat.push_back(std::chrono::duration<double,std::micro>(t1-t0).count());
        agg.add(st);
        const auto& tr=truth[qi];
        if (!tr.empty()){ int hit=0; for(int i=0;i<nr;i++) if(tr.count(ids[i])) hit++; rec_sum+=double(hit)/tr.size(); rec_den++; }
    }
    norm_stats(agg, ds.nq);
    Row r; r.baseline=bname; r.graph="exact"; r.mode="EXACT"; r.M=idx.M; r.gamma=0; r.M_beta=0; r.ef=0; r.k=k; r.cand=cand;
    r.recall = rec_den?rec_sum/rec_den:0;
    r.mean=std::accumulate(lat.begin(),lat.end(),0.0)/lat.size();
    r.p50=pct(lat,50);r.p95=pct(lat,95);r.p99=pct(lat,99);r.mx=pct(lat,100);
    r.qps=r.mean>0?1e6/r.mean:0; r.st=agg; r.build_ms=0; r.gbytes=0; r.avgd0=0; r.maxd0=0;
    return r;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "concurrency") {
        // bench concurrency <n> <d> <l2|ip> <out.csv> [tag]
        int n=atoi(argv[2]), d=atoi(argv[3]); std::string ms=argv[4]; const char* out=argv[5];
        const char* tag = argc>6?argv[6]:"conc";
        faiss::MetricType metric = (ms=="ip")?faiss::METRIC_INNER_PRODUCT:faiss::METRIC_L2;
        Dataset ds = gen_clustered(n, d, 500, 1234);
        AcornIndex gidx; AcornGammaBuildParameters bp; bp.enabled=true; bp.M=16; bp.gamma=4; bp.M_beta=16;
        build_acorn_gamma(gidx, d, metric, bp, n, ds.x.data());
        auto mask = std::vector<char>(n,0);
        auto ids = make_filter(ds, "random", 0.10, 99);
        for (auto id: ids) mask[id]=1;
        VecSel sel; sel.mask=&mask;
        FILE* f=fopen(out,"a");
        for (int nt : {1,4,8,16}) {
            std::atomic<long> done{0};
            auto t0=clk::now();
            auto worker=[&](){
                std::vector<faiss::idx_t> oi(10); std::vector<float> od(10); long c=0;
                for (int qi=0; qi<ds.nq; qi++){ filtered_search(gidx, &ds.queries[size_t(qi)*d],10,128,FilteredHnswSearchMode::ACORN,&sel,oi.data(),od.data(),nullptr); c++; }
                done += c;
            };
            std::vector<std::thread> ts; for(int t=0;t<nt;t++) ts.emplace_back(worker);
            for(auto&t:ts) t.join();
            auto t1=clk::now(); double sec=std::chrono::duration<double>(t1-t0).count();
            double qps = done.load()/sec;
            fprintf(f,"%s,%d,%d,%s,threads=%d,qps=%.1f,queries=%ld,sec=%.3f\n",tag,n,d,ms.c_str(),nt,qps,done.load(),sec);
            printf("[conc] threads=%d qps=%.1f\n", nt, qps);
        }
        fclose(f); return 0;
    }

    if (argc < 7) { fprintf(stderr,"usage: bench <n> <d> <l2|ip> <nq> <seed> <out.csv> [tag]\n"); return 1; }
    int n=atoi(argv[1]), d=atoi(argv[2]); std::string ms=argv[3]; int nq=atoi(argv[4]); int seed=atoi(argv[5]);
    const char* out=argv[6]; const char* tag = argc>7?argv[7]:"run";
    faiss::MetricType metric = (ms=="ip")?faiss::METRIC_INNER_PRODUCT:faiss::METRIC_L2;

    fprintf(stderr,"[bench] gen data n=%d d=%d nq=%d metric=%s\n", n,d,nq,ms.c_str());
    Dataset ds = gen_clustered(n, d, nq, seed);
    if (metric == faiss::METRIC_INNER_PRODUCT) {
        // IP/cosine search operates on normalized embeddings; normalize so that
        // inner product reflects cluster/angle structure, not vector norm.
        auto norm=[&](std::vector<float>& v){ for (size_t i=0;i+d<=v.size(); i+=d){ double s=0; for(int j=0;j<d;j++) s+=double(v[i+j])*v[i+j]; float inv = s>0?1.0f/std::sqrt((float)s):0; for(int j=0;j<d;j++) v[i+j]*=inv; } };
        norm(ds.x); norm(ds.queries);
    }

    // ---- build graphs once ----
    // gamma set is overridable via env ACORN_GAMMAS="4,8,16" (1M run caps lower).
    const int M = 16, efc = 100;
    fprintf(stderr,"[bench] build standard M=%d\n", M);
    AcornIndex A_std;  build_standard_hnsw(A_std, d, metric, M, efc, n, ds.x.data());

    // Two larger-M controls to isolate "simple densification" from ACORN-γ:
    //   large1 M=24 (level0=48, ~memory of ACORN-γ) and large2 M=48 (level0=96).
    std::vector<int> large_Ms = {24, 48};
    std::vector<AcornIndex> D_lrg(large_Ms.size());
    for (size_t li=0; li<large_Ms.size(); li++) {
        fprintf(stderr,"[bench] build large-M M=%d\n", large_Ms[li]);
        build_standard_hnsw(D_lrg[li], d, metric, large_Ms[li], efc, n, ds.x.data());
    }

    std::vector<int> gammas = {4,8,16,32};
    if (const char* g = getenv("ACORN_GAMMAS")) {
        gammas.clear(); std::string s=g,cur;
        for (char c: s+",") { if (c==',') { if(!cur.empty()) gammas.push_back(atoi(cur.c_str())); cur.clear(); } else cur+=c; }
    }
    std::vector<AcornIndex> C_acorn(gammas.size());
    for (size_t gi=0; gi<gammas.size(); gi++) {
        AcornGammaBuildParameters bp; bp.enabled=true; bp.M=M; bp.gamma=gammas[gi]; bp.M_beta=M;
        fprintf(stderr,"[bench] build acorn gamma=%d\n", gammas[gi]);
        build_acorn_gamma(C_acorn[gi], d, metric, bp, n, ds.x.data());
    }

    FILE* f = fopen(out, "w"); emit_header(f);

    std::vector<std::string> ftypes = {"random","correlated","cluster","adversarial","multicluster"};
    std::vector<double> sels = {0.50,0.25,0.10,0.05,0.02,0.01,0.005,0.001};
    std::vector<int> efs = {50,100,250,500};
    std::vector<int> ks = {10,100};

    for (auto& ft : ftypes) {
      for (double sel : sels) {
        auto ids = make_filter(ds, ft, sel, seed*131 + int(sel*100000));
        long cand = ids.size();
        if (cand == 0) continue;
        std::vector<char> mask(n,0); for (auto id: ids) mask[id]=1;
        double sel_actual = double(cand)/n;
        for (int k : ks) {
          if (k==100 && sel < 0.02) continue; // not enough candidates to be meaningful
          // precompute exact truth per query for this (filter,k)
          std::vector<std::unordered_set<faiss::idx_t>> truth(ds.nq);
          { VecSel vs; vs.mask=&mask; std::vector<faiss::idx_t> oi(k); std::vector<float> od(k);
            for (int qi=0; qi<ds.nq; qi++){ int nrr=exact_filtered_search(A_std,&ds.queries[size_t(qi)*d],k,&vs,oi.data(),od.data(),nullptr);
              for(int i=0;i<nrr;i++) truth[qi].insert(oi[i]); } }

          auto emit=[&](Row r){ r.filter=ft; r.sel_target=sel; r.sel_actual=sel_actual; emit_row(f,tag,n,d,ms.c_str(),r); };

          // F: exact once per (filter,k) — ef-independent
          emit(run_exact(A_std, "F_exact", ds, mask, k, truth, cand));

          for (int ef : efs) {
            emit(run_baseline(A_std, FilteredHnswSearchMode::STANDARD, "A_std_standard","std", ds,mask,k,ef,truth,cand));
            emit(run_baseline(A_std, FilteredHnswSearchMode::ACORN,    "B_std_acorn1",  "std", ds,mask,k,ef,truth,cand));
            for (size_t li=0; li<large_Ms.size(); li++) {
              char dn[40], en[40];
              snprintf(dn,sizeof(dn),"D_large%d_standard", large_Ms[li]);
              snprintf(en,sizeof(en),"E_large%d_acorn1", large_Ms[li]);
              emit(run_baseline(D_lrg[li], FilteredHnswSearchMode::STANDARD, dn,"large",ds,mask,k,ef,truth,cand));
              emit(run_baseline(D_lrg[li], FilteredHnswSearchMode::ACORN,    en,"large",ds,mask,k,ef,truth,cand));
            }
            for (size_t gi=0; gi<gammas.size(); gi++) {
              char bn[32]; snprintf(bn,sizeof(bn),"C_acorn_g%d", gammas[gi]);
              emit(run_baseline(C_acorn[gi], FilteredHnswSearchMode::ACORN, bn, "acorn", ds,mask,k,ef,truth,cand));
            }
          }
        }
      }
      fprintf(stderr,"[bench]   done filter=%s\n", ft.c_str());
    }
    fclose(f);
    fprintf(stderr,"[bench] wrote %s\n", out);
    return 0;
}
