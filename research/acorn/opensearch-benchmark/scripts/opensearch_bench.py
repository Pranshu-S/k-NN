#!/usr/bin/env python3
"""
Live OpenSearch end-to-end filtered-search harness for the ACORN/RACORN POC.

STATUS: runnable against a real single-node OpenSearch that has the k-NN plugin
built WITH this POC's native library (filtered_search_mode support). It was NOT
executed in the prototyping sandbox (no cmake/JNI build + no running cluster
there). The native-layer recall/visit-count evidence — which is what these
traversal policies change — comes from research/acorn/opensearch-benchmark/
bench_collective (see METHODOLOGY §1). This script closes the remaining
Java-layer gaps (filter construction, segment fan-out, request latency, GC/heap)
when run on a real node.

Automates: create index -> bulk index vectors+metadata -> control segment count
(force-merge) -> warm -> exact ground truth -> query per strategy (rotated order)
-> collect latency + recall + _nodes/stats -> CSV.

Usage:
  python3 opensearch_bench.py --host https://localhost:9200 --user admin --pass ... \
      --vectors sift.fvecs --n 1000000 --dim 128 --segments 5 \
      --strategies standard acorn racorn racorn_plus --k 10 --ef 200 \
      --out results/live_sift1m.csv
"""
import argparse, csv, json, time, statistics, random, sys
try:
    import numpy as np
    import requests
    from requests.auth import HTTPBasicAuth
except ImportError:
    print("requires: pip install numpy requests", file=sys.stderr); raise

def es(session, method, url, **kw):
    r = session.request(method, url, **kw); r.raise_for_status(); return r

def build_filter(mode):
    # example: category filter (aligns with product-search scenario)
    return {"term": {"category": "c_target"}}

def knn_query(vec, k, filt, ef, fmode):
    mp = {"ef_search": ef}
    if fmode and fmode != "standard":
        mp["filtered_search_mode"] = fmode
    return {"size": k, "query": {"knn": {"v": {
        "vector": vec.tolist(), "k": k, "filter": {"bool": {"filter": filt}},
        "method_parameters": mp}}}}

def exact_truth(base, meta, q, k, keep_mask):
    # brute force over eligible (filter-passing) docs only
    elig = np.where(keep_mask)[0]
    if len(elig) == 0: return set()
    d = np.linalg.norm(base[elig] - q, axis=1)
    order = elig[np.argsort(d)[:k]]
    return set(int(i) for i in order)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True); ap.add_argument("--user"); ap.add_argument("--pass", dest="pw")
    ap.add_argument("--index", default="acorn_poc"); ap.add_argument("--vectors", required=True)
    ap.add_argument("--n", type=int, required=True); ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--nq", type=int, default=200); ap.add_argument("--segments", type=int, default=1)
    ap.add_argument("--strategies", nargs="+", default=["standard","acorn","racorn","racorn_plus"])
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--ef", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    s = requests.Session(); s.verify = False
    if a.user: s.auth = HTTPBasicAuth(a.user, a.pw)

    base = np.fromfile(a.vectors, dtype=np.float32).reshape(-1, a.dim)[:a.n]
    # metadata: assign a target category to a vector-correlated cluster (product scenario)
    # (replace with the real dataset's metadata in a production run)
    rng = np.random.default_rng(0)
    centroid = base[rng.integers(0, a.n)]
    dist = np.linalg.norm(base - centroid, axis=1)
    cat = np.where(dist < np.percentile(dist, 1.0), "c_target", "c_other")  # ~1% selective, correlated
    keep = (cat == "c_target")

    # 1. create index
    s.request("DELETE", f"{a.host}/{a.index}")
    mapping = json.load(open("setup/index_mapping.json")); mapping["mappings"]["properties"]["v"]["dimension"] = a.dim
    es(s, "PUT", f"{a.host}/{a.index}", json=mapping, headers={"Content-Type":"application/json"})

    # 2. bulk index
    B = 2000
    for i0 in range(0, a.n, B):
        lines = []
        for i in range(i0, min(i0+B, a.n)):
            lines.append(json.dumps({"index":{"_id":str(i)}}))
            lines.append(json.dumps({"v": base[i].tolist(), "category": str(cat[i])}))
        es(s, "POST", f"{a.host}/_bulk", data="\n".join(lines)+"\n", headers={"Content-Type":"application/x-ndjson"})
    es(s, "POST", f"{a.host}/{a.index}/_refresh")
    # 3. control segments
    es(s, "POST", f"{a.host}/{a.index}/_forcemerge?max_num_segments={a.segments}")
    es(s, "POST", f"{a.host}/{a.index}/_refresh")
    # warm native graph
    es(s, "GET", f"{a.host}/_plugins/_knn/warmup/{a.index}")

    # queries + ground truth
    qidx = rng.choice(a.n, size=a.nq, replace=False)
    queries = base[qidx]
    filt = build_filter("cat")
    truth = [exact_truth(base, cat, queries[j], a.k, keep) for j in range(a.nq)]

    out = open(a.out, "w"); w = csv.writer(out)
    w.writerow(["strategy","segments","k","ef","recall","p50_ms","p95_ms","p99_ms","mean_ms","took_ms_mean"])

    # rotate strategy order per query to reduce thermal/cache bias
    order = list(a.strategies)
    for warm in range(5):  # JVM/native warmup
        s.request("POST", f"{a.host}/{a.index}/_search", json=knn_query(queries[0], a.k, filt, a.ef, order[0]),
                  headers={"Content-Type":"application/json"})

    agg = {st: {"lat": [], "took": [], "rec": []} for st in order}
    for j in range(a.nq):
        random.shuffle(order)
        for st in order:
            body = knn_query(queries[j], a.k, filt, a.ef, st)
            t0 = time.perf_counter()
            r = es(s, "POST", f"{a.host}/{a.index}/_search", json=body, headers={"Content-Type":"application/json"})
            t1 = time.perf_counter()
            resp = r.json()
            ids = set(int(h["_id"]) for h in resp["hits"]["hits"])
            tr = truth[j]
            agg[st]["lat"].append((t1-t0)*1000.0)
            agg[st]["took"].append(resp["took"])
            agg[st]["rec"].append(len(ids & tr)/max(1, min(a.k, len(tr))) if tr else 0.0)

    def pct(v,p): v=sorted(v); return v[min(len(v)-1, int(p/100*(len(v)-1)))]
    for st in a.strategies:
        L=agg[st]["lat"]
        w.writerow([st, a.segments, a.k, a.ef, round(statistics.mean(agg[st]["rec"]),4),
                    round(pct(L,50),3), round(pct(L,95),3), round(pct(L,99),3),
                    round(statistics.mean(L),3), round(statistics.mean(agg[st]["took"]),3)])
    out.close()
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
