#!/usr/bin/env python3
"""Analyze ACORN-γ benchmark CSVs -> markdown tables + decision-framework
classification. Pure stdlib. Usage: analyze.py <csv> [csv...]"""
import csv, sys
from collections import defaultdict

def family(bl):
    if bl == "A_std_standard": return "standard-HNSW"
    if bl == "B_std_acorn1":  return "ACORN-1"
    if bl == "F_exact":       return "exact"
    if bl.startswith("C_acorn"): return "ACORN-γ"
    if bl.startswith("D_large"): return "larger-M"
    if bl.startswith("E_large"): return "larger-M+ACORN"
    return bl

def load(paths):
    rows=[]
    for p in paths:
        try: fh=open(p)
        except FileNotFoundError: continue
        for r in csv.DictReader(fh):
            for k in ("recall","mean_us","p50_us","p95_us","p99_us","qps","sel_actual","sel_target","build_ms","avg_deg0"):
                r[k]=float(r[k])
            for k in ("ndis","cand","ef","k","graph_bytes","gamma","max_deg0","n","d","second_hop"):
                r[k]=int(float(r[k]))
            rows.append(r)
    return rows

def best_meeting(cands, thr):
    """per baseline: min-latency row with recall>=thr."""
    out={}
    for r in cands:
        if r["recall"]>=thr:
            c=out.get(r["baseline"])
            if c is None or r["mean_us"]<c["mean_us"]: out[r["baseline"]]=r
    return out

def decision_framework(rows):
    scen=defaultdict(list)
    for r in rows:
        if r["k"]==10: scen[(r["tag"],r["filter"],r["sel_target"])].append(r)
    print("## Decision framework — best family per workload (k=10)\n")
    print("Winner = lowest mean latency among baselines reaching the recall tier "
          "(tries 0.90, else 0.80, 0.50, any).\n")
    print("| dataset | filter | sel_target | cand | tier | WINNER | recall | µs | runner-up | µs |")
    print("|---|---|---:|---:|---:|---|---:|---:|---|---:|")
    wins=defaultdict(int); winrows=[]
    for key in sorted(scen):
        tag,filt,sel=key; cands=scen[key]
        for thr in (0.90,0.80,0.50,0.0):
            bb=best_meeting(cands,thr)
            if bb: break
        if not bb: continue
        ranked=sorted(bb.values(), key=lambda r:r["mean_us"])
        w=ranked[0]; s=ranked[1] if len(ranked)>1 else None
        wins[family(w["baseline"])]+=1
        winrows.append((key,w,thr))
        srU = f"{family(s['baseline'])} ({s['mean_us']:.1f})" if s else "-"
        smu = f"{s['mean_us']:.1f}" if s else "-"
        print(f"| {tag} | {filt} | {sel:.3f} | {w['cand']} | {thr:.2f} | "
              f"{family(w['baseline'])} `{w['baseline']}` | {w['recall']:.3f} | {w['mean_us']:.1f} | "
              f"{family(s['baseline']) if s else '-'} `{s['baseline'] if s else '-'}` | {smu} |")
    print("\n**Workload wins by family:** " +
          ", ".join(f"**{k}**={v}" for k,v in sorted(wins.items(), key=lambda x:-x[1])))
    print()
    return winrows

def q1_traversal(rows):
    """A vs B recall at ef=100, per filter x sel (k=10), averaged over datasets."""
    print("## Q1 — ACORN traversal vs standard filtered HNSW (recall @ ef=100, k=10)\n")
    agg=defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["k"]==10 and r["ef"]==100 and r["baseline"] in ("A_std_standard","B_std_acorn1"):
            agg[(r["filter"],r["sel_target"])][r["baseline"]].append(r["recall"])
    print("| filter | sel | A standard recall | B ACORN-1 recall | Δ |")
    print("|---|---:|---:|---:|---:|")
    for key in sorted(agg):
        a=agg[key].get("A_std_standard",[]); b=agg[key].get("B_std_acorn1",[])
        if not a or not b: continue
        am=sum(a)/len(a); bm=sum(b)/len(b)
        print(f"| {key[0]} | {key[1]:.3f} | {am:.3f} | {bm:.3f} | {bm-am:+.3f} |")
    print()

def q_gamma(rows):
    """gamma sensitivity: recall by gamma for cluster/correlated at ef=250,k=10."""
    print("## Q8 — γ sensitivity (recall by γ, ACORN-γ, ef=250, k=10)\n")
    tab=defaultdict(dict)
    gammas=set()
    for r in rows:
        if r["k"]==10 and r["ef"]==250 and r["baseline"].startswith("C_acorn"):
            tab[(r["tag"],r["filter"],r["sel_target"])][r["gamma"]]=r["recall"]; gammas.add(r["gamma"])
    gammas=sorted(gammas)
    print("| dataset | filter | sel | " + " | ".join(f"γ={g}" for g in gammas) + " |")
    print("|---|---|---:|" + "---:|"*len(gammas))
    for key in sorted(tab):
        if key[1] not in ("cluster","correlated","adversarial","multicluster"): continue
        cells=" | ".join(f"{tab[key].get(g,float('nan')):.3f}" if g in tab[key] else "-" for g in gammas)
        print(f"| {key[0]} | {key[1]} | {key[2]:.3f} | {cells} |")
    print()

def build_table(rows):
    print("## Index build & memory overhead (per graph, from build stats)\n")
    seen={}
    for r in rows:
        key=(r["tag"],r["baseline"])
        if r["graph_bytes"]==0: continue
        if key not in seen: seen[key]=r
    print("| dataset | baseline | build_ms | graph_bytes | bytes/vec | avg_deg0 | max_deg0 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for key in sorted(seen):
        r=seen[key]
        bpv=r["graph_bytes"]/max(1,r["n"])
        print(f"| {r['tag']} | {r['baseline']} | {r['build_ms']:.0f} | {r['graph_bytes']:,} | {bpv:.1f} | {r['avg_deg0']:.1f} | {r['max_deg0']} |")
    print()

if __name__=="__main__":
    rows=load(sys.argv[1:])
    if not rows: print("no rows loaded"); sys.exit(0)
    decision_framework(rows)
    q1_traversal(rows)
    q_gamma(rows)
    build_table(rows)
