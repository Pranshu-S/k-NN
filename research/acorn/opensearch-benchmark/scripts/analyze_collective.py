#!/usr/bin/env python3
"""Analyze collective ACORN/RACORN benchmark CSVs -> markdown tables.
Usage: analyze_collective.py <csv> [csv...]"""
import csv, sys
from collections import defaultdict

ORDER = ["Ex_exact","H_hnsw_infilt","Hl_hnsw_large","Fx_faiss_std","A1_acorn1",
         "Ag_acorn_gamma","R1_racorn1","R1q_racorn1_br25","R1p_racorn1plus"]
LABEL = {"Ex_exact":"exact","H_hnsw_infilt":"HNSW-infilt","Hl_hnsw_large":"HNSW-largeM",
         "Fx_faiss_std":"faiss-native","A1_acorn1":"ACORN-1","Ag_acorn_gamma":"ACORN-γ",
         "R1_racorn1":"RACORN-1","R1q_racorn1_br25":"RACORN-1(BR.25)","R1p_racorn1plus":"RACORN-1+"}

def load(paths):
    rows=[]
    for p in paths:
        try: fh=open(p)
        except FileNotFoundError: continue
        for r in csv.DictReader(fh):
            for k in ("recall_med","recall_sd","p95_med","mean_us_med","ndis_med","bridges_med","aef_med","sel"):
                r[k]=float(r[k])
            for k in ("cand","ef","n"): r[k]=int(float(r[k]))
            rows.append(r)
    return rows

def best_ef(rows, corr, cand, strat):
    """row with max recall (tie: min ndis) over ef for a strategy; exact is ef-independent."""
    cs=[r for r in rows if r["correlation"]==corr and r["cand"]==cand and r["strategy"]==strat]
    if not cs: return None
    return sorted(cs, key=lambda r:(-r["recall_med"], r["ndis_med"]))[0]

def table(rows, corr):
    cands=sorted({r["cand"] for r in rows})
    print(f"### Correlation = {corr}\n")
    print("recall (best-ef) — visit-count `ndis` in (); **bold** = recall≥0.90\n")
    hdr="| strategy | " + " | ".join(f"{c}" for c in cands) + " |"
    print(hdr); print("|---|" + "---:|"*len(cands))
    for strat in ORDER:
        cells=[]
        for c in cands:
            r=best_ef(rows,corr,c,strat)
            if not r: cells.append("-"); continue
            rec=r["recall_med"]; nd=r["ndis_med"]
            s=f"{rec:.2f} ({nd:.0f})"
            if rec>=0.90: s=f"**{s}**"
            cells.append(s)
        print(f"| {LABEL[strat]} | " + " | ".join(cells) + " |")
    print()

def winners(rows, corr, target=0.90):
    """per cand: lowest ndis strategy reaching recall>=target (visit-count proxy for speed)."""
    cands=sorted({r["cand"] for r in rows})
    print(f"#### {corr}: cheapest strategy (min visits) reaching recall ≥ {target}\n")
    print("| cand | winner | recall | ndis | runner-up | ndis |")
    print("|---:|---|---:|---:|---|---:|")
    tally=defaultdict(int)
    for c in cands:
        cand_rows=[]
        for strat in ORDER:
            r=best_ef(rows,corr,c,strat)
            if r and r["recall_med"]>=target: cand_rows.append((LABEL[strat], r["recall_med"], r["ndis_med"]))
        cand_rows.sort(key=lambda x:x[2])
        if not cand_rows: print(f"| {c} | (none ≥{target}) | | | | |"); continue
        w=cand_rows[0]; ru=cand_rows[1] if len(cand_rows)>1 else ("-",0,0)
        tally[w[0]]+=1
        print(f"| {c} | **{w[0]}** | {w[1]:.2f} | {w[2]:.0f} | {ru[0]} | {ru[2]:.0f} |")
    print("\nwins: " + ", ".join(f"{k}={v}" for k,v in sorted(tally.items(),key=lambda x:-x[1])) + "\n")

if __name__=="__main__":
    rows=load(sys.argv[1:])
    if not rows: print("no rows"); sys.exit(0)
    for corr in ["no","pos","neg"]:
        if any(r["correlation"]==corr for r in rows):
            table(rows,corr); winners(rows,corr)
