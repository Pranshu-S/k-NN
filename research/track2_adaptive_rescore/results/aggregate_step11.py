#!/usr/bin/env python3
# Aggregates Track2 step-11 raw per-segment CSVs into the step-4..14 summary tables.
# Pure stdlib. Run from the results/ dir.
import csv, math, collections, sys

def rd(p):
    with open(p) as f: return list(csv.DictReader(f))

feas = rd("adaptive_feasible.csv")
perf = rd("adaptive_rep_perf.csv")
pred = rd("adaptive_predictors.csv")

TARGET=0.95; PRIMARY_CAL=150; BITS=[1,2,4,32]
segids = sorted({int(r["seg_id"]) for r in pred}, key=int)
seg = {int(r["seg_id"]): r for r in pred}          # predictors carry type/dim/N
N = {s:int(seg[s]["N"]) for s in segids}
D = {s:int(seg[s]["dim"]) for s in segids}
def bname(b): return "fp32" if int(b)==32 else f"{b}bit"

# feasible_ef lookup: (seg,bits,target,calN) -> ef (int, -1 if infeasible) ; ceiling_maxef per (seg,bits)
fe={}; ceil={}
for r in feas:
    k=(int(r["seg_id"]),int(r["bits"]),float(r["target"]),int(r["calN"]))
    fe[k]=int(r["feasible_ef"])
    ceil[(int(r["seg_id"]),int(r["bits"]))]=float(r["ceiling_maxef"])

def select(s,target,calN):
    for b in BITS:
        if fe.get((s,b,target,calN),-1) > 0: return b
    return 32  # fp32 fallback (may itself be infeasible -> flagged separately)

# ---- primary selection @0.95/PRIMARY_CAL ----
sel = {s: select(s,TARGET,PRIMARY_CAL) for s in segids}
def unachievable(s):  # even fp32 infeasible at primary
    return fe.get((s,32,TARGET,PRIMARY_CAL),-1) <= 0

# per-rep perf lookup
P={}
for r in perf:
    P[(int(r["seg_id"]),int(r["bits"]))]=r

# ---------- adaptive_segment_selection.csv ----------
with open("adaptive_segment_selection.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["seg_id","type","dim","N","selected_bits","selected_ef","ceiling_1b","ceiling_2b","ceiling_4b","ceiling_fp32","fp32_unachievable"])
    for s in segids:
        b=sel[s]; ef=fe.get((s,b,TARGET,PRIMARY_CAL),-1)
        w.writerow([s,seg[s]["type"],D[s],N[s],bname(b),ef,
            f'{ceil.get((s,1),0):.3f}',f'{ceil.get((s,2),0):.3f}',f'{ceil.get((s,4),0):.3f}',f'{ceil.get((s,32),0):.3f}',
            "YES" if unachievable(s) else "no"])

# ---------- distributions: by segment & by vector ----------
seg_ct=collections.Counter(bname(sel[s]) for s in segids)
vec_ct=collections.Counter();
for s in segids: vec_ct[bname(sel[s])]+=N[s]
totseg=len(segids); totvec=sum(N.values())
with open("adaptive_vector_distribution.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["representation","segments","pct_segments","vectors","pct_vectors"])
    for b in ["1bit","2bit","4bit","fp32"]:
        w.writerow([b,seg_ct.get(b,0),f'{100*seg_ct.get(b,0)/totseg:.1f}',vec_ct.get(b,0),f'{100*vec_ct.get(b,0)/totvec:.1f}'])

# ---------- effective bits/dim + storage ----------
def bits_of(name): return {"1bit":1,"2bit":2,"4bit":4,"fp32":32}[name]
def code_bytes(s,bits): return N[s]*D[s]*bits/8.0
adaptive_bytes=sum(code_bytes(s,bits_of(bname(sel[s]))) for s in segids)
all1=sum(code_bytes(s,1) for s in segids); all2=sum(code_bytes(s,2) for s in segids)
all4=sum(code_bytes(s,4) for s in segids); allf=sum(code_bytes(s,32) for s in segids)
eff_bits=sum(N[s]*D[s]*bits_of(bname(sel[s])) for s in segids)/sum(N[s]*D[s] for s in segids)
with open("adaptive_storage_summary.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["policy","total_code_MB","vs_all_1bit","vs_fp32","effective_bits_per_dim"])
    for name,b in [("all_1bit",all1),("adaptive",adaptive_bytes),("all_2bit",all2),("all_4bit",all4),("fp32",allf)]:
        eb = 1 if name=="all_1bit" else 2 if name=="all_2bit" else 4 if name=="all_4bit" else 32 if name=="fp32" else eff_bits
        w.writerow([name,f'{b/1e6:.1f}',f'{b/all1:.2f}x',f'{b/allf:.3f}x',f'{eb:.3f}'])

# ---------- policy comparison A-E (vector-weighted) ----------
def policy_metrics(bitfn):  # bitfn(s)->bits
    rec=miss=p50=p99=nodes=byt=stor=0.0; W=0
    for s in segids:
        b=bitfn(s); r=P.get((s,b))
        if r is None: continue
        w=N[s]; W+=w
        er=float(r["eval_recall"]); rec+=er*w; miss+= (1 if er<TARGET else 0)*w
        p50+=float(r["warm_p50"])*w; p99+=float(r["warm_p99"])*w
        nodes+=float(r["mean_nodes"])*w; byt+=float(r["bytes_per_query"])*w
        stor+=code_bytes(s,b)
    return dict(recall=rec/W,miss=100*miss/W,p50=p50/W,p99=p99/W,nodes=nodes/W,bytes=byt/W,storage_MB=stor/1e6,
               eff_bits=sum(N[s]*D[s]*bitfn(s) for s in segids)/sum(N[s]*D[s] for s in segids))
def lat_opt(s):  # cheapest-p99 feasible rep at 0.95 (fallback fp32)
    cand=[b for b in BITS if fe.get((s,b,TARGET,PRIMARY_CAL),-1)>0]
    if not cand: return 32
    return min(cand, key=lambda b: float(P[(s,b)]["warm_p99"]))
def cap4(s):  # production adaptive-quantized: never fp32, cap at 4-bit (fp32 fallbacks -> 4-bit)
    b=bits_of(bname(sel[s])); return 4 if b==32 else b
policies={"A_all_1bit":lambda s:1,"B_all_2bit":lambda s:2,"C_all_4bit":lambda s:4,
          "D_adaptive_min":lambda s:bits_of(bname(sel[s])),"E_latency_opt":lat_opt,"F_adaptive_cap4":cap4}
with open("adaptive_policy_comparison.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["policy","recall","SLA_miss_pct","p50_us","p99_us","nodes_per_q","bytes_per_q","storage_MB","effective_bits_per_dim"])
    pm={}
    for name,fn in policies.items():
        m=policy_metrics(fn); pm[name]=m
        w.writerow([name,f'{m["recall"]:.4f}',f'{m["miss"]:.1f}',f'{m["p50"]:.1f}',f'{m["p99"]:.1f}',f'{m["nodes"]:.0f}',f'{m["bytes"]:.0f}',f'{m["storage_MB"]:.1f}',f'{m["eff_bits"]:.3f}'])

# ---------- escalation reasons (step 7) ----------
reasons=collections.Counter()
for s in segids:
    b=sel[s]
    if b==1: reasons["1bit: meets SLA directly"]+=1
    elif b==2: reasons["2bit: 1-bit ceiling < target"]+=1
    elif b==4:
        if ceil.get((s,2),0) < TARGET: reasons["4bit: 2-bit ceiling < target"]+=1
        else: reasons["4bit: 2-bit feasible but not certified (conservative)"]+=1
    else:
        if unachievable(s): reasons["fp32: no quantized meets SLA (or unachievable)"]+=1
        else: reasons["fp32: quantized ceilings < target"]+=1
with open("adaptive_escalation_reasons.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["reason","count","pct"])
    for r,c in reasons.most_common(): w.writerow([r,c,f'{100*c/totseg:.1f}'])

# ---------- recall sensitivity (step 13) ----------
with open("adaptive_recall_sensitivity.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["target_recall","effective_bits_per_dim","pct1bit_vec","pct2bit_vec","pct4bit_vec","pctfp32_vec"])
    for tgt in [0.90,0.95,0.97]:
        sl={s:select(s,tgt,PRIMARY_CAL) for s in segids}
        eb=sum(N[s]*D[s]*bits_of(bname(sl[s])) for s in segids)/sum(N[s]*D[s] for s in segids)
        vc=collections.Counter();
        for s in segids: vc[bname(sl[s])]+=N[s]
        w.writerow([tgt,f'{eb:.3f}']+[f'{100*vc.get(b,0)/totvec:.1f}' for b in ["1bit","2bit","4bit","fp32"]])

# ---------- calibration stability (step 11) ----------
with open("adaptive_calibration_stability.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["calN","agreement_vs_cal150_pct","segments_changed","mean_abs_bit_diff"])
    ref={s:bits_of(bname(select(s,TARGET,150))) for s in segids}
    for calN in [25,50,100,150]:
        agree=0; diff=0
        for s in segids:
            b=bits_of(bname(select(s,TARGET,calN)))
            if b==ref[s]: agree+=1
            diff+=abs(math.log2(b)-math.log2(ref[s]))
        w.writerow([calN,f'{100*agree/totseg:.1f}',totseg-agree,f'{diff/totseg:.3f}'])

# ---------- predictor correlation (step 10) ----------
def pearson(xs,ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys)); dx=math.sqrt(sum((x-mx)**2 for x in xs)); dy=math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx>0 and dy>0 else 0.0
selbits=[math.log2(bits_of(bname(sel[s]))) for s in segids]   # log2 bits as difficulty proxy
predcols=["dim","N","anisotropy","recon_err_1b","recon_err_2b","recon_err_4b","intrinsic_dim","nn_spread","cluster_sep","probe_ceil_1b_ef50"]
with open("adaptive_predictor_analysis.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["predictor","pearson_corr_with_log2bits","abs_corr"])
    rows=[]
    for c in predcols:
        xs=[float(seg[s][c]) for s in segids]
        r=pearson(xs,selbits); rows.append((c,r))
    for c,r in sorted(rows,key=lambda t:-abs(t[1])): w.writerow([c,f'{r:.3f}',f'{abs(r):.3f}'])

# ---------- master summary ----------
print("=== SELECTION (segments / vectors) ===")
for b in ["1bit","2bit","4bit","fp32"]:
    print(f'  {b:5s}: {seg_ct.get(b,0):2d} segs ({100*seg_ct.get(b,0)/totseg:4.1f}%)  {vec_ct.get(b,0):>9d} vec ({100*vec_ct.get(b,0)/totvec:4.1f}%)')
cap4_bytes=sum(code_bytes(s,cap4(s)) for s in segids)
cap4_eff=sum(N[s]*D[s]*cap4(s) for s in segids)/sum(N[s]*D[s] for s in segids)
print(f'\nadaptive_MIN (incl fp32 fallback): eff bits/dim={eff_bits:.3f}  storage={adaptive_bytes/1e6:.1f}MB  vs4bit={adaptive_bytes/all4:.2f}x  vsfp32={adaptive_bytes/allf:.3f}x')
print(f'adaptive_CAP4 (production, no fp32): eff bits/dim={cap4_eff:.3f}  storage={cap4_bytes/1e6:.1f}MB  vs4bit={cap4_bytes/all4:.2f}x (saves {100*(1-cap4_bytes/all4):.0f}%)  vsfp32 compression={32/cap4_eff:.1f}x')
print(f'  one iid-128 seg escalated to fp32 (conservative) inflates adaptive_MIN; cap4 is the realistic quantized policy.')
print("\n=== MASTER POLICY TABLE ===")
print(f'{"metric":<20}{"all_1bit":>10}{"all_2bit":>10}{"all_4bit":>10}{"adaptive":>10}')
mp={n:policy_metrics(fn) for n,fn in policies.items()}
def row(lbl,key,fmt):
    a=mp["A_all_1bit"][key];b=mp["B_all_2bit"][key];c=mp["C_all_4bit"][key];d=mp["D_adaptive_min"][key]
    print(f'{lbl:<20}{fmt.format(a):>10}{fmt.format(b):>10}{fmt.format(c):>10}{fmt.format(d):>10}')
row("recall SLA miss %","miss","{:.1f}"); row("effective bits/dim","eff_bits","{:.2f}")
row("total code MB","storage_MB","{:.1f}"); row("avg nodes/query","nodes","{:.0f}")
row("avg bytes/query","bytes","{:.0f}"); row("p99 us","p99","{:.1f}")
print(f'\nsegments profiled: {totseg}   total vectors: {totvec:,}')
