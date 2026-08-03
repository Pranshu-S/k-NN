# RACORN-1 / RACORN-1+ — algorithm spec (for implementation + reports)

Source: Kim & Choe, *"RACORN-1: Adaptive Recall-Preserving Speedup for
Low-Selectivity Filtered Vector Search"*, arXiv:2607.00768v1 (Naver, 2026).
R = Resilient/Recall-preserving. Search-time only, in-place on a standard HNSW
graph (no build change, no extra memory) — same scope as ACORN-1.

## Beam (ACORN-SEARCH-LAYER, Alg. 1) — bottom layer, ef = ef_search
V=visited, C=candidate min-heap(dist), W=result max-heap capped at ef.
Start V={ep},C={ep},W={ep if F(ep)}. Loop while |C|>0:
- c = nearest from C; if |W|≥ef and dist(c)>max(W): break
- for e in EXPAND(c,F,V,W,ef): V∪={e}; if dist(e)<max(W) or |W|<ef: C∪={e};
  **if F(e): W∪={e}** (cap ef)  ← the F(e) guard EXCLUDES bridges from results
Upper layers: standard greedy descent (ef=1), unfiltered.

## ACORN1-EXPAND (baseline)
C1 = unvisited passing 1-hop of u. If |C1| insufficient, gather unvisited passing
2-hop C2 (capped at M). Return C1∪C2. Filter-failing nodes skipped, NOT marked
visited. → recall collapses below ~1/M selectivity (C2 ≈ M²·s → 0).

## RACORN1-EXPAND (Alg. 2) — adds Adaptive Search Fallback (ASF)
Params: bridge_ratio b (default 1.0, range [0,M]). M = base-level degree = 2·M_user (=32).
1. n = #unvisited 1-hop; C1 = unvisited passing 1-hop.
2. Scan all 2-hop w (unvisited): F(w)→C2 (pass), else→Bpool (fail).
3. target = n·b.
4. **ASF trigger: if |C2| < target:**
   - mark ALL unvisited 1-hop FAILS visited; mark ALL Bpool visited (prevents re-eval)
   - **bridge-skip gate: if |W| < ef:** needed = target−|C2|;
     B = stride_sample(Bpool, needed) if |Bpool|>needed else Bpool
5. C2 cap: if |C1|+|C2| > M: C2 = stride_sample(C2, M−|C1|).
6. return C1 ∪ C2 ∪ B  (**B = failing bridges → candidate queue only, never W**)
Bridges = 2-hop-FAILING nodes admitted as transient waypoints to detour around
severed predicate-subgraph paths. Theorem 1: E[|expand|] ≥ M·b for s<b/M
(selectivity-independent frontier) → recall no longer collapses.

## Stride sampling (Alg., §4.2)
Given pool P, target T: if |P|≤T return P; else σ=⌊|P|/T⌋, take P[0],P[σ],P[2σ],…
Preserves radial/spatial diversity vs prefix truncation (each 1-hop's 2-hop block
represented proportionally). Applied at bridge-selection and C2-cap sites.

## RACORN-1+ = RACORN-1 + Adaptive Exact Fallback (AEF, §5)
Monitor running predicate-pass ratio among visited (measured accurately because
ASF registers failing nodes in V). After enough visited, if pass_ratio <
AEF_FB_Threshold → switch to Exact Search (pre-filter brute force over passing
set) → recall 1.0. Threshold user-set (0.003 at 1M, scales ~1/N and ∝ ef).
Equivalent operationally to an absolute eligible-count floor (≈ threshold·N).

## Key defaults
M_user=16 (degree 32), EF_c=100, EF_s=200, K=100, bridge_ratio=1.0, AEF thr=0.003@1M.

## Reported results (paper) — the value proposition
- ACORN-1 collapse (SIFT 1M recall @1/0.5/0.1/0.01%): 0.72/0.40/0.01/0.00.
- RACORN-1 recovers: 0.96/0.97/0.99/0.98, 5–10× speedup over HNSW.
- RACORN-1+ : recall 1.00 with 20–75× speedup at 1M ≤0.1%; 13× at 40M 0.01%.
- **Negative correlation (cluster filter far from query)** K=100: HNSW recall
  0.87–0.99 but latency 0.5–8.7 s; ACORN-1 recall COLLAPSES 0.08–0.22;
  RACORN-1 recall 0.82–0.98 at 5–9× HNSW speedup. ← matches our adversarial/cluster finding.
- BR sensitivity: No-correlation → BR↑ helps recall (headroom). Negative → BR↓
  (0.25–0.5) is the efficient frontier (recall within −0.02, latency ~50%).
- Stride sampling: small auxiliary gain (−0.006 L2, up to −0.04 cosine 40M).

## Mapping to our benchmark
- Our clustered Gaussian mixture == the paper's K-means clusters.
- Positive correlation = filter cluster == query's cluster (valid near query).
- Negative correlation = filter cluster ≠ query's cluster (valid far)  ← ACORN-1 killer.
- No correlation = random filter (our rho=0 / "random" family).
