# ACORN-γ for native Faiss HNSW — Benchmarks, Analysis & Recommendation

Companion to [`ACORN_GAMMA_DESIGN.md`](ACORN_GAMMA_DESIGN.md). Quantitative
evidence for the RFC decision. All numbers are from the prototype in
`research/acorn/` against **real Faiss 1.11.0** `IndexHNSWFlat` graphs (standard
and ACORN-γ), single-threaded, macOS/arm64, Accelerate BLAS.

Raw CSVs: [`raw-results/`](raw-results/). Regenerate with
`bash research/acorn/scripts/run_all.sh`; re-derive tables with
`python3 research/acorn/scripts/analyze.py raw-results/*.csv`.

---

## 1. Baselines

| id | graph | search | isolates |
|---|---|---|---|
| **A** | standard HNSW (M=16) | stock Faiss `IDSelector` | current OpenSearch path |
| **B** | standard HNSW (M=16) | ACORN traversal (γ=1) | **ACORN-1**: traversal benefit |
| **C** | **ACORN-γ** (M=16, γ∈{4,8,16,32}) | ACORN traversal | ACORN-γ graph benefit |
| **D** | larger-M HNSW (M∈{24,48}) | stock Faiss `IDSelector` | simple densification |
| **E** | larger-M HNSW (M∈{24,48}) | ACORN traversal (γ=1) | densification + traversal |
| **F** | — | exact filtered brute force | oracle / lower bound |

ACORN-γ level-0 degree = `M_β+1.5M = 40` (M_β=M=16); larger-M level-0 = `2M` =
48 / 96. `efConstruction`: 100 (standard/large), `M·γ` (ACORN-γ).

## 2. Methodology

- **Data**: clustered Gaussian mixture (`nc≈n/2000` centers, σ=0.08) so cluster/
  correlated filters and HNSW structure are meaningful; queries drawn from the
  same mixture. Corpus sizes 100K (primary) and 1M (scale); dims 128 (primary)
  and 768; metrics L2 and inner-product (IP vectors L2-normalized → cosine).
- **Filters** (7 relationships from the brief): `random` (= `shuffled`/`sparse`
  — id layout is irrelevant since ordinal==id), `correlated` (top of a random
  projection), `cluster` (single vector-space region near a random anchor),
  `multicluster` (union of 5 blobs), `adversarial` (low-density outliers, far
  from the global centroid — poorly connected in a standard graph).
- **Selectivities**: 0.5, 0.25, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001 — reported
  with **absolute candidate counts** (`cand`), which drive behaviour more than %.
- **Queries**: 100 (100K) / 50 (1M), 3-query warmup; per-query wall-clock →
  p50/p95/p99/max, QPS. `k∈{10,100}`; `efSearch∈{50,100,250,500}`.
- **Recall@k**: vs an exact filtered oracle computed per query (F). All
  per-query counters (distance comps, hops, selector checks, first/second-hop
  expansions, accepted/rejected) recorded and normalized per query.
- **Correctness**: `research/acorn/build/tests` — **33/33 assertions pass**
  (degree bounds, valid/well-formed adjacency, determinism, selector
  correctness, 0/<k/all-match edge cases, duplicate suppression, cycles/
  termination, serialization round-trip + bad-magic rejection, 8-thread
  concurrency parity, brute-force parity at 3 selectivities, mode separation).

---

## 3. Headline: ACORN traversal benefit (Q1) — 100K, d=128, L2, ef=100, k=10

Recall of stock filtered HNSW (**A**) vs ACORN-1 traversal (**B**), same graph:

| filter | sel | A standard | B ACORN-1 | Δ |
|---|---:|---:|---:|---:|
| cluster | 0.050 | 0.142 | **0.844** | **+0.702** |
| cluster | 0.100 | 0.202 | 0.607 | +0.405 |
| cluster | 0.250 | 0.374 | 0.710 | +0.336 |
| correlated | 0.020 | 0.046 | 0.571 | +0.525 |
| correlated | 0.050 | 0.059 | 0.586 | +0.527 |
| correlated | 0.100 | 0.190 | 0.605 | +0.415 |
| multicluster | 0.100 | 0.062 | 0.683 | +0.621 |
| multicluster | 0.250 | 0.364 | 0.645 | +0.281 |
| adversarial | 0.050 | 0.180 | 0.313 | +0.133 |
| **random** | 0.010 | **0.813** | 0.026 | **−0.787** |
| **random** | 0.020 | 0.883 | 0.086 | −0.797 |
| **random** | 0.050 | 0.942 | 0.330 | −0.612 |
| random | 0.500 | 0.969 | 0.945 | −0.024 |
| any | ≤0.01 | ~0.0–0.1 | ~0.0 | reachability-limited |

**The ACORN traversal's value is entirely conditional on filter–vector
correlation.** It is transformative when the valid set is a coherent region a
standard graph routes *away* from (cluster/correlated/multicluster: +0.3…+0.7
recall). It is actively **harmful** on random/uniform filters (−0.6…−0.8),
because stock Faiss routes *through* rejected nodes (it pushes every neighbour to
the beam) and thus naturally finds uniformly-scattered valid nodes, whereas
ACORN restricts the beam to predicate-passing nodes + two-hop bridges. At very
low selectivity (cand ≤ ~1000) both collapse — the valid subgraph is
unreachable.

Mechanism (cluster 1% @100K, k=10, per query): A does 534 dist-comps → recall
0.02 (beam converges to the query's own cluster, which holds no valid nodes);
ACORN-γ g8 does 888 dist-comps → recall 0.96 (predicate-subgraph traversal
navigates to the valid cluster). Stock filtered HNSW is not just lower-recall —
at ef=500 on hard filters its latency balloons ~7× (the result threshold never
tightens, so the ef stop triggers late).

---

## 4. Decision framework — best family per workload (100K, d=128, L2, k=10)

Winner = lowest mean latency among baselines reaching recall ≥ 0.90 (else the
best-achievable tier). Full table: [`raw-results/analysis_100k_d128_l2.md`](raw-results/analysis_100k_d128_l2.md).

**Wins by family (of 40 workloads): exact = 23, larger-M+ACORN = 8,
larger-M = 5, ACORN-γ = 3, standard-HNSW = 1.**

Representative rows:

| filter | sel | cand | WINNER | recall | µs | runner-up | µs |
|---|---:|---:|---|---:|---:|---|---:|
| cluster | 0.001 | 100 | **ACORN-γ g32** | 1.000 | **60** | ACORN-γ g16 | 61 |
| cluster | 0.005 | 500 | **ACORN-γ g32** | 0.970 | **52** | exact | 140 |
| cluster | 0.010 | 1000 | **ACORN-γ g8** | 0.959 | **92** | ACORN-γ g16 | 95 |
| cluster | 0.020 | 2000 | larger-M+ACORN (E24) | 0.913 | 117 | E48 | 145 |
| cluster | 0.050 | 5000 | larger-M+ACORN (E48) | 0.966 | 269 | exact | 477 |
| cluster | 0.250 | 25000 | larger-M+ACORN (E48) | 0.954 | 1029 | exact | 2357 |
| correlated | ≤0.02 | ≤2000 | exact | 1.000 | 112–242 | — | — |
| correlated | 0.500 | 50000 | larger-M+ACORN (E48) | 0.909 | 645 | exact | 4389 |
| adversarial | all | any | exact | 1.000 | 111–4371 | — | — |
| random | 0.010–0.25 | — | larger-M (D, no ACORN) | 0.90–0.94 | 81–132 | — | — |
| random | 0.500 | 50000 | standard-HNSW (A) | 0.900 | 73 | larger-M | 91 |
| * | ≤0.005 (cand≤500) | | **exact** | 1.000 | 110–140 | — | — |

Reading it:
- **Exact owns the low-candidate-count regime** (cand ≲ 1–2K): fastest *and*
  recall 1.0. Every filter type, every low selectivity. This is already what
  OpenSearch does (`ExactSearcher`, gate at `filterIdsCount ≤ k` and the
  distance-computation cost model).
- **ACORN-γ's *unique* wins are 3 workloads** — cluster at 0.1–1% selectivity —
  where it beats even exact by 2–3× (52 µs vs 140 µs at 0.5%). This is exactly
  its designed regime: a tight, query-distant cluster with too few candidates
  for a standard graph to reach but too many for exact to be cheap.
- **Larger-M + ACORN traversal (E) wins the mid/high-selectivity clustered
  cases** — and is memory-cheaper to build than ACORN-γ (§6). The ACORN-γ *graph*
  rarely beats "a bigger standard graph + ACORN traversal."
- **Random filters never want ACORN** — larger-M or plain standard win.
- **ACORN-1 over the base M=16 graph (B) wins 0 workloads** — whenever ACORN
  traversal helps, a slightly larger graph under ACORN traversal (E) helps more.

---

## 5. Q2 / Q3 — ACORN-γ vs ACORN-1 vs larger-M

**Q2 (ACORN-γ > ACORN-1?)** Sometimes, and only via graph density. On the base
graph ACORN-1 (B) plateaus and wins nothing; ACORN-γ reaches clustered low-sel
regions B cannot (recall 0→1). But the *same* effect is obtained by ACORN
traversal over a larger standard graph (E). So ACORN-γ ≻ ACORN-1 **only because
it is denser**, not because of the two-hop construction per se.

**Q3 (ACORN-γ > simply increasing M?)** **No, not repeatably.** Larger-M+ACORN
(E) wins 8 workloads to ACORN-γ's 3, builds 3–10× faster (§6), and behaves
predictably. ACORN-γ's construction only pulls ahead in the narrow
cluster/very-low-sel band, and there its advantage over E is 1–2 workloads, not a
general trend.

## 6. Index build & memory overhead (Q7) — 100K, d=128, L2

| graph | build (s) | bytes/vec | max deg L0 |
|---|---:|---:|---:|
| standard M=16 (A/B) | 11.6 | 140 | 32 |
| larger-M M=24 (D/E) | 10.8 | 204 | 48 |
| larger-M M=48 (D/E) | 10.3 | 396 | 96 |
| ACORN-γ γ=4 | 12.9 | 185 | 40 |
| ACORN-γ γ=8 | 20.9 | 203 | 40 |
| ACORN-γ γ=16 | 39.3 | 237 | 40 |
| ACORN-γ γ=32 | **101.5** | 307 | 40 |

ACORN-γ construction cost scales ~linearly in γ (it collects `2·M·γ` candidates
per node): **γ=32 costs 8.7× the standard build time** and 2.2× the memory,
while a larger-M standard graph of comparable memory (M=24) builds in the *same*
11 s. ACORN-γ's `edges_pruned` by the two-hop compression is small at these
sizes (level-0 lists rarely overflow their `M_β+1.5M` cap), so most of the cost
buys denser *upper* levels + wider construction search, not compression savings.

## 7. Q8 / Q9 — γ sensitivity & generalization (the operability problem)

Recall by γ (ACORN-γ, ef=250, k=10, 100K L2) is **all-or-nothing and
non-monotonic**:

| filter | sel | γ=4 | γ=8 | γ=16 | γ=32 |
|---|---:|---:|---:|---:|---:|
| cluster | 0.001 | 0.00 | 0.00 | **1.00** | **1.00** |
| cluster | 0.005 | 0.00 | 0.00 | 0.00 | **1.00** |
| cluster | 0.010 | 0.00 | **0.98** | **1.00** | **0.00** |
| cluster | 0.020 | 0.00 | 0.00 | 0.95 | 0.96 |
| correlated | 0.010 | 0.00 | 0.00 | 0.74 | 0.74 |
| correlated | 0.001 | 0.00 | 0.52 | 0.00 | 0.38 |

Recall snaps between ~0 and ~1 as γ crosses a **reachability threshold** for that
(filter, selectivity), and the threshold is not monotone — cluster 1% is 0.98 at
γ=8, 1.0 at γ=16, then **collapses to 0.0 at γ=32**. **Q8: results are
extremely sensitive to γ. Q9: no single γ generalizes** across filters or
selectivities. Because the valid subgraph is either connected to the search
frontier or not, averaging over 100 queries yields 0.0 or 1.0, not a smooth
curve. For a production system this is the decisive problem: γ cannot be chosen
robustly, and mis-choosing it silently returns near-zero recall.

## 8. Q4 / Q5 — where each approach wins

- **Q4 (where ACORN-γ helps):** cluster/correlated filters at **0.1–2%**
  selectivity (cand ~100–2000) whose valid set is a coherent, query-distant
  region — with γ tuned to that selectivity (γ ≈ 1/s). Outside that band it is
  matched or beaten by larger-M+ACORN or exact.
- **Q5 (where exact wins):** whenever the candidate count is low in absolute
  terms (cand ≲ 1–2K here, i.e. selectivity ≤ ~2% at 100K) — across *all* filter
  types. Exact is both fastest and exact there. At 100K, exact at cand=1000 ≈
  170 µs; it only becomes expensive past ~10–25K candidates (0.9–4.4 ms).

## 9. Q6 — regressions

ACORN traversal **regresses recall by 0.3–0.8** on random/uniform filters vs
stock filtered HNSW (§3), and regresses at very high selectivity where the filter
barely matters. ACORN-γ additionally regresses to **recall 0.0** unpredictably
under mis-tuned γ (§7). Stock filtered HNSW itself regresses in *latency* (~7×)
at high ef on selective filters (threshold never tightens).

## 10. Q10 — concurrency

ACORN-γ index, random 10% filter, k=10, ef=128, 100K/d=128
([`raw-results/concurrency.csv`](raw-results/concurrency.csv)):

| threads | QPS | scaling |
|---:|---:|---:|
| 1 | 5,431 | 1.0× |
| 4 | 20,011 | 3.7× |
| 8 | 40,515 | **7.5×** |
| 16 | 37,575 | 6.9× (15-core box; oversubscribed) |

Near-linear to 8 threads. The ACORN read path is stateless per query
(thread-local `DistanceComputer` + `VisitedTable`, `const` graph); the
correctness suite verifies 8-thread result parity with a single-thread
reference. Concurrency does not change the recall conclusions — it only confirms
ACORN adds no shared mutable state and scales like standard HNSW.

## 10b. Generalization across dimension & metric

- **d=768 (high-dim), L2, 100K** reproduces d=128 almost exactly — wins by
  family: exact=22, larger-M+ACORN=10, larger-M=4, **ACORN-γ=3**,
  standard=1; ACORN-γ's wins are the same narrow cluster/correlated
  0.5–1% band ([`raw-results/analysis_100k_d768_l2.md`](raw-results/analysis_100k_d768_l2.md)).
  Higher dimension only raises absolute latencies (distance cost ∝ d), pushing
  the exact-vs-ANN crossover to slightly lower selectivity, without changing the
  family ordering.
- **Inner-product (cosine, normalized), 100K/d=128** reproduces it — wins by
  family: exact=24, larger-M+ACORN=6, larger-M=6, **ACORN-γ=3**, ACORN-1=1;
  ACORN-γ's wins are again cluster 0.5–5% (e.g. cluster 0.5%: C_g32 55 µs vs
  exact 135 µs), and ACORN traversal gives +0.83/+0.84 recall on cluster 2–5%
  ([`raw-results/analysis_100k_d128_ip.md`](raw-results/analysis_100k_d128_ip.md)).
- **1M scale, d=128, L2** (γ∈{4,8,16}) **reinforces the verdict, not reverses
  it** ([`raw-results/analysis_1m_d128_l2.md`](raw-results/analysis_1m_d128_l2.md)).
  Wins by family: **exact=31, larger-M=5, larger-M+ACORN=2, ACORN-γ=1,
  standard=1.** At scale ACORN-γ's advantage *narrows* to a single workload
  (cluster 0.1%, cand=1000: C_g4 **0.974 recall @ 101 µs vs exact 1119 µs — 11×
  faster**). Elsewhere the ANN methods are recall-starved on clustered/correlated
  filters, so **exact wins even at 0.5–5% selectivity** (cand 5K–50K, 1.5–8 ms) —
  it is the only method reaching recall ≥0.9. At high selectivity
  (cluster 50%, cand=500K) larger-M+ACORN wins big (1.2 ms vs exact's 50 ms).
  Net: at scale, the useful regimes are **exact (low cand)** and
  **larger-M+ACORN (high cand)** — ACORN-γ's unique niche shrinks. (γ capped at 16
  here to bound build cost; higher γ might recover a few low-sel clustered points
  but with the same un-tunable, all-or-nothing behaviour of §7.)

**Consistency across L2/IP × d128/d768:** every configuration yields the same
family ordering — **exact 22–24, larger-M+ACORN 6–10, larger-M 4–6, ACORN-γ
exactly 3 (the cluster/correlated low-selectivity band), standard/ACORN-1 0–1.**

---

## 11. Analysis summary (the 12 questions)

1. **Does ACORN traversal improve standard filtered search?** Yes — dramatically
   (+0.3…+0.7 recall) on clustered/correlated filters; **no** (−0.6…−0.8) on
   random filters. Conditional on filter–vector correlation.
2. **ACORN-γ over ACORN-1?** Only by being denser; the gain is reproduced by
   ACORN traversal over a larger standard graph.
3. **ACORN-γ over larger-M?** No, not repeatably (8 vs 3 workload wins for
   larger-M+ACORN), and larger-M builds 3–10× faster.
4. **Where does ACORN-γ help?** cluster/correlated at 0.1–2% selectivity with γ
   matched to selectivity.
5. **Where is exact better?** All filters at low candidate counts (≲1–2K here).
6. **Regressions?** Recall −0.3…−0.8 on random filters; recall→0 under mis-tuned
   γ; stock-HNSW latency blow-up at high ef.
7. **Overhead?** ACORN-γ: up to 8.7× build time, 1.3–2.2× memory.
8. **Sensitivity to γ?** Extreme; all-or-nothing, non-monotonic.
9. **Does one γ generalize?** No.
10. **Useful under concurrency?** Read path scales like standard HNSW (no shared
    state); does not change the recall conclusions.
11. **Large enough to justify an OpenSearch mapping feature?** No — the robust
    wins (low-selectivity → exact) are already covered by OpenSearch, and the
    ACORN-γ-unique wins are narrow and operationally fragile.
12. **Upstreamable to Faiss?** Yes in principle — the reference ACORN *is* a Faiss
    fork, and this prototype is a clean `faiss::HNSW`-layer module. But upstream
    value is limited by the same instability.

---

## 12. Faiss upstreaming feasibility (report §15)

Mechanically straightforward: the ACORN traversal is a self-contained addition to
the HNSW search layer reusing `IDSelector`, and ACORN-γ is a `set_default_probas`
+ neighbor-selection variant — the reference implementation is already a Faiss
fork. A production patch would be `0011-Add-ACORN-hnsw-search-and-acorn-gamma-build`.
The blocker is not feasibility but value: the γ graph's instability and cost make
it a hard sell upstream; the *traversal* alone (an `IDSelector`-aware
"predicate-subgraph" search mode) is the more defensible upstream contribution.

## 13. Production gaps remaining (report §16)

Serialization is a prototype format (not Faiss-compatible); no Java mapping / SQ /
PQ / byte / binary / nested / IVF / GPU support; γ auto-selection unsolved (and,
per §7, may be unsolvable robustly); no real-embedding validation; single-machine
scale (≤1M here vs the paper's 1M–100M). See design doc §scope.

## 14. Proposed OpenSearch surface *if* results justified it (report §17)

They do not (§11.11). Had they, the minimal surface would be a per-field method
parameter block (test-only today), e.g.
`method.parameters.filter_search = { mode: "acorn", gamma: <int> }`, plus a
query-time `ef_search` override — deliberately **not** an automatic planner.

---

## 15. Final recommendation

The **ACORN traversal** is a real, valuable idea for *metadata-correlated* filters
(the common OpenSearch case): it recovers 0.3–0.7 recall where stock filtered
HNSW routes away from the valid set. But:

- Its benefit is **conditional** (helps clustered/correlated; hurts random).
- The dedicated **ACORN-γ graph does not repeatably beat a larger-M standard
  graph under ACORN traversal**, costs up to ~9× to build, and is **operationally
  fragile** (all-or-nothing, non-monotonic γ; no γ generalizes).
- The regime where ACORN-γ is uniquely best (very low selectivity, clustered) is
  narrow, and the broader low-selectivity regime is **already** owned by
  OpenSearch's existing **exact** fallback.

A full ACORN-γ mapping feature is therefore **not** justified on this evidence.
The portable, defensible win is the ACORN *traversal* as an optional filtered
search mode over the existing (optionally larger-M) graph — i.e. **ACORN-1** —
not the ACORN-γ graph-construction machinery. The 1M scale run (§10b) **reinforces**
this: ACORN-γ's unique wins shrink from 3 (at 100K) to 1 (at 1M) while exact and
larger-M+ACORN absorb the rest. The result is consistent across 100K/1M × d128/d768
× L2/IP (§10b), so it is not an artifact of one configuration.

### VERDICT

```
ACORN-1 ONLY
```

*(ACORN traversal has conditional value and is worth prototyping further as an
optional search mode; the ACORN-γ graph-construction feature is not justified —
it fails to repeatably beat larger-M+ACORN while adding large build cost and
un-tunable γ. Revisit only with real-embedding workloads at ≥10M scale.)*
