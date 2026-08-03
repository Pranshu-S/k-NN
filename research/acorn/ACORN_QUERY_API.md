# ACORN on OpenSearch native Faiss HNSW — Final Report

Experimental POC. Two phases: **ACORN-1** (search-time only, no index build) and
**ACORN-γ** (denser graph built at index time). All benchmarks are on **real SIFT1M**
(128-D, L2), 50 real held-out queries, **per-query-local correlation** (positive =
eligible nearest the query, negative = farthest, none = a random/scattered filter),
selector = **bitset** (as OpenSearch ships). "standard" = stock faiss
`IndexIDMap::search` + `IDSelector` (the real OpenSearch path). Cells report the best
recall over ef ∈ {50,100,250,500}; `ndis` = distance computations; `examined` =
graph-neighbour slots inspected; latency = mean wall-clock per query.

---

# Phase 1 — ACORN-1

## 1. How it was implemented
`jni/src/acorn_hnsw.cpp` — a **filtered, two-hop, bounded best-first traversal over the
standard OpenSearch-built HNSW graph** (no index/build change). Key points:

- **Predicate subgraph:** only filter-**passing** nodes enter the result heap and the
  frontier; filter-**rejected** nodes act as **two-hop routing bridges** — the walk
  routes *through* a rejected node into its passing 2-hop neighbours. This is what lets
  it reach eligible nodes on a sparse graph without a denser build.
- **ID space:** HNSW traverses internal ordinals; the OpenSearch selector is defined
  over external Lucene doc ids. Every `is_member` call translates internal→external via
  `IndexIDMap::id_map` (bounds-checked), matching faiss's `IDSelectorTranslated`.
- **Correctness fixes:** separate `admitted` / `bridged` visited-state (each node bridged
  at most once), unique eligible counting, normalized `ef`, plain greedy upper-level
  descent, `gamma==1` enforced (ACORN-γ rejected here).
- **Performance:** distances computed **SIMD-batched** 4-at-a-time
  (`DistanceComputer::distances_batch_4`); per-query state uses `VisitedTable`, not hash
  sets.
- **Verification:** 16/16 native unit tests (`test_acorn_jni.cpp`); differential vs the
  research reference on a shared graph (recall parity/better; repeated bridge expansions
  eliminated).

## 2. How it is queried
A normal k-NN query with a filter, plus one field in `method_parameters`:
```json
"method_parameters": { "ef_search": 100, "filtered_search_mode": "acorn" }
```
Validated to **FAISS engine + HNSW method + a filter + non-radial**; unknown values
rejected (no silent fallback). Dispatched in `faiss_wrapper.cpp`; opt-in, default off.

## 3. Results — standard vs ACORN-1 (SIFT 128-D), "no correlation"

### @ 100,000
| sel | standard: rec / ndis / examined / lat | ACORN-1: rec / ndis / examined / lat |
|---|---|---|
| 0.1% | 0.40 / 4385 / 16248 / 511µs | 0.04 / 89 / 673 / 12µs |
| 0.5% | 0.87 / 4385 / 16248 / 515µs | 0.51 / 192 / 41283 / 307µs |
| 1% | 0.95 / 4385 / 16248 / 517µs | 0.84 / 572 / 159642 / 1232µs |
| 2% | 0.99 / 4385 / 16248 / 517µs | 0.98 / 879 / 175017 / 1461µs |
| 5% | 1.00 / 4385 / 16248 / 527µs | 1.00 / 1254 / 84711 / 792µs |
| 10% | 1.00 / 4385 / 16248 / 516µs | 1.00 / 2135 / 77584 / 857µs |
| 25% | 1.00 / 4385 / 16248 / 525µs | 1.00 / 4348 / 61865 / 984µs |
| 50% | 1.00 / 2678 / 8250 / 241µs | 1.00 / 6588 / 39116 / 996µs |

### @ 1,000,000
| sel | standard: rec / ndis / examined / lat | ACORN-1: rec / ndis / examined / lat |
|---|---|---|
| 0.1% | 0.43 / 6602 / 16327 / 923µs | 0.06 / 121 / 1369 / 63µs |
| 0.5% | 0.82 / 6602 / 16327 / 906µs | 0.69 / 738 / 218517 / 5717µs |
| 1% | 0.91 / 6602 / 16327 / 852µs | 0.92 / 1307 / 243332 / 4658µs |
| 2% | 0.97 / 6602 / 16327 / 888µs | 0.99 / 2151 / 235001 / 4805µs |
| 5% | 0.99 / 6602 / 16327 / 857µs | 1.00 / 4386 / 218977 / 4453µs |
| 10% | 1.00 / 6602 / 16327 / 856µs | 1.00 / 7636 / 198287 / 4498µs |
| 25% | 1.00 / 6602 / 16327 / 867µs | 1.00 / 7348 / 54214 / 1825µs |
| 50% | 1.00 / 6602 / 16327 / 878µs | 0.99 / 15333 / 58466 / 2936µs |

**Other correlations (both scales):** *positive* — standard is ~1.00 and fast (~850µs @1M);
ACORN-1 matches recall but is slower (more examination). *negative* — standard is ~0.00
(beam never reaches the far eligible region → OpenSearch exact fallback); ACORN-1 recovers
partial recall (100K: up to ~0.8; 1M: ~0.3) but expensively.

## 4. Why ACORN-1 is worse than standard — and where it wins (high dimensions)

**The mechanism:** ACORN-1 **trades distance computations for neighbour inspections.**
It computes far *fewer* distances (only passing nodes) but *inspects far more* neighbours
(the 2-hop bridge = neighbours-of-neighbours). Measured at 100K/5%:

| | distances (`ndis`) | neighbours examined | latency |
|---|---|---|---|
| standard | 4385 | 16,248 | 527µs |
| ACORN-1 | **1254 (3.5× fewer)** | **84,711 (5× more)** | 792µs |

On SIFT (128-D) a distance is **cheap** (SIMD, ~0.1µs), so the **inspection overhead
dominates** and ACORN-1 loses. At 1M it inspects even more (~200K) → ~5× slower. This is
structural, not an implementation gap — 2-hop bridging is O(M²) graph work per node vs
O(M) for standard.

**Where ACORN-1 WINS — high dimensions (expensive distances).** When a distance costs
far more than an inspection, ACORN-1's *fewer distances* outweigh its *more inspections*.
Reference (SIFT tiled to raise dimensionality, same NN structure, "no correlation"):

| dimensionality | 5% selectivity: standard lat | ACORN-1 lat | winner |
|---|---|---|---|
| 128-D (native SIFT) | 527µs | 792µs | standard (1.5× faster) |
| 1024-D (×8) | 1146µs | 1303µs | ~tie (gap shrinks to 1.14×) |
| **4096-D (×32)** | **1946µs** | **1134µs** | **ACORN-1 1.7× faster** |

At 4096-D, 10% selectivity ACORN-1 is also ~1.4× faster (805µs vs 1103µs). So ACORN-1 is
worth it **only when distance computation dominates** — very high-dimensional vectors (e.g.
GIST-960 and beyond) — at moderate selectivity. On typical 128-D embeddings it is not.

---

# Phase 2 — ACORN-γ

## 1. How it was implemented
A **denser HNSW graph built once at index time**, then **filtered 1-hop** search over the
predicate subgraph (no 2-hop needed — the dense graph already has enough passing
neighbours per node). Prototype: `research/acorn/src/bench_gamma.cpp`.

- **Graph:** standard faiss HNSW with **M = γ · M_base** (e.g. γ=12 → M=192 → level-0
  degree ~384 vs standard's 32). Pays the neighbour expansion **once at build** instead of
  per query.
- **Search (`gamma_search`):** plain greedy descent to level 0, then bounded best-first
  where the **frontier is passing-nodes-only**, distances **SIMD-batched**, 1-hop only.
- **Status:** benchmarked standalone (faithful, optimized). **Not yet wired into the
  OpenSearch JNI index-build path** — that is the follow-up if the win matches the workload.

## 2. How it would be queried
Same `filtered_search_mode` channel, but ACORN-γ additionally requires the **denser index
built at ingest** (a mapping/index-time setting for γ), because the graph itself differs.
`gamma` is an index-build parameter, not a pure query knob.

## 3. Results — standard vs ACORN-γ (SIFT 128-D), "no correlation"

### @ 100,000 (γ = 12, dense M = 192)
| sel | standard: rec / ndis / lat | ACORN-γ: rec / ndis / lat | winner |
|---|---|---|---|
| 1% | 0.95 / 4385 / 521µs | 0.44 / 410 / 121µs | 🔴 γ recall collapse |
| 5% | 1.00 / 4385 / 510µs | 0.95 / 1453 / 588µs | 🔴 γ slower |
| **10%** | 1.00 / 4385 / 516µs | **1.00 / 1623 / 357µs** | 🟢 **γ 1.4×** |
| **25%** | 1.00 / 4385 / 522µs | **1.00 / 1310 / 128µs** | 🟢 **γ 4.0×** |
| 50% | 1.00 / 2678 / 244µs | 1.00 / 3205 / 347µs | ⚪ tie/slower |

**Other correlations @100K:** *positive* — standard already fast (~95µs); γ slower
(~160–250µs). *negative* — both fail (γ ~0.1, standard ~0.0). γ's win is confined to
**scattered ("no correlation") filters at ~10–30% selectivity**.

### @ 1,000,000 (γ = 8, dense M = 128)
(γ=8 not 12 here — the 1M dense build had to stay feasible; less dense ⇒ smaller win.)

| sel | standard: rec / ndis / lat | ACORN-γ: rec / ndis / lat | winner |
|---|---|---|---|
| 1% | 0.91 / 6602 / 896µs | 0.14 / 482 / 197µs | 🔴 γ recall collapse |
| 5% | 0.99 / 6602 / 842µs | 0.97 / 2192 / 849µs | ⚪ tie |
| 10% | 1.00 / 6602 / 840µs | 0.98 / 3598 / 1000µs | 🔴 γ slower |
| **25%** | 1.00 / 6602 / 851µs | **1.00 / 4588 / 740µs** | 🟢 **γ 1.15×** |
| **50%** | 1.00 / 6602 / 868µs | **1.00 / 4182 / 519µs** | 🟢 **γ 1.67×** |

At 1M the win is **smaller and shifts to higher selectivity (25–50%, 1.15–1.67×)** vs 100K's
(10–25%, up to 4×) — because γ=8 is less dense than the γ=12 used at 100K. A denser 1M build
(γ=12+) would likely restore a larger win at 10–25%, at proportionally larger index size.
Low selectivity (<5%) still collapses; a higher γ or a pre-filter fallback is needed there.

## 4. Index size cost — this is what you pay for the query speedup

Measured (`measure_size.cpp`, d=128, per-node so N-independent; graph =
`hnsw.neighbors` array, total = graph + flat vectors):

| γ | M | graph B/node | **graph size ×** | **total index ×** (vectors+graph) |
|---|---|---|---|---|
| 1 (standard) | 16 | 132 | 1.0× | 1.00× |
| 4 | 64 | 516 | 3.9× | 1.60× |
| 8 | 128 | 1028 | 7.9× | 2.39× |
| **12** | 192 | 1540 | **11.8×** | **3.18×** |
| 24 | 384 | 3075 | 23.5× | 5.57× |

So the γ=12 graph that gave the **4× query speedup at 25% selectivity costs a 3.2× larger
total index** (the *graph* alone is ~12×; vectors are unchanged, and at 128-D they dilute
the total to ~3.2×). γ=24 is 5.6× total. Build time also rises sharply (dense
neighbour selection per insert). **This is the size-for-speed trade, quantified.**

### Compression — IMPLEMENTED and measured (it cuts size but loses the speed)
The paper's two-hop compression heuristic (`shrink_neighbor_list`: keep the first `M_beta`
neighbours, drop any further neighbour already two-hop-reachable) is implemented in
`bench_compress.cpp`, with adaptive 2-hop at search time (fires only when 1-hop finds
< M_base passing) to recover pruned nodes. Measured @100K, γ=12:

| config | level-0 edges kept | **total index ×** | 25%-sel latency (no-corr) |
|---|---|---|---|
| standard | — | 1.00× | 557µs |
| **γ-dense (1-hop)** | 100% | **3.21×** | **159µs (4× faster)** 🟢 |
| γ-compressed M_beta=32 | 12.6% | **1.10×** | 673µs (slower) 🔴 |
| γ-compressed M_beta=96 | ~30% | 1.41× | 1091µs (slower) 🔴 |

**Compression slashes the size (3.2× → 1.1×) and keeps recall (adaptive 2-hop recovers
pruned nodes) — but it destroys the speed win: the compressed γ is *slower than standard*.**
This is fundamental: the γ speedup comes from the dense graph (1-hop suffices); compression
re-sparsifies it → 1-hop finds too few → adaptive 2-hop fires → back to the expensive
neighbour inspection (the ACORN-1 cost). Raising M_beta adds size without reliably
recovering speed.

**Trilemma — no config is both small and fast:**
- standard: small + baseline-fast (best all-rounder)
- γ-dense: fast (up to 4×) but **3.2× the index**
- γ-compressed: small (**1.1×**) but **slower than standard**

The dense graph's size **is** the price of the speed. Compression converts "big+fast" into
"small+slow" (strictly worse than standard), so it does not rescue the size/speed trade —
it just moves along it.

---

# Phase 3 — RACORN (ASF + AEF over all three bases)

RACORN adds two fallbacks to a base filtered traversal, to recover the regimes where
graph search collapses (low selectivity, negative correlation). Applied to all three
bases: **STD** (faiss-style in-filtering), **ACORN-1** (2-hop), **ACORN-γ** (dense 1-hop).
Prototype: `research/acorn/src/bench_racorn.cpp`.

## 1. How it was implemented
- **ASF (Adaptive Search Fallback):** when a popped node's expansion yields fewer than
  `M_base` passing neighbours, admit **stride-sampled *failing* neighbours** as transient
  frontier bridges (never results) — keeps the walk alive when the predicate subgraph
  fragments. Recovers recall without leaving the graph.
- **AEF (Adaptive Exact Fallback):** track the running pass-ratio (passed/examined); once
  enough nodes are explored, if it is below `aef_thr`, abandon the walk and run **exact
  search over the eligible set** (recall 1.0; cheap when selectivity is tiny).

## 2. How it would be queried
Same `filtered_search_mode` channel (`racorn` / `racorn_plus`). Note **AEF is essentially
OpenSearch's existing exact-search-below-a-cardinality-threshold** behaviour, surfaced
inside the traversal; `aef_thr` is the trigger knob.

## 3. Results (real SIFT1M, recall / latency; AEF fires → exact)

### Negative correlation @ 100K (eligible far from query)
| sel | STD | STD+ASF+AEF | ACORN-1 | +ASF | +ASF+AEF | γ | +ASF+AEF |
|---|---|---|---|---|---|---|---|
| 0.1% | 0.00 | **1.00**/135µs | 0.00 | 0.00 | **1.00**/127µs | 0.00 | **1.00**/131µs |
| 5% | 0.00 | **1.00**/355µs | 0.00 | 0.00 | **1.00**/353µs | 0.00 | **1.00**/351µs |
| 25% | 0.00 | **1.00**/973µs | 0.00 | 0.00 | **1.00**/979µs | 0.00 | **1.00**/970µs |

Every base 0.00 → 1.00, **all via AEF** (exact). **ASF alone stays 0.00** — bridges cannot
reach eligible nodes that are simply far away. Latency = exact cost over the eligible set
(grows with selectivity).

### No correlation @ 100K, low selectivity (ACORN collapse regime)
| sel | ACORN-1 | +ASF | +ASF+AEF | γ | +ASF | +ASF+AEF |
|---|---|---|---|---|---|---|
| 0.1% | 0.04 | **0.93**/1033µs | **1.00**/135µs | 0.02 | **0.80**/1180µs | **1.00**/140µs |
| 1% | 0.84 | **1.00**/1060µs | **1.00**/234µs | 0.44 | **1.00**/1196µs | **1.00**/233µs |

Both fallbacks recover recall; **ASF via graph bridging (~1000µs), AEF via exact (~135µs)**.
At very low selectivity AEF is cheaper *and* higher-recall, so it dominates ASF.

### Moderate–high selectivity (5–25%, no correlation)
All bases already ~1.00 without fallback (AEF does not fire; ASF barely changes it). Latency
is set by the base traversal — γ usually fastest (dense graph), e.g. 10%: γ 303µs vs STD 560µs.

### @ 1,000,000 (γ=8) — same shape, but AEF's exact cost grows with N × selectivity
| corr | sel | base (STD/ACORN1/γ) | +ASF+AEF (all bases) |
|---|---|---|---|
| neg | 0.1% | 0.00 | **1.00** / ~1.2ms (exact) |
| neg | 5% | 0.00 | **1.00** / ~4.9ms (exact over 5% of 1M) |
| neg | 25% | 0.00 | **1.00** / ~11.2ms (exact over 250k) |
| no | 0.1% | 0.01–0.43 | **1.00** / ~1.3ms (AEF) |
| no | 1% | 0.14–0.92 | **1.00** / ~2.5ms (AEF); ASF alone recovers too (γ 0.14→1.00) |
| no | 10–25% | ~1.00 | ~1.00 (base traversal; γ fastest, ~1.0ms) |

Same conclusion at scale — **but note AEF's exact fallback is O(N·selectivity)**: cheap at
low selectivity (tiny eligible set, ~1ms), but at **25% negative correlation on 1M it costs
~11ms** (exact over 250k eligible). So AEF is the right move only when the eligible set is
small; at high selectivity + negative correlation there is no cheap option (the graph can't
reach the eligible region, and exact is inherently O(eligible)).

## 4. Verdict on RACORN
1. **AEF (exact fallback) is the workhorse** — base-agnostic, it fixes **both** hard regimes
   (negative correlation *and* low selectivity), taking every base to recall 1.0, at exact
   cost (cheap at low sel). But **AEF ≈ what OpenSearch already ships** (exact below a
   cardinality threshold), so its value is largely already present.
2. **ASF is secondary** — recovers scattered low-selectivity recall via the graph (useful
   when N is huge and an exact scan is unaffordable), but it is expensive and **dominated by
   AEF when exact is affordable**, and it does **nothing for negative correlation**.
3. **With RACORN all three bases become robust** (recall 1.0 everywhere). The base then only
   matters for latency in the moderate-selectivity band where the walk actually runs — where
   γ is usually fastest (at 3.2× index size).

---

# Verdict

| | ACORN-1 | ACORN-γ |
|---|---|---|
| Index build | none (search-time only) | denser graph: **3.2× index** (γ=12) for the speed win. Compression (implemented) shrinks it to **1.1×** but then loses the speed → strict trilemma, no small+fast config. |
| Beats standard on 128-D SIFT? | **No** (1.5–5× slower; inspection-bound) | **Yes, 1.4–4×**, but only for scattered filters at ~10–30% selectivity |
| Where it wins | high-dimensional vectors (expensive distances), moderate selectivity | scattered filters, moderate selectivity |
| Loses | positive correlation, low & high selectivity, low dimensions | positive correlation, low & high selectivity |
| Negative correlation | partial recall recovery (expensive) | both fail |

**Bottom line:** on ordinary 128-D embeddings, ACORN-1 has no wall-clock advantage over
faiss's native filtered search — its 2-hop inspection cost dominates. ACORN-γ *does* win
(1.4–4×) for scattered, moderately-selective filters, but only by paying for a much denser
index. ACORN-1 becomes competitive only when distances are expensive (very high
dimensions). Neither is a universal win; both are opt-in tools for specific workloads.
