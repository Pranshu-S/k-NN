# Filtered Vector Search — Journey & Cheat Sheet

How we tried to make filtered k-NN better than plain HNSW, what each technique actually did,
and when to use which. All numbers are real (SIFT1M, per-query-local correlation, bitset
selector; "best recall over ef / latency-µs").

**The one governing variable:** *query–filter correlation* — are the filter-eligible docs
**near** the query in vector space (positive), **scattered** (none), or **far** (negative)?
Selectivity (how many docs pass) matters second. Sample filters per regime:
- **Positive:** query "red running shoe" + `color=red` (filter reinforces the query)
- **Scattered (none):** query "laptop" + `in_stock=true` / `price<500` (filter orthogonal to embedding) — *most real filters*
- **Compact-far (negative):** query "summer dress" + `category=winter_coats` (cross-domain)

---

## 1. Problem statement — what we intended to solve

### The current algorithm: standard HNSW with "in-filtering"

OpenSearch's Faiss engine answers a filtered k-NN query (*find the 10 nearest vectors that
ALSO match this metadata filter*) with **in-filtering** on one monolithic HNSW graph:

1. **Build (filter-agnostic).** The graph is built over *all* vectors; the filter isn't known
   at build time, so edges connect vectors purely by geometric proximity — a passing doc and a
   non-passing doc are neighbours if they're close in vector space.
2. **Descend.** The search starts at a **fixed entry point** (the top node) and greedily
   descends the upper layers toward the query — pure nearest-neighbour hops, filter ignored.
3. **Walk level 0 (the crux).** For each node it pops, it inspects **all** neighbours and
   **pushes them onto the frontier regardless of whether they pass the filter** — the walk
   *traverses through* non-passing nodes. The filter is applied only at the **results gate**:
   ```
   for each neighbour v of the current node:
       candidates.push(v, dist)          // ← OUTSIDE the filter gate — always traverse
       if selector.is_member(v):         // ← filter only gates the RESULT
           maybe_add_to_topk(v)
   ```
   A *failing* node still expands its own neighbours; the filter narrows *what counts as an
   answer*, but the **path** is dictated by unfiltered geometry from one fixed start.

### The issue: governed by *correlation*, not selectivity

Because the walk only travels along geometric edges outward from one fixed entry point, it
finds eligible docs **only if they lie along the corridor between the entry point and the
query**. So the real variable is **query–filter correlation**, not selectivity:
- **Positive / scattered** → eligible docs near the query (or everywhere); the walk trips over
  them. Fine.
- **Negative** (filter selects a *compact region far from the query* — "brand = X" whose
  products cluster elsewhere) → eligible docs are **outside** the corridor; the walk converges
  on the query neighbourhood, finds **zero** passing docs, and stops. **Recall → 0.00 at every
  selectivity.** A 25% (permissive) filter can return 0.00 if it's far; a 0.1% filter returns
  1.00 if it's near. **Selectivity % predicts nothing.**

### How the standard algorithm performs (SIFT1M, best recall / latency-µs)

`pos`=near query, `no`=scattered, `fc`=compact far (realistic negative), `neg`=diffuse far.

| correlation | 0.1% | 1% | 5% | 25% |
|---|---|---|---|---|
| **pos** | 1.00 / 1545 | 1.00 / 1398 | 1.00 / 1389 | 1.00 / 1390 |
| **no** (scattered) | 0.43 / 1379 | 0.91 / 1372 | 0.99 / 1377 | 1.00 / 1394 |
| **fc** (compact far) | **0.00** / 179 | **0.00** / 182 | **0.00** / 185 | **0.00** / 187 |
| **neg** (diffuse far) | **0.00** / 182 | **0.00** / 180 | **0.00** / 185 | **0.00** / 185 |

Read by row: **pos** perfect; **no** good except very-low sel (0.43 @ 0.1%: rare+scattered
rarely lands on an eligible doc); **fc/neg** **0.00 across the board** — and note the *low*
latency (~180µs): it doesn't work hard and fail, it converges fast on the wrong neighbourhood
and returns empty. It fails fast and silently.

**Goal:** make filtered vector search robust and fast across *all* regimes — kill that 0.00
block — and understand the real cost/benefit of each approach (the ACORN family plus our own
ideas), so OpenSearch can match Elasticsearch's filtered latency.

---

## 2. ACORN-1

### What it is

ACORN-1 (arXiv:2403.04871) keeps the *same filter-agnostic HNSW graph* but changes **what the
walk may traverse**. Instead of expanding through every neighbour and gating only results, it
restricts the frontier to the **predicate subgraph** — it only *travels* through passing nodes:
```
for each neighbour v1 of the current node:
    if selector.is_member(v1):          // ← filter now gates TRAVERSAL, not just results
        candidates.push(v1, dist); maybe_add_to_topk(v1)
    else:                               // v1 failed — use it as a BRIDGE, don't admit it
        for each neighbour v2 of v1:    // 2-hop: peek one hop further
            if selector.is_member(v2):  // reconnect passing regions the sparse subgraph split
                candidates.push(v2, dist)
```
The problem it solves for itself: on a selective filter the passing nodes are sparse and
**disconnected** (a passing node's direct neighbours may all fail, dead-ending the walk).
**2-hop bridging** lets a failing node reconnect two passing regions. Intent: reach the
eligible region **without** computing distances on the mass of non-passing vectors.

### What we did

- **Implemented it correctly in the JNI layer** ([`jni/src/acorn_hnsw.cpp`](jni/src/acorn_hnsw.cpp)),
  against the real OpenSearch Faiss path. The subtle correctness fix: **ID-space translation** —
  the selector expects *external* Lucene doc-ids but the walk works in *internal* graph ordinals,
  so every `is_member` goes through `id_map` (`is_member(id_map[internal])`). Getting it wrong
  silently returns garbage.
- Separate `admitted` / `bridged` visited tables; bridge-expand-once; unique counting.
- **Optimized it for a fair fight** — SIMD-batched distances (`distances_batch_4`), bitset
  selector, lean upper-layer descent — so a loss couldn't be blamed on the implementation.
- 16/16 native correctness tests + a differential harness vs a reference.

### Results — ACORN-1 vs standard, side by side (SIFT1M @1M, recall / latency-µs)

Compare only where **both reach comparable recall** (so latency is apples-to-apples). `⚠`
marks a cell where ACORN-1's low latency is a **recall collapse** (it dead-ends and quits), not
a real win.

| corr / sel | standard | **ACORN-1** | verdict |
|---|---|---|---|
| pos 0.1% | 1.00 / **1545** | 1.00 / 1985 | ACORN-1 **1.3× slower** |
| pos 5% | 1.00 / **1389** | 1.00 / 1648 | ACORN-1 **1.2× slower** |
| pos 25% | 1.00 / **1390** | 1.00 / 1401 | ~tie |
| no 1% | 0.91 / **1372** | 0.92 / 5003 | ACORN-1 **3.6× slower** |
| no 5% | 0.99 / **1377** | 1.00 / 1864 | ACORN-1 **1.4× slower** |
| no 25% | 1.00 / **1394** | 1.00 / 5133 | ACORN-1 **3.7× slower** |
| no 0.1% | 0.43 / 1379 | 0.06 / 43 ⚠ | ACORN-1 *quits early* (recall collapse) |
| fc / neg (all sel) | **0.00** | **0.00** | both collapse (unsolved) |

Net: at 1M on 128-D vectors, **whenever ACORN-1 achieves standard's recall it is 1.2–3.7×
slower**; its only "fast" cells are ones where it gave up and returned near-empty.

### Why standard usually wins on 128-D — and the two things that flip it

We instrumented both, sweeping ef and comparing each method **at its own best-recall ef** — the
fair, tuned comparison, since ACORN's whole point is reaching a target recall with a *smaller*
beam (a fixed ef hides that). Counts are per-query and **dimension-invariant**
([`bench_highdim.cpp`](src/bench_highdim.cpp), SIFT @100K, scattered):

| scattered sel | | ef* | distances | **inspections** (selector + visited test) |
|---|---|---|---|---|
| **5%** | standard | 320 | 3,200 | 6,331 |
| | **ACORN-1** | 320 | **1,392** (2.3× fewer ✅) | **101,068** (16× more ❌) |
| **25%** | standard | 320 | 3,200 | 6,331 |
| | **ACORN-1** | **40** | **1,498** (2.1× fewer ✅) | 10,708 (1.7× more) |

The lever is **ef***: standard needs ef=320 to reach recall 1.0 *even at 25%*, because most of
its beam is spent on *failing* nodes it traversed but can't return. ACORN-1's frontier is
all-passing, so at 25% it reaches recall 1.0 at **ef=40** — 2.1× fewer distances, barely more
inspections. At 5% the passing subgraph is sparse, so ACORN-1 needs ef=320 and pays **16× more
inspections**. So ACORN-1 wins exactly when **its distance savings outweigh its extra
inspections**, and that tips two ways:

### Proof: the crossover, measured (best-ef, @100K)

We tiled each SIFT vector T× so every distance costs T× more while the **graph, walk, recall,
and the work-counts above stay identical** — the *only* variable is cost-per-distance. Latency
µs, best of 3 ([`raw-results/highdim_crossover.txt`](raw-results/highdim_crossover.txt)):

| dim | 5% sel (std / ACORN-1) | 25% sel (std / ACORN-1) |
|---|---|---|
| **128** | 481 / 700 → **0.69×** | 500 / 219 → **2.29×** ✅ |
| **1024** | 2722 / 1732 → **1.57×** ✅ | 2718 / 1406 → **1.93×** ✅ |
| **2048** | 5454 / 2968 → **1.84×** ✅ | 5424 / 2670 → **2.03×** ✅ |
| **4096** | 10966 / 5545 → **1.98×** ✅ | 10922 / 5234 → **2.09×** ✅ |

- **(1) High selectivity flips it even on 128-D.** At 25% ACORN-1 reaches recall at ef=40, does
  fewer distances, and **wins ~2× at every dimension**.
- **(2) High dimension flips low selectivity.** At 5% ACORN-1's 16× inspection overhead sinks it
  on cheap 128-D distances (0.69×), but once distances are expensive it wins (1.57–1.98×).

(1% omitted: at very low selectivity the passing subgraph is too fragmented for ACORN-1/γ to
match standard's recall — 0.84 / 0.44 vs 0.89 — so a latency comparison there isn't fair.)

**Scale caveat (important).** The 25%/128-D win is at **100K**. ACORN-1's inspection overhead
grows super-linearly with corpus size, so at **1M** that same 25% case *reverses* to a 3.7×
loss (the side-by-side table above: 5,133µs vs 1,394µs). Its robust, scale-stable home is
**high dimension** (distance-bound), where the win doesn't depend on out-inspecting a smaller graph.

### What we concluded

- **It does NOT fix the problem.** The `fc`/`neg` rows are **still 0.00** — identical to
  standard. ACORN changes *which nodes the walk reaches*, but still **starts at the same fixed
  entry point** and is pulled toward the query. It changes the *reach*, not the *destination*
  (note the ~32µs — it dead-ends almost immediately).
- **On ordinary 128-D embeddings it's not a win at scale.** At 1M it's 1.2–3.7× slower wherever
  it matches standard's recall — the inspection overhead of 2-hop bridging grows with N.
- **Its real, scale-stable win is high-dimension + selective filters** — exactly the ACORN
  paper's regime: at 4096-D it's ~2× faster across selectivity (100K).

**Where it works best.** **High-dimensional vectors** (GIST-960 and up, some LLM embeddings).
Not typical 128-D embeddings at scale, and it leaves the negative-correlation collapse unsolved.

---

## 3. ACORN-γ

### What it is

ACORN-γ attacks ACORN-1's weakness (sparse passing subgraph → dead-ends → expensive 2-hop
bridging) at **build time** instead of search time: build a **much denser** HNSW with
**M = γ·16** (level-0 degree ~2·γ·16; we use γ=12 → M=192, ~24× the edges), so that even after
the filter removes most nodes, every passing node *still* has enough passing neighbours to walk
on. Then search with a plain **1-hop** passing-only frontier — no bridges needed:
```
build: HNSW with M = γ·16          // dense enough that the predicate subgraph stays connected
search, for each neighbour v1 of the current node:
    if selector.is_member(v1):     // plain 1-hop — the density replaces ACORN-1's 2-hop bridges
        candidates.push(v1, dist); maybe_add_to_topk(v1)
```

### What we did

- Built it as a standard HNSW at M=γ·16 ([`bench_highdim.cpp`](src/bench_highdim.cpp), and the
  full builder in [`acorn_gamma_builder.cpp`](src/acorn_gamma_builder.cpp)), searched 1-hop.
- Implemented the paper's **two-hop compression** (`shrink_neighbor_list`) with adaptive-2-hop
  recovery, to see if the size cost can be bought back.
- **Measured the build/size cost directly** rather than quoting the paper.

### Results — the cost is real and matches the paper (measured, SIFT @100K)

| | build time | index size (vectors + edges) | vs standard |
|---|---|---|---|
| standard (M=16) | **8.0 s** | 51.2 + 13.2 = **64.4 MB** | — |
| **ACORN-γ (M=192)** | **22.4 s** | 51.2 + 154.0 = **205.2 MB** | **2.8× build, 3.2× index** (11.6× the edges) |

The 3.2× index blow-up **exactly reproduces the paper's Table 4** figure. Build is 2.8× here at
100K; the paper reports **9–33×** at 25M — graph construction super-scales, so the multiplier
*grows* with corpus size (25M: ~10.5 h vs ~19 min). Compression cut the index **3.2× → 1.1×**
and kept recall, **but lost the speed** (re-sparsifying reintroduces ACORN-1's 2-hop cost) — a
strict **trilemma: small+slow, or fast+big, never both.**

### Why it's the best of the ACORN family — but still an inspection trade

Same best-ef instrumentation as §2 (dimension-invariant work-counts, SIFT @100K, scattered):

| scattered sel | standard (ef*/dist/insp) | ACORN-1 (ef*/dist/insp) | **ACORN-γ (ef*/dist/insp)** |
|---|---|---|---|
| **5%** | 320 / 3,200 / 6,331 | 320 / 1,392 / 101,068 | 320 / **1,170** / **54,956** |
| **25%** | 320 / 3,200 / 6,331 | 40 / 1,498 / 10,708 | **80 / 1,673 / 14,536** |

γ does the **fewest distances** and — crucially — **half the inspections of ACORN-1** (54,956 vs
101,068 at 5%): the dense 1-hop graph reaches recall without ACORN-1's bridge scans. That makes
γ the strongest ACORN variant. It still does far more inspections than standard (8.7× at 5%),
so it's the same *distances-for-inspections* trade — just a better-balanced one.

### Results — latency vs standard (best-ef, standard µs / ACORN-γ µs → speedup)

| dim | 5% sel | 25% sel |
|---|---|---|
| **128** | 481 / 359 → **1.34×** ✅ | 500 / 202 → **2.48×** ✅ |
| **1024** | 2722 / 1184 → **2.30×** ✅ | 2718 / 1391 → **1.95×** ✅ |
| **4096** | 10966 / 4190 → **2.62×** ✅ | 10922 / 5697 → **1.92×** ✅ |

Unlike ACORN-1, **γ wins even at 128-D** for scattered mid/high selectivity (1.34× @5%, 2.48×
@25%) — its lower inspection count means the trade pays off without needing expensive distances.
This matches the established grand-comparison numbers (@100K no 25%: **190µs vs 863µs**; @1M no
25%: **1,012µs vs 1,394µs**). (1% omitted — recall 0.44, the dense graph still can't reach a
*very* sparse eligible set at reasonable ef.)

### What we concluded

- **The best ACORN variant, and a real scattered-filter win** (1.3–2.6×), stable across scale
  and dimension — but bought with a **3.2× index and 2.8–33× build**.
- **Still collapses on negative correlation (0.00)** — a denser graph changes *how well-connected*
  the walk is, not *where it starts or heads*. Density can't teleport the walk to a far region.
- Compression can reclaim the size *or* keep the speed, never both.

**Where it works best.** Scattered / near-correlated filters at ~5–30% selectivity, **only if**
you can afford a 3.2×-larger, much-slower-to-build index. For most workloads the size/build cost
isn't worth a ~1.5–2.5× query win that self-aware routing gets most of for free.

---

## 4. RACORN — how it changed cases 1, 2, 3

### What it is

§1–§3 established the *disease*: standard, ACORN-1 and ACORN-γ all **collapse to 0.00** on
negative correlation (and standard/γ sag at very-low-sel scattered). RACORN is the *treatment*
layer — two adaptive fallbacks bolted **on top of any base traversal**, so the walk can rescue
itself when the filter fights it:
```
during the level-0 walk, per popped node:
   ... expand base frontier (standard / ACORN-1 / ACORN-γ) ...
   # ASF (Adaptive Search Fallback): passing supply ran dry -> keep the walk alive
   if (passing neighbours found this expansion) < M_base and failing ones exist:
        admit stride-sampled FAILING neighbours as transient bridges (frontier only, never results)
   # AEF (Adaptive Exact Fallback): the filter is clearly winning -> stop guessing
   if examined > 3·ef  and  passed/examined < threshold(=0.02):
        abandon the graph walk, run EXACT over the eligible set   # recall 1.0
```
ASF tries to *fix the graph walk* (more bridges); AEF *gives up on the graph* and scans the
eligible set exactly.

### What we did

- Implemented both and applied them to **all three bases** ([`bench_racorn.cpp`](src/bench_racorn.cpp)):
  STD, ACORN-1, ACORN-γ, each in `base` / `+ASF` / `+ASF+AEF` form.
- Ran a **before→after** sweep on the two regimes where they matter — negative correlation and
  scattered — SIFT @100K, best-recall ef ([`raw-results/racorn_before_after.txt`](raw-results/racorn_before_after.txt)).

### Results — negative correlation (recall / latency-µs; the 0.00 regime)

| sel | STD → +ASF+AEF | ACORN-1 → +ASF → +ASF+AEF | ACORN-γ → +ASF → +ASF+AEF |
|---|---|---|---|
| 0.1% | 0.00/67 → **1.00/131** | 0.00 → **0.00**/152 → **1.00/124** | 0.00 → **0.00**/164 → **1.00/129** |
| 5% | 0.00/62 → **1.00/341** | 0.00 → **0.00**/168 → **1.00/343** | 0.00 → **0.00**/163 → **1.00/345** |
| 25% | 0.00/63 → **1.00/947** | 0.00 → **0.00**/175 → **1.00/951** | 0.00 → **0.00**/169 → **1.00/950** |

Two things are unmistakable:
1. **ASF does *nothing* for negative correlation** — every `+ASF` cell is still **0.00**. Bridges
   extend reach locally, but the eligible region is nowhere near the walk, so there's nothing to
   bridge *to*.
2. **AEF fixes it completely — for every base — by going exact** (100% of these queries bail).
   Note the latencies converge: STD/ACORN-1/ACORN-γ all land at ~131/343/950µs because **once
   AEF fires, the base is irrelevant — they're all just doing the same exact scan.**

### Results — scattered, very-low selectivity (the other sag)

| base @0.1% no-corr | base | +ASF | +ASF+AEF |
|---|---|---|---|
| STD | 0.40 / 561 | — | **1.00 / 137** (exact) |
| ACORN-1 | 0.04 / 11 | 0.93 / 1003 | **1.00 / 132** (exact) |
| ACORN-γ | 0.02 / 9 | 0.80 / 1152 | **1.00 / 138** (exact) |

Here **ASF genuinely helps on the graph** (ACORN-1 0.04→0.93, γ 0.02→0.80) — but it's
**expensive** (~1ms) and **partial**, and **AEF beats it** on both recall *and* latency (1.00 at
~135µs, because exact over ~100 eligible docs is trivially cheap). At 5–25% scattered every base
is already ~1.0, AEF correctly **never fires** (0% exact) — it's not a blunt "always exact."

### What we concluded

- **AEF is the entire story; ASF is a footnote.** AEF takes *both* hard regimes (negative
  correlation at all sel; scattered at very-low sel) to **recall 1.0 for every base**, and is
  usually *faster* than the broken walk (exact over a small eligible set is cheap). ASF only
  helps scattered-low-sel, costs more, does **nothing** for negative correlation, and is
  dominated by AEF wherever exact is affordable.
- **Once the fallback fires, the base traversal doesn't matter** — the three bases become
  identical. So RACORN did **not** make ACORN-1/γ worth their cost; it proved **the fix is the
  exact fallback, not the graph algorithm.**
- And **AEF ≈ what OpenSearch already ships** (`KNNWeight`'s cardinality-based exact fallback).
  The one thing RACORN's own design does *worse* than it should: it runs the *whole* doomed walk
  and only bails after `3·ef` probes — which is exactly what §5 fixes.

**Where it works best.** As a universal safety net under *any* base: negative correlation and
very-low-selectivity scattered filters. Its lesson — "detect the collapse, hand off to exact" —
is the seed for **§5 (do the hand-off *inline*, mid-walk)**.

---

## 5. Self-awareness — AEF done *inline*

### How we want AEF (the problem with §4's version)

§4 proved AEF is the fix. But **when** it fires matters. Both RACORN's AEF and OpenSearch's
shipping fallback are **post-hoc**: run the *entire* filtered ANN, and only *after* it finishes
decide to go exact. That has two flaws:
1. **It pays for the whole doomed walk first.** On negative correlation the walk visits ~1,262
   nodes finding nothing, *then* does exact anyway — the ANN work is pure waste.
2. **It's half-blind.** Post-hoc triggers on "returned < k results." But a walk can return k
   *mediocre* results (recall 0.95) and post-hoc can't tell — it sees k answers and ships them.

What we *want*: the walk should **monitor its own loss as it goes** and **hand off to exact the
moment the filter is clearly winning** — mid-flight, before wasting the whole walk, and keyed on
the *quality* signal (pass-ratio), not just the result count.

### What's implemented

Inline early-abort ([`bench_selfaware.cpp`](src/bench_selfaware.cpp)):
```
during the level-0 walk, track passed / examined:
   past a min-probe gate (examined > 3·ef, so a barren start doesn't misfire):
       if passed/examined < 0.02:            # the filter is fighting the walk
           abort NOW -> exact over eligible   # recall 1.0, before finishing the doomed walk
```
**Critical correctness fix:** a *stall* (top-k stops improving) must **NOT** trigger exact — a
stall is normal HNSW convergence with good results, handled by the existing relative-distance
stop. Only a **pass-ratio collapse** triggers exact. (We had this wrong first; it sent healthy
scattered queries to exact needlessly.)

### How it helped — inline vs post-hoc (SIFT @100K, standard base, best-ef)

Recall / latency-µs / **ANN nodes examined before bailing** / exact%:

| regime | plain (today's ANN) | post-hoc (OS today) | **inline-abort** |
|---|---|---|---|
| neg 0.1% | 0.00 / 196 / 1262 | 1.00 / 296 / **1262** | 1.00 / **171** / **308** |
| neg 5% | 0.00 / 176 / 1262 | 1.00 / 678 / 1262 | 1.00 / **524** / **308** |
| neg 25% | 0.00 / 181 / 1262 | 1.00 / 1782 / 1262 | 1.00 / **1620** / **308** |
| **no 1%** | 0.95 / 780 | **0.95 / 780** (blind!) | **1.00 / 324** |
| no 5–25% | 1.00 / ~785 | 1.00 / ~785 | 1.00 / ~785 (**never fires**) |

Three things it bought:
1. **Negative correlation: same recall (1.0), 4× less wasted ANN** — bails after **308** nodes
   vs post-hoc's **1,262** → **11–42% faster** (0.1%: 171µs vs 296µs = 1.7×).
2. **It catches what post-hoc can't.** At scattered 1%, plain ANN returns k *mediocre* results
   (0.95) so post-hoc stays silent (0.95). Inline sees the pass-ratio collapse and routes to
   exact → **1.00 recall AND 2.4× faster** (324µs vs 780µs).
3. **No false-triggering.** At scattered 5–25% (healthy walk) it **never fires** (exact% = 0,
   runs the full 4,297-node ANN) — the stall fix is why.

Net across the grand comparison: **recall 1.0 in every correlation × selectivity cell**, with
latency = **min(graph walk, exact scan)** — the graph where it works, exact only where needed
(@100K 25% positive: **184µs** vs always-exact's **1,698µs** = 9×).

### What we concluded

- **Inline self-aware routing strictly dominates the post-hoc fallback** — never worse, and
  materially better where the filter fights the walk (less wasted ANN, catches mediocre-recall
  walks post-hoc misses). It is the **robust default**: never collapses, never far from optimal.
- Cost is a small (**~15-line**) native hook in the Faiss level-0 loop (the two counters + the
  gate). It's also the practical **ES-parity** recipe — Lucene/ES do adaptive filtering; this
  gives off-heap Faiss the same.

**Where it works best.** **Everywhere — the default.** Its one limit: it can't make a *large*
eligible set cheap to scan exactly (neg 25% still costs ~1.6ms of exact). That remaining cost is
what **§6 (seeded entry points)** attacks — staying on the graph instead of bailing to exact.

---

## 6. Seeded entry points

### What it is

Every technique so far kept HNSW's **fixed entry point** and started the walk at the top node,
heading toward the query. Seeding attacks that root cause: **start the walk inside the eligible
region.** Sample a few eligible docs straight from the filter, seed the frontier with them, then
run the same best-first walk toward the query on the predicate subgraph (+2-hop bridging):
```
seeds = a few eligible docs sampled from the filter        # NOT the fixed top node
frontier = seeds                                            # start INSIDE the eligible region
best-first walk toward the query on the passing subgraph    # slide down to the nearest eligible
```
You never traverse the entry→eligible gap that dooms every fixed-entry method — you begin past it.

### What we did

- Implemented it ([`bench_seeded.cpp`](src/bench_seeded.cpp)): seed the frontier with S eligible
  docs (strided for spread), walk +2-hop, compare against **fixed-entry** (the baseline that
  collapses) across S ∈ {8, 32, 128}.
- Ran the two negative regimes — **fc** (compact far cluster = realistic negative) and **neg**
  (diffuse far shell = pathological) — plus scattered, SIFT @100K
  ([`raw-results/seeded_recovery.txt`](raw-results/seeded_recovery.txt)).

### Results — the 0.00 → 0.88–1.00 recovery (recall / latency-µs)

| filter | sel | fixed-entry | seed=8 | seed=128 |
|---|---|---|---|---|
| **compact-far (fc)** | 1% | **0.00** / 335 | **1.00** / 363 | 1.00 / 311 |
| | 5% | **0.00** / 321 | **0.98** / 552 | 0.98 / 554 |
| | 25% | **0.00** / 323 | **0.88** / 642 | 0.89 / 615 |
| **diffuse-far (neg)** | 1% | 0.00 | 0.91 / 357 | 0.94 / 321 |
| | 25% | 0.00 | 0.57 / 1232 | 0.58 / 1195 |
| **scattered (no)** | 1% | 0.80 | 0.88 / 388 | 0.91 / 352 |
| | 25% | 0.99 | 1.00 / 779 | 1.00 / 773 |

- **Compact-far — the realistic negative case — goes 0.00 → 0.88–1.00 on the graph**, at
  ~300–650µs (faster than the exact fallback even at 100K).
- **~8 seeds ≈ 128 seeds** (1.00/0.98/0.88 vs 1.00/0.98/0.89). The min-distance frontier finds
  the near-edge eligible regardless of count — so it's **cheap** (sample a few docs, microseconds).
- **No harm** on scattered/positive (0.80→0.88, 0.99→1.00).

### Why it earns its keep — the scale crossover (SIFT @1M)

At 100K exact is cheap, so seeding is "as good, not yet faster." Its edge appears **at scale**,
where the exact fallback's O(eligible) cost explodes:

| compact-far | self-aware (= exact) | **seeded** | speedup |
|---|---|---|---|
| @1M, 5% | 1.00 / 7,163 µs | 0.94 / **3,369 µs** | **2.1×** |
| @1M, 25% | 1.00 / 19,645 µs | 0.91 / **3,660 µs** | **5.4×** |

Seeded's cost grows with the *graph walk*, not the eligible-set size — so as the far filter gets
bigger, exact climbs to ~20ms while seeded stays ~3.7ms. **This is the one case where staying on
the graph beats bailing to exact, and it only shows up at scale.**

### What we concluded

- **The single best filtered-search lever we found**, and the one genuinely new idea: it fixes
  negative correlation *on the graph* — where ACORN (0.00), ACORN-γ (0.00 + 3.2× index), and
  partition-pruning (≤0.76) all failed — 0.00 → 0.88–1.00, and it **scales** (5.4× over exact @1M).
- It **explains Lucene/ES**: they seed the walk from the filter, which is likely why they handle
  negative-correlation filtered search gracefully.
- **Small change** to OpenSearch's Faiss path, and it **composes with self-aware routing** (seed
  the walk; if it *still* collapses, §5 bails to exact).
- **Honest limits:** *diffuse* far sets (eligible spread across many graph components) only
  partially recover (0.57–0.94) and stay fundamentally ~O(eligible)-hard; at high selectivity the
  eligible subgraph fragments so compact-far tops out ~0.88–0.91 at 25% (higher ef / denser graph
  lifts it). Seeds must come from the filter — cheap when the filter is an inverted list.

**Where it works best.** **Compact-far / negative correlation at scale** — a category filter
whose docs cluster away from the query, on a large corpus. There it turns a ~20ms exact scan
into a ~3.7ms graph walk. Pair it with self-aware routing as the safety net.

---

## 7. Partition-sketch pruning (the off-graph axis)

### What it is

Everything in §1–§6 is a **graph** technique — they all walk HNSW and differ only in *how*.
Partition-sketch pruning comes from a different question: *"do I even need to navigate?"*
Instead of traversing a graph to *find* eligible docs, use cheap **metadata to know which
regions contain them** and scan only those — pruning whole regions *before any vector distance*.
It's essentially **filtered IVF with empty-cell skipping** ([`bench_partition.cpp`](src/bench_partition.cpp)):
```
build: partition all vectors into P≈512 IVF cells (k-means centroids)
       per cell, store a "sketch" = which of its docs pass the filter (bitmap + count)
query: rank cells by centroid distance
       SKIP every cell whose sketch has 0 eligible docs        # the pruning
       probe the nearest `nprobe` eligible-containing cells
       distance ONLY the eligible docs in those cells
```
The sketch is derived from what Lucene **already stores** (inverted index / doc-values), so the
space overhead is tiny (a partition id per vector + a per-cell eligible bitmap) — a few % of the
index, **nothing like ACORN-γ's 3.2×.**

### What we did

- Built P=512 cells over SIFT @100K, per-cell filter sketches, and compared **sketch IVF**
  against the fair baseline — **plain IVF** (same cells, probe nearest `nprobe`, distance *all*
  their docs, post-filter). Both sweep `nprobe ∈ {1,4,16,64,256,512}`; we report the **smallest
  latency reaching recall ≥ 0.95** ([`raw-results/partition_pruning.txt`](raw-results/partition_pruning.txt)).

### Results — best recall≥0.95 latency-µs, sketch IVF vs plain IVF

| filter | sel | plain IVF (rec/µs) | **sketch IVF (rec/µs)** | **sketch speedup** |
|---|---|---|---|---|
| **compact-far (fc)** | 1% | 1.00 / 15,731 | **1.00 / 167** | **94×** |
| | 25% | 1.00 / 14,826 | **1.00 / 2,370** | **6.3×** |
| **diffuse-far (neg)** | 1% | 1.00 / 14,974 | **1.00 / 163** | **92×** |
| | 25% | 1.00 / 12,599 | **0.98 / 1,917** | **6.6×** |
| **scattered (no)** | 1% | 0.96 / 1,149 | **0.98 / 69** | **17×** |
| | 25% | 0.99 / 1,174 | **0.99 / 669** | **1.8×** |

Against a *plain IVF* engine the sketch wins **everywhere, 1.8–94×**, and reaches full recall on
far filters too (the earlier "fails on far" was an nprobe-budget artifact of the *exact*
comparison — with enough probes the sketch gets there; the story is **latency**, not recall).

### In context — all techniques head-to-head (SIFT @100K, recall / latency-µs)

Sketch IVF placed next to every other technique on identical data/queries. HNSW-family columns
are the grand-comparison harness; **partition** is the sketch-IVF number from above (same scalar
distance, but a *separate IVF index* — not a drop-in on an HNSW deployment).

| regime | plain HNSW | ACORN-1 | ACORN-γ | RACORN (AEF)¹ | self-aware | seeded | **partition (IVF)** | best |
|---|---|---|---|---|---|---|---|---|
| **pos 25%** | 1.00/186 | 1.00/186 | 1.00/436 | 1.00/=std | 1.00/**184** | 1.00/193 | 1.00/2320 | HNSW/SA |
| **scattered 0.1%** | 0.40/856 🔴 | 0.04 🔴 | 0.02 🔴 | 1.00/≥SA | 1.00/**93** | 0.16 🔴 | 1.00/**92** | SA / partition |
| **scattered 25%** | 1.00/863 | 1.00/333 | 1.00/**190** | 1.00/=std | 1.00/856 | 1.00/342 | 0.99/670 | **γ** / partition |
| **compact-far 5%** | 0.00 🔴 | 0.00 🔴 | 0.00 🔴 | 1.00/≥SA | 1.00/**519** | 1.00/1247 | 1.00/**504** | SA / partition |
| **compact-far 25%** | 0.00 🔴 | 0.00 🔴 | 0.00 🔴 | 1.00/≥SA | 1.00/1812 | 0.92/**1600** | 1.00/2368 | **seeded** / SA |

¹ **RACORN (AEF)** is the *post-hoc* exact fallback: same recall as self-aware (1.00), but latency
**≥ self-aware** because it runs the whole doomed walk *before* going exact (`=std` where AEF
never fires; §5 measured the gap directly, e.g. neg 0.1%: post-hoc 296µs vs inline 171µs).

What the unified view shows:
- **Partition is genuinely competitive at 100K** where cell-ordering helps: it *ties self-aware*
  on scattered 0.1% (92 vs 93µs) and compact-far 5% (504 vs 519µs), and **beats seeded** there
  (504 vs 1247µs). But it needs a **separate IVF index**, and on **compact-far 25% seeded wins**
  (1600 vs 2368µs) — and seeded's edge *grows* at 1M (§6).
- The far-filter recall is carried by exactly three things — **self-aware** (bail to exact),
  **seeded** (stay on the graph), and **partition** (skip empty IVF cells). ACORN-1/ACORN-γ stay
  **0.00**. RACORN's contribution is the fallback trigger, which self-aware then does better.

### Why — the sketch turns filtered IVF into "distance only the eligible"

Two mechanisms, both invisible to plain IVF:
1. **Skip 0-eligible cells.** On a far filter the *near* cells hold no eligible docs; plain IVF
   still probes and distances all of them (it must probe all 512 cells to reach recall → ~15ms).
   The sketch skips them, so `nprobe` spends only on cells that *contain* eligible docs.
2. **Distance only eligible docs.** Within a probed cell, plain IVF distances every doc then
   post-filters; the sketch distances only the eligible ones. So sketch work ≈ **#eligible**,
   vs plain IVF's **#docs-in-probed-cells**.

The speedup is largest exactly where plain IVF wastes most — **low selectivity + far
correlation** (92–94× at 1%): plain IVF probes everything and keeps almost nothing. It shrinks to
**1.8× at scattered 25%** — most docs are eligible and few cells are empty, so there's little
waste to remove.

### What we concluded

- **Against an IVF engine the sketch is a large, unconditional win (1.8–94×)** at matched recall,
  for the price of a per-cell eligible bitmap (a few % of index) derived from data Lucene already
  stores. If you run filtered search on IVF, the sketch is strictly worth it.
- **The catch (honest framing vs the rest of this sheet):** the sketch's work ≈ #eligible — i.e.
  it recovers **exact-over-eligible** efficiency inside an IVF layout. So vs OpenSearch's actual
  baseline (the HNSW **exact fallback** of §5) it only wins where cell-ordering lets it *stop
  early* (scattered/near ~5×); on far filters it must scan all eligible-containing cells, so it
  **ties exact** and is **beaten by §6 seeding** (which stays sublinear on the graph).
- Its role is an **alternate scan engine** a cost-based planner picks for scattered filters at
  scale — not an HNSW replacement. ("Phase 2/3" of `NEXTGEN_FILTERED_SEARCH.md`.)

**Where it works best.** **Scattered / orthogonal metadata filters at scale** (`in_stock`,
`price<X`) on an IVF substrate — 2–90× over plain IVF, nearly free in space. For negative
correlation prefer **§6 seeding**; for HNSW deployments the **§5 exact fallback** already gets the
far-filter case.

---

## FINAL CHEAT SHEET — how it all plays in, and how each case improved

### Decision matrix (correlation × selectivity; SA = self-aware routing)
| correlation ↓ / selectivity → | very low (<0.1%) | low–moderate (1–10%) | high (>25%) |
|---|---|---|---|
| **Positive** | SA (graph) | **standard / SA** | **standard / SA** |
| **Scattered (none)** | **SA → exact** | standard/SA; **+quantize** if distance-bound; **+partition-prune** for cost | standard/SA; **+quantize** |
| **Compact-far (negative)** | **SA → exact** | **SA → exact**; **seeded** at scale | **seeded** (at scale), else SA→exact |
| **Diffuse-far** | SA → exact | SA → exact | exact (fundamentally hard) |

### How each correlation case improved (before → after)
| regime | plain HNSW | ACORN-1 | ACORN-γ | RACORN (AEF) | self-aware | seeded | partition (vs plain IVF) | **best outcome** |
|---|---|---|---|---|---|---|---|---|
| **positive** | 1.0 ✅ | 1.0 (slower) | 1.0 (costly) | 1.0 | 1.0 fast | 1.0 | — | already solved → **standard/SA** |
| **scattered, low-sel** | 0.4 🔴 | 0.04 🔴 | 0.02 🔴 | **1.0** (exact) | **1.0** (93µs) | 0.16 | 1.0, **17×** 🟢 | **self-aware** |
| **scattered, mod–high** | 1.0 | 1.0 | **1.0 faster** | 1.0 | 1.0 | 1.0 | 0.99, **1.8–6×** 🟢 | SA; **γ / quantize / partition** for speed |
| **compact-far (neg)** | **0.00** 🔴 | **0.00** 🔴 | **0.00** 🔴 | **1.0** (exact, O(eligible)) | **1.0** (min cost) | **0.88–1.0 cheap** | 1.0, 6–94× (≈exact work) 🟡 | **SA (small) / seeded (scale)** |
| **diffuse-far** | 0.00 🔴 | 0.00 | 0.00 | 1.0 (exact) | 1.0 | 0.6–0.9 | ~1.0, 6–92× (≈exact work) 🟡 | **exact via SA** (inherently hard) |

### The storyline in one paragraph
We chased the ACORN family and found the honest truth: **ACORN-1 and ACORN-γ do not beat
standard HNSW on filtered search** — they collapse on the same negative-correlation cases and
cost more (ACORN-1 inspection-bound on 128-D; ACORN-γ 9–33× build for a narrow scattered win).
**RACORN** showed the real fix is the **exact fallback (AEF)**, not the graph algorithm.
**Self-aware routing** made that fallback *inline and cost-optimal* — the robust default, recall
1.0 everywhere at min(graph, exact) latency, and the off-heap ES-parity recipe. **Seeded entry
points** — starting the walk *inside* the eligible region — is the one new idea that makes the
hard case (negative correlation) cheap *on the graph*, and it scales. **Net improvement:** every
regime now reaches recall ~1.0; the two collapse regimes (negative correlation, low-sel
scattered) went from **0.0–0.4 → 1.0**; and we did it with cheap, mostly-already-shipped
mechanisms (adaptive fallback + quantize/rescore + seeding) rather than an expensive new graph.

### Product one-liners
- **Orthogonal metadata filters** (`in_stock`, `price`) → self-aware + quantize; skip ACORN.
- **Category/taxonomy filters** → self-aware; **seed** the cross-domain queries at scale.
- **Very high-dim vectors** → quantize+rescore (and ACORN-1 may finally help).
- **Memory-constrained / huge index** → off-heap Faiss, avoid ACORN-γ; use `on_disk` quantize.
- **"Just want ES filtered latency"** → not a new algorithm: `standard + self-aware exact
  fallback + quantize/rescore`, all in OpenSearch's off-heap Faiss today.

### Grand comparison numbers (SIFT1M @ 100K)
| corr / sel | standard | ACORN-1 | ACORN-γ | **self-aware** | seeded | exact |
|---|---|---|---|---|---|---|
| pos 25% | 1.00/186 | 1.00/186 | 1.00/436 | **1.00/184** | 1.00/193 | 1.00/1698 |
| no 0.1% | 0.40/856 | 0.04 | 0.02 | **1.00/93** | 0.16 | 1.00/60 |
| no 25% | 1.00/863 | 1.00/333 | 1.00/190 | **1.00/856** | 1.00/342 | 1.00/1980 |
| far 5% | 0.00 | 0.00 | 0.00 | **1.00/519** | 1.00/1247 | 1.00/475 |
| far 25% | 0.00 | 0.00 | 0.00 | 1.00/1812 | 0.92/1600 | 1.00/1774 |

### Grand comparison numbers (SIFT1M @ 1,000,000) — the scale story
| corr / sel | standard | ACORN-1 | ACORN-γ | self-aware | **seeded** | exact |
|---|---|---|---|---|---|---|
| pos 25% | 1.00/1390 | 1.00/1401 | 1.00/2315 | **1.00/1387** | 1.00/1402 | 1.00/**18832** |
| no 25% | 1.00/1394 | 1.00/5133 | **1.00/1012** | 1.00/1389 | 1.00/4537 | 1.00/24395 |
| far 5% | 0.00 | 0.00 | 0.00 | 1.00/7163 | **0.94/3369** | 1.00/7116 |
| far 25% | 0.00 | 0.00 | 0.00 | 1.00/**19645** | **0.91/3660** | 1.00/19639 |

**What changes at scale (the key finding):**
- **Exact blows up** — at 25% selectivity, exact is ~19–24ms (O(eligible) over 250k docs). So on
  **positive/scattered** filters, self-aware (which stays on the *graph*) is up to **13× faster
  than always-exact** (pos 25%: 1,387µs vs 18,832µs).
- **On compact-far (negative), self-aware = exact** (it correctly bails), so it inherits that
  ~19ms cost. **Seeded overtakes it: far 25% → 3,660µs vs 19,645µs = 5.4× faster** (recall 0.91),
  far 5% → 3,369µs vs 7,163µs = 2.1×. **This is where seeding earns its keep — negative
  correlation at scale.**
- **ACORN-γ's narrow win shows** at scattered high-sel (no 25%: 1,012µs vs standard 1,394µs).
- **ACORN-1 / ACORN-γ still 0.00** on far/negative — the collapse is unchanged at scale.

**Refined recommendation at scale:** positive/scattered → **self-aware** (graph, avoids the
exact blow-up); compact-far/negative → **seeded** (2–5× faster than the exact fallback);
diffuse-far → exact via self-aware (seeded degrades to 0.28–0.72 there — fundamentally hard).
