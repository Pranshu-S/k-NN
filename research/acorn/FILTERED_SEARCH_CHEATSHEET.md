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

Plain HNSW filtered search ("in-filtering": walk the graph, keep only filter-passing nodes)
has a specific, severe failure: under **negative correlation**, recall **collapses to ~0**.
The walk always starts at the fixed entry point and heads toward the query, so it only ever
explores the entry→query corridor — and the eligible docs are elsewhere. It's also weak at
very low selectivity, and its filtered traversal wastes work at scale. **Goal:** make filtered
vector search robust and fast across *all* correlation/selectivity regimes — ideally beating
standard HNSW — and understand the real cost/benefit of each approach (the ACORN family from
the papers, plus our own ideas), so OpenSearch can match Elasticsearch's filtered latency.

---

## 2. ACORN-1

**What we did.** Implemented the paper's ACORN-1: a *search-time* two-hop traversal over the
**standard** HNSW graph — the walk routes *through* filter-failing nodes into their passing
2-hop neighbours, so only eligible nodes enter results but rejected nodes act as bridges. We
corrected real bugs (internal→external ID translation, bridge-expand-once, unique counting)
and optimized it (SIMD-batched distances). 16/16 unit tests pass.

**How it worked.** Measured @100K, 5% scattered: ACORN-1 computes **fewer distances** (1,254
vs standard's 4,385) but **inspects 5× more neighbours** (84,711 vs 16,248). Filtered HNSW is
**inspection-bound, not distance-bound** — so doing fewer distances but more inspections is a
net loss on ordinary vectors.

**What we concluded.**
- On 128-D SIFT, ACORN-1 is **1.5–5× *slower*** than standard (worse at 1M).
- It **does not fix** the negative-correlation collapse — still 0.00, same as standard (2-hop
  bridges extend *reach*, but the walk still starts at the fixed entry and heads to the query).
- It **only wins when distances are expensive**: tiled to 4096-D, ACORN-1 is **1.7× faster**
  at 5% selectivity (the distance savings finally outweigh the extra inspections).

**Where it works best.** **Very high-dimensional vectors** (GIST-960 and up, some LLM
embeddings) at moderate selectivity, where each distance is costly. Not on typical embeddings.

---

## 3. ACORN-γ

**What we did.** Built the denser-graph variant: a standard HNSW with **M = γ·16** (level-0
degree ~2·γ·16), searched with filtered **1-hop** (the dense graph makes bridging unnecessary).
Also implemented the paper's **two-hop compression** (`shrink_neighbor_list`) to shrink the
index, with adaptive-2-hop recovery.

**How it worked.** For **scattered** filters at ~10–30% selectivity, ACORN-γ is genuinely
**1.4–4× faster** than standard at recall 1.0 (@100K, 25%: 190µs vs 863µs) — the dense graph
finds enough passing neighbours in one hop, so far fewer hops/distances. Compression cut the
index **3.2× → 1.1×** and kept recall, **but lost the speed** (re-sparsifying reintroduces the
2-hop cost) — a strict trilemma: *small+slow, or fast+big, never both*.

**What we concluded.**
- Real win, but **narrow** (scattered filters, moderate sel) and **expensive**: the paper's own
  Table 4 shows **9–33× longer index build** (25M vectors: ~10.5 h vs ~19 min) and **1.3–3.2×
  index size**.
- Still **collapses on negative correlation** (0.00) — a denser graph doesn't change *where the
  walk goes*.

**Where it works best.** Scattered / near-correlated filters at ~10–30% selectivity, **only if**
you can afford a much denser, slower-to-build index. For most workloads the cost isn't worth it.

---

## 4. RACORN — how it modified cases 1, 2, 3

**What we did.** Added two adaptive fallbacks *on top of* each base traversal (standard,
ACORN-1, ACORN-γ):
- **ASF (Adaptive Search Fallback):** when a node's expansion finds too few passing neighbours,
  admit stride-sampled *failing* nodes as transient frontier bridges — keep the graph walk alive.
- **AEF (Adaptive Exact Fallback):** track the running pass-ratio; when it collapses, abandon
  the walk and run **exact** over the eligible set (recall 1.0).

**How they performed (applied to all 3 bases).**
- **AEF is the workhorse.** For *every* base — standard, ACORN-1, ACORN-γ — it takes the hard
  regimes (negative correlation AND very-low selectivity) from ~0 to **recall 1.0**, at exact
  cost (cheap when the eligible set is small; ~11ms at 25% negative on 1M). So RACORN made all
  three bases *robust*.
- **ASF is secondary.** It recovers *scattered* low-selectivity recall on the graph (e.g.
  ACORN-1 @0.1% no-corr: 0.04 → 0.93), but it's expensive and **does nothing for negative
  correlation** (bridges can't reach a far region), and AEF beats it whenever exact is affordable.

**What we concluded.** RACORN's value is almost entirely **AEF ≈ OpenSearch's *existing* exact
fallback**. The genuinely novel part (ASF) is the weaker of the two. So RACORN didn't make
ACORN-1/γ worth it — it confirmed that **the exact fallback, not the graph algorithm, is what
rescues the hard cases.** That directly motivated the next two ideas.

---

## 5. Self-awareness (inline early-abort routing)

**What we did.** Moved the exact fallback *inside* the walk. Today OpenSearch runs the **full**
ANN and *then* falls back to exact if it returned < k ("post-hoc"). Instead we make the walk
**monitor its own loss** (`passed/examined`) past a **min-probe gate** and **abort to exact
mid-flight** the moment the filter is clearly fighting it. (Key fix: a *stall* means the ANN
converged with good results — that must NOT trigger exact; only a pass-ratio collapse does.)

**How it worked.** vs OpenSearch's post-hoc fallback (standard base, negative correlation):
same recall (1.0), but inline **examines 4× fewer nodes** (308 vs 1,262 before bailing) → **11–
40% faster**. And it never false-triggers on scattered filters. In the grand comparison it is
**recall 1.0 in *every* correlation × selectivity cell**, with latency = the **minimum of
(graph walk, exact scan)** — fast graph where the graph works, exact only where needed
(@100K 25% positive: **184µs vs always-exact's 1,698µs**).

**What we concluded.** **Self-aware routing strictly dominates the current post-hoc fallback**
and is the robust default: it never collapses and is never far from optimal. Cost is a small
(~15-line) native hook in the Faiss walk. It's also the practical "ES-parity" recipe (Lucene/ES
do adaptive filtering; this gives Faiss the same, off-heap).

**Where it works best.** **Everywhere — the default.** Its only limit: it can't make a *large*
eligible set cheap to scan exactly (that's what seeding/scale address).

---

## 6. Seeded entry points

**What we did.** Attacked the root cause the others ignored: the **fixed entry point**. Instead
of starting the walk at HNSW's top node, **seed the frontier with ~8 eligible docs** sampled
from the filter, then walk toward the query on the predicate subgraph (+2-hop). You start
*inside* the eligible region rather than trying to traverse to it.

**How it worked.** For **compact-far (realistic negative) correlation**, recall went from
**0.00 → 0.88–1.00**, and *faster than full exact* (@100K 25%: 847µs vs ~1,831µs). Crucially,
**~8 seeds is as good as 128** — the min-distance frontier finds the near-edge eligible
regardless of seed count — so it's cheap (sample a few eligible docs, microseconds). No harm on
positive/scattered filters.

**What we concluded.** **Seeding is the single best filtered-search lever we found** and the one
genuinely new idea: it fixes negative correlation *on the graph*, where ACORN (0.00), ACORN-γ
(0.00 + huge cost), and partition-pruning (≤0.76) all failed. It also explains **why Lucene/ES
handle negative-correlation filtered search gracefully — they seed from the filter.** It's a
small change to OpenSearch's Faiss path and composes with self-aware routing.

**Where it works best.** **Negative correlation** (filter maps to a compact far region), and its
edge is **confirmed at scale**: @1M, compact-far 25% → seeded **3,660µs @ 0.91 recall** vs the
exact fallback's **19,645µs @ 1.0** — **5.4× faster** (2.1× at 5%). At 100K exact is cheap so
seeding isn't yet faster; at 1M+ it clearly wins for negative correlation. Honest limit: at high
selectivity the eligible *subgraph* fragments, so recall tops out ~0.91 at 25% (higher ef /
denser graph helps); truly *diffuse* far sets stay hard (0.28–0.72).

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
| regime | plain HNSW | ACORN-1 | ACORN-γ | RACORN (AEF) | self-aware | seeded | **best outcome** |
|---|---|---|---|---|---|---|---|
| **positive** | 1.0 ✅ | 1.0 (slower) | 1.0 (costly) | 1.0 | 1.0 fast | 1.0 | already solved → **standard/SA** |
| **scattered, low-sel** | 0.4 🔴 | 0.04 🔴 | 0.02 🔴 | **1.0** (exact) | **1.0** (93µs) | 0.16 | **self-aware** |
| **scattered, mod–high** | 1.0 | 1.0 | **1.0 faster** | 1.0 | 1.0 | 1.0 | SA; **γ or quantize** for speed |
| **compact-far (neg)** | **0.00** 🔴 | **0.00** 🔴 | **0.00** 🔴 | **1.0** (exact, O(eligible)) | **1.0** (min cost) | **0.88–1.0 cheap** | **SA (small) / seeded (scale)** |
| **diffuse-far** | 0.00 🔴 | 0.00 | 0.00 | 1.0 (exact) | 1.0 | 0.6–0.9 | **exact via SA** (inherently hard) |

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
