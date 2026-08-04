# Filtered Vector Search — Learnings & Next-Gen Direction

A durable summary of what this investigation established, the self-aware-ANN result, and
the forward roadmap. Companion to `ACORN_QUERY_API.md` (the ACORN/RACORN report).

---

## Part A — What we learned (durable, evidence-backed)

1. **Correlation, not selectivity, governs filtered HNSW.** Recall collapses to ~0 under
   *negative* (cross-domain) correlation at *every* selectivity, because the query-directed
   walk never reaches the far-away eligible region. Selectivity % alone predicts nothing.
2. **Filtered HNSW is inspection-bound, not distance-bound.** Measured (SIFT1M, 5% sel):
   standard examines ~16k neighbours / 4,385 distances; ACORN-1 examines ~85k neighbours /
   1,254 distances. ACORN does *fewer distances* but *5× more neighbour inspections* — and
   inspection (selector test + visited check) is what costs the wall-clock.
3. **ACORN-1 has no wall-clock win on normal (128-D) vectors** — it only wins when distances
   are *expensive* (≈4096-D → 1.7× faster at 5%). On SIFT it is 1.5–5× *slower* than standard.
4. **ACORN-γ is a niche win at a brutal cost.** 1.4–4× faster for scattered filters at
   ~10–30% selectivity, but the paper's own Table 4 shows **9–33× longer index build**
   (25M vectors: ~10.5 h vs ~19 min) and a denser graph (3.2× index uncompressed; ~1.3×
   with the paper's two-hop compression).
5. **Compression cuts size but loses the speed** — a strict trilemma: standard (small+fast),
   γ-dense (fast, 3.2× big), γ-compressed (1.1× small, but slower than standard).
6. **The workhorse is the exact fallback, not the graph algorithm.** Across all of RACORN,
   AEF (Adaptive Exact Fallback) is what fixes *both* negative correlation and low
   selectivity — for every base — taking recall to 1.0. ASF (graph bridging) is secondary
   and does nothing for negative correlation. And AEF ≈ what OpenSearch already ships.
7. **Quantize+rescore helps distance-bound search, not inspection-bound.** It speeds up the
   STANDARD base (~2× at 25% sel) by making the wide first-pass beam cheap; it *hurts* the
   ACORN bases (their cost is inspection, which quantization can't cheapen).
8. **The Elasticsearch filtered-search win = adaptive filtering + quantize/rescore — and
   OpenSearch has both, off-heap.** `KNNWeight` already has a cardinality-based exact
   fallback; `QFrameBitEncoder` provides BBQ-class binary quantization (random rotation +
   ADC); `RescoreContext` provides oversample+rescore; `NativeEngineKnnVectorQuery` composes
   filter × quantized-first-pass × full-precision-rescore. The gap is tuning/defaults, not
   architecture. **You do not need on-heap Lucene to get ES-like filtered latency.**

**Net recommendation:** for a general filtered-vector-search system, **standard HNSW +
adaptive (self-aware) fallback + quantize/rescore** beats ACORN/ACORN-γ. Keep ACORN-γ only
for the high-dimensional / specific-selectivity niche.

---

## Part B — Self-aware ANN: inline early-abort (proven)

**Idea (refined from RACORN's AEF):** don't probe-then-decide, and don't run the whole
doomed ANN and fall back *afterwards*. Instead the ANN **monitors its own loss during the
walk** and **self-routes to exact mid-flight** the moment the filter is clearly fighting it.

**Mechanism** (`bench_selfaware.cpp`):
- Track running `passed/examined` during the level-0 walk.
- A **min-probe gate** (`examined > 3·ef`) prevents aborting a walk that just started in a
  barren patch.
- Past the gate, if `passed/examined < passThr` (default 0.02) → **abort → exact** over the
  eligible set. (A *stall* — top-k stops improving — is normal convergence, handled by the
  existing relative-distance stop; it must NOT trigger exact. That was a bug we caught.)

**Result vs OpenSearch's current post-hoc fallback (SIFT1M @100K, standard base):**

_Negative correlation_ — inline wins (same recall, 4× less wasted ANN):
| sel | post-hoc (OS today): rec / lat / ANN-examined | inline-abort: rec / lat / ANN-examined |
|---|---|---|
| 0.1% | 1.00 / 276µs / **1262** | 1.00 / **167µs** / **308** |
| 1%   | 1.00 / 345µs / 1262 | 1.00 / **260µs** / 308 |
| 5%   | 1.00 / 756µs / 1262 | 1.00 / **521µs** / 308 |
| 10%  | 1.00 / 1364µs / 1262 | 1.00 / **933µs** / 308 |
| 25%  | 1.00 / 1706µs / 1262 | 1.00 / 1520µs / 308 |

_No correlation_ — fires only when it should:
| sel | post-hoc | inline-abort |
|---|---|---|
| 1%  | 0.95 / 708µs (no fallback) | **1.00 / 326µs** (abort→exact: faster AND higher recall) |
| 5–25% | 1.00 / ~800µs (full ANN) | 1.00 / ~800µs (**no abort** — correctly runs full ANN) |

**Verdict:** inline early-abort **strictly dominates** the post-hoc fallback — never worse,
materially better where the filter fights the walk (4× less wasted ANN, 11–40% faster), and
no false-triggering. Cost: a small **native hook** (the Faiss walk returns early past the
min-probe gate) rather than a Java post-hoc check. The exact-fallback cost at *high*
cardinality is unchanged — that is a separate lever (quantized-exact / partition pruning).

**Design lesson (generalizes):** an access method should **monitor its own progress and hand
off mid-flight**. Inline ANN→exact is the first, proven hand-off.

---

## Part C — Roadmap: a self-aware, cost-based vector query planner

The redefinition is not a better graph — it is **killing the monolithic index and routing
per query** over a partitioned, scan-native, filter-native substrate (the DB-optimizer move,
applied to vectors).

- **Phase 1 — Self-aware routing (this doc, Part B).** ANN monitors its own loss and
  self-routes ANN→exact inline. Highest ROI, mostly already in OpenSearch; the inline hook is
  the one small addition. Kills the negative-correlation trap without paying for the doomed walk.
- **Phase 2 — Filter sketches / metadata pruning.** Per-partition attribute metadata (filter
  bitmaps, cardinality, min/max) so the planner prunes *which regions to even touch* before
  any vector distance — turning "traverse to find eligible docs" into "look up where they are."
- **Phase 3 — Scan substrate.** IVF-style partitions + existing binary quant (`QFrameBitEncoder`)
  + rescore as a first-class engine beside HNSW; the planner arbitrates {exact, filtered-ANN,
  quantized-partition-scan} per query. When 1-bit SIMD scans are microseconds, navigation
  stops paying for itself.
- **Phase 4 — Learn the router.** Fit the cost model online from query logs; eventually a
  learned partitioner that jointly organizes vectors *and* filter predicates — the index
  literally organizes around how users filter.

**One line:** HNSW answers "how do I navigate to the neighbourhood?" The next generation
answers "**do I even need to navigate?**" — and the data says: with metadata pruning, 1-bit
SIMD scans, and a self-aware router, mostly you don't.

---

## Part D — Phase 2 prototype: partition-sketch pruning (results)

`bench_partition.cpp`: partition vectors into P IVF-style cells; per-cell "sketch" = which
docs are eligible (filter ∩ cell); rank cells by centroid distance, **skip cells with 0
eligible**, probe the nearest `nprobe` eligible-containing cells, distance only their eligible
docs. Compared to full exact (scan ALL eligible). SIFT1M @100K, P=512.

**Size cost — small (the main question):** partition id ~4 B/vector; per-cell sketch is
derived from filter data Lucene already stores (inverted index / doc-values) — a per-cell
doc-bitmap + counts, a few % of index. **Nothing like ACORN-γ (9–33× build, 1.3–3.2× size).**

**Results (recall / latency / #distances):**
| filter type | full exact | pruned np=64 |
|---|---|---|
| scattered (`no`) 25% | 1.00 / 2033µs / 25000 | **0.99 / 417µs / 3432** (5× faster) 🟢 |
| scattered (`no`) 5% | 1.00 / 580µs / 5000 | **0.98 / 130µs / 698** (4.5×) 🟢 |
| compact far cluster 25% | 1.00 / 1831µs | 0.76 / 283µs (6.5× faster, but recall 0.76) 🟡 |
| diffuse far shell 25% | 1.00 / 1700µs | ≤0.11 (fails — eligible spread across ~1250 cells) 🔴 |

**Verdict (honest correction):** Phase 2 is **filtered IVF with sketch-based empty-cell
skipping**, and it inherits IVF's weakness — centroid-distance ranking guides well when
eligible docs are near/scattered, poorly when they're far (far cells are ~equidistant, so
ordering is noisy). So:
- ✅ scattered / near-correlated filters (the common case): a real, cheap win (~5× at ~0.98),
  negligible size overhead.
- 🟡 compact far cluster: a graceful **recall/latency knob** (0.76 @ 6.5×; raise nprobe for
  more), still far better than ANN's 0.00.
- ❌ it does **not** cheaply solve negative correlation — finding the *nearest* members of a
  *far* set is fundamentally ~O(eligible). Phase 2 softens it into a tunable tradeoff; it
  doesn't erase it. Lever to improve far-cluster recall: more/tighter cells (higher P).

---

## Part E — Seeded entry points (the one lever we never pulled)

**Observation:** every technique above (ACORN, γ, RACORN, sketches) keeps HNSW's **fixed entry
point** and tries to *reach out* from the entry→query corridor. None change *where the walk
starts*. Negative correlation fails because the walk only ever sees that corridor + the query
neighbourhood, and the eligible docs are outside it.

**Idea:** for a filtered query, **seed the graph walk from a few eligible docs** (sampled from
the filter, or one representative per eligible-containing cell from Phase 2's sketch) instead
of the fixed top node — start *inside* the eligible region, then walk toward the query on the
predicate subgraph to reach the nearest eligible. This sidesteps the entry→eligible traversal
that ACORN fights. (Lucene's filtered HNSW already does a form of this.)

**Verified (`bench_seeded.cpp`, SIFT1M @100K, seed frontier with ~8 eligible docs + 2-hop):**

| filter type | sel | fixed-entry (ACORN/standard) | **seeded (8 seeds)** |
|---|---|---|---|
| compact far cluster | 1% | 0.00 | **1.00** / 370µs |
| compact far cluster | 5% | 0.00 | **0.98** / 615µs |
| compact far cluster | 25% | 0.00 | **0.88** / 847µs (vs exact ~1831µs) |
| diffuse far shell | 5% / 25% | 0.00 | 0.75 / 0.57 (partial) |
| no correlation | any | 0.98–0.99 | 1.00 (no harm) |

**This is the best filtered-search lever found in the whole investigation.** For realistic
negative correlation (a filter that maps to a *compact far region*), seeding takes recall from
**0.00 → 0.88–1.00**, *faster than exact*, where ACORN collapses, ACORN-γ needs 9–33× build,
and partition pruning tops out at 0.76. Key properties:
- **Cheap:** ~8 seeds is as good as 128 (the min-distance frontier finds the near-edge eligible
  regardless of count) → sample a few eligible docs from the filter, microseconds.
- **Composes** with the 2-hop bridging already built.
- **Explains Lucene/ES:** they seed the walk from the filter rather than always starting at the
  fixed top node — a likely reason they handle negative-correlation filtered search gracefully.
- **Small change** to OpenSearch's Faiss path: seed the frontier from sampled eligible docs.

**Honest limit:** at high selectivity the eligible *subgraph* on the sparse HNSW graph is
fragmented, so seeded tops out ~0.88 at 25% (some true neighbours sit in unreached components);
more seeds barely helps (connectivity, not count) — higher `ef` or a denser graph would. Diffuse
far sets stay fundamentally O(eligible)-hard. But **0.00 → 0.88–1.00 for free** is the result
nothing else delivered — and it reframes the next-gen direction around **variable entry points**,
not better graphs.
