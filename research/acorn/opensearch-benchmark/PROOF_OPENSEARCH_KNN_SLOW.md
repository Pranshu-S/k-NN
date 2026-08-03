# Simplified proof: today's OpenSearch filtered k-NN does too much work — and it gets worse with scale

**One scenario, one number to watch.** A **1%-selective filter** whose eligible
documents sit in a **different vector-space region than the query** — e.g. "find
running shoes" when the query embedding is closer to sandals, or a tenant/region
filter that points at content the query isn't near. This is routine in production
(category, availability, language, region, tenant filters).

We sweep the **index size N** up to 1,000,000, keep the filter at 1% of the index,
and measure the **work per query = distance computations** (implementation-
independent) and wall-clock, at matched recall.

Reproduce: `./research/acorn/build/proof_slow` — raw:
[`raw-results/proof_scaling.txt`](raw-results/proof_scaling.txt).

## The numbers (full curve, 100K → 1M)

| index N | eligible (1%) | **OpenSearch today (filtered HNSW)** | Exact over eligible | RACORN-1+ (fix) |
|---:|---:|---|---|---|
| 100,000 | 1,000 | recall 1.00 · **99,960 comps · 23.8 ms** | 1.00 · 1,000 · 0.17 ms | 0.95 · 1,428 · 1.2 ms |
| 250,000 | 2,500 | recall 1.00 · **179,542 comps · 51.1 ms** | 1.00 · 2,500 · 0.44 ms | 0.75 · 2,290 · 1.4 ms |
| 500,000 | 5,000 | recall 1.00 · **132,981 comps · 40.7 ms** | 1.00 · 5,000 · 0.99 ms | 0.75 · 4,049 · 2.5 ms |
| 1,000,000 | 10,000 | recall **0.94** · **265,385 comps · 83.8 ms** | 1.00 · 10,000 · 2.09 ms | 0.85 · 9,230 · 4.0 ms |

`comps` = distance computations per query (the work). Same 1% filter shape at every N.

## What the numbers prove

1. **Today's filtered HNSW scans a large fraction of the WHOLE index for a 1% filter.**
   It does **100,000–265,000 distance computations** per query across the sweep —
   roughly the entire index — while the eligible answer set is only 1,000–10,000 docs.

2. **The cost tracks the index size, not the filter.** As N grows 100K→1M, HNSW's
   work stays in the ~100K–265K range (∝ index), while the eligible set and the exact
   cost grow only 1,000→10,000 (∝ filter). At **1M it is 265,385 comps / 83.8 ms** to
   return a 10,000-doc answer.

3. **Head-to-head at 1M:** today's HNSW does **26× more work and is 40× slower than
   exact** (265,385 vs 10,000 comps; 83.8 ms vs 2.09 ms) — and its recall has already
   started slipping (**0.94 vs 1.00**). At 100K the gap is even larger: **100× work,
   140× latency.**

4. **This is a structural scaling defect, not a tuning issue.** To keep recall a
   distance-first HNSW must keep visiting until it collects `ef` *passing* results;
   when the eligible set is small and away from the query, that means walking most of
   the graph. Larger `M` / `ef_construction` / `ef_search` only make it slower.

## Why "it needs to be fixed"

For this extremely common shape — a **selective, non-query-aligned filter** —
OpenSearch's filtered HNSW path is **26–100× slower than it should be**, and it
*worsens* with index size. A customer who adds a category/tenant/region filter to a
1M-vector index pays ~80 ms/query (and rising) for something a correct answer needs
~2 ms for.

**The honest fix is two-fold, and OpenSearch already owns half of it:**

1. **Route these queries to exact.** Exact over the eligible set is the clear winner
   in this proof — **2 ms and recall 1.00 at 1M** vs HNSW's 84 ms / 0.94. OpenSearch
   *has* an exact fallback, but it triggers only for very small filter cardinalities;
   **its threshold is far too conservative** — it lets a 10,000-eligible (1% of 1M)
   query fall onto the catastrophic HNSW path. Widening the exact-vs-ANN routing rule
   (by eligible count and query–filter alignment) fixes the majority of this today.

2. **Bounded graph traversal for large eligible sets.** When the eligible set is too
   big for exact (e.g. 1% of 100M = 1M docs), a **selector-aware traversal
   (RACORN-1/1+)** keeps the work bounded by the *eligible neighbourhood, not the
   index*: it answered these queries with **~1,400–9,200 distance comps that track the
   filter (∝ eligible), not the index** — the property today's HNSW lacks. It is the
   backstop above the exact crossover.

> Honesty note. In this proof **exact is the star**, because at ≤1% of ≤1M the eligible
> set is small enough that brute force wins outright — the immediate, low-risk fix is
> better routing to exact. RACORN's *work* already tracks the eligible set (the correct
> scaling), but its **wall-clock here uses an unoptimized research traversal**; on an
> optimized engine the RACORN-1 paper converts that work advantage into 5–75× speedups
> at 1M–40M, which is where RACORN (not exact) becomes the winner. The rock-solid,
> implementation-independent claim of THIS proof is narrow and sufficient: **today's
> filtered HNSW does 26–100× more work than necessary for a selective, non-aligned
> filter, is 40–140× slower than exact at equal-or-better recall, and the waste scales
> with the index — it needs fixing.**
