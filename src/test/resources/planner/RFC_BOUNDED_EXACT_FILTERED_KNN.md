# RFC: Bounded exact search for selective filtered Lucene k-NN queries

Status: draft for maintainer review. Full research and all benchmark tables:
[CARDINALITY_AWARE_PLANNER_BENCHMARK.md](CARDINALITY_AWARE_PLANNER_BENCHMARK.md).

## 1. Problem

For a filtered top-k k-NN query, restricting HNSW to a selective filter can make the graph search slow
and/or lossy: filtered ANN latency spikes at moderate selectivity (measured 8–16 ms at 5–10 % on 100k×128)
and recall can fall to **0.72–0.87** for graph-local filters, while a brute-force exact scan of the small
filtered set is both fast and exact. Users get unpredictable latency and silent recall loss precisely
when the filter is selective — the case where exact search is cheap.

## 2. Existing Lucene behaviour (inherited unchanged by OpenSearch)

Lucene's `AbstractKnnVectorQuery` (10.5.0) already, per segment:

```
if (filteredLiveCount <= k)          -> exact search
else                                  -> approximate (HNSW) search
if (ANN early-terminated on visit limit, or returned < k)
                                      -> exact fallback over the same AcceptDocs
```

So Lucene handles the tiny-filter case and the "ANN ran out of budget" case, reusing the filter and
guaranteeing exactness. `OSKnnFloatVectorQuery` overrides only `mergeLeafResults`, so this is inherited.

## 3. Gap addressed by OpenSearch

Lucene switches to exact only at `count <= k` (e.g. 10) or after ANN exhausts its (large) visit limit.
It does **not** proactively use exact for the band `k < count <= candidateLimit`, where exact is a small,
bounded, guaranteed-recall scan but ANN may be slower and/or lossy. This proposal fills exactly that band.
It also never catches ANN that *completes* with poor recall — the exact path sidesteps that by not using
ANN at all when the filter is that selective.

## 4. Proposed execution flow

At query build time (`KNNQueryFactory.create`), for the supported scope only, wrap the existing ANN query
in `BoundedExactKnnFloatVectorQuery`. At execution (`createWeight`, data node, one reader generation):

```
bounded filter count (stops at candidateLimit + 1, retains per-leaf matched doc ids)
  count == 0                      -> MatchNoDocsQuery
  count in [1, candidateLimit]    -> exact scan over the retained ids (filter NOT re-evaluated)
  count  > candidateLimit         -> delegate to the unchanged Lucene ANN (+ its native fallback)
candidateLimit = min(HARD_MAX_EXACT_CANDIDATES=10000, max(1000, 10 * k))   // overflow-safe
```

The hard cap keeps the worst-case exact scan bounded even for large `k`: without it, `k = 10,000` would
allow a 100,000-candidate exact scan measured at ~153 ms p95 (dim 768). The cap is a no-op for every
`k <= 1000` (where `max(1000, 10k) <= 10000`), so it only affects the pathological large-`k` regime,
reducing the worst-case exact p95 to ~19 ms.

Exact reuses the plugin's existing `ExactSearcher` (same scorer the rescore path uses) via a sparse
`IntArrayDocIdSetIterator` built from the retained ids.

## 5. Why bounded exact search is safe

- **Bounded work.** Exact scans at most `candidateLimit` (= `max(1000, 10·k)`) candidate vectors.
  Calibration showed exact latency is driven by the number of vectors scanned, not by `card × dim`; the
  cardinality cap bounds the dominant cost uniformly across dimensions — measured worst case ≈ 1–2 ms at
  `candidateLimit = 1000` (128–768 dim). (A dimension-aware `ops/dim` limit was evaluated and rejected: it
  permits ~7,500-vector scans at dim 128 → ~8.7 ms, raising the worst case.)
- **Bounded counting.** The filter count stops at `candidateLimit + 1`; large filters are never fully
  counted.
- **Filter reuse.** The exact path reuses the doc ids gathered during counting — no second filter pass.
- **Memory scales with matches.** `O(matches)` via `IntArrayDocIdSetIterator` (56 bytes for 10 matches in
  a 10M-doc segment vs 1.25 MB for `FixedBitSet`).
- **Correctness.** Live-docs honoured (deletions excluded), per-segment ids never merged into a global
  space, single reader generation, results identical to the existing exact/rescore scorer.

## 6. Benchmark evidence (100k×128, k=10, ef=100; full tables linked)

| Distribution | Card | ANN p50 | Exact p50 | ANN recall |
| ------------ | ---: | ------: | --------: | ---------: |
| random (scattered) | 5,000 | 8,175 µs | 4,193 µs | 1.000 |
| correlated (graph-local) | 5,000 | 742 µs | 1,801 µs | 0.996 |
| contig_rand (adversarial) | 10,000 | 373 µs | 500 µs | **0.790** |

Within the exact band (`count ≤ candidateLimit ≈ 1,000`), exact is competitive-to-faster and always
recall = 1.0; worst-case exact latency at the band edge is ~1–2 ms across 128–768 dim.

## 7. Recall implications

Exact search guarantees recall = 1.0. In the selective band, filtered ANN recall was measured as low as
0.72–0.87 for graph-local / adversarial filters; the proposal removes that loss for the covered band.
Outside the band, recall is unchanged (Lucene ANN + fallback).

## 8. Supported scope

Lucene engine; FLOAT vectors; top-k (`k`) queries; non-nested fields; filter present; rescore disabled.

## 9. Non-goals

- Not a general ANN cost planner; does **not** claim to pick the globally fastest path.
- No new public setting, mapping, REST, codec, or HNSW change.
- No behaviour change for native engines, BYTE/BINARY, radial, nested/expand_nested, rescore, or
  no-filter queries.

## 10. Alternatives considered and rejected (with evidence)

- **Universal cardinality cost model** — cardinality alone predicts the faster path ≤ 73 %; the optimal
  path flips with filter distribution at equal cardinality.
- **Document-ID locality / segment concentration** — insertion-order artifacts; a model fitted on natural
  distributions collapsed to 56–69 % on shuffled/adversarial held-out data.
- **Sampled vector-space dispersion** — did not generalize and added 40–215 % feature overhead.
- **Runtime ANN probe** — Lucene exposes no resumable traversal state, so a probe is duplicate work; probe
  signals don't discriminate at cheap budgets (50 % selection) and a discriminative budget ≈ 85 % of full
  ANN cost, net slower than both paths in the majority of cases; it also cannot detect completed-but-lossy
  ANN.
- **Dimension-aware `ops/dim` exact limit** — raises worst-case exact latency (see §5).

## 11. Risks

- The rule underuses exact in some cases (e.g. random filters where exact wins well past the limit). This
  is intentional — it prioritizes a bounded worst case over maximal benchmark wins.
- Exact runs eagerly in `createWeight` (like the existing rescore path); bounded to `candidateLimit`
  vectors, but does not check `QueryTimeout` mid-scan. Follow-up: thread `QueryTimeout` into `ExactSearcher`.
- `candidateLimit` is a single internal constant, not yet workload-tunable (deliberate — see non-goals).

## 12. Rollout and testing plan

- Ship behind the fixed internal `candidateLimit`; no setting. Optionally gate as experimental.
- Unit + integration tests cover: zero/one/<k/=k/limit/limit+1 matches, deletions, multi-segment,
  reader-generation safety, sparse-iterator contract, equality/hashCode, and that every unsupported
  configuration bypasses the wrapper unchanged.
- Benchmarks (test-source, not shipped) reproduce the latency/recall and calibration evidence.

## 13. Open questions

1. Should `candidateLimit` eventually be an index-level *experimental* setting for recall-sensitive users?
2. Should the exact path thread `QueryTimeout` for very large `k` (candidateLimit up to `10·k`)?
3. Is the `MATCH_NONE` / small-filter overlap with Lucene worth removing, or is the extra clarity fine?
