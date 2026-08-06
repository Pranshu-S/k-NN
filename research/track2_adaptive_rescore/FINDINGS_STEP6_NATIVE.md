# Track 2, Step 6 — Native HNSW two-knob (ef_search × rerank_depth): results & verdict

**Branch:** `track2-adaptive-rescore`. **Tier:** a REAL Lucene HNSW graph (genuine traversal, real
`ef_search`), *not* a flat scan. Graph built once per segment on 1-bit codes (Hamming — the Faiss
binary-graph analogue); the three rankers SEARCH the same graph with different query scorers
(ADC / Hamming / Corrected-8-byte), like Faiss ADC-on-binary-graph — same graph settings for a fair
comparison. Work metrics (nodes_visited, distance_comps, exact_reranks) are noise-free; **latency is
WARM/indicative only** — this sandbox can't pin CPUs or evict page cache, so cold-cache and
production p99 are deferred to real hardware (per the task's own noise-control requirements). Real
`OneBitScalarQuantizer` / `transformWithADC` / `KNNScoringUtil` + step-3 8-byte metadata. Harness:
[`Track2NativeHnswTests.java`](../../src/test/java/org/opensearch/knn/research/Track2NativeHnswTests.java).
Raw: [`results/track2_native_configs.csv`](results/track2_native_configs.csv),
[`results/track2_native_selected.csv`](results/track2_native_selected.csv).

## Central question — answered with a caveat

*Can per-segment calibration jointly reduce HNSW traversal AND exact reranking while matching the
current OpenSearch fixed policy's recall?* **The calibration/feasibility machinery works, but on a
1-bit-code graph high recall is candidate-GENERATION-bound (needs high ef), so the flat-scan rerank
savings largely do not materialize — and the corrected ranker's flat-scan edge does NOT transfer to
graph navigation.** The native tier is a reality check, not a confirmation of automatic savings.

## Finding 1 — the two knobs are genuinely SEPARATE now (the thing flat-scan couldn't show)

Candidate-pool ceiling **grows with `ef_search`** (SIFT, ADC), and so does graph work:

| ef_search | ceiling | mean nodes visited | warm p50 µs |
|---|---|---|---|
| 20 | 0.559 | 333 | 378 |
| 50 | 0.769 | 596 | 100 |
| 100 | 0.894 | 929 | 163 |
| 200 | 0.956 | 1423 | 273 |
| 500 | **0.995** | 2372 | 539 |

So `ef_search` is a real candidate-generation knob that sets the recall **ceiling** (independent of
rerank depth); to meet SLA 0.95 you must first pick `ef` high enough that ceiling ≥ 0.95 (SIFT needs
`ef ≥ 200`), *then* choose rerank depth. Flat-scan (ef=∞) hid this entirely.

## Finding 2 — on a 1-bit graph, high recall is GENERATION-bound; rerank savings mostly vanish

The per-segment optimum on SIFT is **ef=500, rerank=500** — "search wide, rerank deep" — because the
ceiling only reaches 0.95 at ef≥200 and the conservative bound needs ef=500. The rerank-depth knob
can't save work when the *graph* is the bottleneck. vs the OpenSearch-style default:

| policy | ef | rerank | held-out recall | work cost | warm p50/p99 µs | status |
|---|---|---|---|---|---|---|
| **os_default** (ADC) | 100 | 100 | **0.904** | 1929 | 146 / 233 | **MISSES SLA** |
| **per-segment calibrated** | 500 | 500 | 0.996 | 7372 | 518 / 986 | CALIBRATED |

Honest reading: the cheap default **doesn't reach 0.95**; genuinely meeting the SLA costs ~2.8× more
work — and the calibrator correctly finds that minimum (it matches best-fixed-at-equal-recall). So
this is **not** "calibration reduces cost below the default" — on a 1-bit graph the default is simply
non-compliant, and 0.95 recall genuinely requires wide `ef`.

## Finding 3 — corrected ranking does NOT transfer to graph navigation (native-only, decisive)

Candidate ceiling at ef=500, per ranker:

| segment | ADC | Hamming | Corrected |
|---|---|---|---|
| **sift_real** (real) | **0.995** | 0.957 | **0.904** 🔴 |
| isotropic (768) | 0.741 | 0.545 | **0.853** |
| clustered (128) | 0.582 | 0.383 | **0.852** |

On **SIFT, ADC navigates the Hamming-built graph far better than Corrected** (0.995 vs 0.904) — the
*opposite* of flat-scan, where Corrected was a strong ranker. The graph's edges were chosen by the
binary-code (Hamming) geometry, which ADC's transformed-query scoring aligns with; Corrected's
fp32-query·reconstruction geometry mismatches the graph on real data. On synthetic, Corrected's
better distance estimate wins despite the mismatch. **This directly confirms the task's caution:
flat-scan ranker advantages do NOT automatically translate to HNSW navigation** — the corrected
ranker is a native win only where it *also* raises the graph ceiling (synthetic), and a native loss
on SIFT.

## Finding 4 — hard/high-dim segments are graph-INFEASIBLE at 1-bit even at max ef (flagged)

isotropic-768 and clustered-128 top out at ceiling 0.74/0.58 (ADC) or 0.85/0.85 (Corrected) at
ef=500 — **< 0.95 at max config → `SLA_UNACHIEVABLE_AT_MAX_CONFIG`**, correctly reported (not
silently maxed out). Remedy is `INCREASE_EF_SEARCH` beyond 500, a **higher-fidelity graph
representation (more bits / fp32 graph)**, or a better ranker — not calibration. (Also note the
isotropic warm latencies are large — p50 2–6 ms — from 768-D Hamming hops on a poorly-connected
high-dim graph; a real cost signal, warm-only.)

## Verdict

```
PARTIAL-GO with strong native caveats — a reality check, not a green light.
Validated:
  - Two-knob structure is REAL: ef_search is a candidate-generation/ceiling knob genuinely separate
    from rerank depth; per-segment calibration + explicit feasibility statuses work correctly.
  - The conservative calibrator finds the honest minimum config that meets the SLA (matches
    best-fixed-at-equal-recall); the cheap OS-style default simply misses 0.95 on a 1-bit graph.
Native caveats that flat-scan hid:
  - On 1-bit-code HNSW, high recall is GENERATION-bound (needs wide ef) -> the rerank-depth savings
    promised by flat-scan largely do NOT materialize; the joint optimum is "wide ef, deep rerank".
  - The corrected 8-byte ranker's flat-scan edge does NOT transfer to graph navigation (native LOSS
    on real SIFT; win only on synthetic where it also raises the ceiling).
  - Hard/high-dim segments are graph-infeasible at 1-bit even at max ef -> need a better graph
    representation, not calibration.
  - Latency is warm-only; cold-cache / production p99 require real hardware.
Decision: the calibration+feasibility system is sound and worth keeping, but the DOMINANT lever for
native cost/recall is graph/candidate-generation quality (bits per code, ef), not rerank-depth
calibration. Do not claim flat-scan rerank savings as native latency savings.
```

## Where this points

The binding native constraint is the **candidate-generation ceiling of the 1-bit graph**. The single
change that would move every number here is a **higher-fidelity ranker/graph** — a rotation-based
multi-bit **RaBitQ/OSQ** representation, which raises the ceiling at a given `ef` (fewer hops for the
same recall) *and* aligns navigation with the estimate. That is the same conclusion Track 1 (cosine
correction) and Track 2 steps 3–5 reached from other directions, now confirmed natively.

## Production design note (minimal future integration — not implemented)

At flush/merge (fp32 present): for a ~50-query sample, sweep a few `ef` values on the segment's
quantized graph, record the candidate ceiling and the `recall@depth` curve, pick the cheapest
conservative `(ranker, ef, rerank_depth)` for the SLA + margin (or emit `SLA_UNACHIEVABLE` +
fallback), and store `(ranker, ef_search, candidate_count, rerank_depth, status)` in **segment
metadata**. At query time, read the segment's config; provide a user override and a conservative
fallback (global config / full rerank) for small or infeasible segments. No per-query bound math.

## Reproduce
```bash
./gradlew :test --tests "org.opensearch.knn.research.Track2NativeHnswTests" \
    -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
# writes results/track2_native_{configs,selected}.csv
```
Hardware note: run cold-cache / true p99 on a controlled machine (pinned cores, dropped page cache);
these sandbox numbers are warm/indicative and must not be read as production latency.
