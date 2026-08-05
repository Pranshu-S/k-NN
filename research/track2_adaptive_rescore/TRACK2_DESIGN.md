# Track 2 — Target-Adaptive Rescoring: design

**Branch:** `track2-adaptive-rescore` (off upstream `main` @ `6ce7035`, independent of Track 1).

## Goal

Choose FP32 rerank depth **per query** from a calibrated uncertainty bound, instead of a fixed
oversampling factor. Ideal behaviour:

```
For each candidate, the quantized search yields: estimated distance ê, and an error bound so that
the TRUE distance d lies in [ê − Δ, ê + Δ] with confidence 1−α.
Keep pulling candidates (deepening rerank) while:
    best-possible(next remaining candidate)  ≤  worst-possible(current k-th)
    i.e.  (ê_next − Δ_next)  ≤  (ê_k + Δ_k)
Stop when the top-k boundary is provably separated → skip / shallow rerank for easy queries,
deepen for uncertain ones, fall back to a configured max when inconclusive.
Batched: after each batch, recompute the EXACT k-th distance to tighten the boundary before pulling more.
```

## Mandated first step (this commit): is the QFrame score calibrated?

Per the plan, **do not assume the current QFrame score provides valid bounds.** A bound-based policy
is only sound if the quantized distance is a calibrated estimator of true distance with a known,
well-behaved error. [`Track2CalibrationTests`](../../src/test/java/org/opensearch/knn/research/Track2CalibrationTests.java)
measures, for the real Hamming (1-bit) and ADC estimators:

- **affine calibration** `true_L2 ≈ a·ê + b` (slope/intercept — is there a linear map at all?);
- **residual std** normalized by exact-distance std (how noisy is ê after calibration?);
- **skew / excess kurtosis** of residuals (a Gaussian bound assumes ~0/~0);
- **homoscedasticity** — residual std in low vs high estimate bins (a single global σ is only valid
  if constant);
- **bound coverage** — empirical `P(|true − pred| ≤ z·σ)` vs nominal 90/95/99% (under-coverage ⇒
  the bound is **unsafe** and would silently skip rerank on truly-close candidates — a recall bug).

**Decision rule for the prototype:**
- If coverage ≈ nominal and residuals are near-Gaussian & homoscedastic → build the bound from the
  measured global σ.
- If heteroscedastic → bound must use an **estimate-dependent** σ(ê).
- If residual variance is so large that bounds are never tight enough to skip → the current score
  is too weak; **document the dependency**: add per-vector correction metadata (RaBitQ/OSQ-style
  norms/intervals) to get a tighter, unbiased estimator. This is the smallest necessary addition.

## Comparison baselines (next commits)

no-rescore · current OpenSearch default (dim-keyed) · fixed 1×/2×/3×/5× · min-100 · best-fixed
(offline) · exact FP32. Metrics: recall@k, rerank-skip rate, selected-depth distribution, avg/p99
reranked candidates, FP32 bytes read, latency, bound-computation overhead, **confidence-bound
coverage**, **false-safe rate** (rerank skipped but true top-k not contained), max-fallback rate —
**segmented by query difficulty** (easy = large k-th margin, hard = dense competition near rank k).

## Correctness requirements

- Report whether the true FP32 top-k was **contained** in the selected rerank set (the only thing
  that matters for recall safety).
- Bound-failure frequency and the confidence level used.
- **Multiple-comparison correction:** the boundary test is applied across many candidates, so the
  per-candidate α must be corrected (e.g. Bonferroni / union bound) to hold a query-level guarantee —
  the harness will measure whether an uncorrected α under-covers.
- Attribute misses correctly: a miss because the quantizer mis-ranked a *scanned* candidate
  (rescoring's job) vs a miss because **HNSW never discovered** the true neighbour (traversal's job,
  NOT rescoring). Flat-scan tier isolates the former (no graph); the latter is native-tier.

## Tiers

- **Flat-scan (this branch, now):** calibration + the adaptive policy over an exhaustive quantized
  scan → measures scoring-side skip/depth/coverage/false-safe cleanly (no graph confound).
- **Native (staged):** real HNSW + rescore, p50/p99, FP32 bytes read cold/warm, bound overhead in a
  live query, and the HNSW-discovery-vs-quantization attribution.

## Files (this branch)
- `src/test/java/org/opensearch/knn/research/Track2CalibrationTests.java` — step 1, calibration.
- `research/track2_adaptive_rescore/TRACK2_DESIGN.md` — this file.
- `research/track2_adaptive_rescore/results/track2_calibration.csv` — generated.
- (next) adaptive-policy harness + FINDINGS.md.
