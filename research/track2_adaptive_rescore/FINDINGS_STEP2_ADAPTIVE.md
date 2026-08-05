# Track 2, Step 2 — Bound-based adaptive rerank-depth: results & go/no-go

**Branch:** `track2-adaptive-rescore`. **Tier:** flat-scan simulator (no OpenSearch integration; no
per-vector correction estimator — both explicitly out of scope). Real scorers (Hamming, ADC). Config:
k=10, candidate_pool=1000, min_rerank=10, max_rerank=1000, check_step=1, confidence=0.99. Combos:
{Hamming-rank+Hamming-bound, ADC+ADC, ADC-rank+Hamming-bound} × {none, Bonferroni}. Datasets:
SIFT-128 (real), isotropic-768, clustered-128. NQ=100.
Raw: [`results/track2_adaptive_summary.csv`](results/track2_adaptive_summary.csv),
[`results/track2_adaptive_perquery.csv`](results/track2_adaptive_perquery.csv). Harness:
[`Track2AdaptiveRescoreTests.java`](../../src/test/java/org/opensearch/knn/research/Track2AdaptiveRescoreTests.java).

## Algorithm (as implemented)

Safe-stop, all-remaining form: after exact-reranking prefix `[0,r)` (ranking-estimator order) with
`exact_k` = current k-th best exact distance, **stop iff** `min over all remaining j of lower_j >
exact_k`, where `lower_j = pred_j − z·σ` (calibrated bound-estimator prediction). Because calibration
found σ ~homoscedastic, a single global σ is used, so `min lower = suffix_min_pred[r] − z·σ`, tested in
O(1)/checkpoint from a precomputed suffix-min array + an incremental top-k max-heap. Bonferroni:
`m = remaining candidates`, per-comparison `α' = α_query/m`, two-sided `z = Φ⁻¹(1 − α'/2)` (Acklam),
recomputed as r grows. Ranking and bound estimators are independent; bounds always use the same
candidate's bound-estimator score.

## Headline result table

| combo | corr | recall | false-safe | mean depth (ovs) | oracle depth | savings vs best-fixed |
|---|---|---|---|---|---|---|
| **SIFT-128** ADC/Hamming | Bonferroni | 0.999 | **0.000** | **1000 (100×)** | 167 | **−9.0** |
| SIFT-128 ADC/ADC | none | 0.977 | 0.060 | 941 (94×) | 167 | −8.4 |
| SIFT-128 Hamming/Hamming | Bonferroni | 0.990 | 0.000 | 1000 (100×) | 342 | −9.0 |
| **iso-768** ADC/Hamming | Bonferroni | 0.904 | 0.010 | 999 (100×) | 697 | −9.0 |
| iso-768 ADC/ADC | none | 0.654 | **0.890** | 323 (32×) | 697 | −2.2 |
| **clus-128** ADC/Hamming | Bonferroni | 0.674 | 0.000 | 1000 (100×) | 803 | −9.0 |

(Full 18-row matrix in the CSV. `savings = 1 − mean_depth/best_fixed_at_equal_recall`; negative = the
policy reranks *more* than the best fixed baseline. Metric note: `max_fallback_rate` reads 0 because
the r=1000 checkpoint trivially certifies once zero candidates remain — functionally the policy
reranks the whole pool, i.e. ~100× oversample.)

## The 9 interpretation questions, answered

1. **Are the bounds safe after multiple-comparison correction?** **Yes.** With Bonferroni the
   false-safe rate is **0.000** (max 0.02, isotropic ADC/ADC). The correction + all-remaining rule
   gives a real query-level guarantee.
2. **False-safe rate?** With correction: ~0 (0.000–0.020). **Without correction: up to 0.890**
   (isotropic ADC/ADC) — uncorrected 95/99% intervals do NOT provide query-level safety, as warned.
3. **How often max fallback?** **Effectively always** under safe correction — mean depth = 1000
   (= full pool), ~100× oversample, in every dataset/combo. The bound never certifies an early stop.
4. **Work saved vs best fixed at equal recall?** **None — it's negative (−8.4 to −9.0).** No fixed
   depth reaches the adaptive recall, and the policy reranks the whole pool, so at equal recall the
   adaptive policy does ~10× *more* work than fixed-100.
5. **Adaptive opportunity per oracle?** **Real but unreachable.** Oracle mean depth is **167** on
   SIFT with ADC ranking (6× headroom vs the 1000-pool) and 697–836 on synthetic (~1.2–1.4×). So the
   opportunity exists, especially on real data.
6. **Is the gap to oracle primarily estimator variance?** **Yes, entirely.** Oracle 167 vs policy
   1000 on SIFT. Bounds are valid (boundary coverage 0.96–1.00) and the correction is correct; the
   *only* reason the policy can't approach the oracle is that residual σ (0.5–1.0× the distance spread)
   makes `±z·σ` intervals too wide to separate the boundary — worse after the Bonferroni z-inflation
   (m≈1000 ⇒ z≈4.4).
7. **ADC-rank + Hamming-bound vs ADC-rank + ADC-bound?** **Hamming bounds are safer** — without
   correction, ADC/Hamming false-safe 0.00–0.02 vs ADC/ADC 0.06–0.89, at equal-or-higher recall.
   Confirms step-1's insight: **Hamming is the better *stopping* signal even when ADC ranks.** But
   neither yields savings under valid correction.
8. **Which difficulty segments benefit?** **None.** Under safe correction, mean depth ≈ 1000 across
   easy / medium / hard boundary-margin tertiles alike — even large-margin "easy" queries can't be
   certified early, because per-candidate σ dwarfs the k-th boundary margin.
9. **Integrate now, or is correction metadata required first?** **NO-GO for the current 1-bit
   estimator. A tighter estimator (per-vector correction metadata) is required first.** The bound
   framework is correct and safe; the 1-bit score is simply too noisy to certify early stopping under
   valid query-level safety. The oracle proves the opportunity is real, so the unlock is estimator
   variance, not the policy.

## Go / No-Go

```
Go/No-Go (current 1-bit QFrame estimator, bound-based adaptive rerank depth):
  NO-GO for integration.
  - Under correct query-level safety (Bonferroni + all-remaining rule), the policy certifies NO early
    stop: ~100x oversample, negative savings vs fixed baselines, no difficulty segment benefits.
  - Uncorrected intervals reach lower depth but at false-safe rates up to 0.89 — unsafe; do NOT ship.
  - The bound math is validated (false-safe ~0 with correction; boundary coverage ~nominal). The
    blocker is estimator VARIANCE (residual sigma 0.5-1.0x distance spread), confirmed by the
    oracle gap (167 achievable vs 1000 used on SIFT).
  Decision: the adaptive-depth idea is sound and safe, but wasteful on the current estimator.
  Revisit AFTER adding a tighter per-vector estimator; the harness is ready to re-measure.
```

## Where a future per-vector correction estimator plugs in (design note, not implemented)

Exactly one seam: the **bound model** `(pred_j, σ_j)`. Today `pred_j = a·score_j + b` with a global σ
from `calibrate(...)`. A corrected estimator (RaBitQ/OSQ-style per-vector norms/intervals) would:
- replace `pred_j` with the corrected unbiased distance estimate, and
- replace the global σ with a **per-candidate σ_j** (the estimator's own error bound), which the
  all-remaining rule already supports in principle (revert `suffix_min_pred` to a true
  `suffix_min(pred_j − z·σ_j)` when σ is non-constant).
Nothing else in the policy, correction, or metrics changes. Success criterion for the re-run:
`mean_depth` approaching the **oracle** (167 on SIFT) at false-safe ≈ 0 — i.e. σ shrinking from
~0.5–1.0× to a small fraction of the distance spread. Until then, a **fixed** oversample remains the
better operating point than bound-based adaptive on the current estimator.

## Reproduce
```bash
./gradlew :test --tests "org.opensearch.knn.research.Track2AdaptiveRescoreTests" \
    -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
# writes results/track2_adaptive_{summary,perquery}.csv
```
CLI-equivalent config (constants in the test; see TRACK2_DESIGN.md): `--k 10 --candidate-pool 1000
--min-rerank 10 --max-rerank 1000 --check-step 1 --confidence 0.99 --correction {none,bonferroni}
--ranking-estimator {hamming,adc} --bound-estimator {hamming,adc} --dataset {sift_real,isotropic,clustered}`.
