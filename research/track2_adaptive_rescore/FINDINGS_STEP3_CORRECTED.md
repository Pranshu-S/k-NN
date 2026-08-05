# Track 2, Step 3 — Minimal per-vector corrected estimator: results & go/no-go

**Branch:** `track2-adaptive-rescore`. Only the `(pred_j, σ_j)` seam changed; the adaptive policy,
Bonferroni correction, candidate generation (ADC ranking), baselines, and metrics are unchanged.
No OpenSearch integration. Harness:
[`Track2CorrectedEstimatorTests.java`](../../src/test/java/org/opensearch/knn/research/Track2CorrectedEstimatorTests.java).
Raw: [`results/track2_corrected_summary.csv`](results/track2_corrected_summary.csv),
[`results/track2_corrected_perquery.csv`](results/track2_corrected_perquery.csv).

## 1. Corrected estimator — formula & derivation

`‖q−x‖² = ‖q‖² + ‖x‖² − 2·q·x`. The 1-bit code reconstructs `x̂` from the per-dim below/above
threshold means (already in the trained state), so the current score implicitly uses `‖x̂‖²`; the
dominant, cheaply-fixable error is the **norm term**.

```
pred_j   = ‖q‖² + ‖x_j‖² − 2·(q·x̂_j)          # store the TRUE norm; q·x̂ = the ADC LUT estimate
residual = exact − pred = −2·q·(x_j − x̂_j)     # the irreducible dot-product quantization error
σ_j      = c · 2·‖q‖·‖x_j − x̂_j‖ / √d          # E[(q·r)²]=‖q‖²‖r‖²/d ; c fit for coverage, σ floored
```

## 2. Metadata (2 floats/vector = 8 bytes)

`‖x_j‖²` and residual norm `‖x_j − x̂_j‖`. Query feature `‖q‖` (once/query). Both computable at
index time. Extra compute/candidate ≈ free in production: `q·x̂_j` is exactly the ADC LUT value
already computed for ranking, plus one add (norm) and a few ops for `σ_j`.

## 3. Calibration / evaluation split (no leakage)

`a,b,c` fit on a **held-out** set — SIFT `learn` (80 queries) / synthetic seed+1; evaluated on
disjoint SIFT `query` (80) / synthetic seed+2. Docs N=10000. Fixed seeds.

## 4. Calibration results — the correction TIGHTENS the estimator (clear win)

Global residual σ ÷ distance-spread, and RMSE, under the **safe** (Bonferroni) policy:

| dataset | σ/dist ADC | σ/dist Hamming | **σ/dist Corrected** | RMSE ADC → Ham → **Corr** |
|---|---|---|---|---|
| sift_real 128 | 1.115 | 0.525 | **0.217** | 1.66e5 → 4.1e4 → **3.5e4** |
| isotropic 768 | 0.822 | 0.976 | **0.483** | 66.7 → 71.1 → **33.7** |
| clustered 128 | 1.158 | 1.193 | **0.543** | 1555 → 1058 → **466** |

**The 8-byte norm correction cuts global σ ~2× and RMSE ~2× vs the best prior estimator** — a real,
cheap estimator-quality improvement. Validates that per-vector metadata reduces distance-estimation
variance.

## 5. Coverage — a caveat: tight on synthetic, UNDER-COVERS on real SIFT

| dataset | cover90 | cover95 | cover99 | boundary cover95 |
|---|---|---|---|---|
| **sift_real** | 0.708 | **0.784** | 0.888 | **0.679** 🔴 |
| isotropic | 0.900 | 0.950 | 0.990 | 0.887 |
| clustered | 0.904 | 0.953 | 0.995 | 0.889 |

On synthetic the corrected σ is **well-calibrated (nominal coverage) AND tighter** — the ideal. But
on **real SIFT it under-covers** (cover95 0.78, boundary 0.68): the random-direction σ model
(`‖q‖·‖r‖/√d`) breaks on structured real data, so part of SIFT's apparent tightness is unsafe
under-coverage. **Per the task's rule, the SIFT σ is not acceptable as-is** — it needs a
boundary-conservative / geometry-derived error bound (RaBitQ's actual bound), not the isotropic
approximation.

## 6. Adaptive result under the UNCHANGED safe policy — still NO savings

| dataset | bound | false-safe | recall | mean depth | oracle | best-fixed | savings | gap-to-oracle |
|---|---|---|---|---|---|---|---|---|
| sift_real | Corrected | **0.000** | 0.999 | **971.8** | 175 | 90 | **−9.8** | 796 |
| isotropic | Corrected | 0.000 | 0.915 | 995.2 | 626 | 100 | −9.0 | 370 |
| clustered | Corrected | 0.000 | 0.629 | 998.9 | 761 | 90 | −10.1 | 238 |

Even with σ halved, **mean depth stays ~972–999 (≈ full pool)**; savings vs best-fixed remain
≈ **−10** (reranks ~10× more than fixed-90/100); gap-to-oracle barely closes (SIFT 825→796). No
difficulty segment benefits (easy-query mean depth 983–999). Uncorrected (diagnostic) stops a bit
earlier (SIFT 887) but at 6.2% false-safe — unsafe.

**Why:** Bonferroni over m≈1000 gives z≈4.4, so even at the well-calibrated synthetic σ/dist≈0.48,
the interval half-width `z·σ ≈ 2×` the distance spread — vastly wider than the k-th boundary margin
(a tiny fraction of the spread in ANN). To certify a stop, `z·σ` must drop **below the boundary
margin**, needing σ/dist ~ 0.01–0.05 — **another ~10× reduction**. The norm correction delivers ~2×;
the rest is out of reach for this minimal estimator.

## 7. Evaluation-question answers

- **Preserves false-safe ≈ 0?** Yes under Bonferroni (0.000) — but on SIFT only because it never
  stops early; its σ under-covers, so it would be unsafe *if* it did.
- **Reduces mean/tail depth?** No (≈ unchanged, ~972–999; p99 = 1000).
- **Reduces max-fallback?** No.
- **Closes gap to oracle?** Barely (SIFT 825→796).
- **Beats best fixed at equal recall?** No (savings −9 to −10).

## 8. Metadata & runtime cost

- **8 bytes/vector** (2× float32). vs the 1-bit code (`d/8` B): **+50% at 128-D, +8% at 768-D,
  +4% at 1536-D**.
- Totals: **1M → 8 MB, 100M → 800 MB, 1B → 8 GB** (on top of codes + graph).
- Query-time: `q·x̂` reuses the ADC LUT (free); +1 add (norm) + a few ops for σ_j per candidate —
  negligible. Calibration model: 3 scalars (a,b,c) + 1 floor.

## 9. Go / Partial-go / No-go

```
NO-GO for adaptive rerank-depth on this minimal corrected estimator.
- The estimator itself is a WIN: 8 bytes/vec halves global σ and RMSE; coverage nominal on
  synthetic. Direction confirmed: per-vector metadata reduces variance.
- But it does NOT unlock the policy: mean depth stays ~full pool, savings vs best-fixed ~ -10,
  gap-to-oracle barely closes, no segment benefits. Bonferroni z-inflation (z~4.4 over ~1000
  candidates) means σ/dist must reach ~0.01-0.05; the norm correction only reaches ~0.22-0.54.
- On real SIFT the σ model UNDER-COVERS (cover95 0.78, boundary 0.68) — its tightness there is
  partly unsafe; not acceptable without a geometry-derived boundary bound.
Decision: a fixed oversample still beats bound-based adaptive. Do not integrate.
```

## 10. Remaining gap to a full RaBitQ/OSQ implementation

The missing ~10× σ reduction (and the SIFT coverage fix) is exactly what the full machinery
provides, and where the next work goes:
1. **Random rotation before quantization** — makes the dot-product error genuinely random-direction
   (so the `‖q‖·‖r‖/√d` σ model becomes valid on real data, fixing SIFT under-coverage) *and*
   concentrates/shrinks it (RaBitQ's provable bound).
2. **More bits (OSQ 4-bit query / multi-bit doc)** — directly shrinks `‖x−x̂‖`, the residual driving σ.
3. **RaBitQ's per-vector unbiased estimator + closed-form error bound** — replaces the isotropic σ
   approximation with the geometry-exact one (restores nominal boundary coverage at smaller σ).
Only after σ/dist reaches the ~0.05 regime does the *already-validated, already-safe* adaptive
policy become worthwhile — at which point this harness re-runs unchanged to confirm it.

## Reproduce
```bash
./gradlew :test --tests "org.opensearch.knn.research.Track2CorrectedEstimatorTests" \
    -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
# writes results/track2_corrected_{summary,perquery}.csv
```
