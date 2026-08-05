# Track 2 — Adaptive Rescoring: Step 1 findings (is the QFrame score calibrated?)

**Branch:** `track2-adaptive-rescore`. **Tier:** flat exhaustive quantized scan (no HNSW graph), so
this isolates the *estimator* from traversal. Real scorers: 1-bit Hamming and ADC (`l2SquaredADC`).
Raw: [`results/track2_calibration.csv`](results/track2_calibration.csv). N=5000, NQ=100, 2000 sampled
pairs/query.

A bound-based adaptive policy needs, per candidate, an estimated distance **and a trustworthy
uncertainty bound**. Step 1 (mandated) measures whether the shipping score supports that. Two things
must both hold: the bound must be **valid** (covers the true distance at the claimed confidence) and
**tight** (narrow enough to separate the top-k boundary and let us stop early).

## Result 1 — the bound is statistically VALID (good news)

Under a Gaussian `pred ± z·σ` model with the measured global σ, empirical coverage ≈ nominal
**almost everywhere**:

| estimator | datatype | dim | cover90 | cover95 | cover99 | skew | exkurt |
|---|---|---|---|---|---|---|---|
| Hamming | isotropic | 1536 | 0.901 | 0.951 | 0.991 | 0.12 | −0.04 |
| ADC | isotropic | 1536 | 0.900 | 0.950 | 0.990 | 0.06 | −0.01 |
| Hamming | clustered | 768 | 0.903 | 0.950 | 0.991 | 0.00 | −0.09 |
| Hamming | sift_real | 128 | 0.904 | 0.951 | 0.988 | 0.40 | 0.13 |

Residuals are near-Gaussian (|skew|<0.5, small kurtosis) and roughly homoscedastic
(σ_lowbin ≈ σ_highbin), so a `z·σ` bound covers as claimed. **The policy will not silently lose
recall if built with per-candidate σ + a multiple-comparison correction.** (Mild exception:
anisotropic residuals are right-skewed (0.5–0.84) with fatter tails (exkurt up to 1.4) → the 99%
bound slightly under-covers (0.983–0.988); use a larger z there.)

## Result 2 — but the estimator is TOO NOISY for tight bounds (the blocker)

`resid_std_norm` = residual std ÷ exact-distance std, after best-fit affine calibration:

| estimator | isotropic | anisotropic | clustered | **sift_real** |
|---|---|---|---|---|
| Hamming | 0.89–0.90 | 0.97–0.997 | 0.88 | **0.54** |
| ADC | 0.70–0.74 | 0.74–0.80 | 0.83–0.90 | **0.99** |

The calibrated 1-bit estimate leaves **70–99% of the distance spread as residual noise** on synthetic
data. A bound of `±z·σ` with σ ≈ 0.7–1.0× the whole distance std is **enormous** — candidate
intervals overlap massively, so the boundary test `(ê_next − z·σ) ≤ (ê_k + z·σ)` will **almost never
be satisfiable**, i.e. the policy cannot confidently stop early and will hit the **max fallback on
most queries**. **The headroom for adaptive depth reduction is limited by estimator variance, not by
bound validity.**

Two important nuances:
- **Real data is much better calibrated.** SIFT-128 Hamming residual is **0.54** (vs ~0.89 synthetic),
  with the *lowest* residual in the close-pair bin (σ_lowbin 0.43) — exactly where the top-k boundary
  lives. So on real embeddings the adaptive policy has materially more room than synthetic suggests.
- **Better ranking ≠ better distance estimate.** ADC *ranks* better (Track 1) yet on SIFT its
  *calibrated distance* is noisier than Hamming (0.99 vs 0.54). The adaptive bound depends on the
  distance-estimate variance, not the ranking — so for bound-based rerank, **Hamming may be the better
  signal on real data**, a counter-intuitive finding worth carrying forward.

## Step-1 verdict & the documented dependency

- **Bounds are valid → a correct, recall-safe adaptive policy is buildable.**
- **But on the raw 1-bit score the bounds are wide → aggressive skip/shallow-rerank has little
  headroom on synthetic data** (more on real data). This is exactly the plan's contingency: the
  smallest necessary addition to unlock adaptive depth is a **tighter estimator** — per-vector
  correction metadata (RaBitQ/OSQ-style norms/intervals) or more bits — to shrink σ. That is the
  **documented dependency** for the prototype's upside.

## Next (this branch)
1. Build the adaptive-depth policy over this scan and **quantify** the headroom: skip-rate,
   selected-depth distribution, avg/p99 reranked, **false-safe rate**, vs fixed 1×/2×/3×/5×/min-100
   and best-fixed-offline — segmented by query difficulty (k-th margin).
2. Add a minimal per-vector correction estimator; re-measure σ and re-run the policy to show the
   headroom it unlocks.
3. Native tier: the same under real HNSW (attributing misses to quantization vs HNSW discovery),
   with p50/p99 and FP32 bytes read.
