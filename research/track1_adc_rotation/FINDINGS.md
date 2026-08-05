# Track 1 — ADC & Rotation: validated findings (flat-scan / scoring-isolation tier)

**Branch:** `track1-adc-rotation`. **Tier:** brute-force quantized scan over ALL docs → **no HNSW
graph**, so every miss is *approximate-scoring / candidate-pool* loss and **zero traversal loss**.
This isolates "distance estimation" from "graph navigation" (validation-plan item 3, flat side).
Driven by the **real** shipping classes (`OneBitScalarQuantizer` with rotation via the 2-arg
`TrainingRequest`, `transformWithADC`, `KNNScoringUtil.{l2SquaredADC,innerProductADC}`; symmetric =
Hamming). Data: N=5000, NQ=100, seed=1234. Metric `candidateRecall@N = |true top-k ∩ quant top-N| / k`
(= final recall after FP32 rerank of top-N, since exact rerank surfaces every true neighbour in the pool).

Raw: [`results/track1_flatscan.csv`](results/track1_flatscan.csv),
[`results/track1_cosine_estimators.csv`](results/track1_cosine_estimators.csv).

> **Scope honesty.** This tier measures *scoring quality* only. It does **not** measure p99/latency,
> index size, FP32 bytes read, cold/warm cache, indexing throughput, or the HNSW-traversal interaction.
> Those require the native JNI build + a node + opensearch-benchmark and remain **NOT MEASURED** here.
> Real embeddings covered by one real anchor (SIFT-128, image descriptors); real 768/1536-D text/image
> embeddings need downloads and are also staged.

---

## 1. The rotation no-op from run 1 was a harness bug (now fixed & verified)

`QuantizerHelper` reads the rotation flag from `trainingRequest.isEnableRandomRotation()`, not from
the quantizer constructor. Run 1 used the 1-arg `TrainingRequest` → flag false → matrix never built →
rotation inert. Fixed by the 2-arg `TrainingRequest(n, enableRotation)`. The v2 run asserts
`state.getRotationMatrix() != null` iff rotation enabled — **BUILD SUCCESSFUL confirms rotation is now
applied** (rotation differs from baseline in 21/21 L2 cells). No rotation conclusion was ever drawn
from the buggy run.

## 2. Where the ADC gain comes from — scoring, not traversal (item 2 & 3-flat, VALIDATED)

Because the scan has no graph, the candidateRecall@N growth is *purely* better scoring. SIFT-real, L2, k=10:

| config | cr@k | cr@2k | cr@3k | cr@5k | cr@10k(=100) | Spearman |
|---|---|---|---|---|---|---|
| baseline | 0.324 | 0.474 | 0.553 | 0.652 | 0.786 | 0.849 |
| **ADC** | **0.407** | **0.592** | **0.709** | **0.819** | **0.922** | **0.908** |
| rotation | 0.393 | 0.583 | 0.664 | 0.772 | 0.898 | 0.908 |
| **ADC+rot** | **0.511** | **0.717** | **0.813** | **0.900** | **0.971** | **0.931** |

**"ADC improves recall after reranking because it places more true neighbours in the rerank pool" —
CONFIRMED and measured directly**: ADC lifts pool coverage at N=100 from 0.786 → 0.922 (+0.136), and
ADC+rot → 0.971. Spearman rises 0.849→0.931, i.e. genuinely better *ordering*, not luck.

## 3. Low absolute recall = test-data difficulty, not a bug/graph/ef (item 4, DIAGNOSED)

Flat scan has no graph, so ef_search/graph construction are ruled out by construction. The cause is the
dataset: random-Gaussian synthetic is pathologically hard for 1-bit; real data is ~3–4× easier.
L2, k=10, cr@k (baseline / ADC):

| datatype | baseline cr@k | ADC cr@k | baseline cr@100 |
|---|---|---|---|
| **sift_real (128, real)** | **0.324** | **0.407** | **0.786** |
| isotropic (synthetic) | 0.096 | 0.188 | 0.374 |
| uniform | 0.172 | 0.282 | 0.531 |
| clustered | 0.031 | 0.059 | ~0.30 |
| anisotropic | 0.020 | 0.032 | ~0.18 |

Verdict: **the low synthetic numbers are data difficulty; there is no evidence of a scoring/ground-truth
bug** (Spearman is positive everywhere; real data behaves well). ADC does **not** rescue anisotropic
(the hardest case) at the pool top (+0.005 only).

## 4. Rotation is data-dependent — it is NOT a free win (item 5)

Δ vs baseline, L2, k=10 (cr@k / cr@5k):

| datatype | ADC | rotation | ADC+rot | additive? |
|---|---|---|---|---|
| **sift_real** | +0.083 / +0.167 | **+0.069 / +0.120** | **+0.187 / +0.248** | **super-additive (synergy)** |
| anisotropic | +0.005 / +0.005 | +0.008 / +0.014 | +0.005 / +0.011 | ~additive |
| isotropic | +0.072 / +0.173 | **−0.013** / +0.013 | +0.065 / +0.181 | ~additive |
| clustered | +0.047 / +0.107 | −0.006 / +0.015 | +0.030 / +0.099 | sub-additive |
| **uniform** | +0.080 / +0.177 | **−0.067 / −0.134** | +0.027 / +0.072 | rotation *drags down* |

Answers: **Does rotation hurt naturally isotropic/axis-aligned data? YES** — slightly on isotropic
(−0.013) and materially on uniform (−0.067). **Does rotation help the weak cases?** Marginally on
anisotropic (+0.008); strongly on real data (+0.069). **Additive with ADC?** Roughly, except
**super-additive on real data** (adc+rot 0.187 > adc 0.083 + rot 0.069) and sub-additive on clustered.

## 5. "ADC hurts cosine" is FALSE — it's a missing metric-specific estimator (item 6)

`shouldDoADCCorrection(spaceType)` returns true **only for L2**
(`OneBitScalarQuantizer.java:147-152`); cosine/IP get the *uncorrected* `innerProductADC` path. Comparing
three asymmetric-cosine scorers (unit vectors), cr@k / cr@5k:

| scorer | sift_real | isotropic | uniform | anisotropic |
|---|---|---|---|---|
| baseline Hamming | 0.326 / 0.651 | 0.136 / 0.349 | 0.235 / 0.531 | 0.124 / 0.294 |
| **cos_current_adc** (no correction) | **0.196 / 0.521** 🔴 | 0.124 / 0.336 | 0.147 / 0.380 🔴 | 0.108 / 0.290 |
| **cos_normalized→L2adc (corrected)** | **0.406 / 0.823** 🟢 | 0.271 / 0.595 | 0.384 / 0.765 | 0.228 / 0.535 |

**The current cosine ADC path REGRESSES vs baseline** (SIFT 0.196 < 0.326). But **normalizing and
applying the L2 ADC correction RECOVERS and beats baseline** (0.406 > 0.326), matching the L2 story.
Since cosine rank == L2 rank on unit vectors, this is mathematically sound. **Conclusion: ADC is NOT
fundamentally unsuitable for cosine; the shipping cosine estimator is simply uncorrected.** A
metric-specific (normalized-L2) correction is the fix — a small implementation change.

---

## 6. Success / failure matrix (flat-scan tier; p99 / FP32-bytes are native-tier = not measured)

| configuration | dataset | dim | metric | distribution | rerank depth | candidate recall | success/failure | reason |
|---|---|---|---|---|---|---|---|---|
| ADC | sift_real | 128 | L2 | real | N=50 vs base N=100 | 0.819 ≥ base 0.786 | **SUCCESS** | equal recall at ≥30% fewer reranked candidates |
| ADC | synthetic (iso/uniform/clustered) | 128–1536 | L2 | synthetic | N=k…10k | +0.03–0.11 cr@k | **SUCCESS (ordering)** | consistent pool-coverage gain, grows with rerank |
| ADC | any | 128–1536 | L2 | anisotropic | any | +0.005 cr@k | **NO MEASURABLE BENEFIT** | hardest case; ADC ≈ baseline at pool top |
| ADC (current path) | any | 128–1536 | cosine | all | any | −0.03 to −0.13 cr@k | **REGRESSION** | uncorrected `innerProductADC`; needs metric-specific correction |
| ADC (normalized+L2 corr) | any | 128–1536 | cosine | all | any | +0.08–0.15 cr@k | **SUCCESS (needs impl)** | corrected estimator recovers the L2-class gain |
| rotation | sift_real | 128 | L2 | real | any | +0.069 cr@k | **SUCCESS (opt-in)** | helps real data; synergistic with ADC |
| rotation | any | 128–1536 | L2 | uniform | any | −0.067 cr@k | **REGRESSION** | rotation destroys axis-aligned structure |
| rotation | any | 128–1536 | L2 | isotropic | any | −0.013 cr@k | **NO/NEGATIVE** | isotropic is rotation-invariant; slight noise cost |
| ADC+rotation | sift_real | 128 | L2 | real | N=100 | 0.971 cr@100 | **SUCCESS (opt-in)** | best config on real data (super-additive) |

Classification:
- **L2-only default (candidate ordering): ADC** on L2 — validated on real + synthetic.
- **Opt-in: rotation** (and ADC+rotation) — real/anisotropic data only; **regresses uniform/isotropic**.
- **Needs metric-specific correction: cosine ADC** — current path regresses; corrected path succeeds.
- **No measurable benefit: ADC on anisotropic** at the pool top.
- **Blocked / not measured: everything latency/size/HNSW** — native tier.

---

## 7. Final recommendation (conservative, evidence-gated)

```
Validated conclusions:
- ADC improves L2 candidate ordering by placing more true neighbours in the rerank pool
  (SIFT-128 L2: cr@100 0.786 → 0.922; Spearman 0.849 → 0.908). Gain grows with rerank depth.
- On SIFT-128 L2, ADC reaches baseline's N=100 recall at N=50 (≥30% fewer reranked candidates).
- Low absolute recall on synthetic data is TEST-DATA DIFFICULTY, not a graph/ef/ground-truth bug
  (real data is 3–4× higher; flat scan has no graph to blame).
- Rotation is DATA-DEPENDENT: helps real/anisotropic, HURTS uniform (−0.067) and isotropic (−0.013).
- ADC+rotation is SUPER-ADDITIVE on real data (SIFT-128 L2 cr@k 0.511, cr@100 0.971).

Invalidated conclusions:
- "Rotation does nothing" — was a harness bug (flag not set on TrainingRequest); now measured.
- "ADC hurts cosine (fundamental)" — FALSE. The CURRENT cosine path is uncorrected and regresses;
  normalized→L2-ADC-with-correction recovers and beats baseline (SIFT cr@k 0.196 → 0.406).

Safe to enable:
- Nothing yet at production scope — latency/index-size/HNSW recall are unmeasured (native tier).
  Within the SCORING tier, ADC on L2 is a clear, consistent win with no observed scoring downside.

Keep opt-in:
- Rotation, and ADC+rotation: real-data/anisotropic only; explicitly NOT for uniform/isotropic
  (axis-aligned) data, where rotation regresses.

Do not enable:
- The CURRENT (uncorrected) cosine ADC path — it regresses vs baseline on every dataset tested.

Next implementation work:
1. Add a metric-specific (normalized-L2) ADC correction for cosine/IP; re-benchmark (flat + native).
2. Native tier: FP32-HNSW + OSQ baselines, p99/index-size/FP32-bytes-read/cold-cache, graph-param
   sweeps (ef_construction/m/ef_search/first-pass), equal-latency operating points — the item 1/3-HNSW
   /4/7/8 axes this tier cannot measure.
3. Real 768/1536-D text & image embeddings (downloads) to confirm the SIFT-128 real-data result at
   production dimensions.
4. Decide rotation gating by data profile (detect axis-aligned/isotropic → keep rotation off).
```

Every number above is flat-scan (scoring-quality); none is a latency/production claim. Confidence
intervals / multi-seed variance (item 8) are native-tier once latency is in scope.
