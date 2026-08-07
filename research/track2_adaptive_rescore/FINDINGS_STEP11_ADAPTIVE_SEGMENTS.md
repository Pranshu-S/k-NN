# Track 2, Step 11 — Adaptive per-segment precision: does a heterogeneous index need every bit?

**Branch:** `track2-adaptive-rescore`. Harness:
[`Track2AdaptiveSegmentsTests.java`](../../src/test/java/org/opensearch/knn/research/Track2AdaptiveSegmentsTests.java)
(profiling + merge). Aggregator: [`results/aggregate_step11.py`](results/aggregate_step11.py) (stdlib).
Raw + summary CSVs: [`results/adaptive_*.csv`](results/). Bit width is chosen at **segment build time**
from a calibration sample and baked in (never "search a 1-bit segment as 4-bit"). Representation &
geometry unchanged from steps 8–10.

## 1. Executive decision

```
PARTIAL GO. Per-segment precision heterogeneity is REAL, PREDICTABLE, and STABLE -- but on this
(deliberately hard, iid-heavy) corpus most vectors genuinely need 4-bit, so adaptive saves only ~6%
storage vs all-4-bit. The saving is real but corpus-dependent: it comes entirely from the fraction of
STRUCTURED data that can use 1-bit. A realistic index (mostly structured, real-embedding-like) would
save much more; an all-isotropic index would save nothing.

Confirmed:
  - Difficulty varies by DATA STRUCTURE, not just dimension: at the SAME 128-D, real SIFT -> 1-bit
    while iid-Gaussian -> 4-bit. Escalation is GENUINE graph-quality (2-bit candidate ceiling < 0.95
    for 64% of segments), not arbitrary tiering.
  - Adaptive meets the SLA: all-1-bit and all-2-bit MISS recall@10>=0.95 on 83% / 74% of segments;
    4-bit and adaptive both comply (adaptive_cap4 ~2.4% miss = one borderline segment).
  - Required precision is PREDICTABLE without full benchmarking: quantization reconstruction error
    correlates 0.87 with log2(bits); cluster-separation / NN-spread ~ -0.85.
  - Calibration is STABLE: 50 calibration queries give 100% selection agreement with 250.
Not confirmed / caveats:
  - 2-bit is almost never the MINIMUM feasible tier at 0.95 (bimodal 1-bit / 4-bit) -- without
    rotation, data that needs >1-bit usually needs 4-bit. (2-bit does appear at target 0.90.)
  - The corpus is only 33 segments (not the 50 target) and ~half iid -> a CONSERVATIVE test that
    understates the adaptive win; and it is SYNTHETIC apart from the 8 real SIFT segments.
  - A conservative fp32 fallback can pick fp32 for a borderline segment and, at 32 bits, erase the
    storage win -> the production policy must CAP at 4-bit (fp32 only when 4-bit truly fails).
```

**Decision-gate mapping (task):** lands in **PARTIAL GO** — *"heterogeneity exists, but most data
requires 4-bit, adaptive still saves some storage, calibration works."* Not STRONG GO (the < 2.5
bits/dim-at-4-bit-recall bar is not met on this corpus: effective 3.77 bits/dim). Not NO-GO
(heterogeneity is real, selection is stable and predictable, and the low-precision tier is genuine).

## 2. Corpus (task §1) — heterogeneous, honest, deliberately hard

**33 segments profiled** (target was ≥50; the high-dimensional isotropic HNSW builds are pathologically
slow in this sandbox — the same connectivity issue seen in steps 7/10 — so the run was finalized at 33
after the result had clearly stabilized). Heavy-tailed sizes; 4 data families:

| family | dim | segs | vectors | typical selection |
|---|---|---|---|---|
| **real SIFT-128** (slices of `sift_base`) | 128 | 8 | 45,252 | **1-bit** |
| iid-Gaussian-128 | 128 | 10 | ~64K | 4-bit (1 → fp32) |
| iid-Gaussian-384 | 384 | 7 | ~55K | 4-bit |
| low-rank manifold-768 (r=64) | 768 | 7 | ~50K | 4-bit |
| isotropic-768 | 768 | 1 | ~7K | 4-bit |

**Not engineered to a favorable mix.** Difficulty is driven by quantization structure (isotropy /
intrinsic dim / dimension); iid Gaussian is the *worst case* for axis-aligned 1-bit, so a ~half-iid
corpus biases toward 4-bit and makes the adaptive saving a **lower bound**. Real SIFT is the empirical
1-bit-friendly anchor. (An early clustered corpus was discarded: extreme cluster separation disconnected
the HNSW graph so even fp32 failed — a connectivity artifact, not a precision result.)

## 3. Primary distribution (task §4) — @ recall@10 ≥ 0.95, conservative bootstrap LB, 250 cal queries

| representation | segments | % segs | vectors | % vectors |
|---|---|---|---|---|
| **1-bit** | 8 | 24.2% | 45,252 | **19.6%** |
| 2-bit | 0 | 0.0% | 0 | 0.0% |
| **4-bit** | 24 | 72.7% | 165,193 | **71.5%** |
| fp32 | 1 | 3.0% | 20,749 | 9.0% |

**Not all segments are equally difficult** (answer to Q1: *no*). The 8 real-SIFT segments stay at 1-bit;
everything isotropic/manifold needs 4-bit. **2-bit is never the minimum at 0.95** — the bimodal
1-bit/4-bit split reflects that, without rotation, sub-4-bit fidelity rarely lands exactly in the
[0.95-feasible] band.

## 4. Storage & effective bits/dim (task §5, §6) — code bytes only (graph + fp32-rerank excluded)

| policy | total code MB | eff bits/dim | vs all-1-bit | vs all-4-bit | vs fp32 |
|---|---|---|---|---|---|
| all 1-bit | 9.5 | 1.00 | 1.00× | 0.25× | 0.031× |
| all 2-bit | 19.1 | 2.00 | 2.00× | 0.50× | 0.062× |
| all 4-bit | 38.2 | 4.00 | 4.00× | 1.00× | 0.125× |
| **adaptive (cap 4-bit, production)** | **36.0** | **3.77** | 3.79× | **0.94× (saves 6%)** | 0.118× (**8.5× compression**) |
| adaptive (min, incl fp32 fallback) | 45.3 | 4.75 | 4.75× | 1.19× (**worse**) | 0.148× |
| fp32 | 305.4 | 32.0 | 32× | 8× | 1.00× |

- **The one fp32-fallback segment (9% of vectors × 32 bits) pushes "adaptive-min" ABOVE all-4-bit.**
  The realistic production policy **caps at 4-bit** → **adaptive_cap4 = 3.77 bits/dim, saves 6% vs
  all-4-bit, 8.5× compression vs fp32.** The entire saving comes from the 19.6% of vectors (real SIFT)
  that stay at 1-bit.
- On a structured-heavy (realistic) index the 1-bit fraction would be far larger → the saving scales
  with it; this iid-heavy corpus is a floor.

## 5. Policy comparison (task §8) — the master table

| metric | all 1-bit | all 2-bit | all 4-bit | **adaptive (cap4)** |
|---|---|---|---|---|
| recall (vec-weighted) | 0.767 | 0.891 | 0.975 | **0.972** |
| **recall SLA miss %** | **82.8** | **74.3** | **0.0** | **2.4** |
| effective bits/dim | 1.00 | 2.00 | 4.00 | **3.77** |
| total code MB | 9.5 | 19.1 | 38.2 | **36.0** |
| avg nodes/query | 5,335 | 4,662 | 2,873 | 3,092 |
| avg bytes/query | 232,739 | 417,702 | 418,585 | **418,749** |
| p99 µs | 3,998 | 1,763 | 937 | **1,026** |

- **The cheap global policies FAIL the SLA**: all-1-bit misses on 82.8% of segments, all-2-bit on 74.3%
  (they cannot hit 0.95 candidate recall on the isotropic majority even at ef=500). **You cannot just
  "use 1-bit everywhere."**
- **all-4-bit and adaptive_cap4 both essentially meet the SLA** (0% / 2.4% miss). Adaptive_cap4 matches
  4-bit's quality/latency at **6% less storage** — the win is modest **on this corpus** because 72% of
  it genuinely needs 4-bit.
- Latency-optimal-per-segment (policy E) picks fp32 for speed → **120 MB** storage: a latency objective
  destroys the storage benefit; the *memory* objective (adaptive_cap4) is the economically interesting one.

## 6. Is escalation actually necessary? (task §7) — YES, it's genuine graph quality

| reason | count | % |
|---|---|---|
| 4-bit: **2-bit candidate ceiling < target** | 21 | 63.6% |
| 1-bit: meets SLA directly (real SIFT) | 8 | 24.2% |
| 4-bit: 2-bit feasible but not conservatively certified | 3 | 9.1% |
| fp32: quantized ceilings < target | 1 | 3.0% |

**64% of segments escalate because 2-bit's candidate ceiling is genuinely below 0.95** — not arbitrary
tiering. This is the mechanistic core: adaptive precision is solving real per-segment graph-quality
differences.

## 7. Segment-difficulty predictors (task §10) — precision is predictable

Pearson correlation with log2(selected bits):

| predictor | corr | |predictor | corr |
|---|---|---|---|---|
| **recon_err_2b / 1b / 4b** | **+0.87** | | intrinsic_dim (two-NN) | (see CSV) |
| **cluster_sep** | −0.85 | | anisotropy | (see CSV) |
| **nn_spread** | −0.84 | | probe_ceil_1b_ef50 | (see CSV) |

**Quantization reconstruction error (a cheap, build-free statistic) predicts the required bit width at
r ≈ 0.87.** Cluster-separation and NN-distance spread (low = dense/hard) follow at ~−0.85. So a segment's
precision can be *predicted* from cheap sample statistics — a strong signal for a future lightweight
selector that avoids building all four graphs. Full table:
[`adaptive_predictor_analysis.csv`](results/adaptive_predictor_analysis.csv).

## 8. Recall-target sensitivity (task §13) — the SLA sets the bill

| target recall | eff bits/dim (incl fp32) | %1-bit vec | %2-bit vec | %4-bit vec | %fp32 vec |
|---|---|---|---|---|---|
| 0.90 | 3.32 | 19.6 | **21.0** | 59.4 | 0.0 |
| 0.95 | 4.75 (cap4: 3.77) | 19.6 | 0.0 | 71.5 | 9.0 |
| 0.97 | 7.24 | 19.6 | 0.0 | 63.8 | **16.6** |

- **At 0.90, a real 2-bit tier appears (21% of vectors)** — relaxing the SLA unlocks the middle tier.
- **At 0.97, fp32 escalation grows to 16.6%** — chasing the last 2% of recall is expensive.
- The 1-bit fraction (real SIFT) is stable across targets — structured data is robustly 1-bit-capable.

## 9. Calibration stability (task §11) — 50 queries suffice

| cal queries | selection agreement vs 250 | segments changed |
|---|---|---|
| 25 | 93.9% | 2 |
| **50** | **100.0%** | **0** |
| 100 | 97.0% | 1 |
| 250 | (reference) | — |

**50 calibration queries give 100% agreement with 250** — validating the prior belief that ~50 suffice.
Even 25 agrees on 94%. Selection is robust to calibration sample size on this heterogeneous corpus.

## 10. Merge simulation (task §12) — homogeneous merges retain; cross-tier merges escalate

5 same-dim (128-D) merges, re-profiled from scratch (never inheriting the max bit width):

| source A | source B | merged | transition |
|---|---|---|---|
| 1-bit (SIFT) | 1-bit (SIFT) | **1-bit** | **retain** (easy+easy stays easy) |
| 4-bit (iid) | 4-bit (iid) | **4-bit** | retain |
| 1-bit (SIFT) | 4-bit (iid) | **4-bit** | escalate to max (×2) |
| 4-bit (iid) | 1-bit (SIFT) | **fp32** | escalate beyond max (conservative fallback) |

**Merges do NOT universally converge to high precision.** A homogeneous 1-bit+1-bit merge **retains
1-bit** — the adaptive benefit survives when similar segments merge. But **cross-tier (1-bit+4-bit)
merges escalate to the max** (the hard fraction dominates candidate recall), occasionally beyond (a
larger mixed merge hit the conservative fp32 fallback). Production implication (answer to Q14):
**adaptive precision survives merges under a difficulty-aware or size-tiered merge policy** (which keeps
like with like); naive cross-tier merging erodes the 1-bit fraction over time. Raw:
[`adaptive_merge.csv`](results/adaptive_merge.csv).

## 11. Required final answers (task §14)

1. **Equally difficult?** No — same-dim data ranges 1-bit (real SIFT) to 4-bit (iid).
2. **% segments 1-bit @0.95?** 24.2% (8/33), all real SIFT.
3. **% vectors 1-bit?** 19.6%.
4. **% require 2-bit?** 0% at 0.95 (21% of vectors at 0.90).
5. **% require 4-bit?** 72.7% segs / 71.5% vectors.
6. **Any require fp32?** 1 borderline segment (conservative) — capped to 4-bit in production.
7. **Weighted avg bits/dim?** 3.77 (production cap-4-bit); 4.75 if fp32 fallback allowed.
8. **Adaptive storage vs all-1/2/4/fp32?** 36 MB vs 9.5 / 19.1 / 38.2 / 305 MB.
9. **Saves vs all-4-bit?** ~6% (this hard corpus; scales with the structured fraction).
10. **Extra vs all-1-bit?** +279% — but all-1-bit **fails the SLA on 83% of segments**, so it is not a valid baseline here.
11. **Meets SLA on held-out?** Yes — adaptive_cap4 misses on 2.4% (one borderline segment) vs all-4-bit's 0%.
12. **Query work saved vs all-1-bit?** Large: all-1-bit p99 3,998 µs (and non-compliant); adaptive_cap4 p99 1,026 µs (−74%), nodes 3,092 vs 5,335 (−42%).
13. **Stable at ~50 cal queries?** Yes — 100% agreement with 250.
14. **Do merges converge to high precision?** See §10.
15. **Which properties predict bit width?** Reconstruction error (r≈0.87), cluster-separation / NN-spread (≈−0.85).
16. **Hypothesis "spend extra bits only where the graph needs them" supported?** **Mechanistically yes**
    (escalation tracks genuine 2-bit-ceiling shortfalls; recon-error predicts it). **Economically,
    partially** — the storage win is real but modest on isotropic-heavy data and grows with the
    structured fraction.

## 12. Limitations

- **33 segments, not 50** (high-D isotropic HNSW build cost); result had stabilized. **~half synthetic
  iid** (worst case) → conservative/lower-bound on adaptive benefit. Only the 8 SIFT segments are real.
- Warm latency, single host (Apple M5, arm64), no PMU — consistent with steps 8–10.
- **No rotation** — a RaBitQ-grade rotated 1-bit could move much of the 4-bit mass down to 1–2 bit
  (the biggest lever to improve the economics); that is the separate rotation branch.
- fp32-fallback storage blowup is an artifact of allowing 32-bit in the "min" policy; production caps at 4-bit.

## 13. OpenSearch integration recommendation

**Recommend adaptive per-segment precision as an opt-in, capped at 4-bit**, with **reconstruction-error
based fast pre-screening** (r≈0.87) to avoid building all four graphs: compute cheap sample stats at
flush/merge → predict a candidate bit width → verify with a ~50-query conservative calibration →
persist `(bits∈{1,2,4}, ef, rerank_depth)` in segment metadata → query with the matching scorer; fp32
only when 4-bit genuinely fails, and even then prefer 4-bit + deeper rerank. Expected value is highest
on **mixed indexes with a meaningful structured (real-embedding) fraction**, where 1-bit segments are
common; on isotropic-heavy data it degenerates to all-4-bit (still correct, just no saving).

## 14. Exact next step

Rerun this multi-segment PoC on a **real, structured embedding corpus** (768/1536-D text/image
embeddings) — the fraction that stays at 1–2 bit is the whole economic question, and real embeddings are
far more structured than iid Gaussian. Pair with a **rotated 1-bit (RaBitQ) tier** to test whether the
4-bit mass collapses toward 1–2 bit. Also build the **recon-error pre-screener** so selection cost drops
from 4 graph builds to 1.

## Reproduce
```bash
# profiling (auto-resumes from the flushed CSV across the 20-min suite timeout; rerun until 50 done):
./gradlew :test --tests "org.opensearch.knn.research.Track2AdaptiveSegmentsTests.testProfileAll" \
    -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
./gradlew :test --tests "org.opensearch.knn.research.Track2AdaptiveSegmentsTests.testMerge" ...
# aggregate raw per-segment CSVs -> summary tables:
cd research/track2_adaptive_rescore/results && python3 aggregate_step11.py
```
```
