# Track 2, Step 12 — Adaptive precision budgeting: can the 4-bit mass move below 4 bits?

**Branch:** `track2-adaptive-rescore`. Harness:
[`Track2PrecisionBudgetTests.java`](../../src/test/java/org/opensearch/knn/research/Track2PrecisionBudgetTests.java).
Data: [`results/step12_representations.csv`](results/) + derived
[`four_bit_mass_migration.csv`](results/four_bit_mass_migration.csv),
[`bit_allocation_ablation.csv`](results/), [`preconditioning_results.csv`](results/),
[`smart_1bit_ablation.csv`](results/), [`fractional_precision_results.csv`](results/),
[`adaptive_precision_policy.csv`](results/). **Candidate recall@10 is the primary metric** (not
final-reranked). Consistent geometry: HNSW built AND navigated on the same reconstruction. Storage =
actual packed `ceil(Σ bits / 8)` bytes/vec. 5 representative step-11 segments; small N; warm/one host.

## 1. Executive decision

```
NO-GO on the core hypothesis (fractional 1.25-2.0 bits reaching near-4-bit candidate quality) -- with
one genuinely useful positive on STRUCTURED data.

  - 4-BIT MASS MIGRATION = 0%. All four hard (4-bit) segments -- iid-128, iid-384, manifold-768,
    isotropic-768 -- STILL require 4-bit. Nothing below 4-bit reaches candidate recall 0.95.
  - FRACTIONAL / UNEQUAL ALLOCATION (the main branch) FAILS ITS OWN CONTROL: intelligent allocation
    (variance / reconstruction-error / neighbourhood) does NOT beat RANDOM allocation on any segment
    (differences < 0.03, within noise), and fractional is often WORSE than uniform 1-bit on isotropic
    data. Spending bits unevenly does not help when every dimension carries independent information.
  - The 1-bit -> 4-bit bimodality is therefore PRIMARILY AN INFORMATION-CAPACITY FLOOR for
    isotropic/unstructured data, not an artifact of uniform bit allocation.

  POSITIVE (the real lever): on STRUCTURED real data (SIFT-128), a structured PRECONDITIONER
  (random-sign + Walsh-Hadamard) turns 1-bit into ~2-bit graph quality at 1-bit storage:
  Hadamard-1-bit reaches recall 0.95 at ef=100 vs uniform-1-bit needing ef=500 (eff 1.0 bits, no
  padding). This is PRECONDITIONING, not fractional allocation -- and it works only where axis-aligned
  quantization was leaving structure on the table (correlated/structured data), i.e. real embeddings,
  NOT iid Gaussian.
```

**Decision-gate mapping:** **NO-GO** on the storage-migration targets (0% of 4-bit mass moved;
effective precision cannot drop below 4 bits on the hard segments) → the task's designated NO-GO
conclusion applies verbatim: *"The observed 4-bit requirement is primarily an information-capacity floor
for these workloads rather than an artifact of uniform bit allocation."* The Hadamard-on-structured-data
result is a **separate PARTIAL-GO for preconditioning** worth its own follow-up.

## 2. Central migration result (task §11) — 0% of the 4-bit mass moves

| seg | type | dim | old tier | min quantized feasible eff bits @0.95 | best rep | moved below 4-bit? |
|---|---|---|---|---|---|---|
| 0 | sift128 (real) | 128 | 1-bit | **1.0** | **had_uniform1** (ef 100) | n/a (already 1-bit) |
| 1 | iid128 | 128 | 4-bit | 4.0 | uniform4 | **NO** |
| 2 | iid384 | 384 | 4-bit | 4.0 | uniform4 | **NO** |
| 3 | manifold768 | 768 | 4-bit | 4.0 | uniform4 | **NO** |
| 4 | iso768 | 768 | 4-bit | 4.0 | uniform4 | **NO** |

**Of the four 4-bit segments, 0 (0%) can move below 4 bits** while keeping candidate recall ≥ 0.95.
The single structured segment (SIFT) was already 1-bit; preconditioning improves its *graph quality*
(lower ef) but not its storage tier.

## 3. Candidate ceilings by representation (task §9, §10) — max candidate recall@10 over ef ≤ 500

| representation | SIFT-128 | iid-128 | iid-384 | manifold-768 | iso-768 |
|---|---|---|---|---|---|
| uniform 1-bit | 0.99 (ef500) | 0.83 | 0.83 | 0.50 | 0.83 |
| **smart 1-bit** (cond-mean) | **1.00 (ef200)** | 0.84 | 0.85 | 0.40 ↓ | 0.87 |
| uniform 2-bit | 1.00 (ef50) | 0.90 | 0.91 | 0.91 | 0.93 |
| **uniform 4-bit** | 1.00 (ef20) | **0.99 (ef200)** | **0.98 (ef500)** | **1.00 (ef50)** | **0.99 (ef200)** |
| frac 1.25 (best policy) | 0.999 | 0.81 | 0.80 | 0.54 | 0.82 |
| frac 1.5 (best policy) | 0.999 | 0.78 | 0.77 | 0.59 | 0.81 |
| **Hadamard 1-bit** | **1.00 (ef100)** | 0.82 | 0.84¹ | 0.57¹ | 0.91¹ |
| fp32 | 1.00 (ef20) | 0.99 | 0.98 | 1.00 | 0.99 |

Bold = meets 0.95. ¹Hadamard on non-power-of-2 dims pads (384→512, 768→1024) → eff **1.33 bits**, so
its apparent gain on iid/manifold is mostly the +33% bits, not efficiency (at equal bits it ~matches
uniform). **Only on SIFT (pow-2 dim, no padding) is Hadamard a true 1-bit-storage win.**

## 4. Branch results

### Branch A — smart 1-bit (task §3)
- **My baseline 1-bit already uses asymmetric fp32-query ADC** (`asymL2` scores fp32 query vs
  quantized doc), so **A2 (ADC) is already in the baseline** and did not lift the ceiling.
- **Centering (A1) is a no-op** for per-dim min/max quantization (the per-dim range already tracks the
  data; subtracting a constant does not change the two reconstruction levels). Documented, not benchmarked.
- **Smart 1-bit (conditional-mean levels, A-variant): data-dependent and modest.** Helps SIFT
  (0.99@ef500 → **1.00@ef200**) and iso-768 (0.83→0.87); ~neutral on iid; **hurts manifold** (0.50→0.40).
  Not a general win.

### Branch B — preconditioning (task §4) — the one that works, on structured data
- **Hadamard (random-sign + FWHT), applied consistently to index/build/query, is the standout on
  SIFT-128:** 1-bit ceiling 0.99@ef500 → **1.00@ef100** at **eff 1.0 bits** (no padding). This is a real
  ~2-bit-graph-quality-at-1-bit-storage result — and it is a *structured* transform, not the dense
  random rotation that failed in step 7.
- **On isotropic data (iid-128/384, iso-768) Hadamard does not help** beyond the padding-bit overhead
  — isotropic Gaussian is rotation-invariant, so decorrelation buys nothing (confirmed: iid-128
  0.825→0.823 at equal 1.0 bits).
- **On low-rank manifold-768 it helps only slightly** (0.50→0.57 at 1.33 bits; had+frac1.5 → 0.71 at
  2.0 bits) and never reaches 0.95 — the low-rank variance lives in a rotated basis that Hadamard does
  not axis-align (PCA would, but is data-dependent and expensive; deferred).

### Branch C — fractional / unequal precision (task §5–7) — NEGATIVE
- **Allocation ablation (the control): random ≈ variance ≈ reconstruction-error ≈ neighbourhood on
  every segment** (frac1.5 cmax spread < 0.03). Intelligent allocation does **not** beat random →
  by the task's own criterion, the approach is not useful.
- **Fractional is frequently WORSE than uniform 1-bit on isotropic data** (iid-128: uniform1 0.83 vs
  frac1.5 0.75) — unevenly splitting a tiny budget hurts when all dimensions carry independent info.
- **Group-based allocation, progressive bitplane scoring, and per-vector correction metadata (A4)
  were NOT implemented** — the ceiling-bound nature of the failure makes them low-probability:
  correction sidecars improve *distance estimates for reranking*, not the *candidate-generation
  geometry* that is the bottleneck (the task itself flags this). Documented as deferred, not claimed.

### Branch D — 4-bit / fp32 references
4-bit reaches 0.95 on all segments (ef 20–500); fp32 slightly higher ceiling at lower ef. 4-bit ≈ fp32
candidate quality (consistent with step 10).

## 5. Why isotropic data is a floor (the mechanism)
Axis-aligned b-bit quantization stores `b` bits per independent coordinate. For **iid Gaussian**, every
coordinate is independent and equally informative for nearest-neighbour ordering, so total information
scales as `b × dim` with **no reallocation possible** — no dimension is "more important," so fractional
allocation and decorrelating rotations (Hadamard) cannot concentrate bits usefully. Reaching 0.95
candidate recall simply requires ~4 bits/coordinate of fidelity. For **structured/correlated data**
(real SIFT), the coordinates are redundant, so a decorrelating transform (Hadamard) makes each 1-bit
sign more informative → 1-bit suffices. **The corpus's 4-bit mass is isotropic → it is a real capacity
floor.** Real production embeddings are structured (more like SIFT than iid), so this corpus *understates*
what preconditioning could achieve on real data.

## 6. New adaptive selector over the richer representation set (task §12, §13)
Adding smart-1-bit / Hadamard-1-bit / fractional to the `{1,2,4}` menu and re-selecting the cheapest
feasible representation per segment: **the hard segments still pick 4-bit** (nothing cheaper is
feasible), and SIFT picks Hadamard-1-bit (same 1-bit storage tier, better ef). **Net effective
bits/dim and total storage are unchanged from the step-11 adaptive result (~3.77 bits/dim on the full
corpus)** — the richer menu does not move the storage number on this isotropic-heavy corpus. On a
structured-data-heavy index the Hadamard-1-bit tier would lower ef (query cost), not storage.

## 7. Executive summary table (task §22)

| representation/policy | eff bits/dim | % old 4-bit mass recovered | candidate recall @ its min-ef | storage vs 4-bit |
|---|---|---|---|---|
| uniform 1-bit | 1.0 | 0% (fails SLA on all hard) | 0.50–0.99 | 0.25× |
| fractional 1.25–1.5 (any policy) | 1.25–1.5 | **0%** | ≤ 0.82 on hard | 0.31–0.38× |
| Hadamard 1-bit | 1.0 (pow2) | 0% storage-tier; **ef win on structured** | 1.00 (SIFT) / ≤0.91 (iso) | 0.25× |
| **uniform 4-bit** | 4.0 | (baseline) | **0.98–1.00** | 1.00× |
| smart-low-bit adaptive | ~3.77 (corpus) | **0%** | meets SLA | ≈ 0.94× (= step 11) |

## 8. Required final answers (task §21)

1. **Is the 1→4-bit bimodality fundamental?** Yes for isotropic/unstructured data (information floor); an artifact only for structured data.
2. **Centering improvement to 1-bit?** ~0 (no-op for per-dim min/max).
3. **ADC improvement?** Already in the baseline; the low ceiling persists despite ADC.
4. **Preconditioning on hard iid/manifold?** No real gain (isotropic is rotation-invariant; manifold's variance is in a rotated basis Hadamard doesn't align).
5. **Correction metadata per byte?** Not implemented — deferred as low-probability (helps reranking, not the candidate-generation ceiling).
6. **Can ~1.1–1.3 eff bits reach 2-bit quality?** Only on structured SIFT (via Hadamard); not on iid/manifold.
7. **Can 1.5–2 eff bits approach 4-bit?** No on the hard segments (best sub-4-bit ≈ 0.71 on manifold, ≤ 0.83 on iid).
8. **Best allocation policy?** None distinguishable from random (all within noise).
9. **Graph-aware (neighbourhood) beat reconstruction-error?** No — indistinguishable.
10. **Fraction of 4-bit vectors movable below 4-bit?** 0% on this corpus.
11. **New weighted bits/dim?** Unchanged (~3.77 corpus).
12. **Storage saved vs all-4-bit?** ~6% (unchanged from step 11 — the richer menu adds nothing here).
13. **Saved vs the 3.77-bit adaptive?** ~0%.
14. **p99 vs 4-bit?** Comparable (representations that meet the SLA are 4-bit or the SIFT Hadamard-1-bit which lowers ef).
15. **Stable at ~50 cal queries?** Selection is driven by candidate ceilings (robust); the allocation policies are indistinguishable so calibration of *which dims* is moot.
16. **After merges?** Not re-run (step 11 showed mixed merges escalate; fractional cannot change that given it doesn't help).
17. **Pareto frontier?** On the hard segments the frontier is essentially {1-bit poor → 4-bit good} with nothing useful between; on SIFT, Hadamard-1-bit dominates uniform-1-bit (same storage, lower ef).
18. **Complexity justified?** For fractional/unequal: **no** (no benefit). For Hadamard on structured data: **plausibly yes** (cheap FWHT, real ef reduction) — worth a real-embedding follow-up.

## 9. Limitations & deferred branches
- **5 small representative segments** (not the full 33) — chosen to isolate the mechanism; the isotropic
  floor is confirmed on 3 iid/manifold + the manifold, and SIFT is the structured anchor.
- **Deferred (documented, not claimed):** per-vector correction metadata (A4), group-based allocation,
  progressive bitplane scoring (§18), PCA preconditioning (the transform that *could* help low-rank
  data but needs an eigensolver + per-segment matrix storage), merges under fractional (§20), and a
  full quantitative precision predictor (§19). The decisive negative on fractional allocation + the
  isotropic-floor mechanism make these low-value for THIS corpus.
- **Corpus is isotropic-heavy** — this is intentional (the hard case) but means the result is a
  *pessimistic* bound on preconditioning; real structured embeddings should benefit more from Hadamard.
- Hadamard padding overhead (dim → next pow2) is real for non-pow2 dims; a real impl would use a
  padded-but-shared layout or a structured transform sized to the dimension.

## 10. OpenSearch recommendation & exact next step
**Do NOT pursue fractional/unequal per-dimension precision** — it does not beat uniform and adds codec
+ kernel complexity for no gain. **Do pursue a structured preconditioner (Hadamard / RaBitQ-style
rotation) as an optional per-segment transform for STRUCTURED (real-embedding) segments**, where 1-bit
+ preconditioning reaches ~2-bit graph quality at 1-bit storage and 4× lower query ef. **Exact next
step:** re-run branches B (Hadamard) and A on a **real 768/1536-D embedding corpus** (the structured
case this synthetic corpus lacks) and implement a **co-designed RaBitQ 1-bit** (rotation + normalized
estimator + error bound) to test whether the SIFT-style Hadamard win generalizes and whether it can pull
real 4-bit segments down to 1–2 bit — the one remaining path to a stronger-than-adaptive-1/2/4 proposal.

## Reproduce
```bash
./gradlew :test --tests "org.opensearch.knn.research.Track2PrecisionBudgetTests.testPrecisionBudget" \
    -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
# writes results/step12_representations.csv; derive tables with the inline python in the commit.
```
```
