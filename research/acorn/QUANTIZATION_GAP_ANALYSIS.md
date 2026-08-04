# Quantization + Rescore in OpenSearch k-NN — Gap Analysis vs. the Aug-2026 Frontier

**Scope.** Evaluates the k-NN plugin's *quantization* and *rescore* strategy against the
state of the art as of Aug 2026, identifies where we lag, recommends a prioritized program,
and estimates the customer-experience (CX) impact of each fix.

**Evidence tiers used throughout** (so estimates are honest):
- `[code]` — verified in this repo by reading the implementation (file:line cited).
- `[pub]` — published result from a vendor blog or paper (cited).
- `[proj]` — *our projection*, derived from `[code]` + `[pub]` with stated assumptions.
  Projections are ranges, not measured numbers for this plugin.

> We did **not** re-benchmark the plugin's quantizer here (that's the recommended next step,
> §7). Every plugin-specific number below is either `[code]` (a fact about the implementation)
> or `[proj]` (an estimate). Frontier numbers are `[pub]`.

---

## 1. Executive summary

The plugin's **architecture is right** — per-segment training, two-phase quantized→fp32
rescore, off-heap graph, and (crucially) a genuinely modern **Optimized Scalar Quantization
(OSQ)** implementation already vendored in-tree via the Lucene 10.4 codec. But the **default
path and the flagship `on_disk` path are ~two generations behind** what sits in the same repo:

1. The shipped x32 default is **plain per-dimension-mean binarization**, with **random rotation
   OFF and ADC OFF by default** `[code]`, and recall recovered almost entirely by **fp32
   rescoring**. This is a 2023-era code compensated by I/O.
2. Rescoring is **uncalibrated** — oversample is keyed on *dimension* (a proxy for error),
   the compression-level table is **partly unreachable due to an override bug**, and at
   **dim ≥ 1000 the default oversample is 1.0× (none)** `[code]`. The frontier (Elastic OSQ
   auto-calibration) *measures* the error and *solves* for the minimal rerank depth.
3. There is **no unbiased distance estimator / error bound** in the QFrame path — the thing
   that makes RaBitQ/OSQ able to skip or right-size rescoring.

**Highest-leverage moves:** (P0) route the default `on_disk` path through the OSQ scorer we
already ship and flip rotation/ADC on; (P1) replace dimension-keyed oversampling with
**error-model-calibrated, per-query-adaptive rescoring** — which is the *same* "monitor your own
uncertainty and spend only where needed" principle as the self-aware ANN routing in
[`FILTERED_SEARCH_CHEATSHEET.md`](FILTERED_SEARCH_CHEATSHEET.md) §5.

---

## 2. What the plugin actually does today (code-verified inventory)

Three parallel quantization stacks coexist:

| stack | where | mechanism | default? |
|---|---|---|---|
| **QFrame binary** | `index/engine/faiss/QFrameBitEncoder.java` | 1/2/4-bit scalar; threshold = per-dim `mean + iCoef·σ`; Welford stats on 25K sample/segment | the main `on_disk` x32/x16/x8 path |
| **Lucene OSQ bridge** | `codec/…/KNN1040*`, faiss `sq bits=1` (≥3.6.0) | Lucene 10.4 `OptimizedScalarQuantizer` (RaBitQ-derived): 1-bit doc × 4-bit query asymmetric + per-vector corrections; AVX-512 | only for `sq bits=1` |
| **Legacy faiss** | `FaissSQEncoder` (fp16, x2), PQ (≤x64) | mature; not the focus | opt-in |

**Binary threshold formula** `[code]` (`quantization/quantizer/QuantizerHelper.java:103-115`):
```
coef = bits+1 ; iCoef = -1 + 2*(b+1)/coef      # b = 0..bits-1
threshold[b][d] = mean[d] + iCoef * stdDev[d]
```
- 1-bit ⇒ `iCoef = 0` ⇒ **threshold = per-dimension mean** (no percentile, no rotation-aware calibration).
- 2-bit ⇒ `mean ± σ/3` (note: class javadoc says `±1σ` — **doc contradicts code**, `MultiBitScalarQuantizer.java:52`).
- 4-bit ⇒ `mean + {−0.6,−0.2,+0.2,+0.6}·σ` — **uniform**, not energy-allocated.

**Random rotation** `[code]`: `RandomGaussianRotation.java` — dense `d×d` Gaussian +
Gram-Schmidt, **fixed seed `1212121212`** (same matrix for every index/segment of a given `d`),
**O(d²) apply per vector**, **O(d²) bytes serialized per segment**. **Default OFF**
(`QFrameBitEncoder.java:38-39`).

**ADC (asymmetric distance)** `[code]`: query kept full-precision vs 1-bit docs. Java transform
`q ← (q−x)/(y−x)` with `x,y` = below/above-threshold means; L2 correction `(y−x)²·(q−0.5)+0.5`.
Native side: batched 8-bit LUT with `distances_batch_4` (`jni/include/faiss_index_bq.h`).
**1-bit only** (2/4-bit rejected). **Default OFF** (`QFrameBitEncoder.java:36-37`).

**Rescore** `[code]` (`NativeEngineKnnVectorQuery.java`, `RescoreContext.java`,
`ExactSearcher.java`):
- Two-phase: quantized first pass → **global** cross-segment `reduceToTopK(firstPassK)` →
  **fp32 exact** re-scoring, fp32 read from the mmap'd Lucene `.vec` (not a separate copy).
- Oversample default `getFirstPassK(k, dim)`: **dim<768 → 3×, 768–1000 → 2×, ≥1000 → 1.0×**,
  then `firstPassK = clamp(ceil(k·factor), 100, 10000)`.
- **Override wart:** this dimension table **overwrites** the `CompressionLevel` table (3×/5×)
  whenever the user didn't set `oversample_factor` and `allowOverrideOversampleFactor==true`
  — so the compression-level defaults are largely unreachable (`#3460` fixed one 32x instance;
  structural issue remains).

**Training/state** `[code]`: per-segment at flush/merge; 25K reservoir sample; state cached
(Guava, 5% heap, 60-min expiry); fetched at query time via a fake `searchNearestVectors` call.
**No recalibration, no drift handling, single global rotation seed.**

---

## 3. The frontier (Aug 2026)

| technique | year | what it adds |
|---|---|---|
| **RaBitQ** (SIGMOD'24) | 2024 | binary codes of randomly-rotated normalized vectors + **unbiased distance estimator with provable error bound** + per-vector norm correction |
| **Extended RaBitQ** (SIGMOD'25) | 2025 | arbitrary bit widths; **4/5/7-bit ⇒ ~90/95/99% recall *without reranking*** `[pub]` |
| **Elastic OSQ / BBQ** (8.16→8.18) | 2024-25 | per-vector optimized intervals minimizing anisotropic loss; "Robust OSQ"; 1-bit doc × 4-bit query asymmetric |
| **ES auto-calibration** | 2025 | intrinsic-dim distance model + Gaussian error model from tiny samples → **closed-form recall@k at rerank depth n (R²>0.98)** → auto-pick bits + rerank depth `[pub]` |
| **SAQ** (arXiv 2509.12086) | 2025 | PCA segmentation + **energy-based bit allocation** + coordinate-descent code adjustment |
| **TurboQuant** (ICLR'26) | 2026 | data-oblivious near-optimal distortion at all bit widths (random rotation + optimal per-coord quantizers) |
| **LVQ / Turbo-LVQ** (Intel SVS) | 2024-25 | per-vector locally-adaptive scaling; merge/stream-stable |
| **Adaptive re-ranking** (arXiv 2606.25249) | 2026 | per-query routing of rerank effort dominates any fixed policy; ~40% of queries gain nothing from rerank `[pub]` |

The unifying theme: **calibration, not just compression.** Modern quantizers ship a *statistical
model of their own error*, which is what lets them (a) choose bit width and rerank depth
analytically and (b) skip rerank when the margin is safe.

---

## 4. Gap analysis (ranked by severity)

**G1 — The default path is a 2023 code with 2026 compensation.** `[code]` Rotation + ADC — both
already implemented — are OFF. RaBitQ/OSQ show rotation is *not optional* (it isotropizes
per-dimension statistics) and asymmetric estimation is near-free accuracy. We instead pay fp32
rescore I/O to forgive a weak code.

**G2 — No unbiased estimator / error bound ⇒ blind rescoring.** The real gift of RaBitQ/OSQ is
knowing each estimate's *variance*, enabling analytic rerank-depth choice and rerank-skipping.
Our first-pass scores are biased/uncalibrated, so oversampling must be a superstition constant.
**This blocks G3.**

**G3 — Oversampling keyed on the wrong variable, and partly dead.** `[code]` Dimension is a
proxy; ES *measures* error and solves for depth. Our table `{<768:3×, <1000:2×, ≥1000:1×}` is
inverted in spirit (high-D is where the un-rotated code is *weakest*), and the compression-level
table is unreachable via the override. **Concrete footgun:** at **dim ≥ 1000, default oversample
= 1.0×**; for `k=10` the `MIN_FIRST_PASS_RESULTS=100` floor hides it, but for **k ≥ 100 you get
zero oversampling on the weakest 1-bit code** `[proj: recall regression here]`.

**G4 — No per-vector corrections in QFrame.** OSQ/LVQ/RaBitQ store 2–4 scalars/vector
(norms/intervals/component sums) that sharply tighten 1-bit estimates. QFrame stores only global
per-dim thresholds. The plugin *has* the right code (Lucene OSQ) but only routes
`faiss sq bits=1` to it — the flagship `on_disk`/QFrame path doesn't use it.

**G5 — Multi-bit (2/4-bit) is uniform and asymmetric-less.** `[code]` Extended-RaBitQ 4-bit
reaches ~99% recall with no rerank `[pub]`; our 4-bit is uniform `mean±{0.2,0.6}σ` + symmetric
distance. No SAQ-style energy allocation. The x8/x16 tiers lag most.

**G6 — No rescore cascade; int8 tier dormant.** `[code]` We jump 1-bit → fp32 random reads. The
frontier cascades 1-bit → int8/residual → fp32-for-a-handful (or skips). **SQ8/int8 exists in
vendored Faiss but is not exposed** through the mapping API — the middle tier is unused.

**G7 — Rotation implementation is dated.** `[code]` Dense O(d²) matrix (apply + serialize) vs.
fast structured rotations (FHT/Kac-walk, O(d log d), seed-regenerable, zero stored state). At
3072-D: ~9.4M mul-adds/vector + ~37 MB state/segment vs. ~34K ops + 0 state. **This is likely
*why* rotation is off by default** — the cost was made prohibitive.

**G8 — Static forever.** `[code]` No drift detection, no recalibration on merge beyond
re-training thresholds, single global rotation seed. LVQ/TurboQuant reduce drift sensitivity.

---

## 5. Recommendations (prioritized program)

| phase | change | research risk | effort |
|---|---|---|---|
| **P0** | Route default `on_disk` x32 through the in-tree **OSQ scorer** (1-bit doc × 4-bit query asymmetric + per-vector corrections), or at minimum flip `random_rotation`+`enable_adc` ON for QFrame. Fix the oversample-override bug; fix 2-bit doc/code contradiction. | **none** (wiring existing code) | weeks |
| **P1** | **Calibrated rescoring**: at train time measure quantized-vs-exact error on the 25K sample (we already hold both), fit the Gaussian error model, solve `firstPassK(k, target_recall)` in closed form. Then **per-query adaptive**: skip/shrink rescore when the k-vs-(k+1) margin exceeds the error bound. | low–med (ES has shipped the static half) | 1–2 mo |
| **P2** | Extended-RaBitQ / SAQ-class **multi-bit codes** for x8/x16 (asymmetric at all widths, PCA energy allocation); **expose int8 as a cascade tier** (1-bit scan → int8 rerank → fp32 for ≤2k). | medium | 2–4 mo |
| **P3** | **FHT rotation** (O(d log d), seed-derived, no stored matrix, per-index seeds); merge-stable per-vector scaling (LVQ); align CAGRA/GPU with GPU IVF-RaBitQ. | medium | 3–6 mo |

P1 is the flagship research contribution and **unifies with self-aware ANN routing**: one
per-query planner, two uncertainty-gated knobs — *traversal fallback* (§5 of the cheat sheet)
and *rescore depth*. Both say "measure your own error, spend only where it buys recall."

---

## 6. CX (customer-experience) impact & ROI estimates

> **How to read this.** Each item lists the CX axis, the mechanism, and a **projected** delta
> with its assumptions and confidence. Projections combine `[code]` facts about the current
> path with `[pub]` frontier results; they are **estimates for planning, not measured plugin
> numbers**. Baseline assumed: `on_disk` x32, 1-bit, rotation/ADC off, dim-keyed oversample —
> the shipping default.

### 6.1 Search quality (recall@k) — the headline CX metric

- **P0 (OSQ + rotation/ADC on the default path).**
  Mechanism: isotropized code + asymmetric estimation + per-vector corrections raise *first-pass*
  ranking quality, so the same final recall needs far less rescue.
  `[proj]` At 1536-D, x32: recall@10 **~0.90→~0.97 at equal rescore depth**, *or* hold ~0.95 at
  **2×→1× oversample** (trade quality for latency). Basis: ES reports OSQ/BBQ holds high recall
  at 1-bit where plain binary needs heavy rerank `[pub]`; our baseline leans on rescore to reach
  parity `[code]`. **Confidence: medium-high** (routing proven code).
- **P0 also removes a real regression**: the **dim ≥ 1000, k ≥ 100, oversample=1.0×** corner
  `[code]`. `[proj]` recall@100 there today is materially below target (no oversampling on the
  weakest code); calibrated/OSQ path lifts it back to target. **Confidence: high the bug exists;
  medium on magnitude.**
- **P2 (multi-bit).** `[proj]` x8 (4-bit) could hit **~0.98–0.99 recall with *no* fp32 rescore**
  (Extended-RaBitQ 4-bit ≈99% `[pub]`), vs today's 4-bit-uniform+rescore. Turns x8 into a
  "quality tier that needs no I/O." **Confidence: medium.**

**CX translation:** fewer "why did my filtered/vector search miss the obvious result" tickets;
higher answer quality for RAG/semantic search at the *same* cost; and elimination of a silent
high-k/high-dim recall cliff that today only shows up in customer A/Bs.

### 6.2 Latency (p50 / p99)

- **P0/P1 (less and smarter rescore).** Rescore reads fp32 from mmap'd `.vec` `[code]`; on
  **cold page cache** at 1536-D that's ~6 KB × firstPassK × segments of random I/O — a p99
  driver. `[proj]`
  - Calibrated depth (P1) typically **cuts firstPassK 30–60%** at fixed recall (ES auto-cal
    picks minimal depth `[pub]`) → proportional **rescore-I/O and p99 reduction**.
  - Per-query rerank-skipping (P1): ~40% of queries need no rerank `[pub]` → for those, **rescore
    latency → ~0**; **p50 down, tail smoothed**. **Confidence: medium.**
- **P2 (cascade + int8).** `[proj]` replacing 100+ fp32 random reads with an int8 in-RAM rerank
  and fp32 for only the final ≤k **cuts rescore memory traffic ~4–10×** on cold cache; biggest
  win on large, memory-pressured indices. **Confidence: medium.**
- **P0 caveat (honest):** enabling **dense O(d²) rotation** at query+index time is *itself* a
  cost `[code]` — at 3072-D it is not free. This is why **P3 (FHT rotation)** matters: it makes
  rotation ~O(d log d) so P0's quality win doesn't cost latency. Recommend P0-via-OSQ (whose
  transform is already optimized) rather than P0-via-enabling-the-dense-QFrame-rotation.

**CX translation:** lower and *more predictable* tail latency (the p99 is what customers feel),
especially for large indices that don't fully fit in page cache — exactly the scale where they
adopted quantization in the first place.

### 6.3 Cost / RAM

- Quantization already delivers the 4–32× RAM win; these changes **protect it** rather than
  extend it. `[proj]`
- **P2 int8 cascade** lets more customers stay at **higher compression (x16/x32) without a recall
  penalty**, because the middle tier recovers quality cheaply — i.e., they buy the RAM saving
  *without* falling back to x8/x4. Estimated effect: **a tier-shift of ~1 step** for
  recall-sensitive workloads (x8→x16 at equal recall). **Confidence: low-medium** (needs the P2
  benchmark).
- **P3 FHT rotation** removes the **O(d²) per-segment rotation-matrix state** `[code]` — at
  3072-D ~37 MB/segment of on-disk + heap state disappears (seed-derived). Small vs vectors, but
  real on many-segment indices. **Confidence: high (deterministic).**

**CX translation:** same or better recall at a *higher* compression tier = fewer nodes / smaller
instances for the same quality — the metric that shows up on the customer's bill.

### 6.4 Operability / "it just works"

- **P1 auto-calibration** removes the **`oversample_factor` tuning burden** and the current
  **dimension-keyed + override footguns** `[code]`. Customers stop hand-tuning a superstition
  constant; the system targets a recall SLA directly (ES's model-based selection `[pub]`).
  **Confidence: high (design-level).**
- Fixing the **override bug + 2-bit doc/code mismatch** `[code]` removes two "documented behavior
  ≠ actual behavior" traps that generate support load.

**CX translation:** fewer knobs, fewer surprises, a **recall-target API instead of a depth
guess** — the single biggest reduction in "vector search is hard to operate" friction.

### 6.5 ROI summary

| phase | primary CX win | projected magnitude | confidence | cost |
|---|---|---|---|---|
| **P0** | recall@k at equal cost; kills high-k/high-dim cliff | +5–8 pts recall @1536-D x32 *or* 2×→1× rescore | med-high | weeks, ~0 research risk |
| **P1** | p99 latency + zero-tuning recall-SLA | −30–60% rescore depth; rerank-skip ~40% queries | medium | 1–2 mo |
| **P2** | higher compression at equal recall; less I/O | x8 no-rescore ~99%; ~4–10× less rescore traffic | medium | 2–4 mo |
| **P3** | makes P0 latency-free; less segment state | rotation O(d²)→O(d log d); −tens MB/seg | med-high | 3–6 mo |

**Net CX thesis:** P0 is a *weeks-long, near-zero-risk* change that plausibly closes most of the
recall/latency gap to Elasticsearch at x32 **using code already in the repo**. P1 is the
*differentiating* research win — error-bound-driven, per-query-adaptive rescoring — that turns
quantization from a hand-tuned compression knob into a **recall-SLA-targeting, self-calibrating
subsystem**, and it composes with the self-aware ANN routing thesis into a single per-query
"spend only where it buys recall" planner.

---

## 7. Recommended next step (to turn `[proj]` into `[measured]`)

Stand up a quant A/B harness on a real high-D set (e.g. 1536-D OpenAI-style or GIST-960):
baseline (x32, rotation/ADC off, dim-keyed oversample) vs. (a) OSQ path, (b) rotation+ADC on,
(c) calibrated depth. Measure recall@{10,100} × oversample × cold/warm cache latency. That
converts §6's projections into plugin-specific measured deltas and validates the P0 claim before
committing the P1 research.

---

*Companion to [`FILTERED_SEARCH_CHEATSHEET.md`](FILTERED_SEARCH_CHEATSHEET.md) (filtered-search
techniques) and [`NEXTGEN_FILTERED_SEARCH.md`](NEXTGEN_FILTERED_SEARCH.md) (roadmap). Code
inventory verified Aug 2026 against this worktree; frontier survey current to Aug 2026.*
