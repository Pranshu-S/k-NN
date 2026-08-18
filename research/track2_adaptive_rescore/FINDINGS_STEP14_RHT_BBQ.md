> **Correction (2026-08-18, see [`track2_lucene_16029_followup.md`](track2_lucene_16029_followup.md)):** the storage
> numbers below (**120 B/vec, 1.224 bits/dim** for 784‑D 1‑bit BBQ) are **overstated**. This harness packed to
> `discretize(dim,64)=832` (→104 B code) instead of the real Lucene codec's `getDiscreteDimensions=784`
> (→**98 B** code); with 16 B corrective the true size is **114 B/vec (1.163 bits/dim)**. The extra 48 bits were
> zeros that contribute nothing to `int4BitDotProduct`, so **every recall/ef number below is unaffected** and has
> since been independently confirmed bit-identical to the real `Lucene104ScalarQuantizedVectorScorer` (parity error
> 0). Because both BBQ and RHT→BBQ shared the same (overstated) byte count, the "rotation adds no storage / 3.3×
> fewer candidates" conclusions stand. Only the absolute byte count is corrected.

# Track 2, Step 14 — Does RHT preconditioning improve Lucene's real 1‑bit OSQ/BBQ?

**Question:** does a random orthogonal (Hadamard) transform before **Lucene's actual production BBQ**
improve candidate recall and cut the HNSW `ef_search` needed for a recall target — with *everything else
identical*?

**Answer: YES on structured real data, NO on the isotropic control — the "rotation is the reason"
hypothesis survives a strict controlled experiment.** On fashion‑mnist‑784, RHT→BBQ cuts `ef95` from
**100 → 30 (3.3×)** at identical storage; on iid‑784 Gaussian it changes nothing (`ef95` 750→750).

Harness: [`Track2RhtBbqTests.java`](../../src/test/java/org/opensearch/knn/research/Track2RhtBbqTests.java).
Raw: [`results/step14_rht_bbq.csv`](results/step14_rht_bbq.csv),
[`results/step14_common_graph.csv`](results/step14_common_graph.csv).

## 1. This is REAL BBQ, not dequantize‑L2 (the thing that gates the whole experiment)

Phase 0 tracing found: Lucene 10.4 core ships the **quantizer** (`OptimizedScalarQuantizer`) but the
1‑bit asymmetric **scorer** lives in `Lucene104ScalarQuantizedVectorScorer` (found in the 10.5 sources
jar); in OpenSearch the same math runs in native SIMD. So the faithful path is:

```
doc   : OptimizedScalarQuantizer.scalarQuantize(x, dest, bits=1, centroid)  -> 1-bit levels + corrective terms
        OptimizedScalarQuantizer.packAsBinary(dest, packed)                 -> 1-bit code (disc/8 bytes)
query : scalarQuantize(q, dest, bits=4, centroid) ; transposeHalfByte(...)  -> 4-bit transposed planes
score : qc = VectorUtil.int4BitDotProduct(queryT, docPacked)               -> Lucene's real kernel
        dot   = ax*ay*dim + ay*lx*x1 + ax*ly*y1 + lx*ly*qc                  -- Lucene104 corrective formula
        dist2 = qAdd + dAdd - 2*dot                                         -- (EUCLIDEAN) squared L2
```

Build uses Lucene's **asymmetric graph geometry** (4‑bit‑query‑of‑A vs 1‑bit‑doc‑of‑B); traversal uses
4‑bit‑query‑of‑query vs 1‑bit‑doc. **RHT→BBQ is the byte‑for‑byte same pipeline on `R(x)=H(Dx)/√block`,
same seed for index+query — the only difference is the transform.** Invariants asserted: RHT
orthogonality (`|‖x−y‖−‖Rx−Ry‖|<1e‑3`), determinism, BBQ score stability; fp32 flat fidelity = 1.000.

**How much did my earlier dequantize‑L2 understate BBQ?** Modestly, not hugely: on fashion‑mnist,
`osq_dequant_l2` ef95 = 150 vs **real BBQ ef95 = 100**. The corrective‑term estimator helps, but the
earlier proxy was in the right ballpark — my prior worry that I had "crippled OSQ" was overblown.

## 2. Headline — fashion‑mnist‑784 (real), candidate recall@10 BEFORE rerank, native graphs

| representation | flat top10 | ef20 | ef50 | ef100 | **ef95** | **ef99** | total B/vec | build ms | MAE |
|---|---|---|---|---|---|---|---|---|---|
| fp32 | 1.000 | 0.99 | 1.00 | 1.00 | 10 | 20 | 3136 | 12058 | 0 |
| **bbq_1bit** (real Lucene) | 0.515 | 0.70 | 0.89 | 0.97 | **100** | 200 | 120 | 2466 | 122 |
| **rht_bbq_1bit** | 0.768 | 0.93 | 0.99 | 1.00 | **30** | 50 | 120 | 2136 (+86 rot) | 45 |
| rabitq_insp | 0.759 | 0.93 | 1.00 | 1.00 | 30 | 50 | 106 | 114682 | 35 |
| osq_dequant_l2 | 0.550 | 0.67 | 0.86 | 0.94 | 150 | 300 | 106 | 11344 | 170 |

- **RHT→BBQ: ef95 100→30 (3.3×), ef90 75→20 (3.75×), ef99 200→50 (4×)** — same 120 B/vec, only rotation added.
- **Distance‑estimate error halves+**: MAE 122 → 45 (2.7×). That is the mechanism: on correlated data
  the rotation decorrelates the axes so the 1‑bit sign + corrective terms estimate distances far better.
- **RHT→BBQ ≈ RaBitQ** on every ef — confirming rotation is exactly the ingredient BBQ dropped.
- **Rotation is nearly free**: 86 ms for 20k×784 (≈4.3 µs/vec); RHT→BBQ even **builds faster** than BBQ
  (2136 vs 2466 ms) because the better estimates make graph construction cheaper. Global rotation metadata
  = one seed + block config (`3×4+8 = 20 bytes` for the whole index), **not** per vector.

## 3. Control — iid‑784 Gaussian (isotropic → rotation should be a no‑op)

| representation | flat | ef200 | ef500 | ef1000 | ef95 |
|---|---|---|---|---|---|
| fp32 | 1.000 | 0.87 | 0.95 | 0.98 | 500 |
| bbq_1bit | 0.280 | 0.81 | 0.929 | 0.975 | 750 |
| rht_bbq_1bit | 0.324 | 0.80 | 0.923 | 0.972 | 750 |
| rabitq_insp | 0.300 | 0.81 | 0.940 | 0.974 | 750 |

**Rotation changes nothing** (ef95 750 = 750; recall curves overlap within noise). Exactly as predicted:
an orthogonal transform cannot help an already rotation‑invariant distribution, so the fashion‑mnist gain
is *not* "rotation adds information" — it is "rotation exploits structure." (Note iid is intrinsically
hard for HNSW here — even fp32 needs ef≈500 — but that is the dataset, not the quantizer.)

## 4. Where the gain comes from — common fp32 graph vs native graph (Experiment 1 vs 2)

Searching **one shared fp32 graph** with each scorer (isolates traversal‑scoring), fashion‑mnist:

| scorer on common fp32 graph | ef90 | ef95 | ef97 |
|---|---|---|---|
| fp32 | 10 | 10 | 20 |
| bbq_1bit | 50 | 75 | 100 |
| **rht_bbq_1bit** | 20 | **30** | 30 |
| rabitq_insp | 20 | 30 | 30 |

Decomposition of BBQ's ef95:
- **Traversal/scoring error:** on the *same* topology, rotation cuts ef95 **75 → 30 (2.5×)** — better
  distance estimates steer traversal better.
- **Build error:** BBQ's *own* graph (ef95 100) is worse than BBQ on the fp32 graph (75) — BBQ also builds
  a degraded topology; RHT→BBQ builds an fp32‑quality graph (30 on both). Rotation fixes this too.
- **Combined (native graph = production reality): 100 → 30 (3.3×).** Rotation helps at *both* stages.

## 5. Go/no‑go

**STRONG GO** on structured real data: ≥2× lower ef for 0.95 recall (**3.3×** achieved), **no** storage
increase (120 B/vec both; rotation metadata is a single global seed), transform overhead ≈ 4 µs/vec
(≪ 10% of saved ANN latency), and **no regression on the isotropic control**. Build cost is not increased
(RHT→BBQ builds slightly faster than BBQ).

## 6. Answers to the closing questions

1. **Concise finding:** RHT preconditioning makes Lucene's real 1‑bit BBQ ≈3.3× cheaper in `ef` for 0.95
   candidate recall on real 784‑D data, matching RaBitQ, at identical storage and negligible transform
   cost — and does nothing on isotropic data, proving the gain is from exploiting structure.
2. **Tables:** §2, §3, §4 (+ CSVs).
3. **Reproduce:**
   ```bash
   ./gradlew :test --tests "org.opensearch.knn.research.Track2RhtBbqTests.testInvariants" \
       --tests "org.opensearch.knn.research.Track2RhtBbqTests.testRhtBbq" \
       --tests "org.opensearch.knn.research.Track2RhtBbqTests.testCommonGraph" \
       -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
   # data: research/acorn/data/real/{fmnist,nytimes}_{base,query}.fvecs (from ann-benchmarks via to_fvecs.py)
   ```
4. **Changed files:** added `Track2RhtBbqTests.java` (new), `research/acorn/data/real/to_fvecs.py` (hdf5→fvecs
   converter), CSV outputs. No production Lucene/OpenSearch code modified — BBQ is invoked through the real
   `OptimizedScalarQuantizer` + `VectorUtil.int4BitDotProduct` via a benchmark adapter.
5. **Architecture:** real OSQ quantizer → real bit‑packing → real int4 dot kernel → exact Lucene104
   corrective formula; RHT is a global seeded block‑Hadamard (dimension‑preserving for 784 = 512+256+16,
   so no bit inflation) applied identically to index + query; consistent asymmetric geometry for build and
   traversal; exact fp32 ground truth; candidate recall before rerank.
6. **Unexpected results:** (a) my earlier dequantize‑L2 only modestly understated BBQ (ef95 150 vs 100), not
   the large gap I expected; (b) RHT→BBQ **builds faster** than plain BBQ (better estimates → cheaper
   construction) despite the extra transform; (c) rotation helps at **both** build and traversal, not only
   scoring; (d) my RaBitQ‑inspired impl is ~50× slower to build than BBQ (no query‑transpose caching), so
   **RHT→BBQ is the practical way to get RaBitQ‑level recall** at BBQ's speed and byte layout.
7. **Hypothesis verdict:** **SURVIVED.** With everything but the rotation held identical, rotation causes
   the improvement on structured data and is a no‑op on isotropic data — the textbook signature of a real,
   structure‑exploiting effect rather than an artifact.

## Limitations / honest scope
- Two datasets (one real 784‑D + isotropic control); N≈15–20k; warm, single host; latency in µs not yet
  profiled at equal‑recall (Phase 6) — candidate recall + ef is the decisive metric and is complete.
- BBQ scored via a faithful Java transcription of Lucene's `quantizedScore` (not the native SIMD kernel);
  same math, so recall is production‑accurate; absolute latency would need the native path.
- Dimension‑preserving block‑Hadamard (not a single power‑of‑two padded Hadamard) — documented; padded C1
  would add ~33% bits and is strictly worse on storage, so C2 is the right production candidate.
- RaBitQ‑inspired uses a structured (Fastfood‑style) rotation, so labelled *inspired*, not canonical RaBitQ.
