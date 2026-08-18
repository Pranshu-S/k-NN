## Follow-up: exact #16092 rotation reproduces the 1-bit candidate-recall gain; permutation helps the estimator but not candidate count; iid control flat

I re-ran my earlier Fashion-MNIST experiment using the **exact rotation from this PR** (verbatim `HadamardRotation`, PR head `132f8b17715db53a54608a0bbc8cb75b16be7f87`) instead of my home-grown sign-only transform, in front of Lucene's **real** 1-bit BBQ scoring path (`OptimizedScalarQuantizer` 1-bit doc + 4-bit asymmetric query + `int4BitDotProduct` + the Lucene104 corrective formula). My fast scoring loop is verified **bit-identical** to `Lucene104ScalarQuantizedVectorScorer` (max rel error `0.000e+00` over 3000 random vectors incl. the zero-centroid case).

**Setup:** Fashion-MNIST-784 (real), 20 000 index / 200 held-out queries, Euclidean, `HnswGraphBuilder` M=16 beam=100. Metric = **candidate recall@10 BEFORE fp32 rerank** ("candidate count" = `TopKnnCollector` k, not `ef_search`). Rotated numbers are the mean over 3 seeds (candidate-count targets had zero seed variance). iid Gaussian-784 is the isotropic control.

**Native graph (build + search both quantized):**

| scorer | flat@10 | k@0.90 | k@0.95 | k@0.99 | MAE |
|---|---:|---:|---:|---:|---:|
| fp32 | 1.00 | 10 | 10 | 20 | 0 |
| 1-bit BBQ | 0.52 | 75 | **100** | 200 | 122 |
| sign-only rotation → BBQ | 0.76 | 20 | **30** | 50 | 47 |
| **#16092 → BBQ** | 0.77 | 20 | **30** | 50 | **40** |

**Fixed fp32 graph (only the search scorer changes → isolates traversal scoring):** BBQ k@0.95 = 75 → **#16092 30** (2.5×).

**iid-784 control:** BBQ / sign-only / #16092 all need k@0.95 = **750** (native and fixed-graph). The rotation is a no-op on isotropic data, as expected — the Fashion-MNIST gain is structure-exploitation, not added information.

**Two honest caveats / negatives:**
1. **The permutation didn't lower the candidate count vs my sign-only rotation** (both hit k@0.95 = 30). It did consistently lower the distance-estimate error (MAE 40 vs 47, RMSE 53 vs 63 across all seeds), so it improves the estimator — just not enough to move the discrete candidate threshold on this data. I'd expect it to matter more when structure is trapped inside a single power-of-two block, which the permutation breaks up.
2. **On #16030 data-blind (`enableCentering=false`, centroid = 0):** *without* rotation, data-blind actually beat centered on Fashion-MNIST (k@0.95 75 vs 100); *with* #16092, centered pulled ahead (30 vs 50). So preconditioning did **not** make data-blind match centered here — it favored centered. (On iid they tie.)

**Not measured:** latency/QPS — the sweeps use the proven-equal scalar formula, not the SIMD scorer, and I didn't equal-recall profile the O(d·log d) rotation cost. This is a candidate-count result, not a wall-clock one.

Storage is unchanged by the rotation (one global seed; per-vector code+corrective = 98 + 16 = 114 B/vec at 784-D → 1.163 bits/dim).

Harness + raw CSVs (reproducible against Lucene 10.5.0 on JDK 21):
`Track2Lucene16092Tests.java`, `HadamardRotation.java` (verbatim, sha256 `1044cf4e…a6e7c`), `research/track2_adaptive_rescore/results/step15_*.csv`, `step16_*.csv`.

**Question for maintainers:** is the permutation intended primarily to guarantee mixing across the block-diagonal boundaries (robustness on adversarial/blocked layouts), rather than to improve average recall? And given the residual fp32↔1-bit gap that survives rotation (k=10 vs 30 for 0.95), would a corrected/RaBitQ-style distance estimator on top of #16092 be in scope, or is that considered orthogonal to this PR?
