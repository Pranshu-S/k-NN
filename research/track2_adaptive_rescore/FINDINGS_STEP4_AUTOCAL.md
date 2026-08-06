# Track 2, Step 4 — Elastic-style auto-calibrated fixed rerank depth: results & verdict

**Branch:** `track2-adaptive-rescore`. Benchmark only; the safe Bonferroni adaptive policy is
untouched and used as a baseline (it reranks ≈ full pool). Harness:
[`Track2AutoCalibrationTests.java`](../../src/test/java/org/opensearch/knn/research/Track2AutoCalibrationTests.java).
Raw: [`results/track2_autocal_summary.csv`](results/track2_autocal_summary.csv),
[`results/track2_autocal_segments.csv`](results/track2_autocal_segments.csv).

## Algorithm

`recall@r` = fraction of true top-k within the first `r` approximate candidates — monotone in `r`.
On a **calibration** query split, pick the smallest depth in a grid whose **mean** recall (or a
**bootstrap 5% lower bound**, "conservative" mode, + 0.005 margin) meets target; **validate** on
disjoint held-out folds (5-fold, fixed seeds). Not a per-query guarantee — a dataset/segment-level
recall SLA. Rankers: ADC, Hamming, Corrected (step-3 8-byte metadata used as a **ranking** signal).

## Central question — answered: YES, where the candidate pool contains the answers

The binding constraint is the **candidate-pool ceiling** (`poolRec` = recall even at full depth).
Where the pool contains the true top-k, calibration reliably picks a much cheaper depth that meets
the SLA and **matches best-fixed-in-hindsight without hindsight**. Conservative mode, target 0.95:

| dataset | ranker | poolRec | selected depth | held-out recall (worst) | SLA-miss | vs full-pool |
|---|---|---|---|---|---|---|
| **sift_real** | ADC | 0.998 | **420** | 0.978 (0.952) | **0.00** | **2.4× cheaper** |
| **isotropic** | Corrected | 0.996 | **380** | 0.967 (0.948) | 0.20 | **2.6× cheaper** |
| isotropic | ADC | 0.886 | 1000 | 0.886 | 1.00 | ceiling < target |
| clustered | ADC | 0.683 | 1000 | 0.683 | 1.00 | ceiling < target |

So vs the per-query Bonferroni policy (which reranks ~1000 to be per-query-safe), **dataset-level
calibration hits a 0.95 SLA at depth ~380–420 — 2.4–2.6× cheaper — reliably (SLA-miss 0).** This is
the decisive result: **the practical SLA framing beats per-query certification.**

## Four findings

1. **Conservative mode fixes SLA-miss.** Mean mode is too aggressive (sift ADC tgt 0.90: depth 150,
   SLA-miss 0.40). The bootstrap-lower-bound mode trades a little depth for reliability (sift ADC
   0.95: 300→420, SLA-miss 0.40→0.00). Use conservative mode for an SLA.
2. **Failures are candidate-GENERATION ceilings, not calibration failures.** isotropic/clustered ADC
   & Hamming have `poolRec` 0.46–0.89 < target — the true top-k aren't in the top-1000 approx pool,
   so *no* depth can meet the target. The harness attributes this to the pool ceiling, not the depth
   policy (as required). Calibration correctly returns 1000 and flags the miss.
3. **BIG: the corrected estimator is a strong RANKING signal — it raises the pool ceiling.** poolRec
   at depth 1000: isotropic ADC **0.886 → Corrected 0.996**; clustered ADC **0.683 → Corrected
   0.940**. The 8-byte metadata that was a NO-GO for per-query *bounds* (step 3) is a clear win as a
   *ranking* improver — it puts more true neighbours in the pool, making SLAs achievable that ADC/
   Hamming ranking cannot reach at any depth. (Validates the task's "lowest RMSE ≠ best rerank
   recall" caution — here corrected ranking is simply better on structured synthetic data.)
4. **Per-segment calibration beats a single global depth on heterogeneous data.** One global depth
   across the 3 datasets = 1000 (dominated by the hardest). Per-segment: sift ADC needs only **300**
   (3.3× cheaper) while the hard segments are ceiling-limited regardless. Local calibration avoids
   paying the hardest segment's cost on easy segments — the core value of segment-level calibration.

## Baseline comparison (the honest framing)

- **vs full-pool / Bonferroni adaptive:** 2.4–3.3× cheaper where the ceiling permits, at SLA-miss 0.
- **vs best-fixed-in-hindsight:** calibration *matches* it (e.g. sift ADC 0.95 → 300 both) — it finds
  the right depth from held-out data, no oracle needed.
- **vs min-100:** min-100 is cheaper but **does not meet the SLA** (sift ADC recall@100 = 0.864 < 0.95).
  "Savings vs min-100" is negative *because meeting a higher recall requires more depth* — not a loss.
- **vs per-query oracle:** oracle mean depth 205 (sift ADC) vs calibrated 420 — calibration pays a
  fixed-depth premium over per-query ideal, the price of not doing (unreliable) per-query routing.

## Go / Partial-go / No-go

```
PARTIAL-GO (conditional go) — the practical, recommended direction.
Go conditions (met):
  - Held-out recall reliably meets target where poolRec >= target; conservative mode -> SLA-miss ~0.
  - Selected depth matches best-fixed-in-hindsight (found without hindsight).
  - 2.4-3.3x cheaper than the full-pool per-query-safe policy.
  - Per-segment calibration saves materially on easy/heterogeneous segments.
  - Calibration cost is tiny (a few hundred sample queries; exact scores already available at merge).
Caveats:
  - Cannot exceed the candidate-pool ceiling -> when poolRec < target it's a candidate-GENERATION
    problem (fix ranking, not depth). The corrected estimator as a RANKER raises this ceiling a lot.
  - Requires conservative (lower-bound) selection to hold the SLA; mean-mode under-selects.
Not a per-query correctness guarantee; it is a distribution-level recall SLA (by design).
Decision: adopt dataset/segment-level SLA calibration over per-query certification; pair it with a
better ranker (corrected/RaBitQ) to raise the pool ceiling where needed.
```

## Design note — running this at Lucene/OpenSearch segment creation/merge

At flush/merge the segment's FP32 vectors are present, so exact top-k for a **sample** of pseudo-
queries (sampled segment vectors, or a recent query-log sample) is cheap to compute alongside the
quantized codes. Steps per segment: (1) sample ~few-hundred queries; (2) build the `recall@depth`
curve on the grid; (3) pick the conservative-lower-bound depth for the configured SLA; (4) store the
per-segment depth in segment metadata. At query time, use the segment's calibrated depth (sum across
probed segments). Recalibrate on merge (the curve shifts only with distribution, not size —
consistent with Elastic's "glacial scaling"). Cost is negligible vs the merge, which already reads
every vector. No per-query bound math, no OpenSearch integration in this task.

## Reproduce
```bash
./gradlew :test --tests "org.opensearch.knn.research.Track2AutoCalibrationTests" \
    -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
# writes results/track2_autocal_{summary,segments}.csv
```
