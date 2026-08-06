# Track 2, Step 5 — Joint (pool, depth) SLA calibration across segments: results & verdict

**Branch:** `track2-adaptive-rescore`. Benchmark only; per-query Bonferroni policy untouched (baseline).
Harness: [`Track2JointCalibrationTests.java`](../../src/test/java/org/opensearch/knn/research/Track2JointCalibrationTests.java).
Raw: [`results/track2_joint_summary.csv`](results/track2_joint_summary.csv),
[`results/track2_joint_segments.csv`](results/track2_joint_segments.csv).

## Algorithm

Joint search over `(candidate_pool P, rerank_depth d≤P)` minimizing
`total_cost = CA[ranker]·P + CE·d` (CA = {ADC 2, Hamming 1, Corrected 3}; CE = 10; raw P/d also
reported), subject to: pool ceiling `recall@P ≥ target` **and** conservative bootstrap-5%-lower-bound
of `recall@d` on the calibration split `≥ target + margin`. Status:
`CALIBRATED / INCREASE_CANDIDATE_POOL / SLA_UNACHIEVABLE_AT_MAX_POOL`. 5-fold held-out validation.

**Empirical degeneracy (reported honestly):** in the flat-scan (exhaustive-candidate) regime
`recall@d` is pool-independent for `d≤P`, so the cost-optimal `P = d` in **every** CALIBRATED row.
The pool knob's real role here is the **ceiling / feasibility**; a genuine `P > d` optimum needs
graph-limited `ef` (native tier). This is a limitation of the simulator, not of the method.

## Research-question answers

**Q1 — cheapest (pool, depth) for the SLA.** Where feasible, `(d*, d*)` with `d*` the smallest
conservative-sufficient depth: SIFT ADC @0.95 → **(400,400)**, cost 4800 = **2.5× cheaper than
full-pool** (12000); isotropic Corrected @0.95 → (400,400), cost 5200.

**Q2 — calibration data required.** **~50 queries.** Selected depth and SLA-miss stabilize by
calSize 50 (isotropic Corrected: 50→400 = 500→400, SLA-miss 0 throughout). calSize 25 over-selects
(760±80 — small-sample conservatism) but never misses. → ~50–100 queries is cheap and stable.

**Q3 — safety margin to zero SLA-miss.** The **conservative (bootstrap lower-bound) rule alone gives
SLA-miss 0** on feasible cases even at margin 0.000; an explicit margin barely moves the depth
(isotropic Corrected: 0→340, 0.005→380, 0.01→400) at ~10–20% extra cost, and a large margin (0.02)
can over-constrain a borderline case into INCREASE_POOL. → use lower-bound mode + margin ≈ 0.005.
(This resolves step-4's SLA-miss: step 4 used *mean* mode; conservative mode fixes it.)

**Q4 — per-segment vs global (decisive YES).** One global config across the 3 heterogeneous segments
= **(1000,1000)** (dominated by the hardest) and *still* fails isotropic/clustered under ADC. Per
segment, SIFT needs only **(400,400) → 2.5× cheaper**, and each segment can pick its own ranker.
Global fixed policy is both wasteful on easy segments and infeasible on hard ones.

**Q5 — stability.** Selected-depth std = 0 at calSize ≥ 50 (sift, isotropic); SLA-miss 0 across folds
where feasible. Churn only at calSize 25.

**Q6/Q9 — corrected-ranker contribution (the production evidence, nuanced).** The 8-byte metadata as a
**ranker** changes feasibility, not just cost:
| case | ADC | Corrected |
|---|---|---|
| isotropic @0.95 | ceiling 0.907 → **SLA_UNACHIEVABLE** | ceiling 0.998 → **CALIBRATED (400,400)** |
| clustered @0.90 | 0.77 → **UNACHIEVABLE** | 0.957 → **CALIBRATED (840)** |
| isotropic @0.90 | INCREASE_POOL, SLA-miss 0.40 | **(200,200)**, SLA-miss 0 |
| **SIFT @0.95** | **(400,400) cost 4800** ✅ cheaper | (1000,1000) cost 13000 |
**Corrected is the ONLY feasible ranker on hard structured data (isotropic/clustered), but on easy
real data (SIFT) ADC ranking is cheaper** (better ceiling + lower CA). → **pick the ranker per
segment**; corrected's value is unlocking feasibility, not a universal cost win.

**Q7 — beats the best practical fixed baseline?** It *matches* best-fixed-in-hindsight (finds the
same depth without hindsight); beats **min-100** (which undershoots the SLA where the curve needs
more); beats **full-pool** 2.5×. So yes at equal recall, where feasible.

**Q8 — pool ceiling below SLA.** Handled explicitly: `SLA_UNACHIEVABLE_AT_MAX_POOL` (ceiling < target
at P=1000) vs `INCREASE_CANDIDATE_POOL` (ceiling ok, lower bound short) vs `CALIBRATED`. The remedy is
`USE_BETTER_RANKER` — the corrected ranker resolves most of the synthetic ceilings. 0.99 targets are
largely unachievable (ceilings top out ~0.98–0.998) → needs a stronger ranker (RaBitQ) or larger pool.

## Decision gate

```
PARTIAL-GO, leaning strong — recommended production prototype direction.
Met:
  - Conservative segment-level calibration hits the SLA with held-out SLA-miss 0 where feasible.
  - 2.5x cheaper than full-pool; matches best-fixed-in-hindsight; beats min-100 at equal recall.
  - Cheap calibration (~50 queries); low churn; 8-byte metadata overhead.
  - Per-segment materially beats one global config on heterogeneous data.
  - Corrected ranker unlocks feasibility on hard/structured segments.
Caveats (why not full strong-go):
  - Candidate-generation CEILINGS bind at high targets (0.99) and hard segments even with correction
    -> those need a stronger ranker (RaBitQ) or larger pool; the harness flags them, doesn't hide them.
  - The joint pool-vs-depth benefit is latent in flat-scan (P=d optimal); realizing P>d needs
    graph-limited ef (native tier).
  - Ranker must be chosen per segment (ADC on easy real data, corrected on hard) — no single winner.
Decision: adopt conservative per-segment (pool,depth) SLA calibration with per-segment ranker choice;
pair with a stronger ranker (RaBitQ) to raise ceilings where infeasible.
```

## Production design note — computing this at Lucene/OpenSearch flush or merge

At flush/merge the FP32 vectors are present, so for a **sample of ~50–100 pseudo-queries** (sampled
segment vectors or a recent query-log sample) exact top-k is cheap to compute alongside the quantized
codes. Per segment: (1) build `recall@depth` on the depth grid and the pool ceiling; (2) pick the
conservative-lower-bound `(pool, depth)` for the configured SLA + small margin; (3) if
`SLA_UNACHIEVABLE`, record the status and fall back (below); (4) store `(pool, depth, ranker, status)`
in **segment metadata**. Small-segment fallback (modeled): use the **global** calibrated config, or a
conservative default, or full rerank for tiny segments. At query time, use each probed segment's
stored `(pool, depth)`. Recalibrate on merge; the curve shifts only with distribution, not size
("glacial scaling"). Calibration cost ≈ 50–100 exact top-k computations/segment — negligible vs a
merge that already reads every vector. No per-query bound math; no OpenSearch integration in this task.

## Reproduce
```bash
./gradlew :test --tests "org.opensearch.knn.research.Track2JointCalibrationTests" \
    -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
# writes results/track2_joint_{summary,segments}.csv
```
