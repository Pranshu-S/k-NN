/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.annotations.VisibleForTesting;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

/**
 * Decides whether a <em>filtered</em> top-k Lucene FLOAT vector query is eligible for
 * <b>bounded exact search</b>.
 *
 * <p><b>This is deliberately not a general ANN cost planner.</b> It does not attempt to predict the
 * globally fastest path — prior research showed that cannot be done cheaply and reliably (cardinality,
 * document-id locality, segment concentration, sampled vector dispersion and runtime ANN probing all
 * failed to generalize). The rule here is narrow and conservative: run exact search only when its work
 * is <em>bounded and predictable</em>, and otherwise fall through to Lucene's existing filtered ANN
 * (which itself already falls back to exact when the graph search terminates early or returns fewer
 * than {@code k}). Against that ANN, exact search additionally guarantees recall = 1.0.
 *
 * <p>For a filtered top-k request one of three strategies is chosen from the shard-local live-match
 * count:
 * <ul>
 *   <li>{@link KNNFilterExecutionStrategy#MATCH_NONE} when the filter matches zero documents;</li>
 *   <li>{@link KNNFilterExecutionStrategy#EXACT} when the count is in {@code [1, candidateLimit]} where
 *       {@code candidateLimit = min(HARD_MAX_EXACT_CANDIDATES, max(1000, k * 10))} — a brute-force scan
 *       of only the filtered docs, hard-capped so a large {@code k} cannot trigger an unbounded scan;</li>
 *   <li>{@link KNNFilterExecutionStrategy#APPROXIMATE} when the count exceeds the limit (unchanged ANN).</li>
 * </ul>
 *
 * <h2>Why a cardinality bound (and not a dimension-aware {@code ops/dim} bound)</h2>
 * Calibration ({@code BoundedExactCalibrationBenchmarkTests}) showed exact filtered-search latency is
 * driven by the <em>number of candidate vectors scanned</em> (random-access memory cost per vector),
 * not by the float-multiply count {@code cardinality * dimension}. Two cells with equal {@code
 * card*dim} differed ~7x in latency. A cardinality bound therefore caps the dominant cost directly and
 * uniformly across dimensions (worst-case exact ≈ 1–2 ms at {@code candidateLimit = 1000}). A
 * dimension-aware {@code maxExactDimensionOperations / dim} bound was evaluated and rejected: it permits
 * large-cardinality exact scans at low dimension (e.g. ~7,500-vector scans at dim 128 → ~8.7 ms),
 * raising and de-stabilizing the worst case — the opposite of the goal.
 *
 * <h2>Why shard-local, at execution time</h2>
 * The count must be taken against the documents physically resident on this shard, so the decision is
 * made on the data node in {@link BoundedExactKnnFloatVectorQuery#createWeight} — where the authoritative
 * shard searcher and a single reader generation are available — never during JSON parsing or
 * coordinating-node rewrite. The count is <em>bounded</em>: {@link BoundedFilterResult} stops after
 * {@code candidateLimit + 1} live matches (all {@link #strategyForBoundedCount} needs) and retains the
 * collected doc ids so the exact path reuses them without re-evaluating the filter.
 *
 * <h2>Scope</h2>
 * Only the Lucene engine, FLOAT vectors, non-nested fields, {@code k}-based top-k queries with a filter
 * present and rescore disabled are eligible ({@link #isSupported}). Every other configuration — native
 * engines, BYTE/BINARY, radial search, nested / expand_nested, rescore, no filter — bypasses this
 * decider and preserves existing behavior exactly.
 *
 * <p>{@code candidateLimit} is an internal constant, not a public setting. It is overridable in tests
 * via {@link #setThresholdOverrideForTesting(Integer)}.
 */
final class BoundedExactSearchDecider {

    /** Absolute floor for the exact-search candidate limit (bounds vectors scanned). Internal constant. */
    @VisibleForTesting
    static final int ABSOLUTE_THRESHOLD = 1_000;

    /** {@code k} multiplier contributing to the candidate limit. Internal constant. */
    @VisibleForTesting
    static final int K_MULTIPLIER = 10;

    /**
     * Hard upper bound on the number of candidate vectors an exact scan may touch, regardless of {@code
     * k}. Without it, a large {@code k} (e.g. {@code k = 10,000}) would allow {@code candidateLimit =
     * 100,000}, an exact scan the calibration measured at tens of milliseconds — and the exact path does
     * no {@code QueryTimeout} check mid-scan, so it is uninterruptible. This cap bounds the worst-case
     * exact latency. Chosen conservatively: it is above {@code max(1000, 10*k)} for every {@code k <=
     * 1000}, so it changes nothing for the common case and only bites the pathological large-{@code k}
     * regime. See {@code BoundedExactHardCapBenchmarkTests}.
     */
    @VisibleForTesting
    static final int HARD_MAX_EXACT_CANDIDATES = 10_000;

    /**
     * Test-only override of the exact-search candidate limit. When non-null it replaces the computed
     * value so integration tests can cross the boundary with a handful of documents rather than indexing
     * more than a thousand. Must be reset to {@code null} after each test.
     */
    private static volatile Integer thresholdOverrideForTesting = null;

    private BoundedExactSearchDecider() {}

    /**
     * The exact-search candidate limit for the given {@code k}:
     * {@code min(HARD_MAX_EXACT_CANDIDATES, max(ABSOLUTE_THRESHOLD, K_MULTIPLIER * k))} — unless a test
     * override is installed. Bounds the number of candidate vectors an exact scan may touch, and hence
     * its worst-case cost. Overflow-safe: if {@code K_MULTIPLIER * k} overflows an {@code int} it would
     * exceed the hard cap anyway, so the cap is returned.
     */
    static long threshold(final int k) {
        final Integer override = thresholdOverrideForTesting;
        if (override != null) {
            return override;
        }
        final int kBound;
        try {
            kBound = Math.multiplyExact(K_MULTIPLIER, k);
        } catch (final ArithmeticException overflow) {
            // K_MULTIPLIER * k exceeds Integer.MAX_VALUE -> far above the cap.
            return HARD_MAX_EXACT_CANDIDATES;
        }
        return Math.min(HARD_MAX_EXACT_CANDIDATES, Math.max(ABSOLUTE_THRESHOLD, kBound));
    }

    @VisibleForTesting
    static void setThresholdOverrideForTesting(final Integer threshold) {
        thresholdOverrideForTesting = threshold;
    }

    /**
     * Whether the request falls within the supported MVP surface. Any {@code false} here means the
     * caller must preserve existing behavior (approximate path).
     *
     * @param vectorDataType the mapped vector data type
     * @param knnEngine      the mapped engine
     * @param filterQuery    the converted Lucene filter, or {@code null} when no filter is present
     * @param parentFilter   non-null for nested fields
     * @param needsRescore   whether rescore is enabled for this request
     * @param expandNested   whether expand_nested is requested
     */
    static boolean isSupported(
        final VectorDataType vectorDataType,
        final KNNEngine knnEngine,
        final Query filterQuery,
        final BitSetProducer parentFilter,
        final boolean needsRescore,
        final boolean expandNested
    ) {
        return knnEngine == KNNEngine.LUCENE
            && vectorDataType == VectorDataType.FLOAT
            && filterQuery != null
            && parentFilter == null
            && needsRescore == false
            && expandNested == false;
    }

    /**
     * Maps a {@link BoundedFilterResult}'s outcome to a strategy:
     * <pre>
     *   0 matches               -&gt; MATCH_NONE
     *   1..threshold matches     -&gt; EXACT
     *   threshold + 1 observed   -&gt; APPROXIMATE
     * </pre>
     */
    static KNNFilterExecutionStrategy strategyForBoundedCount(final long observedMatches, final boolean thresholdExceeded) {
        if (thresholdExceeded) {
            return KNNFilterExecutionStrategy.APPROXIMATE;
        }
        if (observedMatches == 0) {
            return KNNFilterExecutionStrategy.MATCH_NONE;
        }
        return KNNFilterExecutionStrategy.EXACT;
    }

    /**
     * Pure decision from an already-known (fully counted) shard-local cardinality. Retained for
     * documentation and unit testing of the threshold band; the execution path uses the bounded variant
     * {@link #strategyForBoundedCount}.
     */
    static KNNFilterExecutionStrategy decide(final long filteredCardinality, final int k) {
        final long threshold = threshold(k);
        if (filteredCardinality == 0) {
            return KNNFilterExecutionStrategy.MATCH_NONE;
        }
        if (filteredCardinality <= threshold) {
            return KNNFilterExecutionStrategy.EXACT;
        }
        return KNNFilterExecutionStrategy.APPROXIMATE;
    }
}
