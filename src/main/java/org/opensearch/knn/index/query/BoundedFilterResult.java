/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.Bits;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Result of a <em>bounded</em>, single-pass evaluation of a filter query on one reader generation.
 *
 * <p>The collector walks the filter's matching, live documents segment by segment and stops as soon
 * as it has observed {@code threshold + 1} matches. This answers the only question the planner needs
 * — "is the filtered cardinality {@code 0}, in {@code [1, threshold]}, or {@code > threshold}?" —
 * without ever fully counting a large filter, and without a second filter pass:
 * <ul>
 *   <li>When the threshold is <b>not</b> exceeded, the exact per-segment matched doc ids are retained
 *       ({@link #getLeafMatches()}) so the exact search path can reuse them directly instead of
 *       re-evaluating the filter.</li>
 *   <li>When the threshold <b>is</b> exceeded, the partial collected state is released
 *       ({@link #release()}) because the ANN path will re-evaluate the filter itself.</li>
 * </ul>
 *
 * <h2>Segment safety</h2>
 * Doc ids are kept per {@link LeafReaderContext} and never merged into a global id space, so segment
 * context is preserved. All state is derived from the {@link IndexSearcher} passed to the collector,
 * i.e. a single reader generation; callers must consume it within the same {@code createWeight} call.
 *
 * <h2>Complexity</h2>
 * Time: {@code O(min(matches, threshold + 1))} iterator advances (plus one scorer per segment).
 * Memory: {@code O(min(matches, threshold + 1))} ints — bounded by {@code threshold + 1} regardless of
 * how large the filter is; no {@code FixedBitSet(maxDoc)} is allocated during counting.
 *
 * <p>Not thread-safe; intended to be built and consumed on a single query-planning thread. The
 * enclosing query is immutable.
 */
final class BoundedFilterResult {

    /** Ascending, segment-local matched doc ids for one leaf. */
    static final class LeafMatches {
        private final LeafReaderContext leaf;
        private final int[] docIds;
        private final int count;

        LeafMatches(final LeafReaderContext leaf, final int[] docIds, final int count) {
            this.leaf = leaf;
            this.docIds = docIds;
            this.count = count;
        }

        LeafReaderContext getLeaf() {
            return leaf;
        }

        int[] getDocIds() {
            return docIds;
        }

        int getCount() {
            return count;
        }
    }

    private final long observedMatches;
    private final boolean thresholdExceeded;
    private final long filterAdvances;
    private final int filterScorersCreated;
    private List<LeafMatches> leafMatches;

    private BoundedFilterResult(
        final long observedMatches,
        final boolean thresholdExceeded,
        final long filterAdvances,
        final int filterScorersCreated,
        final List<LeafMatches> leafMatches
    ) {
        this.observedMatches = observedMatches;
        this.thresholdExceeded = thresholdExceeded;
        this.filterAdvances = filterAdvances;
        this.filterScorersCreated = filterScorersCreated;
        this.leafMatches = leafMatches;
    }

    /**
     * The number of live documents matched by the filter, capped at {@code threshold + 1}. When
     * {@link #isThresholdExceeded()} is {@code true} this equals {@code threshold + 1} and is <b>not</b>
     * the exact cardinality; otherwise it is the exact cardinality.
     */
    long getObservedMatches() {
        return observedMatches;
    }

    /** Whether more than {@code threshold} matches exist (exact cardinality is unknown, only bounded). */
    boolean isThresholdExceeded() {
        return thresholdExceeded;
    }

    /** Total filter iterator advances performed (instrumentation; proves early termination). */
    long getFilterAdvances() {
        return filterAdvances;
    }

    /** Number of per-segment filter scorers created (instrumentation). */
    int getFilterScorersCreated() {
        return filterScorersCreated;
    }

    /**
     * The retained per-segment matches for reuse by the exact path. Empty after {@link #release()} or
     * when the threshold was exceeded.
     */
    List<LeafMatches> getLeafMatches() {
        return leafMatches;
    }

    /** Drops the retained per-segment matches so they can be garbage-collected once no longer needed. */
    void release() {
        leafMatches = List.of();
    }

    /**
     * Evaluates {@code filterWeight} across all segments of {@code searcher}, keeping only live matches,
     * and stops after observing {@code threshold + 1} of them.
     *
     * @param searcher     the authoritative searcher (its reader generation defines the doc ids)
     * @param filterWeight a weight for the (already rewritten) filter, created with
     *                     {@code ScoreMode.COMPLETE_NO_SCORES}
     * @param threshold    the exact-search threshold; collection stops once {@code threshold + 1} live
     *                     matches are seen
     */
    static BoundedFilterResult collect(final IndexSearcher searcher, final Weight filterWeight, final long threshold) throws IOException {
        final long limit = threshold + 1; // stop once this many live matches have been observed
        final List<LeafMatches> leafMatches = new ArrayList<>();
        long observed = 0;
        long advances = 0;
        int scorersCreated = 0;
        boolean exceeded = false;

        outer: for (final LeafReaderContext leaf : searcher.getIndexReader().leaves()) {
            final Scorer scorer = filterWeight.scorer(leaf);
            if (scorer == null) {
                continue;
            }
            scorersCreated++;
            final Bits liveDocs = leaf.reader().getLiveDocs();
            final DocIdSetIterator iterator = scorer.iterator();
            int[] docIds = new int[16];
            int count = 0;
            for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                advances++;
                if (liveDocs != null && liveDocs.get(doc) == false) {
                    continue; // skip deleted documents
                }
                if (count == docIds.length) {
                    docIds = java.util.Arrays.copyOf(docIds, docIds.length * 2);
                }
                docIds[count++] = doc;
                observed++;
                if (observed >= limit) {
                    exceeded = true;
                    // Keep this partial leaf so accounting is consistent; caller releases on exceed.
                    leafMatches.add(new LeafMatches(leaf, docIds, count));
                    break outer;
                }
            }
            if (count > 0) {
                leafMatches.add(new LeafMatches(leaf, docIds, count));
            }
        }

        if (exceeded) {
            // The ANN path re-evaluates the filter, so the partial collection is not reused.
            leafMatches.clear();
        }
        return new BoundedFilterResult(observed, exceeded, advances, scorersCreated, leafMatches);
    }
}
