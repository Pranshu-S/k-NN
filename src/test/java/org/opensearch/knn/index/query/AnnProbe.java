/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;

import java.io.IOException;

/**
 * Benchmark-only (test-source) helper that runs a <em>bounded ANN probe</em> against the Lucene HNSW
 * graph using only public APIs — {@link org.apache.lucene.index.LeafReader#searchNearestVectors} with a
 * {@link TopKnnCollector} whose visit limit is the probe budget. It exposes the signals a probe could
 * observe pre-commit: graph nodes visited, whether the search early-terminated on the visit limit, and
 * how many candidates were collected. Used to test whether these signals predict the exact-vs-ANN
 * faster path (feasibility study; not production).
 */
final class AnnProbe {

    private AnnProbe() {}

    static final class Result {
        final long visited;
        final boolean earlyTerminated;
        final int collected;
        final long visitLimitTotal;
        final int leavesProbed;

        Result(long visited, boolean earlyTerminated, int collected, long visitLimitTotal, int leavesProbed) {
            this.visited = visited;
            this.earlyTerminated = earlyTerminated;
            this.collected = collected;
            this.visitLimitTotal = visitLimitTotal;
            this.leavesProbed = leavesProbed;
        }
    }

    /**
     * Runs a per-leaf HNSW probe with a {@code budget} visit limit, over the filter's live matches.
     *
     * @param filterWeight a {@code COMPLETE_NO_SCORES} weight for the (rewritten) filter
     * @param budget       per-leaf visit limit (probe budget)
     */
    static Result probe(
        final IndexSearcher searcher,
        final Weight filterWeight,
        final String field,
        final float[] query,
        final int k,
        final int budget
    ) throws IOException {
        long visited = 0;
        boolean early = false;
        int collected = 0;
        int leaves = 0;
        for (final LeafReaderContext leaf : searcher.getIndexReader().leaves()) {
            final Bits acceptBits = filterBits(filterWeight, leaf);
            if (acceptBits == null) {
                continue;
            }
            leaves++;
            final AcceptDocs acceptDocs = AcceptDocs.fromLiveDocs(acceptBits, leaf.reader().maxDoc());
            final TopKnnCollector collector = new TopKnnCollector(k, budget);
            leaf.reader().searchNearestVectors(field, query, collector, acceptDocs);
            visited += collector.visitedCount();
            early |= collector.earlyTerminated();
            collected += collector.topDocs().scoreDocs.length;
        }
        return new Result(visited, early, collected, (long) budget * Math.max(1, leaves), leaves);
    }

    /** Filter matches on this leaf intersected with live docs, or {@code null} if none. */
    private static Bits filterBits(final Weight filterWeight, final LeafReaderContext leaf) throws IOException {
        final Scorer scorer = filterWeight.scorer(leaf);
        if (scorer == null) {
            return null;
        }
        final Bits liveDocs = leaf.reader().getLiveDocs();
        final int maxDoc = leaf.reader().maxDoc();
        final FilteredDocIdSetIterator live = new FilteredDocIdSetIterator(scorer.iterator()) {
            @Override
            protected boolean match(final int doc) {
                return liveDocs == null || liveDocs.get(doc);
            }
        };
        final BitSet bits = BitSet.of(live, maxDoc);
        return bits.cardinality() == 0 ? null : bits;
    }
}
