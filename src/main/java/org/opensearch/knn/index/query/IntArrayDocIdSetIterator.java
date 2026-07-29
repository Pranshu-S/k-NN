/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.DocIdSetIterator;

/**
 * A {@link DocIdSetIterator} over a pre-collected, ascending, segment-local {@code int[]} of doc ids.
 *
 * <p>Used by the exact-search path of {@link BoundedExactKnnFloatVectorQuery} to feed the doc ids
 * gathered during bounded filter counting straight into {@link org.opensearch.knn.index.query.exactsearch.ExactSearcher}
 * without materialising a {@code FixedBitSet(maxDoc)}. For sparse filters this makes the exact path's
 * memory {@code O(matches)} instead of {@code O(maxDoc)} — e.g. 20 matches in a 10M-doc segment cost a
 * ~80-byte array rather than a 1.25&nbsp;MB bit set.
 *
 * <p>The backing array must be sorted ascending in {@code [0, length)} (which is exactly how
 * {@link BoundedFilterResult} collects matches, iterating each segment's filter in doc-id order). Only
 * the first {@code length} entries are consumed; the array may have spare capacity beyond that. The
 * iterator is single-pass and not thread-safe, matching the {@link DocIdSetIterator} contract.
 */
final class IntArrayDocIdSetIterator extends DocIdSetIterator {

    private final int[] docIds;
    private final int length;
    private int index = -1;
    private int doc = -1;

    /**
     * @param docIds ascending doc ids; only indices {@code [0, length)} are used
     * @param length number of valid entries in {@code docIds}
     */
    IntArrayDocIdSetIterator(final int[] docIds, final int length) {
        assert length <= docIds.length;
        this.docIds = docIds;
        this.length = length;
    }

    @Override
    public int docID() {
        return doc;
    }

    @Override
    public int nextDoc() {
        if (++index >= length) {
            doc = NO_MORE_DOCS;
        } else {
            doc = docIds[index];
        }
        return doc;
    }

    @Override
    public int advance(final int target) {
        // Binary search within the remaining (ascending) region for the first doc >= target.
        int lo = index + 1;
        int hi = length - 1;
        int found = length;
        while (lo <= hi) {
            final int mid = (lo + hi) >>> 1;
            if (docIds[mid] >= target) {
                found = mid;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        index = found;
        doc = index >= length ? NO_MORE_DOCS : docIds[index];
        return doc;
    }

    @Override
    public long cost() {
        return length;
    }
}
