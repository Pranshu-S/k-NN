/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.test.OpenSearchTestCase;

/**
 * Unit tests for {@link IntArrayDocIdSetIterator}: the {@link DocIdSetIterator} contract, ascending
 * traversal, {@code advance}, exhaustion, and the memory property (independent of {@code maxDoc}).
 */
public class IntArrayDocIdSetIteratorTests extends OpenSearchTestCase {

    public void testEmpty() {
        final IntArrayDocIdSetIterator it = new IntArrayDocIdSetIterator(new int[0], 0);
        assertEquals(-1, it.docID());
        assertEquals(0, it.cost());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, it.nextDoc());
    }

    public void testSingleDoc() throws Exception {
        final IntArrayDocIdSetIterator it = new IntArrayDocIdSetIterator(new int[] { 7 }, 1);
        assertEquals(1, it.cost());
        assertEquals(7, it.nextDoc());
        assertEquals(7, it.docID());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, it.nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, it.docID());
    }

    public void testAscendingTraversalAndExhaustion() throws Exception {
        final int[] docs = { 0, 3, 5, 9, 12 };
        final IntArrayDocIdSetIterator it = new IntArrayDocIdSetIterator(docs, docs.length);
        int prev = -1;
        int seen = 0;
        for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
            assertTrue("must be strictly ascending", doc > prev);
            prev = doc;
            seen++;
        }
        assertEquals(docs.length, seen);
        // Still exhausted on further calls.
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, it.nextDoc());
    }

    public void testHonoursLengthBelowArrayCapacity() throws Exception {
        // Backing array has spare capacity (as produced by BoundedFilterResult's growable buffer).
        final int[] docs = { 1, 4, 8, 0, 0, 0 };
        final IntArrayDocIdSetIterator it = new IntArrayDocIdSetIterator(docs, 3);
        assertEquals(3, it.cost());
        assertEquals(1, it.nextDoc());
        assertEquals(4, it.nextDoc());
        assertEquals(8, it.nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, it.nextDoc());
    }

    public void testAdvanceToExactAndBetween() throws Exception {
        final int[] docs = { 2, 4, 6, 8, 10 };
        IntArrayDocIdSetIterator it = new IntArrayDocIdSetIterator(docs, docs.length);
        assertEquals(6, it.advance(6)); // exact hit
        assertEquals(8, it.advance(7)); // first >= 7
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, it.advance(11)); // past the end

        it = new IntArrayDocIdSetIterator(docs, docs.length);
        assertEquals(2, it.advance(0)); // before first
    }

    public void testAdvanceFromMiddle() throws Exception {
        final int[] docs = { 1, 5, 9, 13, 17 };
        final IntArrayDocIdSetIterator it = new IntArrayDocIdSetIterator(docs, docs.length);
        assertEquals(1, it.nextDoc());
        assertEquals(5, it.nextDoc());
        assertEquals(13, it.advance(10)); // first >= 10, skipping 9
        assertEquals(17, it.nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, it.nextDoc());
    }

    public void testMemoryIsIndependentOfMaxDoc() {
        // 20 matches "in a 10M-doc segment": the iterator holds only the 20 ids; maxDoc is irrelevant.
        final int[] docs = new int[20];
        for (int i = 0; i < docs.length; i++) {
            docs[i] = i * 500_000; // spread across a 10M id space
        }
        final IntArrayDocIdSetIterator it = new IntArrayDocIdSetIterator(docs, docs.length);
        assertEquals(20, it.cost()); // cost reflects matches, not maxDoc
    }
}
