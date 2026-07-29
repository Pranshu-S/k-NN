/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.test.OpenSearchTestCase;

import static org.mockito.Mockito.mock;

/**
 * Deterministic unit tests for {@link BoundedExactSearchDecider}'s pure decision logic.
 */
public class BoundedExactSearchDeciderTests extends OpenSearchTestCase {

    private static final Query FILTER = new MatchAllDocsQuery();

    @Override
    public void tearDown() throws Exception {
        BoundedExactSearchDecider.setThresholdOverrideForTesting(null);
        super.tearDown();
    }

    public void testThresholdFloorAndMultiplier() {
        assertEquals(1_000L, BoundedExactSearchDecider.threshold(1));   // floor
        assertEquals(1_000L, BoundedExactSearchDecider.threshold(10));  // floor
        assertEquals(1_000L, BoundedExactSearchDecider.threshold(100)); // multiplier boundary: 10*100 == floor
        assertEquals(1_010L, BoundedExactSearchDecider.threshold(101)); // just above the floor
        assertEquals(2_000L, BoundedExactSearchDecider.threshold(200)); // multiplier region
    }

    public void testHardCapBoundary() {
        assertEquals(10_000, BoundedExactSearchDecider.HARD_MAX_EXACT_CANDIDATES);
        // k = 1000 -> 10*1000 = 10000 == cap (unchanged; the common case is never capped)
        assertEquals(10_000L, BoundedExactSearchDecider.threshold(1_000));
        // k = 1001 -> 10010, capped to 10000
        assertEquals(10_000L, BoundedExactSearchDecider.threshold(1_001));
        assertEquals(10_000L, BoundedExactSearchDecider.threshold(1_500));
    }

    public void testValuesAboveHardCapAreCapped() {
        assertEquals(10_000L, BoundedExactSearchDecider.threshold(5_000));   // was 50000 before the cap
        assertEquals(10_000L, BoundedExactSearchDecider.threshold(10_000));  // was 100000 before the cap
        assertEquals(10_000L, BoundedExactSearchDecider.threshold(100_000));
    }

    public void testMaxLegalKAndMultiplicationOverflow() {
        // 10 * k overflows int for k > Integer.MAX_VALUE / 10; must not throw, must return the cap.
        assertEquals(10_000L, BoundedExactSearchDecider.threshold(Integer.MAX_VALUE));
        assertEquals(10_000L, BoundedExactSearchDecider.threshold(Integer.MAX_VALUE / 10 + 1));
        // just below overflow still computes and caps
        assertEquals(10_000L, BoundedExactSearchDecider.threshold(Integer.MAX_VALUE / 10));
    }

    public void testExactAtLimitAndAnnAtLimitPlusOne() {
        // k = 1500 -> candidateLimit = 10000 (capped). EXACT at the limit, APPROXIMATE just past it.
        assertEquals(10_000L, BoundedExactSearchDecider.threshold(1_500));
        assertEquals(KNNFilterExecutionStrategy.EXACT, BoundedExactSearchDecider.decide(10_000, 1_500));
        assertEquals(KNNFilterExecutionStrategy.APPROXIMATE, BoundedExactSearchDecider.decide(10_001, 1_500));
    }

    public void testDecideBands() {
        // threshold(10) = 1000
        assertEquals(KNNFilterExecutionStrategy.MATCH_NONE, BoundedExactSearchDecider.decide(0, 10));
        assertEquals(KNNFilterExecutionStrategy.EXACT, BoundedExactSearchDecider.decide(1, 10));
        assertEquals(KNNFilterExecutionStrategy.EXACT, BoundedExactSearchDecider.decide(999, 10));
        assertEquals(KNNFilterExecutionStrategy.EXACT, BoundedExactSearchDecider.decide(1_000, 10));
        assertEquals(KNNFilterExecutionStrategy.APPROXIMATE, BoundedExactSearchDecider.decide(1_001, 10));
    }

    public void testKMultiplierExceedsAbsoluteThreshold() {
        // k = 200 -> threshold = max(1000, 2000) = 2000.
        assertEquals(KNNFilterExecutionStrategy.EXACT, BoundedExactSearchDecider.decide(1_500, 200));
        assertEquals(KNNFilterExecutionStrategy.EXACT, BoundedExactSearchDecider.decide(2_000, 200));
        assertEquals(KNNFilterExecutionStrategy.APPROXIMATE, BoundedExactSearchDecider.decide(2_001, 200));
    }

    public void testStrategyForBoundedCount() {
        // 0 matches -> MATCH_NONE.
        assertEquals(KNNFilterExecutionStrategy.MATCH_NONE, BoundedExactSearchDecider.strategyForBoundedCount(0, false));
        // 1..threshold matches (not exceeded) -> EXACT.
        assertEquals(KNNFilterExecutionStrategy.EXACT, BoundedExactSearchDecider.strategyForBoundedCount(1, false));
        assertEquals(KNNFilterExecutionStrategy.EXACT, BoundedExactSearchDecider.strategyForBoundedCount(1_000, false));
        // threshold + 1 observed (exceeded) -> APPROXIMATE.
        assertEquals(KNNFilterExecutionStrategy.APPROXIMATE, BoundedExactSearchDecider.strategyForBoundedCount(1_001, true));
    }

    public void testThresholdOverrideForTesting() {
        BoundedExactSearchDecider.setThresholdOverrideForTesting(5);
        assertEquals(5L, BoundedExactSearchDecider.threshold(10));
        assertEquals(KNNFilterExecutionStrategy.EXACT, BoundedExactSearchDecider.decide(5, 10));
        assertEquals(KNNFilterExecutionStrategy.APPROXIMATE, BoundedExactSearchDecider.decide(6, 10));
    }

    public void testIsSupportedHappyPath() {
        assertTrue(BoundedExactSearchDecider.isSupported(VectorDataType.FLOAT, KNNEngine.LUCENE, FILTER, null, false, false));
    }

    public void testUnsupportedConfigurationsBypassPlanning() {
        // No filter present.
        assertFalse(BoundedExactSearchDecider.isSupported(VectorDataType.FLOAT, KNNEngine.LUCENE, null, null, false, false));
        // Non-Lucene engine.
        assertFalse(BoundedExactSearchDecider.isSupported(VectorDataType.FLOAT, KNNEngine.FAISS, FILTER, null, false, false));
        // Non-FLOAT vectors.
        assertFalse(BoundedExactSearchDecider.isSupported(VectorDataType.BYTE, KNNEngine.LUCENE, FILTER, null, false, false));
        assertFalse(BoundedExactSearchDecider.isSupported(VectorDataType.BINARY, KNNEngine.LUCENE, FILTER, null, false, false));
        // Nested field (parentFilter present).
        final BitSetProducer parentFilter = mock(BitSetProducer.class);
        assertFalse(BoundedExactSearchDecider.isSupported(VectorDataType.FLOAT, KNNEngine.LUCENE, FILTER, parentFilter, false, false));
        // Rescore enabled.
        assertFalse(BoundedExactSearchDecider.isSupported(VectorDataType.FLOAT, KNNEngine.LUCENE, FILTER, null, true, false));
        // expand_nested enabled.
        assertFalse(BoundedExactSearchDecider.isSupported(VectorDataType.FLOAT, KNNEngine.LUCENE, FILTER, null, false, true));
    }
}
