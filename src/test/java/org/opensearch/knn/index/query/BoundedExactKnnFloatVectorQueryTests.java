/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.index.query.lucenelib.OSKnnFloatVectorQuery;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase.randomVector;
import static org.mockito.Mockito.mock;

/**
 * Tests for {@link BoundedExactKnnFloatVectorQuery}: bounded counting, early termination, exact
 * filter-state reuse, deletions, multi-segment correctness, and strategy selection.
 */
public class BoundedExactKnnFloatVectorQueryTests extends OpenSearchTestCase {

    private static final String FIELD = "vector-field";
    private static final String ID_POINT = "id-point";
    private static final String ID_STORED = "id-stored";
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.EUCLIDEAN;
    private static final int DIM = 4;
    private static final int EF_SEARCH = 50;
    private static final int THRESHOLD = 5;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        BoundedExactSearchDecider.setThresholdOverrideForTesting(THRESHOLD);
    }

    @Override
    public void tearDown() throws Exception {
        BoundedExactSearchDecider.setThresholdOverrideForTesting(null);
        super.tearDown();
    }

    private static final class Result {
        final TopDocs topDocs;
        final KNNFilterPlanningStats stats;
        final List<Integer> ids;

        Result(final TopDocs topDocs, final KNNFilterPlanningStats stats, final List<Integer> ids) {
            this.topDocs = topDocs;
            this.stats = stats;
            this.ids = ids;
        }
    }

    // -------------------------------------------------------------------- tests

    @SneakyThrows
    public void testZeroMatchesSelectsMatchNone() {
        withIndex(20, new int[0], (searcher, vectors) -> {
            final Result r = run(searcher, randomVector(DIM), noMatchFilter(), null);
            assertEquals(KNNFilterExecutionStrategy.MATCH_NONE, r.stats.getStrategy());
            assertEquals(0, r.topDocs.scoreDocs.length);
            assertFalse(r.stats.isAnnDelegateExecuted());
        });
    }

    @SneakyThrows
    public void testOneMatchSelectsExact() {
        withIndex(20, new int[0], (searcher, vectors) -> {
            final float[] q = randomVector(DIM);
            final Result r = run(searcher, q, rangeFilter(0, 1), null);
            assertEquals(KNNFilterExecutionStrategy.EXACT, r.stats.getStrategy());
            assertEquals(1, r.stats.getObservedMatches());
            assertEquals(List.of(0), r.ids); // only id 0 matches
            assertEquals(oracle(vectors, ids(0, 1), q, 10), scores(r.topDocs));
        });
    }

    @SneakyThrows
    public void testExactlyThresholdSelectsExact() {
        withIndex(20, new int[0], (searcher, vectors) -> {
            final float[] q = randomVector(DIM);
            final Result r = run(searcher, q, rangeFilter(0, THRESHOLD), null);
            assertEquals(KNNFilterExecutionStrategy.EXACT, r.stats.getStrategy());
            assertEquals(THRESHOLD, r.stats.getObservedMatches());
            assertFalse(r.stats.isThresholdExceeded());
            assertEquals(THRESHOLD, r.stats.getFilterAdvances()); // no early stop; counts exactly threshold
            assertEquals(oracle(vectors, ids(0, THRESHOLD), q, 10), scores(r.topDocs));
        });
    }

    @SneakyThrows
    public void testThresholdPlusOneSelectsApproximate() {
        withIndex(20, new int[0], (searcher, vectors) -> {
            final Result r = run(searcher, randomVector(DIM), rangeFilter(0, THRESHOLD + 1), null);
            assertEquals(KNNFilterExecutionStrategy.APPROXIMATE, r.stats.getStrategy());
            assertTrue(r.stats.isThresholdExceeded());
            assertEquals(THRESHOLD + 1, r.stats.getObservedMatches());
            assertTrue(r.stats.isAnnDelegateExecuted());
        });
    }

    @SneakyThrows
    public void testLargeFilterStopsEarlyAtThresholdPlusOne() {
        withIndex(100, new int[0], (searcher, vectors) -> {
            // 40 docs match but bounded counting must stop after threshold + 1 advances.
            final Result r = run(searcher, randomVector(DIM), rangeFilter(0, 40), null);
            assertEquals(KNNFilterExecutionStrategy.APPROXIMATE, r.stats.getStrategy());
            assertEquals(THRESHOLD + 1, r.stats.getObservedMatches());
            assertEquals(THRESHOLD + 1, r.stats.getFilterAdvances()); // did NOT scan all 40
            assertTrue(r.stats.isThresholdExceeded());
        });
    }

    @SneakyThrows
    public void testDeletedMatchingDocsExcludedFromExact() {
        withIndex(20, new int[] { 1, 3 }, (searcher, vectors) -> {
            final float[] q = randomVector(DIM);
            // ids [0,5) match but 1 and 3 are deleted -> 3 live matches, still EXACT.
            final Result r = run(searcher, q, rangeFilter(0, THRESHOLD), null);
            assertEquals(KNNFilterExecutionStrategy.EXACT, r.stats.getStrategy());
            assertEquals(3, r.stats.getObservedMatches());
            assertEquals(oracle(vectors, ids(0, THRESHOLD, 1, 3), q, 10), scores(r.topDocs));
        });
    }

    @SneakyThrows
    public void testExactMatchesOracleMultiSegment() {
        withIndex(30, new int[0], (searcher, vectors) -> {
            final float[] q = randomVector(DIM);
            // 4 live matches across (potentially) multiple segments; exact must preserve segment identity.
            final Result r = run(searcher, q, rangeFilter(0, 4), null);
            assertEquals(KNNFilterExecutionStrategy.EXACT, r.stats.getStrategy());
            assertEquals(oracle(vectors, ids(0, 4), q, 10), scores(r.topDocs));
            assertEquals(ids(0, 4), sorted(r.ids)); // exactly the filtered ids, correctly mapped via docBase
        });
    }

    @SneakyThrows
    public void testExactPathEvaluatesFilterOnlyOnce() {
        withIndex(20, new int[0], (searcher, vectors) -> {
            final AtomicInteger scorerCreations = new AtomicInteger();
            final Query countingFilter = new CountingFilterQuery(rangeFilter(0, THRESHOLD), scorerCreations);
            final Result r = run(searcher, randomVector(DIM), countingFilter, null);
            assertEquals(KNNFilterExecutionStrategy.EXACT, r.stats.getStrategy());
            // The filter scorer is created once per matched leaf during bounded counting and NOT again
            // for the exact scan (which reuses the collected doc ids).
            assertEquals(r.stats.getFilterScorersCreated(), scorerCreations.get());
            assertTrue("filter must be evaluated at least once", scorerCreations.get() >= 1);
        });
    }

    // -------------------------------------------------------------------- harness

    @FunctionalInterface
    private interface IndexConsumer {
        void accept(IndexSearcher searcher, List<float[]> vectors) throws IOException;
    }

    private void withIndex(final int docCount, final int[] deletedIds, final IndexConsumer consumer) throws IOException {
        final List<float[]> vectors = new ArrayList<>();
        for (int i = 0; i < docCount; i++) {
            vectors.add(randomVector(DIM));
        }
        try (Directory directory = newDirectory()) {
            try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
                for (int id = 0; id < docCount; id++) {
                    final Document document = new Document();
                    document.add(new KnnFloatVectorField(FIELD, vectors.get(id), SIMILARITY));
                    document.add(new IntPoint(ID_POINT, id));
                    document.add(new StoredField(ID_STORED, String.valueOf(id)));
                    writer.addDocument(document);
                    writer.commit(); // per-doc commit => multiple segments
                }
                for (final int deletedId : deletedIds) {
                    writer.deleteDocuments(IntPoint.newExactQuery(ID_POINT, deletedId));
                }
                writer.commit();
            }
            try (IndexReader reader = DirectoryReader.open(directory)) {
                try (MockedStatic<ModelDao.OpenSearchKNNModelDao> mocked = Mockito.mockStatic(ModelDao.OpenSearchKNNModelDao.class)) {
                    mocked.when(ModelDao.OpenSearchKNNModelDao::getInstance).thenReturn(mock(ModelDao.OpenSearchKNNModelDao.class));
                    final IndexSearcher searcher = newSearcher(reader, true, false);
                    // Disable the query cache so bounded-count filter evaluations are deterministic
                    // (no cache wrapper re-invoking scorerSupplier) and not order-sensitive.
                    searcher.setQueryCache(null);
                    consumer.accept(searcher, vectors);
                }
            }
        }
    }

    private Result run(final IndexSearcher searcher, final float[] queryVector, final Query filter, final Query ignored)
        throws IOException {
        final int k = 10;
        final Query annDelegate = new OSKnnFloatVectorQuery(FIELD, queryVector, EF_SEARCH, filter, k, RescoreContext.NO_RESCORE_NEEDED);
        final KNNFilterPlanningStats stats = new KNNFilterPlanningStats();
        final ExactSearcher exactSearcher = new ExactSearcher(ModelDao.OpenSearchKNNModelDao.getInstance());
        final BoundedExactKnnFloatVectorQuery query = new BoundedExactKnnFloatVectorQuery(
            annDelegate,
            filter,
            FIELD,
            k,
            queryVector,
            0,
            exactSearcher,
            stats
        );
        final TopDocs topDocs = searcher.search(query, k);
        return new Result(topDocs, stats, idsOf(searcher, topDocs.scoreDocs));
    }

    private Query rangeFilter(final int loInclusive, final int hiExclusive) {
        return IntPoint.newRangeQuery(ID_POINT, loInclusive, hiExclusive - 1);
    }

    private Query noMatchFilter() {
        return IntPoint.newRangeQuery(ID_POINT, 1_000_000, 2_000_000);
    }

    private List<Integer> ids(final int loInclusive, final int hiExclusive, final int... excluded) {
        final List<Integer> ids = new ArrayList<>();
        for (int id = loInclusive; id < hiExclusive; id++) {
            boolean isExcluded = false;
            for (final int e : excluded) {
                if (e == id) {
                    isExcluded = true;
                    break;
                }
            }
            if (!isExcluded) {
                ids.add(id);
            }
        }
        return ids;
    }

    private List<Integer> sorted(final List<Integer> ids) {
        final List<Integer> copy = new ArrayList<>(ids);
        copy.sort(Comparator.naturalOrder());
        return copy;
    }

    private List<Float> scores(final TopDocs topDocs) {
        final List<Float> scores = new ArrayList<>();
        for (final ScoreDoc scoreDoc : topDocs.scoreDocs) {
            scores.add(scoreDoc.score);
        }
        return scores;
    }

    private List<Float> oracle(final List<float[]> vectors, final List<Integer> allowedIds, final float[] q, final int k) {
        final List<Float> scores = new ArrayList<>();
        for (final int id : allowedIds) {
            scores.add(SIMILARITY.compare(q, vectors.get(id)));
        }
        scores.sort(Comparator.reverseOrder());
        return scores.subList(0, Math.min(k, scores.size()));
    }

    private List<Integer> idsOf(final IndexSearcher searcher, final ScoreDoc[] scoreDocs) throws IOException {
        final StoredFields storedFields = searcher.storedFields();
        final List<Integer> ids = new ArrayList<>(scoreDocs.length);
        for (final ScoreDoc scoreDoc : scoreDocs) {
            ids.add(Integer.parseInt(storedFields.document(scoreDoc.doc).get(ID_STORED)));
        }
        return ids;
    }

    /** Filter wrapper that counts how many times a per-leaf {@link Scorer} is created for it. */
    private static final class CountingFilterQuery extends Query {
        private final Query inner;
        private final AtomicInteger scorerCreations;

        CountingFilterQuery(final Query inner, final AtomicInteger scorerCreations) {
            this.inner = inner;
            this.scorerCreations = scorerCreations;
        }

        @Override
        public Weight createWeight(final IndexSearcher searcher, final ScoreMode scoreMode, final float boost) throws IOException {
            final Weight innerWeight = inner.createWeight(searcher, scoreMode, boost);
            return new Weight(this) {
                @Override
                public org.apache.lucene.search.Explanation explain(final LeafReaderContext context, final int doc) throws IOException {
                    return innerWeight.explain(context, doc);
                }

                // Weight.scorer(...) is final in Lucene 10 and routes to scorerSupplier(...); count here.
                @Override
                public org.apache.lucene.search.ScorerSupplier scorerSupplier(final LeafReaderContext context) throws IOException {
                    final org.apache.lucene.search.ScorerSupplier supplier = innerWeight.scorerSupplier(context);
                    if (supplier != null) {
                        scorerCreations.incrementAndGet();
                    }
                    return supplier;
                }

                @Override
                public boolean isCacheable(final LeafReaderContext ctx) {
                    return false;
                }
            };
        }

        @Override
        public Query rewrite(final IndexSearcher indexSearcher) throws IOException {
            return this; // keep identity so bounded counting sees this weight
        }

        @Override
        public String toString(final String field) {
            return "CountingFilterQuery(" + inner.toString(field) + ")";
        }

        @Override
        public void visit(final QueryVisitor visitor) {
            visitor.visitLeaf(this);
        }

        @Override
        public boolean equals(final Object obj) {
            return sameClassAs(obj) && inner.equals(((CountingFilterQuery) obj).inner);
        }

        @Override
        public int hashCode() {
            return classHash() * 31 + inner.hashCode();
        }
    }
}
