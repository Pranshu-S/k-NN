/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase.randomVector;
import static org.mockito.Mockito.mock;

/**
 * Lucene-level integration tests for {@link ExactFilteredKNNVectorQuery}. They build small
 * multi-segment indices of FLOAT vectors plus a filterable {@code id} field and assert exact
 * filtered top-k results against a brute-force oracle, including deletions and segment merges.
 */
public class ExactFilteredKNNVectorQueryTests extends OpenSearchTestCase {

    private static final String FIELD_NAME = "vector-field";
    private static final String ID_FIELD = "id";
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.EUCLIDEAN;

    private List<float[]> generateVectors(final int count, final int dimension) {
        final List<float[]> vectors = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            vectors.add(randomVector(dimension));
        }
        return vectors;
    }

    /** Adds one document per vector (committing per doc so the index spans multiple segments). */
    private void addDocuments(final List<float[]> vectors, final Directory directory, final int[] deletedIds) throws IOException {
        try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
            for (int id = 0; id < vectors.size(); id++) {
                final Document document = new Document();
                document.add(new KnnFloatVectorField(FIELD_NAME, vectors.get(id), SIMILARITY));
                document.add(new IntPoint(ID_FIELD, id));
                writer.addDocument(document);
                writer.commit();
            }
            for (final int deletedId : deletedIds) {
                writer.deleteDocuments(IntPoint.newExactQuery(ID_FIELD, deletedId));
            }
            writer.commit();
        }
    }

    /** Brute-force oracle: top-k similarity scores over the allowed (filtered, live) doc ids. */
    private List<Float> oracleScores(final List<float[]> vectors, final List<Integer> allowedIds, final float[] queryVector, final int k) {
        final List<Float> scores = new ArrayList<>();
        for (final int id : allowedIds) {
            scores.add(SIMILARITY.compare(queryVector, vectors.get(id)));
        }
        scores.sort(Collections.reverseOrder());
        return scores.subList(0, Math.min(k, scores.size()));
    }

    private List<Integer> idsInRange(final int loInclusive, final int hiExclusive, final int... excluded) {
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

    private List<Float> runAndCollectScores(final Directory directory, final float[] queryVector, final Query filter, final int k)
        throws IOException {
        try (IndexReader reader = DirectoryReader.open(directory)) {
            try (MockedStatic<ModelDao.OpenSearchKNNModelDao> mocked = Mockito.mockStatic(ModelDao.OpenSearchKNNModelDao.class)) {
                mocked.when(ModelDao.OpenSearchKNNModelDao::getInstance).thenReturn(mock(ModelDao.OpenSearchKNNModelDao.class));
                final IndexSearcher searcher = newSearcher(reader, true, false);
                final ExactFilteredKNNVectorQuery query = new ExactFilteredKNNVectorQuery(filter, FIELD_NAME, k, queryVector, 0);
                final TopDocs topDocs = searcher.search(query, Math.max(k, 1));
                final List<Float> scores = new ArrayList<>();
                for (final ScoreDoc scoreDoc : topDocs.scoreDocs) {
                    scores.add(scoreDoc.score);
                }
                return scores;
            }
        }
    }

    @SneakyThrows
    public void testFilterMatchesSubsetBelowK() {
        final int dimension = 4;
        final int k = 5;
        try (Directory directory = newDirectory()) {
            final List<float[]> vectors = generateVectors(20, dimension);
            addDocuments(vectors, directory, new int[0]);
            final float[] queryVector = randomVector(dimension);
            // Filter selects ids [0, 12); a subset larger than k across multiple segments.
            final Query filter = IntPoint.newRangeQuery(ID_FIELD, 0, 11);
            final List<Float> actual = runAndCollectScores(directory, queryVector, filter, k);
            assertEquals(k, actual.size());
            assertEquals(oracleScores(vectors, idsInRange(0, 12), queryVector, k), actual);
        }
    }

    @SneakyThrows
    public void testFilterMatchesFewerThanK() {
        final int dimension = 4;
        final int k = 10;
        try (Directory directory = newDirectory()) {
            final List<float[]> vectors = generateVectors(20, dimension);
            addDocuments(vectors, directory, new int[0]);
            final float[] queryVector = randomVector(dimension);
            // Only 3 docs match -> all 3 returned (fewer than k).
            final Query filter = IntPoint.newRangeQuery(ID_FIELD, 0, 2);
            final List<Float> actual = runAndCollectScores(directory, queryVector, filter, k);
            assertEquals(3, actual.size());
            assertEquals(oracleScores(vectors, idsInRange(0, 3), queryVector, k), actual);
        }
    }

    @SneakyThrows
    public void testFilterMatchesExactlyK() {
        final int dimension = 4;
        final int k = 6;
        try (Directory directory = newDirectory()) {
            final List<float[]> vectors = generateVectors(20, dimension);
            addDocuments(vectors, directory, new int[0]);
            final float[] queryVector = randomVector(dimension);
            // Exactly k docs match.
            final Query filter = IntPoint.newRangeQuery(ID_FIELD, 0, 5);
            final List<Float> actual = runAndCollectScores(directory, queryVector, filter, k);
            assertEquals(k, actual.size());
            assertEquals(oracleScores(vectors, idsInRange(0, 6), queryVector, k), actual);
        }
    }

    @SneakyThrows
    public void testZeroMatchesReturnsNoResults() {
        final int dimension = 4;
        final int k = 5;
        try (Directory directory = newDirectory()) {
            final List<float[]> vectors = generateVectors(20, dimension);
            addDocuments(vectors, directory, new int[0]);
            final float[] queryVector = randomVector(dimension);
            // No document has an id in this range.
            final Query filter = IntPoint.newRangeQuery(ID_FIELD, 1_000, 2_000);
            final List<Float> actual = runAndCollectScores(directory, queryVector, filter, k);
            assertEquals(0, actual.size());
        }
    }

    @SneakyThrows
    public void testDeletedMatchingDocsAreIgnored() {
        final int dimension = 4;
        final int k = 5;
        try (Directory directory = newDirectory()) {
            final List<float[]> vectors = generateVectors(20, dimension);
            // Delete two docs that fall inside the filter range.
            final int[] deleted = new int[] { 2, 7 };
            addDocuments(vectors, directory, deleted);
            final float[] queryVector = randomVector(dimension);
            final Query filter = IntPoint.newRangeQuery(ID_FIELD, 0, 11);
            final List<Float> actual = runAndCollectScores(directory, queryVector, filter, k);
            assertEquals(k, actual.size());
            // Oracle excludes the deleted ids.
            assertEquals(oracleScores(vectors, idsInRange(0, 12, 2, 7), queryVector, k), actual);
        }
    }

    @SneakyThrows
    public void testResultsStableAcrossSegmentMerge() {
        final int dimension = 4;
        final int k = 5;
        final List<float[]> vectors = generateVectors(20, dimension);
        final float[] queryVector = randomVector(dimension);
        final Query filter = IntPoint.newRangeQuery(ID_FIELD, 0, 11);

        try (Directory multiSegment = newDirectory(); Directory merged = newDirectory()) {
            addDocuments(vectors, multiSegment, new int[0]);
            addDocuments(vectors, merged, new int[0]);
            // Force the second index down to a single segment.
            try (IndexWriter writer = new IndexWriter(merged, new IndexWriterConfig())) {
                writer.forceMerge(1);
            }

            final List<Float> multiSegmentScores = runAndCollectScores(multiSegment, queryVector, filter, k);
            final List<Float> mergedScores = runAndCollectScores(merged, queryVector, filter, k);

            assertEquals(k, multiSegmentScores.size());
            assertEquals(multiSegmentScores, mergedScores);
            assertEquals(oracleScores(vectors, idsInRange(0, 12), queryVector, k), mergedScores);
        }
    }
}
