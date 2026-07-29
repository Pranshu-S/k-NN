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
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.List;

/**
 * Unit tests for the benchmark-only {@link FilterLocalityFeatures} extractor: segment distribution,
 * doc-id scatter, deterministic bounded vector sampling, deletions, and the contiguous-vs-scattered
 * distinction that the adversarial benchmark relies on.
 */
public class FilterLocalityFeaturesTests extends OpenSearchTestCase {

    private static final String FIELD = "vec";
    private static final String IDF = "idf";
    private static final int DIM = 4;

    @FunctionalInterface
    private interface WithSearcher {
        void run(IndexSearcher searcher) throws IOException;
    }

    private void build(final Directory dir, final int n, final int commitEvery, final int[] deleted, final WithSearcher body)
        throws IOException {
        build(dir, n, commitEvery, deleted, false, body);
    }

    /** Adds {@code n} docs id=0..n-1, vector = idBucket-based, committing per {@code commitEvery} docs. */
    private void build(
        final Directory dir,
        final int n,
        final int commitEvery,
        final int[] deleted,
        final boolean noMerge,
        final WithSearcher body
    ) throws IOException {
        final IndexWriterConfig config = new IndexWriterConfig();
        if (noMerge) {
            config.setMergePolicy(org.apache.lucene.index.NoMergePolicy.INSTANCE);
        }
        try (IndexWriter w = new IndexWriter(dir, config)) {
            for (int i = 0; i < n; i++) {
                final Document d = new Document();
                final float[] v = new float[DIM];
                v[0] = i; // deterministic, distinct
                d.add(new KnnFloatVectorField(FIELD, v, VectorSimilarityFunction.EUCLIDEAN));
                d.add(new IntPoint(IDF, i));
                w.addDocument(d);
                if ((i + 1) % commitEvery == 0) {
                    w.commit();
                }
            }
            for (final int del : deleted) {
                w.deleteDocuments(IntPoint.newExactQuery(IDF, del));
            }
            w.commit();
        }
        try (IndexReader r = DirectoryReader.open(dir)) {
            body.run(newSearcher(r, true, false));
        }
    }

    private FilterLocalityFeatures.ScatterFeatures scatterFor(final IndexSearcher s, final Query filter) throws IOException {
        final Weight w = s.createWeight(s.rewrite(filter), ScoreMode.COMPLETE_NO_SCORES, 1f);
        final List<FilterLocalityFeatures.LeafSample> sample = FilterLocalityFeatures.collectSample(s, w, 100_000);
        return FilterLocalityFeatures.scatterFeatures(sample, s.getIndexReader().leaves().size());
    }

    @SneakyThrows
    public void testZeroMatches() {
        try (Directory dir = newDirectory()) {
            build(dir, 50, 50, new int[0], (s) -> {
                final FilterLocalityFeatures.ScatterFeatures f = scatterFor(s, IntPoint.newRangeQuery(IDF, 1_000, 2_000));
                assertEquals(0, f.sampledMatches);
                assertEquals(0, f.numLeavesWithMatches);
            });
        }
    }

    @SneakyThrows
    public void testSingleMatchSingleLeaf() {
        try (Directory dir = newDirectory()) {
            build(dir, 50, 50, new int[0], (s) -> {
                final FilterLocalityFeatures.ScatterFeatures f = scatterFor(s, IntPoint.newExactQuery(IDF, 7));
                assertEquals(1, f.sampledMatches);
                assertEquals(1, f.numLeavesWithMatches);
                assertEquals(1.0, f.maxLeafConcentration, 1e-9);
                assertEquals(0.0, f.weightedNormalizedSpan, 1e-9); // single doc, no span
            });
        }
    }

    @SneakyThrows
    public void testContiguousVsScatteredDocIds() {
        try (Directory dir = newDirectory()) {
            // single segment so doc ids == build ids
            build(dir, 1000, 1000, new int[0], (s) -> {
                // contiguous ids [0,100)
                final FilterLocalityFeatures.ScatterFeatures contig = scatterFor(s, IntPoint.newRangeQuery(IDF, 0, 99));
                // scattered ids: multiples of 10 -> gaps of 10
                final Query scatter = IntPoint.newSetQuery(IDF, java.util.stream.IntStream.range(0, 100).map(i -> i * 10).boxed().toList());
                final FilterLocalityFeatures.ScatterFeatures scat = scatterFor(s, scatter);

                // Contiguous: mean gap ~1, most matches in runs, small normalized span.
                assertEquals(1.0, contig.weightedMeanGap, 1e-6);
                assertTrue("contiguous should be mostly in runs", contig.fracInContiguousRuns > 0.95);
                // Scattered: mean gap ~10, no contiguous runs, larger span.
                assertTrue("scattered mean gap should be ~10", scat.weightedMeanGap > 5.0);
                assertEquals(0.0, scat.fracInContiguousRuns, 1e-9);
                assertTrue(scat.weightedNormalizedSpan > contig.weightedNormalizedSpan);
            });
        }
    }

    @SneakyThrows
    public void testMultiLeafSegmentDistribution() {
        try (Directory dir = newDirectory()) {
            // commit every 100 docs with no merging -> 10 segments; filter spans all -> matches in many leaves
            build(dir, 1000, 100, new int[0], true, (s) -> {
                final FilterLocalityFeatures.ScatterFeatures f = scatterFor(s, IntPoint.newRangeQuery(IDF, 0, 999));
                assertTrue("expected multiple leaves with matches", f.numLeavesWithMatches >= 2);
                assertTrue("entropy should be high for even distribution", f.leafEntropy > 0.5);
                assertTrue(f.maxLeafConcentration < 1.0);
            });
        }
    }

    @SneakyThrows
    public void testDeletedDocsExcluded() {
        try (Directory dir = newDirectory()) {
            build(dir, 100, 100, new int[] { 2, 5, 8 }, (s) -> {
                final FilterLocalityFeatures.ScatterFeatures f = scatterFor(s, IntPoint.newRangeQuery(IDF, 0, 9));
                // ids 0..9 minus deleted {2,5,8} = 7 live matches
                assertEquals(7, f.sampledMatches);
            });
        }
    }

    @SneakyThrows
    public void testVectorDispersionSampleBoundAndDeterminism() {
        try (Directory dir = newDirectory()) {
            build(dir, 1000, 1000, new int[0], (s) -> {
                final Weight w = s.createWeight(s.rewrite(IntPoint.newRangeQuery(IDF, 0, 499)), ScoreMode.COMPLETE_NO_SCORES, 1f);
                final List<FilterLocalityFeatures.LeafSample> sample = FilterLocalityFeatures.collectSample(s, w, 100_000);
                final float[] q = new float[DIM];
                q[0] = 0;
                final FilterLocalityFeatures.VectorDispersion d1 = FilterLocalityFeatures.vectorDispersion(sample, FIELD, q, 32);
                final FilterLocalityFeatures.VectorDispersion d2 = FilterLocalityFeatures.vectorDispersion(sample, FIELD, q, 32);
                assertTrue("sample size bounded", d1.sampleSize <= 32);
                assertEquals("sampling deterministic", d1.meanDistToQuery, d2.meanDistToQuery, 1e-9);
                // vectors are v[0]=id, query at 0 -> distances increase with id; dispersion > 0
                assertTrue(d1.meanDistToQuery > 0);
                assertTrue(d1.meanDistToCentroid > 0);
            });
        }
    }

    @SneakyThrows
    public void testCollectSampleRespectsCap() {
        try (Directory dir = newDirectory()) {
            build(dir, 1000, 1000, new int[0], (s) -> {
                final Weight w = s.createWeight(s.rewrite(IntPoint.newRangeQuery(IDF, 0, 999)), ScoreMode.COMPLETE_NO_SCORES, 1f);
                final List<FilterLocalityFeatures.LeafSample> sample = FilterLocalityFeatures.collectSample(s, w, 50);
                long total = 0;
                for (final FilterLocalityFeatures.LeafSample ls : sample) {
                    total += ls.count;
                }
                assertEquals("cap honoured", 50, total);
            });
        }
    }
}
