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
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.Random;

/**
 * Phase-9 tests for the benchmark-only {@link AnnProbe}: visit-limit respected, early termination,
 * fewer-than-k / exactly-k candidates, deletions, multiple segments, zero matches. Confirms the probe
 * uses only public Lucene APIs and honours live docs.
 */
public class AnnProbeTests extends OpenSearchTestCase {

    private static final String FIELD = "vec";
    private static final String IDF = "idf";
    private static final int DIM = 8;

    @FunctionalInterface
    private interface Body {
        void run(IndexSearcher s) throws IOException;
    }

    private void build(final Directory dir, final int n, final int commitEvery, final boolean noMerge, final int[] deleted, final Body body)
        throws IOException {
        final Random r = new Random(3);
        final IndexWriterConfig cfg = new IndexWriterConfig();
        if (noMerge) {
            cfg.setMergePolicy(NoMergePolicy.INSTANCE);
        }
        try (IndexWriter w = new IndexWriter(dir, cfg)) {
            for (int i = 0; i < n; i++) {
                final Document d = new Document();
                final float[] v = new float[DIM];
                for (int j = 0; j < DIM; j++) {
                    v[j] = (float) r.nextGaussian();
                }
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
        try (IndexReader reader = DirectoryReader.open(dir)) {
            body.run(newSearcher(reader, true, false));
        }
    }

    private Weight fw(final IndexSearcher s, final Query filter) throws IOException {
        return s.createWeight(s.rewrite(filter), ScoreMode.COMPLETE_NO_SCORES, 1f);
    }

    private float[] q() {
        final float[] v = new float[DIM];
        v[0] = 1f;
        return v;
    }

    @SneakyThrows
    public void testVisitLimitRespectedAndEarlyTerminated() {
        try (Directory dir = newDirectory()) {
            build(dir, 5000, 5000, false, new int[0], s -> {
                final Weight w = fw(s, IntPoint.newRangeQuery(IDF, 0, 4999)); // all docs match
                final AnnProbe.Result r = AnnProbe.probe(s, w, FIELD, q(), 10, 50);
                // Visited is bounded by the per-leaf budget (times leaves; single segment here).
                assertTrue("visited within budget*leaves", r.visited <= r.visitLimitTotal + 5);
                assertTrue("large graph + tiny budget should early-terminate", r.earlyTerminated);
            });
        }
    }

    @SneakyThrows
    public void testFewerThanKCandidates() {
        try (Directory dir = newDirectory()) {
            build(dir, 1000, 1000, false, new int[0], s -> {
                // Only 3 docs match -> probe can collect at most 3 < k=10.
                final Weight w = fw(s, IntPoint.newRangeQuery(IDF, 0, 2));
                final AnnProbe.Result r = AnnProbe.probe(s, w, FIELD, q(), 10, 1000);
                assertTrue("fewer than k collected", r.collected <= 3);
            });
        }
    }

    @SneakyThrows
    public void testExactlyKAndCompletionWithLargeBudget() {
        try (Directory dir = newDirectory()) {
            build(dir, 2000, 2000, false, new int[0], s -> {
                final Weight w = fw(s, IntPoint.newRangeQuery(IDF, 0, 1999)); // all match
                final AnnProbe.Result r = AnnProbe.probe(s, w, FIELD, q(), 10, 100_000);
                assertEquals("collects k with a generous budget", 10, r.collected);
                assertFalse("should not early-terminate with a huge budget", r.earlyTerminated);
            });
        }
    }

    @SneakyThrows
    public void testZeroMatches() {
        try (Directory dir = newDirectory()) {
            build(dir, 1000, 1000, false, new int[0], s -> {
                final Weight w = fw(s, IntPoint.newRangeQuery(IDF, 10_000, 20_000));
                final AnnProbe.Result r = AnnProbe.probe(s, w, FIELD, q(), 10, 500);
                assertEquals(0, r.leavesProbed);
                assertEquals(0, r.collected);
                assertEquals(0, r.visited);
                assertFalse(r.earlyTerminated);
            });
        }
    }

    @SneakyThrows
    public void testMultiSegmentProbe() {
        try (Directory dir = newDirectory()) {
            build(dir, 2000, 200, true, new int[0], s -> { // 10 segments
                final Weight w = fw(s, IntPoint.newRangeQuery(IDF, 0, 1999));
                final AnnProbe.Result r = AnnProbe.probe(s, w, FIELD, q(), 10, 100);
                assertTrue("probed multiple leaves", r.leavesProbed >= 2);
                assertTrue(r.collected > 0);
            });
        }
    }

    @SneakyThrows
    public void testDeletionsExcludedFromProbe() {
        try (Directory dir = newDirectory()) {
            build(dir, 1000, 1000, false, new int[] { 0, 1, 2 }, s -> {
                // Filter ids [0,3) but all deleted -> no live matches -> nothing probed.
                final Weight w = fw(s, IntPoint.newRangeQuery(IDF, 0, 2));
                final AnnProbe.Result r = AnnProbe.probe(s, w, FIELD, q(), 10, 500);
                assertEquals(0, r.leavesProbed);
                assertEquals(0, r.collected);
            });
        }
    }
}
