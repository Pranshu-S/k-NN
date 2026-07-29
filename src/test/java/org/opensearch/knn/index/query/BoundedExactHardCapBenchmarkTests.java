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
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import static org.mockito.Mockito.mock;

/**
 * Phase 1-2 hard-cap calibration. The current rule {@code candidateLimit = max(1000, 10*k)} lets a large
 * {@code k} trigger exact scans of up to 100,000 candidates. This measures worst-case EXACT filtered
 * latency (random/scattered filter → worst random-access cost) across (dimension, candidate cardinality),
 * then reports, for each candidate hard cap, the worst p95 / max exact latency it would still permit — so
 * a cap can be chosen against a "maximum acceptable p95 exact latency" target rather than benchmark wins.
 *
 * <p>The exact path performs no {@code QueryTimeout} check mid-scan, so an uncapped large-cardinality
 * scan is an uninterruptible cost — the reason the cap exists.
 */
public class BoundedExactHardCapBenchmarkTests extends OpenSearchTestCase {

    private static final String FIELD = "vec";
    private static final String RND = "rnd";
    private static final VectorSimilarityFunction SIM = VectorSimilarityFunction.EUCLIDEAN;
    private static final int CORPUS = 100_000;
    private static final int K = 10;
    private static final int NUM_CLUSTERS = 100;
    private static final int NQ = 20;
    private static final long SEED = 42L;
    private static final int[] DIMS = { 128, 384, 768 };
    private static final int[] CARDS = { 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000 };
    private static final int[] CANDIDATE_CAPS = { 2_500, 5_000, 10_000, 25_000 };

    private static final class Cell {
        final int dim, card;
        final double exP50, exP95, exMax;

        Cell(int dim, int card, double exP50, double exP95, double exMax) {
            this.dim = dim;
            this.card = card;
            this.exP50 = exP50;
            this.exP95 = exP95;
            this.exMax = exMax;
        }
    }

    private final List<Cell> cells = new ArrayList<>();

    @SneakyThrows
    public void testHardCapCalibration() {
        final StringBuilder out = new StringBuilder(
            "\n\n$$$$$$ HARD-CAP CALIBRATION: EXACT latency vs cardinality (100k, random filter) $$$$$$\n"
        );
        out.append(env());
        for (final int dim : DIMS) {
            try (Directory dir = newFSDirectory(createTempDir())) {
                buildIndex(dir, dim);
                withSearcher(dir, s -> {
                    final List<float[]> queries = new ArrayList<>();
                    for (int i = 0; i < NQ; i++) {
                        queries.add(unit(new Random(SEED + dim * 131L + i), dim));
                    }
                    for (final int c : CARDS) {
                        cells.add(measure(s, dim, c, IntPoint.newRangeQuery(RND, 0, c - 1), queries));
                    }
                });
            }
        }
        out.append(renderCells());
        out.append(renderCapComparison());
        out.append("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
        System.out.println(out);
    }

    private Cell measure(final IndexSearcher s, final int dim, final int card, final Query filter, final List<float[]> queries)
        throws IOException {
        final long[] ex = new long[NQ];
        for (int i = 0; i < 3; i++) {
            searchExact(s, queries.get(i), filter);
        }
        for (int qi = 0; qi < NQ; qi++) {
            final long t = System.nanoTime();
            searchExact(s, queries.get(qi), filter);
            ex[qi] = System.nanoTime() - t;
        }
        return new Cell(dim, card, pct(ex, 50), pct(ex, 95), max(ex) / 1_000.0);
    }

    private String renderCells() {
        final StringBuilder sb = new StringBuilder("\n### Exact latency by (dim, candidate cardinality) — random filter, k=10\n");
        sb.append("| Dim | Cardinality | Exact p50 µs | Exact p95 µs | Exact MAX µs | int[] mem (bytes) |\n");
        sb.append("| --: | ----------: | -----------: | -----------: | -----------: | ----------------: |\n");
        for (final Cell c : cells) {
            sb.append(
                String.format(
                    Locale.ROOT,
                    "| %3d | %11d | %12.1f | %12.1f | %12.1f | %17d |%n",
                    c.dim,
                    c.card,
                    c.exP50,
                    c.exP95,
                    c.exMax,
                    16L + (long) c.card * Integer.BYTES
                )
            );
        }
        return sb.toString();
    }

    private String renderCapComparison() {
        final StringBuilder sb = new StringBuilder(
            "\n### Candidate hard caps: worst exact latency still permitted (max over dims of the largest card <= cap)\n"
        );
        sb.append("| Hard cap | worst permitted card | worst p95 µs | worst MAX µs |\n");
        sb.append("| -------: | -------------------: | -----------: | -----------: |\n");
        // include "no cap" baseline
        appendCapRow(sb, Integer.MAX_VALUE, "none (current)");
        for (final int cap : CANDIDATE_CAPS) {
            appendCapRow(sb, cap, String.valueOf(cap));
        }
        sb.append("\n(A cap of C limits exact to <= C candidates; worst permitted card = largest tested cardinality <= C.)\n");
        return sb.toString();
    }

    private void appendCapRow(final StringBuilder sb, final int cap, final String label) {
        // For each dim, find the largest tested cardinality <= cap; take the worst (max) p95/max across dims.
        double worstP95 = 0, worstMax = 0;
        int worstCard = 0;
        for (final int dim : DIMS) {
            Cell best = null;
            for (final Cell c : cells) {
                if (c.dim == dim && c.card <= cap && (best == null || c.card > best.card)) {
                    best = c;
                }
            }
            if (best != null) {
                if (best.exP95 > worstP95) {
                    worstP95 = best.exP95;
                }
                if (best.exMax > worstMax) {
                    worstMax = best.exMax;
                }
                worstCard = Math.max(worstCard, best.card);
            }
        }
        sb.append(String.format(Locale.ROOT, "| %-8s | %20d | %12.1f | %12.1f |%n", label, worstCard, worstP95, worstMax));
    }

    // ------------------------------------------------------------------ helpers

    private TopDocs searchExact(final IndexSearcher s, final float[] q, final Query f) throws IOException {
        return s.search(new ExactFilteredKNNVectorQuery(f, FIELD, K, q, 0), K);
    }

    private void buildIndex(final Directory dir, final int dim) throws IOException {
        final Random r = new Random(SEED + dim);
        final int clusterSize = CORPUS / NUM_CLUSTERS;
        final int[] perm = permutation(new Random(SEED + 55));
        try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
            for (int i = 0; i < CORPUS; i++) {
                final int cluster = Math.min(NUM_CLUSTERS - 1, i / clusterSize);
                final Document d = new Document();
                d.add(new KnnFloatVectorField(FIELD, perturb(center(cluster, dim), 0.1f, r), SIM));
                d.add(new IntPoint(RND, perm[i]));
                w.addDocument(d);
            }
            w.commit();
        }
    }

    private float[] center(final int c, final int dim) {
        return unit(new Random(SEED * 1_000_003L + c), dim);
    }

    private float[] unit(final Random r, final int dim) {
        final float[] v = new float[dim];
        double norm = 0;
        for (int i = 0; i < dim; i++) {
            v[i] = (float) r.nextGaussian();
            norm += (double) v[i] * v[i];
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < dim; i++) {
            v[i] /= (float) (norm == 0 ? 1 : norm);
        }
        return v;
    }

    private float[] perturb(final float[] base, final float scale, final Random r) {
        final float[] v = new float[base.length];
        double norm = 0;
        for (int i = 0; i < base.length; i++) {
            v[i] = base[i] + scale * (float) r.nextGaussian();
            norm += (double) v[i] * v[i];
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < base.length; i++) {
            v[i] /= (float) (norm == 0 ? 1 : norm);
        }
        return v;
    }

    private int[] permutation(final Random r) {
        final int[] p = new int[CORPUS];
        for (int i = 0; i < CORPUS; i++) {
            p[i] = i;
        }
        for (int i = CORPUS - 1; i > 0; i--) {
            final int j = r.nextInt(i + 1);
            final int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    @FunctionalInterface
    private interface SC {
        void run(IndexSearcher s) throws IOException;
    }

    private void withSearcher(final Directory dir, final SC body) throws IOException {
        try (IndexReader r = DirectoryReader.open(dir)) {
            try (MockedStatic<ModelDao.OpenSearchKNNModelDao> mocked = Mockito.mockStatic(ModelDao.OpenSearchKNNModelDao.class)) {
                mocked.when(ModelDao.OpenSearchKNNModelDao::getInstance).thenReturn(mock(ModelDao.OpenSearchKNNModelDao.class));
                final IndexSearcher s = newSearcher(r, true, false);
                s.setQueryCache(null);
                body.run(s);
            }
        }
    }

    @SuppressWarnings("unused")
    private long[] idSet(final TopDocs td) {
        final long[] ids = new long[td.scoreDocs.length];
        int i = 0;
        for (final ScoreDoc sd : td.scoreDocs) {
            ids[i++] = sd.doc;
        }
        return ids;
    }

    private String env() {
        final Runtime rt = Runtime.getRuntime();
        return String.format(
            Locale.ROOT,
            "env: jvm=%s, os=%s/%s, cores=%d, maxHeapMB=%d, queries=%d, seed=%d; exact does NO QueryTimeout check mid-scan%n",
            System.getProperty("java.version"),
            System.getProperty("os.name"),
            System.getProperty("os.arch"),
            rt.availableProcessors(),
            rt.maxMemory() / (1024 * 1024),
            NQ,
            SEED
        );
    }

    private static double pct(final long[] s, final int p) {
        final long[] c = s.clone();
        Arrays.sort(c);
        return c[Math.max(0, Math.min((int) Math.ceil(p / 100.0 * c.length) - 1, c.length - 1))] / 1_000.0;
    }

    private static long max(final long[] s) {
        long m = 0;
        for (final long v : s) {
            m = Math.max(m, v);
        }
        return m;
    }
}
