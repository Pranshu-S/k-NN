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
import org.opensearch.knn.index.query.lucenelib.OSKnnFloatVectorQuery;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

import static org.mockito.Mockito.mock;

/**
 * Iteration-6 production calibration: measures how worst-case EXACT filtered-search latency scales with
 * (dimension, cardinality) so a conservative exact-work bound can be chosen, and compares eligibility
 * Candidate A ({@code max(1000,10k)}) against Candidate B (dimension-aware {@code
 * maxExactDimensionOperations / dim}). The worst case for exact is a scattered (random) filter (random
 * memory access), so that is the primary measurement; a correlated filter is included for the benefit
 * view. Reports exact p50/p95/max, ANN p50, and ANN recall; then, for each candidate rule, the redirected
 * cardinality band and the worst exact latency it would introduce.
 */
public class BoundedExactCalibrationBenchmarkTests extends OpenSearchTestCase {

    private static final String FIELD = "vec";
    private static final String RND = "rnd";
    private static final String SEQ = "seq";
    private static final VectorSimilarityFunction SIM = VectorSimilarityFunction.EUCLIDEAN;
    private static final int CORPUS = 100_000;
    private static final int K = 10;
    private static final int EF = 100;
    private static final int NUM_CLUSTERS = 100;
    private static final int NQ = 30;
    private static final long SEED = 42L;
    private static final int[] DIMS = { 128, 384, 768 };
    private static final int[] CARDS = { 500, 1_000, 2_500, 5_000, 7_500, 10_000 };
    // Candidate-B constants to evaluate (total float ops budget for the exact scan).
    private static final long[] MAX_OPS = { 500_000L, 1_000_000L, 2_000_000L };
    private static final int HARD_CAP = 20_000;

    private static final class Cell {
        final int dim, card;
        final double exP50, exP95, exMax, annP50, annP95, recall;

        Cell(int dim, int card, double exP50, double exP95, double exMax, double annP50, double annP95, double recall) {
            this.dim = dim;
            this.card = card;
            this.exP50 = exP50;
            this.exP95 = exP95;
            this.exMax = exMax;
            this.annP50 = annP50;
            this.annP95 = annP95;
            this.recall = recall;
        }
    }

    private final List<Cell> cells = new ArrayList<>();

    @SneakyThrows
    public void testCalibration() {
        final StringBuilder out = new StringBuilder(
            "\n\n###### BOUNDED-EXACT CALIBRATION (100k, k=10, ef=100, random/worst-case filter) ######\n"
        );
        out.append(env());
        for (final int dim : DIMS) {
            try (Directory dir = newFSDirectory(createTempDir())) {
                buildIndex(dir, dim);
                withSearcher(dir, s -> {
                    final List<float[]> queries = new ArrayList<>();
                    final Random qr = new Random(SEED + dim);
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
        out.append(renderRuleComparison());
        out.append("################################################################################\n");
        System.out.println(out);
    }

    private Cell measure(final IndexSearcher s, final int dim, final int card, final Query filter, final List<float[]> queries)
        throws IOException {
        final long[] ex = new long[NQ], ann = new long[NQ];
        final double[] rec = new double[NQ];
        for (int i = 0; i < 3; i++) {
            searchExact(s, queries.get(i), filter);
            searchAnn(s, queries.get(i), filter, dim);
        }
        for (int qi = 0; qi < NQ; qi++) {
            final float[] q = queries.get(qi);
            long t = System.nanoTime();
            final TopDocs e = searchExact(s, q, filter);
            ex[qi] = System.nanoTime() - t;
            t = System.nanoTime();
            final TopDocs a = searchAnn(s, q, filter, dim);
            ann[qi] = System.nanoTime() - t;
            rec[qi] = recall(idSet(a), idSet(e), card);
        }
        return new Cell(dim, card, pct(ex, 50), pct(ex, 95), max(ex) / 1_000.0, pct(ann, 50), pct(ann, 95), mean(rec));
    }

    private String renderCells() {
        final StringBuilder sb = new StringBuilder("\n### Exact vs ANN latency by (dim, cardinality) — random filter\n");
        sb.append("| Dim | Card | Exact p50 µs | Exact p95 µs | Exact MAX µs | ANN p50 µs | ANN p95 µs | ANN recall | ops(card*dim) |\n");
        sb.append("| --: | ---: | -----------: | -----------: | -----------: | ---------: | ---------: | ---------: | ------------: |\n");
        for (final Cell c : cells) {
            sb.append(
                String.format(
                    Locale.ROOT,
                    "| %3d | %5d | %12.1f | %12.1f | %12.1f | %10.1f | %10.1f | %.3f | %13d |%n",
                    c.dim,
                    c.card,
                    c.exP50,
                    c.exP95,
                    c.exMax,
                    c.annP50,
                    c.annP95,
                    c.recall,
                    (long) c.card * c.dim
                )
            );
        }
        return sb.toString();
    }

    private String renderRuleComparison() {
        final StringBuilder sb = new StringBuilder("\n### Candidate rules: redirected band + worst exact latency introduced (per dim)\n");
        sb.append("| Rule | Dim | candidateLimit | max redirected card (<=limit, tested) | worst exact p95 µs | worst exact MAX µs |\n");
        sb.append("| ---- | --: | -------------: | ------------------------------------: | -----------------: | -----------------: |\n");
        for (final int dim : DIMS) {
            appendRuleRow(sb, "A: max(1000,10k)", dim, Math.max(1000, 10L * K));
            for (final long ops : MAX_OPS) {
                appendRuleRow(sb, "B: " + (ops / 1000) + "k ops/dim", dim, clamp(ops / dim));
            }
        }
        sb.append("\n(worst exact = the largest tested cardinality <= candidateLimit; this is the exact scan the rule can trigger.)\n");
        return sb.toString();
    }

    private void appendRuleRow(final StringBuilder sb, final String rule, final int dim, final long limit) {
        Cell worst = null;
        int maxCard = -1;
        for (final Cell c : cells) {
            if (c.dim == dim && c.card > K && c.card <= limit && c.card > maxCard) {
                maxCard = c.card;
                worst = c;
            }
        }
        sb.append(
            String.format(
                Locale.ROOT,
                "| %-16s | %3d | %14d | %37s | %18s | %18s |%n",
                rule,
                dim,
                limit,
                worst == null ? "none" : String.valueOf(maxCard),
                worst == null ? "-" : String.format(Locale.ROOT, "%.1f", worst.exP95),
                worst == null ? "-" : String.format(Locale.ROOT, "%.1f", worst.exMax)
            )
        );
    }

    private long clamp(final long v) {
        return Math.max(K + 1, Math.min(HARD_CAP, v));
    }

    // ------------------------------------------------------------------ helpers

    private TopDocs searchExact(final IndexSearcher s, final float[] q, final Query f) throws IOException {
        return s.search(new ExactFilteredKNNVectorQuery(f, FIELD, K, q, 0), K);
    }

    private TopDocs searchAnn(final IndexSearcher s, final float[] q, final Query f, final int dim) throws IOException {
        return s.search(new OSKnnFloatVectorQuery(FIELD, q, Math.max(EF, K), f, K, RescoreContext.NO_RESCORE_NEEDED), K);
    }

    private void buildIndex(final Directory dir, final int dim) throws IOException {
        final Random r = new Random(SEED + dim);
        final int clusterSize = CORPUS / NUM_CLUSTERS;
        final int[] perm = randomPermutation(new Random(SEED + 55));
        try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
            for (int i = 0; i < CORPUS; i++) {
                final int cluster = Math.min(NUM_CLUSTERS - 1, i / clusterSize);
                final Document d = new Document();
                d.add(new KnnFloatVectorField(FIELD, perturb(center(cluster, dim), 0.1f, r), SIM));
                d.add(new IntPoint(RND, perm[i]));
                d.add(new IntPoint(SEQ, i));
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

    private int[] randomPermutation(final Random r) {
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

    private Set<Integer> idSet(final TopDocs td) {
        final Set<Integer> ids = new HashSet<>();
        for (final ScoreDoc sd : td.scoreDocs) {
            ids.add(sd.doc);
        }
        return ids;
    }

    private double recall(final Set<Integer> ann, final Set<Integer> ex, final int card) {
        final int denom = Math.min(K, card);
        if (denom == 0 || ex.isEmpty()) {
            return 1.0;
        }
        int hits = 0;
        for (final int id : ann) {
            if (ex.contains(id)) {
                hits++;
            }
        }
        return (double) hits / denom;
    }

    private String env() {
        final Runtime rt = Runtime.getRuntime();
        return String.format(
            Locale.ROOT,
            "env: jvm=%s, os=%s/%s, cores=%d, maxHeapMB=%d, queries=%d, hardCap=%d, seed=%d%n",
            System.getProperty("java.version"),
            System.getProperty("os.name"),
            System.getProperty("os.arch"),
            rt.availableProcessors(),
            rt.maxMemory() / (1024 * 1024),
            NQ,
            HARD_CAP,
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

    private static double mean(final double[] v) {
        double s = 0;
        for (final double x : v) {
            s += x;
        }
        return v.length == 0 ? 0 : s / v.length;
    }
}
