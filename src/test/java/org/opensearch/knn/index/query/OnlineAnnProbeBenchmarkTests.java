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
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
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
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

import static org.mockito.Mockito.mock;

/**
 * Phase 5-7 feasibility experiment for an online adaptive ANN fallback. Runs a bounded HNSW probe (via
 * {@link AnnProbe}, public APIs only) at several budgets, then measures whether the probe signals
 * (early-terminated, candidates collected) predict the exact-vs-ANN faster path and ANN recall loss —
 * charging the probe as pure duplicate work (probe + chosen full path), since Lucene exposes no
 * resumable traversal state. Distributions reuse the iteration-4 adversarial constructions.
 */
public class OnlineAnnProbeBenchmarkTests extends OpenSearchTestCase {

    private static final String FIELD = "vec";
    private static final String CORR = "corr";
    private static final String RND = "rnd";
    private static final String SEQ = "seq";
    private static final VectorSimilarityFunction SIM = VectorSimilarityFunction.EUCLIDEAN;
    private static final int CORPUS = 100_000;
    private static final int DIM = 128;
    private static final int K = 10;
    private static final int EF = 100;
    private static final int NUM_CLUSTERS = 100;
    private static final int CLUSTER_SIZE = CORPUS / NUM_CLUSTERS;
    private static final int NQ = 30;
    private static final long SEED = 42L;
    private static final int[] CARDS = { 2_500, 5_000, 10_000, 25_000 };
    private static final int[] BUDGETS = { 50, 100, 250, 500, 1_000 };
    private static final double RECALL_TARGET = 0.95;

    private static final class Row {
        final String dist;
        final int card;
        final double annP50, exactP50, annRecall, fullVisited;
        // per-budget probe stats
        final double[] probeP50 = new double[BUDGETS.length];
        final double[] probeVisited = new double[BUDGETS.length];
        final double[] probeEarlyFrac = new double[BUDGETS.length];
        final double[] probeCollected = new double[BUDGETS.length];

        Row(String dist, int card, double annP50, double exactP50, double annRecall, double fullVisited) {
            this.dist = dist;
            this.card = card;
            this.annP50 = annP50;
            this.exactP50 = exactP50;
            this.annRecall = annRecall;
            this.fullVisited = fullVisited;
        }

        boolean exactFaster() {
            return exactP50 <= annP50;
        }

        boolean recallLoss() {
            return annRecall < RECALL_TARGET;
        }
    }

    private final List<Row> rows = new ArrayList<>();

    @SneakyThrows
    public void testProbeBudgetStudy() {
        final StringBuilder out = new StringBuilder("\n\n====== ONLINE ANN PROBE FEASIBILITY (100k/dim128/k10/ef100) ======\n");
        out.append(env());

        try (
            Directory ordered = newFSDirectory(createTempDir());
            Directory shuffled = newFSDirectory(createTempDir());
            Directory randomVec = newFSDirectory(createTempDir())
        ) {
            buildClustered(ordered, false);
            buildClustered(shuffled, true);
            buildRandomVec(randomVec);

            final List<float[]> clusterQ = new ArrayList<>();
            final Random qr = new Random(SEED + 7);
            for (int i = 0; i < NQ; i++) {
                clusterQ.add(perturb(clusterCenter(0), 0.15f, qr));
            }
            final List<float[]> randomQ = new ArrayList<>();
            for (int i = 0; i < NQ; i++) {
                randomQ.add(unit(new Random(SEED + 200 + i)));
            }

            withSearcher(ordered, s -> {
                for (final int c : CARDS) {
                    rows.add(measure(s, "random", c, IntPoint.newRangeQuery(RND, 0, c - 1), clusterQ));
                    rows.add(measure(s, "correlated", c, IntPoint.newRangeQuery(CORR, 0, c - 1), clusterQ));
                }
            });
            withSearcher(shuffled, s -> {
                for (final int c : CARDS) {
                    rows.add(measure(s, "scatter_corr", c, IntPoint.newRangeQuery(CORR, 0, c - 1), clusterQ));
                }
            });
            withSearcher(randomVec, s -> {
                for (final int c : CARDS) {
                    rows.add(measure(s, "contig_rand", c, IntPoint.newRangeQuery(SEQ, 0, c - 1), randomQ));
                }
            });
        }

        out.append(renderScenarioTable());
        out.append(renderBudgetTable());
        out.append(renderRecallPrediction());
        out.append("=================================================================\n");
        System.out.println(out);
    }

    private Row measure(final IndexSearcher s, final String dist, final int card, final Query filter, final List<float[]> queries)
        throws IOException {
        final Weight fw = s.createWeight(s.rewrite(filter), ScoreMode.COMPLETE_NO_SCORES, 1f);
        final long[] annT = new long[NQ], exT = new long[NQ];
        final double[] recalls = new double[NQ];
        double fullVisited = 0;
        // warmup
        for (int i = 0; i < 3; i++) {
            searchAnn(s, queries.get(i), filter);
            searchExact(s, queries.get(i), filter);
        }
        for (int qi = 0; qi < NQ; qi++) {
            final float[] q = queries.get(qi);
            long t = System.nanoTime();
            final TopDocs ann = searchAnn(s, q, filter);
            annT[qi] = System.nanoTime() - t;
            t = System.nanoTime();
            final TopDocs ex = searchExact(s, q, filter);
            exT[qi] = System.nanoTime() - t;
            recalls[qi] = recall(idSet(ann), idSet(ex), card);
            fullVisited += AnnProbe.probe(s, fw, FIELD, q, K, 1_000_000).visited; // reference full graph work
        }
        final Row row = new Row(dist, card, pct(annT, 50), pct(exT, 50), mean(recalls), fullVisited / NQ);

        for (int bi = 0; bi < BUDGETS.length; bi++) {
            final int budget = BUDGETS[bi];
            final long[] pT = new long[NQ];
            double vis = 0, early = 0, coll = 0;
            for (int qi = 0; qi < NQ; qi++) {
                final long t = System.nanoTime();
                final AnnProbe.Result r = AnnProbe.probe(s, fw, FIELD, queries.get(qi), K, budget);
                pT[qi] = System.nanoTime() - t;
                vis += r.visited;
                early += r.earlyTerminated ? 1 : 0;
                coll += r.collected;
            }
            row.probeP50[bi] = pct(pT, 50);
            row.probeVisited[bi] = vis / NQ;
            row.probeEarlyFrac[bi] = early / NQ;
            row.probeCollected[bi] = coll / NQ;
        }
        return row;
    }

    // ------------------------------------------------------------------ rendering

    private String renderScenarioTable() {
        final StringBuilder sb = new StringBuilder("\n### Scenario latency/recall + probe@250 (visited, earlyFrac, collected)\n");
        sb.append(
            "| Distribution | Card | ANN p50 | Exact p50 | FasterPath | ANN recall | probe250 p50 | probeVisited | earlyFrac | collected | probeRule | correct? |\n"
        );
        sb.append(
            "| ------------ | ---: | ------: | --------: | ---------- | ---------: | -----------: | -----------: | --------: | --------: | --------- | -------- |\n"
        );
        final int b250 = indexOf(250);
        for (final Row r : rows) {
            final boolean predExact = r.probeCollected[b250] < K || r.probeEarlyFrac[b250] >= 0.5;
            final boolean correct = predExact == r.exactFaster();
            sb.append(
                String.format(
                    Locale.ROOT,
                    "| %-12s | %5d | %7.1f | %9.1f | %-10s | %.3f | %12.1f | %12.0f | %9.2f | %9.1f | %-9s | %-8s |%n",
                    r.dist,
                    r.card,
                    r.annP50,
                    r.exactP50,
                    r.exactFaster() ? "EXACT" : "ANN",
                    r.annRecall,
                    r.probeP50[b250],
                    r.probeVisited[b250],
                    r.probeEarlyFrac[b250],
                    r.probeCollected[b250],
                    predExact ? "EXACT" : "ANN",
                    correct ? "yes" : "NO"
                )
            );
        }
        return sb.toString();
    }

    private String renderBudgetTable() {
        final StringBuilder sb = new StringBuilder("\n### Probe-budget analysis (end-to-end charges probe + chosen full path)\n");
        sb.append(
            "| Budget | Probe p50 µs | % of full ANN visits | Selection acc | Mean regret µs | p95 regret µs | Recall viol. | Slower-than-both |\n"
        );
        sb.append(
            "| -----: | -----------: | -------------------: | ------------: | -------------: | ------------: | -----------: | ---------------: |\n"
        );
        for (int bi = 0; bi < BUDGETS.length; bi++) {
            double probeSum = 0, visitFrac = 0, correct = 0, recallViol = 0, slowerBoth = 0;
            final List<Double> regret = new ArrayList<>();
            for (final Row r : rows) {
                probeSum += r.probeP50[bi];
                visitFrac += r.fullVisited > 0 ? r.probeVisited[bi] / r.fullVisited : 0;
                final boolean predExact = r.probeCollected[bi] < K || r.probeEarlyFrac[bi] >= 0.5;
                if (predExact == r.exactFaster()) {
                    correct++;
                }
                // recall violation: probe rule keeps ANN but ANN loses recall
                if (!predExact && r.recallLoss()) {
                    recallViol++;
                }
                // end-to-end = probe + chosen full path (no continuation => duplicate work)
                final double chosen = predExact ? r.exactP50 : r.annP50;
                final double e2e = r.probeP50[bi] + chosen;
                final double oracle = Math.min(r.annP50, r.exactP50);
                regret.add(e2e - oracle);
                if (e2e > Math.max(r.annP50, r.exactP50)) {
                    slowerBoth++;
                }
            }
            final int n = rows.size();
            final double[] rg = regret.stream().mapToDouble(Double::doubleValue).toArray();
            sb.append(
                String.format(
                    Locale.ROOT,
                    "| %6d | %12.1f | %19.1f%% | %12.1f%% | %14.1f | %13.1f | %12.0f | %16.0f |%n",
                    BUDGETS[bi],
                    probeSum / n,
                    100.0 * visitFrac / n,
                    100.0 * correct / n,
                    mean(rg),
                    pctileD(rg, 95),
                    recallViol,
                    slowerBoth
                )
            );
        }
        sb.append("\nProbe rule: EXACT if (collected < k) OR (early-terminated). Regret = (probe + chosen p50) - min(ANN,Exact) p50.\n");
        sb.append("'Slower-than-both' = end-to-end p50 exceeds BOTH raw ANN and raw exact (probe overhead net-harmful).\n");
        return sb.toString();
    }

    private String renderRecallPrediction() {
        final StringBuilder sb = new StringBuilder("\n### Does the probe predict ANN recall loss? (budget 250)\n");
        sb.append(
            "| Distribution | Card | ANN recall | recallLoss(<0.95) | probe earlyFrac | probe collected>=k | probe predicts loss? |\n"
        );
        sb.append(
            "| ------------ | ---: | ---------: | ----------------: | --------------: | -----------------: | -------------------: |\n"
        );
        final int b = indexOf(250);
        for (final Row r : rows) {
            final boolean lossPredicted = r.probeCollected[b] < K || r.probeEarlyFrac[b] >= 0.5;
            sb.append(
                String.format(
                    Locale.ROOT,
                    "| %-12s | %5d | %.3f | %-17s | %15.2f | %18s | %-20s |%n",
                    r.dist,
                    r.card,
                    r.annRecall,
                    r.recallLoss() ? "YES" : "no",
                    r.probeEarlyFrac[b],
                    r.probeCollected[b] >= K ? "yes" : "no",
                    lossPredicted ? "predicts EXACT" : "predicts ANN"
                )
            );
        }
        return sb.toString();
    }

    // ------------------------------------------------------------------ queries / index

    private TopDocs searchAnn(final IndexSearcher s, final float[] q, final Query f) throws IOException {
        return s.search(new OSKnnFloatVectorQuery(FIELD, q, Math.max(EF, K), f, K, RescoreContext.NO_RESCORE_NEEDED), K);
    }

    private TopDocs searchExact(final IndexSearcher s, final float[] q, final Query f) throws IOException {
        return s.search(new ExactFilteredKNNVectorQuery(f, FIELD, K, q, 0), K);
    }

    private void buildClustered(final Directory dir, final boolean shuffle) throws IOException {
        final Random r = new Random(SEED + (shuffle ? 1 : 0));
        final Integer[] order = new Integer[CORPUS];
        for (int i = 0; i < CORPUS; i++) {
            order[i] = i;
        }
        if (shuffle) {
            Collections.shuffle(Arrays.asList(order), r);
        }
        final int[] perm = randomPermutation(CORPUS, new Random(SEED + 55));
        try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
            int seq = 0;
            for (final int corrRank : order) {
                final int cluster = Math.min(NUM_CLUSTERS - 1, corrRank / CLUSTER_SIZE);
                final Document d = new Document();
                d.add(new KnnFloatVectorField(FIELD, perturb(clusterCenter(cluster), 0.1f, r), SIM));
                d.add(new IntPoint(CORR, corrRank));
                d.add(new IntPoint(RND, perm[corrRank]));
                d.add(new IntPoint(SEQ, seq++));
                w.addDocument(d);
            }
            w.commit();
        }
    }

    private void buildRandomVec(final Directory dir) throws IOException {
        final Random r = new Random(SEED + 2);
        try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
            for (int i = 0; i < CORPUS; i++) {
                final Document d = new Document();
                d.add(new KnnFloatVectorField(FIELD, unit(r), SIM));
                d.add(new IntPoint(SEQ, i));
                w.addDocument(d);
            }
            w.commit();
        }
    }

    private float[] clusterCenter(final int c) {
        return unit(new Random(SEED * 1_000_003L + c));
    }

    private float[] unit(final Random r) {
        final float[] v = new float[DIM];
        double norm = 0;
        for (int i = 0; i < DIM; i++) {
            v[i] = (float) r.nextGaussian();
            norm += (double) v[i] * v[i];
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < DIM; i++) {
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

    private int[] randomPermutation(final int n, final Random r) {
        final int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            final int j = r.nextInt(i + 1);
            final int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    // ------------------------------------------------------------------ helpers

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

    private int indexOf(final int budget) {
        for (int i = 0; i < BUDGETS.length; i++) {
            if (BUDGETS[i] == budget) {
                return i;
            }
        }
        return 0;
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
            "env: jvm=%s, os=%s/%s, cores=%d, maxHeapMB=%d, queries=%d, budgets=%s, queryCache=DISABLED, seed=%d%n",
            System.getProperty("java.version"),
            System.getProperty("os.name"),
            System.getProperty("os.arch"),
            rt.availableProcessors(),
            rt.maxMemory() / (1024 * 1024),
            NQ,
            Arrays.toString(BUDGETS),
            SEED
        );
    }

    private static double pct(final long[] s, final int p) {
        final long[] c = s.clone();
        Arrays.sort(c);
        return c[Math.max(0, Math.min((int) Math.ceil(p / 100.0 * c.length) - 1, c.length - 1))] / 1_000.0;
    }

    private static double pctileD(final double[] v, final int p) {
        if (v.length == 0) {
            return 0;
        }
        final double[] c = v.clone();
        Arrays.sort(c);
        return c[Math.max(0, Math.min((int) Math.ceil(p / 100.0 * c.length) - 1, c.length - 1))];
    }

    private static double mean(final double[] v) {
        double s = 0;
        for (final double x : v) {
            s += x;
        }
        return v.length == 0 ? 0 : s / v.length;
    }
}
