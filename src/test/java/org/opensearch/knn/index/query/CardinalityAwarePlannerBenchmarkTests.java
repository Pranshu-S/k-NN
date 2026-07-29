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
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
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
 * Multi-query crossover benchmark (iteration 3). For each scenario (corpus, dim, k, ef, filter
 * distribution, cardinality) it runs many independent query vectors against a shared index and reports
 * latency (p50/p95) and the full recall@k distribution for raw ANN, raw exact, and the optimised
 * planner. Exact results are the recall oracle. Query order is rotated per query vector and the Lucene
 * query cache is disabled (compute isolation). Two filter distributions are tested: {@code random}
 * (scattered doc ids) and {@code correlated} (a contiguous block of vector-space clusters). The class
 * also evaluates candidate planner decision models against an oracle that always picks the faster
 * measured path. Tables are printed to stdout (captured in the JUnit XML {@code <system-out>}).
 */
public class CardinalityAwarePlannerBenchmarkTests extends OpenSearchTestCase {

    private static final String FIELD = "vec";
    private static final String SEQ = "seq"; // build order -> correlated (clustered) filter
    private static final String RND = "rnd"; // random permutation rank -> scattered filter
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.EUCLIDEAN;
    private static final int NUM_CLUSTERS = 100;
    private static final long SEED = 42L;
    private static final int[] CARDINALITIES = { 100, 500, 1_000, 2_500, 5_000, 7_500, 10_000, 25_000, 50_000 };

    /** One measured scenario, retained for the decision-model analysis. */
    private static final class Scenario {
        final int corpus, dim, k, ef, cardinality;
        final String distribution;
        final double annP50, annP95, exactP50, exactP95, plannerP50, plannerP95;
        final double meanRecall, medianRecall, p10Recall, minRecall, pctRecall1;
        final String plannerStrategy;

        Scenario(
            int corpus,
            int dim,
            int k,
            int ef,
            String distribution,
            int cardinality,
            double annP50,
            double annP95,
            double exactP50,
            double exactP95,
            double plannerP50,
            double plannerP95,
            double meanRecall,
            double medianRecall,
            double p10Recall,
            double minRecall,
            double pctRecall1,
            String plannerStrategy
        ) {
            this.corpus = corpus;
            this.dim = dim;
            this.k = k;
            this.ef = ef;
            this.distribution = distribution;
            this.cardinality = cardinality;
            this.annP50 = annP50;
            this.annP95 = annP95;
            this.exactP50 = exactP50;
            this.exactP95 = exactP95;
            this.plannerP50 = plannerP50;
            this.plannerP95 = plannerP95;
            this.meanRecall = meanRecall;
            this.medianRecall = medianRecall;
            this.p10Recall = p10Recall;
            this.minRecall = minRecall;
            this.pctRecall1 = pctRecall1;
            this.plannerStrategy = plannerStrategy;
        }
    }

    private final List<Scenario> scenarios = new ArrayList<>();

    @SneakyThrows
    public void testBenchmark100k() {
        final int corpus = 100_000;
        final StringBuilder out = new StringBuilder();
        out.append("\n\n############ MULTI-QUERY CROSSOVER BENCHMARK (100k) ############\n");
        out.append(env());

        // Index A: 100k x 128 — reused for k10 ef{50,100,200} and k100 ef100.
        try (Directory dir128 = newFSDirectory(createTempDir())) {
            buildIndex(dir128, corpus, 128);
            try (IndexReader reader = DirectoryReader.open(dir128)) {
                withSearcher(reader, (searcher) -> {
                    runConfig(out, searcher, corpus, 128, 10, 50, 50);
                    runConfig(out, searcher, corpus, 128, 10, 100, 50);
                    runConfig(out, searcher, corpus, 128, 10, 200, 50);
                    runConfig(out, searcher, corpus, 128, 100, 100, 50);
                });
            }
        }
        // Index B: 100k x 768 — k10 ef100 (fewer queries; 768-dim exact is costly).
        try (Directory dir768 = newFSDirectory(createTempDir())) {
            buildIndex(dir768, corpus, 768);
            try (IndexReader reader = DirectoryReader.open(dir768)) {
                withSearcher(reader, (searcher) -> runConfig(out, searcher, corpus, 768, 10, 100, 30));
            }
        }

        out.append(renderScenarioTable());
        out.append(renderRecallTable());
        out.append(renderModelAnalysis());
        out.append("################################################################\n");
        System.out.println(out);
    }

    @SneakyThrows
    public void testBenchmark1M() {
        // Stretch goal. Uses FSDirectory to keep vectors off-heap; may be slow. Reported separately so a
        // failure here does not lose the 100k results.
        final int corpus = 1_000_000;
        final StringBuilder out = new StringBuilder();
        out.append("\n\n############ MULTI-QUERY CROSSOVER BENCHMARK (1M) ############\n");
        out.append(env());
        try (Directory dir = newFSDirectory(createTempDir())) {
            buildIndex(dir, corpus, 128);
            try (IndexReader reader = DirectoryReader.open(dir)) {
                withSearcher(reader, (searcher) -> runConfig(out, searcher, corpus, 128, 10, 100, 30));
            }
        }
        out.append(renderScenarioTable());
        out.append(renderRecallTable());
        out.append("##############################################################\n");
        System.out.println(out);
    }

    // ------------------------------------------------------------------ per-config run

    private void runConfig(
        final StringBuilder out,
        final IndexSearcher searcher,
        final int corpus,
        final int dim,
        final int k,
        final int ef,
        final int numQueries
    ) throws IOException {
        final Random random = new Random(SEED * 31 + dim * 7L + k * 13L + ef);
        final List<float[]> queries = new ArrayList<>(numQueries);
        // Queries near cluster 0 (always inside the correlated filter's low block); distinct per query.
        final float[] center0 = clusterCenter(0, dim);
        for (int i = 0; i < numQueries; i++) {
            queries.add(perturb(center0, 0.15f, random));
        }
        for (final String dist : new String[] { "random", "correlated" }) {
            for (final int cardinality : CARDINALITIES) {
                if (cardinality > corpus) {
                    continue;
                }
                scenarios.add(runScenario(searcher, corpus, dim, k, ef, dist, cardinality, queries));
            }
        }
        out.append(String.format(Locale.ROOT, "  ran config corpus=%d dim=%d k=%d ef=%d (%d queries)%n", corpus, dim, k, ef, numQueries));
    }

    private Scenario runScenario(
        final IndexSearcher searcher,
        final int corpus,
        final int dim,
        final int k,
        final int ef,
        final String dist,
        final int cardinality,
        final List<float[]> queries
    ) throws IOException {
        final Query filter = filterFor(dist, cardinality);
        final int luceneK = Math.max(ef, k);

        final int warmup = Math.min(5, queries.size());
        final int n = queries.size();
        final long[] annT = new long[n];
        final long[] exactT = new long[n];
        final long[] planT = new long[n];
        final double[] recalls = new double[n];
        String plannerStrategy = "?";

        for (int w = 0; w < warmup; w++) {
            searchAnn(searcher, queries.get(w), filter, luceneK, k);
            searchExact(searcher, queries.get(w), filter, k);
        }

        for (int qi = 0; qi < n; qi++) {
            final float[] q = queries.get(qi);
            Set<Integer> exactIds = null;
            Set<Integer> annIds = null;
            // rotate order to share JIT / page-cache state fairly
            for (final int s : rotate(qi)) {
                if (s == 0) {
                    final long t = System.nanoTime();
                    final TopDocs td = searchAnn(searcher, q, filter, luceneK, k);
                    annT[qi] = System.nanoTime() - t;
                    annIds = idSet(td);
                } else if (s == 1) {
                    final long t = System.nanoTime();
                    final TopDocs td = searchExact(searcher, q, filter, k);
                    exactT[qi] = System.nanoTime() - t;
                    exactIds = idSet(td);
                } else {
                    final KNNFilterPlanningStats stats = new KNNFilterPlanningStats();
                    final long t = System.nanoTime();
                    searchPlanner(searcher, q, filter, luceneK, k, stats);
                    planT[qi] = System.nanoTime() - t;
                    plannerStrategy = String.valueOf(stats.getStrategy());
                }
            }
            recalls[qi] = recall(annIds, exactIds, k, cardinality);
        }

        return new Scenario(
            corpus,
            dim,
            k,
            ef,
            dist,
            cardinality,
            pct(annT, 50),
            pct(annT, 95),
            pct(exactT, 50),
            pct(exactT, 95),
            pct(planT, 50),
            pct(planT, 95),
            mean(recalls),
            pctile(recalls, 50),
            pctile(recalls, 10),
            min(recalls),
            fractionEqual(recalls, 1.0) * 100.0,
            plannerStrategy
        );
    }

    private int[] rotate(final int i) {
        final int shift = i % 3;
        return new int[] { shift % 3, (shift + 1) % 3, (shift + 2) % 3 };
    }

    // ------------------------------------------------------------------ strategies

    private TopDocs searchAnn(final IndexSearcher s, final float[] q, final Query filter, final int luceneK, final int k)
        throws IOException {
        return s.search(new OSKnnFloatVectorQuery(FIELD, q, luceneK, filter, k, RescoreContext.NO_RESCORE_NEEDED), k);
    }

    private TopDocs searchExact(final IndexSearcher s, final float[] q, final Query filter, final int k) throws IOException {
        return s.search(new ExactFilteredKNNVectorQuery(filter, FIELD, k, q, 0), k);
    }

    private TopDocs searchPlanner(
        final IndexSearcher s,
        final float[] q,
        final Query filter,
        final int luceneK,
        final int k,
        final KNNFilterPlanningStats stats
    ) throws IOException {
        final Query ann = new OSKnnFloatVectorQuery(FIELD, q, luceneK, filter, k, RescoreContext.NO_RESCORE_NEEDED);
        final ExactSearcher exact = new ExactSearcher(ModelDao.OpenSearchKNNModelDao.getInstance());
        return s.search(new BoundedExactKnnFloatVectorQuery(ann, filter, FIELD, k, q, 0, exact, stats), k);
    }

    private Query filterFor(final String dist, final int cardinality) {
        final String field = dist.equals("random") ? RND : SEQ;
        return IntPoint.newRangeQuery(field, 0, cardinality - 1);
    }

    // ------------------------------------------------------------------ index + data

    private void buildIndex(final Directory directory, final int totalDocs, final int dim) throws IOException {
        final Random random = new Random(SEED + dim);
        final int[] perm = randomPermutation(totalDocs, random);
        final int clusterSize = Math.max(1, totalDocs / NUM_CLUSTERS);
        try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
            for (int i = 0; i < totalDocs; i++) {
                final int cluster = Math.min(NUM_CLUSTERS - 1, i / clusterSize);
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD, perturb(clusterCenter(cluster, dim), 0.1f, random), SIMILARITY));
                doc.add(new IntPoint(SEQ, i));
                doc.add(new IntPoint(RND, perm[i]));
                writer.addDocument(doc);
            }
            writer.commit();
        }
    }

    /** Deterministic cluster center from a per-cluster seed (not retained in memory). */
    private float[] clusterCenter(final int cluster, final int dim) {
        final Random r = new Random(SEED * 1_000_003L + cluster);
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

    private float[] perturb(final float[] base, final float scale, final Random random) {
        final float[] v = new float[base.length];
        double norm = 0;
        for (int i = 0; i < base.length; i++) {
            v[i] = base[i] + scale * (float) random.nextGaussian();
            norm += (double) v[i] * v[i];
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < base.length; i++) {
            v[i] /= (float) (norm == 0 ? 1 : norm);
        }
        return v;
    }

    private int[] randomPermutation(final int n, final Random random) {
        final int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            final int j = random.nextInt(i + 1);
            final int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    private Set<Integer> idSet(final TopDocs td) {
        final Set<Integer> ids = new HashSet<>();
        for (final ScoreDoc sd : td.scoreDocs) {
            ids.add(sd.doc);
        }
        return ids;
    }

    private double recall(final Set<Integer> annIds, final Set<Integer> exactIds, final int k, final int cardinality) {
        final int denom = Math.min(k, cardinality);
        if (denom == 0 || exactIds == null || exactIds.isEmpty()) {
            return 1.0;
        }
        int hits = 0;
        for (final int id : annIds) {
            if (exactIds.contains(id)) {
                hits++;
            }
        }
        return (double) hits / denom;
    }

    // ------------------------------------------------------------------ tables

    private String renderScenarioTable() {
        final StringBuilder sb = new StringBuilder("\n### Latency (µs) by scenario\n");
        sb.append(
            "| Corpus | Dim | k | ef | Distribution | Cardinality | Selectivity | ANN p50 | ANN p95 | Exact p50 | Exact p95 | Planner p50 | Planner p95 | Strategy |\n"
        );
        sb.append(
            "| -----: | --: | -: | -: | ------------ | ----------: | ----------: | ------: | ------: | --------: | --------: | ----------: | ----------: | -------- |\n"
        );
        for (final Scenario s : scenarios) {
            sb.append(
                String.format(
                    Locale.ROOT,
                    "| %6d | %3d | %2d | %3d | %-11s | %11d | %9.3f%% | %7.1f | %7.1f | %9.1f | %9.1f | %11.1f | %11.1f | %-8s |%n",
                    s.corpus,
                    s.dim,
                    s.k,
                    s.ef,
                    s.distribution,
                    s.cardinality,
                    100.0 * s.cardinality / s.corpus,
                    s.annP50,
                    s.annP95,
                    s.exactP50,
                    s.exactP95,
                    s.plannerP50,
                    s.plannerP95,
                    s.plannerStrategy
                )
            );
        }
        return sb.toString();
    }

    private String renderRecallTable() {
        final StringBuilder sb = new StringBuilder("\n### ANN recall@k distribution by scenario\n");
        sb.append("| Corpus | Dim | k | ef | Distribution | Cardinality | Mean | Median | p10 | Min | %recall=1.0 |\n");
        sb.append("| -----: | --: | -: | -: | ------------ | ----------: | ---: | -----: | --: | --: | ----------: |\n");
        for (final Scenario s : scenarios) {
            sb.append(
                String.format(
                    Locale.ROOT,
                    "| %6d | %3d | %2d | %3d | %-11s | %11d | %.3f | %.3f | %.3f | %.3f | %8.1f%% |%n",
                    s.corpus,
                    s.dim,
                    s.k,
                    s.ef,
                    s.distribution,
                    s.cardinality,
                    s.meanRecall,
                    s.medianRecall,
                    s.p10Recall,
                    s.minRecall,
                    s.pctRecall1
                )
            );
        }
        return sb.toString();
    }

    // ------------------------------------------------------------------ Phase 5: decision models

    private interface Model {
        String name();

        boolean picksExact(Scenario s); // true=exact, false=ANN
    }

    private String renderModelAnalysis() {
        // Calibrate cost-model constants on dim128/k10/ef100 rows; validate on the rest.
        final List<Scenario> calib = new ArrayList<>();
        for (final Scenario s : scenarios) {
            if (s.dim == 128 && s.k == 10 && s.ef == 100) {
                calib.add(s);
            }
        }
        final double[] cost = fitCostModel(calib); // {costPerCardDim, annBaseline}
        final int aStar = fitEfMultiplier(calib); // threshold = aStar * ef

        final List<Model> models = new ArrayList<>();
        models.add(model("A: max(1000,10k)", s -> s.cardinality <= Math.max(1000, 10L * s.k)));
        for (final int t : new int[] { 2500, 5000, 7500, 10000 }) {
            models.add(model("B: fixed " + t, s -> s.cardinality <= t));
        }
        models.add(model("C: " + aStar + "*ef", s -> s.cardinality <= (long) aStar * s.ef));
        models.add(model("D: 5000*128/dim", s -> s.cardinality <= 5000L * 128 / s.dim));
        models.add(model("E: cost card*dim", s -> (double) s.cardinality * s.dim * cost[0] < cost[1]));

        final StringBuilder sb = new StringBuilder("\n### Decision-model comparison vs. oracle (faster measured path)\n");
        sb.append(
            String.format(
                Locale.ROOT,
                "(fitted on dim128/k10/ef100: costPerCardDim=%.3e ns, annBaseline=%.1f µs, ef-multiplier a*=%d)%n",
                cost[0],
                cost[1],
                aStar
            )
        );
        sb.append(
            "| Model | Faster-path selection rate | Mean regret µs | p95 regret µs | Mean rel-regret | ANN chosen w/ recall loss | Exact chosen w/ ANN>=2x faster |\n"
        );
        sb.append(
            "| ----- | -------------------------: | -------------: | ------------: | --------------: | ------------------------: | -----------------------------: |\n"
        );
        for (final Model m : models) {
            sb.append(evaluateModel(m));
        }
        sb.append("\nRegret = selected-path p50 - min(ANN p50, Exact p50). 'recall loss' = model picks ANN while mean recall < 0.99.\n");
        return sb.toString();
    }

    private Model model(final String name, final java.util.function.Predicate<Scenario> picksExact) {
        return new Model() {
            public String name() {
                return name;
            }

            public boolean picksExact(final Scenario s) {
                return picksExact.test(s);
            }
        };
    }

    private String evaluateModel(final Model m) {
        int faster = 0, annRecallLoss = 0, exactWhenAnnMuchFaster = 0;
        final List<Double> regrets = new ArrayList<>();
        final List<Double> relRegrets = new ArrayList<>();
        for (final Scenario s : scenarios) {
            final boolean pickExact = m.picksExact(s);
            final double chosen = pickExact ? s.exactP50 : s.annP50;
            final double oracle = Math.min(s.annP50, s.exactP50);
            final double regret = chosen - oracle;
            regrets.add(regret);
            relRegrets.add(oracle > 0 ? regret / oracle : 0.0);
            final boolean exactIsFaster = s.exactP50 <= s.annP50;
            if (pickExact == exactIsFaster) {
                faster++;
            }
            if (!pickExact && s.meanRecall < 0.99) {
                annRecallLoss++;
            }
            if (pickExact && s.annP50 * 2 <= s.exactP50) {
                exactWhenAnnMuchFaster++;
            }
        }
        final double[] rg = regrets.stream().mapToDouble(Double::doubleValue).toArray();
        return String.format(
            Locale.ROOT,
            "| %-18s | %25.1f%% | %14.1f | %13.1f | %14.1f%% | %25d | %30d |%n",
            m.name(),
            100.0 * faster / scenarios.size(),
            mean(rg),
            pct(toLong(rg), 95),
            100.0 * mean(relRegrets.stream().mapToDouble(Double::doubleValue).toArray()),
            annRecallLoss,
            exactWhenAnnMuchFaster
        );
    }

    private double[] fitCostModel(final List<Scenario> calib) {
        // exactP50 ~ costPerCardDim * (card*dim); fit slope via least-squares through origin.
        double num = 0, den = 0;
        for (final Scenario s : calib) {
            final double x = (double) s.cardinality * s.dim;
            num += x * s.exactP50;
            den += x * x;
        }
        final double slope = den > 0 ? num / den : 0;
        // ANN baseline = median ANN p50 over calibration (roughly card-independent).
        final double[] ann = calib.stream().mapToDouble(s -> s.annP50).toArray();
        return new double[] { slope, ann.length > 0 ? pctile(ann, 50) : 1000.0 };
    }

    private int fitEfMultiplier(final List<Scenario> calib) {
        int best = 25;
        double bestRegret = Double.MAX_VALUE;
        for (final int a : new int[] { 10, 25, 50, 75, 100 }) {
            double total = 0;
            for (final Scenario s : calib) {
                final boolean pickExact = s.cardinality <= (long) a * s.ef;
                final double chosen = pickExact ? s.exactP50 : s.annP50;
                total += chosen - Math.min(s.annP50, s.exactP50);
            }
            if (total < bestRegret) {
                bestRegret = total;
                best = a;
            }
        }
        return best;
    }

    // ------------------------------------------------------------------ helpers

    @FunctionalInterface
    private interface SearcherConsumer {
        void accept(IndexSearcher searcher) throws IOException;
    }

    private void withSearcher(final IndexReader reader, final SearcherConsumer consumer) throws IOException {
        try (MockedStatic<ModelDao.OpenSearchKNNModelDao> mocked = Mockito.mockStatic(ModelDao.OpenSearchKNNModelDao.class)) {
            mocked.when(ModelDao.OpenSearchKNNModelDao::getInstance).thenReturn(mock(ModelDao.OpenSearchKNNModelDao.class));
            final IndexSearcher searcher = newSearcher(reader, true, false);
            searcher.setQueryCache(null);
            consumer.accept(searcher);
        }
    }

    private String env() {
        final Runtime rt = Runtime.getRuntime();
        return String.format(
            Locale.ROOT,
            "env: jvm=%s, os=%s/%s, cores=%d, maxHeapMB=%d, queryCache=DISABLED, orderRotated=true, seed=%d%n",
            System.getProperty("java.version"),
            System.getProperty("os.name"),
            System.getProperty("os.arch"),
            rt.availableProcessors(),
            rt.maxMemory() / (1024 * 1024),
            SEED
        );
    }

    private static double pct(final long[] samples, final int p) {
        final long[] c = samples.clone();
        Arrays.sort(c);
        final int idx = Math.max(0, Math.min((int) Math.ceil(p / 100.0 * c.length) - 1, c.length - 1));
        return c[idx] / 1_000.0;
    }

    private static long[] toLong(final double[] micros) {
        final long[] l = new long[micros.length];
        for (int i = 0; i < micros.length; i++) {
            l[i] = (long) (micros[i] * 1_000);
        }
        return l;
    }

    private static double pctile(final double[] values, final int p) {
        final double[] c = values.clone();
        Arrays.sort(c);
        final int idx = Math.max(0, Math.min((int) Math.ceil(p / 100.0 * c.length) - 1, c.length - 1));
        return c[idx];
    }

    private static double mean(final double[] v) {
        double sum = 0;
        for (final double x : v) {
            sum += x;
        }
        return v.length == 0 ? 0 : sum / v.length;
    }

    private static double min(final double[] v) {
        double m = Double.MAX_VALUE;
        for (final double x : v) {
            m = Math.min(m, x);
        }
        return v.length == 0 ? 0 : m;
    }

    private static double fractionEqual(final double[] v, final double target) {
        int c = 0;
        for (final double x : v) {
            if (x >= target - 1e-9) {
                c++;
            }
        }
        return v.length == 0 ? 0 : (double) c / v.length;
    }
}
