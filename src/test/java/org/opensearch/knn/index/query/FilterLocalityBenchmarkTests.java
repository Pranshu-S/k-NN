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
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
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
import java.util.function.Predicate;

import static org.mockito.Mockito.mock;

/**
 * Iteration-4 study: can a cheap filter-locality signal (from the bounded filter matches) beat the
 * ~73% faster-path ceiling of cardinality-only rules — and does it survive adversarial distributions
 * that decouple document-id locality from vector-space locality?
 *
 * <p>Four distributions cross (doc-id locality) x (vector locality) at 100k/dim128:
 * <ul>
 *   <li><b>rand_rand</b>   — scattered ids, scattered vectors (calibration);</li>
 *   <li><b>contig_corr</b> — contiguous ids, correlated vectors (calibration);</li>
 *   <li><b>scatter_corr</b>— scattered ids, correlated vectors (VALIDATION / adversarial: shuffled
 *       insertion, so doc-id scatter is high while vectors are clustered);</li>
 *   <li><b>contig_rand</b> — contiguous ids, random vectors (VALIDATION / adversarial: doc-id
 *       contiguous while vectors are scattered).</li>
 * </ul>
 * Calibration = the two "natural" distributions (where doc-id locality happens to track vector
 * locality); validation = the two adversarial ones. A doc-id feature fitted on calibration but failing
 * on validation is proven to be an insertion-order artifact; a vector feature that holds is real.
 */
public class FilterLocalityBenchmarkTests extends OpenSearchTestCase {

    private static final String FIELD = "vec";
    private static final String CORR = "corr"; // cluster-major rank -> correlated-vector filter
    private static final String RND = "rnd";   // random permutation rank -> scattered filter
    private static final String SEQ = "seq";   // insertion order
    private static final VectorSimilarityFunction SIM = VectorSimilarityFunction.EUCLIDEAN;
    private static final int CORPUS = 100_000;
    private static final int DIM = 128;
    private static final int K = 10;
    private static final int EF = 100;
    private static final int NUM_CLUSTERS = 100;
    private static final int CLUSTER_SIZE = CORPUS / NUM_CLUSTERS;
    private static final int NUM_QUERIES = 50;
    private static final int SAMPLE_CAP = 1024; // bounded feature sample
    private static final int VEC_SAMPLE = 48;
    private static final long SEED = 42L;
    private static final int[] CARDS = { 500, 1_000, 2_500, 5_000, 7_500, 10_000, 25_000, 50_000 };

    private static final class Scenario {
        final String dist;
        final boolean validation;
        final int card;
        final double annP50, exactP50, plannerP50, annP95, exactP95;
        final double meanRecall, p10Recall, minRecall, pctRecall1;
        final FilterLocalityFeatures.ScatterFeatures sf;
        final double vecMeanDistToCentroid, vecMeanDistToQuery, featureOverheadUs, vecFeatureOverheadUs;

        Scenario(
            String dist,
            boolean validation,
            int card,
            double annP50,
            double exactP50,
            double plannerP50,
            double annP95,
            double exactP95,
            double meanRecall,
            double p10Recall,
            double minRecall,
            double pctRecall1,
            FilterLocalityFeatures.ScatterFeatures sf,
            double vecMeanDistToCentroid,
            double vecMeanDistToQuery,
            double featureOverheadUs,
            double vecFeatureOverheadUs
        ) {
            this.dist = dist;
            this.validation = validation;
            this.card = card;
            this.annP50 = annP50;
            this.exactP50 = exactP50;
            this.plannerP50 = plannerP50;
            this.annP95 = annP95;
            this.exactP95 = exactP95;
            this.meanRecall = meanRecall;
            this.p10Recall = p10Recall;
            this.minRecall = minRecall;
            this.pctRecall1 = pctRecall1;
            this.sf = sf;
            this.vecMeanDistToCentroid = vecMeanDistToCentroid;
            this.vecMeanDistToQuery = vecMeanDistToQuery;
            this.featureOverheadUs = featureOverheadUs;
            this.vecFeatureOverheadUs = vecFeatureOverheadUs;
        }

        boolean exactIsFaster() {
            return exactP50 <= annP50;
        }
    }

    private final List<Scenario> scenarios = new ArrayList<>();

    @SneakyThrows
    public void testLocalityFeatureStudy() {
        final StringBuilder out = new StringBuilder("\n\n@@@@@@@@ FILTER-LOCALITY FEATURE STUDY (100k/dim128/k10/ef100) @@@@@@@@\n");
        out.append(env());

        // Build the three indices.
        try (
            Directory ordered = newFSDirectory(createTempDir());
            Directory shuffled = newFSDirectory(createTempDir());
            Directory randomVec = newFSDirectory(createTempDir())
        ) {

            buildClustered(ordered, false); // cluster-major insertion order
            buildClustered(shuffled, true); // shuffled insertion order
            buildRandomVec(randomVec);

            final Random qr = new Random(SEED + 7);
            final List<float[]> queries = new ArrayList<>();
            for (int i = 0; i < NUM_QUERIES; i++) {
                queries.add(perturb(clusterCenter(0), 0.15f, qr)); // near cluster 0 (present in low corr-rank band)
            }
            final List<float[]> randomQueries = new ArrayList<>();
            for (int i = 0; i < NUM_QUERIES; i++) {
                randomQueries.add(unit(new Random(SEED + 100 + i)));
            }

            withSearcher(ordered, s -> {
                for (final int c : CARDS) {
                    scenarios.add(run(s, "rand_rand", false, c, IntPoint.newRangeQuery(RND, 0, c - 1), queries));
                    scenarios.add(run(s, "contig_corr", false, c, IntPoint.newRangeQuery(CORR, 0, c - 1), queries));
                }
            });
            withSearcher(shuffled, s -> {
                for (final int c : CARDS) {
                    scenarios.add(run(s, "scatter_corr", true, c, IntPoint.newRangeQuery(CORR, 0, c - 1), queries));
                }
            });
            withSearcher(randomVec, s -> {
                for (final int c : CARDS) {
                    scenarios.add(run(s, "contig_rand", true, c, IntPoint.newRangeQuery(SEQ, 0, c - 1), randomQueries));
                }
            });
        }

        out.append(renderFeatureTable());
        out.append(renderModelAnalysis());
        out.append(renderRecallConstrained());
        out.append("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
        System.out.println(out);
    }

    // ------------------------------------------------------------------ scenario execution

    private Scenario run(
        final IndexSearcher s,
        final String dist,
        final boolean validation,
        final int card,
        final Query filter,
        final List<float[]> queries
    ) throws IOException {
        // Features (query-independent scatter; vector dispersion averaged over a few queries).
        final Weight fw = s.createWeight(s.rewrite(filter), ScoreMode.COMPLETE_NO_SCORES, 1f);
        long t0 = System.nanoTime();
        final List<FilterLocalityFeatures.LeafSample> sample = FilterLocalityFeatures.collectSample(s, fw, SAMPLE_CAP);
        final FilterLocalityFeatures.ScatterFeatures sf = FilterLocalityFeatures.scatterFeatures(
            sample,
            s.getIndexReader().leaves().size()
        );
        final double scatterOverheadUs = (System.nanoTime() - t0) / 1_000.0;

        double vecCentroid = 0, vecQuery = 0, vecOverhead = 0;
        final int vecQ = Math.min(5, queries.size());
        for (int i = 0; i < vecQ; i++) {
            final long v0 = System.nanoTime();
            final FilterLocalityFeatures.VectorDispersion vd = FilterLocalityFeatures.vectorDispersion(
                sample,
                FIELD,
                queries.get(i),
                VEC_SAMPLE
            );
            vecOverhead += (System.nanoTime() - v0) / 1_000.0;
            vecCentroid += vd.meanDistToCentroid;
            vecQuery += vd.meanDistToQuery;
        }
        vecCentroid /= vecQ;
        vecQuery /= vecQ;
        vecOverhead /= vecQ;

        // Latency + recall over all queries, rotated order.
        final int n = queries.size();
        final long[] annT = new long[n], exT = new long[n], plT = new long[n];
        final double[] recalls = new double[n];
        for (int w = 0; w < 5; w++) {
            searchAnn(s, queries.get(w), filter);
            searchExact(s, queries.get(w), filter);
        }
        for (int qi = 0; qi < n; qi++) {
            final float[] q = queries.get(qi);
            Set<Integer> ann = null, ex = null;
            for (final int strat : rotate(qi)) {
                if (strat == 0) {
                    final long t = System.nanoTime();
                    final TopDocs td = searchAnn(s, q, filter);
                    annT[qi] = System.nanoTime() - t;
                    ann = idSet(td);
                } else if (strat == 1) {
                    final long t = System.nanoTime();
                    final TopDocs td = searchExact(s, q, filter);
                    exT[qi] = System.nanoTime() - t;
                    ex = idSet(td);
                } else {
                    final long t = System.nanoTime();
                    searchPlanner(s, q, filter);
                    plT[qi] = System.nanoTime() - t;
                }
            }
            recalls[qi] = recall(ann, ex, card);
        }
        return new Scenario(
            dist,
            validation,
            card,
            pct(annT, 50),
            pct(exT, 50),
            pct(plT, 50),
            pct(annT, 95),
            pct(exT, 95),
            mean(recalls),
            pctile(recalls, 10),
            min(recalls),
            fracEqual(recalls, 1.0) * 100,
            sf,
            vecCentroid,
            vecQuery,
            scatterOverheadUs,
            vecOverhead
        );
    }

    private int[] rotate(final int i) {
        final int sh = i % 3;
        return new int[] { sh % 3, (sh + 1) % 3, (sh + 2) % 3 };
    }

    private TopDocs searchAnn(final IndexSearcher s, final float[] q, final Query f) throws IOException {
        return s.search(new OSKnnFloatVectorQuery(FIELD, q, Math.max(EF, K), f, K, RescoreContext.NO_RESCORE_NEEDED), K);
    }

    private TopDocs searchExact(final IndexSearcher s, final float[] q, final Query f) throws IOException {
        return s.search(new ExactFilteredKNNVectorQuery(f, FIELD, K, q, 0), K);
    }

    private TopDocs searchPlanner(final IndexSearcher s, final float[] q, final Query f) throws IOException {
        final Query ann = new OSKnnFloatVectorQuery(FIELD, q, Math.max(EF, K), f, K, RescoreContext.NO_RESCORE_NEEDED);
        final ExactSearcher ex = new ExactSearcher(ModelDao.OpenSearchKNNModelDao.getInstance());
        return s.search(new BoundedExactKnnFloatVectorQuery(ann, f, FIELD, K, q, 0, ex, null), K);
    }

    // ------------------------------------------------------------------ index construction

    private void buildClustered(final Directory dir, final boolean shuffleInsertion) throws IOException {
        final Random r = new Random(SEED + (shuffleInsertion ? 1 : 0));
        // Doc definition: corrRank in cluster-major order -> cluster = corrRank/CLUSTER_SIZE.
        final Integer[] insertionOrder = new Integer[CORPUS];
        for (int i = 0; i < CORPUS; i++) {
            insertionOrder[i] = i; // i == corrRank
        }
        if (shuffleInsertion) {
            Collections.shuffle(Arrays.asList(insertionOrder), r);
        }
        final int[] perm = randomPermutation(CORPUS, new Random(SEED + 55));
        try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
            int seq = 0;
            for (final int corrRank : insertionOrder) {
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

    private float[] clusterCenter(final int cluster) {
        final Random r = new Random(SEED * 1_000_003L + cluster);
        return unit(r);
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

    // ------------------------------------------------------------------ tables

    private String renderFeatureTable() {
        final StringBuilder sb = new StringBuilder("\n### Features & measured latency/recall by scenario\n");
        sb.append(
            "| Distribution | Set | Card | normSpan | meanGap | fracInRuns | vecCentroidDisp | vecDistToQ | ANN p50 | Exact p50 | FasterPath | ANN recall |\n"
        );
        sb.append(
            "| ------------ | --- | ---: | -------: | ------: | ---------: | --------------: | ---------: | ------: | --------: | ---------- | ---------: |\n"
        );
        for (final Scenario s : scenarios) {
            sb.append(
                String.format(
                    Locale.ROOT,
                    "| %-12s | %-3s | %5d | %8.4f | %7.2f | %10.3f | %15.4f | %10.4f | %7.1f | %9.1f | %-10s | %.3f |%n",
                    s.dist,
                    s.validation ? "VAL" : "CAL",
                    s.card,
                    s.sf.weightedNormalizedSpan,
                    s.sf.weightedMeanGap,
                    s.sf.fracInContiguousRuns,
                    s.vecMeanDistToCentroid,
                    s.vecMeanDistToQuery,
                    s.annP50,
                    s.exactP50,
                    s.exactIsFaster() ? "EXACT" : "ANN",
                    s.meanRecall
                )
            );
        }
        return sb.toString();
    }

    // ------------------------------------------------------------------ Phase 6: models

    private interface Model {
        String name();

        Predicate<Scenario> rule();
    }

    private String renderModelAnalysis() {
        // Fit locality models on calibration scenarios only.
        final List<Scenario> calib = scenarios.stream().filter(x -> !x.validation).toList();
        final int ambigLo = 1000, ambigHi = 25000;

        // Fit F: in the ambiguous band, pick exact if normSpan >= spanThr (scatter => exact faster, per calibration).
        final double spanThr = fitThreshold(calib, ambigLo, ambigHi, x -> x.sf.weightedNormalizedSpan, true);
        // Fit H: pick exact if vecMeanDistToCentroid >= dispThr (dispersed => exact faster).
        final double dispThr = fitThreshold(calib, ambigLo, ambigHi, x -> x.vecMeanDistToCentroid, true);
        // Fit G: pick exact if fracLeavesWithMatches >= leafThr.
        final double leafThr = fitThreshold(calib, ambigLo, ambigHi, x -> x.sf.fracLeavesWithMatches, true);

        final List<Model> models = new ArrayList<>();
        models.add(m("A: max(1000,10k)", x -> x.card <= Math.max(1000, 10L * K)));
        models.add(m("B: fixed 5000", x -> x.card <= 5000));
        models.add(m("B: fixed 10000", x -> x.card <= 10000));
        models.add(m("C: 100*ef", x -> x.card <= 100L * EF));
        models.add(m("E: cost card*dim", x -> (double) x.card * DIM * 3.2e-3 < 900.0));
        models.add(m("F: docID scatter", x -> staged(x, ambigLo, ambigHi, x.sf.weightedNormalizedSpan >= spanThr)));
        models.add(m("G: segment leaves", x -> staged(x, ambigLo, ambigHi, x.sf.fracLeavesWithMatches >= leafThr)));
        models.add(m("H: vector dispersion", x -> staged(x, ambigLo, ambigHi, x.vecMeanDistToCentroid >= dispThr)));
        models.add(m("I: staged (H in band)", x -> staged(x, ambigLo, ambigHi, x.vecMeanDistToCentroid >= dispThr)));

        final StringBuilder sb = new StringBuilder("\n### Latency-oracle model comparison (calibration vs held-out validation)\n");
        sb.append(
            String.format(
                Locale.ROOT,
                "(fitted on calibration: spanThr=%.4f, dispThr=%.4f, leafThr=%.4f; ambiguous band (%d,%d])%n",
                spanThr,
                dispThr,
                leafThr,
                ambigLo,
                ambigHi
            )
        );
        sb.append("| Model | CAL faster-path | VAL faster-path | VAL mean regret µs | VAL p95 regret µs | VAL max regret µs |\n");
        sb.append("| ----- | --------------: | --------------: | -----------------: | ----------------: | ----------------: |\n");
        for (final Model model : models) {
            sb.append(evalModel(model));
        }
        sb.append("\nFeature overhead (median µs, from bounded sample): see per-card below.\n");
        sb.append(overheadTable());
        return sb.toString();
    }

    private boolean staged(final Scenario x, final int lo, final int hi, final boolean featureSaysExact) {
        if (x.card <= lo) {
            return true; // small -> exact
        }
        if (x.card > hi) {
            return false; // large -> ANN
        }
        return featureSaysExact;
    }

    private double fitThreshold(
        final List<Scenario> calib,
        final int lo,
        final int hi,
        final java.util.function.ToDoubleFunction<Scenario> feat,
        final boolean higherMeansExact
    ) {
        // Grid-search the feature threshold minimizing regret on calibration ambiguous scenarios.
        final List<Scenario> band = calib.stream().filter(x -> x.card > lo && x.card <= hi).toList();
        double best = 0, bestRegret = Double.MAX_VALUE;
        final double[] vals = band.stream().mapToDouble(feat).sorted().toArray();
        if (vals.length == 0) {
            return 0;
        }
        for (final double thr : vals) {
            double regret = 0;
            for (final Scenario x : band) {
                final boolean pickExact = higherMeansExact ? feat.applyAsDouble(x) >= thr : feat.applyAsDouble(x) <= thr;
                final double chosen = pickExact ? x.exactP50 : x.annP50;
                regret += chosen - Math.min(x.annP50, x.exactP50);
            }
            if (regret < bestRegret) {
                bestRegret = regret;
                best = thr;
            }
        }
        return best;
    }

    private Model m(final String name, final Predicate<Scenario> rule) {
        return new Model() {
            public String name() {
                return name;
            }

            public Predicate<Scenario> rule() {
                return rule;
            }
        };
    }

    private String evalModel(final Model model) {
        final double calSel = selectionRate(scenarios.stream().filter(x -> !x.validation).toList(), model.rule());
        final List<Scenario> val = scenarios.stream().filter(x -> x.validation).toList();
        final double valSel = selectionRate(val, model.rule());
        final double[] reg = val.stream().mapToDouble(x -> {
            final boolean e = model.rule().test(x);
            return (e ? x.exactP50 : x.annP50) - Math.min(x.annP50, x.exactP50);
        }).toArray();
        return String.format(
            Locale.ROOT,
            "| %-20s | %14.1f%% | %14.1f%% | %18.1f | %17.1f | %17.1f |%n",
            model.name(),
            calSel * 100,
            valSel * 100,
            mean(reg),
            pctileD(reg, 95),
            max(reg)
        );
    }

    private double selectionRate(final List<Scenario> set, final Predicate<Scenario> rule) {
        int ok = 0;
        for (final Scenario x : set) {
            if (rule.test(x) == x.exactIsFaster()) {
                ok++;
            }
        }
        return set.isEmpty() ? 0 : (double) ok / set.size();
    }

    private String overheadTable() {
        final StringBuilder sb = new StringBuilder("\n| Card | Scatter feature µs (A+B) | Vector feature µs (C) | % of exact p50 |\n");
        sb.append("| ---: | -----------------------: | --------------------: | -------------: |\n");
        for (final int c : CARDS) {
            final List<Scenario> at = scenarios.stream().filter(x -> x.card == c).toList();
            final double sc = at.stream().mapToDouble(x -> x.featureOverheadUs).average().orElse(0);
            final double ve = at.stream().mapToDouble(x -> x.vecFeatureOverheadUs).average().orElse(0);
            final double ex = at.stream().mapToDouble(x -> x.exactP50).average().orElse(1);
            sb.append(String.format(Locale.ROOT, "| %4d | %24.1f | %21.1f | %13.1f%% |%n", c, sc, ve, 100.0 * (sc + ve) / ex));
        }
        return sb.toString();
    }

    // ------------------------------------------------------------------ Phase 7: recall-constrained

    private String renderRecallConstrained() {
        final StringBuilder sb = new StringBuilder(
            "\n### Recall-constrained oracle (pick ANN only if ANN mean recall >= target, else exact)\n"
        );
        sb.append("| Recall target | Exact selections | ANN selections | Mean latency µs | Violations (ANN chosen, recall<target) |\n");
        sb.append("| ------------: | ---------------: | -------------: | --------------: | --------------------------------------: |\n");
        for (final double target : new double[] { 0.90, 0.95, 0.99 }) {
            int exactSel = 0, annSel = 0, violations = 0;
            double latSum = 0;
            for (final Scenario x : scenarios) {
                final boolean annOk = x.meanRecall >= target;
                if (annOk) {
                    annSel++;
                    latSum += x.annP50;
                } else {
                    exactSel++;
                    latSum += x.exactP50;
                }
                // Violation is impossible for the oracle by construction; report current rule A's violations instead.
                final boolean aPicksAnn = !(x.card <= Math.max(1000, 10L * K));
                if (aPicksAnn && x.meanRecall < target) {
                    violations++;
                }
            }
            sb.append(
                String.format(
                    Locale.ROOT,
                    "| %13.2f | %16d | %14d | %15.1f | %39d |%n",
                    target,
                    exactSel,
                    annSel,
                    latSum / scenarios.size(),
                    violations
                )
            );
        }
        sb.append("(Violations column = how many scenarios the CURRENT rule A would violate this recall target by choosing ANN.)\n");
        return sb.toString();
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

    private Set<Integer> idSet(final TopDocs td) {
        final Set<Integer> ids = new HashSet<>();
        for (final ScoreDoc sd : td.scoreDocs) {
            ids.add(sd.doc);
        }
        return ids;
    }

    private double recall(final Set<Integer> ann, final Set<Integer> ex, final int card) {
        final int denom = Math.min(K, card);
        if (denom == 0 || ex == null || ex.isEmpty()) {
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
            "env: jvm=%s, os=%s/%s, cores=%d, maxHeapMB=%d, queries=%d, sampleCap=%d, vecSample=%d, queryCache=DISABLED, seed=%d%n",
            System.getProperty("java.version"),
            System.getProperty("os.name"),
            System.getProperty("os.arch"),
            rt.availableProcessors(),
            rt.maxMemory() / (1024 * 1024),
            NUM_QUERIES,
            SAMPLE_CAP,
            VEC_SAMPLE,
            SEED
        );
    }

    private static double pct(final long[] s, final int p) {
        final long[] c = s.clone();
        Arrays.sort(c);
        return c[Math.max(0, Math.min((int) Math.ceil(p / 100.0 * c.length) - 1, c.length - 1))] / 1_000.0;
    }

    private static double pctile(final double[] v, final int p) {
        final double[] c = v.clone();
        Arrays.sort(c);
        return c[Math.max(0, Math.min((int) Math.ceil(p / 100.0 * c.length) - 1, c.length - 1))];
    }

    private static double pctileD(final double[] v, final int p) {
        if (v.length == 0) {
            return 0;
        }
        return pctile(v, p);
    }

    private static double mean(final double[] v) {
        double s = 0;
        for (final double x : v) {
            s += x;
        }
        return v.length == 0 ? 0 : s / v.length;
    }

    private static double min(final double[] v) {
        double m = Double.MAX_VALUE;
        for (final double x : v) {
            m = Math.min(m, x);
        }
        return v.length == 0 ? 0 : m;
    }

    private static double max(final double[] v) {
        double m = 0;
        for (final double x : v) {
            m = Math.max(m, x);
        }
        return m;
    }

    private static double fracEqual(final double[] v, final double t) {
        int c = 0;
        for (final double x : v) {
            if (x >= t - 1e-9) {
                c++;
            }
        }
        return v.length == 0 ? 0 : (double) c / v.length;
    }
}
