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
import org.apache.lucene.search.LRUQueryCache;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryCachingPolicy;
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
import java.util.List;
import java.util.Locale;
import java.util.Random;

import static org.mockito.Mockito.mock;

/**
 * Phase-6 cache-behaviour check with Lucene query caching ENABLED (earlier benchmarks disabled it for
 * compute isolation). Measures cold (first) vs warm (repeated) latency for the bounded-exact wrapper vs.
 * the raw ANN baseline, using the same filter across many query vectors, at an exact-band cardinality
 * (500) and an ANN-band cardinality (5000). Purpose is observation, not optimization.
 */
public class BoundedExactCacheBenchmarkTests extends OpenSearchTestCase {

    private static final String FIELD = "vec";
    private static final String IDF = "idf";
    private static final VectorSimilarityFunction SIM = VectorSimilarityFunction.EUCLIDEAN;
    private static final int CORPUS = 100_000;
    private static final int DIM = 128;
    private static final int K = 10;
    private static final int EF = 100;
    private static final int WARM = 30;
    private static final long SEED = 42L;

    @SneakyThrows
    public void testCacheBehaviour() {
        final StringBuilder out = new StringBuilder("\n\n****** BOUNDED-EXACT CACHE BEHAVIOUR (query cache ENABLED, 100k/dim128) ******\n");
        final Random r = new Random(SEED);
        try (Directory dir = newFSDirectory(createTempDir())) {
            try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
                for (int i = 0; i < CORPUS; i++) {
                    final Document d = new Document();
                    d.add(new KnnFloatVectorField(FIELD, unit(r), SIM));
                    d.add(new IntPoint(IDF, i));
                    w.addDocument(d);
                }
                w.commit();
            }
            try (IndexReader reader = DirectoryReader.open(dir)) {
                try (MockedStatic<ModelDao.OpenSearchKNNModelDao> mocked = Mockito.mockStatic(ModelDao.OpenSearchKNNModelDao.class)) {
                    mocked.when(ModelDao.OpenSearchKNNModelDao::getInstance).thenReturn(mock(ModelDao.OpenSearchKNNModelDao.class));
                    final IndexSearcher searcher = newSearcher(reader, true, false);
                    // Force caching so filter reuse is observable.
                    searcher.setQueryCache(new LRUQueryCache(1000, 50_000_000L));
                    searcher.setQueryCachingPolicy(new QueryCachingPolicy() {
                        @Override
                        public void onUse(final Query query) {}

                        @Override
                        public boolean shouldCache(final Query query) {
                            return true; // force caching so filter reuse is observable
                        }
                    });

                    final List<float[]> queries = new ArrayList<>();
                    for (int i = 0; i < WARM + 1; i++) {
                        queries.add(unit(new Random(SEED + 100 + i)));
                    }

                    out.append("| Cardinality | Path | Cold µs (1st) | Warm µs (median of repeats, same filter) |\n");
                    out.append("| ----------: | ---- | ------------: | ----------------------------------------: |\n");
                    for (final int card : new int[] { 500, 5000 }) {
                        final Query filter = IntPoint.newRangeQuery(IDF, 0, card - 1);
                        measure(out, searcher, "ANN baseline", card, filter, queries, false);
                        measure(out, searcher, "BoundedExact", card, filter, queries, true);
                    }
                    out.append("\n(Same filter, different query vector each repeat: exercises filter-bitset cache reuse.)\n");
                    out.append("****************************************************************************\n");
                    System.out.println(out);
                }
            }
        }
    }

    private void measure(
        final StringBuilder out,
        final IndexSearcher s,
        final String label,
        final int card,
        final Query filter,
        final List<float[]> queries,
        final boolean wrapper
    ) throws IOException {
        final long coldStart = System.nanoTime();
        run(s, queries.get(0), filter, wrapper);
        final double cold = (System.nanoTime() - coldStart) / 1_000.0;

        final long[] warm = new long[WARM];
        for (int i = 0; i < WARM; i++) {
            final long t = System.nanoTime();
            run(s, queries.get(i + 1), filter, wrapper); // different query vector, same filter
            warm[i] = System.nanoTime() - t;
        }
        Arrays.sort(warm);
        out.append(String.format(Locale.ROOT, "| %11d | %-12s | %13.1f | %41.1f |%n", card, label, cold, warm[warm.length / 2] / 1_000.0));
    }

    private TopDocs run(final IndexSearcher s, final float[] q, final Query filter, final boolean wrapper) throws IOException {
        if (wrapper) {
            final Query ann = new OSKnnFloatVectorQuery(FIELD, q, Math.max(EF, K), filter, K, RescoreContext.NO_RESCORE_NEEDED);
            final ExactSearcher ex = new ExactSearcher(ModelDao.OpenSearchKNNModelDao.getInstance());
            return s.search(new BoundedExactKnnFloatVectorQuery(ann, filter, FIELD, K, q, 0, ex, null), K);
        }
        return s.search(new OSKnnFloatVectorQuery(FIELD, q, Math.max(EF, K), filter, K, RescoreContext.NO_RESCORE_NEEDED), K);
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
}
