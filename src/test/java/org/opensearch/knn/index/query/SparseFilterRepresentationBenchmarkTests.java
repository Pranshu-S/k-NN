/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.SparseFixedBitSet;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher.ExactSearcherContext;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

import static org.mockito.Mockito.mock;

/**
 * Phase-6 benchmark comparing sparse per-leaf filter-match representations for the exact path:
 * {@link FixedBitSet} (the previous representation), {@link SparseFixedBitSet}, and
 * {@link IntArrayDocIdSetIterator} (the new representation). Reports allocation (via
 * {@code ramBytesUsed} / analytic), build time, and iteration time across {@code maxDoc} x {@code
 * matches}, plus exact end-to-end latency on a real index for the bit-set vs. int-array iterators.
 */
public class SparseFilterRepresentationBenchmarkTests extends OpenSearchTestCase {

    private static final int[] MAX_DOCS = { 100_000, 1_000_000, 10_000_000 };
    private static final int[] MATCHES = { 10, 100, 1_000, 5_000 };
    private static final int REPS = 200;

    @SneakyThrows
    public void testRepresentationMemoryAndSpeed() {
        final Random random = new Random(7);
        final StringBuilder out = new StringBuilder();
        out.append("\n\n============ SPARSE FILTER REPRESENTATION BENCHMARK ============\n");
        out.append("(build/iterate are medians over ").append(REPS).append(" reps; memory is bytes)\n\n");
        out.append("| Representation | maxDoc | Matches | Memory (bytes) | Build µs | Iterate µs |\n");
        out.append("| -------------- | -----: | ------: | -------------: | -------: | ---------: |\n");

        for (final int maxDoc : MAX_DOCS) {
            for (final int matches : MATCHES) {
                final int[] docIds = ascendingDistinct(random, matches, maxDoc);

                // FixedBitSet
                measureBitSet(out, "FixedBitSet", maxDoc, matches, docIds, false);
                // SparseFixedBitSet
                measureBitSet(out, "SparseFixedBitSet", maxDoc, matches, docIds, true);
                // IntArrayDocIdSetIterator
                measureIntArray(out, maxDoc, matches, docIds);
            }
        }
        out.append("\n(Analytic sizes: FixedBitSet = ceil(maxDoc/64)*8 + overhead; IntArray = matches*4 + overhead;\n");
        out.append(" SparseFixedBitSet ~ (maxDoc/4096) index longs + touched blocks. Bit sets use ramBytesUsed().)\n");
        out.append("===============================================================\n");
        System.out.println(out);
    }

    private void measureBitSet(
        final StringBuilder out,
        final String name,
        final int maxDoc,
        final int matches,
        final int[] docIds,
        final boolean sparse
    ) throws IOException {
        long buildNanos = Long.MAX_VALUE;
        long iterNanos = Long.MAX_VALUE;
        long ramBytes = 0;
        for (int r = 0; r < REPS; r++) {
            long start = System.nanoTime();
            final org.apache.lucene.util.BitSet bitSet = sparse ? new SparseFixedBitSet(maxDoc) : new FixedBitSet(maxDoc);
            for (final int doc : docIds) {
                bitSet.set(doc);
            }
            buildNanos = Math.min(buildNanos, System.nanoTime() - start);
            ramBytes = bitSet.ramBytesUsed();

            start = System.nanoTime();
            final DocIdSetIterator it = new BitSetIterator(bitSet, matches);
            int sink = 0;
            for (int d = it.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
                sink += d;
            }
            iterNanos = Math.min(iterNanos, System.nanoTime() - start);
            if (sink == Integer.MIN_VALUE) {
                out.append(""); // prevent dead-code elimination
            }
        }
        out.append(row(name, maxDoc, matches, ramBytes, buildNanos, iterNanos));
    }

    private void measureIntArray(final StringBuilder out, final int maxDoc, final int matches, final int[] docIds) {
        long buildNanos = Long.MAX_VALUE;
        long iterNanos = Long.MAX_VALUE;
        for (int r = 0; r < REPS; r++) {
            long start = System.nanoTime();
            final int[] copy = Arrays.copyOf(docIds, docIds.length); // model per-query allocation
            final IntArrayDocIdSetIterator itBuild = new IntArrayDocIdSetIterator(copy, copy.length);
            buildNanos = Math.min(buildNanos, System.nanoTime() - start);

            start = System.nanoTime();
            int sink = 0;
            for (int d = itBuild.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = itBuild.nextDoc()) {
                sink += d;
            }
            iterNanos = Math.min(iterNanos, System.nanoTime() - start);
            if (sink == Integer.MIN_VALUE) {
                out.append("");
            }
        }
        final long ramBytes = 16L + (long) matches * Integer.BYTES; // array header + ints
        out.append(row("IntArrayDISI", maxDoc, matches, ramBytes, buildNanos, iterNanos));
    }

    private String row(
        final String name,
        final int maxDoc,
        final int matches,
        final long ramBytes,
        final long buildNanos,
        final long iterNanos
    ) {
        return String.format(
            Locale.ROOT,
            "| %-14s | %6d | %7d | %14d | %8.2f | %10.2f |%n",
            name,
            maxDoc,
            matches,
            ramBytes,
            buildNanos / 1_000.0,
            iterNanos / 1_000.0
        );
    }

    private static int[] ascendingDistinct(final Random random, final int count, final int bound) {
        final java.util.TreeSet<Integer> set = new java.util.TreeSet<>();
        while (set.size() < count) {
            set.add(random.nextInt(bound));
        }
        final int[] ids = new int[count];
        int i = 0;
        for (final int v : set) {
            ids[i++] = v;
        }
        return ids;
    }

    // ------------------------------------------------------------------ exact end-to-end (bitset vs int[])

    @SneakyThrows
    public void testExactLatencyBitSetVsIntArray() {
        final int totalDocs = 100_000;
        final int dim = 128;
        final int k = 10;
        final int[] cardinalities = { 100, 1_000, 5_000 };
        final int warmup = 5;
        final int measured = 30;
        final Random random = new Random(11);

        try (Directory directory = newFSDirectory(createTempDir())) {
            try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
                for (int i = 0; i < totalDocs; i++) {
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField("vec", unit(random, dim), VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();
            }
            try (IndexReader reader = DirectoryReader.open(directory)) {
                try (MockedStatic<ModelDao.OpenSearchKNNModelDao> mocked = Mockito.mockStatic(ModelDao.OpenSearchKNNModelDao.class)) {
                    mocked.when(ModelDao.OpenSearchKNNModelDao::getInstance).thenReturn(mock(ModelDao.OpenSearchKNNModelDao.class));
                    final IndexSearcher searcher = newSearcher(reader, true, false);
                    searcher.setQueryCache(null);
                    final ExactSearcher exactSearcher = new ExactSearcher(ModelDao.OpenSearchKNNModelDao.getInstance());
                    final LeafReaderContext leaf = reader.leaves().get(0);
                    final int leafMaxDoc = leaf.reader().maxDoc();

                    final StringBuilder out = new StringBuilder(
                        "\n\n===== EXACT PATH: FixedBitSet vs IntArray end-to-end (100k x 128, single leaf) =====\n"
                    );
                    out.append("| Cardinality | FixedBitSet exact µs (med) | IntArray exact µs (med) |\n");
                    out.append("| ----------: | -------------------------: | ----------------------: |\n");

                    for (final int cardinality : cardinalities) {
                        final int[] docIds = ascendingDistinct(random, cardinality, leafMaxDoc);
                        final float[] q = unit(random, dim);

                        final long[] bitSetT = new long[measured];
                        final long[] intArrT = new long[measured];
                        for (int i = 0; i < warmup + measured; i++) {
                            final boolean measure = i >= warmup;
                            final int m = i - warmup;

                            long start = System.nanoTime();
                            final FixedBitSet bitSet = new FixedBitSet(leafMaxDoc);
                            for (final int d : docIds) {
                                bitSet.set(d);
                            }
                            runExactLeaf(exactSearcher, leaf, new BitSetIterator(bitSet, cardinality), cardinality, k, q);
                            final long bt = System.nanoTime() - start;

                            start = System.nanoTime();
                            runExactLeaf(exactSearcher, leaf, new IntArrayDocIdSetIterator(docIds, cardinality), cardinality, k, q);
                            final long it = System.nanoTime() - start;

                            if (measure) {
                                bitSetT[m] = bt;
                                intArrT[m] = it;
                            }
                        }
                        out.append(
                            String.format(
                                Locale.ROOT,
                                "| %11d | %26.1f | %23.1f |%n",
                                cardinality,
                                median(bitSetT) / 1_000.0,
                                median(intArrT) / 1_000.0
                            )
                        );
                    }
                    out.append("=================================================================================\n");
                    System.out.println(out);
                }
            }
        }
    }

    private TopDocs runExactLeaf(
        final ExactSearcher exactSearcher,
        final LeafReaderContext leaf,
        final DocIdSetIterator matched,
        final int cardinality,
        final int k,
        final float[] q
    ) throws IOException {
        final ExactSearcherContext ctx = ExactSearcherContext.builder()
            .matchedDocsIterator(matched)
            .numberOfMatchedDocs(cardinality)
            .useQuantizedVectorsForSearch(false)
            .k(k)
            .field("vec")
            .floatQueryVector(q)
            .build();
        return exactSearcher.searchLeaf(leaf, ctx);
    }

    private static float[] unit(final Random random, final int dim) {
        final float[] v = new float[dim];
        double norm = 0;
        for (int i = 0; i < dim; i++) {
            v[i] = (float) random.nextGaussian();
            norm += (double) v[i] * v[i];
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < dim; i++) {
            v[i] /= (float) (norm == 0 ? 1 : norm);
        }
        return v;
    }

    private static long median(final long[] samples) {
        final long[] c = samples.clone();
        Arrays.sort(c);
        return c[c.length / 2];
    }
}
