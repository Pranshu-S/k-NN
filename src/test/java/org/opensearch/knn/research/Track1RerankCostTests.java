/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.research;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.plugin.script.KNNScoringUtil;
import org.opensearch.knn.quantization.models.quantizationOutput.BinaryQuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.quantizer.OneBitScalarQuantizer;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Track 1 — cost of "baseline rerank 100" vs "ADC rerank 50" (the equal-recall operating point
 * validated in FINDINGS.md: SIFT-128 L2 ADC cr@50 0.819 >= baseline cr@100 0.786).
 *
 * Measures, with the REAL {@link KNNScoringUtil#l2Squared} FP32 scorer:
 *   - RERANK-STEP p50/p99 latency: read N fp32 candidate vectors, compute exact L2, partial-sort top-k.
 *     This is the operation that differs between the two recipes (N=100 vs N=50). Warm / in-memory.
 *   - FP32 bytes read during rerank = N * dim * 4 (deterministic).
 *   - FIRST-PASS per-doc scoring cost: ADC asymmetric (l2SquaredADC) vs symmetric Hamming — context
 *     for whether ADC's costlier first pass offsets the rerank savings.
 *
 * NOT measured (native tier): total query p50/p99 incl. HNSW traversal, COLD-cache disk reads
 * (where the 2x-fewer bytes matters MORE, not less), index size, throughput.
 */
public class Track1RerankCostTests extends KNNTestCase {

    private static final int N = 20000;      // corpus (random candidate access -> some cache pressure)
    private static final int NQ = 300;       // timed queries
    private static final int REPS = 5;
    private static final int K = 10;
    private static final int[] DIMS = { 128, 384, 768, 1024, 1536 };
    private static final long SEED = 99L;

    public void testRerankCost() throws IOException {
        List<String> rows = new ArrayList<>();
        rows.add("dim,recipe,rerank_N,fp32_bytes_per_query,rerank_p50_us,rerank_p99_us,firstpass_ns_per_doc");
        Random rng = new Random(SEED);
        for (int d : DIMS) {
            float[][] docs = new float[N][d];
            for (int i = 0; i < N; i++) for (int j = 0; j < d; j++) docs[i][j] = (float) rng.nextGaussian();
            float[][] queries = new float[NQ][d];
            for (int i = 0; i < NQ; i++) for (int j = 0; j < d; j++) queries[i][j] = (float) rng.nextGaussian();
            // random candidate id sets per query (rerank cost depends on count+dim, not which docs)
            int[][] cand100 = new int[NQ][100], cand50 = new int[NQ][50];
            for (int q = 0; q < NQ; q++) { for (int i = 0; i < 100; i++) cand100[q][i] = rng.nextInt(N);
                for (int i = 0; i < 50; i++) cand50[q][i] = cand100[q][i]; }   // ADC's 50 = subset of the 100

            double fpAdc = firstPassNsPerDoc(docs, queries, true);    // ADC asymmetric l2SquaredADC
            double fpBase = firstPassNsPerDoc(docs, queries, false);  // symmetric Hamming

            double[] base = rerankLatency(docs, queries, cand100);    // baseline: rerank 100
            double[] adc = rerankLatency(docs, queries, cand50);      // ADC: rerank 50
            rows.add(String.format(java.util.Locale.ROOT, "%d,baseline_rerank100,100,%d,%.1f,%.1f,%.2f",
                d, 100L * d * 4, base[0], base[1], fpBase));
            rows.add(String.format(java.util.Locale.ROOT, "%d,adc_rerank50,50,%d,%.1f,%.1f,%.2f",
                d, 50L * d * 4, adc[0], adc[1], fpAdc));
            System.out.printf(java.util.Locale.ROOT, "d=%4d  base@100 p50=%.1f p99=%.1f (%dB)  |  adc@50 p50=%.1f p99=%.1f (%dB)  | firstpass ns/doc base=%.2f adc=%.2f%n",
                d, base[0], base[1], 100 * d * 4, adc[0], adc[1], 50 * d * 4, fpBase, fpAdc);
        }
        Path out = Paths.get(System.getProperty("user.dir"), "research", "track1_adc_rotation", "results");
        Files.createDirectories(out);
        try (Writer w = Files.newBufferedWriter(out.resolve("track1_rerank_cost.csv"))) {
            for (String r : rows) { w.write(r); w.write("\n"); }
        }
        System.out.println("WROTE rerank cost -> " + out.resolve("track1_rerank_cost.csv"));
    }

    /** p50/p99 (us) of one rerank: for each candidate read fp32 vec, real l2Squared, partial-sort top-k. */
    private static double[] rerankLatency(float[][] docs, float[][] queries, int[][] cand) {
        int nq = queries.length;
        // warmup (JIT)
        for (int w = 0; w < 3; w++) for (int q = 0; q < nq; q++) rerankOnce(docs, queries[q], cand[q]);
        double[] samples = new double[nq * REPS]; int idx = 0;
        for (int r = 0; r < REPS; r++) {
            for (int q = 0; q < nq; q++) {
                long t0 = System.nanoTime();
                rerankOnce(docs, queries[q], cand[q]);
                long t1 = System.nanoTime();
                samples[idx++] = (t1 - t0) / 1000.0;   // us
            }
        }
        Arrays.sort(samples);
        return new double[] { pct(samples, 50), pct(samples, 99) };
    }

    private static float rerankOnce(float[][] docs, float[] q, int[] cand) {
        float[] dist = new float[cand.length];
        for (int i = 0; i < cand.length; i++) dist[i] = KNNScoringUtil.l2Squared(q, docs[cand[i]]);  // REAL fp32 scorer
        // partial top-k (selection of K smallest)
        Integer[] idx = new Integer[cand.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(dist[a], dist[b]));
        float sink = 0; for (int i = 0; i < K && i < idx.length; i++) sink += dist[idx[i]];
        return sink;
    }

    /** ns per doc for the FIRST-PASS quantized scorer: ADC (l2SquaredADC) vs Hamming. */
    private static double firstPassNsPerDoc(float[][] docs, float[][] queries, boolean adc) throws IOException {
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer(false);
        TrainingRequest<float[]> req = new TrainingRequest<float[]>(docs.length, false) {
            @Override public float[] getVectorAtThePosition(int position) { return docs[position]; }
            @Override public void resetVectorValues() { }
        };
        OneBitScalarQuantizationState st = (OneBitScalarQuantizationState) quantizer.train(req);
        int scan = 4000;
        byte[][] bits = new byte[scan][];
        for (int j = 0; j < scan; j++) { BinaryQuantizationOutput o = new BinaryQuantizationOutput(1); quantizer.quantize(docs[j].clone(), st, o); bits[j] = o.getQuantizedVector().clone(); }
        float sink = 0; long t0, t1; int iters = 20;
        if (adc) {
            float[] qt = queries[0].clone(); quantizer.transformWithADC(qt, st, org.opensearch.knn.index.SpaceType.L2);
            for (int w = 0; w < 3; w++) for (int j = 0; j < scan; j++) sink += KNNScoringUtil.l2SquaredADC(qt, bits[j]);
            t0 = System.nanoTime();
            for (int it = 0; it < iters; it++) for (int j = 0; j < scan; j++) sink += KNNScoringUtil.l2SquaredADC(qt, bits[j]);
            t1 = System.nanoTime();
        } else {
            BinaryQuantizationOutput qo = new BinaryQuantizationOutput(1); quantizer.quantize(queries[0].clone(), st, qo); byte[] qb = qo.getQuantizedVector();
            for (int w = 0; w < 3; w++) for (int j = 0; j < scan; j++) sink += hamming(qb, bits[j]);
            t0 = System.nanoTime();
            for (int it = 0; it < iters; it++) for (int j = 0; j < scan; j++) sink += hamming(qb, bits[j]);
            t1 = System.nanoTime();
        }
        if (sink == Float.NaN) throw new IllegalStateException();
        return (double) (t1 - t0) / ((long) iters * scan);
    }

    private static int hamming(byte[] a, byte[] b) { int s = 0; for (int i = 0; i < a.length; i++) s += Integer.bitCount((a[i] ^ b[i]) & 0xFF); return s; }
    private static double pct(double[] sorted, int p) { return sorted[Math.min(sorted.length - 1, (int) Math.ceil(p / 100.0 * sorted.length) - 1)]; }
}
